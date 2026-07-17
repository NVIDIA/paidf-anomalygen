# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from imaginaire.utils import log
from roi_generate.box_to_mask import BoxToMaskPostProcess
from roi_generate.default_config import DefaultConfig, validate_sample_config
from roi_generate.grayscale_to_mask import GrayscaleToMaskPostProcess, grayscale_binarize
from roi_generate.template_box_to_masks import build_template_box_to_masks_stages, load_cached_context
from roi_generate.utils import sample_resize


def prepare_samples(samples):
    """
    Normalize and validate per-sample config.

    For each sample:
      - Ensure required fields (`image_path`, `boxes`) exist.
      - Merge its `config` (if provided) with DefaultConfig.
      - Validate the merged config.
      - Store the final OmegaConf config back into sample["config"].

    Returns a new list of samples with resolved configs.
    """
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a non-empty list")

    base_schema = OmegaConf.structured(DefaultConfig)

    prepared = []
    for i, sample in enumerate(samples):
        # Validate image_path, boxes, and config for each sample
        if not isinstance(sample, dict):
            raise TypeError(f"Sample {i} must be a dict, got {type(sample)}")
        if "image_path" not in sample:
            raise ValueError(f"Sample {i} missing required field: 'image_path'")
        if "boxes" not in sample:
            raise ValueError(f"Sample {i} missing required field: 'boxes'")

        user_cfg_dict = sample.get("config", {})
        if not isinstance(user_cfg_dict, dict):
            raise TypeError(f"Sample {i} 'config' must be a dict if provided, got {type(user_cfg_dict)}")

        user_cfg = OmegaConf.create(user_cfg_dict)
        cfg = OmegaConf.merge(base_schema, user_cfg)
        validate_sample_config(cfg)
        sample["config"] = cfg
        prepared.append(sample)

    return prepared


def load_one_sample(sample):
    """
    load a single sample.
    Validates image path, and returns (ctx).
    """
    image_path = sample["image_path"]

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    cfg = sample["config"]
    boxes = sample["boxes"]
    need_boxes = cfg.box_to_mask.enabled or cfg.template_box_to_masks.enabled
    if need_boxes:
        ensure_valid_boxes(boxes, img_w, img_h)
    else:
        boxes = []

    ctx = {"input": {"image_path": image_path, "image": image, "boxes": boxes,}}
    return ctx


def ensure_valid_boxes(raw_boxes, img_w, img_h):
    """Validate box format and ensure boxes lie within image bounds."""
    if not raw_boxes:
        raise ValueError(
            "At least one bounding box is required when Box-to-Mask or Template-Box-to-Masks is enabled. "
            "Please provide a bounding box or disable these modes."
        )
    for box in raw_boxes:
        # Format check
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"Invalid box format: {box!r}; expected [x0, y0, x1, y1].")

        x0, y0, x1, y1 = box

        # Area check
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid box with non-positive area: {box!r}")

        # Boundary check
        if not (0 <= x0 < img_w and 0 <= x1 <= img_w and 0 <= y0 < img_h and 0 <= y1 <= img_h):
            raise ValueError(f"Box {box!r} is out of image bounds. " f"Image size = ({img_w} * {img_h}).")
    return True


@torch.inference_mode()
def run_pipeline(samples, output_dir, roi_generate_models):
    """Unified ROI-Generate pipeline for both modes."""
    samples = prepare_samples(samples)

    total = len(samples)
    for idx, sample in enumerate(samples, start=1):
        log.info(f"[Sample {idx}/{total}] Processing image: {sample['image_path']}")
        ctx = load_one_sample(sample)
        ctx["input"]["output_dir"] = os.path.join(output_dir, f"sample_{idx:05d}")
        t0 = time.perf_counter()
        config = sample["config"]

        enabled_modes = [
            name
            for name, flag in [
                ("Grayscale-to-Mask", config.grayscale_to_mask.enabled),
                ("Box-to-Mask", config.box_to_mask.enabled),
                ("Template-Box-to-Masks", config.template_box_to_masks.enabled),
            ]
            if flag
        ]

        log.info(f"Enabled modes: {', '.join(enabled_modes)}")
        if config.grayscale_to_mask.enabled:
            run_graymask(ctx, config)
        if config.box_to_mask.enabled:
            run_box2mask(ctx, config, roi_generate_models)
        if config.template_box_to_masks.enabled:
            run_template(ctx, config, roi_generate_models)

        log.info(f"[Sample {idx}/{total}] Done ({time.perf_counter() - t0:.3f}s)")


def run_graymask(ctx, config):
    image = np.array(ctx["input"]["image"])
    t_binary = time.perf_counter()

    mask = grayscale_binarize(image, config)
    binarize_time = time.perf_counter() - t_binary
    post_process = GrayscaleToMaskPostProcess(config)

    t_post_process = time.perf_counter()
    post_process.run(mask)
    post_process.save_result(ctx)
    if config.save_visualization:
        post_process.save_visualization(ctx)
    post_process_time = time.perf_counter() - t_post_process
    log.info(f"[Grayscale-to-Mask] binarize: {binarize_time:.3f}s | post_process: {post_process_time:.3f}s")


def run_box2mask(ctx, config, roi_generate_models):
    post_process = BoxToMaskPostProcess(config)

    t_sam = time.perf_counter()
    resized_img, resized_boxes = sample_resize(
        ctx["input"]["image"], ctx["input"]["boxes"], config.box_to_mask.image_resize
    )
    ctx_resized = {
        "input": {
            "image_path": ctx["input"]["image_path"],
            "image": resized_img,
            "boxes": resized_boxes,
            "ori_image_size": ctx["input"]["image"].size,
            "output_dir": ctx["input"]["output_dir"],
        }
    }

    masks, _ = roi_generate_models.forward_segmentation(
        ctx_resized["input"]["image"], boxes=ctx_resized["input"]["boxes"]
    )
    sam_time = time.perf_counter() - t_sam

    t_post_process = time.perf_counter()
    post_process.run(masks)
    post_process.save_result(ctx_resized)
    if config.save_visualization:
        post_process.save_visualization(ctx_resized)
    post_process_time = time.perf_counter() - t_post_process
    log.info(f"[Box-to-Mask] sam_inference: {sam_time:.3f}s | post_process: {post_process_time:.3f}s")


def run_template(ctx, config, roi_generate_models):
    log.info(f"[Template-Box-to-Masks] Start")
    t0 = time.perf_counter()

    max_template = config.template_box_to_masks.max_template
    num_boxes = len(ctx["input"]["boxes"])
    if num_boxes > max_template:
        raise ValueError(
            f"Too many template boxes detected: {num_boxes} provided, "
            f"but the maximum allowed is {max_template}.\n\n"
            f"Note: Template-Box-to-Masks is computationally intensive, "
            f"even a single template box with max_proposal=300 may take ~20 seconds.\n\n"
            f"Recommended actions:\n"
            f"  1. Reduce boxes to {max_template} or fewer\n"
            f"  2. Increase 'max_template' and lower 'max_proposal' "
            f"to balance speed and coverage\n"
            f"  3. Set 'template_box_to_masks.enabled: false' to skip Template-Box-to-Masks"
        )

    resized_img, resized_boxes = sample_resize(
        ctx["input"]["image"], ctx["input"]["boxes"], config.template_box_to_masks.image_resize
    )

    ctx_resized = {
        "input": {
            "image_path": ctx["input"]["image_path"],
            "image": resized_img,
            "boxes": resized_boxes,
            "ori_image_size": ctx["input"]["image"].size,
            "output_dir": ctx["input"]["output_dir"],
        }
    }

    stages = build_template_box_to_masks_stages(config, roi_generate_models)

    start_idx = 0
    prev_hash = None
    if config.template_box_to_masks.resume_cache:
        cache_dir = os.path.join(ctx_resized["input"]["output_dir"], "template_box_to_masks", "cache")
        ctx_resized, start_idx, prev_hash = load_cached_context(stages, ctx_resized, cache_dir)
        log.info(f"    [resume] From stage: {start_idx}-{stages[start_idx].name}")

    for i, stage in enumerate(stages[start_idx:], start=start_idx):
        t_stage = time.perf_counter()
        is_last = i == len(stages) - 1

        if config.template_box_to_masks.save_cache and not is_last:
            stage_hash = stage.compute_dependency_hash(ctx_resized, prev_hash)

        ctx_resized = stage.run(ctx_resized)

        if config.template_box_to_masks.save_cache and not is_last:
            stage.save_cache(ctx_resized, stage_hash)
            prev_hash = stage_hash
        if config.save_visualization:
            stage.save_visualization(ctx_resized)
        if is_last:
            stage.save_result(ctx_resized)
        log.info(f"    [{i}-{stage.name}] {time.perf_counter() - t_stage:.3f}s")
    log.info(f"[Template-Box-to-Masks] Done ({time.perf_counter() - t0:.3f}s)")
