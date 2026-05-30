#!/usr/bin/env python3
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

"""
Unified ROI → AMP pipeline: auto-routes between CAD-based and Text-based ROI generation.

Routing logic: if a sample has "cad_mask" → cad2roi path, otherwise → text2roi path.

Usage:
    python -m scripts.run_auto_roi_amp \
        --input amp_configs/amp_samples_template.json \
        --defect-desc amp_configs/defect_spec_template.jsonl \
        --output results/output \
        --n_seeds 5 --seed 42

Input JSON format: list of samples, each with:
    Common (required):
        - clean_image:  path to image (PNG)
        - defect_type:  defect category string

    CAD route (when cad_mask is present):
        - cad_mask:     path to CAD semantic segmentation mask (PNG)
        - cad_mask_label:   path to color label JSON
        - submask:      path to defect submask (null for missing)

    Text route (when cad_mask is absent):
        - submask:      path to defect submask (null → ROI only)
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from automatic_mask_placement.cad2roi.mask_utils import mask_area_from_path, preprocess_submask
from automatic_mask_placement.amp_visualize import (
    overlay, make_amp_overlay, save_cad_seed, sum_masks,
    ROI_COLOR, CAD_ROI_COLOR, MASK_COLOR, CAD_MASK_COLOR, BBOX_COLOR, POINT_COLOR,
)
from automatic_mask_placement.text2roi.sam_utils import (
    resize_for_sam, postprocess, pick_mask, IMAGE_RESIZE,
)
from automatic_mask_placement.pipeline_config import load_defect_descriptions, get_sample_route
from automatic_mask_placement.core import AutomaticMaskPlacement
from automatic_mask_placement.cad2roi import (
    CADToROIGenerator, MissingMaskPlacer, LessSolderMaskPlacer,
    ExcessSolderMaskPlacer, BridgeMaskPlacer,
)
from automatic_mask_placement.cad2roi.defects.bridge import get_bridge_groups
from automatic_mask_placement.text2roi.text2box import Text2BoxDetector
from roi_generate.model import ROIGenerateModels
import torch
from imaginaire.utils import log




# ── Free (whole-image ROI) processing ────────────────────────────────────────

def process_free_sample(sample, output_dir, n_seeds, base_seed,
                        min_area_ratio=0.5, max_area_retries=10):
    clean_image_path = sample["clean_image"]
    defect_type = sample["defect_type"]
    submask_path = sample.get("submask")

    name = Path(clean_image_path).stem
    out = Path(output_dir) / name / defect_type
    assets = out / "assets"
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    pil = Image.open(clean_image_path).convert("RGB")
    ori_w, ori_h = pil.size
    img_np = np.array(pil)

    # ROI = entire image (white mask)
    roi_mask = np.full((ori_h, ori_w), 255, dtype=np.uint8)
    roi_mask_path = str(assets / "roi_mask.png")
    Image.fromarray(roi_mask).save(roi_mask_path)

    if not submask_path or not os.path.exists(submask_path):
        log.warning(f"  {name}/{defect_type}: free ROI, no submask for AMP")
        return {"name": name, "defect_type": defect_type, "route": "free",
                "status": "ROI_ONLY", "n_candidates": 1, "seeds": {}}

    submask_path = preprocess_submask(submask_path, sample)
    submask_stem = Path(submask_path).stem
    base_area = mask_area_from_path(submask_path)
    min_area = int(base_area * min_area_ratio)

    seed_results = {}

    for seed_i in range(n_seeds):
        final_mask_path = out / f"{submask_stem}__seed{seed_i}.png"
        overlay_path = assets / f"{submask_stem}__seed{seed_i}_overlay.png"

        if final_mask_path.exists() and overlay_path.exists():
            seed_results[seed_i] = {"status": "CACHED"}
            continue

        placed_ok = False
        amp_results = None

        with tempfile.TemporaryDirectory() as seed_work:
            for attempt in range(max_area_retries):
                sub_seed = base_seed + seed_i * max_area_retries + attempt
                amp = AutomaticMaskPlacement(
                    image_width=ori_w, image_height=ori_h,
                    random_seed=sub_seed,
                    separate_rois=False,
                    max_retry_per_mask=15,
                )
                amp.load_combined_rois(roi_image_paths=[roi_mask_path])
                amp_results = amp.process_submask(
                    submask_path=submask_path,
                    n_instances=1,
                    output_dir=seed_work,
                )
                if not amp_results:
                    continue
                area = mask_area_from_path(amp_results[0]["output_path"])
                if area > min_area:
                    placed_ok = True
                    break

            final_area = (mask_area_from_path(amp_results[0]["output_path"])
                          if amp_results else 0)
            if amp_results and final_area > 0:
                shutil.copy(amp_results[0]["output_path"], str(final_mask_path))

                placed_np = cv2.imread(str(final_mask_path), cv2.IMREAD_GRAYSCALE)
                ov = make_amp_overlay(img_np, None, placed_np, None, None)
                Image.fromarray(ov).save(overlay_path)

                seed_results[seed_i] = {"status": "OK" if placed_ok else "SMALL"}
            else:
                seed_results[seed_i] = {"status": "FAILED"}

    n_ok = sum(1 for s in seed_results.values() if s["status"] in ("OK", "CACHED"))
    log.info(f"  {name}/{defect_type}: free ROI, {n_ok}/{n_seeds} seeds OK")

    return {
        "name": name, "defect_type": defect_type, "route": "free",
        "status": "OK", "n_candidates": 1, "seeds": seed_results,
    }


# ── Text2ROI processing ─────────────────────────────────────────────────────

def process_text2roi_sample(sample, defect_prompts, detector, sam, output_dir,
                            n_seeds, base_seed, roi_only=False,
                            min_area_ratio=0.5, max_area_retries=10):
    clean_image_path = sample["clean_image"]
    defect_type = sample["defect_type"]
    submask_path = sample.get("submask") if not roi_only else None

    prompt = defect_prompts.get(defect_type)
    if prompt is None:
        raise ValueError(
            f"No roi_prompt_defect_location for '{defect_type}' in defect description JSONL."
        )

    name = Path(clean_image_path).stem
    cat, dc = defect_type.split("+", 1)
    out = Path(output_dir) / name / defect_type
    assets = out / "assets"
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    pil = Image.open(clean_image_path).convert("RGB")
    ori_w, ori_h = pil.size
    img_np = np.array(pil)

    # Stage 1: Text2Box
    det_results = detector.detect(pil, prompt)
    if not det_results:
        log.warning(f"  {name}/{defect_type}: no detection, skipping")
        return {"name": name, "defect_type": defect_type, "route": "text2roi",
                "status": "NO_DETECTION", "n_candidates": 0, "seeds": {}}

    top = max(det_results, key=lambda r: r["confidence"])
    bbox = top["bbox"]
    point = top["point"]

    # Save bbox overlay
    bbox_viz = img_np.copy()
    x0, y0, x1, y1 = [int(v) for v in bbox]
    cv2.rectangle(bbox_viz, (x0, y0), (x1, y1), BBOX_COLOR, 3)
    if point:
        px, py = int(point[0]), int(point[1])
        cv2.circle(bbox_viz, (px, py), 10, POINT_COLOR, -1)
        cv2.circle(bbox_viz, (px, py), 10, (0, 0, 0), 2)
    Image.fromarray(bbox_viz).save(assets / "bbox.png")

    # Stage 2: SAM2 → ROI mask
    res_pil, scale = resize_for_sam(pil, IMAGE_RESIZE)
    sam.set_image(np.array(res_pil))

    bx = [int(v * scale) for v in bbox]
    pt_s = [point[0] * scale, point[1] * scale] if point else None

    if pt_s:
        sam_masks, scores, _ = sam.predict(
            point_coords=np.asarray([pt_s]),
            point_labels=np.asarray([1]),
            box=np.asarray(bx),
            multimask_output=True,
        )
    else:
        sam_masks, scores, _ = sam.predict(
            point_coords=None, point_labels=None,
            box=np.asarray(bx),
            multimask_output=True,
        )

    roi_mask = postprocess(pick_mask(sam_masks, scores), ori_w, ori_h)

    Image.fromarray(roi_mask).save(assets / "roi_mask.png")
    roi_viz = overlay(img_np, roi_mask, ROI_COLOR, alpha=0.5)
    if bbox:
        cv2.rectangle(roi_viz, (x0, y0), (x1, y1), BBOX_COLOR, 3)
    if point:
        cv2.circle(roi_viz, (int(point[0]), int(point[1])), 10, POINT_COLOR, -1)
    Image.fromarray(roi_viz).save(assets / "roi_overlay.png")

    # Stage 3: AMP
    if not submask_path or not os.path.exists(submask_path):
        log.warning(f"  {name}/{defect_type}: ROI OK, no submask for AMP")
        return {"name": name, "defect_type": defect_type, "route": "text2roi",
                "status": "ROI_ONLY", "bbox": bbox, "point": point,
                "n_candidates": 1, "seeds": {}}

    roi_mask_path = str(assets / "roi_mask.png")
    submask_path = preprocess_submask(submask_path, sample)
    submask_stem = Path(submask_path).stem
    base_area = mask_area_from_path(submask_path)
    min_area = int(base_area * min_area_ratio)

    seed_results = {}

    for seed_i in range(n_seeds):
        final_mask_path = out / f"{submask_stem}__seed{seed_i}.png"
        overlay_path = assets / f"{submask_stem}__seed{seed_i}_overlay.png"

        if final_mask_path.exists() and overlay_path.exists():
            seed_results[seed_i] = {"status": "CACHED"}
            continue

        placed_ok = False
        amp_results = None

        with tempfile.TemporaryDirectory() as seed_work:
            for attempt in range(max_area_retries):
                sub_seed = base_seed + seed_i * max_area_retries + attempt
                amp = AutomaticMaskPlacement(
                    image_width=ori_w, image_height=ori_h,
                    random_seed=sub_seed,
                    separate_rois=False,
                    max_retry_per_mask=15,
                )
                amp.load_combined_rois(roi_image_paths=[roi_mask_path])
                amp_results = amp.process_submask(
                    submask_path=submask_path,
                    n_instances=1,
                    output_dir=seed_work,
                )
                if not amp_results:
                    continue
                area = mask_area_from_path(amp_results[0]["output_path"])
                if area > min_area:
                    placed_ok = True
                    break

            final_area = (mask_area_from_path(amp_results[0]["output_path"])
                          if amp_results else 0)
            if amp_results and final_area > 0:
                shutil.copy(amp_results[0]["output_path"], str(final_mask_path))

                placed_np = cv2.imread(str(final_mask_path), cv2.IMREAD_GRAYSCALE)
                ov = make_amp_overlay(img_np, roi_mask, placed_np, bbox, point)
                Image.fromarray(ov).save(overlay_path)
            else:
                seed_results[seed_i] = {"status": "FAILED"}

    n_ok = sum(1 for s in seed_results.values() if s["status"] in ("OK", "CACHED"))
    log.info(f"  {name}/{defect_type}: ROI OK, {n_ok}/{n_seeds} seeds OK")

    return {
        "name": name, "defect_type": defect_type, "route": "text2roi",
        "status": "OK", "bbox": bbox, "point": point,
        "n_candidates": 1, "seeds": seed_results,
    }


# ── CAD2ROI processing ──────────────────────────────────────────────────────

def process_cad2roi_sample(sample, output_dir, n_seeds, base_seed,
                           n_instances=-1, roi_only=False):
    cad_mask_path = sample["cad_mask"]
    clean_image_path = sample["clean_image"]
    defect_type = sample["defect_type"]
    submask_path = sample.get("submask")
    cad_mask_label = sample["cad_mask_label"]

    # defect_type is "PCB+missing" → cad_defect is "missing"
    cad_defect = defect_type.split("+", 1)[1] if "+" in defect_type else defect_type

    name = Path(cad_mask_path).stem
    out = Path(output_dir) / name / defect_type
    assets = out / "assets"
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    cad_img = np.array(Image.open(cad_mask_path))[:, :, :3]
    clean_img = np.array(Image.open(clean_image_path).convert("RGB"))

    gen = CADToROIGenerator(cad_mask_label)
    candidates = gen.generate_all_candidates(cad_mask_path)
    cands = candidates.get(cad_defect, [])

    if not cands:
        log.warning(f"  {name}/{defect_type}: no ROI candidates, skipping")
        return {"name": name, "defect_type": defect_type, "route": "cad2roi",
                "status": "NO_CANDIDATES", "n_candidates": 0, "seeds": {}}

    # Save ROI overlays to assets
    roi_viz_cad = overlay(cad_img, sum_masks(cands), CAD_ROI_COLOR)
    roi_viz_real = overlay(clean_img, sum_masks(cands), CAD_ROI_COLOR)
    Image.fromarray(roi_viz_cad).save(assets / "roi_cad.png")
    Image.fromarray(roi_viz_real).save(assets / "roi_real.png")

    for i, cand in enumerate(cands):
        Image.fromarray(cand.mask).save(assets / f"roi_{i}_mask.png")

    if roi_only:
        log.info(f"  {name}/{defect_type}: {len(cands)} ROIs (roi-only)")
        return {"name": name, "defect_type": defect_type, "route": "cad2roi",
                "status": "ROI_ONLY", "n_candidates": len(cands), "seeds": {}}

    # AMP placement per seed
    seed_results = {}
    components, img_shape = gen.parser.parse(cad_mask_path)
    # Preprocess submask if needed
    if submask_path:
        submask_path = preprocess_submask(submask_path, sample)
    sub_stem = Path(submask_path).stem if submask_path else ""
    prefix = f"{sub_stem}__" if sub_stem else ""

    if cad_defect == "missing":
        placer = MissingMaskPlacer()
        results = placer.place_all(cands, n_instances=n_instances,
                                   n_seeds=n_seeds, base_seed=base_seed)
        for seed_i, (n_sel, n_placed, mask_bw) in results.items():
            idx = seed_i - 1  # normalize to 0-based
            save_cad_seed(mask_bw, idx, prefix, out, assets, cad_img, clean_img)
            seed_results[idx] = {"status": "OK", "n_selected": n_sel}

    elif cad_defect in ("less_solder", "excess_solder"):
        PlacerCls = ExcessSolderMaskPlacer if cad_defect == "excess_solder" else LessSolderMaskPlacer
        placer = PlacerCls()
        results = placer.place_all(submask_path, cands, img_shape,
                                   n_instances=n_instances,
                                   n_seeds=n_seeds, base_seed=base_seed)
        for seed_i, (n_req, n_placed, mask) in results.items():
            idx = seed_i - 1
            if mask is None:
                seed_results[idx] = {"status": "FAILED",
                                     "n_requested": n_req, "n_placed": 0}
                continue
            save_cad_seed(mask, idx, prefix, out, assets, cad_img, clean_img)
            status = "OK" if n_placed >= n_req else "PARTIAL"
            seed_results[idx] = {"status": status,
                                 "n_requested": n_req, "n_placed": n_placed}

    elif cad_defect == "bridge":
        groups = get_bridge_groups(components, img_shape,
                                   classes=("pad", "solder"),
                                   min_area_abs=gen.parser.min_area_abs)
        placer = BridgeMaskPlacer()
        results = placer.place_all(submask_path, cands, groups, img_shape,
                                   n_instances=n_instances,
                                   n_seeds=n_seeds, base_seed=base_seed)
        for seed_i, (n_req, n_placed, mask) in results.items():
            idx = seed_i - 1
            if mask is None:
                seed_results[idx] = {"status": "FAILED",
                                     "n_requested": n_req, "n_placed": 0}
                continue
            save_cad_seed(mask, idx, prefix, out, assets, cad_img, clean_img)
            status = "OK" if n_placed >= n_req else "PARTIAL"
            seed_results[idx] = {"status": status,
                                 "n_requested": n_req, "n_placed": n_placed}

    n_ok = sum(1 for s in seed_results.values() if s["status"] == "OK")
    log.info(f"  {name}/{defect_type}: {len(cands)} ROIs, {n_ok}/{n_seeds} seeds OK")

    return {
        "name": name, "defect_type": defect_type, "route": "cad2roi",
        "status": "OK", "n_candidates": len(cands), "seeds": seed_results,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified ROI → AMP pipeline (auto-routes cad2roi / text2roi)")
    parser.add_argument("--input", required=True, help="Path to input samples JSON")
    parser.add_argument("--defect-desc", required=True,
                        help="JSONL with roi_prompt_defect_location per defect_type")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n_seeds", type=int, default=1, help="Number of seeds per sample")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--n_instances", type=int, default=-1,
                        help="Number of instances per image for cad2roi (-1 = all ROIs)")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Qwen VL model for text2box")
    parser.add_argument("--max-area-retries", type=int, default=10,
                        help="Max retry attempts per seed for mask placement")
    parser.add_argument("--roi-only", action="store_true",
                        help="Skip AMP stage, only generate ROI")
    args = parser.parse_args()

    # Load inputs
    defect_prompts, spatial_deps = load_defect_descriptions(args.defect_desc)
    log.info(f"Loaded {len(spatial_deps)} defect descriptions from {args.defect_desc}")

    with open(args.input) as f:
        samples = json.load(f)

    # Partition samples into three routes
    text_indices = []
    cad_indices = []
    free_indices = []
    for i, s in enumerate(samples):
        route = get_sample_route(s, spatial_deps)
        if route == "text":
            text_indices.append(i)
        elif route == "cad":
            cad_indices.append(i)
        else:
            free_indices.append(i)

    log.info(f"Loaded {len(samples)} samples "
          f"({len(text_indices)} text2roi, {len(cad_indices)} cad2roi, "
          f"{len(free_indices)} free)")
    log.info(f"Output: {args.output}, n_seeds={args.n_seeds}, base_seed={args.seed}")

    all_results = [None] * len(samples)
    total = len(samples)

    # ── Text2ROI path (GPU-heavy, run first) ─────────────────────────────
    if text_indices:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"\nLoading Text2Box (Qwen) on {device}...")
        detector = Text2BoxDetector(model_id=args.model_id, device=str(device))
        detector.load()
        log.info("Loading SAM2...")
        models = ROIGenerateModels(device)
        sam = models._get_model("sam2")
        log.info("Models ready.\n")

        for idx in text_indices:
            sample = samples[idx]
            log.info(f"[{idx+1}/{total}] {sample['clean_image']} -> {sample['defect_type']}")
            sample_seed = args.seed + idx * args.n_seeds * args.max_area_retries
            result = process_text2roi_sample(
                sample, defect_prompts, detector, sam,
                args.output, args.n_seeds, sample_seed,
                roi_only=args.roi_only,
                max_area_retries=args.max_area_retries,
            )
            all_results[idx] = result

        # Free VRAM
        detector.unload()
        del sam, models, detector
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        log.info("\nText2ROI models unloaded.\n")

    # ── Free path (whole-image ROI, CPU-only) ────────────────────────────
    if free_indices:
        log.info("Processing free (whole-image ROI) samples...\n")
        for idx in free_indices:
            sample = samples[idx]
            log.info(f"[{idx+1}/{total}] {sample['clean_image']} -> {sample['defect_type']}")
            sample_seed = args.seed + idx * args.n_seeds * args.max_area_retries
            result = process_free_sample(
                sample, args.output, args.n_seeds, sample_seed,
                max_area_retries=args.max_area_retries,
            )
            all_results[idx] = result

    # ── CAD2ROI path (CPU-bound) ─────────────────────────────────────────
    if cad_indices:
        log.info("Processing CAD2ROI samples...\n")
        for idx in cad_indices:
            sample = samples[idx]
            log.info(f"[{idx+1}/{total}] {sample['cad_mask']} -> {sample['defect_type']}")
            sample_seed = args.seed + idx * args.n_seeds * args.max_area_retries
            result = process_cad2roi_sample(
                sample, args.output, args.n_seeds, sample_seed,
                n_instances=args.n_instances, roi_only=args.roi_only,
            )
            all_results[idx] = result

    # ── Summary ──────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nSummary saved to {summary_path}")

    # Stats
    log.info(f"\nTotal: {total} samples "
          f"({len(text_indices)} text2roi, {len(cad_indices)} cad2roi, "
          f"{len(free_indices)} free)")

    for route, indices in [("text2roi", text_indices), ("free", free_indices)]:
        if not indices:
            continue
        results = [all_results[i] for i in indices]
        has_roi = sum(1 for r in results if r["status"] not in ("NO_DETECTION",))
        n_ok = sum(
            sum(1 for s in r["seeds"].values() if s["status"] in ("OK", "CACHED"))
            for r in results
        )
        n_st = sum(len(r["seeds"]) for r in results)
        seeds_str = f", {n_ok}/{n_st} seeds OK" if n_st > 0 else ""
        log.info(f"{route}: {has_roi}/{len(results)} with ROI{seeds_str}")

    if cad_indices:
        cad_results = [all_results[i] for i in cad_indices]
        log.info(f"\ncad2roi:")
        for dt in sorted(set(r["defect_type"] for r in cad_results)):
            dt_results = [r for r in cad_results if r["defect_type"] == dt]
            n_with_roi = sum(1 for r in dt_results if r["n_candidates"] > 0)
            n_ok = sum(
                sum(1 for s in r["seeds"].values() if s["status"] == "OK")
                for r in dt_results
            )
            n_st = sum(len(r["seeds"]) for r in dt_results)
            log.info(f"  {dt}: {n_with_roi}/{len(dt_results)} with ROI, "
                  f"{n_ok}/{n_st} seeds OK")


if __name__ == "__main__":
    main()