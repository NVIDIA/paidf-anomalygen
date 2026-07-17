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

import argparse
import json
import os
import secrets

from cosmos_predict2.inference.anomaly_gen.inpaint_condition import (
    AnomalyInpaintCondition,
)
from cosmos_predict2.inference.anomaly_gen.tsne import sample_by_tsne
from cosmos_predict2.utils.random import secure_randint
from imaginaire.utils import log


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--SDG_RATIO", type=int, default=16)
    parser.add_argument("--GUIDANCE", default="7.0")
    parser.add_argument("--CROP_RATIO", default="2.0")
    parser.add_argument("--POISSON_BLEND", default="False")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help=(
            "Dataset root. Masks are read from "
            "<dataset_dir>/<texture>/mask/<anomaly_type>/*, and the <texture> and "
            "<anomaly_type> folder names form the 'texture+anomaly_type' label. "
            "By default each mask is paired with a clean image of its OWN texture "
            "from <dataset_dir>/<texture>/clean_image/ (multi-texture safe); a "
            "texture with no clean_image/ is an error unless --clean_image_dir is "
            "given."
        ),
    )
    parser.add_argument(
        "--clean_image_dir",
        type=str,
        default=None,
        help=(
            "Optional: a single flat folder of clean images shared across all "
            "textures, overriding the per-texture <texture>/clean_image/ default. "
            "Use for single-texture data, or datasets that keep masks and clean "
            "images in separate roots. With multiple textures this pairs clean "
            "images across textures (a warning is printed)."
        ),
    )
    parser.add_argument("--name", type=str)
    parser.add_argument("--tSNE_sample", action="store_true", default=False)
    parser.add_argument(
        "--disable_augmentation",
        action="store_true",
        default=False,
        help=(
            "Disable random augmentation (shift, rotation, morph). Useful when "
            "masks are already placed via automatic_mask_placement."
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if args.disable_augmentation and args.SDG_RATIO > 1:
        log.warning(
            "--disable_augmentation is enabled with "
            f"--SDG_RATIO={args.SDG_RATIO}. Since augmentation is disabled, "
            "all generations will be identical. Consider setting --SDG_RATIO=1 "
            "to avoid redundant outputs."
        )


def main(args):
    # We are using `secrets` for random sampling, so no need to set random seed.

    DATASET_DIR = args.dataset_dir
    CLEAN_IMAGE_DIR = args.clean_image_dir

    def _list_images(directory):
        return [
            os.path.join(directory, filename)
            for filename in sorted(os.listdir(directory))
            if filename != "Thumbs.db"
            and os.path.isfile(os.path.join(directory, filename))
        ]

    MASK_AND_ANOMALY_TYPES = [
        dict(
            mask_filename=os.path.join(
                DATASET_DIR, texture, "mask", anomaly_type, mask_filename
            ),
            anomaly_type=f"{texture}+{anomaly_type}",
            texture=texture,
        )
        for texture in os.listdir(os.path.join(DATASET_DIR))
        if os.path.isdir(os.path.join(DATASET_DIR, texture, "mask"))
        for anomaly_type in os.listdir(os.path.join(DATASET_DIR, texture, "mask"))
        if os.path.isdir(os.path.join(DATASET_DIR, texture, "mask", anomaly_type))
        for mask_filename in os.listdir(
            os.path.join(DATASET_DIR, texture, "mask", anomaly_type)
        )
        if mask_filename != "Thumbs.db"
    ]
    TEXTURES = sorted({m["texture"] for m in MASK_AND_ANOMALY_TYPES})

    # Resolve clean/OK images. Two modes:
    #   default (per-texture): each mask pairs with a clean image of its OWN
    #     texture from <dataset_dir>/<texture>/clean_image/. A texture with no
    #     clean_image/ is an error. Multi-texture datasets never cross-pair.
    #   --clean_image_dir override: a single flat folder of clean images shared
    #     across all textures (single-texture / split-layout data).
    if CLEAN_IMAGE_DIR is not None:
        SHARED_OK_IMAGES = _list_images(CLEAN_IMAGE_DIR)
        if not SHARED_OK_IMAGES:
            raise ValueError(
                f"No clean images found in --clean_image_dir {CLEAN_IMAGE_DIR!r}."
            )
        OK_IMAGES_BY_TEXTURE = {texture: SHARED_OK_IMAGES for texture in TEXTURES}
        if len(TEXTURES) > 1:
            log.warning(
                f"{len(TEXTURES)} textures found but --clean_image_dir is a single "
                "flat pool; clean images will be paired ACROSS textures. Omit "
                "--clean_image_dir to pair per-texture from "
                "<dataset_dir>/<texture>/clean_image/."
            )
        log.info(
            f"OK images (shared flat pool of {len(SHARED_OK_IMAGES)}): {CLEAN_IMAGE_DIR}"
        )
    else:
        SHARED_OK_IMAGES = None
        OK_IMAGES_BY_TEXTURE = {}
        for texture in TEXTURES:
            clean_dir = os.path.join(DATASET_DIR, texture, "clean_image")
            if not os.path.isdir(clean_dir):
                raise ValueError(
                    f"No clean_image/ for texture {texture!r} at {clean_dir!r}. "
                    "Add <texture>/clean_image/ to the dataset, or pass "
                    "--clean_image_dir to use a single shared folder."
                )
            OK_IMAGES_BY_TEXTURE[texture] = _list_images(clean_dir)
            if not OK_IMAGES_BY_TEXTURE[texture]:
                raise ValueError(f"No clean images found in {clean_dir!r}")
        log.info(
            "OK images (per-texture): "
            + str({t: len(imgs) for t, imgs in OK_IMAGES_BY_TEXTURE.items()})
        )
    log.info(f"Found {len(MASK_AND_ANOMALY_TYPES)} anomaly masks in {DATASET_DIR}")

    # Parameters.
    # Parameters are fixed if there is only one value.
    SDG_RATIO = args.SDG_RATIO
    GUIDANCE = [float(guidance) for guidance in args.GUIDANCE.split(",")]
    NUM_STEPS = [35]
    CROP_AND_PASTE = [True]
    CROP_RATIO = (
        [None]
        if args.CROP_RATIO == "None"
        else [float(crop_ratio) for crop_ratio in args.CROP_RATIO.split(",")]
    )
    CROP_GRID_X = [192] if args.CROP_RATIO == "None" else ["none"]
    CROP_GRID_Y = [192] if args.CROP_RATIO == "None" else ["none"]
    NUM_GENERATED_IMAGES = [1]
    POISSON_BLEND = [
        True if poisson_blend == "True" else False
        for poisson_blend in args.POISSON_BLEND.split(",")
    ]
    SHIFT_VALUES = [-100, 100]
    ROTATION_ANGLE = [0, 180]
    MORPH_OPERATION = ["dilate", "open", "close", "none"]
    ITERATION_GENERATION_MAX_INSTANCE = [5]

    if args.tSNE_sample:
        if SHARED_OK_IMAGES is None:
            raise ValueError(
                "--tSNE_sample requires --clean_image_dir (a single flat pool to "
                "cluster over); it is not supported in per-texture mode."
            )
        train_full_OK_images_by_cluster_sampled = sample_by_tsne(SHARED_OK_IMAGES)
    else:
        train_full_OK_images_by_cluster_sampled = None

    # Write the jsonl file.
    OUTPUT_DIR = f"ag_inference/{args.name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    EXP_NAME = (
        f"{OUTPUT_DIR}/testcase_{args.SDG_RATIO}"
        f"x_guidance={args.GUIDANCE}_crop_ratio={args.CROP_RATIO}"
        f"_poisson_blend={args.POISSON_BLEND}"
    )
    if args.tSNE_sample:
        EXP_NAME = f"{EXP_NAME}_tSNE_sample"
    EXP_NAME = f"{EXP_NAME}.jsonl"
    with open(EXP_NAME, "w") as fp:
        # For each mask and anomaly type, generate SDG_RATIO test cases
        for mask_and_anomaly_type in MASK_AND_ANOMALY_TYPES:
            mask_filename = mask_and_anomaly_type["mask_filename"]
            anomaly_type = mask_and_anomaly_type["anomaly_type"]
            texture = mask_and_anomaly_type["texture"]
            if not os.path.exists(mask_filename):
                raise ValueError(f"Mask file {mask_filename} does not exist")
            # Pair with OK image
            for idx in range(SDG_RATIO):
                if train_full_OK_images_by_cluster_sampled is not None:
                    # First sample by cluster, then sample by image.
                    # This avoids from sampling over and over from the same
                    # cluster.
                    cluster_label = secrets.choice(
                        list(train_full_OK_images_by_cluster_sampled.keys())
                    )
                    OK_image = secrets.choice(
                        train_full_OK_images_by_cluster_sampled[cluster_label]
                    )
                else:
                    OK_image = secrets.choice(OK_IMAGES_BY_TEXTURE[texture])
                ok_image_filename = OK_image
                if not os.path.exists(ok_image_filename):
                    raise ValueError(f"Image file {ok_image_filename} does not exist")

                # Sample parameters
                num_steps = secrets.choice(NUM_STEPS)
                crop_and_paste = secrets.choice(CROP_AND_PASTE)
                guidance = secrets.choice(GUIDANCE)
                crop_ratio = secrets.choice(CROP_RATIO)
                crop_grid_x = secrets.choice(CROP_GRID_X)
                crop_grid_y = secrets.choice(CROP_GRID_Y)
                num_generated_images = secrets.choice(NUM_GENERATED_IMAGES)
                poisson_blend = secrets.choice(POISSON_BLEND)
                iteration_generation_max_instance = secrets.choice(
                    ITERATION_GENERATION_MAX_INSTANCE
                )

                # Disable augmentation if flag is set
                if args.disable_augmentation:
                    shift_values = "0,0"
                    rotation_angle = 0
                    morph_operation = "none"
                else:
                    shift_values = secrets.choice(SHIFT_VALUES)
                    rotation_angle = secrets.choice(ROTATION_ANGLE)
                    morph_operation = secrets.choice(MORPH_OPERATION)

                    shift_min, shift_max = SHIFT_VALUES
                    shift_values_x, shift_values_y = (
                        secure_randint(shift_min, shift_max),
                        secure_randint(shift_min, shift_max),
                    )
                    shift_values = f"{shift_values_x},{shift_values_y}"

                    rotation_min, rotation_max = ROTATION_ANGLE
                    rotation_angle = secure_randint(rotation_min, rotation_max)

                data = {
                    "image_filename": ok_image_filename,
                    "mask_filename": mask_filename,
                    "anomaly_type": anomaly_type,
                    "guidance": guidance,
                    "num_steps": num_steps,
                    "crop_and_paste": crop_and_paste,
                    "crop_ratio": crop_ratio,
                    "crop_grid_X": crop_grid_x,
                    "crop_grid_Y": crop_grid_y,
                    "num_generated_images": num_generated_images,
                    "poisson_blend": poisson_blend,
                    "shift_values": shift_values,
                    "rotation_angle": rotation_angle,
                    "morph_operation": morph_operation,
                    "iteration_generation_max_instance": iteration_generation_max_instance,
                }
                fp.write(json.dumps(data) + "\n")

    # Sanity check.
    with open(EXP_NAME, "r") as fp:
        for line in fp:
            data = json.loads(line)
            try:
                AnomalyInpaintCondition(**data)
            except Exception as e:
                log.error(f"Error: {e}")
    log.info(f"Output file: {EXP_NAME}")


if __name__ == "__main__":
    args = get_args()
    validate_args(args)
    main(args)
