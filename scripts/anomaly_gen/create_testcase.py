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
    parser.add_argument("--OK_image_path", type=str)
    parser.add_argument("--NG_mask_path", type=str)
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

    OK_IMAGE_PATH = args.OK_image_path
    NG_MASK_PATH = args.NG_mask_path
    OK_IMAGE_FILENAMES = [
        os.path.join(OK_IMAGE_PATH, filename)
        for filename in os.listdir(OK_IMAGE_PATH)
        if filename != "Thumbs.db"
    ]
    MASK_AND_ANOMALY_TYPES = [
        dict(
            mask_filename=os.path.join(
                NG_MASK_PATH, texture, "mask", anomaly_type, mask_filename
            ),
            anomaly_type=f"{texture}+{anomaly_type}",
        )
        for texture in os.listdir(os.path.join(NG_MASK_PATH))
        for anomaly_type in os.listdir(os.path.join(NG_MASK_PATH, texture, "mask"))
        for mask_filename in os.listdir(
            os.path.join(NG_MASK_PATH, texture, "mask", anomaly_type)
        )
        if mask_filename != "Thumbs.db"
    ]
    log.info(f"Found {len(OK_IMAGE_FILENAMES)} OK images in {OK_IMAGE_PATH}")
    log.info(f"Found {len(MASK_AND_ANOMALY_TYPES)} anomaly masks in {NG_MASK_PATH}")

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
        train_full_OK_images_by_cluster_sampled = sample_by_tsne(OK_IMAGE_FILENAMES)
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
                    OK_image = secrets.choice(OK_IMAGE_FILENAMES)
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
