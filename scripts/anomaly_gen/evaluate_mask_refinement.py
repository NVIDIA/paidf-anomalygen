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
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm

from imaginaire.utils import log
from pseudo_label import bbox as pl_bbox
from pseudo_label import mask as pl_mask
from pseudo_label import utils as pl_utils
from pseudo_label.infosam import build_infosam2
from pseudo_label.iou_metric import MeanIoUMeter


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate mask refinement using the dataset with ground truth masks."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the images directory",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Path to the masks directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Path to the InfoSAM2 checkpoint. If not provided, use the "
            "pretrained SAM2. Default is None."
        ),
    )
    parser.add_argument(
        "--crop_ratio",
        type=float,
        default=2.0,
        help="The expand ratio for the ROI. Default is 2.0.",
    )
    parser.add_argument(
        "--dilate_sizes",
        nargs="+",
        type=int,
        default=[7, 9, 11, 13],
        help=(
            "The candidate sizes for dilation. Default is [7, 9, 11, 13] for "
            "512x512 images. Use [3, 5, 7, 9] for 300x300 images."
        ),
    )
    parser.add_argument(
        "--rectangular_mask",
        action="store_true",
        help="If set, use rectangular mask.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help=(
            "The strength for the mask prompt in mask refinement. Set to 0.0 "
            "to disable this feature. Default is 1.0."
        ),
    )
    parser.add_argument(
        "--fallback_ratio",
        type=float,
        default=0.5,
        help=(
            "The threshold ratio of the area for the refined mask that needs "
            "to be restored to the original one. 0.5 means that if the area of "
            "the refined mask is smaller than the area of the input mask by "
            "50%, the refined mask will be replaced with the input mask. Use "
            "0.0 to disable this mechanism. Default is 0.5."
        ),
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    log.info(f"{args}")

    np.random.seed(42)
    sam2_model = build_infosam2(
        "checkpoints/sam2/sam2.1_hiera_large.pt",
        args.checkpoint_path,
        use_pretrained_sam2=args.checkpoint_path is None,
    )
    predictor = SAM2ImagePredictor(sam2_model)

    # Prepare paths for original masks and generated images.
    original_mask_paths = sorted(pl_utils.get_image_paths(args.mask_dir))
    generated_image_paths = sorted(pl_utils.get_image_paths(args.image_dir))
    if len(original_mask_paths) == 0:
        raise ValueError(f"No original masks found in {args.mask_dir}.")
    if len(original_mask_paths) != len(generated_image_paths):
        raise ValueError(
            f"Number of original masks ({len(original_mask_paths)}) and "
            f"generated images ({len(generated_image_paths)}) do not match."
        )
    log.info(f"Found {len(original_mask_paths)} images and masks.")
    refined_mask_dir = pathlib.Path(args.output_dir)
    os.makedirs(refined_mask_dir, exist_ok=True)

    # IoU metric.
    dilated_running_metric = MeanIoUMeter(n_class=2)
    refined_running_metric = MeanIoUMeter(n_class=2)

    # Fallback records.
    fallback_records = []

    pbar = tqdm(
        zip(original_mask_paths, generated_image_paths),
        total=len(original_mask_paths),
        desc="Refining masks...",
        dynamic_ncols=True,
    )
    for original_mask_path, generated_image_path in pbar:
        image = Image.open(generated_image_path).convert("RGB")
        gt_mask = Image.open(original_mask_path).convert("L")
        corrected_gt_mask = np.where(np.array(gt_mask) > 127, 255, 0).astype(np.uint8)
        gt_mask = Image.fromarray(corrected_gt_mask)

        # Dilate the mask with random kernel size.
        kernel_size = np.random.choice(args.dilate_sizes)
        kernel = np.ones((kernel_size, kernel_size))
        dilated_gt_mask_array = cv2.dilate(np.array(gt_mask), kernel, iterations=1)
        dilated_gt_mask = Image.fromarray(dilated_gt_mask_array)

        # Convert to binary masks.
        gt_mask_array = np.array(gt_mask) > 127
        dilated_gt_mask_array = np.array(dilated_gt_mask) > 127

        instance_masks = pl_mask.cluster_mask(dilated_gt_mask)
        if len(instance_masks) == 0:
            # DBSCAN will fail when only 1 group of pixels is detected.
            instance_masks = [dilated_gt_mask]
        bboxes = pl_bbox.get_bboxes(instance_masks, format="xyxy")

        if args.rectangular_mask:
            for i, (bbox, instance_mask) in enumerate(zip(bboxes, instance_masks)):
                dilated_gt_mask_array[bbox[1] : bbox[3], bbox[0] : bbox[2]] = True

        # Init figure.
        fig = plt.figure(figsize=(30, 7.5))

        # Show original image.
        ax = fig.add_subplot(1, 4, 1)
        ax.imshow(image)
        ax.set_title("Anomaly Image", fontsize=18)
        ax.set_axis_off()

        # Show GT mask.
        gt_image = pl_utils.visualize(image, [None], [None], [gt_mask])
        ax = fig.add_subplot(1, 4, 2)
        ax.imshow(gt_image)
        ax.set_title("GT Mask", fontsize=18)
        ax.set_axis_off()

        # Show dilated mask.
        dilated_image = pl_utils.visualize(image, [None], [None], [dilated_gt_mask])
        ax = fig.add_subplot(1, 4, 3)
        ax.imshow(dilated_image)
        ax.set_title(f"Dilated Mask (kernel_size={kernel_size})", fontsize=18)
        ax.set_axis_off()

        refined_masks = []
        for i, (bbox, instance_mask) in enumerate(zip(bboxes, instance_masks)):
            # Convert to rectangular mask.
            if args.rectangular_mask:
                instance_mask_array = np.array(instance_mask)
                instance_mask_array[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 255
                instance_mask = Image.fromarray(instance_mask_array.astype(np.uint8))

            # Crop the image and calculate the new bbox.
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            long_side = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            crop_size = int(long_side * args.crop_ratio)
            crop_x1 = int(max(center_x - crop_size // 2, 0))
            crop_y1 = int(max(center_y - crop_size // 2, 0))
            crop_x2 = min(crop_x1 + crop_size, image.width)
            crop_y2 = min(crop_y1 + crop_size, image.height)
            cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            cropped_instance_mask = instance_mask.crop(
                (crop_x1, crop_y1, crop_x2, crop_y2)
            )
            new_bbox = (
                bbox[0] - crop_x1,
                bbox[1] - crop_y1,
                bbox[2] - crop_x1,
                bbox[3] - crop_y1,
            )

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                mask_input = pl_mask.compute_sam2_mask_prompt(
                    cropped_instance_mask,
                    strength=float(args.strength),
                    device=predictor.device,
                )

                # SAM2 prediction.
                predictor.set_image(np.array(cropped_image))
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array(new_bbox),
                    mask_input=mask_input,
                    multimask_output=True,
                )
                best_idx = np.argmax(scores)
                masks = masks[best_idx : best_idx + 1]
                masks = (masks > 0).astype(np.float32)
                assert len(masks) == 1

            # Restore original size.
            restored_masks_array = np.zeros(
                (masks.shape[0], image.height, image.width), dtype=masks.dtype
            )
            restored_masks_array[:, crop_y1:crop_y2, crop_x1:crop_x2] = masks
            restored_mask_array = restored_masks_array[0]

            # Post-processing.
            instance_mask_array = (
                np.array(instance_mask, dtype=restored_mask_array.dtype) / 255
            )
            intersection = restored_mask_array * instance_mask_array
            if np.sum(intersection) == 0:
                intersection = restored_mask_array
            restored_mask_array = intersection
            if args.fallback_ratio > 0.0:
                instance_mask_array = (
                    np.array(instance_mask, dtype=restored_mask_array.dtype) / 255
                )
                original_area = np.sum(instance_mask_array)
                refined_area = np.sum(restored_mask_array)
                if refined_area < args.fallback_ratio * original_area:
                    fallback_records.append(
                        (
                            original_mask_path.name,
                            int(original_area),
                            int(refined_area),
                            refined_area / original_area,
                        )
                    )
                    restored_mask_array = instance_mask_array

            # Show predicted masks.
            refined_masks.append(restored_mask_array)

        refined_mask = np.any(np.stack(refined_masks, axis=0), axis=0)

        # Show refined masks.
        refined_image = pl_utils.visualize(
            image, [None], [None], [Image.fromarray(refined_mask).convert("L")]
        )
        ax = fig.add_subplot(1, 4, 4)
        ax.imshow(refined_image)
        ax.set_title("Refined Mask", fontsize=18)
        ax.set_axis_off()

        # Save the figure.
        fig.tight_layout(pad=3)
        fig.savefig(pathlib.Path(refined_mask_dir, original_mask_path.name))
        plt.close(fig)

        # Calculate IoU.
        dilated_running_metric.update_cm(
            pr=dilated_gt_mask_array.astype(np.uint8), gt=gt_mask_array.astype(np.uint8)
        )
        refined_running_metric.update_cm(
            pr=refined_mask.astype(np.uint8), gt=gt_mask_array.astype(np.uint8)
        )

    # Save evaluation metrics.
    # metric_name_0: background, metric_name_1: anomaly
    scores = dilated_running_metric.get_scores()
    dilated_metrics = {
        "miou": float(scores["iou_1"]),
        "mf1": float(scores["f1_1"]),
        "mprecision": float(scores["precision_1"]),
        "mrecall": float(scores["recall_1"]),
    }
    scores = refined_running_metric.get_scores()
    refined_metrics = {
        "miou": float(scores["iou_1"]),
        "mf1": float(scores["f1_1"]),
        "mprecision": float(scores["precision_1"]),
        "mrecall": float(scores["recall_1"]),
    }
    with open(pathlib.Path(refined_mask_dir, "evaluation_metrics.json"), "w") as f:
        f.write(
            json.dumps(
                {
                    "dilated_metrics": dilated_metrics,
                    "refined_metrics": refined_metrics,
                },
                indent=4,
            )
        )
    log.info(f"Dilated masks merics:\n{dilated_metrics}")
    log.info(f"Refined masks merics:\n{refined_metrics}")
    with open(pathlib.Path(refined_mask_dir, "fallback_records.csv"), "w") as f:
        f.write("image_name,original_area,refined_area,ratio\n")
        for record in fallback_records:
            f.write(f"{record[0]},{record[1]},{record[2]},{record[3]:.3f}\n")


if __name__ == "__main__":
    main(get_args())
