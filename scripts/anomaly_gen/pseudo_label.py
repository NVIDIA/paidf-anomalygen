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
Usage:

python3 -m scripts.anomaly_gen.pseudo_label \
        --ori_image_dir path/to/original_image \
        --gen_image_dir path/to/reconstructed_image \
        --mask_dir path/to/original_mask \
        --csv_path path/to/SDG_result.csv \
        --captioner_prompt_path path/to/caption_prompt.yaml \
        --output_dir path/to/output_dir

python3 -m scripts.anomaly_gen.pseudo_label -h
"""

import argparse
import csv
import json
import os
import pathlib
import typing

if "HF_HOME" not in os.environ:
    # CR model will be downloaded to ./checkpoints if HF_HOME is not specified.
    os.environ["HF_HOME"] = "checkpoints"
import numpy as np
import torch
import yaml
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm

from imaginaire.utils import log
from pseudo_label import bbox as pl_bbox
from pseudo_label import caption as pl_caption
from pseudo_label import mask as pl_mask
from pseudo_label import utils as pl_utils
from pseudo_label.infosam import build_infosam2


def get_args():
    parser = argparse.ArgumentParser(
        description="Pseudo-labeling script for anomaly generated data."
    )
    parser.add_argument(
        "--ori_image_dir",
        type=str,
        required=True,
        help="Path to clean images directory",
    )
    parser.add_argument(
        "--gen_image_dir",
        type=str,
        required=True,
        help="Path to generated images directory",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Path to input masks directory",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the SDG CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--no_mask_refinement",
        action="store_true",
        help="If set, do not refine the masks. This will speed up the process.",
    )
    parser.add_argument(
        "--no_caption",
        action="store_true",
        help="If set, do not generate captions. This will speed up the process.",
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=0.2,
        help="DBSCAN eps parameter for clustering masks. Default is 0.2.",
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=5,
        help="DBSCAN min_sample parameter for clustering masks. Default is 5.",
    )
    parser.add_argument(
        "--mask_refinement_checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint for the mask refinement model.",
    )
    parser.add_argument(
        "--mask_refinement_strength",
        type=float,
        default=1.0,
        help=(
            "The strength for the mask prompt in mask refinement. Set to 0.0 "
            "to disable this feature. Default is 1.0."
        ),
    )
    parser.add_argument(
        "--mask_refinement_fallback_ratio",
        type=float,
        default=0.5,
        help=(
            "The threshold ratio of the area for the refined mask that needs "
            "to be filtered. 0.5 means that if the area of "
            "the refined mask is smaller than the area of the input mask by "
            "50%%, the refined mask will be filtered. Set to 0.0 to disable "
            "this feature. Default is 0.5."
        ),
    )
    parser.add_argument(
        "--captioner_prompt_path",
        default="pseudo_label/default_caption_prompt.yaml",
        type=str,
        help="Path to the prompt configuration for captioning.",
    )
    parser.add_argument(
        "--captioner_num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for captioning. Default is 1.",
    )
    parser.add_argument(
        "--captioner_temperature",
        type=float,
        default=0.01,
        help="Captioner temperature parameter for generating captions. Default is 0.01.",
    )
    parser.add_argument(
        "--captioner_max_tokens",
        type=int,
        default=4096,
        help="Captioner max_tokens parameter for generating captions. Default is 4096.",
    )
    parser.add_argument(
        "--captioner_seed",
        type=int,
        default=42,
        help="Captioner seed parameter for generating captions. Default is 42.",
    )
    return parser.parse_args()


def mask_refinement(
    predictor: SAM2ImagePredictor,
    instance_mask: Image.Image,
    generated_image: Image.Image,
    bbox: typing.Tuple[int, int, int, int],
    crop_ratio: float,
    strength: float,
    fallback_ratio: float,
) -> Image.Image:
    bbox_xyxy = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
    cropped_bbox = pl_bbox.compute_cropped_bbox(
        bbox_xyxy,
        instance_mask.height,
        instance_mask.width,
        crop_ratio,
    )
    cropped_generated_image = generated_image.crop(cropped_bbox)
    cropped_instance_mask = instance_mask.crop(cropped_bbox)
    box_prompt = (
        bbox_xyxy[0] - cropped_bbox[0],
        bbox_xyxy[1] - cropped_bbox[1],
        bbox_xyxy[2] - cropped_bbox[0],
        bbox_xyxy[3] - cropped_bbox[1],
    )
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        mask_input = pl_mask.compute_sam2_mask_prompt(
            cropped_instance_mask,
            strength=strength,
            device=predictor.device,
        )
        predictor.set_image(np.array(cropped_generated_image))
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.expand_dims(np.array(box_prompt), axis=0),
            mask_input=mask_input,
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        refined_mask = (masks[best_idx] > 0).astype(np.float32)

    return pl_mask.post_process_sam2_mask(
        refined_mask,
        np.array(instance_mask),
        generated_image.height,
        generated_image.width,
        cropped_bbox,
        fallback_ratio,
    )


def compute_annotation_dict(
    annotation_id: int,
    image_id: int,
    category_id: str,
    instance_mask: Image.Image,
) -> typing.Dict[str, typing.Any]:
    binary_mask = np.array(instance_mask) > 127
    area = int(np.sum(binary_mask))
    uncompressed_rle = pl_mask.binary_mask_to_rle(binary_mask)
    coco_rle = pl_mask.coco_encode_rle(uncompressed_rle)
    # Assert that the RLE encoding is correct.
    np.testing.assert_array_equal(pl_mask.coco_decode_rle(coco_rle), binary_mask)
    bbox = pl_bbox.get_bboxes(instance_mask)[0]
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": coco_rle,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
    }


def save_coco_annotations(
    coco_dict: typing.Dict[str, typing.Any], output_path: pathlib.Path
):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f)
    log.info(
        f"COCO JSON saved to {str(output_path)} with "
        f"{len(coco_dict['annotations'])} annotations."
    )


def main(args: argparse.Namespace):
    log.info(f"{args}")

    # Load images and masks.
    ori_image_paths = sorted(pl_utils.get_image_paths(args.ori_image_dir))
    gen_image_paths = sorted(pl_utils.get_image_paths(args.gen_image_dir))
    mask_paths = sorted(pl_utils.get_image_paths(args.mask_dir))
    if len(ori_image_paths) != len(mask_paths):
        raise ValueError(
            "The number of original images and masks must be the same. "
            f"Found {len(ori_image_paths)} original images and {len(mask_paths)} masks."
        )
    if len(gen_image_paths) != len(mask_paths):
        raise ValueError(
            "The number of generated images and masks must be the same. "
            f"Found {len(gen_image_paths)} generated images and {len(mask_paths)} masks."
        )

    # Get the corresponding anomaly types from the CSV file.
    anomaly_type_map: typing.Dict[str, str] = dict()
    crop_ratio_map: typing.Dict[str, float] = dict()
    with open(args.csv_path, "r") as csv_file:
        rows = csv.DictReader(csv_file)
        for row in rows:
            image_filename = pathlib.Path(str(row["output_filename"]).strip()).name
            anomaly_type = str(row["anomaly_type"]).strip()
            anomaly_type_map[image_filename] = anomaly_type
            crop_ratio = str(row["crop_ratio"]).strip().lower()
            crop_ratio = float(crop_ratio) if crop_ratio != "none" else 2.0
            crop_ratio_map[image_filename] = crop_ratio

    # Creat the output dirs.
    # By default: (images, masks, visualization)
    # If args.no_caption is false: (captions, captions_with_extra_info)
    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = pathlib.Path(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    mask_dir = pathlib.Path(args.output_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    visualization_dir = pathlib.Path(args.output_dir, "visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    if not args.no_caption:
        caption_dir = pathlib.Path(args.output_dir, "captions")
        os.makedirs(caption_dir, exist_ok=True)
        caption_with_meta_dir = pathlib.Path(args.output_dir, "captions_with_meta")
        os.makedirs(caption_with_meta_dir, exist_ok=True)

    # Initialize the mask refinement model.
    if not args.no_mask_refinement:
        sam2_model = build_infosam2(
            "checkpoints/sam2/sam2.1_hiera_large.pt",
            args.mask_refinement_checkpoint_path,
        )
        predictor = SAM2ImagePredictor(sam2_model)

    # Main loop.
    #   COCO format reference:
    #   https://cocodataset.org/#format-data
    #   https://www.v7labs.com/blog/coco-dataset-guide
    images_list = []
    annotations_list = []
    ori_annotations_list = []
    categories_list = []
    captions_list = []

    # Process the categories.
    class_names = set(anomaly_type_map.values())
    category_to_id = {}
    for i, class_name in enumerate(class_names, start=1):
        categories_list.append(
            {
                "id": i,
                "name": class_name,
            }
        )
        category_to_id[class_name] = i

    # Process the images and annotations.
    annotation_id = 1
    for image_id, (gen_image_path, ori_image_path, mask_path) in tqdm(
        enumerate(zip(gen_image_paths, ori_image_paths, mask_paths), start=1),
        desc="Processing images and annotations",
        total=len(gen_image_paths),
        dynamic_ncols=True,
    ):
        gen_image_filename = gen_image_path.name
        anomaly_type = anomaly_type_map[gen_image_filename]
        generated_image = Image.open(gen_image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        instance_masks = pl_mask.cluster_mask(
            mask,
            eps=float(args.dbscan_eps),
            min_samples=int(args.dbscan_min_samples),
        )
        bboxes = pl_bbox.get_bboxes(instance_masks)

        image_dict = {
            "id": image_id,
            "width": mask.width,
            "height": mask.height,
            "file_name": gen_image_filename,
            # "caption": None,  # TODO: Add caption.
        }
        images_list.append(image_dict)
        new_instance_masks = []

        for bbox, instance_mask in zip(bboxes, instance_masks):
            if bbox is None:
                continue

            # Mask refinement.
            if not args.no_mask_refinement:
                # Compute the original annotation dictionary.
                ori_annotation_dict = compute_annotation_dict(
                    annotation_id,
                    image_id,
                    category_to_id[anomaly_type],
                    instance_mask,
                )
                ori_annotations_list.append(ori_annotation_dict)

                instance_mask = mask_refinement(
                    predictor,
                    instance_mask,
                    generated_image,
                    bbox,
                    crop_ratio_map[gen_image_filename],
                    float(args.mask_refinement_strength),
                    float(args.mask_refinement_fallback_ratio),
                )

            new_instance_masks.append(instance_mask)
            annotation_dict = compute_annotation_dict(
                annotation_id,
                image_id,
                category_to_id[anomaly_type],
                instance_mask,
            )
            annotation_id += 1
            annotations_list.append(annotation_dict)

        # Update the bboxes and instance_masks.
        bboxes = pl_bbox.get_bboxes(new_instance_masks)
        instance_masks = new_instance_masks

        # Visualization.
        visualization_image = pl_utils.visualize(
            generated_image,
            [anomaly_type] * len(bboxes),
            bboxes,
            instance_masks,
        )
        visualization_image.save(pathlib.Path(visualization_dir, gen_image_path.name))

        # Save the original image and mask.
        generated_image.save(pathlib.Path(image_dir, gen_image_path.name))
        mask.save(pathlib.Path(mask_dir, mask_path.name))

        if not args.no_caption:
            # Prepare the data for captioning.
            anomaly_type = anomaly_type_map[mask_path.name]
            if "+" in anomaly_type:
                image_type, anomaly_type = anomaly_type.split("+", maxsplit=1)
            else:
                image_type, anomaly_type = "unknown", anomaly_type
            caption_bboxes = pl_bbox.get_bboxes(instance_masks, format="xyxy")
            num_bboxes = len(caption_bboxes)
            bboxes_str = ", ".join(
                [f"({b[0]}, {b[1]}, {b[2]}, {b[3]})" for b in caption_bboxes]
            )
            meta = {
                "image_type": image_type,
                "anomaly_type": anomaly_type,
                "bboxes": bboxes_str,
                "num_bboxes": num_bboxes,
            }
            captions_list.append((ori_image_path, mask_path, gen_image_path, meta))

    # Release the GPU memory if needed.
    if not args.no_mask_refinement:
        del sam2_model
        del predictor
        torch.cuda.empty_cache()

    # COCO JSON.
    coco_output_path = pathlib.Path(args.output_dir, "coco_annotations.json")
    coco_dict = {
        "annotations": annotations_list,
        "images": images_list,
        "categories": categories_list,
    }
    save_coco_annotations(coco_dict, coco_output_path)
    if len(ori_annotations_list) > 0:
        ori_coco_output_path = coco_output_path.with_stem("ori_coco_annotations")
        coco_dict["annotations"] = ori_annotations_list
        save_coco_annotations(coco_dict, ori_coco_output_path)

    # Classification.
    classification_dir = pathlib.Path(args.output_dir, "classification")
    os.makedirs(classification_dir, exist_ok=True)
    # Save the class names to classes.txt.
    classes_txt_path = pathlib.Path(classification_dir, "classes.txt")
    with open(classes_txt_path, "w") as f:
        f.write("original\n")  # Add original class.
        os.makedirs(pathlib.Path(classification_dir, "original"), exist_ok=True)
        for i, class_name in enumerate(
            sorted([item["name"] for item in categories_list])
        ):
            os.makedirs(pathlib.Path(classification_dir, class_name), exist_ok=True)
            f.write(class_name)
            if i != len(categories_list) - 1:
                f.write("\n")
    log.info(
        f"Class names saved to {classes_txt_path} with "
        f"{len(categories_list) + 1} classes."
    )
    # Save images to the classification dir.
    for ori_image_path in ori_image_paths:
        path = pathlib.Path(classification_dir, "original", ori_image_path.name)
        path.write_bytes(ori_image_path.read_bytes())
    for gen_image_path in gen_image_paths:
        anomaly_type = anomaly_type_map[gen_image_path.name]
        path = pathlib.Path(classification_dir, anomaly_type, gen_image_path.name)
        path.write_bytes(gen_image_path.read_bytes())
    log.info(
        f"Images are organized to {classification_dir} with "
        f"{len(ori_image_paths) + len(gen_image_paths)} images."
    )

    # Captioning. Using the batch inference to speed up the process.
    if not args.no_caption:
        # Initialize the captioner.
        with open(args.captioner_prompt_path, "r") as f:
            prompt_data: typing.Dict[str, str] = yaml.safe_load(f)
        if "system_prompt" not in prompt_data or "user_prompt" not in prompt_data:
            raise ValueError(
                "The prompt configuration file must contain 'system_prompt' "
                "and 'user_prompt' fields. You can refer to "
                "`pseudo_label/default_caption_prompt.yaml` for an example."
            )
        captioner_args = {
            "prompt_data": prompt_data,
            "model_name": "nvidia/Cosmos-Reason1-7B",
            # Default parameters from the CR1 captioning example.
            "temperature": float(args.captioner_temperature),
            "max_tokens": int(args.captioner_max_tokens),
            "seed": int(args.captioner_seed),
            "num_gpus": int(args.captioner_num_gpus),
        }
        log.info(f"The captioner arguments:\n{json.dumps(captioner_args, indent=4)}")
        log.info("Initializing the Cosmos Reason captioner...")
        captioner = pl_caption.Captioner(**captioner_args)
        log.info("Successfully initialized the captioner.")

        batch_size = max(1, int(args.captioner_num_gpus))
        total_len = -(-len(captions_list) // batch_size)
        for idx in tqdm(
            range(0, len(captions_list), batch_size),
            desc="Processing captions",
            total=total_len,
            dynamic_ncols=True,
        ):
            batch_data = captions_list[idx : idx + batch_size]
            ori_image_paths = [item[0] for item in batch_data]
            ori_mask_paths = [item[1] for item in batch_data]
            gen_image_paths = [item[2] for item in batch_data]
            metas = [item[3] for item in batch_data]
            responses = captioner.batch_generate_caption(
                ori_image_paths, ori_mask_paths, gen_image_paths, metas
            )
            for response, gen_image_path, meta in zip(
                responses, gen_image_paths, metas
            ):
                response, response_with_meta = pl_caption.format_response(
                    response, meta
                )
                caption_path = caption_dir / f"{gen_image_path.stem}.txt"
                with open(caption_path, "w") as f:
                    f.write(response)
                caption_with_meta_path = (
                    caption_with_meta_dir / f"{gen_image_path.stem}.txt"
                )
                with open(caption_with_meta_path, "w") as f:
                    f.write(response_with_meta)
        log.info(
            f"Captions are saved to {caption_dir} and {caption_with_meta_dir} "
            f"with {len(captions_list)} captions."
        )


if __name__ == "__main__":
    main(get_args())
