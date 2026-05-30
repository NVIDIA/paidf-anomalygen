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
This script is used to finetune the InfoSAM2 model on a custom dataset.

The finetuned model will be further used for mask refinement in Pseudo-labeling.

Usage:

python3 -m scripts.anomaly_gen.finetune_sam \
        --image_dir path/to/image \
        --mask_dir path/to/mask \
        --output_dir path/to/output_dir

python3 -m scripts.anomaly_gen.finetune_sam -h
"""

import argparse
import os
import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.transforms import SAM2Transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from imaginaire.utils import log
from pseudo_label import bbox as pl_bbox
from pseudo_label import mask as pl_mask
from pseudo_label import utils as pl_utils
from pseudo_label.infosam import (
    DualMiLoss,
    InfoSAM2Dataset,
    InfoSAM2EvalDataset,
    RelationModel,
    StructureLoss,
    build_infosam2,
    get_parameter_names,
    prepare_prompts,
    secure_uniform,
)
from pseudo_label.iou_metric import MeanIoUMeter


def get_args():
    parser = argparse.ArgumentParser(
        description="Finetune SAM2 based on an information-theoretic approach."
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
        "--train_val_ratio",
        type=float,
        default=0.8,
        help=(
            "The ratio to split the train/val dataset. When the total size of "
            "the dataset is less than 100 samples, the entire dataset will be "
            "used as the validation set. Default is 0.8."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs for training. Default is 20.",
    )
    parser.add_argument(
        "--patient_epochs",
        type=int,
        default=5,
        help="Number of epochs to wait without validation metric improvement before early stopping. Default is 5.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The batch size for training. Default is 4.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="The learning rate. Default is 2e-4.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="The weight decay. Default is 1e-4.",
    )
    parser.add_argument(
        "--eval_dilate_sizes",
        nargs="+",
        type=int,
        default=[7, 9, 11, 13],
        help=(
            "The candidate sizes for dilation during evaluation. "
            "Default is [7, 9, 11, 13] for 512x512 images. "
            "Recommended to use [3, 5, 7, 9] for 300x300 images."
        ),
    )
    parser.add_argument(
        "--eval_dbscan_eps",
        type=float,
        default=0.2,
        help=(
            "DBSCAN eps parameter for clustering masks during evaluation. "
            "Default is 0.2."
        ),
    )
    parser.add_argument(
        "--eval_dbscan_min_samples",
        type=int,
        default=5,
        help=(
            "DBSCAN min_sample parameter for clustering masks during evaluation. "
            "Default is 5."
        ),
    )
    parser.add_argument(
        "--eval_crop_ratio",
        type=float,
        default=2.0,
        help="The expand ratio for the ROI during evaluation. Default is 2.0.",
    )
    parser.add_argument(
        "--eval_strength",
        type=float,
        default=1.0,
        help=(
            "The strength for the mask prompt during evaluation. Set to 0.0 "
            "to disable this feature. Default is 1.0."
        ),
    )
    parser.add_argument(
        "--eval_fallback_ratio",
        type=float,
        default=0.5,
        help=(
            "The threshold ratio of the area for the refined mask that needs "
            "to be restored to the original one during evaluation. 0.5 means "
            "that if the area of the refined mask is smaller than the area of "
            "the input mask by 50%, the refined mask will be replaced with the "
            "input mask. Use 0.0 to disable this mechanism. Default is 0.5."
        ),
    )
    return parser.parse_args()


def get_datasets(
    image_dir: pathlib.Path,
    mask_dir: pathlib.Path,
    image_size: int,
    train_val_ratio: float,
    image_transforms: SAM2Transforms,
    eval_dilate_sizes: list[int],
):
    is_insufficient_data = False
    image_paths = sorted(pl_utils.get_image_paths(image_dir))
    mask_paths = sorted(pl_utils.get_image_paths(mask_dir))
    if len(image_paths) != len(mask_paths):
        raise ValueError(
            "Mismatch in number of images and masks. "
            f"images: {len(image_paths)}, masks: {len(mask_paths)}"
        )
    random_indices = np.argsort([secure_uniform() for _ in range(len(image_paths))])
    if len(image_paths) < 100:
        log.info(
            f"Dataset size is too small: {len(image_paths)} < 100. Using all "
            f"data for training and validation."
        )
        is_insufficient_data = True
        train_indices = random_indices
        val_indices = random_indices
    else:
        train_indices = random_indices[: int(train_val_ratio * len(random_indices))]
        val_indices = random_indices[len(train_indices) :]

    train_dataset = InfoSAM2Dataset(
        image_paths=[image_paths[i] for i in train_indices],
        mask_paths=[mask_paths[i] for i in train_indices],
        size=image_size,
        training=True,
        image_transforms=image_transforms,
    )
    val_dataset = InfoSAM2EvalDataset(
        image_paths=[image_paths[i] for i in val_indices],
        mask_paths=[mask_paths[i] for i in val_indices],
        size=image_size,
        image_transforms=image_transforms,
        dilate_sizes=eval_dilate_sizes,
    )
    return train_dataset, val_dataset, is_insufficient_data


def create_models(checkpoint: str):
    stages = [2, 6, 36, 4]
    adapter_config = [0 for _ in range(sum(stages))]
    for i in range(stages[0]):
        adapter_config[i] = -1
    adapter_mlp_ratio = [0.25 for _ in range(sum(stages) * 2)]
    teacher = build_infosam2(checkpoint)
    student = build_infosam2(
        checkpoint, adapter_config=adapter_config, adapter_mlp_ratio=adapter_mlp_ratio
    )
    student.train()
    relation_model = RelationModel()
    relation_model.to(device=student.device)
    relation_model.train()
    # Freeze the model except the adapter, layernorm and decoder.
    for p in teacher.parameters():
        p.requires_grad = False
    for p in student.parameters():
        p.requires_grad = False
    unfrozen_adapter_count = 0
    for block in student.image_encoder.trunk.blocks:  # Adapter.
        for adapter in block.adapters_list:
            unfrozen_adapter_count += 1
            for p in adapter.parameters():
                p.requires_grad = True
    log.info(f"Unfrozen Adapter count: {unfrozen_adapter_count}")
    unfrozen_layernorm_count = 0
    for block in student.image_encoder.trunk.blocks:  # LayerNorm.
        if len(block.adapters_list) == 0:
            continue
        for module in block.modules():
            if isinstance(module, nn.LayerNorm):
                unfrozen_layernorm_count += 1
                for p in module.parameters():
                    p.requires_grad = True
    log.info(f"Unfrozen LayerNorm count: {unfrozen_layernorm_count}")
    for p in student.sam_mask_decoder.parameters():  # Mask decoder.
        p.requires_grad = True
    return student, teacher, relation_model


def create_optimizer(
    student: nn.Module,
    relation_model: nn.Module,
    total_training_steps: int,
    lr: float,
    weight_decay: float,
):
    student_decay_parameters = get_parameter_names(student)
    relation_model_decay_parameters = get_parameter_names(relation_model)
    parameters = [
        # Student.
        {
            "params": [
                p
                for n, p in student.named_parameters()
                if (n in student_decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in student.named_parameters()
                if (n not in student_decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
        # Relation model.
        {
            "params": [
                p
                for n, p in relation_model.named_parameters()
                if (n in relation_model_decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in relation_model.named_parameters()
                if (n not in relation_model_decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_training_steps, eta_min=lr / 10.0
    )
    return optimizer, scheduler


def mask_refinement(
    predictor: SAM2ImagePredictor,
    instance_mask: Image.Image,
    image: Image.Image,
    bbox: typing.Tuple[int, int, int, int],
    crop_ratio: float,
    strength: float,
    fallback_ratio: float,
) -> Image.Image:
    cropped_bbox = pl_bbox.compute_cropped_bbox(
        bbox, instance_mask.height, instance_mask.width, crop_ratio
    )
    cropped_image = image.crop(cropped_bbox)
    cropped_instance_mask = instance_mask.crop(cropped_bbox)
    box_prompt = (
        bbox[0] - cropped_bbox[0],
        bbox[1] - cropped_bbox[1],
        bbox[2] - cropped_bbox[0],
        bbox[3] - cropped_bbox[1],
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
        predictor.set_image(np.array(cropped_image))
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
        image.height,
        image.width,
        cropped_bbox,
        fallback_ratio,
    )


def main(args: argparse.Namespace):
    log.info(f"{args}")

    # Set parameters.
    save_dir = pathlib.Path(args.output_dir)
    train_val_ratio = args.train_val_ratio
    batch_size = args.batch_size
    num_workers = 4
    lr = args.lr
    weight_decay = args.weight_decay
    max_norm = 10.0
    epochs = args.epochs
    patient_epochs = args.patient_epochs
    # InfoSAM.
    image_size = 1024
    mask_num = 1
    checkpoint = "./checkpoints/sam2/sam2.1_hiera_large.pt"

    os.makedirs(save_dir, exist_ok=True)
    # Instantiate the datasets.
    image_transforms = SAM2Transforms(
        resolution=image_size,
        mask_threshold=0,
        max_hole_area=0,
        max_sprinkle_area=0,
    )
    train_dataset, val_dataset, is_insufficient_data = get_datasets(
        pathlib.Path(args.image_dir),
        pathlib.Path(args.mask_dir),
        image_size,
        train_val_ratio,
        image_transforms,
        args.eval_dilate_sizes,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    log.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Instantiate the model.
    student, teacher, relation_model = create_models(checkpoint)

    # Instantiate the optimizer.
    optimizer, scheduler = create_optimizer(
        student, relation_model, epochs * len(train_dataloader), lr, weight_decay
    )

    # Instantiate the losses.
    structure_loss = StructureLoss()
    rkd_loss = DualMiLoss(alpha=1.0, beta=0.5)

    # Training loop.
    optimizer.zero_grad()
    best_miou = -1.0
    best_epoch = 0
    for epoch in range(epochs):
        log.info(f"Epoch {epoch + 1}/{epochs}")

        # Training.
        student.train()
        relation_model.train()
        train_pbar = tqdm(
            enumerate(train_dataloader),
            desc="Training...",
            total=len(train_dataloader),
            dynamic_ncols=True,
        )
        for idx, data in train_pbar:
            optimizer.zero_grad()
            images = data[0].to(student.device)
            gt_masks = data[1].to(device=student.device)
            box_coords = data[2].to(device=student.device)
            box_labels = data[3].to(device=student.device)
            point_coords = data[4].to(device=student.device)
            point_labels = data[5].to(device=student.device)
            orig_hws = data[6]

            # Student forward.
            batch_size = images.shape[0]
            attn_features, high_res_features = student.get_image_embedding(images)
            image_embeddings_repeat = []
            for i in range(batch_size):
                image_embed = attn_features[i]
                image_embed = image_embed.repeat(mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            image_embeddings = torch.cat(image_embeddings_repeat, dim=0)
            if secure_uniform() > 0.5:
                input_point_coords = box_coords
                input_point_labels = box_labels
            else:
                input_point_coords = point_coords
                input_point_labels = point_labels
            input_points = prepare_prompts(
                image_transforms,
                orig_hws,
                input_point_coords,
                input_point_labels,
                device=student.device,
            )
            low_res_masks, iou_predictions, infosam_mask_tokens_out = (
                student.infosam_decode(
                    image_embeddings,
                    high_res_features,
                    input_points=input_points,
                )
            )
            pred_masks = image_transforms.postprocess_masks(low_res_masks, image_size)

            # Teacher forward.
            with torch.no_grad():
                attn_features_t, high_res_features_t = teacher.get_image_embedding(
                    images
                )
                image_embeddings_t_repeat = []
                for i in range(batch_size):
                    image_embed_t = attn_features_t[i]
                    image_embed_t = image_embed_t.repeat(mask_num, 1, 1, 1)
                    image_embeddings_t_repeat.append(image_embed_t)
                image_embeddings_t = torch.cat(image_embeddings_t_repeat, dim=0)
                low_res_masks_t, _, infosam_mask_tokens_out_t = teacher.infosam_decode(
                    image_embeddings_t,
                    high_res_features_t,
                    input_points=input_points,
                )

            # Compute loss.
            loss_structure = structure_loss(pred_masks, gt_masks)
            bs = image_embeddings.shape[0]
            d_model = image_embeddings.shape[1]
            teacher_inputs = (
                image_embeddings_t.permute(0, 2, 3, 1).reshape(bs, -1, d_model),
                infosam_mask_tokens_out_t,
                low_res_masks_t,
            )
            student_inputs = (
                image_embeddings.permute(0, 2, 3, 1).reshape(bs, -1, d_model),
                infosam_mask_tokens_out,
                low_res_masks,
            )
            loss_rkd = rkd_loss(student_inputs, teacher_inputs, relation_model)
            loss = loss_structure + loss_rkd

            # Backprop.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=max_norm)
            torch.nn.utils.clip_grad_norm_(
                relation_model.parameters(), max_norm=max_norm
            )
            optimizer.step()
            scheduler.step()

            # Display loss.
            train_pbar.set_postfix({"loss": f"{loss.item():.3f}"})
        train_pbar.close()

        # Validation.
        visualization_dir = pathlib.Path(
            save_dir, f"epoch{epoch + 1:02d}_visualization"
        )
        os.makedirs(visualization_dir, exist_ok=True)
        student.eval()
        predictor = SAM2ImagePredictor(student)
        dilated_running_metric = MeanIoUMeter(n_class=2)
        refined_running_metric = MeanIoUMeter(n_class=2)
        val_pbar = tqdm(
            val_dataset,
            desc="Validating...",
            total=len(val_dataset),
            dynamic_ncols=True,
        )
        for data in val_pbar:
            image, gt_mask, dilated_gt_mask, kernel_size, mask_path_name = data
            instance_masks = pl_mask.cluster_mask(
                dilated_gt_mask,
                eps=float(args.eval_dbscan_eps),
                min_samples=int(args.eval_dbscan_min_samples),
            )
            bboxes = pl_bbox.get_bboxes(instance_masks, format="xyxy")

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
                refined_mask = mask_refinement(
                    predictor,
                    instance_mask,
                    image,
                    bbox,
                    float(args.eval_crop_ratio),
                    float(args.eval_strength),
                    float(args.eval_fallback_ratio),
                )
                refined_masks.append(np.array(refined_mask))
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
            fig.savefig(pathlib.Path(visualization_dir, mask_path_name))
            plt.close(fig)

            # Calculate IoU.
            gt_mask_array = np.array(gt_mask) > 127
            dilated_gt_mask_array = np.array(dilated_gt_mask) > 127
            dilated_running_metric.update_cm(
                pr=dilated_gt_mask_array.astype(np.uint8),
                gt=gt_mask_array.astype(np.uint8),
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
        val_pbar.close()

        # Log.
        log.info(f"Visualization saved to {visualization_dir}")
        log.info(f"Dilated masks metrics: {dilated_metrics}")
        log.info(f"Refined masks metrics: {refined_metrics}")

        # Save the best model.
        current_miou = refined_metrics["miou"]
        if current_miou > best_miou:
            best_miou = current_miou
            best_epoch = epoch + 1
            save_path = pathlib.Path(
                save_dir, f"epoch{epoch + 1}_iou{current_miou:.3f}.pt"
            )
            torch.save({"model": student.state_dict()}, save_path)
            torch.save(
                {"model": student.state_dict()}, pathlib.Path(save_dir, "best.pt")
            )
            log.info(
                f"Saved the model to {str(save_path)} and "
                f"{str(save_path.with_name('best.pt'))}."
            )
        log.info(
            f"epoch {epoch + 1}, mIoU: {current_miou:.3f}, "
            f"best mIoU: {best_miou:.3f}, best epoch: {best_epoch}"
        )
        if (epoch + 1) - best_epoch >= patient_epochs:
            log.info("Early stopping.")
            break

    log.info("Finished finetuning.")
    if is_insufficient_data:
        log.warning(
            "Since the dataset is insufficient, the validation set is the whole "
            "dataset. You must verify the finetuned SAM2 model using the "
            "evaluation script before utilizing it for mask refinement."
        )


if __name__ == "__main__":
    main(get_args())
