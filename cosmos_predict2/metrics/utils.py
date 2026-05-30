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
from collections import defaultdict

from PIL import Image
from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import torch
from torchvision import transforms

from cosmos_predict2.models.ag_modules.backbone_v2 import c_radio_v3_vit_base_patch16_reg4_dinov2
from cosmos_predict2.models.ag_modules.backbone_v2.ptm_util import load_pretrained_weights
from cosmos_predict2.metrics.fid import compute_fid_on_feats
from cosmos_predict2.metrics.giqa import compute_giqa_on_feats
from cosmos_predict2.metrics.correspondence import compute_correspondence_kpi, DEFAULT_BACKBONE
from imaginaire.utils import log

BACKBONES = {
    "cradio_v3_base": {
        "builder": lambda res: c_radio_v3_vit_base_patch16_reg4_dinov2(resolution=res),
        "ckpt": "checkpoints/nvidia/C-RADIO-V3/model.safetensors",
    },
}

# Module-level cache: {model_id: model} — models live on GPU.
_model_cache: dict = {}


def compute_kpi(real_images_dict, generated_images_dict, backbone_name="cradio_v3_base", correspondence_backbone=DEFAULT_BACKBONE, correspondence_top_k=3):
    result = {}
    fids = []

    for anomaly_name in real_images_dict.keys():
        result[anomaly_name] = {}

        mask_crop_images(real_images_dict[anomaly_name], "original_image")
        mask_crop_images(generated_images_dict[anomaly_name], "reconstructed_image")

        real_crops = real_images_dict[anomaly_name]["mask_cropped_image"]
        gen_crops = generated_images_dict[anomaly_name]["mask_cropped_image"]

        # Single ≥2 gate after feature extraction. Covers (a) raw image count
        # <= 1, (b) mask cropping produced zero crops, (c) feature-extraction
        # loss. compute_feats raises on empty input, so guard those cases.
        feats_real = compute_feats(real_crops, backbone_name=backbone_name) if real_crops else torch.empty(0)
        feats_gen  = compute_feats(gen_crops,  backbone_name=backbone_name) if gen_crops  else torch.empty(0)

        log.info(
            f"[{anomaly_name}] feature counts -> real: {feats_real.size(0)}, generated: {feats_gen.size(0)}"
        )

        if feats_real.size(0) <= 1 or feats_gen.size(0) <= 1:
            log.warning(
                f"[{anomaly_name}] Skipping FID: not enough defect features "
                f"(real: {feats_real.size(0)}, generated: {feats_gen.size(0)}; need at least 2 each)."
            )
            result[anomaly_name][f"{backbone_name}_fid"] = None
            continue

        fid = compute_fid_on_feats(feats_real, feats_gen)

        result[anomaly_name][f"{backbone_name}_fid"] = fid
        fids.append(fid)

    avg_fid = sum(fids) / len(fids) if fids else None

    result["Average"] = {
        f"{backbone_name}_fid": avg_fid,
    }

    # Correspondence scores (DINOv2 ViT-L/14 patch features).
    corr = compute_correspondence_kpi(
        real_images_dict, generated_images_dict,
        backbone=correspondence_backbone,
        top_k=correspondence_top_k,
    )
    for anomaly_name in sorted(real_images_dict.keys()):
        result[anomaly_name]["nn_score"]   = corr.get(anomaly_name, {}).get("nn_score",  float("nan"))
        result[anomaly_name]["mnn_score"]  = corr.get(anomaly_name, {}).get("mnn_score", float("nan"))
        result[anomaly_name]["per_sample"] = corr.get(anomaly_name, {}).get("per_sample", [])
    if "Average" in corr:
        result["Average"]["nn_score"]  = corr["Average"]["nn_score"]
        result["Average"]["mnn_score"] = corr["Average"]["mnn_score"]

    return result

def compute_kpi_per_view_multiview(real_images_dict, generated_images_dict, num_views, backbone_name="cradio_v3_base"):
    """
    Compute KPI separately for each view in multi-view validation.
    
    This function groups images by view index and computes FID for each view independently.
    
    Args:
        real_images_dict: Dictionary with structure {anomaly_name: {key: [images]}}
                         where images are ordered as [sample0_view0, sample0_view1, ..., sample0_viewN-1, 
                                                      sample1_view0, sample1_view1, ..., sample1_viewN-1, ...]
        generated_images_dict: Same structure as real_images_dict
        num_views: Number of views (T)
        backbone_name: Backbone model name for feature extraction
    
    Returns:
        Dictionary with structure:
        {
            anomaly_name: {
                view0: {fid: ...},
                view1: {fid: ...},
                ...
            },
            "Average": {
                view0: {fid: ...},
                view1: {fid: ...},
                ...
            }
        }
    """
    result = {}
    all_view_fids = {}  # {view_idx: [fid1, fid2, ...]}
    
    for view_idx in range(num_views):
        all_view_fids[view_idx] = []
    
    for anomaly_name in real_images_dict.keys():
        if len(generated_images_dict[anomaly_name]["reconstructed_image"]) <= 1:
            raise ValueError(f"[{anomaly_name}] Not enough generated images to compute FID (count={len(generated_images_dict[anomaly_name]['reconstructed_image'])}).")
        
        # Group images by view index
        # Images are ordered as: [sample0_view0, sample0_view1, ..., sample0_viewN-1, sample1_view0, ...]
        # So view_idx images are at indices: view_idx, view_idx + num_views, view_idx + 2*num_views, ...
        num_real_images = len(real_images_dict[anomaly_name]["original_image"])
        num_gen_images = len(generated_images_dict[anomaly_name]["reconstructed_image"])
        
        # Calculate number of samples (should be the same for real and generated)
        num_real_samples = num_real_images // num_views
        num_gen_samples = num_gen_images // num_views
        if num_real_images % num_views != 0:
            raise ValueError(f"[{anomaly_name}] Number of real images ({num_real_images}) is not divisible by num_views ({num_views}). Some images may be skipped.")
        if num_gen_images % num_views != 0:
            raise ValueError(f"[{anomaly_name}] Number of generated images ({num_gen_images}) is not divisible by num_views ({num_views}). Some images may be skipped.")

        log.info(f"[{anomaly_name}] Computing per-view KPI: {num_gen_samples} generated samples, {num_real_samples} real samples, {num_views} views per sample")
        
        result[anomaly_name] = {}
        
        for view_idx in range(num_views):
            # Extract images for this view from all samples
            view_real_dict = {
                "original_image": [],
                "original_mask": [],
            }
            view_gen_dict = {
                "reconstructed_image": [],
                "original_mask": [],  # Need masks for mask_crop_images
            }
            
            # Collect images for this view from all samples
            # Use max to process all available samples (FID doesn't require equal counts)
            for sample_idx in range(max(num_real_samples, num_gen_samples)):
                global_idx = sample_idx * num_views + view_idx
                if global_idx < len(real_images_dict[anomaly_name]["original_image"]):
                    view_real_dict["original_image"].append(real_images_dict[anomaly_name]["original_image"][global_idx])
                    view_real_dict["original_mask"].append(real_images_dict[anomaly_name]["original_mask"][global_idx])
                if global_idx < len(generated_images_dict[anomaly_name]["reconstructed_image"]):
                    view_gen_dict["reconstructed_image"].append(generated_images_dict[anomaly_name]["reconstructed_image"][global_idx])
                    # Also collect the corresponding mask for generated images
                    if "original_mask" in generated_images_dict[anomaly_name] and global_idx < len(generated_images_dict[anomaly_name]["original_mask"]):
                        view_gen_dict["original_mask"].append(generated_images_dict[anomaly_name]["original_mask"][global_idx])
            
            # Check if we have enough images for this view
            if len(view_gen_dict["reconstructed_image"]) <= 1:
                log.warning(f"[{anomaly_name}] View {view_idx}: Not enough images to compute FID (count={len(view_gen_dict['reconstructed_image'])}). Skipping.")
                result[anomaly_name][f"view{view_idx}"] = {f"{backbone_name}_fid": None}
                continue
            
            # Compute mask_cropped_image for this view
            mask_crop_images(view_real_dict, "original_image")
            mask_crop_images(view_gen_dict, "reconstructed_image")
            
            # Check if mask_cropped_image is empty
            mask_cropped_real = view_real_dict.get("mask_cropped_image", [])
            mask_cropped_gen = view_gen_dict.get("mask_cropped_image", [])
            
            if not mask_cropped_real or len(mask_cropped_real) == 0:
                log.warning(f"[{anomaly_name}] View {view_idx}: No cropped images found in real images (all masks may be empty). Skipping FID computation.")
                result[anomaly_name][f"view{view_idx}"] = {f"{backbone_name}_fid": None}
                continue
            if not mask_cropped_gen or len(mask_cropped_gen) == 0:
                log.warning(f"[{anomaly_name}] View {view_idx}: No cropped images found in generated images (all masks may be empty). Skipping FID computation.")
                result[anomaly_name][f"view{view_idx}"] = {f"{backbone_name}_fid": None}
                continue
            
            # Compute features for this view
            feats_real = compute_feats(mask_cropped_real, backbone_name=backbone_name)
            feats_gen = compute_feats(mask_cropped_gen, backbone_name=backbone_name)
            
            log.info(
                f"[{anomaly_name}] View {view_idx}: feature counts -> real: {feats_real.size(0)}, generated: {feats_gen.size(0)}"
            )
            
            # Check if we have enough features to compute FID (need at least 2 for each)
            if feats_real.size(0) <= 1:
                log.warning(f"[{anomaly_name}] View {view_idx}: Not enough real features to compute FID (count={feats_real.size(0)}). Skipping.")
                result[anomaly_name][f"view{view_idx}"] = {f"{backbone_name}_fid": None}
                continue
            if feats_gen.size(0) <= 1:
                log.warning(f"[{anomaly_name}] View {view_idx}: Not enough generated features to compute FID (count={feats_gen.size(0)}). Skipping.")
                result[anomaly_name][f"view{view_idx}"] = {f"{backbone_name}_fid": None}
                continue
            
            # Compute FID for this view
            fid = compute_fid_on_feats(feats_real, feats_gen)
            result[anomaly_name][f"view{view_idx}"] = {f"{backbone_name}_fid": fid}
            all_view_fids[view_idx].append(fid)
    
    # Compute average FID per view across all anomaly types
    result["Average"] = {}
    for view_idx in range(num_views):
        view_fids = [fid for fid in all_view_fids[view_idx] if fid is not None]
        avg_fid = sum(view_fids) / len(view_fids) if view_fids else None
        result["Average"][f"view{view_idx}"] = {f"{backbone_name}_fid": avg_fid}
    
    return result

def compute_sample_wise_kpi(generated_images_dict, real_images_dict, K=1, rotation_range=None, rotation_step=0, backbone_name="cradio_v3_base"):
    for anomaly_name in real_images_dict.keys():
        if len(generated_images_dict[anomaly_name]["reconstructed_image"]) < 1:
            raise ValueError(f"[{anomaly_name}] Not enough generated images to compute KPI (count={len(generated_images_dict[anomaly_name]['reconstructed_image'])}).")
        mask_crop_images(real_images_dict[anomaly_name], "original_image")
        mask_crop_images(generated_images_dict[anomaly_name], "reconstructed_image")
        feats_real = compute_feats(
            real_images_dict[anomaly_name]["mask_cropped_image"],
            rotation_range=rotation_range,
            rotation_step=rotation_step,
            backbone_name=backbone_name
        )
        feats_gen  = compute_feats(
            generated_images_dict[anomaly_name]["mask_cropped_image"],
            backbone_name=backbone_name
        )

        log.info(
            f"[{anomaly_name}] feature counts -> real: {feats_real.size(0)}, generated: {feats_gen.size(0)}"
        )

        K = min(K, len(real_images_dict[anomaly_name]["mask_cropped_image"]))

        # compute instance-wise G-IQA score
        giqa_scores = compute_giqa_on_feats(feats_gen, feats_real, K=1)
        
        # convert to image-wise G-IQA by grouping instances
        image_giqa_scores = []
        image_feats = [] 
        feats_gen = feats_gen.cpu()
        inst_counts = generated_images_dict[anomaly_name]["num_instance"]

        offset = 0
        for count in inst_counts:
            if count == 0:
                image_giqa_scores.append(0.0)
                image_feats.append(np.zeros(feats_gen.shape[1]))
            else:
                inst_scores = giqa_scores[offset:offset + count]
                inst_feats  = feats_gen[offset:offset + count]

                image_score = min(inst_scores)  # min-pooling
                image_giqa_scores.append(image_score)

                # pick feature of the min-score instance
                worst_idx = np.argmin(inst_scores)
                pooled_feat = inst_feats[worst_idx]
                image_feats.append(pooled_feat)

            offset += count

        # store per-image G-IQA for this anomaly type
        generated_images_dict[anomaly_name][f"{backbone_name}_giqa"] = image_giqa_scores
        generated_images_dict[anomaly_name]["feature"] = np.stack(image_feats)  # shape (N_img, D)

    return generated_images_dict
        
def log_kpi_table(kpis):
    # Prepare rows. Skip non-scalar values (e.g., per_sample lists attached by
    # compute_kpi) — only numeric KPIs and None render in the table.
    rows = []
    for anomaly_name, values in kpis.items():
        if anomaly_name == "Average":
            continue
        for kpi_name, score in values.items():
            if not isinstance(score, (int, float, type(None))):
                continue
            score_str = f"{score:.4f}" if score is not None else "None"
            rows.append((anomaly_name, kpi_name, score_str))
    for kpi_name, score in kpis["Average"].items():
        if not isinstance(score, (int, float, type(None))):
            continue
        score_str = f"{score:.4f}" if score is not None else "None"
        rows.append(("Average", kpi_name, score_str))

    # Compute max column widths
    col1_width = max(len("Anomaly Type"), max(len(row[0]) for row in rows))
    col2_width = max(len("KPI"), max(len(row[1]) for row in rows))
    col3_width = max(len("Score"), max(len(row[2]) for row in rows))

    # Construct horizontal border and header
    border = f"+{'-' * (col1_width + 2)}+{'-' * (col2_width + 2)}+{'-' * (col3_width + 2)}+"
    header = f"| {'Anomaly Type'.ljust(col1_width)} | {'KPI'.ljust(col2_width)} | {'Score'.ljust(col3_width)} |"

    # Build table lines
    lines = ["KPI Results:", border, header, border]
    for row in rows:
        lines.append(f"| {row[0].ljust(col1_width)} | {row[1].ljust(col2_width)} | {row[2].ljust(col3_width)} |")
    lines.append(border)

    # Log as one string
    log.info("\n".join(lines))

def log_kpi_table_per_view(kpis, num_views):
    """
    Log KPI table for per-view multi-view validation results.
    
    Args:
        kpis: Dictionary with structure {anomaly_name: {view0: {fid: ...}, view1: {fid: ...}, ...}, "Average": {...}}
        num_views: Number of views
    """
    # Prepare rows: [anomaly_name, view, kpi_name, score]
    rows = []
    anomaly_type_averages = {}  # {anomaly_name: {kpi_name: average_score}}
    
    # Process each anomaly type (excluding "Average")
    for anomaly_name, view_dict in kpis.items():
        if anomaly_name == "Average":
            continue
        
        # Collect scores for this anomaly type across all views
        anomaly_scores = {}  # {kpi_name: [scores]}
        
        # Add per-view rows
        for view_key, values in view_dict.items():
            for kpi_name, score in values.items():
                score_str = f"{score:.4f}" if score is not None else "None"
                rows.append((anomaly_name, view_key, kpi_name, score_str))
                
                # Collect scores for averaging
                if score is not None:
                    if kpi_name not in anomaly_scores:
                        anomaly_scores[kpi_name] = []
                    anomaly_scores[kpi_name].append(score)
        
        # Calculate average for this anomaly type (across all views)
        if anomaly_name not in anomaly_type_averages:
            anomaly_type_averages[anomaly_name] = {}
        for kpi_name, scores in anomaly_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                anomaly_type_averages[anomaly_name][kpi_name] = avg_score
            else:
                anomaly_type_averages[anomaly_name][kpi_name] = None
        
        # Add average row for this anomaly type
        for kpi_name, avg_score in anomaly_type_averages[anomaly_name].items():
            score_str = f"{avg_score:.4f}" if avg_score is not None else "None"
            rows.append((f"{anomaly_name} (avg)", "all views", kpi_name, score_str))
    
    # Add Average row (average across anomaly types, per view)
    for view_key, values in kpis["Average"].items():
        for kpi_name, score in values.items():
            score_str = f"{score:.4f}" if score is not None else "None"
            rows.append(("Average", view_key, kpi_name, score_str))
    
    # Compute max column widths
    col1_width = max(len("Anomaly Type"), max(len(row[0]) for row in rows))
    col2_width = max(len("View"), max(len(row[1]) for row in rows))
    col3_width = max(len("KPI"), max(len(row[2]) for row in rows))
    col4_width = max(len("Score"), max(len(row[3]) for row in rows))
    
    # Construct horizontal border and header
    border = f"+{'-' * (col1_width + 2)}+{'-' * (col2_width + 2)}+{'-' * (col3_width + 2)}+{'-' * (col4_width + 2)}+"
    header = f"| {'Anomaly Type'.ljust(col1_width)} | {'View'.ljust(col2_width)} | {'KPI'.ljust(col3_width)} | {'Score'.ljust(col4_width)} |"
    
    # Build table lines
    lines = ["Per-View KPI Results:", border, header, border]
    for row in rows:
        lines.append(f"| {row[0].ljust(col1_width)} | {row[1].ljust(col2_width)} | {row[2].ljust(col3_width)} | {row[3].ljust(col4_width)} |")
    lines.append(border)
    
    # Log as one string
    log.info("\n".join(lines))

def mask_crop_images(single_type_images_dict, image_key, eps=30, min_samples=1):
    if "mask_cropped_image" in single_type_images_dict and len(single_type_images_dict["mask_cropped_image"]) != 0:
        return

    images = single_type_images_dict[image_key]
    masks = single_type_images_dict["original_mask"]
    cropped = []
    num_instance_list = []

    for i in range(len(images)):
        img = images[i]  # 0~1
        mask_bin = (np.array(masks[i]) >= 0.5).astype(np.uint8)

        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        num_instance = 0
        if contours:
            bbox_centers = []
            bboxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                bbox_centers.append([cx, cy])
                bboxes.append([x, y, x + w, y + h])

            if not bbox_centers:
                num_instance_list.append(0)
                continue

            bbox_centers = np.array(bbox_centers)
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(bbox_centers)
            labels = db.labels_
            for label in np.unique(labels):
                cluster_bboxes = [b for i, b in enumerate(bboxes) if labels[i] == label]
                if not cluster_bboxes:
                    continue

                x1 = min(b[0] for b in cluster_bboxes)
                y1 = min(b[1] for b in cluster_bboxes)
                x2 = max(b[2] for b in cluster_bboxes)
                y2 = max(b[3] for b in cluster_bboxes)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                half_size = max(x2 - x1, y2 - y1) // 2

                x1_sq = max(cx - half_size, 0)
                y1_sq = max(cy - half_size, 0)
                x2_sq = min(cx + half_size, img.shape[1])
                y2_sq = min(cy + half_size, img.shape[0])

                if x2_sq <= x1_sq or y2_sq <= y1_sq:
                    continue
                img_crop = img[y1_sq:y2_sq, x1_sq:x2_sq]
                mask_crop = mask_bin[y1_sq:y2_sq, x1_sq:x2_sq].astype(np.float32)
                mask_crop = np.expand_dims(mask_crop, axis=-1)

                img_crop = img_crop * mask_crop

                img_crop_resized = cv2.resize(img_crop, (512, 512), interpolation=cv2.INTER_AREA)
                cropped.append(img_crop_resized)
                num_instance += 1
        num_instance_list.append(num_instance)

    single_type_images_dict["mask_cropped_image"] = cropped
    single_type_images_dict["num_instance"] = num_instance_list

def compute_feats(images, resize_to=(512, 512), rotation_range=None, rotation_step=0, backbone_name="cradio_v3_base"):
    if backbone_name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {backbone_name}. "
                         f"Available: {list(BACKBONES.keys())}")

    if not images:
        raise ValueError("Empty image list passed to compute_feats.")
    transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backbone_name not in _model_cache:
        log.info(f"Loading {backbone_name} for compute_feats...")
        builder = BACKBONES[backbone_name]["builder"]
        ckpt = BACKBONES[backbone_name]["ckpt"]

        model = builder(resize_to)
        state_dict = load_pretrained_weights(ckpt, weights_only=False)
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            raise RuntimeError(f"Missing keys in model: {load_result.missing_keys}")

        model.eval()
        model.to(device)

        compile_opts = dict(mode="default", fullgraph=False)
        if "cradio_v3_base" in backbone_name:
            log.info(f"torch.compile: {backbone_name}")
            model = torch.compile(model, **compile_opts)

        _model_cache[backbone_name] = model
        log.info(f"{backbone_name} loaded successfully.")
    else:
        model = _model_cache[backbone_name]

    features = []
    with torch.no_grad():
        for img in images:

            augmented_images = [img]
            if rotation_range:
                augmented_images = augment_image(img, rotation_range, rotation_step)

            for augmented_img in augmented_images:
                pil = Image.fromarray((augmented_img * 255).astype(np.uint8))
                x = transform(pil).unsqueeze(0).to(device)
                feat = model(x)
                features.append(feat.squeeze(0))
    return torch.stack(features, dim=0)


def augment_image(image, rotation_range=[-15, 15], rotation_step=15):
    augmented_images = []
    angles = np.arange(rotation_range[0],
                       rotation_range[1] + np.sign(rotation_step) * rotation_step,
                       rotation_step, dtype=int)

    # Include the original image
    if 0 not in angles:
        angles = np.sort(np.append(angles, 0))

    H, W, _ = image.shape
    center = (W / 2.0, H / 2.0)
    for ang in angles:
        M = cv2.getRotationMatrix2D(center, ang, 1.0)

        # Rotate image
        rot_img = cv2.warpAffine(
            (image * 255.0).astype(np.uint8),
            M, (W, H),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        ).astype(np.float32) / 255.0

        # Guarantee (H,W,3)
        if rot_img.ndim == 2:  # sometimes warpAffine can drop channel
            rot_img = np.repeat(rot_img[..., None], 3, axis=-1)
        elif rot_img.shape[2] == 1:
            rot_img = np.repeat(rot_img, 3, axis=2)
        augmented_images.append(rot_img)

    return augmented_images

def load_images(image_dir, mask_dir, image_files, mask_files, image_size):
    """Loads and preprocesses images and masks."""
    images, masks = [], []

    for image_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image = Image.open(img_path).convert("RGB").resize(image_size)
        mask = Image.open(mask_path).convert("L").resize(image_size)

        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.where(mask >= 0.5, 1.0, 0.0)

        if mask.max() != 1.0:
            raise ValueError(f"Mask {mask_path} is not binary.")
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Size mismatch: {img_path} vs {mask_path}")

        images.append(image)
        masks.append(mask)

    return images, masks


def load_real_images(dataset_dir, anomaly_types, image_size=(512, 512)):
    """Loads real anomaly images and masks into the dictionary."""
    real_dict = defaultdict(lambda: defaultdict(list))

    for texture, anomaly_type in anomaly_types:
        key = f"{texture}+{anomaly_type}"

        img_dir = os.path.join(dataset_dir, texture, "anomaly_image", anomaly_type)
        mask_dir = os.path.join(dataset_dir, texture, "mask", anomaly_type)

        file_list = sorted(os.listdir(img_dir))
        image_files = [f for f in file_list if not f.lower().endswith("thumbs.db")]

        ext = lambda f: os.path.splitext(f)[1]
        mask_files = [f.replace(ext(f), f"_mask{ext(f)}") for f in image_files]

        imgs, masks = load_images(img_dir, mask_dir, image_files, mask_files, image_size)
        real_dict[key]["original_image"] = imgs
        real_dict[key]["original_mask"] = masks
    
    return real_dict


def load_generated_images(gen_path, anomaly_types, image_size=(512, 512)):
    """Loads generated (reconstructed) images and masks into the dictionary."""
    gen_dict = defaultdict(lambda: defaultdict(list))

    img_dir = os.path.join(gen_path, "reconstructed_image")
    mask_dir = os.path.join(gen_path, "original_mask")
    orig_dir = os.path.join(gen_path, "original_image")

    all_files = sorted(os.listdir(img_dir))

    for texture, anomaly_type in anomaly_types:
        key = f"{texture}+{anomaly_type}"
        file_list = [f for f in all_files if key in f]

        imgs, masks = load_images(img_dir, mask_dir, file_list, file_list, image_size)
        gen_dict[key]["reconstructed_image"] = imgs
        gen_dict[key]["original_mask"] = masks
        gen_dict[key]["img_path"] = [os.path.join(img_dir, file_name) for file_name in file_list]
        gen_dict[key]["mask_path"] = [os.path.join(mask_dir, file_name) for file_name in file_list]
        gen_dict[key]["orig_path"] = [os.path.join(orig_dir, file_name) for file_name in file_list]

    return gen_dict
