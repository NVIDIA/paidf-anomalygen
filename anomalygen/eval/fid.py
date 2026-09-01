# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Defect FID scoring using a frozen C-RADIO v3 ViT-B/16 backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from cosmos_framework.utils import log
from PIL import Image
from sklearn.cluster import DBSCAN
from torchvision import transforms

from anomalygen.models.vision_encoder.cradio import c_radio_v3_vit_base_patch16_reg4_dinov2
from anomalygen.models.vision_encoder.cradio.ptm_util import load_pretrained_weights

_CRADIO_CKPT = str(Path(__file__).resolve().parents[2] / "checkpoints" / "nvidia" / "C-RADIO-V3" / "model.safetensors")
BACKBONES = {
    "cradio_v3_base": {
        "builder": lambda res: c_radio_v3_vit_base_patch16_reg4_dinov2(resolution=res),
        "ckpt": _CRADIO_CKPT,
    },
}

_model_cache: dict = {}


def _sqrtm_psd(mat: torch.Tensor) -> torch.Tensor:
    """Matrix square root of a symmetric PSD matrix via eigendecomposition."""
    mat = (mat + mat.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(mat)
    sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=0))
    return (eigvecs * sqrt_eigvals) @ eigvecs.T


def _compute_fid_on_feats(feats_1: torch.Tensor, feats_2: torch.Tensor) -> float:
    if feats_1 is None or feats_2 is None:
        raise ValueError("One of the feature sets is None for FID computation.")
    if feats_1.shape[0] <= 1 or feats_2.shape[0] <= 1:
        raise ValueError(f"Not enough samples to compute FID: feats_1={feats_1.shape[0]}, feats_2={feats_2.shape[0]}")

    mu1 = feats_1.mean(dim=0)
    mu2 = feats_2.mean(dim=0)
    sigma1 = torch.cov(feats_1.T)
    sigma2 = torch.cov(feats_2.T)

    # Tr(sqrtm(sigma1 @ sigma2)): sigma1 @ sigma2 is non-symmetric, but the sandwich form
    # sigma1^.5 @ sigma2 @ sigma1^.5 is symmetric PSD and shares its eigenvalues.
    sigma1_sqrt = _sqrtm_psd(sigma1)
    covmean = _sqrtm_psd(sigma1_sqrt @ sigma2 @ sigma1_sqrt)

    fid = torch.sum((mu1 - mu2) ** 2) + torch.trace(sigma1 + sigma2 - 2 * covmean)
    return fid.item()


def mask_crop_images(
    single_type_images_dict: dict, image_key: str, eps: int = 30, min_samples: int = 1, crop_size: int = 512
) -> None:
    """Crop a square patch around each masked defect instance into ``mask_cropped_image``.

    Mask contours are clustered (DBSCAN on centres) so nearby blobs form one instance; each
    cluster is cropped, masked, and resized to ``crop_size`` x ``crop_size`` (the FID backbone's input
    resolution; 512 by default). Idempotent."""
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
                cluster_bboxes = [b for j, b in enumerate(bboxes) if labels[j] == label]
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

                img_crop_resized = cv2.resize(img_crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
                cropped.append(img_crop_resized)
                num_instance += 1
        num_instance_list.append(num_instance)

    single_type_images_dict["mask_cropped_image"] = cropped
    single_type_images_dict["num_instance"] = num_instance_list


@torch.inference_mode()
def compute_feats(
    images: List[np.ndarray], resize_to=(512, 512), backbone_name: str = "cradio_v3_base"
) -> torch.Tensor:
    if backbone_name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {backbone_name}. Available: {list(BACKBONES.keys())}")
    if not images:
        raise ValueError("Empty image list passed to compute_feats.")

    transform = transforms.Compose(
        [
            transforms.Resize(resize_to),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Cache keyed by (backbone, resolution): the model is built for a specific input resolution
    # (builder(resize_to)), so a name-only key would hand back a wrong-resolution model when callers
    # mix resolutions.
    cache_key = (backbone_name, tuple(resize_to))
    if cache_key not in _model_cache:
        log.info(f"Loading {backbone_name} for FID compute_feats at {tuple(resize_to)}...")
        builder = BACKBONES[backbone_name]["builder"]
        ckpt = BACKBONES[backbone_name]["ckpt"]

        model = builder(resize_to)
        state_dict = load_pretrained_weights(ckpt)
        load_result = model.load_state_dict(state_dict, strict=False)

        if load_result.missing_keys:
            raise RuntimeError(f"Missing keys in model: {load_result.missing_keys}")

        model.eval()
        model.to(device)

        # torch.compile only pays off on CUDA; skip it on CPU to avoid a slow, pointless warmup.
        if device.type == "cuda" and "cradio_v3_base" in backbone_name:
            log.info(f"torch.compile: {backbone_name}")
            model = torch.compile(model, mode="default", fullgraph=False)

        _model_cache[cache_key] = model
        log.info(f"{backbone_name} loaded successfully.")
    else:
        model = _model_cache[cache_key]

    features = []
    for img in images:
        pil = Image.fromarray((img * 255).astype(np.uint8))
        x = transform(pil).unsqueeze(0).to(device)
        feat = model(x)
        features.append(feat.squeeze(0))
    return torch.stack(features, dim=0)


@torch.inference_mode()
def compute_fid_kpi(
    real_images_dict: dict,
    generated_images_dict: dict,
    backbone_name: str = "cradio_v3_base",
    crop_size: int = 512,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Per-anomaly-type FID + macro Average, keyed ``fid``.

    ``real[name] = {"original_image": [...], "original_mask": [...]}`` and
    ``generated[name] = {"reconstructed_image": [...], "original_mask": [...]}``; images are
    HxWx3 float arrays in [0, 1], masks HxW in [0, 1]. FID is ``None`` when fewer than two
    defect crops exist on either side.

    ``crop_size`` (multiple of the cradio patch size, 512 by default) is the resolution defect crops
    are resized to and the C-RADIO-V3 backbone runs at. Smaller values trade FID fidelity for speed —
    useful on CPU, where the backbone forward dominates."""
    result: Dict[str, Dict[str, Optional[float]]] = {}
    fids: List[float] = []
    fid_key = "fid"
    resize_to = (crop_size, crop_size)

    for anomaly_name in sorted(real_images_dict.keys()):
        result[anomaly_name] = {}

        mask_crop_images(real_images_dict[anomaly_name], "original_image", crop_size=crop_size)
        mask_crop_images(generated_images_dict[anomaly_name], "reconstructed_image", crop_size=crop_size)

        real_crops = real_images_dict[anomaly_name].get("mask_cropped_image", [])
        gen_crops = generated_images_dict[anomaly_name].get("mask_cropped_image", [])

        feats_real = compute_feats(real_crops, resize_to, backbone_name) if real_crops else torch.empty(0)
        feats_gen = compute_feats(gen_crops, resize_to, backbone_name) if gen_crops else torch.empty(0)

        log.info(f"[{anomaly_name}] FID feature counts -> real: {feats_real.size(0)}, generated: {feats_gen.size(0)}")

        if feats_real.size(0) <= 1 or feats_gen.size(0) <= 1:
            log.warning(
                f"[{anomaly_name}] Skipping FID: not enough defect features "
                f"(real: {feats_real.size(0)}, generated: {feats_gen.size(0)}; need at least 2 each)."
            )
            result[anomaly_name][fid_key] = None
            continue

        fid = _compute_fid_on_feats(feats_real, feats_gen)
        result[anomaly_name][fid_key] = fid
        fids.append(fid)

    result["Average"] = {fid_key: (sum(fids) / len(fids) if fids else None)}

    return result
