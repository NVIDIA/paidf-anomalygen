# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision import transforms

from anomalygen.auto_mask_placement.roi_generation.utils import to_rgb_uint8

# SAM2.1 hiera-large weights, resolved relative to the repo root so the path is CWD-independent
# (mirrors anomalygen.eval.fid). Downloaded by scripts/download_checkpoints.sh.
_SAM2_CKPT = str(
    Path(__file__).resolve().parents[3] / "checkpoints" / "facebook" / "sam2.1-hiera-large" / "sam2.1_hiera_large.pt"
)


def _build_sam2_predictor(device):
    # sam_config is a hydra config name resolved by the sam2 package (not a filesystem path).
    sam_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(sam_config, _SAM2_CKPT, device=device)
    return SAM2ImagePredictor(sam2_model)


def _build_cradiov3_base(device):
    # Imported lazily: the cradio backbone pulls apex (CUDA-only), which isn't installed in CPU-only
    # environments. This builder is a GPU-only path, so keeping the import here lets the module import
    # (and the SAM2 path run) without the cradio/apex stack.
    from anomalygen.eval.fid import BACKBONES
    from anomalygen.models.vision_encoder.cradio.ptm_util import load_pretrained_weights

    cradio_builder = BACKBONES["cradio_v3_base"]["builder"]
    cradio_ckpt = BACKBONES["cradio_v3_base"]["ckpt"]
    cradio_model = cradio_builder((1024, 1024))
    state_dict = load_pretrained_weights(cradio_ckpt)
    load_result = cradio_model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        raise RuntimeError(f"Missing keys in cradio model: {load_result.missing_keys}")
    cradio_model.eval()
    cradio_model.to(device)
    return cradio_model


class ROIGenerationModels:
    """Handle model initialisation and provide shared assets."""

    def __init__(self, device):
        self.device = device

        self.models = {}

        self.cradiov3_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._current_image = None

    def _get_model(self, model_name: str):
        if model_name not in self.models:
            if model_name == "sam2":
                self.models[model_name] = _build_sam2_predictor(self.device)
            elif model_name == "cradiov3_base":
                self.models[model_name] = _build_cradiov3_base(self.device)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

        return self.models[model_name]

    def forward_segmentation(
        self,
        image,
        model_name="sam2",
        boxes=None,
        point_coords=None,
        point_labels=None,
    ):
        if boxes is None and point_coords is None:
            raise ValueError("forward_segmentation requires boxes or point_coords")

        model = self._get_model(model_name)
        if model_name == "sam2":
            return self._forward_sam2(
                model,
                image=image,
                boxes=boxes,
                point_coords=point_coords,
                point_labels=point_labels,
            )
        else:
            raise ValueError(f"Unknown segmentation model: {model_name}")

    def forward_feature_map(self, image_pil, model_name="cradiov3_base"):
        model = self._get_model(model_name)
        if model_name == "cradiov3_base":
            return self._forward_cradiov3_feat_map(model, image_pil)
        else:
            raise ValueError(f"Unknown feature model: {model_name}")

    def _forward_sam2(
        self,
        segmentation_model,
        image,
        boxes=None,
        point_coords=None,
        point_labels=None,
    ):
        """Run per-point / per-box SAM2 inference. Returns (N,H,W) masks, (N,) scores."""
        self._safe_sam2_set_image(segmentation_model, image)

        masks_list, scores_list = [], []

        # Point prompts
        if point_coords is not None:
            point_coords = np.asarray(point_coords)
            point_labels = np.asarray(point_labels)

            for pt, lbl in zip(point_coords, point_labels):
                m, s, _ = segmentation_model.predict(
                    point_coords=np.asarray([pt]),
                    point_labels=np.asarray([lbl]),
                    box=None,
                    multimask_output=True,
                )
                idx = np.argmax(s)
                masks_list.append(m[idx].astype(np.uint8))
                scores_list.append(float(s[idx]))

        # Box prompts
        if boxes is not None:
            boxes = np.asarray(boxes)
            for box in boxes:
                m, s, _ = segmentation_model.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.asarray(box),
                    multimask_output=True,
                )
                idx = np.argmax(s)
                masks_list.append(m[idx].astype(np.uint8))
                scores_list.append(float(s[idx]))

        masks = np.stack(masks_list, axis=0).astype(np.uint8)
        scores = np.asarray(scores_list, dtype=np.float32)

        return masks, scores

    def _safe_sam2_set_image(self, model, image):
        """
        Safely set the image for the SAM2 predictor, skipping if it's the same image.
        """
        # Reuse embeddings when the same image object comes back; retain a
        # reference so its identity stays stable across calls. An id() cache can
        # alias two distinct images after CPython reuses a freed address, which
        # skips set_image and runs SAM2 on stale embeddings.
        if image is not None and image is not self._current_image:
            model.set_image(to_rgb_uint8(image))
            self._current_image = image

    def _forward_cradiov3_feat_map(self, model, image_pil):
        with torch.no_grad():
            x = self.cradiov3_transform(image_pil).unsqueeze(0).to(self.device)
            feat = model.forward_feature_pyramid(x)
        return feat[0]
