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

import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision import transforms

from cosmos_predict2.metrics import utils
from cosmos_predict2.models.ag_modules.backbone_v2.ptm_util import load_pretrained_weights
from roi_generate.utils import to_rgb_uint8


def build_sam2_predictor(device):
    sam_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_checkpoint = "checkpoints/sam2/sam2.1_hiera_large.pt"
    sam2_model = build_sam2(sam_config, sam_checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)


def build_cradiov3_base(device):
    cradio_builder = utils.BACKBONES["cradio_v3_base"]["builder"]
    cradio_ckpt = utils.BACKBONES["cradio_v3_base"]["ckpt"]
    cradio_model = cradio_builder((1024, 1024))
    state_dict = load_pretrained_weights(cradio_ckpt, weights_only=False)
    load_result = cradio_model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        raise RuntimeError(f"Missing keys in cradio model: {load_result.missing_keys}")
    cradio_model.eval()
    cradio_model.to(device)
    return cradio_model


class ROIGenerateModels:
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
        self.current_image_id = None

    def _get_model(self, model_name: str):
        if model_name not in self.models:
            if model_name == "sam2":
                self.models[model_name] = build_sam2_predictor(self.device)
            elif model_name == "cradiov3_base":
                self.models[model_name] = build_cradiov3_base(self.device)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

        return self.models[model_name]

    def forward_segmentation(
        self, image, model_name="sam2", boxes=None, point_coords=None, point_labels=None,
    ):
        if boxes is None and point_coords is None:
            raise ValueError("forward_segmentation requires boxes or point_coords")

        model = self._get_model(model_name)
        if model_name == "sam2":
            return self._forward_sam2(
                model, image=image, boxes=boxes, point_coords=point_coords, point_labels=point_labels,
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
        self, segmentation_model, image, boxes=None, point_coords=None, point_labels=None,
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
                    point_coords=np.asarray([pt]), point_labels=np.asarray([lbl]), box=None, multimask_output=True,
                )
                idx = np.argmax(s)
                masks_list.append(m[idx].astype(np.uint8))
                scores_list.append(float(s[idx]))

        # Box prompts
        if boxes is not None:
            boxes = np.asarray(boxes)
            for box in boxes:
                m, s, _ = segmentation_model.predict(
                    point_coords=None, point_labels=None, box=np.asarray(box), multimask_output=True,
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
        if image is not None:
            new_image_id = id(image)

            if self.current_image_id != new_image_id:
                image_np = to_rgb_uint8(image)
                model.set_image(image_np)
                self.current_image_id = new_image_id

    def _forward_cradiov3_feat_map(self, model, image_pil):
        with torch.no_grad():
            x = self.cradiov3_transform(image_pil).unsqueeze(0).to(self.device)
            feat = model.forward_feature_pyramid(x)
        return feat[0]
