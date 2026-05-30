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

"""AMP result visualization helpers."""

import cv2
import numpy as np
from PIL import Image

# ── Colors ───────────────────────────────────────────────────────────────────
ROI_COLOR = np.array([0, 200, 0], dtype=np.float32)
CAD_ROI_COLOR = np.array([180, 180, 180], dtype=np.float32)
MASK_COLOR = np.array([0, 220, 255], dtype=np.float32)
CAD_MASK_COLOR = np.array([255, 140, 0], dtype=np.float32)
BBOX_COLOR = (0, 0, 255)
POINT_COLOR = (0, 255, 255)


def overlay(bg, mask, color, alpha=0.6):
    """Blend color onto bg where mask > 0."""
    viz = bg.copy()
    m = mask > 0
    viz[m] = (viz[m].astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)
    return viz


def make_amp_overlay(img_np, roi_mask, placed_mask, bbox, point):
    """Create AMP result overlay with ROI, placed mask, bbox, and point."""
    ov = img_np.astype(np.float32).copy()
    if roi_mask is not None:
        roi_bin = roi_mask > 0
        ov[roi_bin] = 0.65 * ov[roi_bin] + 0.35 * ROI_COLOR
    if placed_mask is not None:
        placed_bin = placed_mask > 0
        ov[placed_bin] = 0.45 * ov[placed_bin] + 0.55 * MASK_COLOR
    ov = ov.astype(np.uint8)
    if bbox:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        cv2.rectangle(ov, (x0, y0), (x1, y1), BBOX_COLOR, 3)
    if point:
        px, py = int(point[0]), int(point[1])
        cv2.circle(ov, (px, py), 10, POINT_COLOR, -1)
        cv2.circle(ov, (px, py), 10, (0, 0, 0), 2)
    return ov


def save_cad_seed(mask_bw, idx, prefix, out, assets, cad_img, clean_img):
    """Save mask + CAD/real overlays for one seed of a CAD defect."""
    Image.fromarray(mask_bw).save(out / f"{prefix}seed{idx}.png")
    Image.fromarray(overlay(cad_img, mask_bw, CAD_MASK_COLOR)).save(
        assets / f"{prefix}seed{idx}_cad.png")
    Image.fromarray(overlay(clean_img, mask_bw, CAD_MASK_COLOR)).save(
        assets / f"{prefix}seed{idx}_real.png")


def sum_masks(cands):
    """Combine all candidate masks into one."""
    if not cands:
        return np.zeros((1, 1), dtype=np.uint8)
    result = cands[0].mask.copy()
    for c in cands[1:]:
        result = cv2.bitwise_or(result, c.mask)
    return result
