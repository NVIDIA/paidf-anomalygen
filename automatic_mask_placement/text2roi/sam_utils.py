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

"""SAM2 pre/post-processing helpers for text2roi pipeline."""

import cv2
import numpy as np
from PIL import Image

IMAGE_RESIZE = 1024


def resize_for_sam(pil, max_size=IMAGE_RESIZE):
    w, h = pil.size
    scale = min(max_size / w, max_size / h, 1.0)
    return pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS), scale


def keep_largest_and_fill(u8):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (u8 > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return u8
    lbl = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == lbl).astype(np.uint8)
    h, w = mask.shape
    # Flood the background inward from a guaranteed-background 1px border. If we
    # seeded the flood at (0, 0) of the mask itself and the kept object touched
    # the top-left corner, the seed would land on foreground, the background
    # would never be flooded, and mask[flood == 0] = 1 would turn the whole
    # frame white. Padding with a zero border makes the corner always background.
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:-1, 1:-1] = mask
    ff_buf = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(canvas, ff_buf, (0, 0), 1)
    # After the flood, exterior background == 1 and only truly-enclosed holes
    # remain 0; fill those holes back into the object.
    mask[canvas[1:-1, 1:-1] == 0] = 1
    return (mask * 255).astype(np.uint8)


def postprocess(raw, ori_w, ori_h):
    combined = keep_largest_and_fill(raw.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k)
    return cv2.resize(combined, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)


def pick_mask(masks, scores):
    return masks[np.argmax(scores)]
