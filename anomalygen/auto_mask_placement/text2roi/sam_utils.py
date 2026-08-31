# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM2 pre/post-processing helpers for text2roi pipeline."""

import cv2
import numpy as np
from PIL import Image

SAM_IMAGE_SIZE = 1024


def resize_for_sam(pil, max_size=SAM_IMAGE_SIZE):
    w, h = pil.size
    scale = min(max_size / w, max_size / h, 1.0)
    return pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS), scale


def _keep_largest_and_fill(u8):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((u8 > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return u8
    lbl = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == lbl).astype(np.uint8)
    h, w = mask.shape
    # Flood the exterior background from a guaranteed-background 1px border so
    # only truly-enclosed holes get filled. Seeding the flood at the mask's own
    # (0, 0) inverted the whole frame when the kept object touched the top-left
    # corner (that pixel was foreground, so the background was never flooded).
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:-1, 1:-1] = mask
    ff_buf = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(canvas, ff_buf, (0, 0), 1)
    mask[canvas[1:-1, 1:-1] == 0] = 1
    return (mask * 255).astype(np.uint8)


def postprocess_sam_mask(raw, ori_w, ori_h):
    combined = _keep_largest_and_fill(raw.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k)
    return cv2.resize(combined, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)


def pick_best_mask(masks, scores):
    return masks[np.argmax(scores)]
