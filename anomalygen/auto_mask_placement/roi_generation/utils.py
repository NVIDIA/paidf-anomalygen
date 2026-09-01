# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers used across multiple roi_generation modules.

Single-consumer helpers live next to their caller (``template_box_to_masks``, ``pipeline``);
this module holds only the genuinely-shared conversions.
"""

import numpy as np


def to_rgb_uint8(image):
    arr = np.array(image, copy=True)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        max_val = float(arr.max()) if arr.size else 0.0
        if max_val <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round()
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)

    return arr


def to_jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x
