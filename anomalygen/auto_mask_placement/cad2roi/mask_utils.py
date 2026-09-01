# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared mask utilities for cad2roi defect processing."""

import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def merge_class_masks(components: dict, classes: Sequence[str], img_shape: tuple) -> np.ndarray:
    """Merge component masks for the given class names into a single binary mask."""
    h, w = img_shape
    merged = np.zeros((h, w), dtype=np.uint8)
    for cls in classes:
        for comp in components.get(cls, []):
            merged = cv2.bitwise_or(merged, comp.mask)
    return merged


def mask_area(mask: np.ndarray) -> int:
    """Count white pixels (value 255) in a binary mask."""
    return int(mask.sum() // 255)


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (x, y, w, h) bounding box of nonzero region, or None if empty."""
    nz = cv2.findNonZero(mask)
    if nz is None:
        return None
    return cv2.boundingRect(nz)


def save_temp_mask(mask: np.ndarray) -> str:
    """Save mask to a temporary PNG file and return its path."""
    fd = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fd.close()
    cv2.imwrite(fd.name, mask)
    return fd.name


def load_and_scale_submask(
    submask_path: str, target_area: float, img_shape: tuple, scale_factor: float = 1.0
) -> Optional[str]:
    """Read a submask, scale it relative to target_area, center on canvas, save to temp file.

    Args:
        submask_path: path to the submask image
        target_area: the ROI area to scale against (pixels at value 255)
        img_shape: (H, W) of the output canvas
        scale_factor: additional multiplier on the computed scale

    Returns:
        path to the temporary scaled mask file, or None on failure
    """
    h, w = img_shape
    submask_raw = cv2.imread(submask_path, cv2.IMREAD_GRAYSCALE)
    if submask_raw is None:
        return None
    if submask_raw.ndim > 2:
        submask_raw = submask_raw[:, :, 0]
    _, submask_bin = cv2.threshold(submask_raw, 127, 255, cv2.THRESH_BINARY)

    bbox = mask_bbox(submask_bin)
    if bbox is None:
        return None
    sx, sy, sw, sh = bbox
    crop = submask_bin[sy : sy + sh, sx : sx + sw]
    submask_area = mask_area(crop)

    scale = np.sqrt(target_area / max(submask_area, 1)) * scale_factor
    new_w = max(1, int(sw * scale))
    new_h = max(1, int(sh * scale))
    scaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - new_w // 2)
    y1 = max(0, cy - new_h // 2)
    x2 = min(w, x1 + new_w)
    y2 = min(h, y1 + new_h)
    canvas[y1:y2, x1:x2] = scaled[: y2 - y1, : x2 - x1]

    return save_temp_mask(canvas)


def read_amp_output(tmp_dir: str) -> Optional[np.ndarray]:
    """Read and clean up AMP's file-based output from a temp directory.

    Returns:
        Grayscale mask array, or None if AMP produced no valid output.
    """
    out_files = sorted(Path(tmp_dir).glob("*.png"))
    if not out_files:
        return None
    placed = cv2.imread(str(out_files[0]), cv2.IMREAD_GRAYSCALE)
    for f in out_files:
        f.unlink()
    if placed is None or placed.sum() == 0:
        return None
    return placed
