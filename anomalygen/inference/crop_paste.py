# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Crop & paste helpers for I2I inpainting at inference. Pure PIL / cv2 / numpy.
#
# Conventions:
#   - Mask is PIL ``L`` mode with values in {0, 255}. 255 means edit/regenerate.
#   - All coordinates use PIL's (left, upper, right, lower) convention.

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from anomalygen.data.utils import MASK_FG_THRESHOLD


def mask_bbox(mask: Image.Image) -> Tuple[int, int, int, int]:
    """Tight ``(left, upper, right, lower)`` bounding box of the mask's foreground.

    Raises ``ValueError`` on an empty mask, matching :func:`best_crop`.
    """
    arr = np.array(mask.convert("L")) if mask.mode != "L" else np.array(mask)
    ys, xs = np.nonzero(arr >= MASK_FG_THRESHOLD)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("No mask region found in input mask.")

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def annotate(
    image: Image.Image, mask: Image.Image, crop_offset: Tuple[int, int], crop_size: Tuple[int, int], label: str = ""
) -> Image.Image:
    """Draw the mask bbox (red) and crop window (green) on a copy of ``image``."""
    lx, uy, rx, by = mask_bbox(mask)
    clx, cuy = crop_offset
    crx, cby = clx + crop_size[0], cuy + crop_size[1]

    annotated = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(annotated, "RGBA")
    draw.rectangle(((lx, uy), (rx, by)), outline=(255, 0, 0, 127), width=3)
    draw.rectangle(((clx, cuy), (crx, cby)), outline=(0, 255, 0, 127), width=3)
    if label:
        draw.text((clx + 5, max(0, cuy - 12)), label, fill=(0, 128, 0))

    return annotated.convert("RGB")


def crop_grid_by_ratio(mask: Image.Image, crop_ratio: float) -> int:
    """Square crop side for ``crop_ratio``: ``max(bbox_w, bbox_h) * crop_ratio``, at least 1."""
    lx, uy, rx, by = mask_bbox(mask)
    bbox = max(rx - lx + 1, by - uy + 1)

    return max(1, int(bbox * crop_ratio))


def best_crop(image: Image.Image, mask: Image.Image, grid_x: int = 512, grid_y: int = 512):
    """Crop a ``grid_x × grid_y`` window centered on the mask's centroid, clamped to bounds.

    Returns ``(cropped_image, cropped_mask, (upper_left_x, upper_left_y))``.
    """
    binary_mask = np.array(mask.convert("L")) if mask.mode != "L" else np.array(mask)
    ys, xs = np.nonzero(binary_mask >= MASK_FG_THRESHOLD)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("No mask region found in input mask.")

    lx, rx = int(xs.min()), int(xs.max())
    uy, by = int(ys.min()), int(ys.max())
    center_x = (lx + rx) // 2
    center_y = (uy + by) // 2

    h, w = binary_mask.shape
    crop_lx = max(0, min(center_x - grid_x // 2, w - grid_x))
    crop_rx = min(w, crop_lx + grid_x)
    crop_uy = max(0, min(center_y - grid_y // 2, h - grid_y))
    crop_by = min(h, crop_uy + grid_y)

    cropped_image = image.crop((crop_lx, crop_uy, crop_rx, crop_by))
    cropped_mask = mask.crop((crop_lx, crop_uy, crop_rx, crop_by))

    return cropped_image, cropped_mask, (crop_lx, crop_uy)


def enlarge_mask(pil_mask: Image.Image) -> Image.Image:
    """Replace each connected component with a disk of twice its inscribed radius.

    Softens the hard mask edge at paste time to avoid seams.
    """
    if pil_mask.mode != "L":
        pil_mask = pil_mask.convert("L")

    arr = np.array(pil_mask)
    contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_mask = np.zeros_like(arr)
    for contour in contours:
        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        cv2.circle(new_mask, (int(x), int(y)), int(radius * 2), 255, -1)

    return Image.fromarray(new_mask)


def feathered_alpha(cropped_mask: Image.Image, feather: float) -> np.ndarray:
    """Soft ``[0, 1]`` alpha of the *original* mask shape, ``[H, W]`` float32.

    Unlike :func:`enlarge_mask` (which replaces each component with a 2x-radius disk), this keeps
    the true contour. Alpha is exactly 1 everywhere inside the original mask — so the whole
    reconstruction is preserved there — and ramps linearly to 0 over a ``feather``-pixel band
    *outside* the contour (feathering outward only, never eroding the interior). ``feather`` is the
    ramp width in pixels; ``feather <= 0`` returns the hard binary mask.
    """
    binary = (np.array(cropped_mask.convert("L")) >= MASK_FG_THRESHOLD).astype(np.uint8)
    if feather <= 0:
        return binary.astype(np.float32)
    # Distance from each background pixel to the nearest foreground pixel; foreground stays at 0,
    # so alpha=1 across the entire mask and ramps to 0 over `feather` px outward.
    dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    alpha = np.clip(1.0 - dist / feather, 0.0, 1.0)
    return alpha.astype(np.float32)


def paste_back(
    input_image: Image.Image,
    recon_image: Image.Image,
    cropped_image: Image.Image,
    cropped_mask: Image.Image,
    offset: Tuple[int, int],
    poisson_blend: bool = False,
    feather: float = 16.0,
) -> Image.Image:
    """Composite the model's edited crop back into the full image (out-of-place).

    ``poisson_blend`` selects Poisson seamless cloning over the enlarged-disk mask (see
    :func:`enlarge_mask`); otherwise the reconstruction is soft-blended with an alpha of the
    *original* mask shape that is 1 across the whole mask and feathers outward only (see
    :func:`feathered_alpha`), so every reconstructed pixel inside the contour is kept.
    """
    if recon_image.size != cropped_image.size:
        recon_image = recon_image.resize(cropped_image.size, Image.Resampling.BICUBIC)

    out = input_image.copy()
    tmp_cropped = cropped_image.copy()

    if poisson_blend:
        enlarged = enlarge_mask(cropped_mask)
        image_mode = recon_image.mode
        np_recon = np.array(recon_image.convert("RGB"))
        np_cropped = np.array(tmp_cropped.convert("RGB"))
        np_mask = np.array(enlarged)

        ys, xs = np.nonzero(np_mask >= MASK_FG_THRESHOLD)
        if xs.size == 0 or ys.size == 0:
            out.paste(tmp_cropped, offset)
            return out

        cx = (int(xs.min()) + int(xs.max())) // 2
        cy = (int(ys.min()) + int(ys.max())) // 2
        blended = cv2.seamlessClone(np_recon, np_cropped, np_mask, (cx, cy), cv2.NORMAL_CLONE)
        tmp_cropped = Image.fromarray(blended).convert(image_mode)
    else:
        alpha = feathered_alpha(cropped_mask, feather)[..., None]  # [H,W,1]
        np_recon = np.array(recon_image.convert("RGB"), dtype=np.float32)
        np_cropped = np.array(tmp_cropped.convert("RGB"), dtype=np.float32)
        blended = np_recon * alpha + np_cropped * (1.0 - alpha)
        tmp_cropped = Image.fromarray(blended.round().clip(0, 255).astype(np.uint8)).convert(recon_image.mode)

    out.paste(tmp_cropped, offset)
    return out
