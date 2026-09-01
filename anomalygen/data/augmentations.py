# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Paired (image, mask) augmentations for I2I anomaly inpainting. Operates on PIL Image
# pairs so the mask stays aligned with the image through every transform.

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as tF
from torchvision.transforms import v2

from anomalygen.data.utils import MASK_FG_THRESHOLD


def _fires(rng, p: float) -> bool:
    """Sample a probability gate, drawing ONLY when the outcome is uncertain.

    ``p <= 0`` and ``p >= 1`` short-circuit without touching ``rng``, so turning an augmentation
    fully off or fully on leaves the random stream — and therefore every other augmentation's
    draws — byte-identical to a run without it.
    """
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return bool(rng.random() < p)


class RandomRingJitter:
    """Colour-jitter a band AROUND every defect; the defect pixels themselves stay untouched.

    Motivation: training crops can carry a colour fringe a few pixels from the defect edge that the
    held-out images do not, a leak the model would otherwise absorb as part of "what a defect looks
    like". Recolouring the band breaks the correlation. Because the band lies outside the mask it
    never enters the defect-only training loss — it only changes what the source item SHOWS.

    ``band_px`` is a FIXED width in ORIGINAL-image pixels, not a fraction of the instance: the leak
    sits a fixed distance from the edge, so it does not scale with defect size. One dilation of the
    whole mask suffices — dilation is a union, so this equals dilating each instance separately.
    The band is recoloured by a SINGLE jitter draw, so the whole fringe shifts together. Run BEFORE
    :class:`RandomInstanceDrop`, so bands are cut against the full mask.

    Returns ``mask`` unchanged; only the image is touched.
    """

    def __init__(self, p: float = 0.5, band_px: int = 10, *, rng: np.random.Generator) -> None:
        self.p = float(p)
        self.band_px = int(band_px)
        self.rng = rng
        # ``hue`` spans the full circle on purpose: the goal is to destroy the hue cue, not nudge it.
        self.photometric = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.5)

    def __call__(self, image: Image.Image, mask: Image.Image):
        if self.band_px <= 0 or not _fires(self.rng, self.p):
            return image, mask
        m = (np.asarray(mask) >= MASK_FG_THRESHOLD).astype(np.uint8)
        r = self.band_px
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        ring = (cv2.dilate(m, kern) > 0) & (m == 0)  # subtract the mask: defect pixels never recoloured
        ys, xs = np.nonzero(ring)
        if xs.size == 0:
            return image, mask
        arr = np.asarray(image).copy()
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        patch = torch.from_numpy(arr[y0:y1, x0:x1].astype(np.float32) / 255.0).permute(2, 0, 1)
        jittered = self.photometric(patch).clamp(0.0, 1.0)
        jittered = (jittered.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        sel = ring[y0:y1, x0:x1]
        arr[y0:y1, x0:x1][sel] = jittered[sel]
        return Image.fromarray(arr), mask


class RandomInstanceDrop:
    """Keep one random connected instance of the mask; black the others out in the IMAGE too.

    A no-op when the mask has 0 or 1 instances, so its effective rate is the multi-instance fraction
    of the dataset rather than ``p``. Motivation: inference inpaints ONE instance at a time, while
    training otherwise asks for every defect at once. Instances use the same threshold and
    8-connectivity as the loss and the inference-side splitter, so "one defect" means the same thing
    everywhere.

    The blackout covers each dropped instance PLUS a ``band_px`` margin, unconditionally: it removes
    a jittered ring that would else remain as a coloured halo, and covers the instance edge so no
    trace of a dropped defect survives unmarked.
    """

    def __init__(self, p: float = 0.5, band_px: int = 10, *, rng: np.random.Generator) -> None:
        self.p = float(p)
        self.band_px = int(band_px)
        self.rng = rng

    def __call__(self, image: Image.Image, mask: Image.Image):
        if not _fires(self.rng, self.p):
            return image, mask
        m = (np.asarray(mask) >= MASK_FG_THRESHOLD).astype(np.uint8)
        n_labels, labels = cv2.connectedComponents(m, connectivity=8)
        if n_labels <= 2:  # background + at most one instance
            return image, mask
        keep = int(self.rng.integers(1, n_labels))
        dropped = ((labels > 0) & (labels != keep)).astype(np.uint8)
        r = self.band_px
        if r > 0:
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            dropped = cv2.dilate(dropped, kern)
        dropped = (dropped > 0) & (labels != keep)
        arr = np.asarray(image).copy()
        arr[dropped] = 0  # black in the IMAGE too, so the pair keeps no unmarked defect
        kept_mask = np.where(labels == keep, 255, 0).astype(np.uint8)
        return Image.fromarray(arr), Image.fromarray(kept_mask, mode="L")


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5, *, rng: np.random.Generator) -> None:
        self.p = p
        self.rng = rng

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if _fires(self.rng, self.p):
            image, mask = tF.vflip(image), tF.vflip(mask)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5, *, rng: np.random.Generator) -> None:
        self.p = p
        self.rng = rng

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if _fires(self.rng, self.p):
            image, mask = tF.hflip(image), tF.hflip(mask)
        return image, mask


class RandomOrthogonalRotation:
    """Rotate the pair by a random quarter turn (0 / 90 / 180 / 270 degrees).

    Lossless and border-free: PIL fast-paths exact quarter turns to a transpose, so there is no
    interpolation (the mask stays binary without re-thresholding) and no black corners to crop away.
    That is why this is applied to EVERY sample, unlike the opt-in small-angle
    :class:`RandomRotation`. Note 90/270 swap width and height, so a non-square input comes back
    transposed.
    """

    def __init__(self, *, rng: np.random.Generator) -> None:
        self.rng = rng

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        angle = int(self.rng.integers(0, 4)) * 90
        if angle == 0:
            return image, mask
        # expand=True keeps the full canvas on 90/270 (and triggers PIL's exact transpose path).
        image = tF.rotate(image, angle, expand=True)
        mask = tF.rotate(mask, angle, expand=True)
        return image, mask


class RandomRotation:
    """Small-angle rotation; NEAREST for the mask keeps it binary and preserves thin defects.

    ``p`` defaults to 0: a free rotation leaves black corners and can swing a small edge defect off
    canvas, so it is opt-in and must be followed by a crop that trims the corners away. The lossless
    quarter turn (:class:`RandomOrthogonalRotation`) is the one applied by default.
    """

    def __init__(self, max_angle: int = 20, p: float = 0.0, *, rng: np.random.Generator) -> None:
        self.max_angle = int(max_angle)
        self.p = float(p)
        self.rng = rng

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if not _fires(self.rng, self.p):
            return image, mask
        angle = int(self.rng.integers(0, self.max_angle + 1))
        angle = angle if self.rng.random() < 0.5 else -angle
        image = tF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        # NEAREST keeps the mask binary (no re-binarize needed) and preserves thin defects
        # (bridge / crack / scratch) that BILINEAR + 0.5-threshold could erode.
        mask = tF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
        return image, mask


class RandomRatioCrop:
    """Mask-centered zoom-crop at a random ratio. Crop ONLY — compose it with the flips yourself.

    ``p`` defaults to 1.0 because inference always crops: skipping it would hand the model a
    full-image anisotropic resize, a framing generation never produces.
    """

    def __init__(
        self,
        final_crop_size: int = 512,
        ratio_range: Tuple[float, float] = (1.5, 8.0),
        p: float = 1.0,
        *,
        rng: np.random.Generator,
    ) -> None:
        self.final_crop_size = int(final_crop_size)
        self.ratio_range = (float(ratio_range[0]), float(ratio_range[1]))
        self.p = float(p)
        self.rng = rng

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """A mask-centered SQUARE crop at a random zoom ratio.

        The crop aims to be square, to match inference (``crop_paste.crop_grid_by_ratio``); only the
        zoom ratio and a small centroid jitter are randomised. Run it AFTER
        :class:`RandomRotation` when that is enabled, so it trims the black corners away.

        The side is clamped per axis, so a window wider than the frame comes back rectangular
        (~65% of phone_screen and ~68% of Apple crops, medians 1.78x / 1.58x). Not a bug: inference
        clamps identically (``crop_paste.best_crop``), so the model sees the same framings at
        generation time. Forcing ``min(side, W, H)`` here would make training disagree with it.
        """
        if mask.mode != "L":
            raise ValueError(f"RandomRatioCrop expects an 'L'-mode mask, got {mask.mode!r}")
        if not _fires(self.rng, self.p):
            return image, mask
        W, H = image.size

        mask_arr = np.array(mask)
        ys, xs = np.nonzero(mask_arr >= MASK_FG_THRESHOLD)
        if xs.size == 0:
            return image, mask

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        defect_w, defect_h = x2 - x1, y2 - y1
        defect_size = int(max(defect_w, defect_h))

        ratio_min, ratio_max = self.ratio_range
        ratio = ratio_min + self.rng.random() * (ratio_max - ratio_min)
        base = max(int(defect_size * ratio), self.final_crop_size // 16)

        # Square SIDE, matching crop_paste.crop_grid_by_ratio: max(bbox_w, bbox_h) * ratio. A random
        # train-time aspect would show framings generation never produces. The clamp below can still
        # return a rectangle, which is fine — inference clamps the same way.
        crop_w = crop_h = max(base, defect_w, defect_h, self.final_crop_size // 16)

        jx = max((crop_w - defect_w) // 2, 0)
        jy = max((crop_h - defect_h) // 2, 0)
        if jx > 0:
            cx += int(self.rng.integers(-jx, jx + 1))
        if jy > 0:
            cy += int(self.rng.integers(-jy, jy + 1))

        crop_x1 = max(0, min(cx - crop_w // 2, W - crop_w))
        crop_y1 = max(0, min(cy - crop_h // 2, H - crop_h))
        crop_x2 = min(W, crop_x1 + crop_w)
        crop_y2 = min(H, crop_y1 + crop_h)

        return image.crop((crop_x1, crop_y1, crop_x2, crop_y2)), mask.crop((crop_x1, crop_y1, crop_x2, crop_y2))
