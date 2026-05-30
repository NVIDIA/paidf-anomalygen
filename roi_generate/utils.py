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

import dataclasses
import hashlib
import json
import os
import random

import cv2
import numpy as np
import pycocotools.mask as mask_utils
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def mask_to_compressed_rle(mask):
    """
    Convert binary mask to COCO-compressed RLE string.
    """
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    # rle["counts"] is a bytes object → convert to str for JSON
    rle["counts"] = rle["counts"].decode("utf-8")
    return {"size": list(mask.shape), "counts": rle["counts"]}


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


def crop_pad_resize_square(
    img, box, target_size, is_mask=False,
):
    """
    Crop an image or mask by box, pad to square, and resize to target_size.
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = map(int, box)

    # Clip box to valid image bounds
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((target_size, target_size), dtype=img.dtype)

    # Crop region
    crop = img[y0:y1, x0:x1]

    # Normalize mask to 0/255 binary if needed
    if is_mask:
        crop = (crop > 0).astype(np.uint8) * 255

    # Pad to square (centered)
    h_c, w_c = crop.shape[:2]
    side = max(h_c, w_c)
    if img.ndim == 2:
        square = np.zeros((side, side), dtype=img.dtype)
    else:
        square = np.zeros((side, side, img.shape[2]), dtype=img.dtype)
    y_off = (side - h_c) // 2
    x_off = (side - w_c) // 2
    square[y_off : y_off + h_c, x_off : x_off + w_c] = crop

    # Resize
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(square, (target_size, target_size), interpolation=interp)

    # Re-binarize masks after resize
    if is_mask:
        resized = (resized > 127).astype(np.uint8) * 255

    return resized


def generate_augmented_variants(
    base_images, rotation_degrees, allow_flip_horizontal, allow_flip_vertical, is_mask=False,
):
    """Generate all rotation + flip variants with canonical."""

    def _canonicalize_transform(rotation, flip_lr, flip_ud):
        # --- Hard-coded dihedral group canonicalization ---
        r = round(rotation / 90.0) * 90 % 360
        if not flip_lr and not flip_ud:
            return (r, "none")
        if flip_lr and flip_ud:
            return ((r + 180) % 360, "none")

        lookup = {
            # flip_lr only
            (0, True, False): (0.0, "hflip"),
            (90, True, False): (0.0, "anti-diag"),
            (180, True, False): (0.0, "vflip"),
            (270, True, False): (0.0, "diag"),
            # flip_ud only
            (0, False, True): (0.0, "vflip"),
            (90, False, True): (0.0, "diag"),
            (180, False, True): (0.0, "hflip"),
            (270, False, True): (0.0, "anti-diag"),
        }
        return lookup.get((int(r), flip_lr, flip_ud), (r, "none"))

    flip_lr_opts = [False, True] if allow_flip_horizontal else [False]
    flip_ud_opts = [False, True] if allow_flip_vertical else [False]

    variants, metadata = [], []
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    for i, base in enumerate(base_images):
        seen_xforms, seen_hashes = set(), set()
        for deg in rotation_degrees:
            M = cv2.getRotationMatrix2D((base.shape[1] / 2, base.shape[0] / 2), deg, 1.0)
            rotated = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]), flags=interp, borderValue=0)

            for flr in flip_lr_opts:
                for fud in flip_ud_opts:
                    canon_rot, canon_flip = _canonicalize_transform(deg, flr, fud)
                    key = (canon_rot, canon_flip)
                    if key in seen_xforms:
                        continue

                    aug = rotated
                    if flr:
                        aug = cv2.flip(aug, 1)
                    if fud:
                        aug = cv2.flip(aug, 0)
                    aug = np.ascontiguousarray(aug)
                    if is_mask:
                        aug = (aug > 127).astype(np.uint8) * 255

                    hsh = hashlib.sha1(aug.tobytes()).hexdigest()
                    if hsh in seen_hashes:
                        continue

                    seen_xforms.add(key)
                    seen_hashes.add(hsh)
                    variants.append(aug)
                    metadata.append(
                        {
                            "source": str(i),
                            "rotation": float(deg),
                            "flip_lr": bool(flr),
                            "flip_ud": bool(fud),
                            "canonical": {"rotation": canon_rot, "flip": canon_flip},
                        }
                    )
    return variants, metadata


def _to_serializable(obj):
    """Safely convert numpy / torch / OmegaConf / PIL to serializable JSON form."""
    if dataclasses.is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    if hasattr(obj, "tolist"):  # torch tensor
        return obj.tolist()
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode(errors="ignore")
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]
    if hasattr(obj, "size") and hasattr(obj, "mode"):  # PIL.Image
        return {"size": obj.size, "mode": obj.mode}
    return obj


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


def compute_hash_dict(d):
    """Return short stable SHA256 hash string for dict."""
    dumped = json.dumps(_to_serializable(d), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16]


def sample_resize(image, boxes, max_size=1024):
    """
    Resize image to speed up inference and scale boxes accordingly.
    No padding. Keep aspect ratio.

    Args:
        image: PIL.Image
        boxes: list of [x0, y0, x1, y1]
        max_size: largest edge after resizing

    Returns:
        resized_image (PIL.Image)
        scaled_boxes (list)
        scale_factor (float)
    """
    w, h = image.size  # PIL returns (W, H)

    # If already small, skip
    if max(w, h) <= max_size:
        return image, boxes

    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = image.resize((new_w, new_h))

    # scale boxes
    scaled_boxes = []
    for x0, y0, x1, y1 in boxes:
        scaled_boxes.append(
            [x0 * scale, y0 * scale, x1 * scale, y1 * scale,]
        )

    return resized, scaled_boxes
