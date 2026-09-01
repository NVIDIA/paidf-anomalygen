# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for anomalygen.inference.crop_paste (pure PIL/cv2/numpy geometry).

Reachable from generate.py via inference.iterative; tested directly here for the branches the
orchestrator does not exercise (poisson blend, feathering, annotation, empty-mask guards).
"""

import numpy as np
import pytest
from PIL import Image

from anomalygen.inference.crop_paste import (
    annotate,
    best_crop,
    crop_grid_by_ratio,
    enlarge_mask,
    feathered_alpha,
    mask_bbox,
    paste_back,
)


def _mask(size=50, box=(5, 10, 15, 20)):
    """L-mode mask with a filled foreground rectangle (left, upper, right, lower), inclusive."""
    lx, uy, rx, by = box
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[uy : by + 1, lx : rx + 1] = 255
    return Image.fromarray(arr, mode="L")


def test_mask_bbox_tight_and_inclusive():
    assert mask_bbox(_mask(box=(5, 10, 15, 20))) == (5, 10, 15, 20)


def test_mask_bbox_empty_raises():
    with pytest.raises(ValueError):
        mask_bbox(Image.fromarray(np.zeros((10, 10), np.uint8), mode="L"))


def test_crop_grid_by_ratio():
    # bbox side = max(15-5+1, 20-10+1) = 11; * 2.0 -> 22.
    assert crop_grid_by_ratio(_mask(box=(5, 10, 15, 20)), 2.0) == 22
    assert crop_grid_by_ratio(_mask(box=(5, 10, 15, 20)), 0.0) == 1  # never smaller than 1


def test_best_crop_centers_and_clamps():
    img = Image.fromarray(np.full((100, 100, 3), 50, np.uint8), mode="RGB")
    mask = _mask(size=100, box=(40, 40, 60, 60))
    cropped_image, cropped_mask, offset = best_crop(img, mask, grid_x=32, grid_y=32)
    assert offset == (34, 34)  # center 50; clamp(50-16) = 34
    assert cropped_image.size == (32, 32)
    assert cropped_mask.size == (32, 32)


def test_best_crop_empty_mask_raises():
    img = Image.fromarray(np.zeros((40, 40, 3), np.uint8), mode="RGB")
    empty = Image.fromarray(np.zeros((40, 40), np.uint8), mode="L")
    with pytest.raises(ValueError):
        best_crop(img, empty)


def test_enlarge_mask_grows_foreground():
    mask = _mask(size=50, box=(20, 20, 25, 25))
    enlarged = enlarge_mask(mask)
    assert enlarged.mode == "L"
    assert int((np.array(enlarged) > 127).sum()) > int((np.array(mask) > 127).sum())


def test_feathered_alpha_hard_when_feather_nonpositive():
    alpha = feathered_alpha(_mask(size=40, box=(10, 10, 20, 20)), feather=0.0)
    assert set(np.unique(alpha)).issubset({0.0, 1.0})


def test_feathered_alpha_interior_one_and_ramps_outward():
    mask = _mask(size=60, box=(20, 20, 40, 40))
    alpha = feathered_alpha(mask, feather=8.0)
    binary = np.array(mask.convert("L")) >= 128
    # Alpha is exactly 1 everywhere inside the original mask (interior never eroded).
    assert np.allclose(alpha[binary], 1.0)
    # Values stay in [0, 1] and a soft ramp (strictly between 0 and 1) exists outside.
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    assert np.any((alpha > 0.0) & (alpha < 1.0))


def test_paste_back_feather_zero_keeps_recon_inside_and_source_outside():
    input_image = Image.fromarray(np.zeros((100, 100, 3), np.uint8), mode="RGB")  # black
    cropped_image = Image.fromarray(np.full((50, 50, 3), 10, np.uint8), mode="RGB")  # grey source
    recon_image = Image.fromarray(np.full((50, 50, 3), 200, np.uint8), mode="RGB")  # bright recon
    cropped_mask = _mask(size=50, box=(20, 20, 30, 30))

    out = paste_back(input_image, recon_image, cropped_image, cropped_mask, offset=(25, 25), feather=0.0)
    out_arr = np.array(out)
    assert out_arr[50, 50, 0] == 200  # mask centre -> reconstruction
    assert out_arr[30, 30, 0] == 10  # inside crop, outside mask -> original cropped value
    assert out_arr[5, 5, 0] == 0  # outside the paste window -> untouched input


def test_paste_back_resizes_mismatched_recon():
    input_image = Image.fromarray(np.zeros((80, 80, 3), np.uint8), mode="RGB")
    cropped_image = Image.fromarray(np.full((40, 40, 3), 10, np.uint8), mode="RGB")
    recon_image = Image.fromarray(np.full((20, 20, 3), 200, np.uint8), mode="RGB")  # wrong size
    cropped_mask = _mask(size=40, box=(10, 10, 30, 30))
    out = paste_back(input_image, recon_image, cropped_image, cropped_mask, (10, 10), feather=0.0)
    assert out.size == (80, 80)  # recon resized to crop, no error


def test_paste_back_poisson_empty_mask_falls_back_to_paste():
    input_image = Image.fromarray(np.zeros((60, 60, 3), np.uint8), mode="RGB")
    cropped_image = Image.fromarray(np.full((30, 30, 3), 10, np.uint8), mode="RGB")
    recon_image = Image.fromarray(np.full((30, 30, 3), 200, np.uint8), mode="RGB")
    empty_mask = Image.fromarray(np.zeros((30, 30), np.uint8), mode="L")
    out = paste_back(
        input_image, recon_image, cropped_image, cropped_mask=empty_mask, offset=(15, 15), poisson_blend=True
    )
    # Empty enlarged mask -> the untouched cropped image is pasted (no reconstruction blended in).
    assert np.array(out)[30, 30, 0] == 10


def test_paste_back_poisson_blends_into_mask_region():
    input_image = Image.fromarray(np.zeros((80, 80, 3), np.uint8), mode="RGB")
    cropped_image = Image.fromarray(np.full((40, 40, 3), 10, np.uint8), mode="RGB")
    recon_image = Image.fromarray(np.full((40, 40, 3), 200, np.uint8), mode="RGB")
    cropped_mask = _mask(size=40, box=(12, 12, 28, 28))
    out = paste_back(input_image, recon_image, cropped_image, cropped_mask, offset=(20, 20), poisson_blend=True)
    # Seamless-clone runs and composites the crop back; output stays full-size and valid.
    assert out.size == (80, 80)
    assert out.mode == "RGB"


def test_annotate_returns_rgb_same_size():
    img = Image.fromarray(np.full((60, 60, 3), 100, np.uint8), mode="RGB")
    mask = _mask(size=60, box=(10, 10, 30, 30))
    out = annotate(img, mask, crop_offset=(5, 5), crop_size=(40, 40), label="crop")
    assert out.mode == "RGB"
    assert out.size == img.size
