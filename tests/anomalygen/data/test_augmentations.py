# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for anomalygen.data.augmentations (paired train-time image/mask transforms).

RNG is injected so the probabilistic transforms are deterministic under test.
"""

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as tF

from anomalygen.data.augmentations import (
    RandomHorizontalFlip,
    RandomInstanceDrop,
    RandomOrthogonalRotation,
    RandomRatioCrop,
    RandomRingJitter,
    RandomRotation,
    RandomVerticalFlip,
)


@pytest.mark.parametrize(
    "augmentation",
    [
        RandomRingJitter,
        RandomInstanceDrop,
        RandomVerticalFlip,
        RandomHorizontalFlip,
        RandomOrthogonalRotation,
        RandomRotation,
        RandomRatioCrop,
    ],
)
def test_every_augmentation_requires_an_explicit_rng(augmentation):
    """No augmentation may fall back to an unseeded generator: the caller's seed is the contract.

    An implicit default would still draw, so a run would stay reproducible-looking while quietly
    ignoring base_seed. Refusing to construct says so at the call site instead.
    """
    with pytest.raises(TypeError, match="rng"):
        augmentation()


def _asymmetric_image():
    # Distinct top-left marker so a flip is observable.
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[0, 0] = 255
    return Image.fromarray(arr, mode="RGB")


def _mask_with_square(size=64, lo=20, hi=40):
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[lo:hi, lo:hi] = 255
    return Image.fromarray(arr, mode="L")


def test_horizontal_flip_applies_when_probability_certain():
    img = _asymmetric_image()
    out_img, _ = RandomHorizontalFlip(p=1.0, rng=np.random.default_rng(0))(img, _mask_with_square())
    # Top-left marker moves to top-right after an h-flip.
    assert np.array(out_img)[0, -1, 0] == 255
    assert np.array(out_img)[0, 0, 0] == 0


def test_horizontal_flip_skipped_when_probability_zero():
    img = _asymmetric_image()
    out_img, _ = RandomHorizontalFlip(p=0.0, rng=np.random.default_rng(0))(img, _mask_with_square())
    assert np.array_equal(np.array(out_img), np.array(img))


def test_vertical_flip_applies_when_probability_certain():
    img = _asymmetric_image()
    out_img, _ = RandomVerticalFlip(p=1.0, rng=np.random.default_rng(0))(img, _mask_with_square())
    # Top-left marker moves to bottom-left after a v-flip.
    assert np.array(out_img)[-1, 0, 0] == 255


def _thin_diagonal_mask(size=64):
    """1px-wide diagonal defect — the shape (crack / scratch / bridge) rotation most easily erodes."""
    arr = np.zeros((size, size), dtype=np.uint8)
    for i in range(10, size - 10):
        arr[i, i] = 255
    return Image.fromarray(arr, mode="L")


class _FixedAngleRng:
    """np.random.Generator stand-in pinning RandomRotation to one known positive angle."""

    def __init__(self, angle: int) -> None:
        self._angle = angle

    def integers(self, low, high=None):
        return self._angle

    def random(self):
        return 0.0  # < 0.5 -> keep the positive angle


def test_random_rotation_preserves_thin_mask_area():
    angle = 13
    img = Image.fromarray(np.full((64, 64, 3), 120, np.uint8), mode="RGB")
    mask = _thin_diagonal_mask()
    area = int((np.array(mask) > 0).sum())

    _, out_mask = RandomRotation(max_angle=angle, p=1.0, rng=_FixedAngleRng(angle))(img, mask)
    out = np.array(out_mask)

    # NEAREST rotation: the mask stays binary and every defect pixel survives the turn.
    assert set(np.unique(out)).issubset({0, 255})
    assert int((out > 0).sum()) == area
    # Reference: the BILINEAR + 0.5-threshold path this replaced erodes the same thin defect.
    bilinear = np.array(tF.rotate(mask, angle, interpolation=transforms.InterpolationMode.BILINEAR))
    assert int((bilinear >= 128).sum()) < area


def test_flip_stack_preserves_size_and_mask_binary():
    """The flips are standalone transforms now — composing them is the caller's job."""
    rng = np.random.default_rng(1)
    img = Image.fromarray(np.full((64, 64, 3), 120, np.uint8), mode="RGB")
    mask = _mask_with_square()
    out_img, out_mask = img, mask
    for aug in (
        RandomVerticalFlip(0.5, rng=rng),
        RandomHorizontalFlip(0.5, rng=rng),
        RandomOrthogonalRotation(rng=rng),
    ):
        out_img, out_mask = aug(out_img, out_mask)
    assert out_img.size == img.size
    assert out_mask.size == mask.size
    assert set(np.unique(np.array(out_mask))).issubset({0, 255})


def test_random_ratio_crop_requires_l_mode_mask():
    img = _asymmetric_image()
    with pytest.raises(ValueError):
        RandomRatioCrop(rng=np.random.default_rng(0))(img, img.convert("RGB"))


def test_random_ratio_crop_empty_mask_returns_unchanged():
    rng = np.random.default_rng(2)
    img = Image.fromarray(np.zeros((100, 100, 3), np.uint8), mode="RGB")
    empty_mask = Image.fromarray(np.zeros((100, 100), np.uint8), mode="L")
    out_img, out_mask = RandomRatioCrop(rng=rng)(img, empty_mask)
    # No foreground to centre on -> the pair is returned uncropped; sizes stay full.
    assert out_img.size == (100, 100)
    assert out_mask.size == (100, 100)


def test_random_ratio_crop_produces_aligned_bounded_crop():
    rng = np.random.default_rng(3)
    img = Image.fromarray(np.full((100, 100, 3), 50, np.uint8), mode="RGB")
    mask = _mask_with_square(size=100, lo=40, hi=60)
    out_img, out_mask = RandomRatioCrop(final_crop_size=512, rng=rng)(img, mask)
    # Image and mask are cropped with identical windows, so their sizes must match ...
    assert out_img.size == out_mask.size
    # ... and stay within the original frame.
    w, h = out_img.size
    assert 0 < w <= 100 and 0 < h <= 100
    # Square, matching inference's crop_grid_by_ratio window: a random train-time aspect would show
    # the model framings generation never produces. NOTE this fixture is square (100x100) with a
    # small defect, so the window always fits — see
    # test_random_ratio_crop_clamps_to_a_rectangle_when_the_window_overflows for the case where it
    # does not, which is the common one on real non-square data.
    assert w == h


def test_random_ratio_crop_clamps_to_a_rectangle_when_the_window_overflows():
    """A window bigger than the frame comes back rectangular — matching inference, not a deviation.

    ``crop_paste.best_crop`` clamps each axis the same way, so this framing is one the model also
    sees at generation time. Forcing a square here would make training disagree with inference.
    """
    rng = np.random.default_rng(0)
    img = Image.fromarray(np.zeros((400, 1600, 3), np.uint8), mode="RGB")  # wide frame
    m = np.zeros((400, 1600), np.uint8)
    m[170:230, 770:830] = 255  # 60x60 defect -> side 60*8 = 480 > H
    mask = Image.fromarray(m, mode="L")

    crop = RandomRatioCrop(final_crop_size=512, ratio_range=(8.0, 8.0), rng=rng)
    w, h = crop(img, mask)[0].size

    assert h == 400, "height should clamp to the frame"
    assert w > h, "width does not clamp, so the crop is rectangular"
