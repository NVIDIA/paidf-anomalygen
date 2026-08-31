# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for eval.fid.

The pure-math / numpy helpers run without any model. The real C-RADIO-V3 feature-extraction path
(``compute_feats``) is covered by a golden test that runs on CUDA when available, else CPU, and
skips when the checkpoint is absent.
"""

import io
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import anomalygen
from anomalygen.eval.fid import _compute_fid_on_feats, _sqrtm_psd, compute_feats, mask_crop_images

_CRADIO_CKPT = str(
    Path(anomalygen.__file__).resolve().parent.parent / "checkpoints" / "nvidia" / "C-RADIO-V3" / "model.safetensors"
)
_COCO_URL = "http://images.cocodataset.org/val2017/000000000285.jpg"  # a brown bear

# Golden C-RADIO-V3 feature fingerprint for the COCO bear resized to 256, captured from this
# checkpoint. Regenerate if the checkpoint or the compute_feats preprocessing changes.
_GOLDEN_FEATS_FIRST8 = torch.tensor([-0.70482, 0.61647, 1.11028, -1.03978, -1.64905, -1.29541, 3.09255, -0.05893])
_GOLDEN_FEATS_NORM = 75.49334
_GOLDEN_FEATS_MEAN = 0.004745


def test_sqrtm_psd_diagonal():
    a = torch.diag(torch.tensor([4.0, 9.0]))
    root = _sqrtm_psd(a)
    assert torch.allclose(root, torch.diag(torch.tensor([2.0, 3.0])), atol=1e-5)


def test_sqrtm_psd_reconstructs_matrix():
    torch.manual_seed(0)
    m = torch.randn(4, 4)
    a = m @ m.T  # symmetric PSD
    root = _sqrtm_psd(a)
    assert torch.allclose(root @ root, a, atol=1e-4)


def test_compute_fid_identical_sets_is_zero():
    torch.manual_seed(0)
    feats = torch.randn(8, 3)
    assert _compute_fid_on_feats(feats, feats) == pytest.approx(0.0, abs=1e-3)


def test_compute_fid_mean_shift_equals_squared_distance():
    torch.manual_seed(0)
    feats = torch.randn(8, 3)
    offset = torch.tensor([1.0, 2.0, 0.5])
    shifted = feats + offset  # same covariance, mean shifted by `offset`
    # With equal covariances FID reduces to ||mu1 - mu2||^2 = sum(offset^2).
    assert _compute_fid_on_feats(feats, shifted) == pytest.approx(float((offset**2).sum()), abs=1e-3)


def test_compute_fid_rejects_none_and_too_few_samples():
    feats = torch.randn(4, 3)
    with pytest.raises(ValueError):
        _compute_fid_on_feats(None, feats)
    with pytest.raises(ValueError):
        _compute_fid_on_feats(feats[:1], feats)  # need > 1 sample per side


def _image_and_mask(size=100, blobs=((40, 40),), side=20):
    img = np.random.default_rng(0).random((size, size, 3)).astype(np.float32)
    mask = np.zeros((size, size), np.float32)
    for r, c in blobs:
        mask[r : r + side, c : c + side] = 1.0
    return img, mask


def test_mask_crop_images_single_instance():
    img, mask = _image_and_mask(blobs=((40, 40),))
    d = {"original_image": [img], "original_mask": [mask]}
    mask_crop_images(d, "original_image")
    assert len(d["mask_cropped_image"]) == 1
    assert d["mask_cropped_image"][0].shape == (512, 512, 3)  # resized crop
    assert d["num_instance"] == [1]


def test_mask_crop_images_respects_crop_size():
    img, mask = _image_and_mask(blobs=((40, 40),))
    d = {"original_image": [img], "original_mask": [mask]}
    mask_crop_images(d, "original_image", crop_size=64)
    assert d["mask_cropped_image"][0].shape == (64, 64, 3)


def test_mask_crop_images_empty_mask():
    img = np.random.default_rng(1).random((100, 100, 3)).astype(np.float32)
    mask = np.zeros((100, 100), np.float32)
    d = {"original_image": [img], "original_mask": [mask]}
    mask_crop_images(d, "original_image")
    assert d["mask_cropped_image"] == []
    assert d["num_instance"] == [0]


def test_mask_crop_images_two_separated_instances():
    img, mask = _image_and_mask(blobs=((5, 5), (80, 80)), side=10)  # centres far apart (> eps)
    d = {"original_image": [img], "original_mask": [mask]}
    mask_crop_images(d, "original_image")
    assert d["num_instance"] == [2]
    assert len(d["mask_cropped_image"]) == 2


def test_mask_crop_images_is_idempotent():
    img, mask = _image_and_mask()
    d = {"original_image": [img], "original_mask": [mask]}
    mask_crop_images(d, "original_image")
    first = d["mask_cropped_image"]
    mask_crop_images(d, "original_image")  # early-returns; does not recompute
    assert d["mask_cropped_image"] is first


@pytest.mark.skipif(not os.path.exists(_CRADIO_CKPT), reason="C-RADIO-V3 checkpoint not present")
def test_compute_feats_matches_golden_on_coco_image():
    """Golden: real COCO image -> C-RADIO-V3 feature vector matches the captured fingerprint.

    Exercises the full compute_feats path (ImageNet transform + resize + real backbone forward),
    on CUDA when available else CPU. NB: FID self-match is intentionally NOT used as a golden — with
    few samples the covariance is rank-deficient and sqrtm is numerically unstable (nonzero, not 0).
    """
    try:
        raw = urllib.request.urlopen(_COCO_URL, timeout=30).read()  # noqa: S310 (fixed COCO URL)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip(f"could not download COCO sample image: {exc}")

    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((256, 256))
    arr = np.asarray(img, dtype=np.float32) / 255.0

    feats = compute_feats([arr], backbone_name="cradio_v3_base")
    assert feats.shape == (1, 2304)
    f = feats[0].float().cpu()
    # Fingerprint must match the golden (tolerances absorb cross-platform float noise).
    assert torch.allclose(f[:8], _GOLDEN_FEATS_FIRST8, atol=1e-2)
    assert f.norm().item() == pytest.approx(_GOLDEN_FEATS_NORM, abs=0.3)
    assert f.mean().item() == pytest.approx(_GOLDEN_FEATS_MEAN, abs=1e-2)
