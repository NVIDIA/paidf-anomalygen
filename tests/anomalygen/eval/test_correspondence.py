# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure-math helpers and input validation of eval.correspondence.

The model-dependent scoring path (real DINOv2) is not exercised here — these run on CPU.
"""

import io
import math
import os
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch
from PIL import Image

from anomalygen.eval.correspondence import (
    _MAX_ZOOM_INSTANCES,
    _crop_instance,
    _image_mask_to_tensor,
    _nn_mnn_scores,
    _patch_grid_mask,
    _split_instances,
    compute_correspondence_kpi,
)
from anomalygen.models.vision_encoder.dinov2 import DEFAULT_BACKBONE

_COCO_URL = "http://images.cocodataset.org/val2017/000000000285.jpg"  # a brown bear
_DINOV2_WEIGHTS = os.path.join(DEFAULT_BACKBONE, "model.safetensors")


def test_nn_mnn_scores_identity():
    feats = torch.eye(2)  # two orthonormal patches, self-matched
    nn, mnn = _nn_mnn_scores(feats, feats)
    assert math.isclose(nn.item(), 1.0, abs_tol=1e-6)
    assert math.isclose(mnn.item(), 1.0, abs_tol=1e-6)


def test_nn_mnn_scores_partial_match():
    # gen = {e0, e1}; real = {e0}. gen0 matches (sim 1), gen1 has no partner (sim 0).
    f_g = torch.eye(2)
    f_r = torch.tensor([[1.0, 0.0]])
    nn, mnn = _nn_mnn_scores(f_g, f_r)
    assert math.isclose(nn.item(), 0.5, abs_tol=1e-6)  # mean(best per gen) = mean(1, 0)
    assert math.isclose(mnn.item(), 1.0, abs_tol=1e-6)  # only the mutual gen0<->real0 pair counts


def test_nn_readout_pooling_values():
    # One reference patch e0; four gen patches whose cosine to it is exactly {0.2,0.4,0.6,0.8}.
    f_r = torch.tensor([[1.0, 0.0]])
    cs = [0.2, 0.4, 0.6, 0.8]
    f_g = torch.tensor([[c, (1.0 - c * c) ** 0.5] for c in cs])
    mean = _nn_mnn_scores(f_g, f_r, readout="mean")[0].item()
    p25 = _nn_mnn_scores(f_g, f_r, readout="p25")[0].item()
    worst25 = _nn_mnn_scores(f_g, f_r, readout="worst25")[0].item()
    assert math.isclose(mean, 0.5, abs_tol=1e-5)  # mean of the four sims
    assert math.isclose(p25, 0.35, abs_tol=1e-5)  # 0.25-quantile (linear interp between 0.2 and 0.4)
    assert math.isclose(worst25, 0.2, abs_tol=1e-5)  # mean of the lowest 25% (1 of 4) = the worst sim


def test_crop_instance_letterboxes_non_square_image():
    # Elongated image (80×400): the square `side` gets clamped in the short axis, so the raw crop is
    # non-square. A SQUARE defect must stay ~square after the letterbox+resize (no stretching).
    height, width = 80, 400
    inst = np.zeros((height, width), dtype=bool)
    inst[15:65, 175:225] = True  # 50×50 square defect
    img = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    crop, crop_mask = _crop_instance(img, inst, out=518)
    assert crop.size[0] == crop.size[1] == crop_mask.shape[0] == crop_mask.shape[1]  # square target
    ys, xs = np.where(crop_mask)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    assert 0.85 < h / w < 1.18  # aspect preserved; stretching this crop would give h/w ≈ 1.25


def test_split_instances_caps_to_largest_by_area():
    # A speckled mask (12 disjoint blobs of distinct area) must be capped to the largest
    # `_MAX_ZOOM_INSTANCES` so the downstream DINOv2 batch is bounded (no OOM on a fragmented mask).
    sides = list(range(3, 15))  # 12 blobs, areas 9..196, all >= min_area and mutually disjoint
    mask = np.zeros((len(sides) * 40, 40), dtype=bool)
    for i, s in enumerate(sides):
        mask[i * 40 + 2 : i * 40 + 2 + s, 2 : 2 + s] = True  # 2px margin: keep off the edge so 3×3 closing is a no-op
    insts = _split_instances(mask)
    assert len(insts) == _MAX_ZOOM_INSTANCES  # 12 components capped
    kept = sorted(int(c.sum()) for c in insts)
    assert kept == sorted(s * s for s in sides)[-_MAX_ZOOM_INSTANCES:]  # exactly the largest by area
    # a single defect is returned unchanged (no spurious cap)
    single = np.zeros((100, 100), dtype=bool)
    single[20:80, 20:80] = True
    assert len(_split_instances(single)) == 1


def test_patch_grid_mask_max_pools_onto_grid():
    mask = torch.zeros(4, 4)
    mask[0, 0] = 1.0  # one active pixel in the top-left 2x2 patch
    grid = _patch_grid_mask(mask, height_patch=2, width_patch=2, patch_size=2)
    assert grid.shape == (2, 2)
    assert grid[0, 0] == 1.0  # a patch is active if any pixel inside it is
    assert grid[0, 1] == 0.0 and grid[1, 0] == 0.0 and grid[1, 1] == 0.0


def test_image_mask_to_tensor_snaps_to_patch_multiple():
    image_arr = np.random.default_rng(0).random((300, 300, 3)).astype(np.float32)
    mask_arr = np.zeros((300, 300), np.float32)
    mask_arr[100:200, 100:200] = 1.0
    x, mask_t, new_h, new_w = _image_mask_to_tensor(image_arr, mask_arr, patch_size=14)
    assert new_h % 14 == 0 and new_w % 14 == 0  # snapped to patch multiples
    assert x.shape == (3, new_h, new_w)
    assert mask_t.shape == (new_h, new_w)
    assert set(torch.unique(mask_t).tolist()).issubset({0.0, 1.0})  # mask stays binary


@pytest.mark.parametrize("bad_top_k", [0, -2])
def test_compute_correspondence_kpi_rejects_bad_top_k(bad_top_k):
    with pytest.raises(ValueError):
        compute_correspondence_kpi({}, {}, top_k=bad_top_k)


def test_compute_correspondence_kpi_rejects_unknown_backbone():
    with pytest.raises(ValueError):
        compute_correspondence_kpi({}, {}, backbone="not/a/registered/backbone")


@pytest.mark.skipif(not os.path.exists(_DINOV2_WEIGHTS), reason="DINOv2 checkpoint not present")
def test_correspondence_self_match_is_perfect():
    """Golden: a real image scored against itself yields perfect (1.0) NN/MNN correspondence.

    Runs the full real path through DINOv2 (CUDA when available, else CPU). Because the generated
    and real features over the masked region are identical, their cosine correspondence is exactly
    1.0 — a golden value that is analytic and device-independent.
    """
    try:
        raw = urllib.request.urlopen(_COCO_URL, timeout=30).read()  # noqa: S310 (fixed COCO URL)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip(f"could not download COCO sample image: {exc}")

    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mask = np.zeros((224, 224), np.float32)
    mask[60:160, 60:160] = 1.0  # a defect region over the bear

    real = {"bear": {"original_image": [arr], "original_mask": [mask]}}
    gen = {"bear": {"reconstructed_image": [arr], "original_mask": [mask], "img_path": ["gen0.png"]}}

    result = compute_correspondence_kpi(real, gen, top_k=1)

    assert result["bear"]["nn_score"] == pytest.approx(1.0, abs=1e-4)
    assert result["bear"]["mnn_score"] == pytest.approx(1.0, abs=1e-4)
    assert result["Average"]["nn_score"] == pytest.approx(1.0, abs=1e-4)
    assert result["bear"]["per_sample"][0]["path"] == "gen0.png"


@pytest.mark.skipif(not os.path.exists(_DINOV2_WEIGHTS), reason="DINOv2 checkpoint not present")
def test_correspondence_default_nn_values_are_pinned():
    """Golden: pins the nn/mnn values of the *default* scoring config (zoom / layer-12 / worst25 /
    min) on a fixed non-identical pair, so an accidental change to those defaults fails CI. A
    self-match test can't catch a default flip (it is 1.0 under any config); the old
    full / final-layer / mean defaults give ~0.95 here, far outside the tolerance.
    """

    def _pair(seed):
        rng = np.random.default_rng(seed)
        img = rng.random((224, 224, 3)).astype(np.float32)
        img[80:144, 80:144] = np.clip(img[80:144, 80:144] + 0.4, 0.0, 1.0)  # a bright central defect
        mask = np.zeros((224, 224), np.float32)
        mask[80:144, 80:144] = 1.0
        return img, mask

    gen_img, mask = _pair(0)
    ref_img, _ = _pair(1)  # different image, same mask -> a non-identical pair
    real = {"k": {"original_image": [ref_img], "original_mask": [mask]}}
    gen = {"k": {"reconstructed_image": [gen_img], "original_mask": [mask], "img_path": ["g0.png"]}}

    result = compute_correspondence_kpi(real, gen, top_k=1)  # default zoom / layer-12 / worst25 / min
    assert result["k"]["nn_score"] == pytest.approx(0.6940, abs=0.02)
    assert result["k"]["mnn_score"] == pytest.approx(0.7703, abs=0.02)
