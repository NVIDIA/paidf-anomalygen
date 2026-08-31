# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the texture ``evaluate.py`` KPI CLI.

Mirrors the source repo's ``tests/test_evaluate.py``: a real+generated tree built from deterministic
synthetic images drives the CLI end-to-end through the real DINOv2 (and, for FID, C-RADIO-V3) path — no
patching, no network. These run on CUDA when available else CPU, and skip cleanly without the checkpoint.

The golden ``test_evaluate_self_match_...`` pins the numeric output: generated == real must score
exactly 1.0, an analytic value that would fail on any numeric drift from the migration.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import anomalygen
from anomalygen.models.vision_encoder.dinov2 import DEFAULT_BACKBONE
from anomalygen.scripts.texture import evaluate as evaluate_module

_DINOV2_WEIGHTS = os.path.join(DEFAULT_BACKBONE, "model.safetensors")
_CRADIO_CKPT = str(
    Path(anomalygen.__file__).resolve().parent.parent / "checkpoints" / "nvidia" / "C-RADIO-V3" / "model.safetensors"
)


def _synthetic_images(n, size=224):
    """Return ``n`` distinct deterministic RGB uint8 images — hermetic (no network) for CI."""
    rng = np.random.default_rng(0)
    return [(rng.random((size, size, 3)) * 255).astype(np.uint8) for _ in range(n)]


def _center_mask(size=224):
    mask = np.zeros((size, size), np.uint8)
    q = size // 4
    mask[q : size - q, q : size - q] = 255
    return mask


def _write_rgb(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)


def _build_tree(base_dir, anomaly_types, real_imgs, gen_imgs):
    """Real tree ({texture}/anomaly_image/{defect} + mask) and generated tree (recon/mask/orig)."""
    real_root, gen_root = base_dir / "real", base_dir / "gen"
    mask = _center_mask(real_imgs[0].shape[0])
    for texture, defect in anomaly_types:
        key = f"{texture}+{defect}"
        for idx, arr in enumerate(real_imgs):
            _write_rgb(real_root / texture / "anomaly_image" / defect / f"r{idx}.png", arr)
            _write_mask(real_root / texture / "mask" / defect / f"r{idx}_mask.png", mask)
        for idx, arr in enumerate(gen_imgs):
            name = f"{key}_{idx:05d}.png"
            _write_rgb(gen_root / "reconstructed_image" / name, arr)
            _write_mask(gen_root / "original_mask" / name, mask)
            _write_rgb(gen_root / "original_image" / name, arr)
    return real_root, gen_root


def test_evaluate_raises_when_no_matching_generated(tmp_path):
    """The 'no generated images' guard fires before any model call (CPU-only, no checkpoint)."""
    recon = tmp_path / "gen" / "reconstructed_image"
    recon.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(recon / "A+b_00000.png")

    with pytest.raises(RuntimeError):
        evaluate_module.main(
            [
                "--gen_root",
                str(tmp_path / "gen"),
                "--real_root",
                str(tmp_path / "real"),
                "--anomaly_types",
                "NOPE+x",
                "--output_file",
                str(tmp_path / "k.json"),
            ]
        )


@pytest.mark.gpu  # DINOv2 forward is ~30s/img on the CPU CI runner; run this integration check on GPU
@pytest.mark.skipif(not os.path.exists(_DINOV2_WEIGHTS), reason="DINOv2 checkpoint not present")
def test_evaluate_writes_kpi_json_for_each_anomaly_type(tmp_path):
    imgs = _synthetic_images(2)
    anomaly_types = [("SEM_IC", "crack"), ("OPT", "scratch")]
    real_root, gen_root = _build_tree(tmp_path, anomaly_types, real_imgs=imgs, gen_imgs=imgs)
    out = tmp_path / "kpi.json"

    evaluate_module.main(
        [
            "--gen_root",
            str(gen_root),
            "--real_root",
            str(real_root),
            "--anomaly_types",
            "SEM_IC+crack",
            "OPT+scratch",
            "--top_k",
            "1",
            "--output_file",
            str(out),
        ]
    )

    kpi = json.loads(out.read_text())
    assert set(kpi) == {"SEM_IC+crack", "OPT+scratch", "Average"}
    assert len(kpi["SEM_IC+crack"]["per_sample"]) == 2
    assert isinstance(kpi["Average"]["nn_score"], float)


@pytest.mark.gpu
@pytest.mark.skipif(not os.path.exists(_DINOV2_WEIGHTS), reason="DINOv2 checkpoint not present")
def test_evaluate_infers_anomaly_types_from_generated_filenames(tmp_path):
    imgs = _synthetic_images(1)
    anomaly_types = [("SEM_IC", "crack"), ("OPT", "scratch")]
    real_root, gen_root = _build_tree(tmp_path, anomaly_types, real_imgs=imgs, gen_imgs=imgs)
    out = tmp_path / "kpi.json"

    # No --anomaly_types: types must be inferred from the reconstructed_image/ filenames.
    evaluate_module.main(
        ["--gen_root", str(gen_root), "--real_root", str(real_root), "--top_k", "1", "--output_file", str(out)]
    )

    kpi = json.loads(out.read_text())
    assert set(kpi) == {"SEM_IC+crack", "OPT+scratch", "Average"}


@pytest.mark.skipif(not os.path.exists(_DINOV2_WEIGHTS), reason="DINOv2 checkpoint not present")
def test_evaluate_self_match_golden_is_perfect(tmp_path):
    """Golden correctness: generated == real yields exactly 1.0 NN/MNN through the real pipeline.

    The self-match value is analytic (identical masked features → cosine correspondence 1.0) and
    device-independent, so it pins the migrated evaluate output — numeric drift fails here.
    """
    # _build_tree writes original_image == reconstructed_image, so the diff is empty and
    # anomaly_quality scores nothing. The axes need a real change inside the mask, so paint one.
    clean = _synthetic_images(1)[0]
    mask = _center_mask(clean.shape[0])
    defective = clean.copy()
    defective[mask > 0] = 255 - defective[mask > 0]

    real_root, gen_root = tmp_path / "real", tmp_path / "gen"
    _write_rgb(real_root / "SEM_IC" / "anomaly_image" / "crack" / "r0.png", defective)
    _write_mask(real_root / "SEM_IC" / "mask" / "crack" / "r0_mask.png", mask)
    name = "SEM_IC+crack_00000.png"
    _write_rgb(gen_root / "reconstructed_image" / name, defective)
    _write_mask(gen_root / "original_mask" / name, mask)
    _write_rgb(gen_root / "original_image" / name, clean)
    out = tmp_path / "kpi.json"

    evaluate_module.main(
        [
            "--gen_root",
            str(gen_root),
            "--real_root",
            str(real_root),
            "--anomaly_types",
            "SEM_IC+crack",
            "--top_k",
            "1",
            "--output_file",
            str(out),
        ]
    )

    kpi = json.loads(out.read_text())
    assert kpi["SEM_IC+crack"]["nn_score"] == pytest.approx(1.0, abs=1e-4)
    assert kpi["SEM_IC+crack"]["mnn_score"] == pytest.approx(1.0, abs=1e-4)
    assert kpi["Average"]["nn_score"] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.skipif(
    not (os.path.exists(_DINOV2_WEIGHTS) and os.path.exists(_CRADIO_CKPT)),
    reason="DINOv2 and/or C-RADIO-V3 checkpoint not present",
)
@pytest.mark.gpu
def test_evaluate_merges_fid_when_requested(tmp_path):
    imgs = _synthetic_images(2)  # FID needs >= 2 defect crops per side
    real_root, gen_root = _build_tree(tmp_path, [("SEM_IC", "crack")], real_imgs=imgs, gen_imgs=imgs)
    out = tmp_path / "kpi.json"

    evaluate_module.main(
        [
            "--gen_root",
            str(gen_root),
            "--real_root",
            str(real_root),
            "--anomaly_types",
            "SEM_IC+crack",
            "--top_k",
            "1",
            "--fid_crop_size",
            "64",  # tiny cradio resolution — keeps the CPU forward fast
            "--output_file",
            str(out),
        ]
    )

    kpi = json.loads(out.read_text())
    assert "nn_score" in kpi["SEM_IC+crack"]  # correspondence preserved
    assert "fid" in kpi["SEM_IC+crack"]  # FID merged in (compute_fid_kpi keys it "fid")


_SAM2_CKPT = str(
    Path(anomalygen.__file__).resolve().parent.parent
    / "checkpoints"
    / "facebook"
    / "sam2.1-hiera-large"
    / "sam2.1_hiera_large.pt"
)


@pytest.mark.gpu
@pytest.mark.skipif(
    not (os.path.exists(_DINOV2_WEIGHTS) and os.path.exists(_SAM2_CKPT)),
    reason="DINOv2 or SAM2 checkpoint not present",
)
def test_quality_axes_land_on_rows_and_per_sample_axes_is_stripped(tmp_path):
    """The aq merge folds each axis onto the existing per_sample rows and drops the axis-only block.

    Without the strip, every axis value is written twice — once per row and again under
    ``per_sample_axes`` — and no other assertion notices, because ``compute_anomaly_quality_kpi``
    returns the same top-level keys the correspondence KPI already has, so the set of keys is
    unchanged either way. This is also the only assertion that exercises the aq wiring in
    ``evaluate.py`` rather than the two helpers in isolation.
    """
    # _build_tree writes original_image == reconstructed_image, so the diff is empty and
    # anomaly_quality scores nothing. The axes need a real change inside the mask, so paint one.
    clean = _synthetic_images(1)[0]
    mask = _center_mask(clean.shape[0])
    defective = clean.copy()
    defective[mask > 0] = 255 - defective[mask > 0]

    real_root, gen_root = tmp_path / "real", tmp_path / "gen"
    _write_rgb(real_root / "SEM_IC" / "anomaly_image" / "crack" / "r0.png", defective)
    _write_mask(real_root / "SEM_IC" / "mask" / "crack" / "r0_mask.png", mask)
    name = "SEM_IC+crack_00000.png"
    _write_rgb(gen_root / "reconstructed_image" / name, defective)
    _write_mask(gen_root / "original_mask" / name, mask)
    _write_rgb(gen_root / "original_image" / name, clean)
    out = tmp_path / "kpi.json"

    evaluate_module.main(
        ["--gen_root", str(gen_root), "--real_root", str(real_root), "--top_k", "1", "--output_file", str(out)]
    )

    kpi = json.loads(out.read_text())
    block = kpi["SEM_IC+crack"]
    assert "per_sample_axes" not in block, "axis-only block must not be duplicated into the JSON"
    row = block["per_sample"][0]
    for axis in ("completeness", "precision", "boundary_iou"):
        assert f"{axis}_score" in row, f"{axis} missing from the per-sample row"
    assert "aq_nn_score" in row
