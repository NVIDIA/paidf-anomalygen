# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for eval.anomaly_quality: geometry, gating, the aq_nn composite and the merge structure.

The heavy backbone (SAM2) is mocked — these run on CPU without weights.
"""

import math

import numpy as np

from anomalygen.eval import anomaly_quality as aq


class _FakePredictor:
    """SAM2ImagePredictor stand-in returning a fixed boolean mask."""

    def __init__(self, mask: np.ndarray):
        self._mask = mask.astype(bool)

    def set_image(self, image):  # noqa: D401 — signature-compatible no-op
        pass

    def predict(self, box=None, multimask_output=False):
        return np.asarray([self._mask]), None, None


def _defect_pair(size=20, blob=slice(8, 12)):
    """clean (flat) vs gen (a bright blob) at 255-scale, plus a mask rectangle around the blob."""
    clean = np.zeros((size, size, 3), dtype=np.float64)
    gen = clean.copy()
    gen[blob, blob, :] = 200.0
    mask = np.zeros((size, size), dtype=bool)
    mask[5:15, 5:15] = True  # 100 px, contains the 16 px blob
    return clean, gen, mask


# --------------------------------------------------------------------------- geometry
def test_boundary_iou_identity_and_disjoint():
    m = np.zeros((30, 30), dtype=bool)
    m[10:20, 10:20] = True
    assert math.isclose(aq.boundary_iou(m, m), 1.0, abs_tol=1e-9)

    other = np.zeros((30, 30), dtype=bool)
    other[0:5, 0:5] = True
    assert aq.boundary_iou(m, other) == 0.0


def test_extract_core_recovers_blob_and_empty_when_no_change():
    clean, gen, _ = _defect_pair()
    core, _, t = aq._extract_core(clean, gen)
    assert core.sum() > 0 and t >= aq._OTSU_FLOOR
    # core lands inside the blob footprint
    assert core[8:12, 8:12].any()

    empty, _, _ = aq._extract_core(clean, clean.copy())
    assert empty.sum() == 0


# --------------------------------------------------------------------------- completeness gating
def test_completeness_reliable_uses_sam_union():
    clean, gen, mask = _defect_pair()
    sam = np.zeros((20, 20), dtype=bool)
    sam[6:14, 6:14] = True  # 64 px, contains the core, size_ratio 0.64 <= CAP
    cov, mgen = aq.completeness(clean, gen, mask, _FakePredictor(sam))
    # M_gen = core ∪ sam = sam (superset); coverage = |sam ∩ mask| / |mask| = 64/100
    assert math.isclose(cov, 0.64, abs_tol=1e-6)
    assert mgen.sum() == 64


def test_completeness_falls_back_to_low_otsu_when_sam_oversegments():
    clean, gen, mask = _defect_pair()
    sam = np.ones((20, 20), dtype=bool)  # whole image, size_ratio 4.0 > CAP -> unreliable
    cov, _ = aq.completeness(clean, gen, mask, _FakePredictor(sam))
    # fallback = diff > 0.5*Otsu -> the 16 px blob; coverage = 16/100
    assert math.isclose(cov, 0.16, abs_tol=1e-6)


def test_completeness_nan_on_empty_mask():
    clean, gen, _ = _defect_pair()
    cov, mgen = aq.completeness(clean, gen, np.zeros((20, 20), dtype=bool), _FakePredictor(np.zeros((20, 20), bool)))
    assert math.isnan(cov) and mgen.sum() == 0


def test_completeness_precision_sees_spill_outside_mask():
    """M_gen stays UNCLIPPED so precision detects change spilling past the mask; a clipped M_gen would
    pin precision at 1.0. Coverage is unaffected (re-intersected per part)."""
    clean, gen, mask = _defect_pair()  # mask = [5:15, 5:15], 100 px
    sam = np.zeros((20, 20), dtype=bool)
    sam[6:14, 6:18] = True  # 8×12 = 96 px; cols 15-17 (24 px) spill OUTSIDE the mask
    cov, mgen = aq.completeness(clean, gen, mask, _FakePredictor(sam))
    assert mgen.sum() == 96  # not clipped down to the 72 px inside the mask
    prec = aq.precision(mgen, mask)
    assert prec < 1.0 and math.isclose(prec, 72 / 96, abs_tol=1e-6)
    assert math.isclose(cov, 0.72, abs_tol=1e-6)


def test_completeness_rejects_subfloor_noise():
    """A whole-image diff below the Otsu floor (faithful repaint / sensor noise) must NOT score a full
    defect — the fallback threshold is clamped to the floor rather than 0.5·floor."""
    clean, _, mask = _defect_pair()
    gen = clean + 7.0  # uniform 7/255 diff everywhere, below the floor of 8
    cov, mgen = aq.completeness(clean, gen, mask, _FakePredictor(np.zeros((20, 20), dtype=bool)))
    assert cov == 0.0 and mgen.sum() == 0
    assert math.isnan(aq.precision(mgen, mask))


def test_min_over_parts_penalises_ungenerated_part():
    """A multi-part defect is only as good as its worst-filled part (min over connected mask parts)."""
    mask = np.zeros((20, 40), dtype=bool)
    mask[5:15, 2:12] = True  # part A
    mask[5:15, 28:38] = True  # part B, disjoint from A
    mgen = np.zeros((20, 40), dtype=bool)
    mgen[5:15, 2:12] = True  # only part A filled
    assert aq._min_over_parts(mgen, mask) == 0.0  # part B empty -> min = 0
    mgen[5:15, 28:38] = True  # fill part B too
    assert math.isclose(aq._min_over_parts(mgen, mask), 1.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- aq_nn composite
def test_aq_nn_score_and_nan_handling():
    # aq_nn = completeness + nn_score
    assert math.isclose(aq._aq_nn_score({"completeness": 1.0, "nn_score": 1.0}), 2.0, abs_tol=1e-6)
    assert math.isclose(aq._aq_nn_score({"completeness": 0.3, "nn_score": 0.5}), 0.8, abs_tol=1e-6)
    # NaN terms contribute 0
    assert aq._aq_nn_score(dict.fromkeys(aq._TERMS, float("nan"))) == 0.0


def test_aq_nn_average_tracks_improving_model_across_passes():
    """Regression for the per-pass-zscore bug: as completeness/nn rise across passes, the macro
    Average that EarlyStop / TrainingReport read must rise monotonically (not sit at ~0)."""
    prev = None
    for base in (0.2, 0.4, 0.6, 0.8, 1.0):
        per_type = []
        for n in (4, 6, 5):  # imbalanced type counts — the old code jittered from this
            vals = [aq._aq_nn_score({"completeness": base, "nn_score": base}) for _ in range(n)]
            per_type.append(float(np.mean(vals)))
        avg = float(np.mean(per_type))
        if prev is not None:
            assert avg > prev, f"aq_nn Average must increase with the model; {avg} !> {prev}"
        prev = avg


# --------------------------------------------------------------------------- end-to-end (mocked SAM2)
def test_compute_anomaly_quality_kpi_structure(monkeypatch):
    sam = np.zeros((20, 20), dtype=bool)
    sam[6:14, 6:14] = True
    monkeypatch.setattr(aq, "_sam_predictor", lambda device: _FakePredictor(sam))

    clean, gen, mask = _defect_pair()
    name = "metal_surface+MT_Blowhole"
    generated = {
        name: {
            "original_image": [clean / 255.0, clean / 255.0],
            "reconstructed_image": [gen / 255.0, gen / 255.0],
            "original_mask": [mask.astype(float), mask.astype(float)],
            "img_path": ["a.png", "b.png"],
        }
    }
    real = {name: {"original_image": [clean / 255.0], "original_mask": [mask.astype(float)]}}
    corr = {name: {"per_sample": [{"path": "a.png", "nn_score": 0.8}, {"path": "b.png", "nn_score": 0.2}]}}

    out = aq.compute_anomaly_quality_kpi(real, generated, corr, device="cpu")
    assert set(out.keys()) == {name, "Average"}
    for key in ("aq_nn", "completeness", "precision", "boundary_iou"):
        assert key in out[name] and key in out["Average"]
        assert not math.isnan(out["Average"][key])


def test_aq_nn_drops_wholly_failed_sample_like_axes(monkeypatch):
    """A sample whose scoring raises is dropped from the aq_nn mean (as the axes drop it via
    ``_nanmean``), not counted as a hard 0 that drags the composite down.
    """
    sam = np.zeros((20, 20), dtype=bool)
    sam[6:14, 6:14] = True
    monkeypatch.setattr(aq, "_sam_predictor", lambda device: _FakePredictor(sam))

    clean, gen, mask = _defect_pair()
    name = "metal_surface+MT_Blowhole"
    generated = {
        name: {
            "original_image": [clean / 255.0, clean / 255.0],  # two identical samples
            "reconstructed_image": [gen / 255.0, gen / 255.0],
            "original_mask": [mask.astype(float), mask.astype(float)],
            "img_path": ["a.png", "b.png"],
        }
    }
    real = {name: {"original_image": [clean / 255.0], "original_mask": [mask.astype(float)]}}
    corr = {name: {"per_sample": [{"path": "a.png", "nn_score": 0.5}, {"path": "b.png", "nn_score": 0.5}]}}

    base = aq.compute_anomaly_quality_kpi(real, generated, corr, device="cpu")[name]["aq_nn"]

    # Make the SECOND completeness call raise, so sample "b" is wholly-failed (all terms NaN).
    real_completeness = aq.completeness
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return real_completeness(*args, **kwargs)

    monkeypatch.setattr(aq, "completeness", flaky)
    out = aq.compute_anomaly_quality_kpi(real, generated, corr, device="cpu")[name]

    # Two identical samples -> mean == a single sample's score; dropping the failed one keeps that
    # value, NOT (base + 0) / 2. And the surviving sample's axes stay present (not NaN).
    assert math.isclose(out["aq_nn"], base, abs_tol=1e-9)
    assert not math.isnan(out["completeness"])


def test_compute_returns_empty_when_models_unavailable(monkeypatch):
    def _boom(device):
        raise RuntimeError("no weights")

    monkeypatch.setattr(aq, "_sam_predictor", _boom)
    assert aq.compute_anomaly_quality_kpi({}, {}, {}, device="cpu") == {}
