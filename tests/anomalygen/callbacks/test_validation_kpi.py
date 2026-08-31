# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gather + dedup-by-index logic for ValidationKPI (pure Python; no model/GPU/distributed)."""

import numpy as np

from anomalygen.callbacks.validation_kpi import _flatten_and_dedup


def _rank_dict(name, recon_vals, indices):
    gd = {name: {"reconstructed_image": [np.full((2, 2, 3), v, np.float32) for v in recon_vals]}}
    si = {name: list(indices)}
    return gd, si


def test_flatten_merges_ranks():
    gd0, si0 = _rank_dict("t+d", [0.1, 0.2], [0, 1])
    gd1, si1 = _rank_dict("t+d", [0.3], [2])
    merged = _flatten_and_dedup([gd0, gd1], [si0, si1])
    assert len(merged["t+d"]["reconstructed_image"]) == 3


def test_flatten_dedups_repeated_indices():
    # index 1 is duplicated across ranks (padding); keep first only.
    gd0, si0 = _rank_dict("t+d", [0.1, 0.2], [0, 1])
    gd1, si1 = _rank_dict("t+d", [0.9, 0.3], [1, 2])
    merged = _flatten_and_dedup([gd0, gd1], [si0, si1])
    recon = merged["t+d"]["reconstructed_image"]
    assert len(recon) == 3  # indices {0,1,2}
    # The kept index-1 sample is rank 0's (0.2), not rank 1's duplicate (0.9).
    assert any(np.allclose(r, 0.2) for r in recon)
    assert not any(np.allclose(r, 0.9) for r in recon)
