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

"""Regression tests for the FID computation fix.

The fix replaces the old ``sqrtm`` of the *symmetrized product* ``σ1·σ2`` with
``sqrtm(σ1^½ · σ2 · σ1^½)``.  ``σ1·σ2`` is generally not symmetric, so
symmetrizing it before the eigendecomposition changes its eigenvalues and the
FID trace term is computed on the wrong spectrum.  The symmetric PSD form
``σ1^½·σ2·σ1^½`` shares the eigenvalues of ``σ1·σ2``, so ``Tr(covmean)`` equals
``Tr(sqrtm(σ1·σ2))`` as the FID definition requires.
"""

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fid_mod = importlib.import_module("cosmos_predict2.metrics.fid")
compute_fid_on_feats = fid_mod.compute_fid_on_feats
_sqrtm_psd = fid_mod._sqrtm_psd


def _correlated_feats(n, transform, seed, shift=0.0):
    """Deterministic feature matrix with covariance ≈ ``transform @ transform.T``."""
    generator = torch.Generator().manual_seed(seed)
    d = transform.shape[0]
    base = torch.randn(n, d, generator=generator, dtype=torch.float64)
    return base @ transform.T + shift


# Two lower-triangular mixing matrices chosen so σ1 and σ2 are non-diagonal and
# do NOT commute -> σ1·σ2 is markedly non-symmetric, which is exactly the case
# the buggy symmetrize-the-product code got wrong.
_L1 = torch.tensor([[1.0, 0.0, 0.0], [0.8, 1.0, 0.0], [0.3, 0.5, 1.0]], dtype=torch.float64)
_L2 = torch.tensor([[1.0, 0.0, 0.0], [-0.6, 1.0, 0.0], [0.2, -0.7, 1.0]], dtype=torch.float64)


def _reference_fid_via_scipy(feats_1, feats_2):
    """Ground-truth FID using scipy's dense ``sqrtm`` of σ1·σ2 (the definition)."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    mu1 = feats_1.mean(dim=0).numpy()
    mu2 = feats_2.mean(dim=0).numpy()
    sigma1 = torch.cov(feats_1.T).numpy()
    sigma2 = torch.cov(feats_2.T).numpy()
    covmean = scipy_linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def test_fid_identical_features_is_zero():
    feats = _correlated_feats(256, _L1, seed=0)
    fid = compute_fid_on_feats(feats, feats.clone())
    assert abs(fid) < 1e-6


def test_sqrtm_psd_reconstructs_known_matrix():
    # M = Q diag(d) Q^T with known d >= 0  =>  sqrtm(M) = Q diag(sqrt(d)) Q^T.
    theta = 0.7
    q = torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=torch.float64,
    )
    d = torch.tensor([4.0, 9.0], dtype=torch.float64)
    mat = q @ torch.diag(d) @ q.T
    expected = q @ torch.diag(torch.sqrt(d)) @ q.T
    got = _sqrtm_psd(mat)
    assert torch.allclose(got, expected, atol=1e-10)
    assert torch.allclose(got @ got, mat, atol=1e-10)


def test_sqrtm_psd_clamps_negative_eigenvalues():
    # A (numerically) indefinite symmetric matrix must not produce NaNs.
    mat = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)  # eigenvalues 3, -1
    root = _sqrtm_psd(mat)
    assert torch.isfinite(root).all()


def test_fid_matches_scipy_sqrtm_reference():
    feats1 = _correlated_feats(4000, _L1, seed=1)
    feats2 = _correlated_feats(4000, _L2, seed=2, shift=0.5)
    fid = compute_fid_on_feats(feats1, feats2)
    ref = _reference_fid_via_scipy(feats1, feats2)
    assert fid == pytest.approx(ref, rel=1e-4, abs=1e-4)


def test_fid_differs_from_symmetrized_product_bug():
    """The old symmetrize-the-product formula must give a materially different
    (wrong) answer; the new formula must match the scipy reference."""
    feats1 = _correlated_feats(4000, _L1, seed=3)
    feats2 = _correlated_feats(4000, _L2, seed=4, shift=0.5)

    fid = compute_fid_on_feats(feats1, feats2)
    ref = _reference_fid_via_scipy(feats1, feats2)

    mu1 = feats1.mean(dim=0)
    mu2 = feats2.mean(dim=0)
    sigma1 = torch.cov(feats1.T)
    sigma2 = torch.cov(feats2.T)

    # Old buggy computation: symmetrize the *product* before the eigendecomp.
    cov_prod = sigma1 @ sigma2
    cov_prod = (cov_prod + cov_prod.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(cov_prod)
    sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=0))
    covmean_buggy = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
    fid_buggy = (torch.sum((mu1 - mu2) ** 2) + torch.trace(sigma1 + sigma2 - 2 * covmean_buggy)).item()

    assert fid == pytest.approx(ref, rel=1e-4, abs=1e-4)
    assert abs(fid - fid_buggy) > 1e-2, "the bug should materially change the FID"
