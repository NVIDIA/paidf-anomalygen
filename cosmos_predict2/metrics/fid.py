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

import torch


def compute_fid_on_feats(feats_1, feats_2):
    if feats_1 is None or feats_2 is None:
        raise ValueError("One of the feature sets is None for FID computation.")
    if feats_1.shape[0] <= 1 or feats_2.shape[0] <= 1:
        raise ValueError(f"Not enough samples to compute FID: feats_1={feats_1.shape[0]}, feats_2={feats_2.shape[0]}")
    try:
        mu1 = feats_1.mean(dim=0)
        mu2 = feats_2.mean(dim=0)
        sigma1 = torch.cov(feats_1.T)
        sigma2 = torch.cov(feats_2.T)
        cov_prod = (sigma1 @ sigma2)
        cov_prod = (cov_prod + cov_prod.T) / 2  # Make it symmetric
        # Eigen-decomposition based sqrtm
        eigvals, eigvecs = torch.linalg.eigh(cov_prod)
        eigvals_clamped = torch.clamp(eigvals, min=0)
        sqrt_eigvals = torch.sqrt(eigvals_clamped)
        covmean = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
        fid = torch.sum((mu1 - mu2) ** 2) + torch.trace(sigma1 + sigma2 - 2 * covmean)
        return fid.item()
    except Exception as e:
        raise RuntimeError(f"FID computation failed: {e}")