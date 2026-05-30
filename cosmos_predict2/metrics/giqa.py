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
import torch.nn.functional as F
import numpy as np


def compute_giqa_on_feats(feats_1: torch.Tensor, feats_2: torch.Tensor, K: int = 1) -> float:
    """
    Compute G-IQA score using reciprocal L1 distances (torch tensors).
    Args:
        feats_1: (N, D) generated/test features
        feats_2: (M, D) reference/real features
    Returns:
        G-IQA score (float)
    """
    scores = []
    for i in range(feats_1.size(0)):
        query = feats_1[i].unsqueeze(0).expand(feats_2.size(0), -1)
        dists = F.pairwise_distance(query, feats_2, p=1)
        topk_vals, _ = torch.topk(dists, K, largest=False)
        score = (1.0 / (topk_vals + 1e-6)).sum().item()
        scores.append(score)
    return scores