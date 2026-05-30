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

import re
import secrets

import torch
from sam2.utils.transforms import SAM2Transforms


def secure_uniform(a=0.0, b=1.0) -> float:
    upper_bound = 2**63
    random_fraction = secrets.randbelow(upper_bound) / upper_bound
    return a + (b - a) * random_fraction


def secure_choice(sequence):
    idx = secrets.randbelow(len(sequence))
    return sequence[idx]


def get_parameter_names(model, forbidden_layer_types=None, forbidden_layer_names=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    if forbidden_layer_types is None:
        forbidden_layer_types = [torch.nn.LayerNorm]
    if forbidden_layer_names is None:
        forbidden_layer_names = [
            r"bias",
            r"layernorm",
            r"rmsnorm",
            r"(?:^|\.)norm(?:$|\.)",
            r"_norm(?:$|\.)",
        ]
    forbidden_layer_patterns = (
        [re.compile(pattern) for pattern in forbidden_layer_names]
        if forbidden_layer_names is not None
        else []
    )
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(
            child, forbidden_layer_types, forbidden_layer_names
        )
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(forbidden_layer_types))
            and not any(
                pattern.search(f"{name}.{n}".lower())
                for pattern in forbidden_layer_patterns
            )
        ]
    # Add model specific parameters that are not in any child
    result += [
        k
        for k in model._parameters
        if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
    ]
    return result


def prepare_prompts(
    transforms: SAM2Transforms,
    orig_hw,
    point_coords,
    point_labels,
    normalize_coords=True,
    device=None,
):
    if point_coords is None or point_labels is None:
        raise ValueError("point_coords and point_coords must be provided.")

    point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=device)
    unnorm_coords = []
    for i in range(len(orig_hw)):
        unnorm_coords_i = transforms.transform_coords(
            point_coords[i], normalize=normalize_coords, orig_hw=orig_hw[i]
        )
        unnorm_coords.append(unnorm_coords_i)
    unnorm_coords = torch.stack(unnorm_coords, dim=0)
    labels = torch.as_tensor(point_labels, dtype=torch.int, device=device)
    if len(unnorm_coords.shape) == 2:
        unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
    return unnorm_coords, labels
