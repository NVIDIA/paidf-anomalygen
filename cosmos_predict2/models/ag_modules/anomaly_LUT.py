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
import torch.nn as nn
from loguru import logger as logging
from cosmos_predict2.models.ag_modules.utils import freeze_model

def build_anomaly_LUT(num_tokens, anomaly_types, token_dim, init_embedding = None, freeze = False):
    # Initialize LUT
    if init_embedding is not None: # Initialize with init. embedding
        anomaly_LUT = nn.ParameterDict({
            f"{texture}+{anomaly}":  torch.nn.Parameter(
                init_embedding.unsqueeze(0).repeat(num_tokens, 1),
                requires_grad = True
            )             
            for texture, anomaly in anomaly_types
        })
    else: # Random initialize
        anomaly_LUT = nn.ParameterDict({
            f"{texture}+{anomaly}":   torch.nn.Parameter(
                torch.rand(num_tokens, token_dim),
                requires_grad = True
            )
            for texture, anomaly in anomaly_types
        })

    # Freeze / Unfreeze adapter
    if freeze:
        anomaly_LUT = freeze_model(anomaly_LUT)
        logging.info(f"Anomaly LUT {anomaly_LUT} is freezed.")
    return anomaly_LUT