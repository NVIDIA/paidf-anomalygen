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

"""
Portions provided under the following terms:
<insert contents of https://github.com/haotian-liu/LLaVA/blob/main/LICENSE> 
"""

import torch
import torch.nn as nn
from loguru import logger as logging
from cosmos_predict2.models.ag_modules.utils import freeze_model


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_adapter(adapter_type, adapter_config, freeze = False):
    """ Adapted from LLaVA's projector implementation.
    https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py
    """
    # Initialize adapter
    if adapter_type == 'linear':
        input_hidden_size = adapter_config['input_hidden_size']
        final_hidden_size = adapter_config['final_hidden_size']
        adapter =  nn.Linear(input_hidden_size, final_hidden_size)
    elif adapter_type == 'identity':
        adapter = IdentityMap()
    elif adapter_type == 'mlp_gelu':
        input_hidden_size = adapter_config['input_hidden_size']
        final_hidden_size = adapter_config['final_hidden_size']
        num_layers = adapter_config['num_layers']
        modules = [nn.Linear(input_hidden_size, final_hidden_size)]
        for _ in range(1, num_layers):
            modules.append(nn.GELU())
            modules.append(nn.Linear(final_hidden_size, final_hidden_size))
        adapter =  nn.Sequential(*modules)
    elif adapter_type == 'transformer':
        adapter = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(adapter_config['input_hidden_size'], adapter_config['num_heads']),
            num_layers=adapter_config['num_layers']
        )
    else:
        raise ValueError(f'Unknown projector type: {adapter_type}')

    # Freeze / Unfreeze adapter
    if freeze:
        adapter = freeze_model(adapter)
        logging.info(f"Adapter {adapter} is freezed.")
    return adapter    
