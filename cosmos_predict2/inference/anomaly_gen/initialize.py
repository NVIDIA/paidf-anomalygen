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

from imaginaire.utils import log
from imaginaire.lazy_config import instantiate
import torch

def initialize_anomaly_diffusion_model(config: dict, ad_checkpoint_dir: str, step: int = None):
    # Create the model
    log.info(f"Initializing Cosmos AnomalyGen model from {ad_checkpoint_dir}/checkpoints/iter_{step:>09}_reg_model.pt")
    log.info(f"Initialize Cosmos Predict model")
    model = instantiate(config.model)

    ## Load AnomalyGen Model components
    if step is not None:
        checkpoint_path = f"{ad_checkpoint_dir}/checkpoints/model/iter_{step:>09}.pt"
        log.info(f"Loading pretrained checkpoint {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, weights_only=True)
        log.info(f"Loading checkpoint to model")
        model.load_state_dict(state_dict)
    else:
        log.warning(f"No checkpoint step provided. AnomalyGen components are initialized from scratch.")

    # Freeze everything
    model.requires_grad_(False)
    return model

