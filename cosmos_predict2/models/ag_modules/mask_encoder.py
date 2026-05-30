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

from cosmos_predict2.models.ag_modules.dinov2_vit import VitLargePatch14Dinov2Swiglu
from cosmos_predict2.models.ag_modules.utils import freeze_model
from loguru import logger as logging
from torch import nn

def build_mask_encoder(encoder_type, encoder_config, freeze = True):
    assert encoder_type in ['nvdinov2'], f"[ERROR] Unknown mask encoder type {encoder_type}. "\
                                                                    f"Supported mask encoder: ['nvdinov2']"

    # Initialize mask_encoder
    if encoder_type == 'nvdinov2':
        mask_encoder = VitLargePatch14Dinov2Swiglu(
            init_cfg=dict(checkpoint=encoder_config.get("init_cfg", {}).get("checkpoint", None)),
            pool_kernel=encoder_config.get("pool_kernel", 7),
        ).cuda()
    else:
          raise RuntimeError(f"[ERROR] Unknown mask encoder type {encoder_type}.\nSupported mask encoder: ['nvdinov2']")

    if freeze:
        mask_encoder = freeze_model(mask_encoder)
        logging.info(f"Mask encoder {encoder_type} is freezed.")
    return mask_encoder
