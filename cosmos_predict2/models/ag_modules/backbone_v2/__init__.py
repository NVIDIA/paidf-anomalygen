# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Backbone module.
"""
from cosmos_predict2.models.ag_modules.backbone_v2.registry import BACKBONE_REGISTRY
from cosmos_predict2.models.ag_modules.backbone_v2.radio import (
    c_radio_p1_vit_huge_patch16_mlpnorm,
    c_radio_p2_vit_huge_patch16_mlpnorm,
    c_radio_p3_vit_huge_patch16_mlpnorm,
    c_radio_v2_vit_base_patch16,
    c_radio_v2_vit_large_patch16,
    c_radio_v2_vit_huge_patch16,
    c_radio_v3_vit_base_patch16_reg4_dinov2,
    c_radio_v3_vit_large_patch16_reg4_dinov2,
    c_radio_v3_vit_huge_patch16_reg4_dinov2,
)

__all__ = [
    "BACKBONE_REGISTRY",
    "c_radio_p1_vit_huge_patch16_mlpnorm",
    "c_radio_p2_vit_huge_patch16_mlpnorm",
    "c_radio_p3_vit_huge_patch16_mlpnorm",
    "c_radio_v2_vit_base_patch16",
    "c_radio_v2_vit_large_patch16",
    "c_radio_v2_vit_huge_patch16",
    "c_radio_v3_vit_large_patch16_reg4_dinov2",
    "c_radio_v3_vit_base_patch16_reg4_dinov2",
    "c_radio_v3_vit_huge_patch16_reg4_dinov2",

]
