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

"""Backbone module."""

# isort: off
# registry must be imported before radio: radio imports BACKBONE_REGISTRY from this package.
from anomalygen.models.vision_encoder.cradio.registry import BACKBONE_REGISTRY
from anomalygen.models.vision_encoder.cradio.radio import (
    c_radio_v3_vit_base_patch16_reg4_dinov2,
    c_radio_v3_vit_huge_patch16_reg4_dinov2,
    c_radio_v3_vit_large_patch16_reg4_dinov2,
)
# isort: on

__all__ = [
    "BACKBONE_REGISTRY",
    "c_radio_v3_vit_large_patch16_reg4_dinov2",
    "c_radio_v3_vit_base_patch16_reg4_dinov2",
    "c_radio_v3_vit_huge_patch16_reg4_dinov2",
]
