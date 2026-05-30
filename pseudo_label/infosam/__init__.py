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
Reference: InfoSAM, https://github.com/MuyaoYuan/InfoSAM
"""

from pseudo_label.infosam.build_infosam2 import build_infosam2
from pseudo_label.infosam.dataset import InfoSAM2Dataset, InfoSAM2EvalDataset
from pseudo_label.infosam.infosam2_module import RelationModel
from pseudo_label.infosam.losses import DualMiLoss, StructureLoss
from pseudo_label.infosam.utils import prepare_prompts, secure_uniform, get_parameter_names
