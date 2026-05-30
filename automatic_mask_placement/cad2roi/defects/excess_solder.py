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

"""Excess solder defect: ROI = pad+solder group, AMP places submask within group.

Same placement logic as less_solder, but ROI is the merged pad+solder connected
group instead of individual pad components."""

from typing import List

from automatic_mask_placement.cad2roi.parser import ROICandidate
from automatic_mask_placement.cad2roi.defects.bridge import get_bridge_groups
from automatic_mask_placement.cad2roi.defects.less_solder import LessSolderMaskPlacer


def get_excess_solder_candidates(components: dict, img_shape: tuple,
                                 min_area_abs: int = 5) -> List[ROICandidate]:
    """Each pad+solder connected group is an excess_solder ROI candidate."""
    return get_bridge_groups(components, img_shape,
                             classes=("pad", "solder"),
                             min_area_abs=min_area_abs)


# Placement logic is identical to less_solder (scale submask to ROI, AMP place,
# clip + area check). Reuse directly.
ExcessSolderMaskPlacer = LessSolderMaskPlacer
