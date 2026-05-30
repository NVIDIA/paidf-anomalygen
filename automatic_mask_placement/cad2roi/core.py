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

"""CAD to ROI Generator: unified entry point for all defect types."""

import numpy as np
from typing import Optional, List

from automatic_mask_placement.cad2roi.parser import CADParser, ROICandidate
from automatic_mask_placement.cad2roi.defects.missing import get_missing_candidates
from automatic_mask_placement.cad2roi.defects.less_solder import get_less_solder_candidates
from automatic_mask_placement.cad2roi.defects.excess_solder import get_excess_solder_candidates
from automatic_mask_placement.cad2roi.defects.bridge import get_bridge_candidates, get_bridge_groups


class CADToROIGenerator:
    """Generate ROI masks from CAD semantic segmentation."""

    DEFECT_TYPES = ("missing", "less_solder", "excess_solder", "bridge")

    def __init__(self, label_path: str, min_area_abs: int = 5,
                 fragment_ratio: float = 0.1,
                 bridge_max_cut_ratio: float = 0.25,
                 bridge_classes: tuple = ("pad", "solder")):
        self.parser = CADParser(label_path, min_area_abs=min_area_abs,
                                fragment_ratio=fragment_ratio)
        self._bridge_max_cut_ratio = bridge_max_cut_ratio
        self._bridge_classes = bridge_classes

    def generate_all_candidates(self, mask_path: str) -> dict:
        """Generate ALL possible ROI candidates for each defect type.

        Returns:
            dict with keys 'missing', 'less_solder', 'bridge',
            each mapping to a list of ROICandidate.
        """
        components, img_shape = self.parser.parse(mask_path)

        result = {}
        result["missing"] = get_missing_candidates(
            components, img_shape, min_area_abs=self.parser.min_area_abs)
        result["less_solder"] = get_less_solder_candidates(components, img_shape)
        result["excess_solder"] = get_excess_solder_candidates(
            components, img_shape, min_area_abs=self.parser.min_area_abs)

        result["bridge"] = get_bridge_candidates(
            components, img_shape,
            bridge_classes=self._bridge_classes,
            bridge_max_cut_ratio=self._bridge_max_cut_ratio,
            min_area_abs=self.parser.min_area_abs,
        )

        return result

    def generate(self, mask_path: str, defect_type: str, n: int = 1,
                 seed: Optional[int] = None) -> List[ROICandidate]:
        """Generate n ROI candidates for a given defect type."""
        assert defect_type in self.DEFECT_TYPES
        candidates = self.generate_all_candidates(mask_path)
        cands = candidates[defect_type]
        if not cands:
            return []
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(cands), size=min(n, len(cands)), replace=False)
        return [cands[i] for i in indices]

    def get_bridge_groups(self, mask_path: str):
        """Get pad+solder groups for bridge validation (needed by BridgeMaskPlacer)."""
        components, img_shape = self.parser.parse(mask_path)
        return get_bridge_groups(components, img_shape,
                                 classes=self._bridge_classes,
                                 min_area_abs=self.parser.min_area_abs)