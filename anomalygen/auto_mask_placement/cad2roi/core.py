# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CAD to ROI Generator: unified entry point for all defect types."""

from anomalygen.auto_mask_placement.cad2roi.defects.bridge import get_bridge_candidates
from anomalygen.auto_mask_placement.cad2roi.defects.excess_solder import get_excess_solder_candidates
from anomalygen.auto_mask_placement.cad2roi.defects.less_solder import get_less_solder_candidates
from anomalygen.auto_mask_placement.cad2roi.defects.missing import get_missing_candidates
from anomalygen.auto_mask_placement.cad2roi.parser import CADParser


class CADToROIGenerator:
    """Generate ROI masks from CAD semantic segmentation."""

    DEFECT_TYPES = ("missing", "less_solder", "excess_solder", "bridge")

    def __init__(
        self,
        label_path: str,
        min_area_abs: int = 5,
        fragment_ratio: float = 0.1,
        bridge_max_cut_ratio: float = 0.25,
        bridge_classes: tuple = ("pad", "solder"),
    ):
        self.parser = CADParser(label_path, min_area_abs=min_area_abs, fragment_ratio=fragment_ratio)
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
        result["missing"] = get_missing_candidates(components, img_shape, min_area_abs=self.parser.min_area_abs)
        result["less_solder"] = get_less_solder_candidates(components, img_shape)
        result["excess_solder"] = get_excess_solder_candidates(
            components, img_shape, min_area_abs=self.parser.min_area_abs
        )

        result["bridge"] = get_bridge_candidates(
            components,
            img_shape,
            bridge_classes=self._bridge_classes,
            bridge_max_cut_ratio=self._bridge_max_cut_ratio,
            min_area_abs=self.parser.min_area_abs,
        )

        return result
