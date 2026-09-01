# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Excess solder defect: ROI = pad+solder group, AMP places submask within group.

Same placement logic as less_solder, but ROI is the merged pad+solder connected
group instead of individual pad components."""

from typing import List

from anomalygen.auto_mask_placement.cad2roi.defects.bridge import get_bridge_groups
from anomalygen.auto_mask_placement.cad2roi.defects.less_solder import LessSolderMaskPlacer
from anomalygen.auto_mask_placement.cad2roi.parser import ROICandidate


def get_excess_solder_candidates(components: dict, img_shape: tuple, min_area_abs: int = 5) -> List[ROICandidate]:
    """Each pad+solder connected group is an excess_solder ROI candidate."""
    return get_bridge_groups(components, img_shape, classes=("pad", "solder"), min_area_abs=min_area_abs)


# Placement logic is identical to less_solder (scale submask to ROI, AMP place,
# clip + area check). Reuse directly.
ExcessSolderMaskPlacer = LessSolderMaskPlacer
