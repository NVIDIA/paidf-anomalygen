# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CAD-driven ROI generation: parse a CAD mask, then build per-defect ROI candidates and placers.

Public API:
- ``CADToROIGenerator``: one-shot generation of all defect ROI candidates from a CAD mask.
- ``CADParser`` / ``CADComponent`` / ``ROICandidate``: the parsed CAD data model.
- ``get_*_candidates`` / ``get_bridge_groups``: per-defect ROI-candidate factories.
- ``*MaskPlacer``: per-defect submask placement onto ROI candidates.
- ``visualize_roi_candidates``: overlay helper.
"""

from anomalygen.auto_mask_placement.cad2roi.core import CADToROIGenerator
from anomalygen.auto_mask_placement.cad2roi.defects.bridge import (
    BridgeMaskPlacer,
    get_bridge_candidates,
    get_bridge_groups,
)
from anomalygen.auto_mask_placement.cad2roi.defects.excess_solder import (
    ExcessSolderMaskPlacer,
    get_excess_solder_candidates,
)
from anomalygen.auto_mask_placement.cad2roi.defects.less_solder import (
    LessSolderMaskPlacer,
    get_less_solder_candidates,
)
from anomalygen.auto_mask_placement.cad2roi.defects.missing import MissingMaskPlacer, get_missing_candidates
from anomalygen.auto_mask_placement.cad2roi.parser import CADComponent, CADParser, ROICandidate
from anomalygen.auto_mask_placement.cad2roi.visualize import visualize_roi_candidates

__all__ = [
    # CAD data model
    "CADParser",
    "CADComponent",
    "ROICandidate",
    # Generator
    "CADToROIGenerator",
    # ROI-candidate factories
    "get_missing_candidates",
    "get_less_solder_candidates",
    "get_excess_solder_candidates",
    "get_bridge_candidates",
    "get_bridge_groups",
    # Placers
    "MissingMaskPlacer",
    "LessSolderMaskPlacer",
    "ExcessSolderMaskPlacer",
    "BridgeMaskPlacer",
    # Visualization
    "visualize_roi_candidates",
]
