# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Automatic Mask Placement Package

This package implements the workflow for automatic mask placement within ROI regions.
"""

# Import main classes for easy access
from anomalygen.auto_mask_placement.config import AugmentationParams
from anomalygen.auto_mask_placement.core import AutoMaskPlacement
from anomalygen.auto_mask_placement.data_types import ROI, AlignmentPoint, BoundingBox
from anomalygen.auto_mask_placement.mask_augmentor import MaskAugmentor
from anomalygen.auto_mask_placement.mask_cropper import MaskCropper
from anomalygen.auto_mask_placement.mask_placer import MaskPlacer, PlacementResult
from anomalygen.auto_mask_placement.roi_separator import ROISeparator
from anomalygen.auto_mask_placement.text2roi.text2box import Text2BoxDetector, run_text2box
from anomalygen.auto_mask_placement.utils import validate_binary_mask

__all__ = [
    # Main class
    "AutoMaskPlacement",
    # Configuration
    "AugmentationParams",
    # Models
    "AlignmentPoint",
    "BoundingBox",
    "ROI",
    "PlacementResult",
    # Helper classes
    "MaskAugmentor",
    "MaskCropper",
    "MaskPlacer",
    "ROISeparator",
    "Text2BoxDetector",
    # Utilities
    "run_text2box",
    "validate_binary_mask",
]
