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
Automatic Mask Placement Package

This package implements the workflow for automatic mask placement within ROI regions.
"""

# Import main classes for easy access
from automatic_mask_placement.config import AugmentationParams
from automatic_mask_placement.core import AutomaticMaskPlacement
from automatic_mask_placement.mask_augmentor import MaskAugmentor
from automatic_mask_placement.mask_cropper import MaskCropper
from automatic_mask_placement.mask_placer import MaskPlacer, PlacementResult
from automatic_mask_placement.data_types import AlignmentPoint, BoundingBox, ROI
from automatic_mask_placement.roi_separator import ROISeparator
from automatic_mask_placement.text2roi.text2box import Text2BoxDetector, run_text2box
from automatic_mask_placement.utils import validate_binary_mask

__all__ = [
    # Main class
    'AutomaticMaskPlacement',

    # Configuration
    'AugmentationParams',

    # Models
    'AlignmentPoint',
    'BoundingBox',
    'ROI',
    'PlacementResult',

    # Helper classes
    'MaskAugmentor',
    'MaskCropper',
    'MaskPlacer',
    'ROISeparator',
    'Text2BoxDetector',

    # Utilities
    'run_text2box',
    'validate_binary_mask',
]

