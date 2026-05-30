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
Data types for Automatic Mask Placement.

This module contains the core data structures:
- AlignmentPoint: Enum for alignment modes
- BoundingBox: Represents a bounding box
- ROI: Represents a Region of Interest
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class AlignmentPoint(Enum):
    """Alignment modes for mask placement within ROI"""
    CENTER = "center"           # Align to center of ROI
    TOP_LEFT = "top_left"       # Align to top-left corner
    TOP_RIGHT = "top_right"     # Align to top-right corner
    BOTTOM_LEFT = "bottom_left" # Align to bottom-left corner
    BOTTOM_RIGHT = "bottom_right" # Align to bottom-right corner
    TOP_CENTER = "top_center"   # Align to top center
    BOTTOM_CENTER = "bottom_center" # Align to bottom center
    LEFT_CENTER = "left_center" # Align to left center
    RIGHT_CENTER = "right_center" # Align to right center
    RANDOM = "random"           # Random alignment within ROI


@dataclass
class BoundingBox:
    """Represents a bounding box with x, y, width, height"""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class ROI:
    """Represents a Region of Interest"""
    bbox: BoundingBox
    is_legal: bool  # True for legal regions, False for illegal regions
    roi_id: str

