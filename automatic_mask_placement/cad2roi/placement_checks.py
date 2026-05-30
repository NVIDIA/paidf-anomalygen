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

"""Placement validation checks for cad2roi defect mask placement."""

from typing import List

import cv2
import numpy as np

from automatic_mask_placement.cad2roi.mask_utils import mask_area


def clip_to_roi(placed: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Clip placed mask to ROI boundary."""
    return cv2.bitwise_and(placed, roi_mask)


def check_min_area(mask: np.ndarray, roi_area: int,
                   min_ratio: float = 0.5) -> bool:
    """Check that mask area is at least min_ratio of roi_area."""
    if roi_area <= 0:
        return True
    return mask_area(mask) >= roi_area * min_ratio


def check_touches_regions(mask: np.ndarray, regions: List[np.ndarray],
                          min_count: int = 2) -> bool:
    """Check that mask overlaps with at least min_count regions."""
    n = sum(1 for r in regions if cv2.bitwise_and(mask, r).sum() > 0)
    return n >= min_count


def check_single_component(mask: np.ndarray) -> bool:
    """Check that mask has exactly one connected component."""
    num_cc, _, _, _ = cv2.connectedComponentsWithStats(mask)
    return (num_cc - 1) == 1

