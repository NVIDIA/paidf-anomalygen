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

"""Unified morphological operation helpers for cad2roi.

All operations use elliptical structuring elements for consistent behavior.
"""

import cv2
import numpy as np


def dilate(mask: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    """Dilate binary mask with elliptical kernel."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask, k, iterations=iterations)


def erode(mask: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    """Erode binary mask with elliptical kernel."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_ERODE, k, iterations=iterations)


def close(mask: np.ndarray, ksize: int) -> np.ndarray:
    """Morphological close (dilate then erode) with elliptical kernel."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)