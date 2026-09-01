# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
