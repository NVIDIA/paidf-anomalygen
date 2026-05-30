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
Mask Cropper module for Automatic Mask Placement.

This module handles mask cropping operations.
"""

import numpy as np


class MaskCropper:
    """Handles mask cropping operations"""
    
    @staticmethod
    def crop_mask_by_max_dimensions(mask: np.ndarray) -> np.ndarray:
        """Crop mask to its maximum height and width bounding box"""
        # Find bounding box of the mask
        rows, cols = np.where(mask > 127)
        if len(rows) == 0 or len(cols) == 0:
            return mask
        
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        # Crop to bounding box
        cropped_mask = mask[min_row:max_row+1, min_col:max_col+1]
        return cropped_mask

