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
Utility functions for Automatic Mask Placement.
"""

import numpy as np
from imaginaire.utils import log


def validate_binary_mask(mask: np.ndarray, mask_name: str = "mask") -> np.ndarray:
    """Validate that a mask contains only 0 and 255 values.
    
    Args:
        mask: The mask array to validate
        mask_name: Name of the mask for error messages
        
    Returns:
        The validated mask (normalized to 0 and 255 if needed)
        
    Raises:
        ValueError: If mask contains invalid values
    """
    unique_values = np.unique(mask)
    
    # Check if mask only contains 0 and 255
    valid_values = {0, 255}
    if set(unique_values).issubset(valid_values):
        return mask
    
    # If mask contains other values, try to normalize it
    log.warning(f"Mask '{mask_name}' contains values other than 0 and 255: {unique_values}")
    log.warning(f"Attempting to normalize mask to binary (0 or 255)...")
    
    # Normalize: values > 127 become 255, others become 0
    normalized_mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    
    # Verify normalization worked
    unique_after = np.unique(normalized_mask)
    if not set(unique_after).issubset(valid_values):
        raise ValueError(f"Failed to normalize mask '{mask_name}'. Contains values: {unique_after}")
    
    log.info(f"Successfully normalized mask '{mask_name}' to binary values (0, 255)")
    return normalized_mask

