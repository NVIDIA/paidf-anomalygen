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

import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import shift, rotate

def check_mask_is_binary(mask: np.ndarray):
    """Check if the mask is binary (including only 0 and 255).
    A mask is binary if it contains only 0 and 255 values, even if it's all zeros.
    """
    return np.all(np.isin(mask, [0, 255]))

    
def augment_binary_mask(pil_mask, shift_values=(0, 0), rotation_angle=0, morph_operation=None, kernel_size=3):
    """
    Augments a binary mask with shifting, rotation, and morphological operations.

    Parameters:
        mask (numpy.ndarray): Binary mask to augment.
        shift_values (tuple): Tuple of (shift_x, shift_y) values for shifting the mask.
        rotation_angle (float): Angle in degrees to rotate the mask.
        morph_operation (str): Morphological operation to apply ('dilate', 'erode', 'open', 'close').
        kernel_size (int): Size of the kernel for morphological operations.

    Returns:
        numpy.ndarray: Augmented binary mask.
    """
    mask = np.array(pil_mask)
    
    # Ensure the mask is binary
    if check_mask_is_binary(mask):
        mask = (mask > 0).astype(np.uint8)
    else:
        raise ValueError("Mask is not binary. Please check the mask format.")

    # Apply shifting
    if shift_values is not None:
        height, width = mask.shape
        vstep, hstep = shift_values
        shifted_mask = shift(mask, shift=(vstep, hstep), mode='constant')
        # Calculate bounding box of the masked region
        rows, cols = np.where(mask == 1)
        if len(rows) == 0 or len(cols) == 0:
            return pil_mask, False  # No masked region present

        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        # Check if shifting would move the region out of bounds
        if (((min_row + vstep) < 0) or ((max_row + vstep) >= height) or
            ((min_col + hstep) < 0) or ((max_col + hstep) >= width)):
            return None, False  # Mask moved out of bounds
        mask = shifted_mask
        
    # Apply rotation
    if rotation_angle is not None:
        mask = rotate(mask, angle=rotation_angle, reshape=False, mode='nearest')

    # Apply morphological operation
    if morph_operation is not None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if morph_operation == 'dilate':
            mask = cv2.dilate(mask, kernel, iterations=1)
        elif morph_operation == 'erode':
            mask = cv2.erode(mask, kernel, iterations=1)
        elif morph_operation == 'open':
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif morph_operation == 'close':
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif morph_operation == 'none':
            pass
        else:
            raise ValueError(f"Invalid morphological operation: {morph_operation}")
        
    # Check if no mask given
    mask = (mask * 255).astype(np.uint8)
    mask_region_y, mask_region_x = np.nonzero(mask > 127)
    if mask_region_x.size == 0 or mask_region_y.size == 0:
        return pil_mask, False  # Mask moved out of bounds
    return Image.fromarray(mask), True


def get_bbox_dimensions(mask_image):
    """
    Calculate the width and height of the bounding box of the mask region.

    Parameters:
        mask_image (PIL.Image.Image): Binary mask image (mode "1" or "L").

    Returns:
        tuple: Dimensions of the bounding box as (x_length, y_length).
               Returns (0, 0) if no mask region is found.
    """
    # Ensure the image is in binary or grayscale mode
    if mask_image.mode not in ["1", "L"]:
        raise ValueError("Mask image must be in mode '1' or 'L'.")

    # Get bounding box coordinates
    bbox = mask_image.getbbox()
    
    if bbox:
        left, upper, right, lower = bbox
        x_length = right - left  # Width of the bounding box
        y_length = lower - upper  # Height of the bounding box
        return x_length, y_length
    else:
        return 0, 0  # No mask region found


def get_crop_grid_by_ratio(mask_image, crop_ratio):
    crop_grid_x, crop_grid_y = get_bbox_dimensions(mask_image)
    crop_grid = max(crop_grid_x, crop_grid_y)
    crop_grid = int(crop_grid * crop_ratio)
    return crop_grid


def enlarge_mask(pil_mask):
    # Find contours of the circle
    if pil_mask.mode != "L":
        pil_mask = pil_mask.convert("L")
    opencv_mask = np.array(pil_mask)
    contours, _ = cv2.findContours(opencv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_mask = np.zeros_like(opencv_mask)

    for contour in contours:
        # Calculate contour moments to find the center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Fit a circle around the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)

            # Draw a new larger circle (twice the radius)
            new_radius = int(radius * 2)
            cv2.circle(new_mask, (int(x), int(y)), new_radius, 255, -1)
    new_mask = Image.fromarray(new_mask)
    return new_mask