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
ROI Separator module for Automatic Mask Placement.

This module handles separation and management of ROI regions.
"""

import random
from typing import List, Optional

import cv2
import numpy as np
from imaginaire.utils import log

from .data_types import BoundingBox, ROI


class ROISeparator:
    """Separates and manages different ROI regions using CV algorithms"""
    
    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height
        self.rois: List[ROI] = []
        self.original_legal_masks: List[np.ndarray] = []  # List of all legal ROI masks
        self.illegal_regions_mask: np.ndarray = None  # Combined single illegal ROI mask
        self.separated_roi_masks: List[np.ndarray] = []
    
    def add_roi(self, bbox: BoundingBox, is_legal: bool, roi_id: str = None):
        """Add a ROI region"""
        if roi_id is None:
            roi_id = f"roi_{len(self.rois)}"
        
        # Check for duplicate roi_id
        existing_ids = {roi.roi_id for roi in self.rois}
        if roi_id in existing_ids:
            raise ValueError(f"Duplicate ROI ID: '{roi_id}' already exists. Please use unique ROI IDs.")
        
        # Validate bounding box
        if (bbox.x < 0 or bbox.y < 0 or 
            bbox.x + bbox.width > self.image_width or 
            bbox.y + bbox.height > self.image_height):
            raise ValueError(f"ROI {roi_id} is out of image bounds")
        
        roi = ROI(bbox, is_legal, roi_id)
        self.rois.append(roi)
        return roi
    
    def _clip_bbox_to_bounds(self, x: int, y: int, w: int, h: int) -> tuple:
        """Clip bounding box to image bounds
        
        Returns:
            Tuple of (x_start, y_start, x_end, y_end) clipped to image bounds
        """
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(x + w, self.image_width)
        y_end = min(y + h, self.image_height)
        return x_start, y_start, x_end, y_end
    
    def get_legal_rois(self) -> List[ROI]:
        """Get all legal ROI regions"""
        return [roi for roi in self.rois if roi.is_legal]
    
    def select_random_roi(self) -> Optional[ROI]:
        """Select a random legal ROI"""
        legal_rois = self.get_legal_rois()
        if not legal_rois:
            return None
        return random.choice(legal_rois)
    
    def get_separated_roi_masks(self) -> List[np.ndarray]:
        """Get the separated ROI masks"""
        if not self.separated_roi_masks:
            raise ValueError("Separated ROI masks not found")
        return self.separated_roi_masks
    
    def select_random_separated_roi(self) -> Optional[np.ndarray]:
        """Select a random separated ROI mask"""
        separated_masks = self.get_separated_roi_masks()
        if not separated_masks:
            return None
        return random.choice(separated_masks)
    
    def separate_connected_regions(self, roi_img: np.ndarray, min_area: int = 10) -> List[np.ndarray]:
        """Separate connected ROIs from a binary ROI image"""
        # Use connected components to find separate ROIs
        num_labels, labels = cv2.connectedComponents(roi_img)
        
        separated_masks = []
        skipped_count = 0
        for label in range(1, num_labels):  # Skip background (label 0)
            # Create mask for this connected component
            mask = (labels == label).astype(np.uint8) * 255
            
            # Check area of this ROI
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                area = cv2.contourArea(contours[0])
            else:
                area = 0
            
            if area < min_area:
                log.info(f"Skipping small ROI {label} (area: {area}) < min_area {min_area}")
                skipped_count += 1
                continue
            
            separated_masks.append(mask)
        
        log.info(f"Separated {len(separated_masks)} ROIs from image (skipped {skipped_count} small ROIs)")
        return separated_masks
    
    def create_separated_roi_masks_from_json(self) -> List[np.ndarray]:
        """Load JSON ROIs into illegal_regions_mask and original_legal_masks attributes"""
        original_legal_masks = []
        illegal_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        
        for i, roi in enumerate(self.rois):
            x, y, w, h = roi.bbox.x, roi.bbox.y, roi.bbox.width, roi.bbox.height
            x_start, y_start, x_end, y_end = self._clip_bbox_to_bounds(x, y, w, h)
            
            if roi.is_legal:
                # Create individual legal ROI mask
                mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
                mask[y_start:y_end, x_start:x_end] = 255

                original_legal_masks.append(mask)
            else:
                # Add to illegal regions mask
                illegal_mask[y_start:y_end, x_start:x_end] = 255
        
        # Store the masks
        self.original_legal_masks = original_legal_masks
        self.illegal_regions_mask = illegal_mask
        
        log.info(f"Loaded {len(original_legal_masks)} legal ROIs and {np.sum(illegal_mask > 127)} illegal pixels from JSON definitions")
        return original_legal_masks
    
    def _bboxes_overlap(self, bbox1: BoundingBox, bbox2: BoundingBox) -> bool:
        """Check if two bounding boxes overlap"""
        return not (bbox1.x + bbox1.width <= bbox2.x or 
                   bbox2.x + bbox2.width <= bbox1.x or 
                   bbox1.y + bbox1.height <= bbox2.y or 
                   bbox2.y + bbox2.height <= bbox1.y)

