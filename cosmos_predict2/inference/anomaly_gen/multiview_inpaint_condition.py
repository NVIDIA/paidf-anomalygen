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

from dataclasses import dataclass
from typing import List, Optional, Union
from imaginaire.utils import log

@dataclass
class MultiViewAnomalyInpaintCondition:
    """
    Multi-view version of AnomalyInpaintCondition.
    
    Key differences from single-view:
    - image_filenames: List[List[str]] - Each sample has multiple view images
    - mask_filename: List[List[str]] - Each sample has per-view masks (or shared)
    
    Each condition should be a list of lists for multi-view data, or a list of scalars.
    If num_generated_images is greater than 1, the conditions will be duplicated 
    for multiple generations (same condition, different latents)
    """
    image_filenames: Union[List[List[str]], List[str]]  # List of [list of view paths] OR list of single paths (for backward compat)
    mask_filename: Union[List[List[str]], List[str]]  # List of [list of mask paths per view] OR list of single mask path
    anomaly_type: Union[str, List[str]]
    guidance: float = 1.5
    seed: int = 1
    num_steps: int = 35
    num_generated_images: int = 1
    # Crop & Paste
    crop_and_paste: Union[bool, List[bool]] = True
    crop_grid_X: Union[int, List[int]] = None
    crop_grid_Y: Union[int, List[int]] = None
    crop_ratio: Union[float, None, List[Optional[float]]] = None
    poisson_blend: Union[bool, List[bool]] = None
    # Mask Augmentation
    shift_values: Union[tuple, str, List[Union[tuple, str]]] = None
    rotation_angle: Union[float, List[float]] = None
    morph_operation: Union[str, List[str]] = None
    # Iterative Generation
    iteration_generation_max_instance: int = 5
    # KPI
    PSNR: Union[float, List[float]] = None
    # Index
    index: Union[int, List[int]] = None

    def __post_init__(self):
        """
        Converts non-list inputs to lists and performs sanity checks
        """
        # Convert non-list inputs to lists
        def to_list(value):
            return [value] if not isinstance(value, list) else value
        
        # Handle image_filenames (List[List[str]])
        if not isinstance(self.image_filenames, list):
            raise ValueError("image_filenames must be a list")
        
        # Check if it's already List[List[str]] or needs conversion
        if len(self.image_filenames) > 0 and isinstance(self.image_filenames[0], list):
            # Already List[List[str]]
            pass
        else:
            # Single-view format, convert to multi-view format (for backward compat)
            # This shouldn't happen in practice for multi-view inference
            log.warning("image_filenames is not in multi-view format (List[List[str]]). Auto-converting...")
            self.image_filenames = [[path] for path in self.image_filenames]
        
        B = len(self.image_filenames)
        
        # Handle mask_filename (List[List[str]])
        self.mask_filename = to_list(self.mask_filename)
        if len(self.mask_filename) > 0 and not isinstance(self.mask_filename[0], list):
            # Single mask path per sample, convert to list of lists
            # This could be shared mask format from JSONL
            log.info("mask_filename is not in per-view format (List[List[str]]). Assuming shared masks...")
            # Keep as is for now, will handle in prepare function
        
        self.anomaly_type = to_list(self.anomaly_type)
        self.crop_and_paste = to_list(self.crop_and_paste)
        self.crop_grid_X = to_list(self.crop_grid_X)
        self.crop_grid_Y = to_list(self.crop_grid_Y)
        self.crop_ratio = to_list(self.crop_ratio)
        self.poisson_blend = to_list(self.poisson_blend)
        self.PSNR = to_list(self.PSNR)
        self.index = to_list(self.index)

        def parse_shift_value(s):
            if isinstance(s, tuple):
                return s
            v, h = map(int, s.split(","))
            return (v, h)
        
        try:
            if isinstance(self.shift_values, list):
                self.shift_values = [parse_shift_value(s) for s in self.shift_values]
            else:
                self.shift_values = [parse_shift_value(self.shift_values)]
        except:
            log.warning(f"Invalid shift_values: {self.shift_values}. Use (0, 0) for shifting.")
            self.shift_values = [(0, 0)] * B

        self.rotation_angle = to_list(self.rotation_angle)
        self.morph_operation = to_list(self.morph_operation)

        # Convert shared config from list to int
        def to_scalar(value):
            return value[0] if isinstance(value, list) else value

        self.guidance = to_scalar(self.guidance)
        self.seed = to_scalar(self.seed)
        self.num_steps = to_scalar(self.num_steps)
        self.num_generated_images = to_scalar(self.num_generated_images)
        self.iteration_generation_max_instance = to_scalar(self.iteration_generation_max_instance)

        # Sanity checks for REQUIRED arguments
        if not self.image_filenames:
            raise ValueError("image_filenames is required and must not be empty.")
        if not self.mask_filename:
            raise ValueError("mask_filename is required and must not be empty.")
        if not self.anomaly_type:
            raise ValueError("anomaly_type is required and must not be empty.")
            
        # Sanity checks for OPTIONAL arguments
        # Check if all provided lists have the same length
        if len(self.mask_filename) != B or len(self.anomaly_type) != B:
            raise ValueError(f"The number of image filenames {len(self.image_filenames)}, mask filenames {len(self.mask_filename)}, and anomaly types {len(self.anomaly_type)} must match {B}")
        if len(self.crop_and_paste) != B or len(self.crop_grid_X) != B or len(self.crop_grid_Y) != B or len(self.crop_ratio) != B or len(self.poisson_blend) != B:
            raise ValueError(f"The number of crop_and_paste {len(self.crop_and_paste)}, crop_grid_X {len(self.crop_grid_X)}, crop_grid_Y {len(self.crop_grid_Y)}, crop_ratio {len(self.crop_ratio)}, and poisson_blend {len(self.poisson_blend)} must match {B}")
        if len(self.shift_values) != B or len(self.rotation_angle) != B or len(self.morph_operation) != B:
            raise ValueError(f"The number of shift_values {len(self.shift_values)}, rotation_angle {len(self.rotation_angle)}, and morph_operation {len(self.morph_operation)} must match {B}")

        # Check if num_generated_images is 1 for multi-view (we don't support multiple generations yet)
        if self.num_generated_images > 1:
            log.warning("Multi-view does not fully support num_generated_images > 1. Setting to 1.")
            self.num_generated_images = 1

