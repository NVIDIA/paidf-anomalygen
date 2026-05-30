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
from typing import Any, List, Optional, Union
from imaginaire.utils import log

@dataclass
class AnomalyInpaintCondition:
    """
    This class is used to store the conditions for anomaly inpainting.
    Each condition should be a single string or a list of strings. If they comes in list format, the length of the list should be the same.
    if num_generated_images is greater than 1, the conditions will be duplicated for multiple generations (same condition, different latents)
    """
    image_filename: Union[str, List[str]]
    mask_filename: Union[str, List[str]]
    anomaly_type: Union[str, List[str]]
    guidance: Union[float, List[float]] = 1.5
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
    shift_values: Union[tuple, List[tuple]] = None
    rotation_angle: Union[float, List[float]] = None
    morph_operation: Union[str, List[str]] = None
    # Iterative Generation
    iteration_generation_max_instance: int = 5
    # KPI
    PSNR: Union[float, List[float]] = None
    # Index
    index: Union[int, List[int]] = None
    # Optional preloaded data for DataLoader worker prefetch
    loaded_image_array: Optional[Union[Any, List[Any]]] = None
    loaded_image_mode: Union[None, str, List[Optional[str]]] = None
    loaded_mask_array: Optional[Union[Any, List[Any]]] = None
    loaded_mask_mode: Union[None, str, List[Optional[str]]] = None

    def __post_init__(self):
        """
        Converts non-list inputs to lists and performs sanity checks
        """
        # Convert non-list inputs to lists
        def to_list(value):
            return [value] if not isinstance(value, list) else value
        self.image_filename = to_list(self.image_filename)
        B = len(self.image_filename)

        self.mask_filename = to_list(self.mask_filename)
        self.anomaly_type = to_list(self.anomaly_type)
        self.crop_and_paste = to_list(self.crop_and_paste)
        self.crop_grid_X = to_list(self.crop_grid_X)
        self.crop_grid_Y = to_list(self.crop_grid_Y)
        self.crop_ratio = to_list(self.crop_ratio)
        self.poisson_blend = to_list(self.poisson_blend)
        self.PSNR = to_list(self.PSNR)
        self.index = to_list(self.index)

        def to_optional_list(value):
            if value is None:
                return [None] * B
            return [value] if not isinstance(value, list) else value

        self.loaded_image_array = to_optional_list(self.loaded_image_array)
        self.loaded_image_mode = to_optional_list(self.loaded_image_mode)
        self.loaded_mask_array = to_optional_list(self.loaded_mask_array)
        self.loaded_mask_mode = to_optional_list(self.loaded_mask_mode)

        def parse_shift_value(s):
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

        # Per-sample guidance: keep as list, broadcast single value to batch size
        self.guidance = to_list(self.guidance)
        if len(self.guidance) == 1 and B > 1:
            self.guidance = self.guidance * B
        if len(self.guidance) != B:
            raise ValueError(
                f"guidance length ({len(self.guidance)}) must be 1 or match "
                f"batch size ({B})"
            )

        # Convert shared config from list to int
        def to_scalar(value):
            return value[0] if isinstance(value, list) else value

        self.seed = to_scalar(self.seed)
        self.num_steps = to_scalar(self.num_steps)
        self.num_generated_images = to_scalar(self.num_generated_images)
        self.iteration_generation_max_instance = to_scalar(self.iteration_generation_max_instance)

        # Sanity checks for REQUIRED arguments
        if not self.image_filename:
            raise ValueError("image_filename is required and must not be empty.")
        if not self.mask_filename:
            raise ValueError("mask_filename is required and must not be empty.")
        if not self.anomaly_type:
            raise ValueError("anomaly_type is required and must not be empty.")
            
        # Sanity checks for OPTIONAL arguments
        # Check if all provided lists have the same length
        if len(self.mask_filename) != B or len(self.anomaly_type) != B:
            raise ValueError(f"The number of image filenames {len(self.image_filename)}, mask filenames {len(self.mask_filename)}, and anomaly types {len(self.anomaly_type)} must match {B}")
        if len(self.crop_and_paste) != B or len(self.crop_grid_X) != B or len(self.crop_grid_Y) != B or len(self.crop_ratio) != B or len(self.poisson_blend) != B:
            raise ValueError(f"The number of crop_and_paste {len(self.crop_and_paste)}, crop_grid_X {len(self.crop_grid_X)}, crop_grid_Y {len(self.crop_grid_Y)}, crop_ratio {len(self.crop_ratio)}, and poisson_blend {len(self.poisson_blend)} must match {B}")
        if len(self.shift_values) != B or len(self.rotation_angle) != B or len(self.morph_operation) != B:
            raise ValueError(f"The number of shift_values {len(self.shift_values)}, rotation_angle {len(self.rotation_angle)}, and morph_operation {len(self.morph_operation)} must match {B}")
        if len(self.loaded_image_array) != B or len(self.loaded_image_mode) != B or len(self.loaded_mask_array) != B or len(self.loaded_mask_mode) != B:
            raise ValueError(
                "The number of loaded_image_array, loaded_image_mode, loaded_mask_array, "
                f"and loaded_mask_mode must match {B}"
            )

        # Check if num_generated_images is 1 if number of provided image filenames is 1
        ## TODO: Check if this is needed
        # if self.num_generated_images != 1 and B == 1:
        #     raise ValueError("num_generated_images must be 1 if number of provided image filenames is 1")
        if self.num_generated_images > 1: # Duplicate input conditions for multiple generations (same condition, different latents)
            log.info(f"Generating {self.num_generated_images} images for current input image")
            self.image_filename = self.image_filename * self.num_generated_images
            self.mask_filename = self.mask_filename * self.num_generated_images
            self.anomaly_type = self.anomaly_type * self.num_generated_images
            self.crop_and_paste = self.crop_and_paste * self.num_generated_images
            self.crop_grid_X = self.crop_grid_X * self.num_generated_images
            self.crop_grid_Y = self.crop_grid_Y * self.num_generated_images
            self.crop_ratio = self.crop_ratio * self.num_generated_images
            self.poisson_blend = self.poisson_blend * self.num_generated_images
            self.shift_values = self.shift_values * self.num_generated_images
            self.rotation_angle = self.rotation_angle * self.num_generated_images
            self.morph_operation = self.morph_operation * self.num_generated_images
            self.PSNR = self.PSNR * self.num_generated_images
            self.index = self.index * self.num_generated_images
            self.guidance = self.guidance * self.num_generated_images
            self.loaded_image_array = self.loaded_image_array * self.num_generated_images
            self.loaded_image_mode = self.loaded_image_mode * self.num_generated_images
            self.loaded_mask_array = self.loaded_mask_array * self.num_generated_images
            self.loaded_mask_mode = self.loaded_mask_mode * self.num_generated_images
