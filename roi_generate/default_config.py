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

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TemplateBoxToMasksConfig:
    enabled: bool = True
    save_cache: bool = True
    resume_cache: bool = True
    image_resize: int = 1024
    # Proposal
    proposal_similarity_tol: float = 0.5
    max_proposal: int = 300
    nms_iou_threshold: float = 0.9
    # Template
    max_template: int = 3
    crop_resize: int = 224
    refine_template: bool = True
    rotation_degrees: List[int] = field(default_factory=lambda: [0])
    allow_flip: List[str] = field(default_factory=lambda: [])
    # Box Filter
    box_enabled: bool = True
    size_tol: float = 0.3
    aspect_tol: float = 0.3
    # Mask Filter
    mask_enabled: bool = True
    component_count_tol: int = 0
    chamfer_tol: float = 0.025
    # HOG Filter
    hog_enabled: bool = True
    hog_similarity_tol: float = 0.8
    orientations: int = 9
    pixels_per_cell: List[int] = field(default_factory=lambda: [8, 8])
    cells_per_block: List[int] = field(default_factory=lambda: [2, 2])
    # Color Histogram Filter
    color_hist_enabled: bool = True
    color_bins: int = 8
    lightness_bins: int = 8
    color_tol: float = 0.5
    lightness_tol: float = 0.5


@dataclass
class BoxToMaskConfig:
    enabled: bool = True
    image_resize: int = 1024


@dataclass
class GrayscaleToMaskConfig:
    enabled: bool = True
    threshold_mode: str = "otsu"
    threshold_value: int = 128


@dataclass
class DefaultConfig:
    save_visualization: bool = False
    morphological_operation: Optional[str] = None
    morphological_kernel: List[int] = field(default_factory=lambda: [3, 3])
    box_to_mask: BoxToMaskConfig = field(default_factory=BoxToMaskConfig)
    template_box_to_masks: TemplateBoxToMasksConfig = field(default_factory=TemplateBoxToMasksConfig)
    grayscale_to_mask: GrayscaleToMaskConfig = field(default_factory=GrayscaleToMaskConfig)


def validate_sample_config(cfg):
    """
    Validate ROI Generate config according to constraints.
    Raises ValueError on invalid values.
    Note: The type check is done by OmegaConf.
    """

    op = cfg.morphological_operation
    supported = (None, "dilate", "erode", "close", "open")
    if op not in supported:
        raise ValueError(
            f"Invalid post_process.morphological_operation: {op}. " f"Supported operations are: {supported}."
        )

    kernel = cfg.morphological_kernel
    if len(kernel) != 2 or not all(v > 1 for v in kernel):
        raise ValueError(f"morphological_kernel must be a 2-element list of positive integers >1, got {kernel}")

    # enforce odd kernel sizes
    if any(v % 2 == 0 for v in kernel):
        raise ValueError(f"morphological_kernel must use odd sizes (3,5,7,...), got {kernel}")

    if sum([cfg.box_to_mask.enabled, cfg.template_box_to_masks.enabled, cfg.grayscale_to_mask.enabled]) == 0:
        raise ValueError("At least one mode must be enabled")

    # Grayscale-Binary-Mask
    gcfg = cfg.grayscale_to_mask

    # Validate threshold_mode
    valid_modes = ("otsu", "otsu_inv", "custom", "custom_inv")
    if gcfg.threshold_mode not in valid_modes:
        raise ValueError(
            f"grayscale_to_mask.threshold_mode must be one of {valid_modes}, " f"got '{gcfg.threshold_mode}'"
        )

    # Validate threshold_value when in custom mode
    if gcfg.threshold_mode in ("custom", "custom_inv"):
        if not (0 <= gcfg.threshold_value <= 255):
            raise ValueError(f"grayscale_to_mask.threshold_value must be in [0, 255], got {gcfg.threshold_value}")

    # Box-to-Mask
    if cfg.box_to_mask.image_resize <= 0:
        raise ValueError("box_to_mask.image_resize must be a positive integer")

    # Template-Box-to-Masks
    tcfg = cfg.template_box_to_masks

    if tcfg.image_resize <= 0:
        raise ValueError("template_box_to_masks.image_resize must be a positive integer")

    # Proposal section
    if not 0.0 <= tcfg.proposal_similarity_tol <= 1.0:
        raise ValueError(f"proposal.similarity_tol must be in [0,1], got {tcfg.proposal_similarity_tol}")

    if tcfg.max_proposal <= 0:
        raise ValueError("proposal.max_proposal must be a positive integer")

    if not 0.0 <= tcfg.nms_iou_threshold <= 1.0:
        raise ValueError(f"proposal.nms_iou_threshold must be in [0,1], got {tcfg.nms_iou_threshold}")

    # Template config
    if tcfg.max_template <= 0:
        raise ValueError("template.max_template must be a positive integer")

    if tcfg.crop_resize <= 0:
        raise ValueError("template.crop_resize must be a positive integer")

    if not all(v in ("horizontal", "vertical") for v in tcfg.allow_flip):
        raise ValueError(
            f"template.allow_flip must be a list containing 'horizontal' and/or 'vertical', got {tcfg.allow_flip}"
        )

    # Box filter
    if not 0.0 <= tcfg.size_tol <= 1.0:
        raise ValueError(f"filters.box.size_tol must be in [0,1], got {tcfg.size_tol}")

    if not 0.0 <= tcfg.aspect_tol <= 1.0:
        raise ValueError(f"filters.box.aspect_tol must be in [0,1], got {tcfg.aspect_tol}")

    # Mask filter
    if tcfg.component_count_tol < 0:
        raise ValueError("filters.mask.component_count_tol must be a non-negative integer")

    if not 0.0 <= tcfg.chamfer_tol <= 1.0:
        raise ValueError(f"filters.mask.chamfer_tol must be in [0,1], got {tcfg.chamfer_tol}")

    # HOG filter
    if not 0.0 <= tcfg.hog_similarity_tol <= 1.0:
        raise ValueError(f"filters.hog.similarity_tol must be in [0,1], got {tcfg.hog_similarity_tol}")

    if tcfg.orientations <= 0:
        raise ValueError("filters.hog.orientations must be a positive integer")

    if not (len(tcfg.pixels_per_cell) == 2 and all(v > 0 for v in tcfg.pixels_per_cell)):
        raise ValueError("filters.hog.pixels_per_cell must be a 2-element list of positive integers")

    if not (len(tcfg.cells_per_block) == 2 and all(v > 0 for v in tcfg.cells_per_block)):
        raise ValueError("filters.hog.cells_per_block must be a 2-element list of positive integers")

    # Color Histogram filter
    if tcfg.color_bins <= 0:
        raise ValueError("filters.color_hist.color_bins must be a positive integer")

    if tcfg.lightness_bins <= 0:
        raise ValueError("filters.color_hist.lightness_bins must be a positive integer")

    if not 0.0 <= tcfg.color_tol <= 1.0:
        raise ValueError(f"filters.color_hist.color_tol must be in [0,1], got {tcfg.color_tol}")

    if not 0.0 <= tcfg.lightness_tol <= 1.0:
        raise ValueError(f"filters.color_hist.lightness_tol must be in [0,1], got {tcfg.lightness_tol}")

    return True
