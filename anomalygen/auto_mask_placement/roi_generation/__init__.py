# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Box/template/grayscale → mask ROI-generation pipeline (SAM2 + cradio backbones).

Public API:
- ``ROIGenerationModels``: lazy SAM2 / cradio model holder shared across pipeline stages.
- ``run_roi_pipeline`` / ``prepare_samples`` / ``load_one_sample`` / ``ensure_valid_boxes``: pipeline driver.
- ``ROIGenerationConfig`` / ``validate_sample_config``: per-sample configuration.

The individual post-process backends (``box_to_mask``, ``grayscale_to_mask``,
``template_box_to_masks``) and ``utils`` are internal helpers; import them from their modules.
"""

from anomalygen.auto_mask_placement.roi_generation.default_config import ROIGenerationConfig, validate_sample_config
from anomalygen.auto_mask_placement.roi_generation.model import ROIGenerationModels
from anomalygen.auto_mask_placement.roi_generation.pipeline import (
    ensure_valid_boxes,
    load_one_sample,
    prepare_samples,
    run_roi_pipeline,
)

__all__ = [
    "ROIGenerationModels",
    "run_roi_pipeline",
    "prepare_samples",
    "load_one_sample",
    "ensure_valid_boxes",
    "ROIGenerationConfig",
    "validate_sample_config",
]
