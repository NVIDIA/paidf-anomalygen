# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-prompted ROI generation: Qwen-VL box detection (``text2box``) + SAM2 mask helpers.

Public API:
- ``Text2BoxDetector`` / ``run_text2box``: detect defect-location boxes from a text prompt.
- ``resize_for_sam`` / ``postprocess_sam_mask`` / ``pick_best_mask`` / ``SAM_IMAGE_SIZE``: SAM2 pre/post-processing.
"""

from anomalygen.auto_mask_placement.text2roi.sam_utils import (
    SAM_IMAGE_SIZE,
    pick_best_mask,
    postprocess_sam_mask,
    resize_for_sam,
)
from anomalygen.auto_mask_placement.text2roi.text2box import Text2BoxDetector, run_text2box

__all__ = [
    "Text2BoxDetector",
    "run_text2box",
    "resize_for_sam",
    "postprocess_sam_mask",
    "pick_best_mask",
    "SAM_IMAGE_SIZE",
]
