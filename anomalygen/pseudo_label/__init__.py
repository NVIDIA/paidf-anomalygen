# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pseudo-labeling for generated anomalies: COCO format conversion + Cosmos3-reasoner captioning.

Public API:
- ``get_bboxes``: tight bounding boxes (``xywh``/``xyxy``) from instance masks.
- ``binary_mask_to_rle`` / ``coco_encode_rle`` / ``coco_decode_rle``: COCO RLE (de)serialization.
- ``get_image_paths`` / ``visualize``: image discovery + mask/bbox overlay.
- ``Captioner`` / ``format_response`` / ``DEFAULT_CAPTION_PROMPT_PATH``: anomaly captioning.

Instance splitting reuses :func:`anomalygen.inference.iterative.split_mask_into_instances`.
"""

from anomalygen.pseudo_label.bbox import get_bboxes
from anomalygen.pseudo_label.caption import (
    DEFAULT_CAPTION_PROMPT_PATH,
    Captioner,
    format_response,
)
from anomalygen.pseudo_label.mask import (
    binary_mask_to_rle,
    coco_decode_rle,
    coco_encode_rle,
)
from anomalygen.pseudo_label.utils import get_image_paths, visualize

__all__ = [
    "get_bboxes",
    "binary_mask_to_rle",
    "coco_encode_rle",
    "coco_decode_rle",
    "get_image_paths",
    "visualize",
    "Captioner",
    "format_response",
    "DEFAULT_CAPTION_PROMPT_PATH",
]
