# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounding-box extraction from instance masks (COCO ``xywh`` / ``xyxy``)."""

from __future__ import annotations

from typing import List, Tuple, Union

from PIL import Image


def get_bboxes(
    instance_masks: Union[Image.Image, List[Image.Image]],
    format: str = "xywh",
) -> List[Union[Tuple[int, int, int, int], None]]:
    """Tight bounding box per instance mask; ``None`` for an empty mask.

    ``format="xywh"`` returns COCO ``(x, y, w, h)``; ``"xyxy"`` returns ``(x1, y1, x2, y2)``.
    """
    if format not in ("xywh", "xyxy"):
        raise ValueError("format must be either 'xywh' or 'xyxy'")
    if isinstance(instance_masks, Image.Image):
        instance_masks = [instance_masks]

    bboxes = []
    for mask in instance_masks:
        bbox = mask.getbbox()
        if format == "xywh" and bbox is not None:
            # Convert bbox from (x1, y1, x2, y2) to (x, y, w, h).
            x1, y1, x2, y2 = bbox
            bbox = (x1, y1, x2 - x1, y2 - y1)
        bboxes.append(bbox)
    return bboxes
