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

import typing

from PIL import Image


def get_bboxes(
    instance_masks: typing.Union[Image.Image, typing.List[Image.Image]],
    format="xywh",
) -> typing.List[typing.Union[typing.Tuple[int, int, int, int], None]]:
    if format not in ("xywh", "xyxy"):
        raise ValueError("format must be either 'xywh' or 'xyxy'")
    if isinstance(instance_masks, Image.Image):
        instance_masks = [instance_masks]

    bboxes = []
    for mask in instance_masks:
        bbox = mask.getbbox()
        if format == "xywh" and bbox is not None:
            # Convert bbox from (x1, y1, x2, y2) to (x, y, w, h)
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            bbox = (x1, y1, width, height)
        bboxes.append(bbox)
    return bboxes


def compute_cropped_bbox(
    bbox: typing.Tuple[int, int, int, int],
    image_height: int,
    image_width: int,
    crop_ratio: float = 2.0,
) -> typing.Tuple[int, int, int, int]:
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    long_side = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    crop_size = int(long_side * crop_ratio)
    crop_x1 = int(max(center_x - crop_size // 2, 0))
    crop_y1 = int(max(center_y - crop_size // 2, 0))
    crop_x2 = min(crop_x1 + crop_size, image_width)
    crop_y2 = min(crop_y1 + crop_size, image_height)
    return (crop_x1, crop_y1, crop_x2, crop_y2)
