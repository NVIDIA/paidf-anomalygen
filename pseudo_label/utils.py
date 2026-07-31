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

import pathlib
import typing

import numpy as np
from PIL import Image, ImageDraw


def get_image_paths(
    image_dir: typing.Union[str, typing.List[str]],
) -> typing.List[pathlib.Path]:
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    if isinstance(image_dir, (str, pathlib.Path)):
        image_dir = [image_dir]
    if not isinstance(image_dir, (tuple, list)):
        raise ValueError("image_dir should be a string or a list of strings.")
    image_dir = sorted(image_dir)
    image_patshs = []
    for dir in image_dir:
        image_patshs.extend(
            [
                p
                for p in pathlib.Path(dir).rglob("**/*")
                if p.suffix.lower() in IMG_EXTENSIONS
            ]
        )
    return image_patshs


def visualize(
    image: Image.Image,
    classes,
    bboxes,
    instance_masks: typing.List[Image.Image],
    mask_alpha=0.6
):
    def textsize(text, font):
        im = Image.new(mode="P", size=(0, 0))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height

    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    for class_name, bbox, mask in zip(classes, bboxes, instance_masks):
        # Mask.
        mask_array = np.array(mask)
        color = np.array([30 / 255, 144 / 255, 255 / 255, mask_alpha])
        mask_array = np.expand_dims(mask_array, axis=-1) * color.reshape(1, 1, -1)
        mask = Image.fromarray((mask_array).astype(np.uint8), mode="RGBA")
        alpha_channel = mask.split()[-1]
        image.paste(mask, mask=alpha_channel)
        # Bbox (also fixes the label anchor below).
        if bbox is not None:
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        else:
            # No bbox for this instance: anchor the label at the top-left instead
            # of crashing (x/y are undefined on the first iteration) or silently
            # reusing the previous instance's coordinates.
            x, y = 0, 0
        # Class name.
        if class_name is not None:
            _, text_height = textsize(class_name, font=None)
            draw.text((x, y - text_height), class_name, fill="white")
    return image
