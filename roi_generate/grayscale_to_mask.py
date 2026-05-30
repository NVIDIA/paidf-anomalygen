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

import os

import cv2
import numpy as np
from PIL import Image

from roi_generate.utils import to_rgb_uint8


def grayscale_binarize(image, config):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    mode = config.grayscale_to_mask.threshold_mode
    thres = config.grayscale_to_mask.threshold_value

    if mode == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "otsu_inv":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif mode == "custom":
        _, binary = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY)
    elif mode == "custom_inv":
        _, binary = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY_INV)
    else:
        raise ValueError(
            f"Unsupported threshold mode '{mode}' for Grayscale-to-Mask. "
            "Supported mode: otsu, otsu_inv, custom, custom_inv"
        )

    return binary


class GrayscaleToMaskPostProcess:
    """
    morphological refinement to generate the final binary mask.
    """

    def __init__(self, config):
        self.kernel_size = config.morphological_kernel
        self.op_name = config.morphological_operation
        self.result = {}

    def run(self, binary_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)
        if self.op_name:
            if self.op_name == "close":
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            elif self.op_name == "open":
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            elif self.op_name == "dilate":
                binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            elif self.op_name == "erode":
                binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
            else:
                raise ValueError(
                    f"Unsupported morphological operation '{self.op_name}'. "
                    "Supported ops: close, open, dilate, erode"
                )

        binary_mask = (binary_mask > 0.5).astype(np.uint8) * 255

        self.result.update({"binary_mask": binary_mask})

    def save_result(self, ctx):
        """
        Save:
          - <output_dir>/grayscale_to_mask/output/binary_mask.png
        """
        output_dir = os.path.join(ctx["input"]["output_dir"], "grayscale_to_mask")
        binary_mask = self.result["binary_mask"]

        output_dir = os.path.join(output_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "binary_mask.png")
        cv2.imwrite(output_path, binary_mask)

    def save_visualization(self, ctx):
        """
        Save:
        - <output_dir>/visualization/overlay_mask.png
        """
        base_dir = os.path.join(ctx["input"]["output_dir"], "grayscale_to_mask")
        out_dir = os.path.join(base_dir, "visualization")
        os.makedirs(out_dir, exist_ok=True)

        image_np = to_rgb_uint8(ctx["input"]["image"])
        mask = self.result["binary_mask"] > 0

        # Green overlay
        overlay = image_np.astype(np.float32)
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0], dtype=np.float32)

        Image.fromarray(overlay.astype(np.uint8)).save(os.path.join(out_dir, "overlay_mask.png"))
