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

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sam2.modeling import sam2_utils
from sam2.utils.transforms import SAM2Transforms
from torch.utils.data import Dataset


class InfoSAM2Dataset(Dataset):
    def __init__(
        self, image_paths, mask_paths, size=1024, training=False, image_transforms=None
    ):
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.size = int(size)
        self.training = bool(training)
        if image_transforms is None:
            image_transforms = SAM2Transforms(
                resolution=self.size,
                mask_threshold=0,
                max_hole_area=0,
                max_sprinkle_area=0,
            )
        self.image_transforms = image_transforms
        # Sanity check: images and masks are paired positionally, so the counts
        # must match (zip would otherwise silently drop the trailing extras) and
        # each pair's stems must correspond. Require an exact stem match or an
        # "<image_stem>_<suffix>" mask (e.g. image_0 -> image_0 / image_0_mask);
        # a plain substring test wrongly lets "1" match "10_mask".
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Number of images ({len(self.image_paths)}) and masks "
                f"({len(self.mask_paths)}) differ."
            )
        for image_path, mask_path in zip(self.image_paths, self.mask_paths):
            istem, mstem = image_path.stem, mask_path.stem
            if not (mstem == istem or mstem.startswith(istem + "_")):
                raise ValueError(
                    f"Image and mask names do not match: {image_path} vs {mask_path}"
                )

    def __len__(self):
        return len(self.image_paths)

    def _get_input_prompts(self, mask: Image.Image):
        binary_mask = torch.from_numpy(np.array(mask) > 127)
        binary_mask = binary_mask[None, None, ...]
        if self.training:
            noise = 0.1
            noise_bound = 20
            method = "uniform"
        else:
            noise = 0.0
            noise_bound = 0
            method = "center"
        box_points, box_labels = sam2_utils.sample_box_points(
            binary_mask, noise=noise, noise_bound=noise_bound
        )
        points, labels = sam2_utils.get_next_point(
            binary_mask, pred_masks=None, method=method
        )
        return box_points[0], box_labels[0], points[0], labels[0]

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")
        # Ensure binary mask.
        mask = Image.fromarray((np.array(mask) > 127).astype(np.uint8) * 255)

        orig_mask = np.expand_dims(np.array(mask) > 127, axis=0)
        orig_hw = np.array([image.height, image.width])

        box_coords, box_labels, point_coords, point_labels = self._get_input_prompts(
            mask
        )

        # Resize.
        image = self.image_transforms(image)
        mask = TF.resize(
            mask, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST
        )
        mask = np.expand_dims(np.array(mask) > 127, axis=0)
        result = [
            image,  # (3, H, W)
            mask,  # (1, H, W)
            box_coords,  # (1, 2)
            box_labels,  # (1,)
            point_coords,  # (1, 2)
            point_labels,  # (1,)
            orig_hw,  # (2,)
        ]
        if not self.training:
            result.append(orig_mask)
        return result


class InfoSAM2EvalDataset(InfoSAM2Dataset):
    def __init__(
        self,
        image_paths,
        mask_paths,
        size=1024,
        image_transforms=None,
        dilate_sizes=[7, 9, 11, 13],
    ):
        super().__init__(image_paths, mask_paths, size, False, image_transforms)
        self.dilate_sizes = dilate_sizes

        # Prepare the kernel sizes for each mask.
        self.kernel_sizes = [
            np.random.choice(self.dilate_sizes) for _ in range(len(self))
        ]

    def _dilate_mask(self, mask: Image.Image, kernel_size: int):
        kernel = np.ones((kernel_size, kernel_size))
        dilated_mask_array = cv2.dilate(np.array(mask), kernel, iterations=1)
        dilated_mask = Image.fromarray(dilated_mask_array)
        return dilated_mask, int(kernel_size)

    def __getitem__(self, index):
        mask_path: pathlib.Path = self.mask_paths[index]
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # Ensure binary mask.
        mask = Image.fromarray((np.array(mask) > 127).astype(np.uint8) * 255)

        # Resize.
        image = TF.resize(image, [self.size, self.size])
        mask = TF.resize(
            mask, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST
        )
        dilated_mask, kernel_size = self._dilate_mask(mask, self.kernel_sizes[index])
        return image, mask, dilated_mask, kernel_size, mask_path.name
