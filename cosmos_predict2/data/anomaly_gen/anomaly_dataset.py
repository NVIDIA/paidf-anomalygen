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

"""
Run this command to interactively debug:
PYTHONPATH=. python cosmos_predict1/diffusion/training/datasets/dataset_gear.py

Adapted from:
https://github.com/bytedance/IRASim/blob/main/dataset/dataset_3D.py
"""

import json
import os
import random
from collections import Counter, OrderedDict
from pathlib import Path
from PIL import Image

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2.data.anomaly_gen.anomaly_dataset_utils import (
    Random_Crop,
    RandomRatioCrop,
    RandomRotation90,
)
from cosmos_predict2.inference.anomaly_gen.mask_augmentation import augment_binary_mask
from cosmos_predict2.utils.random import secure_random
from imaginaire.utils import log


ANOMALY_TEXT_TEMPLATE = \
    "A photo showing a close-up look of an anomalous {} with {} and spatial anomaly features {}. " \
    "Defective, Non-conforming, Substandard, Irregular, Damaged, Misaligned, Incomplete, Malformed, Contaminated, Under-processed, " \
    "Faulty, Blemished, Expired."


def _load_image_and_mask(image_file, mask_file):
    # Load single image / mask
    image = Image.open(image_file).convert("RGB")
    mask = Image.open(mask_file).convert("L")
    np_mask = np.array(mask)
    if np_mask.max() != 255:
        raise ValueError(f"Error: Mask {mask_file} is not binary!")
    if image.size != mask.size:
        raise ValueError(f"Error: Image filename {image_file} 's size with mask filename {mask_file}")
    return image, mask

def _preprocess(image, mask):
    """
    Normalize image value to -1~1, and convert mask into binary value
    """
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    mask = np.array(mask).astype(np.float32)

    image= (image / 127.5 - 1.0).astype(np.float32) # -1~1
    mask = mask / 255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    return image, mask

def _pil_image_to_tensor(image, mask):
    """
    Convert PIL Image to array and normalize the values to -1~1 for RGB and [0, 1] for mask.
    """
    image = image.convert("RGB")
    mask = mask.convert("L")
    image = np.array(image).astype(np.float32)
    mask = np.array(mask).astype(np.float32)

    image = (image / 127.5 - 1.0) # -1~1
    mask = mask / 255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    image = torch.from_numpy(image).permute(2, 0, 1)
    mask = torch.from_numpy(mask).permute(2, 0, 1)
    return image, mask


def _resize_pil_pair(image_pil, mask_pil, target_size):
    """BILINEAR for image, NEAREST for mask (preserve binary edges)."""
    h, w = target_size
    return (image_pil.resize((w, h), Image.Resampling.BILINEAR),
            mask_pil.resize((w, h), Image.Resampling.NEAREST))


def _crop_around_mask(image_pil, mask_pil, max_ratio, min_pad, max_rot_deg=0):
    """Crop a window that contains the defect with at least `half` pixels of
    context on the closer side, while preserving the source image's geometric
    center (margin equal on both sides of each axis). The preserved center
    keeps default tF.rotate pivot identical to the full-source pivot.
    Returns (None, None) if the mask has no defect pixels.
    """
    np_mask = np.array(mask_pil)
    ys, xs = np.where(np_mask >= 128)
    if xs.size == 0:
        return None, None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    defect_size = max(x2 - x1, y2 - y1, 1)
    rot_rad = float(np.deg2rad(max_rot_deg))
    sin_t = abs(np.sin(rot_rad))
    cos_t = abs(np.cos(rot_rad))
    rot_factor = cos_t + sin_t
    half = max(int(rot_factor * (max_ratio - 0.5) * defect_size), int(min_pad))
    W, H = image_pil.size
    # Rotation around source center also translates the defect.
    dx = abs(cx - W // 2)
    dy = abs(cy - H // 2)
    rot_dx = int(dx * (1 - cos_t) + dy * sin_t) + 1
    rot_dy = int(dx * sin_t + dy * (1 - cos_t)) + 1
    margin_x = max(0, min(cx, W - cx) - half - rot_dx)
    margin_y = max(0, min(cy, H - cy) - half - rot_dy)
    return (image_pil.crop((margin_x, margin_y, W - margin_x, H - margin_y)),
            mask_pil.crop((margin_x, margin_y, W - margin_x, H - margin_y)))

class Dataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        anomaly_types,
        image_size = [512, 512],
        num_frames=1,  # Should freeze to 1
        data_augprob = 0.5,
        aug_type = None,
        ratio_range = None,  # Only used when aug_type="random_ratio_crop"
        random_rotation90 = False,
        seed = 1
    ):
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            image_size (list): Target size [H,W] for video frames

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        assert num_frames == 1, f"[ERROR] Cosmos Anomaly Diffusion does not support T2V! num_frames = {num_frames}"
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.seed = seed; random.seed(self.seed)
        self.image_size = image_size
        self.data_augprob = data_augprob

        # Anomaly diffusion
        self.placeholder = '*'
        self.anomaly_types = anomaly_types

        self._max_ratio = ratio_range[1] if (ratio_range is not None and len(ratio_range) >= 2) else 8.0
        self._max_rot_deg = 20  # match RandomRotation in Compose()

        # Load dataset
        log.info(f"Now loading dataset:")
        self._load_and_process_image_path(self.dataset_dir)
        log.info(f"{len(self.data)} data loaded.")

        # Augmentation parameters
        if aug_type is None or aug_type == "random_crop":
            self.augment = Random_Crop(crop_size=self.image_size[0])
            log.info("Using random_crop augmentation")
        elif aug_type == "random_ratio_crop":
            self.augment = RandomRatioCrop(crop_size=self.image_size[0], ratio_range=ratio_range)
            log.info(f"Using random_ratio_crop augmentation (ratio_range={self.augment.ratio_range})")
        else:
            raise ValueError(f"Unknown aug_type: '{aug_type}'. Choose from ['random_crop', 'random_ratio_crop']")

        self._rotation90 = RandomRotation90(p=0.5) if random_rotation90 else None
        if self._rotation90 is not None:
            log.info("Using random_rotation90 augmentation (p=0.5)")

    def __str__(self):
        return f"{len(self.data)} samples from {self.dataset_dir}"

    def _load_and_process_image_path(self, image_path):
        """ Load and preprocess all images in memory
        """
        N = len(self.anomaly_types)
        self.data = []

        # Replicate anomaly to almost equal number of samples
        anomaly_data_count  = {
            f"{sample_name}+{anomaly_name}": len([file for file in os.listdir(f"{self.dataset_dir}/{sample_name}/anomaly_image/{anomaly_name}")])
            for sample_name, anomaly_name in self.anomaly_types
        }
        replicate_ratio = {
            anomaly_type: int(max(anomaly_data_count.values()) / anomaly_data_count[anomaly_type])
            for anomaly_type in anomaly_data_count
        }
        log.info(f"Replicate ratio: {replicate_ratio}")

        # Load dataset
        for idx, (sample_name, anomaly_name) in enumerate(self.anomaly_types):
            log.info(f"[{idx+1}/{N}]: Now loading {sample_name}-{anomaly_name}")
            anomaly_image_dir = os.path.join(self.dataset_dir, sample_name, "anomaly_image", anomaly_name)
            mask_image_dir = os.path.join(self.dataset_dir, sample_name, "mask", anomaly_name)

            # Remove Thumbs.db safely (avoid race condition in multi-rank setting)
            for dir_path in [anomaly_image_dir, mask_image_dir]:
                thumbs_path = os.path.join(dir_path, "Thumbs.db")
                try:
                    os.remove(thumbs_path)
                except FileNotFoundError:
                    pass

            anomaly_image_files = sorted(os.listdir(anomaly_image_dir))
            mask_image_files = []
            try:
                for anomaly_image in anomaly_image_files:
                    anomaly_image = Path(anomaly_image)
                    extension = Path(anomaly_image).suffix[1:]
                    mask_image_file = anomaly_image.name.replace(f".{extension}", f"_mask.{extension}")
                    mask_image_files.append(mask_image_file)
            except Exception as e:
                raise RuntimeError(f"Error: {e}")

            assert len(anomaly_image_files) == len(mask_image_files), f"[ERROR] Found {len(anomaly_image_files)} anomaly images, "\
                                                                                                                      f"which is not equal to {len(mask_image_files)} mask images" 

            # Sanity check - Check all anomaly images were paired with a mask image
            for img_file, mask_file in zip(anomaly_image_files, mask_image_files):
                img_filename = Path(img_file).stem
                mask_filename = Path(mask_file).stem
                assert mask_filename == f"{img_filename}_mask", f"Error: Image filename {img_file} is not aligned with mask filename {mask_file}"

            anomaly_image_files=[os.path.join(anomaly_image_dir, file_name) for file_name in anomaly_image_files]
            mask_image_files=[os.path.join(mask_image_dir, file_name) for file_name in mask_image_files]

            # Cache two PIL views per sample: resized (non-aug fast path)
            # and native-res mask-bbox window (aug path; preserves defect detail).
            for anomaly_image_file, mask_image_file in zip(anomaly_image_files, mask_image_files):
                anomaly_image, mask = _load_image_and_mask(anomaly_image_file, mask_image_file)

                image_resized, mask_resized = _resize_pil_pair(
                    anomaly_image, mask, self.image_size
                )
                image_cropped, mask_cropped = _crop_around_mask(
                    anomaly_image, mask,
                    max_ratio=self._max_ratio,
                    min_pad=self.image_size[0] // 8,
                    max_rot_deg=self._max_rot_deg,
                )
                if image_cropped is None:
                    # Empty mask: RandomRatioCrop returns input unchanged, so reuse resized view.
                    image_cropped, mask_cropped = image_resized, mask_resized

                # Caption
                text = ANOMALY_TEXT_TEMPLATE.format(sample_name, anomaly_name, self.placeholder)

                # Cache input
                for _ in range(replicate_ratio[f"{sample_name}+{anomaly_name}"]):
                    self.data.append(
                        (
                            image_resized, # PIL RGB at model input size
                            mask_resized, # PIL L at model input size
                            image_cropped, # PIL RGB at native res, mask-bbox window
                            mask_cropped, # PIL L at native res, mask-bbox window
                            text, # Caption for diffusion text condition
                            sample_name,
                            anomaly_name,
                            anomaly_image_file, # anomaly image's filename
                            mask_image_file # mask's filename
                        )
                    )
        return

    def _collate_fn(self, batch):
        """Non-aug path uses the pre-resized view (no resize at collate);
        aug path uses the pre-cropped native-res window."""
        batched_output = {}

        # Caption
        batched_output['caption'] = [data['caption'] for data in batch]

        # img_filename
        batched_output['img_filename'] = [data['img_filename'] for data in batch]

        # mask_filename
        batched_output['mask_filename'] = [data['mask_filename'] for data in batch]

        # name
        batched_output['name'] = [data['name'] for data in batch]

        # Image / Mask
        original_images = [data['image'] for data in batch]
        original_masks = [data['mask'] for data in batch]

        model_input_images = []
        model_input_masks = []

        for data in batch:
            # Data augmentation
            if secure_random() < self.data_augprob:
                image_tensor, mask_tensor = data['image_cropped'], data['mask_cropped']
                if self._rotation90 is not None:
                    image_tensor, mask_tensor = self._rotation90(image_tensor, mask_tensor)
                model_input_image, model_input_mask = self.augment(image_tensor, mask_tensor)
            else:
                image_tensor, mask_tensor = data['image'], data['mask']
                if self._rotation90 is not None:
                    image_tensor, mask_tensor = self._rotation90(image_tensor, mask_tensor)
                model_input_image = image_tensor
                model_input_mask = mask_tensor

            # resize if needed
            H, W = model_input_image.height, model_input_image.width
            if H != self.image_size[0] or W != self.image_size[1]:
                model_input_image = T.Resize(self.image_size, interpolation=Image.Resampling.BILINEAR)(model_input_image)
                model_input_mask = T.Resize(self.image_size, interpolation=Image.Resampling.NEAREST)(model_input_mask)

            model_input_image, model_input_mask = _pil_image_to_tensor(model_input_image, model_input_mask)
            model_input_images.append(model_input_image)
            model_input_masks.append(model_input_mask)

        ## Stack images 
        batched_output["images"] = torch.stack(model_input_images)
        # batched_output["images"] = rearrange(batched_output["images"], "b c h w -> b c 1 h w").contiguous() # (B. C, 1, H, W). We don't need this for predict2
        batched_output["mask"] = torch.stack(model_input_masks)   # (B, 1, H, W)
        batched_output['original_image'] = original_images
        batched_output['original_masks'] = original_masks

        # Redundant values for anomaly
        B = len(batched_output['caption'])
        batched_output['fps'] = torch.tensor([1] * B)
        H, W = self.image_size
        batched_output['image_size'] = torch.tensor([H, W, H, W])
        batched_output["num_frames"] = torch.tensor([self.sequence_length] * B)
        batched_output["padding_mask"] = torch.zeros(B, 1, H, W)
        batched_output["t5_text_mask"] = torch.ones(B, 512, dtype=torch.bfloat16)
        return batched_output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (
            image_resized,
            mask_resized,
            image_cropped,
            mask_cropped,
            caption,
            sample_name,
            anomaly_name,
            anomaly_image_file,
            mask_image_file,
        ) = self.data[index]

        # Form text caption
        text = ANOMALY_TEXT_TEMPLATE.format(sample_name, anomaly_name, self.placeholder)

        # Form single data
        data = dict()
        data['image'] = image_resized
        data['mask'] = mask_resized
        data['image_cropped'] = image_cropped
        data['mask_cropped'] = mask_cropped
        data['caption'] = text
        data['name'] = f"{sample_name}+{anomaly_name}"
        data['img_filename'] = anomaly_image_file
        data['mask_filename'] = mask_image_file
        return data


class AnomalyInpaintDataset(Dataset):
    def __init__(self, input_data_path):
        log.info(f"Loading generation settings from JSONL file: {input_data_path}")
        with open(input_data_path, "r") as f:
            self.input_data = [json.loads(line) for line in f if line.strip()]
        self._cache_max_items = 128
        self._image_cache = OrderedDict()
        self._mask_cache = OrderedDict()

        # Raise error if there are duplicated samples.
        input_data_string = [json.dumps(data, sort_keys=True) for data in self.input_data]
        seen = set()
        dupes = set()
        for string in input_data_string:
            if string in seen:
                dupes.add(string)
            seen.add(string)
        if len(seen) < len(self.input_data):
            dup_str = "\n".join(dupes)
            raise ValueError(
                f"Found duplicated samples in {input_data_path}. "
                f"Duplicated samples: \n{dup_str}"
            )

        # Sample index: honor a preset if present, else default to line position.
        for i in range(len(self.input_data)):
            self.input_data[i].setdefault("index", i)

        # Sort by the number of instances for more effcient iterative generation
        self.sort_by_instance_num()

    def _get_cached_pil_image(self, cache, cache_key):
        cached_image = cache.get(cache_key)
        if cached_image is None:
            return None
        cache.move_to_end(cache_key)
        return cached_image.copy()

    def _put_cached_pil_image(self, cache, cache_key, image):
        cache[cache_key] = image
        cache.move_to_end(cache_key)
        while len(cache) > self._cache_max_items:
            cache.popitem(last=False)

    def _load_cached_image(self, image_filename):
        cached_image = self._get_cached_pil_image(self._image_cache, image_filename)
        if cached_image is not None:
            return cached_image

        with Image.open(image_filename) as fp:
            image = fp.convert("RGB")

        self._put_cached_pil_image(self._image_cache, image_filename, image)
        return image.copy()

    def _load_cached_mask(self, mask_filename):
        cache_key = (mask_filename, "L")
        cached_mask = self._get_cached_pil_image(self._mask_cache, cache_key)
        if cached_mask is not None:
            return cached_mask

        with Image.open(mask_filename) as fp:
            mask = fp.convert("L")
        # Binarize to 0/255 (threshold 127) so every downstream consumer gets a
        # clean binary mask; warn if the source mask was not already binary.
        if not np.all(np.isin(np.array(mask), (0, 255))):
            log.warning(f"Mask {mask_filename} is not binary; binarizing at threshold 127.")
        mask = mask.point(lambda p: 255 if p > 127 else 0)

        self._put_cached_pil_image(self._mask_cache, cache_key, mask)
        return mask.copy()

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        data = dict(self.input_data[idx % len(self.input_data)])
        image = self._load_cached_image(data["image_filename"])
        mask = self._load_cached_mask(data["mask_filename"])
        data["loaded_image_array"] = np.array(image, copy=True)
        data["loaded_image_mode"] = image.mode
        data["loaded_mask_array"] = np.array(mask, copy=True)
        data["loaded_mask_mode"] = mask.mode
        return data

    def _collate_fn(self, batch):
        batched_output = {}
        for key in batch[0].keys():
            batched_output[key] = [data[key] for data in batch]
        return batched_output

    def sort_by_instance_num(self):
        num_instances = []
        for data in self.input_data:
            data.setdefault("shift_values", "0,0")
            data.setdefault("rotation_angle", 0)
            data.setdefault("morph_operation", "none")
            data.setdefault("iteration_generation_max_instance", 5)
            data.setdefault("poisson_blend", False)
            data.setdefault("crop_and_paste", True)
            data.setdefault("num_generated_images", 1)
            data.setdefault("guidance", 7)
            data.setdefault("index", 0)

            mask_filename = data["mask_filename"]
            shift_values = map(int, data["shift_values"].split(','))
            rotation_angle = data["rotation_angle"]
            morph_operation = data["morph_operation"]

            mask = self._load_cached_mask(mask_filename)
            aug_mask, aug_success = augment_binary_mask(mask, shift_values, rotation_angle, morph_operation)
            if aug_success:
                mask = aug_mask
            else:
                data.update({
                    "shift_values": "0,0",
                    "rotation_angle": 0,
                    "morph_operation": 'none',
                })

            # Handle crop-related fields
            if data.get("crop_and_paste") == False:
                data["crop_grid_X"] = data["crop_grid_Y"] = data["crop_ratio"] = "none"
            elif data.get("crop_ratio") is not None:
                data["crop_grid_X"] = data["crop_grid_Y"] = "none"
            elif data.get("crop_grid_X") is not None and data.get("crop_grid_Y") is not None:
                data["crop_ratio"] = "none"
            else:
                data["crop_ratio"] = 2.0
                data["crop_grid_X"] = data["crop_grid_Y"] = "none"

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.array(mask), connectivity=8)
            num_instances.append(num_labels - 1) # Exclude background

        counts = Counter(num_instances)
        log.info("Instance count distribution in augmented masks:")
        for k in sorted(counts.keys()):
            log.info(f"  {k} instance(s): {counts[k]} image(s)")

        # Sort input_data in ascending order based on the corresponding number of instances
        paired = list(zip(num_instances, self.input_data))
        paired.sort(key=lambda pair: pair[0])
        self.input_data = [data for _, data in paired]


class AnomalyInpaintValidationDataset(AnomalyInpaintDataset):
    """Validation dataset for anomaly inpainting."""

    def __init__(self, input_data_path):
        super().__init__(input_data_path)

        # Raise error if the `num_generated_images` is not 1.
        for data in self.input_data:
            if data["num_generated_images"] != 1:
                raise ValueError(
                    f"Found num_generated_images != 1 in {input_data_path}. "
                    f"num_generated_images is only allowed to be 1 in "
                    "validation."
                )
