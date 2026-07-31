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
Multi-view anomaly dataset for training cosmos anomaly generation with multiple views.

This dataset groups multiple views (e.g., different lighting conditions) of the same
sample together as "frames" in the temporal dimension, enabling multi-view training.
"""

import os
import re
import json
import random
from collections import defaultdict, Counter, OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2.data.anomaly_gen.anomaly_dataset_utils import MultiViewRandomCrop, MultiViewRandomRatioCrop
from cosmos_predict2.inference.anomaly_gen.mask_augmentation import augment_binary_mask
from cosmos_predict2.inference.anomaly_gen.inference_anomaly_diffusion_utils import count_mask_instances
from cosmos_predict2.utils.random import secure_random
from imaginaire.utils import log


ANOMALY_TEXT_TEMPLATE = \
    "A photo showing a close-up look of an anomalous {} with {} and spatial anomaly features {}. " \
    "Defective, Non-conforming, Substandard, Irregular, Damaged, Misaligned, Incomplete, Malformed, Contaminated, Under-processed, " \
    "Faulty, Blemished, Expired."


def _preprocess(image: Image.Image, mask: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize image value to -1~1, and convert mask into binary value.
    """
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    mask = np.array(mask).astype(np.float32)

    image = (image / 127.5 - 1.0).astype(np.float32)  # -1~1
    mask = mask / 255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    return image, mask


def _load_image(image_file: str) -> Tuple[Image.Image, np.ndarray]:
    """Load single image and return both PIL Image and normalized numpy array."""
    image = Image.open(image_file)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    np_image = np.array(image).astype(np.uint8)
    norm_image = (np_image / 127.5 - 1.0).astype(np.float32)
    return image, norm_image


def _load_mask(mask_file: str) -> Tuple[Image.Image, np.ndarray]:
    """Load single mask and return both PIL Image and binary numpy array."""
    mask = Image.open(mask_file).convert("L")
    np_mask = np.array(mask)
    
    # Check if binary: all values should be 0 or 255
    unique_values = np.unique(np_mask)
    if not np.all(np.isin(unique_values, [0, 255])):
        raise ValueError(f"Error: Mask {mask_file} is not binary! Unique values: {unique_values}")
    
    mask_normalized = np_mask.astype(np.float32) / 255.0
    mask_normalized[mask_normalized < 0.5] = 0
    mask_normalized[mask_normalized >= 0.5] = 1
    return mask, mask_normalized


class MultiViewDataset(Dataset):
    """
    Dataset class for loading multi-view anomaly data.
    
    This dataset groups multiple views of the same sample (e.g., different lighting 
    conditions) together as frames in the temporal dimension (T).
    
    Expected data structure:
    dataset_dir/
    ├── anomaly_image/
    │   └── {anomaly_type}/
    │       ├── {base_name}_{view1}.jpg
    │       ├── {base_name}_{view2}.jpg
    │       └── ...
    └── mask/
        └── {anomaly_type}/
            ├── {base_name}_mask.png  (shared mask, used if per-view masks not found)
            OR
            ├── {base_name}_{view1}_mask.png  (per-view masks, automatically detected)
            ├── {base_name}_{view2}_mask.png
            └── ...
    
    Args:
        dataset_dir: Base path to the dataset directory
        anomaly_types: List of (sample_name, anomaly_type) tuples
        view_types: List of view type suffixes (e.g., ['LowAngleLight', 'SolderLight', ...])
        image_size: Target size [H, W] for frames
        num_frames: Number of views/frames (should match len(view_types))
        data_augprob: Probability of applying data augmentation
        seed: Random seed
    
    Returns dict with:
        - video: RGB frames tensor [T, C, H, W] where T is number of views
        - mask: Binary mask tensor [T, H, W] (always per-view, shared masks are replicated)
    """

    def __init__(
        self,
        dataset_dir: str,
        anomaly_types: List[Tuple[str, str]],
        view_types: List[str],
        image_size: List[int] = [512, 512],
        num_frames: int = 4,
        data_augprob: float = 0.5,
        aug_type: Optional[str] = None,
        ratio_range: Optional[tuple] = None,  # Only used when aug_type="random_ratio_crop"
        seed: int = 1,
    ):
        super().__init__()
        
        self.dataset_dir = dataset_dir
        self.view_types = view_types
        self.num_frames = num_frames
        self.image_size = image_size
        self.data_augprob = data_augprob
        self.seed = seed
        random.seed(self.seed)

        # Anomaly diffusion
        self.placeholder = '*'
        self.anomaly_types = anomaly_types

        # Validate view_types matches num_frames
        assert len(view_types) == num_frames, \
            f"[ERROR] Number of view_types ({len(view_types)}) must match num_frames ({num_frames})"

        # Load dataset
        log.info(f"Now loading multi-view dataset:")
        self._load_and_process_multiview_data()
        log.info(f"{len(self.data)} multi-view samples loaded.")
        log.info(f"  Per-view masks: {self.per_view_mask_count} samples")
        log.info(f"  Shared masks (replicated): {self.shared_mask_count} samples")

        # Augmentation parameters
        if aug_type is None or aug_type == "random_crop":
            self.augment = MultiViewRandomCrop(crop_size=self.image_size[0])
            log.info("Using random_crop augmentation")
        elif aug_type == "random_ratio_crop":
            self.augment = MultiViewRandomRatioCrop(crop_size=self.image_size[0], ratio_range=ratio_range)
            log.info(f"Using random_ratio_crop augmentation (ratio_range={self.augment.ratio_range})")
        else:
            raise ValueError(f"Unknown aug_type: '{aug_type}'. Choose from ['random_crop', 'random_ratio_crop']")

    def __str__(self):
        return f"{len(self.data)} multi-view samples from {self.dataset_dir}"

    def _find_base_name(self, filename: str) -> Optional[str]:
        """
        Extract base name from filename by removing view type suffix.
        
        Example: 'sample_001_LowAngleLight.jpg' -> 'sample_001'
        """
        stem = Path(filename).stem
        for view_type in self.view_types:
            if stem.endswith(f"_{view_type}"):
                return stem[:-len(f"_{view_type}")]
        return None

    def _load_and_process_multiview_data(self):
        """Load and preprocess multi-view image data."""
        self.data = []
        N = len(self.anomaly_types)

        # Statistics for logging
        per_view_mask_count = 0
        shared_mask_count = 0

        for idx, (sample_name, anomaly_name) in enumerate(self.anomaly_types):
            log.info(f"[{idx+1}/{N}]: Now loading {sample_name}-{anomaly_name}")
            
            # Build paths with sample_name subdirectory (consistent with single-view dataset structure)
            anomaly_image_dir = os.path.join(self.dataset_dir, sample_name, "anomaly_image", anomaly_name)
            mask_image_dir = os.path.join(self.dataset_dir, sample_name, "mask", anomaly_name)

            # Remove Thumbs.db safely
            for dir_path in [anomaly_image_dir, mask_image_dir]:
                thumbs_path = os.path.join(dir_path, "Thumbs.db")
                try:
                    os.remove(thumbs_path)
                except FileNotFoundError:
                    pass

            # Group images by base name
            all_files = sorted(os.listdir(anomaly_image_dir))
            base_name_to_views: Dict[str, Dict[str, str]] = defaultdict(dict)
            
            for filename in all_files:
                base_name = self._find_base_name(filename)
                if base_name is None:
                    log.warning(f"Could not extract base name from {filename}, skipping...")
                    continue
                    
                # Determine which view type this file is
                for view_type in self.view_types:
                    if filename.endswith(f"_{view_type}.jpg") or filename.endswith(f"_{view_type}.png"):
                        base_name_to_views[base_name][view_type] = os.path.join(anomaly_image_dir, filename)
                        break

            # Process each complete multi-view sample
            for base_name, view_files in base_name_to_views.items():
                # Check if all views are present
                if len(view_files) != len(self.view_types):
                    missing = set(self.view_types) - set(view_files.keys())
                    log.warning(f"Sample {base_name} is missing views: {missing}, skipping...")
                    continue

                # === Auto-detect mask type: per-view or shared ===
                # Try to find per-view masks first
                per_view_mask_files = {}
                has_all_per_view_masks = True
                
                for view_type in self.view_types:
                    per_view_mask = os.path.join(mask_image_dir, f"{base_name}_{view_type}_mask.png")
                    if not os.path.exists(per_view_mask):
                        # Try other extensions
                        found = False
                        for ext in ['.jpg', '.jpeg', '.PNG', '.JPG']:
                            alt_mask = per_view_mask.replace('.png', ext)
                            if os.path.exists(alt_mask):
                                per_view_mask_files[view_type] = alt_mask
                                found = True
                                break
                        if not found:
                            has_all_per_view_masks = False
                            break
                    else:
                        per_view_mask_files[view_type] = per_view_mask
                
                # Load masks based on detected type and unify to [T, H, W]
                mask_files_ordered = []
                if has_all_per_view_masks:
                    # Load per-view masks: [T, H, W]
                    mask_images = []
                    for view_type in self.view_types:
                        _, norm_mask = _load_mask(per_view_mask_files[view_type])
                        mask_images.append(norm_mask)
                        mask_files_ordered.append(per_view_mask_files[view_type])
                    
                    norm_masks = np.stack(mask_images, axis=0)  # [T, H, W]
                    per_view_mask_count += 1
                else:
                    # Load shared mask and replicate for all views: [T, H, W]
                    mask_file = os.path.join(mask_image_dir, f"{base_name}_mask.png")
                    if not os.path.exists(mask_file):
                        # Try other extensions
                        for ext in ['.jpg', '.jpeg', '.PNG', '.JPG']:
                            alt_mask = mask_file.replace('.png', ext)
                            if os.path.exists(alt_mask):
                                mask_file = alt_mask
                                break
                        else:
                            raise ValueError(f"Mask not found for {base_name}")
                    
                    _, norm_mask = _load_mask(mask_file)
                    # Replicate shared mask for all views
                    norm_masks = np.stack([norm_mask] * len(self.view_types), axis=0)  # [T, H, W]
                    mask_files_ordered = [mask_file]  # Single shared mask file
                    shared_mask_count += 1

                # Load all views in order
                view_images = []
                view_files_ordered = []
                for view_type in self.view_types:
                    filepath = view_files[view_type]
                    _, norm_image = _load_image(filepath)
                    view_images.append(norm_image)
                    view_files_ordered.append(filepath)

                # Stack views as frames [T, H, W, C]
                stacked_views = np.stack(view_images, axis=0)  # [T, H, W, C]

                # Caption
                text = ANOMALY_TEXT_TEMPLATE.format(sample_name, anomaly_name, self.placeholder)

                self.data.append({
                    'views': stacked_views,  # [T, H, W, C], normalized to -1~1
                    'masks': norm_masks,  # [T, H, W] (always per-view format)
                    'caption': text,
                    'sample_name': sample_name,
                    'anomaly_name': anomaly_name,
                    'view_files': view_files_ordered,
                    'mask_files': mask_files_ordered,
                })
        
        # Store statistics as instance variables
        self.per_view_mask_count = per_view_mask_count
        self.shared_mask_count = shared_mask_count

    def _collate_fn(self, batch: List[dict]) -> dict:
        """
        Collate function to handle multi-view batches.
        Output format compatible with Video2World pipeline.
        All masks are in [T, H, W] format.
        """
        batched_output = {}

        # Caption
        batched_output['caption'] = [data['caption'] for data in batch]

        # File paths for reference
        batched_output['view_files'] = [data['view_files'] for data in batch]
        batched_output['mask_filename'] = [data['mask_files'] for data in batch]

        # Name
        batched_output['name'] = [data['name'] for data in batch]

        # Process videos (multi-view frames) and masks
        model_input_videos = []
        model_input_masks = []

        resize_transform = T.Resize(self.image_size, interpolation=Image.Resampling.BILINEAR)

        for data in batch:
            views = data['views']  # [T, H, W, C]
            masks = data['masks']  # [T, H, W] (always per-view format)

            # Convert views to tensor [T, C, H, W]
            views_tensor = torch.from_numpy(views).permute(0, 3, 1, 2).float()  # [T, C, H, W]
            
            # Convert masks to tensor [T, H, W]
            masks_tensor = torch.from_numpy(masks).float()  # [T, H, W]

            # Data augmentation - apply multi-view random crop (flip + rotation + crop)
            # All views get the same transformation parameters to maintain consistency
            if secure_random() < self.data_augprob:
                views_tensor, masks_tensor = self.augment(views_tensor, masks_tensor)

            # Resize if needed
            T_frames, C, H, W = views_tensor.shape
            target_H, target_W = self.image_size
            if H != target_H or W != target_W:
                # Resize all views at once
                views_tensor = torch.stack([resize_transform(views_tensor[t]) for t in range(T_frames)])
                
                # Resize masks (each view's mask)
                masks_tensor = torch.stack([resize_transform(masks_tensor[t:t+1])[0] for t in range(T_frames)])
                
                # Re-binarize
                masks_tensor[masks_tensor < 0.5] = 0
                masks_tensor[masks_tensor >= 0.5] = 1

            model_input_videos.append(views_tensor)  # [T, C, H, W]
            model_input_masks.append(masks_tensor)   # [T, H, W]

        # Stack batches
        # video: [B, C, T, H, W] - rearrange from [B, T, C, H, W]
        batched_videos = torch.stack(model_input_videos)  # [B, T, C, H, W]
        batched_output["video"] = batched_videos.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        
        # mask: [B, T, H, W] (always per-view format)
        batched_output["mask"] = torch.stack(model_input_masks)  # [B, T, H, W]

        # Redundant values for video processing
        B = len(batched_output['caption'])
        num_views = self.num_frames
        H, W = self.image_size
        batched_output['fps'] = torch.tensor([1] * B)
        batched_output['image_size'] = torch.tensor([H, W, H, W])
        batched_output["num_frames"] = torch.tensor([num_views] * B)
        batched_output["padding_mask"] = torch.zeros(B, 1, H, W)
        batched_output["t5_text_mask"] = torch.ones(B, 512, dtype=torch.bfloat16)

        return batched_output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        item = self.data[index]
        
        return {
            'views': item['views'],
            'masks': item['masks'],
            'caption': item['caption'],
            'name': f"{item['sample_name']}+{item['anomaly_name']}",
            'view_files': item['view_files'],
            'mask_files': item['mask_files'],
        }


class MultiViewAnomalyInpaintDataset(Dataset):
    """Base dataset for multi-view anomaly inpainting (similar to AnomalyInpaintDataset)."""
    
    def __init__(self, input_data_path):
        log.info(f"Loading multi-view generation settings from JSONL file: {input_data_path}")
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

        # Add sample index
        for i in range(len(self.input_data)):
            self.input_data[i]["index"] = i

        # Sort by the number of instances for more efficient iterative generation
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
        # Preload per-view images and (binarized) masks, mirroring the
        # single-view AnomalyInpaintDataset.__getitem__ so every downstream
        # consumer — including inference — receives clean binary masks.
        images = [self._load_cached_image(f) for f in data["image_filenames"]]
        masks = [self._load_cached_mask(f) for f in data["mask_filename"]]
        data["loaded_image_array"] = [np.array(im, copy=True) for im in images]
        data["loaded_image_mode"] = [im.mode for im in images]
        data["loaded_mask_array"] = [np.array(m, copy=True) for m in masks]
        data["loaded_mask_mode"] = [m.mode for m in masks]
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

            # Handle both shared mask (string) and per-view masks (list) formats
            mask_filenames = data["mask_filename"]
            
            if isinstance(mask_filenames, str):
                # Shared mask format - convert to list by replicating
                # Number of views can be inferred from image_filenames
                num_views = len(data.get("image_filenames", []))
                if num_views == 0:
                    raise ValueError(f"Cannot infer num_views from image_filenames for shared mask")
                data["mask_filename"] = [mask_filenames] * num_views
                mask_filenames_list = [mask_filenames]  # Only validate once for shared mask
                log.info(f"Converted shared mask to per-view format ({num_views} views): {mask_filenames}")
            elif isinstance(mask_filenames, list):
                # Per-view masks format - validate all masks
                mask_filenames_list = mask_filenames
            else:
                raise ValueError(
                    f"MultiViewAnomalyInpaintDataset expects mask_filename to be str or list, got {type(mask_filenames)}"
                )

            # Validate augmentation for all unique masks
            all_aug_success = True
            for mask_filename in mask_filenames_list:
                mask = self._load_cached_mask(mask_filename)
                aug_mask, aug_success = augment_binary_mask(mask, 
                                                           map(int, data["shift_values"].split(',')), 
                                                           data["rotation_angle"], 
                                                           data["morph_operation"])
                if not aug_success:
                    all_aug_success = False
                    break
            
            # If any mask fails augmentation, reset to default
            if not all_aug_success:
                data.update({
                    "shift_values": "0,0",
                    "rotation_angle": 0,
                    "morph_operation": 'none',
                })
                log.warning(f"Augmentation failed for sample, reset to default parameters")

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

            # Count instances for all views and use the maximum (reuses shared utility)
            # TODO(@maxhuang): May need to find all instances which are not overlapped by other instances.
            instances_per_view = []
            for view_idx, mask_filename in enumerate(mask_filenames_list):
                mask = self._load_cached_mask(mask_filename)
                # Re-apply augmentation if it was successful
                if all_aug_success and data["morph_operation"] != 'none':
                    aug_mask, _ = augment_binary_mask(mask, 
                                                      map(int, data["shift_values"].split(',')), 
                                                      data["rotation_angle"], 
                                                      data["morph_operation"])
                    mask = aug_mask
                
                # Use shared count_mask_instances utility
                num_instances_in_view = count_mask_instances(mask)
                instances_per_view.append(num_instances_in_view)
            
            max_instances = max(instances_per_view)
            num_instances.append(max_instances)
            
            # Log if views have different instance counts
            if len(set(instances_per_view)) > 1:
                # Extract sample identifier from image_filenames
                sample_id = 'unknown'
                if 'image_filenames' in data and data['image_filenames']:
                    first_filename = data['image_filenames'][0] if isinstance(data['image_filenames'], list) else data['image_filenames']
                    basename = os.path.basename(first_filename)
                    sample_id = os.path.splitext(basename)[0]
                
                log.info(f"Sample (first view: {first_filename}): Different instance counts across views: {instances_per_view}, using max={max_instances}")

        counts = Counter(num_instances)
        log.info("Instance count distribution in augmented masks:")
        for k in sorted(counts.keys()):
            log.info(f"  {k} instance(s): {counts[k]} image(s)")

        # Sort input_data in ascending order based on the corresponding number of instances
        paired = list(zip(num_instances, self.input_data))
        paired.sort(key=lambda pair: pair[0])
        self.input_data = [data for _, data in paired]


class MultiViewAnomalyInpaintValidationDataset(MultiViewAnomalyInpaintDataset):
    """Validation dataset for multi-view anomaly inpainting."""
    
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

