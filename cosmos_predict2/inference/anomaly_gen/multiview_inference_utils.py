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
Multi-view Inference Utilities for Anomaly Diffusion

This module provides the inference flow for multi-view anomaly generation,
including crop-and-paste and iterative generation logic.
"""

import numpy as np
import torch
from PIL import Image
from imaginaire.utils import log
from cosmos_predict2.data.anomaly_gen.anomaly_dataset import _preprocess, ANOMALY_TEXT_TEMPLATE
from cosmos_predict2.inference.anomaly_gen.inference_anomaly_diffusion_utils import (
    _split_mask, compute_psnr_in_mask, tensor_to_pil_image, _postprocess as _postprocess_single_view,
    count_mask_instances
)
from cosmos_predict2.inference.anomaly_gen.mask_augmentation import get_crop_grid_by_ratio, augment_binary_mask
from cosmos_predict2.inference.anomaly_gen.crop_paste import apply_CP_flow, paste_back, full_image_crop
import torch.distributed as dist


def _resize_if_needed(image, target_size):
    """
    Resize image to target_size if sizes don't match.
    
    Args:
        image: PIL.Image.Image
        target_size: Tuple of (width, height)
    
    Returns:
        PIL.Image.Image (resized if needed)
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)}")
    if image.size != target_size:
        resized = image.resize(target_size, Image.Resampling.BICUBIC)
        assert resized.size == target_size, f"Resize failed: {resized.size} != {target_size}"
        return resized
    return image


def _resize_mask_to_binary(mask, target_size, threshold=127):
    """
    Resize mask to target_size and convert to binary mask (0 or 255).
    
    Args:
        mask: PIL.Image.Image (L mode, binary mask)
        target_size: Tuple of (width, height)
        threshold: Threshold for binarization (default: 127)
    
    Returns:
        PIL.Image.Image (L mode, binary mask with values 0 or 255)
    """
    if not isinstance(mask, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(mask)}")
    if mask.size != target_size:
        resized = mask.resize(target_size, Image.Resampling.NEAREST)  # Use NEAREST for masks to preserve binary nature
        # Convert to binary: values >= threshold become 255, else 0
        mask_array = np.array(resized)
        binary_mask = np.where(mask_array >= threshold, 255, 0).astype(np.uint8)
        return Image.fromarray(binary_mask, mode='L')
    else:
        # Even if size matches, ensure it's binary
        mask_array = np.array(mask)
        if not np.all(np.isin(np.unique(mask_array), [0, 255])):
            binary_mask = np.where(mask_array >= threshold, 255, 0).astype(np.uint8)
            return Image.fromarray(binary_mask, mode='L')
    return mask


def _generate_annotated_image(image, mask, crop_offset, crop_size):
    """
    Generate annotated image showing mask bbox and crop region.
    
    Args:
        image: PIL.Image (original image)
        mask: PIL.Image (mask to show bbox for, typically intersection_mask)
        crop_offset: Tuple of (crop_LX, crop_UY) - upper left corner of crop region
        crop_size: Tuple of (crop_width, crop_height) - size of crop region
    
    Returns:
        PIL.Image: Annotated image with mask bbox and crop region drawn
    """
    from PIL import ImageDraw, ImageFont
    
    # Get mask bbox
    mask_array = np.array(mask.convert("L"))
    mask_region_y, mask_region_x = np.nonzero(mask_array >= 127)
    
    if mask_region_x.size == 0 or mask_region_y.size == 0:
        # Empty mask: just show crop region
        LX, RX, UY, BY = 0, 0, 0, 0
    else:
        LX, RX = min(mask_region_x), max(mask_region_x)
        UY, BY = min(mask_region_y), max(mask_region_y)
    
    crop_LX, crop_UY = crop_offset
    crop_width, crop_height = crop_size
    crop_RX = crop_LX + crop_width
    crop_BY = crop_UY + crop_height
    
    # Draw annotation
    annotated_image = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(annotated_image, "RGBA")
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw mask bbox text and rectangle
    if mask_region_x.size > 0 and mask_region_y.size > 0:
        mask_text = f"Mask BBOX: ({LX},{UY})-({RX},{BY})"
        text_bg_width = draw.textlength(mask_text, font=font) + 10
        draw.rectangle(((LX, UY-30), (LX+text_bg_width, UY)), fill=(255, 255, 255, 180))
        draw.text((LX+5, UY-25), mask_text, fill=(255, 0, 0), font=font)
        draw.rectangle(((LX, UY), (RX, BY)), outline=(255, 0, 0, 127), width=3)  # Mask region, in red
    
    # Draw crop region text and rectangle
    crop_text = f"Crop: ({crop_LX},{crop_UY})-({crop_RX},{crop_BY}), {crop_width}x{crop_height}"
    text_bg_width = draw.textlength(crop_text, font=font) + 10
    draw.rectangle(((crop_LX, crop_UY-30), (crop_LX+text_bg_width, crop_UY)), fill=(255, 255, 255, 180))
    draw.text((crop_LX+5, crop_UY-25), crop_text, fill=(0, 128, 0), font=font)
    draw.rectangle(((crop_LX, crop_UY), (crop_RX, crop_BY)), outline=(0, 255, 0, 127), width=3)  # Cropped area, in green
    
    return annotated_image


def _union_masks(masks):
    """
    Compute the union of multiple masks.
    
    Args:
        masks: List of PIL Image masks (one per view)
    
    Returns:
        PIL.Image: Union mask (binary, 0 or 255)
    """
    if not masks:
        raise ValueError("Cannot compute union of empty mask list")
    
    # Ensure all masks have the same size
    first_size = masks[0].size
    for mask in masks:
        if mask.size != first_size:
            raise ValueError(f"All masks must have the same size. Got {first_size} and {mask.size}")
    
    # Convert all masks to numpy arrays and compute union
    mask_arrays = []
    for mask in masks:
        mask_array = np.array(mask.convert("L"))
        # Binarize: >= 127 -> 255, else 0
        binary_mask = np.where(mask_array >= 127, 255, 0).astype(np.uint8)
        mask_arrays.append(binary_mask)
    
    # Union: any pixel that is 255 in any mask becomes 255
    union_array = np.maximum.reduce(mask_arrays)
    
    return Image.fromarray(union_array, mode='L')


def _intersect_masks(mask1, mask2):
    """
    Compute the intersection of two masks.
    
    Args:
        mask1: PIL.Image mask
        mask2: PIL.Image mask
    
    Returns:
        PIL.Image: Intersection mask (binary, 0 or 255)
    """
    if mask1.size != mask2.size:
        raise ValueError(f"Masks must have the same size. Got {mask1.size} and {mask2.size}")
    
    # Convert to numpy arrays and binarize
    mask1_array = np.array(mask1.convert("L"))
    mask2_array = np.array(mask2.convert("L"))
    
    binary_mask1 = np.where(mask1_array >= 127, 255, 0).astype(np.uint8)
    binary_mask2 = np.where(mask2_array >= 127, 255, 0).astype(np.uint8)
    
    # Intersection: both must be 255
    intersection_array = np.minimum(binary_mask1, binary_mask2)
    
    return Image.fromarray(intersection_array, mode='L')


def _split_union_mask_for_multiview(masks, max_k, sample_name="unknown"):
    """
    Split masks for multi-view using union-based approach.
    
    Strategy:
    1. Compute union of all view masks to get a "super mask"
    2. Split the union mask according to max_k
    3. For each split mask and each view, compute intersection with the view's original mask
       - The split mask is used for crop_and_paste (ensures consistent augmentation across views)
       - The intersection is used as the denoise condition (view-specific)
    
    Args:
        masks: List of PIL Image masks (one per view)
        max_k: Maximum number of instances
        sample_name: Sample identifier for logging
    
    Returns:
        Tuple of (union_instance_masks, view_intersection_masks)
        - union_instance_masks: List of split masks from union (for crop_and_paste)
        - view_intersection_masks: List[List[PIL.Image]] - [num_instances][num_views]
                                  Each entry is the intersection of split_mask with view_mask (for denoise)
    """
    # Step 1: Compute union of all view masks
    union_mask = _union_masks(masks)
    log.info(f"Sample {sample_name}: Computed union mask from {len(masks)} views")
    
    # Step 2: Split the union mask
    union_instance_masks = _split_mask(union_mask, max_k)
    num_instances = len(union_instance_masks)
    log.info(f"Sample {sample_name}: Split union mask into {num_instances} instance(s)")
    
    # Step 3: For each split mask and each view, compute intersection
    view_intersection_masks = []  # [num_instances][num_views]
    for inst_idx, split_mask in enumerate(union_instance_masks):
        view_intersections = []
        for view_idx, view_mask in enumerate(masks):
            intersection = _intersect_masks(split_mask, view_mask)
            view_intersections.append(intersection)
        view_intersection_masks.append(view_intersections)
    
    return union_instance_masks, view_intersection_masks


def inpaint_multiview_image(inpaint_condition, model):
    """
    Multi-view version of inpaint_image with full crop-and-paste support.
    
    Main flow for inpainting a batch of multi-view images with iterative generation.
    """
    # Prepare data_batch for diffusion inference pipeline
    diffusion_data_batches = _prepare_multiview_diffusion_inference_data_batches(inpaint_condition, model)
    max_instance = len(diffusion_data_batches)
    B = len(diffusion_data_batches[0]["original_images_per_view"][0])  # Number of samples
    
    # Collect original images/masks (per-view) for final output
    original_images_per_view = diffusion_data_batches[-1]["original_images_per_view"]  # List[List[PIL]] - [num_views][B]
    original_masks_per_view = diffusion_data_batches[-1]["original_masks_per_view"]
    
    # Collect cropped images/masks/annotations (per instance, per sample, per view)
    cropped_images_per_view = [
        [
            [
                diffusion_data_batches[j]['cropped_images_per_view'][view_idx][i] 
                for j in range(max_instance) 
                if not diffusion_data_batches[j]['dummy'][i]
            ]
            for i in range(B)
        ]
        for view_idx in range(len(original_images_per_view))
    ]
    
    cropped_masks_per_view = [
        [
            [
                diffusion_data_batches[j]['cropped_masks_per_view'][view_idx][i] 
                for j in range(max_instance) 
                if not diffusion_data_batches[j]['dummy'][i]
            ]
            for i in range(B)
        ]
        for view_idx in range(len(original_masks_per_view))
    ]
    
    annotated_images_per_view = [
        [
            [
                diffusion_data_batches[j]['annotated_images_per_view'][view_idx][i] 
                for j in range(max_instance) 
                if not diffusion_data_batches[j]['dummy'][i]
            ]
            for i in range(B)
        ]
        for view_idx in range(len(original_images_per_view))
    ]

    # Iterative inpainting
    prev_reconstructed_videos = None
    prev_batch = None
    for diffusion_data_batch in diffusion_data_batches:
        if prev_reconstructed_videos is not None:
            _replace_multiview_batch_input(diffusion_data_batch, prev_batch, prev_reconstructed_videos, model)
        
        with torch.no_grad():
            reconstructed_videos = model.pipe(
                diffusion_data_batch,
                guidance=inpaint_condition.guidance,
                seed=inpaint_condition.seed,
                num_steps=inpaint_condition.num_steps,
                is_negative_prompt=("neg_t5_text_embeddings" in diffusion_data_batch),
                n_sample=1  # Multi-view doesn't support multiple samples yet
            )
            if dist.is_initialized():
                dist.barrier()
            diffusion_data_batch['denoised_result'] = reconstructed_videos
            reconstructed_videos = _postprocess_multiview(inpaint_condition, diffusion_data_batch)
            prev_reconstructed_videos = reconstructed_videos
            prev_batch = diffusion_data_batch
    
    # Calculate PSNR for each view
    # Note: For multi-view, when crop_and_paste=False, we need to resize reconstructed_videos
    # to match original_images size, since reconstructed_videos are 512x512 (model output)
    # but original_images may be different sizes (e.g., 256x256 for SimCardSet)
    # Also, masks need to be resized and converted to binary (0 or 255) after resize
    num_views = len(original_images_per_view)
    inpaint_condition.PSNR = [
        [
            compute_psnr_in_mask(
                _resize_if_needed(
                    reconstructed_videos[view_idx][i],
                    original_images_per_view[view_idx][i].size
                ),
                original_images_per_view[view_idx][i], 
                _resize_mask_to_binary(
                    original_masks_per_view[view_idx][i],
                    original_images_per_view[view_idx][i].size
                )
            )
            for i in range(B)
        ]
        for view_idx in range(num_views)
    ]
    
    log.info(f"PSNR per view: {inpaint_condition.PSNR}")
    if dist.is_initialized():
        dist.barrier()
    
    # Return a dictionary of all images for recording
    # Format: List[List[PIL]] - [num_views][B]
    result_dict = {
        "original_image": original_images_per_view,
        "original_mask": original_masks_per_view,
        "reconstructed_image": reconstructed_videos,
        "cropped_image": cropped_images_per_view,
        "cropped_mask": cropped_masks_per_view,
        "annotated_image": annotated_images_per_view,
    }
    
    return result_dict, diffusion_data_batches[-1]['name'], diffusion_data_batches[-1]['index']


@torch.no_grad()
def _prepare_multiview_diffusion_inference_data_batches(inpaint_condition, model):
    """
    Multi-view version of _prepare_diffusion_inference_data_batches.
    
    Handles per-view image/mask loading and crop-and-paste preparation.
    """
    B = len(inpaint_condition.image_filenames)
    num_views = len(inpaint_condition.image_filenames[0])  # Assume all samples have same number of views
    
    log.info(f"Preparing {B} samples with {num_views} views each")
    
    # Processed databatch for inference pipeline
    full_instance_batch = {}
    
    # Load & Preprocess images / masks (per-view)
    instance_indices_per_image = []
    image_filenames_per_view = [[] for _ in range(num_views)]
    mask_filenames_per_view = [[] for _ in range(num_views)]
    anomaly_names = []
    
    model_input_videos = []  # [num_instances] of [B, C, T, H, W]
    model_input_masks_per_view = [[] for _ in range(num_views)]  # [num_views][num_instances] of [B, C, H, W]
    
    original_images_per_view = [[] for _ in range(num_views)]  # [num_views][B]
    original_masks_per_view = [[] for _ in range(num_views)]
    
    # For C&P Flow (per-view)
    crop_and_pastes, cropped_images_per_view, cropped_masks_per_view, annotated_images_per_view, upper_lefts_per_view = [], [[] for _ in range(num_views)], [[] for _ in range(num_views)], [[] for _ in range(num_views)], [[] for _ in range(num_views)]
    crop_grid_Xs, crop_grid_Ys, crop_ratios, poisson_blends = [], [], [], []
    shift_valuess, rotation_angles, morph_operations = [], [], []
    indice = []
    
    all_instance_idx = 0
    
    for idx in range(B):
        image_filenames = inpaint_condition.image_filenames[idx]  # List of view paths
        mask_filenames = inpaint_condition.mask_filename[idx]  # List of mask paths (per-view)
        
        # Handle shared mask format (single path replicated for all views)
        if isinstance(mask_filenames, str):
            mask_filenames = [mask_filenames] * num_views
        elif len(mask_filenames) == 1:
            mask_filenames = mask_filenames * num_views
        
        # Load multi-view images and masks
        images = [Image.open(path) for path in image_filenames]
        masks = [Image.open(path).convert("L") for path in mask_filenames]
        
        # Verify all sizes match
        for i in range(num_views):
            assert images[i].size == masks[i].size, f"Error: Image {image_filenames[i]} size mismatch with mask {mask_filenames[i]}"
        
        # Mask Augmentation (apply same augmentation to all views)
        aug_masks = []
        aug_success = True
        for mask in masks:
            aug_mask, success = augment_binary_mask(
                mask, 
                inpaint_condition.shift_values[idx], 
                inpaint_condition.rotation_angle[idx], 
                inpaint_condition.morph_operation[idx]
            )
            if success:
                aug_masks.append(aug_mask)
            else:
                aug_success = False
                break
        
        if aug_success:
            masks = aug_masks
        else:
            log.warning(f"No valid mask found after augmentations for sample {idx}. Use original masks for generation.")
            inpaint_condition.shift_values[idx] = (0, 0)
            inpaint_condition.rotation_angle[idx] = 0
            inpaint_condition.morph_operation[idx] = 'none'

        # Save original masks after augmentation (for storing in original_masks_per_view)
        original_masks = [mask.copy() for mask in masks]
        
        # Split mask using union-based approach
        # Step 1: Compute union of all view masks and split it
        # Step 2: For each split mask, compute intersection with each view's mask
        # - union_instance_masks: used for crop_and_paste (ensures consistent augmentation)
        # - view_intersection_masks: used as denoise condition (view-specific)
        union_instance_masks, view_intersection_masks = _split_union_mask_for_multiview(
            masks,
            inpaint_condition.iteration_generation_max_instance,
            sample_name=f"{idx}"
        )
        
        instance_indices_per_image.append([])
        
        for inst_idx, union_inst_mask in enumerate(union_instance_masks):
            # Crop & Paste flow
            # Use union_inst_mask for crop_and_paste to ensure consistent augmentation across views
            crop_and_paste = inpaint_condition.crop_and_paste[idx]
            crop_grid_X, crop_grid_Y, crop_ratio = inpaint_condition.crop_grid_X[idx], inpaint_condition.crop_grid_Y[idx], inpaint_condition.crop_ratio[idx]
            
            if crop_and_paste:
                use_crop_ratio = crop_ratio not in (None, "none")
                if use_crop_ratio and crop_grid_X is not None:
                    log.warning(f"crop_ratio and crop_grid_X co-exist! Will use crop_ratio {crop_ratio} for generation.")
                
                # Crop w/ fixed grid size (using union mask)
                if use_crop_ratio:
                    crop_grid_size = get_crop_grid_by_ratio(union_inst_mask, crop_ratio)
                    crop_grid_X = crop_grid_size
                    crop_grid_Y = crop_grid_size
            
            # Process each view
            view_tensors = []
            view_mask_tensors = []
            for view_idx in range(num_views):
                image = images[view_idx]
                # Use intersection mask as denoise condition (view-specific)
                intersection_mask = view_intersection_masks[inst_idx][view_idx]
                
                # Use union mask to determine crop region (ensures consistent crop across views)
                # But use intersection_mask for cropped_mask and annotated_image (shows actual denoise condition)
                if crop_and_paste:
                    # Step 1: Use union_inst_mask to determine crop region
                    cropped_image, _, upper_left, _ = apply_CP_flow(image, union_inst_mask, crop_grid_X, crop_grid_Y)
                    model_input_image = cropped_image
                    # Step 2: Crop intersection_mask to the same region for cropped_mask and model_input_mask
                    crop_LX, crop_UY = upper_left
                    crop_RX = crop_LX + cropped_image.size[0]
                    crop_BY = crop_UY + cropped_image.size[1]
                    cropped_mask = intersection_mask.crop((crop_LX, crop_UY, crop_RX, crop_BY))
                    model_input_mask = cropped_mask
                    # Step 3: Generate annotated_image using intersection_mask (shows actual denoise condition)
                    annotated_image = _generate_annotated_image(image, intersection_mask, upper_left, cropped_image.size)
                else:
                    # Step 1: Use union_inst_mask to get bbox info (for consistency check)
                    _, _, upper_left, _ = full_image_crop(image, union_inst_mask)
                    cropped_image = image  # Full image when crop_and_paste=False
                    model_input_image = cropped_image
                    # Step 2: Use intersection_mask for cropped_mask and model_input_mask
                    intersection_array = np.array(intersection_mask)
                    if np.sum(intersection_array >= 127) == 0:
                        # Empty intersection: view doesn't see this instance
                        cropped_mask = Image.new("L", intersection_mask.size, 0)
                        model_input_mask = cropped_mask
                        log.debug(f"Sample {idx}, instance {inst_idx}, view {view_idx}: Empty intersection mask (view doesn't see this instance)")
                    else:
                        cropped_mask = intersection_mask
                        model_input_mask = intersection_mask
                    # Step 3: Generate annotated_image using intersection_mask (shows actual denoise condition)
                    annotated_image = _generate_annotated_image(image, intersection_mask, (0, 0), image.size)
                
                # Resize to 512x512
                if model_input_image.size != (512, 512):
                    model_input_image = model_input_image.resize((512, 512), Image.Resampling.BICUBIC)
                    model_input_mask = model_input_mask.resize((512, 512), Image.Resampling.BICUBIC)
                
                # Preprocess
                norm_image, norm_mask = _preprocess(model_input_image, model_input_mask)
                if norm_mask.ndim == 2:
                    norm_mask = norm_mask[:, :, np.newaxis]
                if norm_image.ndim == 2:
                    norm_image = norm_image[:, :, np.newaxis]
                
                image_tensor = torch.from_numpy(norm_image).permute(2, 0, 1)  # [C, H, W]
                mask_tensor = torch.from_numpy(norm_mask).permute(2, 0, 1)
                
                view_tensors.append(image_tensor)
                view_mask_tensors.append(mask_tensor)
                
                # Store per-view metadata
                cropped_images_per_view[view_idx].append(cropped_image)
                cropped_masks_per_view[view_idx].append(cropped_mask)
                annotated_images_per_view[view_idx].append(annotated_image)
                upper_lefts_per_view[view_idx].append(upper_left)
                image_filenames_per_view[view_idx].append(image_filenames[view_idx])
                mask_filenames_per_view[view_idx].append(mask_filenames[view_idx])
                
                # Store original images/masks (only once per sample, not per instance)
                if inst_idx == 0:
                    original_images_per_view[view_idx].append(image)
                    original_masks_per_view[view_idx].append(original_masks[view_idx])
            
            # Stack views into video tensor [T, C, H, W]
            video_tensor = torch.stack(view_tensors, dim=0)  # [T, C, H, W]
            model_input_videos.append(video_tensor)
            
            # Store per-view mask tensors
            for view_idx in range(num_views):
                model_input_masks_per_view[view_idx].append(view_mask_tensors[view_idx])
            
            # Collect instance metadata (same for all views)
            crop_and_pastes.append(crop_and_paste)
            crop_grid_Xs.append(inpaint_condition.crop_grid_X[idx])
            crop_grid_Ys.append(inpaint_condition.crop_grid_Y[idx])
            crop_ratios.append(inpaint_condition.crop_ratio[idx])
            poisson_blends.append(inpaint_condition.poisson_blend[idx])
            shift_valuess.append(inpaint_condition.shift_values[idx])
            rotation_angles.append(inpaint_condition.rotation_angle[idx])
            morph_operations.append(inpaint_condition.morph_operation[idx])
            
            # Convert anomaly_type to string format 'Sample+Defect'
            anomaly_type = inpaint_condition.anomaly_type[idx]
            if isinstance(anomaly_type, list):
                anomaly_name_str = f"{anomaly_type[0]}+{anomaly_type[1]}"
            else:
                anomaly_name_str = anomaly_type
            anomaly_names.append(anomaly_name_str)
            
            indice.append(inpaint_condition.index[idx])
            
            instance_indices_per_image[-1].append(all_instance_idx)
            all_instance_idx += 1
    
    # Build full_instance_batch
    H, W = 512, 512
    full_instance_batch['fps'] = torch.tensor([1] * B).cuda()
    full_instance_batch['image_size'] = torch.tensor([H, W, H, W]).cuda()
    full_instance_batch["num_frames"] = torch.tensor([num_views] * B).cuda()
    full_instance_batch["padding_mask"] = torch.zeros(B, 1, H, W, dtype=torch.bfloat16).cuda()
    full_instance_batch["t5_text_mask"] = torch.ones(B, 512, dtype=torch.bfloat16).cuda()
    full_instance_batch["neg_t5_text_mask"] = torch.ones(B, 512, dtype=torch.bfloat16).cuda()
    
    full_instance_batch["original_images_per_view"] = original_images_per_view
    full_instance_batch["original_masks_per_view"] = original_masks_per_view
    
    # Stack videos: [num_instances, T, C, H, W] -> [num_instances, C, T, H, W]
    full_instance_batch["video"] = torch.stack([v.permute(1, 0, 2, 3) for v in model_input_videos]).to("cuda")  # [num_instances, C, T, H, W]
    
    # Stack masks: Build [num_instances, T, H, W]
    # Each mask_per_view has [num_instances] tensors of [C, H, W]
    full_instance_batch["mask"] = torch.stack([
        torch.stack([model_input_masks_per_view[view_idx][inst_idx][0] for view_idx in range(num_views)], dim=0)  # [T, H, W]
        for inst_idx in range(len(model_input_videos))
    ]).to("cuda")  # [num_instances, T, H, W]
    
    # Basic information
    full_instance_batch['image_filenames_per_view'] = image_filenames_per_view
    full_instance_batch['mask_filenames_per_view'] = mask_filenames_per_view
    full_instance_batch['name'] = anomaly_names
    
            # Captions
    captions = []
    for anomaly_type in anomaly_names:
        # Handle both list format ['Sample', 'Defect'] and string format 'Sample+Defect'
        if isinstance(anomaly_type, list):
            sample_name, anomaly_name = anomaly_type[0], anomaly_type[1]
        else:
            sample_name, anomaly_name = anomaly_type.split("+")
        caption = ANOMALY_TEXT_TEMPLATE.format(sample_name, anomaly_name, '*')
        captions.append(caption)
    full_instance_batch['caption'] = captions
    
    # Inpainting: Prepare text embedding for anomaly generation
    with torch.no_grad():
        _, latent_state, condition = model.pipe.get_data_and_condition(full_instance_batch)
    full_instance_batch["guided_image"] = latent_state
    full_instance_batch["guided_mask"], _ = model.pipe._get_guided_mask_and_weight(full_instance_batch["guided_image"], full_instance_batch['mask'])
    full_instance_batch["guided_mask"] = (1 - full_instance_batch["guided_mask"].to(torch.bfloat16))
    
    # Crop & Paste flow
    full_instance_batch["crop_grid_Xs"] = crop_grid_Xs
    full_instance_batch["crop_grid_Ys"] = crop_grid_Ys
    full_instance_batch["upper_lefts_per_view"] = upper_lefts_per_view
    full_instance_batch["cropped_images_per_view"] = cropped_images_per_view
    full_instance_batch["cropped_masks_per_view"] = cropped_masks_per_view
    full_instance_batch["annotated_images_per_view"] = annotated_images_per_view
    full_instance_batch["dummy"] = [False] * len(anomaly_names)
    full_instance_batch["index"] = indice
    
    # Split into instance batches for iterative generation
    # (Similar to single-view logic but adapted for multi-view)
    
    local_max_instances = max(len(indices) for indices in instance_indices_per_image)
    max_instances_tensor = torch.tensor([local_max_instances], device="cuda")
    
    if dist.is_initialized():
        dist.all_reduce(max_instances_tensor, op=dist.ReduceOp.MAX)
    
    max_instances = int(max_instances_tensor.item())
    
    # Placeholder objects for padding
    placeholder = {
        "video": torch.zeros_like(model_input_videos[0]).permute(1, 0, 2, 3),  # [C, T, H, W]
        "mask": torch.zeros((num_views, H, W)),  # [T, H, W]
        "orig_images": [Image.new("RGB", original_images_per_view[0][0].size) for _ in range(num_views)],
        "orig_masks": [Image.new("L", original_masks_per_view[0][0].size) for _ in range(num_views)],
        "image_paths": [""] * num_views,
        "mask_paths": [""] * num_views,
        "anomaly_name": anomaly_names[0],
        "caption": captions[0],
        "upper_lefts": [(0, 0)] * num_views,
        "crop_images": [Image.new("RGB", original_images_per_view[0][0].size) for _ in range(num_views)],
        "crop_masks": [Image.new("L", original_masks_per_view[0][0].size) for _ in range(num_views)],
        "annotated_images": [Image.new("RGB", original_images_per_view[0][0].size) for _ in range(num_views)],
        "dummy": True,
        "index": -1
    }
    
    # Keys that never change between instance batches
    static_keys = full_instance_batch.keys() - {
        "video", "mask",
        "cropped_images_per_view", "cropped_masks_per_view",
        "original_images_per_view", "original_masks_per_view",
        "upper_lefts_per_view",
        "image_filenames_per_view", "mask_filenames_per_view",
        "name", "caption",
    }
    static_fields = {k: full_instance_batch[k] for k in static_keys}
    
    instance_batches = []
    for position in range(max_instances):
        # Resolve real or placeholder index for every input image
        flat_indices = [
            indices[position] if position < len(indices) else None
            for indices in instance_indices_per_image
        ]
        
        def gather(source_list, field):
            return [
                source_list[i] if i is not None else placeholder[field]
                for i in flat_indices
            ]
        
        def gather_per_view(source_lists_per_view, field, use_sample_idx=False):
            # source_lists_per_view is [num_views][num_items]
            # If use_sample_idx=True: num_items = B (sample count), use sample index directly
            # If use_sample_idx=False: num_items = num_instances, use instance index from flat_indices
            # Returns [num_views][B]
            if use_sample_idx:
                # For original_images_per_view and original_masks_per_view
                # Use sample index directly (flat_indices index) instead of instance index
                return [
                    [
                        source_lists_per_view[view_idx][sample_idx] if flat_indices[sample_idx] is not None else placeholder[field][view_idx]
                        for sample_idx in range(B)
                    ]
                    for view_idx in range(num_views)
                ]
            else:
                # For cropped_images_per_view, etc.
                # Use instance index from flat_indices
                return [
                    [
                        source_lists_per_view[view_idx][i] if i is not None else placeholder[field][view_idx]
                        for i in flat_indices
                    ]
                    for view_idx in range(num_views)
                ]
        
        instance_batch = {
            **static_fields,
            "video": torch.stack([
                full_instance_batch["video"][i] if i is not None else placeholder["video"]
                for i in flat_indices
            ]).to("cuda"),  # [B, C, T, H, W]
            "mask": torch.stack([
                full_instance_batch["mask"][i] if i is not None else placeholder["mask"]
                for i in flat_indices
            ]).to("cuda"),  # [B, T, H, W]
            "original_images_per_view": gather_per_view(original_images_per_view, "orig_images", use_sample_idx=True),
            "original_masks_per_view": gather_per_view(original_masks_per_view, "orig_masks", use_sample_idx=True),
            "image_filenames_per_view": gather_per_view(image_filenames_per_view, "image_paths", use_sample_idx=False),
            "mask_filenames_per_view": gather_per_view(mask_filenames_per_view, "mask_paths", use_sample_idx=False),
            "name": gather(anomaly_names, "anomaly_name"),
            "caption": gather(captions, "caption"),
            "upper_lefts_per_view": gather_per_view(upper_lefts_per_view, "upper_lefts"),
            "cropped_images_per_view": gather_per_view(cropped_images_per_view, "crop_images"),
            "cropped_masks_per_view": gather_per_view(cropped_masks_per_view, "crop_masks"),
            "annotated_images_per_view": gather_per_view(annotated_images_per_view, "annotated_images"),
            "dummy": gather([False] * len(anomaly_names), "dummy"),
            "index": gather(indice, "index"),
        }
        
        instance_batches.append(instance_batch)
    
    # Reverse so dummy samples are processed first
    instance_batches.reverse()
    return instance_batches


def _postprocess_multiview(inpaint_condition, diffusion_data_batch):
    """
    Multi-view version of _postprocess.
    Reuses core tensor-to-PIL and paste_back logic from single-view.
    
    Returns: List[List[PIL]] - [num_views][B]
    """
    # Reconstruct images from latent
    reconstructed_videos = diffusion_data_batch['denoised_result']  # [B, C, T, H, W]
    reconstructed_videos = ((reconstructed_videos + 1).clamp(0, 2)) / 2.0
    
    B, C, T, H, W = reconstructed_videos.shape
    num_views = T
    
    # Convert to PIL per-view: [num_views][B]
    reconstructed_images_per_view = [
        [tensor_to_pil_image(reconstructed_videos[i, :, view_idx, :, :], diffusion_data_batch['original_images_per_view'][view_idx][i].mode)
         for i in range(B)]
        for view_idx in range(num_views)
    ]
    
    # Paste back logic - iterate per view
    for i in range(B):
        if inpaint_condition.crop_and_paste[i]:
            for view_idx in range(num_views):
                # Reuse paste_back from single-view utils
                CP_reconstructed_image = diffusion_data_batch['original_images_per_view'][view_idx][i].copy()
                paste_back(
                    CP_reconstructed_image,
                    reconstructed_images_per_view[view_idx][i],  # r_image
                    diffusion_data_batch['cropped_images_per_view'][view_idx][i],  # c_image
                    diffusion_data_batch['cropped_masks_per_view'][view_idx][i],  # c_mask
                    diffusion_data_batch['upper_lefts_per_view'][view_idx][i],  # upper_left
                    inpaint_condition.poisson_blend[i]
                )
                reconstructed_images_per_view[view_idx][i] = CP_reconstructed_image
    
    return reconstructed_images_per_view


def _replace_multiview_batch_input(batch, prev_batch, new_images_per_view, model):
    """
    Multi-view version of _replace_batch_input.
    Reuses pipeline's get_data_and_condition and _get_guided_mask_and_weight.
    
    new_images_per_view: [num_views][B] PIL images
    """
    B = len(batch["video"])
    num_views = batch["video"].shape[2]
    
    for i in range(B):
        if prev_batch["dummy"][i]:  # skip for dummy sample
            continue
        
        # Update original images for all views
        for view_idx in range(num_views):
            batch["original_images_per_view"][view_idx][i] = new_images_per_view[view_idx][i]
        
        # Crop and update video tensor for all views
        view_tensors = []
        for view_idx in range(num_views):
            offsets = batch["upper_lefts_per_view"][view_idx][i]
            crop_size = batch["cropped_images_per_view"][view_idx][i].size
            crop_image = new_images_per_view[view_idx][i].crop((offsets[0], offsets[1], offsets[0] + crop_size[0], offsets[1] + crop_size[1]))
            crop_image = crop_image.resize((512, 512), Image.Resampling.BICUBIC)
            
            from torchvision.transforms.functional import to_tensor
            view_tensor = to_tensor(crop_image)  # [C, H, W]
            view_tensors.append(view_tensor)
        
        # Stack and update: [C, T, H, W]
        batch["video"][i] = torch.stack(view_tensors, dim=1).cuda() * 2.0 - 1.0  # [C, T, H, W]
    
    # Reuse pipeline methods for text embedding and guided mask preparation
    with torch.no_grad():
        _, latent_state, condition = model.pipe.get_data_and_condition(batch)
    batch["guided_image"] = latent_state
    batch["guided_mask"], _ = model.pipe._get_guided_mask_and_weight(batch["guided_image"], batch['mask'])
    batch["guided_mask"] = (1 - batch["guided_mask"].to(torch.bfloat16))

