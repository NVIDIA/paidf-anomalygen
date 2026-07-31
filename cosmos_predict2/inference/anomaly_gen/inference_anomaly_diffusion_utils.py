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

import numpy as np
import torch
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import os
from torchvision.transforms.functional import to_tensor

from cosmos_predict2.data.anomaly_gen.anomaly_dataset import  _preprocess, ANOMALY_TEXT_TEMPLATE
from imaginaire.utils import log
from cosmos_predict2.inference.anomaly_gen.mask_augmentation import  get_crop_grid_by_ratio, augment_binary_mask
from cosmos_predict2.inference.anomaly_gen.crop_paste import apply_CP_flow, paste_back, full_image_crop
import torch.distributed as dist


def _pil_image_from_array(image_array, image_mode=None):
    pil_image = Image.fromarray(np.asarray(image_array))
    if image_mode is not None and pil_image.mode != image_mode:
        pil_image = pil_image.convert(image_mode)
    return pil_image


def _load_condition_image(image_filename, image_array=None, image_mode=None):
    if image_array is not None:
        return _pil_image_from_array(image_array, image_mode)
    with Image.open(image_filename) as fp:
        return fp.convert("RGB")


def _load_condition_mask(mask_filename, mask_array=None, mask_mode=None):
    if mask_array is not None:
        mask = _pil_image_from_array(mask_array, mask_mode)
        return mask if mask.mode == "L" else mask.convert("L")
    with Image.open(mask_filename) as fp:
        return fp.convert("L")


def _apply_image_guardrail(model, reconstructed_images):
    """Run the image content-safety guardrail on each generated image.

    The guardrail runner is created once on the pipeline (model.pipe) when
    `guardrail_config.image_enabled` is set; if it is disabled or unavailable,
    every image is treated as safe and left untouched.

    Any image flagged unsafe is replaced in place with a black image so it
    cannot propagate into the dataset (or the validation KPI).

    Returns:
        list[bool]: per-image safe verdict, aligned with `reconstructed_images`.
    """
    runner = getattr(getattr(model, "pipe", None), "image_guardrail_runner", None)
    if runner is None:
        return [True] * len(reconstructed_images)

    from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

    safe_flags = []
    for i, image in enumerate(reconstructed_images):
        is_safe = guardrail_presets.run_image_guardrail(image, runner)
        safe_flags.append(is_safe)
        if not is_safe and isinstance(image, Image.Image):
            # Replace the unsafe image with a black image of the same size/mode.
            reconstructed_images[i] = Image.new(image.mode, image.size)
    num_unsafe = safe_flags.count(False)
    if num_unsafe:
        log.critical(
            f"Image guardrail flagged {num_unsafe}/{len(safe_flags)} generated image(s) as unsafe; "
            "replaced with black image(s)."
        )
    return safe_flags


def inpaint_image(inpaint_condition, model):
    """
    Main flow for inpainting a batch of images
    """
    # Prepare data_batch for diffusion inference pipeline
    diffusion_data_batches = _prepare_diffusion_inference_data_batches(inpaint_condition, model)
    max_instance = len(diffusion_data_batches)
    B = len(diffusion_data_batches[0]["original_image"])
    original_images = [img.copy() for img in diffusion_data_batches[-1]["original_image"]]
    original_masks = [m.copy() for m in diffusion_data_batches[-1]["original_mask"]]
    cropped_images = [
        [
            diffusion_data_batches[j]['cropped_image'][i] 
            for j in range(max_instance) 
            if not diffusion_data_batches[j]['dummy'][i]
        ]
        for i in range(B)
    ]
    cropped_masks = [
        [
            diffusion_data_batches[j]['cropped_mask'][i] 
            for j in range(max_instance) 
            if not diffusion_data_batches[j]['dummy'][i]
        ]
        for i in range(B)
    ]
    annotated_images = [
        [
            diffusion_data_batches[j]['annotated_image'][i] 
            for j in range(max_instance) 
            if not diffusion_data_batches[j]['dummy'][i]
        ]
        for i in range(B)
    ]

    # Inpainting
    prev_reconstructed_images = None
    prev_batch = None
    for diffusion_data_batch in diffusion_data_batches:
        if prev_reconstructed_images:
            _replace_batch_input(diffusion_data_batch, prev_batch, prev_reconstructed_images, model)
        with torch.no_grad():
            reconstructed_images = model.pipe(
                    diffusion_data_batch,
                    guidance=inpaint_condition.guidance,
                    seed=inpaint_condition.seed,
                    num_steps=inpaint_condition.num_steps,
                    is_negative_prompt = ("neg_t5_text_embeddings" in diffusion_data_batch),
                    n_sample=inpaint_condition.num_generated_images,
                    use_cuda_graphs=getattr(model.config, "use_cuda_graphs_for_dit", False),
                )
            if dist.is_initialized():
                dist.barrier()
            diffusion_data_batch['denoised_result'] = reconstructed_images
            reconstructed_images = _postprocess(inpaint_condition, diffusion_data_batch)
            prev_reconstructed_images = reconstructed_images
            prev_batch = diffusion_data_batch

    # Post-generation image guardrail. Runs on every generated image for both
    # training validation and inference (both reach this single chokepoint).
    # Unsafe images are replaced with a black image so they cannot enter the
    # dataset, and the per-image safe/unsafe verdict is recorded on
    # inpaint_condition.guardrail_safe for the caller to log / persist.
    inpaint_condition.guardrail_safe = _apply_image_guardrail(model, reconstructed_images)

    # Calculate PSNR
    inpaint_condition.PSNR = [
        compute_psnr_in_mask(reconstructed_images[i], original_images[i], original_masks[i])
        for i in range(len(reconstructed_images))
    ]

    log.info(f"PSNR: {inpaint_condition.PSNR}")
    if dist.is_initialized():
        dist.barrier()
    # Return a dictionary of all images for recording
    result_dict = {
        "original_image": original_images,
        "original_mask": original_masks,
        "reconstructed_image": reconstructed_images,
        "cropped_image": cropped_images,
        "cropped_mask": cropped_masks,
        "annotated_image": annotated_images,
    }
    return result_dict, diffusion_data_batches[-1]['name'], diffusion_data_batches[-1]['index']

@torch.no_grad()
def _prepare_diffusion_inference_data_batches(inpaint_condition, model):
    """ This function preprocess data_batch to align API usage as training pipeline"""
    B = len(inpaint_condition.image_filename)

    # Processed databatch for inference pipeline
    full_instance_batch = {}
    
    # Load & Preprocess image / mask
    instance_indices_per_image = []
    image_filenames, mask_filenames, anomaly_names = [], [], []
    model_input_images, model_input_masks = [], []
    original_images, original_masks = [], []
    crop_and_pastes, cropped_images, cropped_masks, annotated_images, upper_lefts = [], [], [], [], [] # For C&P Flow
    crop_grid_Xs, crop_grid_Ys, crop_ratios, poisson_blends = [], [], [], []
    shift_valuess, rotation_angles, morph_operations = [], [], []
    indice = []
    all_instance_idx = 0
    for idx, (image_filename, mask_filename) in enumerate(zip(inpaint_condition.image_filename, inpaint_condition.mask_filename)):
        # Load Image & Mask
        image = _load_condition_image(
            image_filename,
            inpaint_condition.loaded_image_array[idx],
            inpaint_condition.loaded_image_mode[idx],
        )
        mask = _load_condition_mask(
            mask_filename,
            inpaint_condition.loaded_mask_array[idx],
            inpaint_condition.loaded_mask_mode[idx],
        )
        assert image.size == mask.size,  f"Error: Image filename {image_filename} 's size with mask filename {mask_filename}"

        # Mask Augmentation
        aug_mask, aug_success = augment_binary_mask(mask, inpaint_condition.shift_values[idx], inpaint_condition.rotation_angle[idx], inpaint_condition.morph_operation[idx])
        if aug_success:
            mask = aug_mask
        else:
            log.warning(f"No valid mask found after augmentations. Use original mask for generation.")
            inpaint_condition.shift_values[idx] = (0,0)
            inpaint_condition.rotation_angle[idx] = 0
            inpaint_condition.morph_operation[idx] = 'none'
        
        instance_masks = _split_mask(mask, inpaint_condition.iteration_generation_max_instance)
        instance_indices_per_image.append([])
        for inst_idx, inst_mask in enumerate(instance_masks):        
            # Crop & Paste flow
            crop_and_paste, crop_grid_X, crop_grid_Y, crop_ratio = inpaint_condition.crop_and_paste[idx], inpaint_condition.crop_grid_X[idx], inpaint_condition.crop_grid_Y[idx], inpaint_condition.crop_ratio[idx]
            if crop_and_paste:
                use_crop_ratio = crop_ratio not in (None, "none")
                if use_crop_ratio  and crop_grid_X is not None:
                    log.warning(f"crop_ratio and crop_grid_X co-exist! Will use crop_ratio  {inpaint_condition.crop_ratio[idx]} for generation.")
                # Crop w/ fixed grid size
                if use_crop_ratio :
                    crop_grid_size = get_crop_grid_by_ratio(inst_mask, crop_ratio)
                    crop_grid_X = crop_grid_size
                    crop_grid_Y = crop_grid_size
                cropped_image, cropped_mask, upper_left, annotated_image = apply_CP_flow(image, inst_mask, crop_grid_X, crop_grid_Y)
                model_input_image = cropped_image
                model_input_mask = cropped_mask
            else: # No crop & paste
                cropped_image, cropped_mask, upper_left, annotated_image = full_image_crop(image, inst_mask)
                model_input_image = cropped_image
                model_input_mask = cropped_mask

            # Collect instance data
            crop_and_pastes.append(crop_and_paste)
            cropped_images.append(cropped_image)
            cropped_masks.append(cropped_mask)
            upper_lefts.append(upper_left)
            annotated_images.append(annotated_image)
            crop_grid_Xs.append(inpaint_condition.crop_grid_X[idx])
            crop_grid_Ys.append(inpaint_condition.crop_grid_Y[idx])
            crop_ratios.append(inpaint_condition.crop_ratio[idx])
            poisson_blends.append(inpaint_condition.poisson_blend[idx])
            shift_valuess.append(inpaint_condition.shift_values[idx])
            rotation_angles.append(inpaint_condition.rotation_angle[idx])
            morph_operations.append(inpaint_condition.morph_operation[idx])

            # Always make sure model's input image / mask size is 512x512
            if model_input_image.size != (512, 512): 
                model_input_image = model_input_image.resize((512, 512), Image.Resampling.BICUBIC)
                model_input_mask = model_input_mask.resize((512, 512), Image.Resampling.BICUBIC)
                
            # Preprocess image / mask
            norm_image, norm_mask = _preprocess(model_input_image, model_input_mask)
            # Preprocess Image & Mask
            if norm_mask.ndim == 2:
                norm_mask = norm_mask[:, :, np.newaxis]
            if norm_image.ndim == 2:   # Some black images will be loaded as 'L' mode
                norm_image = norm_image[:, :, np.newaxis]
            if norm_image.ndim != 3 or norm_mask.ndim != 3:
                log.warning(f"Image or mask ndim != 3! Image: {image_filename} (shape={norm_image.shape}), Mask: {mask_filename} (shape={norm_mask.shape})")
            image_tensor = torch.from_numpy(norm_image).permute(2, 0, 1) # [C, H, W]
            mask_tensor = torch.from_numpy(norm_mask).permute(2, 0, 1) # [C, H, W]
            image_filenames.append(inpaint_condition.image_filename[idx])
            mask_filenames.append(inpaint_condition.mask_filename[idx])
            anomaly_names.append(inpaint_condition.anomaly_type[idx])
            model_input_images.append(image_tensor)
            model_input_masks.append(mask_tensor)
            original_images.append(image)
            original_masks.append(mask)
            indice.append(inpaint_condition.index[idx])
            instance_indices_per_image[-1].append(all_instance_idx)
            all_instance_idx += 1

    # Redundant values for anomaly
    H, W = 512, 512
    full_instance_batch['fps'] = torch.tensor([1] * B).cuda()
    full_instance_batch['image_size'] = torch.tensor([H, W, H, W]).cuda()
    full_instance_batch["num_frames"] = torch.tensor([1] * B).cuda()
    full_instance_batch["padding_mask"] = torch.zeros(B, 1, H, W, dtype=torch.bfloat16).cuda()
    full_instance_batch["t5_text_mask"] = torch.ones(B, 512, dtype=torch.bfloat16).cuda()
    full_instance_batch["neg_t5_text_mask"] = torch.ones(B, 512, dtype=torch.bfloat16).cuda()

    full_instance_batch["original_image"] = original_images
    full_instance_batch["original_mask"] = original_masks
    full_instance_batch["images"] = torch.stack(model_input_images).to("cuda") # (B, C, 1, H, W)
    full_instance_batch["mask"] = torch.stack(model_input_masks).to("cuda")   # (B, 1, H, W)

    # Basic information
    full_instance_batch['image_filename'] = image_filenames
    full_instance_batch['mask_filename'] = mask_filenames
    full_instance_batch['name'] = anomaly_names
    
    # Captions
    captions = []
    for anomaly_type in anomaly_names:
        sample_name, anomaly_name = anomaly_type.split("+")
        caption = ANOMALY_TEXT_TEMPLATE.format(sample_name, anomaly_name, '*')
        captions.append(caption)
    full_instance_batch['caption'] = captions

    # Inpainting. Prepare text embedding for anomaly generation
    with torch.no_grad():
        _, latent_state, condition = model.pipe.get_data_and_condition(full_instance_batch)
    full_instance_batch["guided_image"]  = latent_state
    full_instance_batch["guided_mask"], _  = model.pipe._get_guided_mask_and_weight(full_instance_batch["guided_image"], full_instance_batch['mask'])
    full_instance_batch["guided_mask"] = (1 - full_instance_batch["guided_mask"].to(torch.bfloat16)) # For replacement trick, 0 means using predicted value, 1 means using original value

    # Crop & Paste flow
    full_instance_batch["crop_grid_X"] = crop_grid_Xs
    full_instance_batch["crop_grid_Y"] = crop_grid_Ys
    full_instance_batch["upper_lefts"] = upper_lefts
    full_instance_batch["cropped_image"] = cropped_images
    full_instance_batch["cropped_mask"] = cropped_masks
    full_instance_batch["annotated_image"] = annotated_images
    full_instance_batch["dummy"] = [False] * len(original_images)

    full_instance_batch["index"] = indice
    # Split full_instance_batch into instance_batch for iterative generation
    # max_instances = max(len(indices) for indices in instance_indices_per_image)

    local_max_instances = max(len(indices) for indices in instance_indices_per_image)
    max_instances_tensor = torch.tensor([local_max_instances], device="cuda")

    if dist.is_initialized():
        dist.all_reduce(max_instances_tensor, op=dist.ReduceOp.MAX)

    max_instances = int(max_instances_tensor.item())
    # Placeholder objects for padding
    placeholder = {
        "image"        : torch.zeros_like(model_input_images[0]),
        "mask"         : torch.zeros_like(model_input_masks[0]),
        "orig_image"   : Image.new("RGB", original_images[0].size),
        "orig_mask"    : Image.new("L",   original_masks[0].size),
        "image_path"   : "",
        "mask_path"    : "",
        "anomaly_name" : anomaly_names[0],
        "caption"      : captions[0],
        "upper_left"   : (0, 0),
        "crop_image"   : Image.new("RGB", original_images[0].size),
        "crop_mask"    : Image.new("L",   original_masks[0].size),
        "annotated_image": Image.new("RGB", original_images[0].size),
        "dummy"        : True,
        "index"        : -1
    }

    # Keys that never change between instance batches
    static_keys = full_instance_batch.keys() - {
        "images", "mask",
        "cropped_image", "cropped_mask",
        "original_image", "original_mask",
        "upper_lefts",
        "image_filename", "mask_filename",
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

        instance_batch = {
            **static_fields,
            "images"         : torch.stack(
                gather(model_input_images, "image")
            ).to("cuda").unsqueeze(2),
            "mask"           : torch.stack(
                gather(model_input_masks, "mask")
            ).to("cuda"),
            "original_image" : gather(original_images,  "orig_image"),
            "original_mask"  : gather(original_masks,   "orig_mask"),
            "image_filename" : gather(image_filenames,  "image_path"),
            "mask_filename"  : gather(mask_filenames,   "mask_path"),
            "name"           : gather(anomaly_names,    "anomaly_name"),
            "caption"        : gather(captions,         "caption"),
            "upper_lefts"    : gather(upper_lefts,      "upper_left"),
            "cropped_image"  : gather(cropped_images,   "crop_image"),
            "cropped_mask"   : gather(cropped_masks,    "crop_mask"),
            "annotated_image": gather(annotated_images, "annotated_image"),
            "dummy"          : gather([False] * len(original_images),    "dummy"),
            "index"          : gather(indice,    "index"),
        }

        instance_batches.append(instance_batch)

    # Newest batch first, so the dummy samples will be process first
    instance_batches.reverse()
    return instance_batches


def _split_mask(mask, max_k=5):
    assert max_k>0, "Maximum number of instance in the mask shoulb be >0."
    if max_k==1:
        return [mask]
    
    # Convert mask to binary numpy array
    mask_np = np.array(mask)

    # Threshold to ensure binary
    _, binary_mask = cv2.threshold(mask_np, 0, 255, cv2.THRESH_BINARY)

    # Compute connected components and stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    instance_masks = []

    if num_labels <= 1:
        # No foreground pixels (label 0 is background)
        instance_masks = []

    else:
        num_components = num_labels - 1  # Exclude background

        if num_components <= max_k:
            # Each component becomes its own mask
            for label in range(1, num_labels):
                mask_i = np.where(labels == label, 255, 0).astype(np.uint8)
                instance_masks.append(Image.fromarray(mask_i))
        else:
            # Too many components, cluster centroids into K groups
            component_centroids = centroids[1:]  # skip background centroid at index 0
            kmeans = KMeans(n_clusters=max_k, random_state=0).fit(component_centroids)
            cluster_labels = kmeans.labels_

            for cluster_id in range(max_k):
                mask_k = np.zeros_like(mask_np, dtype=np.uint8)
                labels_in_cluster = np.where(cluster_labels == cluster_id)[0] + 1  # +1 to map back to component label ids
                for lbl in labels_in_cluster:
                    mask_k[labels == lbl] = 255
                instance_masks.append(Image.fromarray(mask_k))
    return instance_masks


def count_mask_instances(mask):
    """
    Count the number of connected components (instances) in a mask.
    Shared utility for both single-view and multi-view datasets.
    
    Args:
        mask: PIL Image or numpy array (grayscale mask)
    
    Returns:
        int: Number of instances (excluding background)
    """
    mask_np = np.array(mask)
    _, binary_mask = cv2.threshold(mask_np, 0, 255, cv2.THRESH_BINARY)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    return num_labels - 1  # Exclude background


def _postprocess(inpaint_condition, diffusion_data_batch):
    # Reconstruct images
    reconstructed_images = diffusion_data_batch['denoised_result'][:, :, 0, :, :]
    reconstructed_images = ((reconstructed_images + 1).clamp(0, 2)) / 2.0
    reconstructed_images = [tensor_to_pil_image(reconstructed_images[i], diffusion_data_batch['original_image'][i].mode) \
                                                   for i in range(len(reconstructed_images))]
    
    # If crop & paste is enabled, paste back the reconstructed images
    for i in range(len(reconstructed_images)):
        if inpaint_condition.crop_and_paste[i]:
            g_image = diffusion_data_batch['original_image'][i]
            r_image = reconstructed_images[i]
            c_image = diffusion_data_batch['cropped_image'][i]
            c_mask = diffusion_data_batch['cropped_mask'][i]
            upper_left = diffusion_data_batch['upper_lefts'][i]
            poisson_blend = inpaint_condition.poisson_blend[i]

            CP_reconstructed_image = g_image.copy()
            paste_back(CP_reconstructed_image, r_image, c_image, c_mask, upper_left, poisson_blend)
            reconstructed_images[i] = CP_reconstructed_image
    return reconstructed_images

def _replace_batch_input(batch, prev_batch, new_image_pil, model):
    for i in range(len(batch["images"])):
        if prev_batch["dummy"][i]: # skip for dummy sample
            continue
        batch["original_image"][i] = new_image_pil[i] # 0~1
        offsets = batch["upper_lefts"][i]
        crop_size = batch["cropped_image"][i].size
        crop_image = new_image_pil[i].crop((offsets[0], offsets[1], offsets[0]+crop_size[0],offsets[1]+crop_size[1]))
        crop_image = crop_image.resize((512, 512), Image.Resampling.BICUBIC).convert("RGB")
        batch["images"][i] = (to_tensor(crop_image).cuda().unsqueeze(1) * 2.0 - 1.0) # -1~1
    # Prepare text embedding
    with torch.no_grad():
        _, latent_state, condition = model.pipe.get_data_and_condition(batch)
    batch["guided_image"]  = latent_state
    batch["guided_mask"], _  = model.pipe._get_guided_mask_and_weight(batch["guided_image"], batch['mask'])
    batch["guided_mask"] = (1 - batch["guided_mask"].to(torch.bfloat16)) # For replacement trick, 0 means using predicted value, 1 means using original value

def compute_psnr_in_mask(sdg_image, ok_image, mask):
    """
    Compute the PSNR between sdg_image and ok_image only for pixels where mask==255.
    Both images must be 8-bit (0-255) and have the same shape.
    Accepts PIL.Image.Image objects.
    """
    sdg_image = np.array(sdg_image)
    ok_image = np.array(ok_image)
    mask = np.array(mask)

    mask_bool = mask.astype(bool)
    sdg_region = sdg_image[mask_bool]
    ok_region  = ok_image[mask_bool]
    
    if sdg_region.size == 0 or ok_region.size == 0:
        log.warning("No pixels found in the mask region.")
        return None
    
    mse = np.mean((sdg_region.astype(np.float64) - ok_region.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX) - 10 * np.log10(mse)
    return psnr

def tensor_to_pil_image(tensor, mode = None):
    """ Expect tensor with 0~1 value ranges """
    if tensor.min() < 0 or tensor.max() > 1:
        raise ValueError("Tensor value ranges must be 0~1")
    if tensor.ndim == 5: # B, C, T, H, W
        tensor = tensor[:, :, 0, :, :]
    if tensor.shape[0] == 3: # N, Ch, H, W
        tensor = tensor.transpose(0, 1).transpose(1, 2).squeeze(-1)# [H', W', C]
    tensor = tensor.to(torch.float32).to("cpu").numpy()
    tensor = (tensor * 255).astype(np.uint8)
    image = Image.fromarray(tensor)
    if mode is not None:
        image = image.convert(mode)
    return image

def save_images(images, save_dir, filename):
    for i, image in enumerate(images):
        image.save(os.path.join(save_dir, f"{filename}_{i}.png"))
