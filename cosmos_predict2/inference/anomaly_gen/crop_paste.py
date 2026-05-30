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

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from cosmos_predict2.inference.anomaly_gen.mask_augmentation import enlarge_mask
from imaginaire.utils import log
import cv2


def best_crop(image, mask, grid_X=512, grid_Y=512):
    if mask.mode != "L":
        binary_mask = mask.convert("L")
        binary_mask = np.array(binary_mask)
    else:
        binary_mask = np.array(mask)

    mask_region_y, mask_region_x = np.nonzero(binary_mask >= 127)

    if mask_region_x.size == 0 or mask_region_y.size == 0:
        raise ValueError("No valid mask region found.")

    LX, RX = min(mask_region_x), max(mask_region_x)
    UY, BY = min(mask_region_y), max(mask_region_y)
    
    center_X, center_Y = (np.around((LX + RX) / 2)).astype(int), (np.around((UY + BY) / 2)).astype(int)
    log.info(f"X: ({LX}-{RX}), Y: ({BY}-{UY}). Center: ({center_X}, {center_Y})")

    # Get image dimensions
    H, W = binary_mask.shape

    # Compute crop boundaries ensuring full 512x512 crop unless at edges
    crop_LX = max(0, min(center_X - grid_X // 2, W - grid_X))
    crop_RX = min(W, crop_LX + grid_X)
    crop_UY = max(0, min(center_Y - grid_Y // 2, H - grid_Y))
    crop_BY = min(H, crop_UY + grid_Y)
    crop_width = crop_RX - crop_LX
    crop_height = crop_BY - crop_UY

    # Draw crop annotation
    # Try to get a default font, fallback gracefully if not available
    annotated_image = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(annotated_image, "RGBA")

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    mask_text = f"Mask BBOX: ({LX},{UY})-({RX},{BY})"
    text_bg_width = draw.textlength(mask_text, font=font) + 10  # Add padding
    draw.rectangle(((LX, UY-30), (LX+text_bg_width, UY)), fill=(255, 255, 255, 180))
    draw.text((LX+5, UY-25), mask_text, fill=(255, 0, 0), font=font)

    # Add crop region coordinates and dimensions
    crop_text = f"Crop: ({crop_LX},{crop_UY})-({crop_RX},{crop_BY}), {crop_width}x{crop_height}"
    text_bg_width = draw.textlength(crop_text, font=font) + 10  # Add padding
    draw.rectangle(((crop_LX, crop_UY-30), (crop_LX+text_bg_width, crop_UY)), fill=(255, 255, 255, 180))
    draw.text((crop_LX+5, crop_UY-25), crop_text, fill=(0, 128, 0), font=font)

    draw.rectangle(((LX, UY), (RX, BY)), outline=(255, 0, 0, 127), width=3)  # Mask region, in red
    draw.rectangle(((crop_LX, crop_UY), (crop_RX, crop_BY)), outline=(0, 255, 0, 127), width=3)  # Cropped area, in green

    # Crop image and mask    
    cropped_image = image.crop((crop_LX, crop_UY, crop_RX, crop_BY))
    cropped_mask = mask.crop((crop_LX, crop_UY, crop_RX, crop_BY))
    return cropped_image, cropped_mask, (crop_LX, crop_UY), annotated_image


def full_image_crop(image, mask):
    if mask.mode != "L":
        binary_mask = mask.convert("L")
        binary_mask = np.array(binary_mask)
    else:
        binary_mask = np.array(mask)

    mask_region_y, mask_region_x = np.nonzero(binary_mask >= 127)

    if mask_region_x.size == 0 or mask_region_y.size == 0:
        raise ValueError("No valid mask region found.")

    LX, RX = min(mask_region_x), max(mask_region_x)
    UY, BY = min(mask_region_y), max(mask_region_y)

    center_X, center_Y = (np.around((LX + RX) / 2)).astype(int), (np.around((UY + BY) / 2)).astype(int)
    log.info(f"X: ({LX}-{RX}), Y: ({BY}-{UY}). Center: ({center_X}, {center_Y})")

    # Get image dimensions
    H, W = binary_mask.shape

    crop_LX = 0
    crop_RX = W
    crop_UY = 0
    crop_BY = H
    crop_width = W
    crop_height = H

    annotated_image = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(annotated_image, "RGBA")

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    mask_text = f"Mask BBOX: ({LX},{UY})-({RX},{BY})"
    text_bg_width = draw.textlength(mask_text, font=font) + 10  # Add padding
    draw.rectangle(((LX, UY-30), (LX+text_bg_width, UY)), fill=(255, 255, 255, 180))
    draw.text((LX+5, UY-25), mask_text, fill=(255, 0, 0), font=font)

    # Add crop region coordinates and dimensions
    crop_text = f"Crop: ({crop_LX},{crop_UY})-({crop_RX},{crop_BY}), {crop_width}x{crop_height}"
    text_bg_width = draw.textlength(crop_text, font=font) + 10  # Add padding
    draw.rectangle(((crop_LX, crop_UY-30), (crop_LX+text_bg_width, crop_UY)), fill=(255, 255, 255, 180))
    draw.text((crop_LX+5, crop_UY-25), crop_text, fill=(0, 128, 0), font=font)

    draw.rectangle(((LX, UY), (RX, BY)), outline=(255, 0, 0, 127), width=3)  # Mask region, in red
    draw.rectangle(((crop_LX, crop_UY), (crop_RX, crop_BY)), outline=(0, 255, 0, 127), width=3)  # Cropped area, in green

    return image, mask, (0, 0), annotated_image


def paste_back(input_image, recon_image, cropped_image, cropped_mask, offset, poisson_blend = False):
    # Resize reconstructed image to cropped size
    if recon_image.size != cropped_image.size:
        recon_cropped_image = recon_image.resize(cropped_image.size, Image.Resampling.BICUBIC) 
    else:
        recon_cropped_image = recon_image
    # Enlarge mask area to avoid anomalies growing out of the image
    enlarged_mask = enlarge_mask(cropped_mask)
    # Paste back
    ## Paste anomaly back to cropped real image
    tmp_cropped_image = cropped_image.copy()
    tmp_recon_cropped_image = recon_cropped_image.copy()
    tmp_enlarged_mask = enlarged_mask.copy()
    if poisson_blend: # Perform seemless cloning
        # Convert PIL images to numpy arrays
        image_mode = tmp_recon_cropped_image.mode
        np_recon_cropped_image = np.array(tmp_recon_cropped_image.convert("RGB"))
        np_tmp_cropped_image = np.array(tmp_cropped_image.convert("RGB"))
        np_enlarged_mask = np.array(tmp_enlarged_mask)
        # Find center of mask
        mask_region_y, mask_region_x = np.nonzero(np_enlarged_mask >= 127)
        if len(mask_region_x) == 0 or len(mask_region_y) == 0:
            log.warning("Empty mask detected for Poisson blending; skipping blending.")
            input_image.paste(tmp_cropped_image, offset)
            return
        LX, RX = min(mask_region_x), max(mask_region_x)
        UY, BY = min(mask_region_y), max(mask_region_y)        
        center = (np.around((LX + RX) / 2)).astype(int), (np.around((UY + BY) / 2)).astype(int)
        log.info(f"Poisson Blend: Center = {center}")
        # Perform seamless cloning
        tmp_cropped_image = cv2.seamlessClone(np_recon_cropped_image, np_tmp_cropped_image, np_enlarged_mask, center, cv2.NORMAL_CLONE)
        # Convert back to PIL image
        tmp_cropped_image = Image.fromarray(tmp_cropped_image).convert(image_mode)
    else:
        tmp_cropped_image.paste(recon_cropped_image, (0, 0), enlarged_mask)
    ## Paste entire cropped image back to input image
    input_image.paste(tmp_cropped_image, offset)
    return

def apply_CP_flow(image, mask, crop_grid_X, crop_grid_Y):
    log.info(f"Using C & P Flow.")
    cropped_image, cropped_mask, upper_left, annotated_image = best_crop(image, mask, crop_grid_X, crop_grid_Y)
    return cropped_image, cropped_mask, upper_left, annotated_image