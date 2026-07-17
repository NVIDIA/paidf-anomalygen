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

from torchvision.transforms import functional as tF
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image

from cosmos_predict2.utils.random import secure_randint, secure_random
from imaginaire.utils import log
            
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if secure_random() < self.flip_prob:
            image = tF.hflip(image)
            if target is not None:
                target = tF.hflip(target)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if secure_random() < self.flip_prob:
            image = tF.vflip(image)
            if target is not None:
                target = tF.vflip(target)
        return image, target

class RandomRotation(object):
    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, image, target=None):
        random_angle = secure_randint(-self.max_angle, self.max_angle)
        image = tF.rotate(image, random_angle,
                          interpolation=transforms.InterpolationMode.BILINEAR)
        if target is not None:
            target = tF.rotate(target, random_angle,
                               interpolation=transforms.InterpolationMode.BILINEAR)
            if isinstance(target, Image.Image):
                target = Image.fromarray(
                    np.where(np.array(target) >= 128, 255, 0).astype(np.uint8),
                    mode=target.mode,
                )
            else:
                hi = 1.0 if target.max() <= 1.0 else 255
                target = (target >= hi / 2).to(target.dtype) * hi
        return image, target

class RandomRotation90(object):
    """Randomly rotate by 90 or 270 degrees (lossless)."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target=None):
        if secure_random() < self.p:
            angle = 90 if secure_random() < 0.5 else 270
            image = tF.rotate(image, angle)
            if target is not None:
                target = tF.rotate(target, angle)
        return image, target


class Compose(object):
    def __init__(self):
        self.transforms = [
                    RandomVerticalFlip(0.5),
                    RandomHorizontalFlip(0.5),
                    RandomRotation(20)
                ]

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
        
class Random_Transform(object):
    def __init__(self):
        self.transform=Compose()
        self.resize_512=transforms.Resize((512,512))
        self.resize_256=transforms.Resize((256,256))
        
    def __call__(self, image, mask):

        C, H, W = image.shape

        """
        if H < 512:
            image=self.resize_512(image)
            mask=self.resize_512(mask)
        """

        image, mask = self.transform(image, mask)
        mask2 = (mask > 0.5).int()
        x = torch.nonzero(mask2)

        if x.numel() == 0: # No mask is given
            return image, mask.float()

        x1, x2 = x[:, 1].min(), x[:, 1].max()
        y1, y2 = x[:, 2].min(), x[:, 2].max()
        crop_x1 = secure_randint(0, x1 - 1)
        crop_y1 = secure_randint(0, y1 - 1)
        crop_x2 = secure_randint(x2, image.size(-1))
        crop_y2 = secure_randint(y2, image.size(-1))

        cropped_image =  image[:, crop_x1:crop_x2, crop_y1:crop_y2]
        cropped_mask = mask[:, crop_x1:crop_x2, crop_y1:crop_y2]

        cropped_mask[cropped_mask < 0.5] = 0.0
        cropped_mask[cropped_mask >= 0.5] = 1.0

        return cropped_image, cropped_mask.float()


class Random_Crop(object):
    def __init__(self, crop_size=512):
        self.crop_size = crop_size
        self.transform=Compose()
        self.resize_512=transforms.Resize((512,512))
        self.resize_256=transforms.Resize((256,256))
        
    def __call__(self, image, mask):
        """
        Input: 
            * Image: torch.tensor in # [3, H, W]
            * Mask: torch.tensor in # [1, H, W]
        Output:
            * cropped_image: torch.tensor in # [3, H', W']
            * cropped_mask: torch.tensor in # [1, H', W']
        """
        if isinstance(image, Image.Image):
            H = image.height
            W = image.width
        else:
            C, H, W = image.shape

        # Rotate + FLIP
        aug_image, aug_mask = self.transform(image, mask)

        # Find masked position
        if isinstance(aug_mask, Image.Image):
            aug_mask_arr = np.array(aug_mask)
        else:
            aug_mask_arr = aug_mask
        if aug_mask_arr.ndim == 2:
            aug_mask_arr = np.expand_dims(aug_mask_arr, axis=0)
        x = np.transpose(np.nonzero(aug_mask_arr))
        if x.size == 0:
            log.warning("No masked pixels after augmentation. Use original images.")
            return aug_image, aug_mask

        x1 = int(x[:, 2].min())
        y1 = int(x[:, 1].min())
        
        # Crop around mask region
        crop_offset = self.crop_size // 4 # (128 for 512 crop, 96 for 384 crop)
        crop_x1 = secure_randint(max(0, x1 - crop_offset), x1)
        crop_x2 = min(crop_x1 + self.crop_size, W)
        crop_y1 = secure_randint(max(0, y1 - crop_offset), y1)
        crop_y2 = min(crop_y1 + self.crop_size, H)

        # Always crop crop_size x crop_size region
        if isinstance(aug_image, Image.Image):
            cropped_image = aug_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            cropped_mask = aug_mask.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        else:
            cropped_image = aug_image[:, crop_y1:crop_y2, crop_x1:crop_x2]
            cropped_mask = aug_mask[:, crop_y1:crop_y2, crop_x1:crop_x2]
            cropped_mask = cropped_mask.astype(np.float32)
        return cropped_image, cropped_mask


class RandomRatioCrop(object):
    def __init__(self, crop_size=512, ratio_range=None):
        self.final_crop_size = crop_size
        self.transform = Compose()
        self.ratio_range = tuple(ratio_range) if ratio_range is not None else (1.5, 16.0)

    def __call__(self, image, mask):
        """
        Crop a window centered on the mask centroid, with a random zoom ratio
        and centroid jitter over the full valid range (defect always stays visible).
        Mirrors the inference crop and paste mechanism so train and inference see
        defects at consistent scales. Note that current default ratio_range is just
        an intuitive choice and worth further investigation.

        Input:
            * image: torch.tensor [C, H, W]
            * mask:  torch.tensor [1, H, W]
        Output:
            * cropped_image: torch.tensor [C, H', W']
            * cropped_mask:  torch.tensor [1, H', W']
        """
        if isinstance(image, Image.Image):
            H = image.height
            W = image.width
        else:
            C, H, W = image.shape

        # Flip + rotate
        aug_image, aug_mask = self.transform(image, mask)

        # Find mask bounding box
        if isinstance(aug_mask, Image.Image):
            aug_mask_arr = np.array(aug_mask)
        else:
            aug_mask_arr = aug_mask
        if aug_mask_arr.ndim == 2:
            aug_mask_arr = np.expand_dims(aug_mask_arr, axis=0)
        x = np.transpose(np.nonzero(aug_mask_arr))
        if x.size == 0:
            log.warning("No masked pixels after augmentation. Use original images.")
            return aug_image, aug_mask

        x1, x2 = x[:, 2].min(), x[:, 2].max()
        y1, y2 = x[:, 1].min(), x[:, 1].max()

        # Mask centroid
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        defect_size = int(max(x2 - x1, y2 - y1))

        # Random crop size proportional to defect size
        ratio_min, ratio_max = self.ratio_range
        ratio = ratio_min + secure_random() * (ratio_max - ratio_min)
        crop_size = max(int(defect_size * ratio), self.final_crop_size // 16)
        # Floor at 1/16 of final crop size to avoid extremely small crops
        # However, this value is just an intuitive choice and worth further investigation

        # Randomly shift centroid over the full valid range (defect stays fully visible).
        jitter = max((crop_size - defect_size) // 2, 0)
        if jitter > 0:
            cx += secure_randint(-jitter, jitter)
            cy += secure_randint(-jitter, jitter)

        # Crop boundaries centered on (cx, cy), clamped to image
        half = crop_size // 2
        crop_x1 = max(0, min(cx - half, W - crop_size))
        crop_y1 = max(0, min(cy - half, H - crop_size))
        crop_x2 = min(W, crop_x1 + crop_size)
        crop_y2 = min(H, crop_y1 + crop_size)

        # Apply crop
        if isinstance(aug_image, Image.Image):
            cropped_image = aug_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            cropped_mask = aug_mask.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        else:
            cropped_image = aug_image[:, crop_y1:crop_y2, crop_x1:crop_x2]
            cropped_mask = aug_mask[:, crop_y1:crop_y2, crop_x1:crop_x2]
            cropped_mask = cropped_mask.astype(np.float32)
        return cropped_image, cropped_mask


class Erosion2d(nn.Module):

    def __init__(self, m=1):
        super(Erosion2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=1e9)
        channel = self.unfold(x_pad).view(batch_size, c, -1, h, w)
        result = torch.min(channel, dim=2)[0]
        return result

def erosion(x, m=1):
    b, c, h, w = x.shape
    x_pad = F.pad(x, pad=[m, m, m, m], mode='constant', value=1e9)
    channel = nn.functional.unfold(x_pad, 2 * m + 1, padding=0, stride=1).view(b, c, -1, h, w)
    result = torch.min(channel, dim=2)[0]
    return result

class Dilation2d(nn.Module):

    def __init__(self, m=1):
        super(Dilation2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=-1e9)
        channel = self.unfold(x_pad).view(batch_size, c, -1, h, w)
        result = torch.max(channel, dim=2)[0]
        return result


def dilation(x, m=1):
    b, c, h, w = x.shape
    x_pad = F.pad(x, pad=[m, m, m, m], mode='constant', value=-1e9)
    channel = nn.functional.unfold(x_pad, 2 * m + 1, padding=0, stride=1).view(b, c, -1, h, w)
    result = torch.max(channel, dim=2)[0]
    return result

class MultiViewRandomCrop(object):
    """
    Multi-view version of Random_Crop that applies the SAME augmentation 
    to all views simultaneously to maintain cross-view consistency.
    
    Works with views in [T, C, H, W] format by treating T as batch dimension.
    torchvision.transforms.functional operations (hflip, vflip, rotate) 
    automatically broadcast over the first dimension, ensuring all views 
    get the same transformation parameters.
    """
    def __init__(self, crop_size=512):
        self.crop_size = crop_size
        self.transform = Compose()  # Contains RandomFlip + RandomRotation
        self.resize_512 = transforms.Resize((512, 512))
        self.resize_256 = transforms.Resize((256, 256))
        
    def __call__(self, views, mask):
        """
        Apply random augmentation (flip + rotation + crop) to multi-view images.
        
        Input: 
            * views: torch.tensor [T, C, H, W] - multiple views
            * mask: torch.tensor [1, H, W] - shared mask across views
        Output:
            * cropped_views: torch.tensor [T, C, H', W']
            * cropped_mask: torch.tensor [1, H', W']
        """
        T, C, H, W = views.shape
        
        # Apply rotation + flip to ALL views and mask simultaneously
        # The Compose transform will apply the same random parameters to all views
        aug_views, aug_mask = self.transform(views, mask)
        
        # Find masked position (from augmented mask)
        x = torch.nonzero(aug_mask)
        if x.numel() == 0:
            print(f"No masked pixels after augmentation. Use original.")
            return aug_views, aug_mask.float()

        x1, x2 = x[:, 2].min(), x[:, 2].max()
        y1, y2 = x[:, 1].min(), x[:, 1].max()
        
        # Crop around mask region (same crop for all views)
        crop_offset = self.crop_size // 4 # (128 for 512 crop, 96 for 384 crop)
        crop_x1 = secure_randint(max(0, x1 - crop_offset), x1)
        crop_x2 = min(crop_x1 + self.crop_size, W)
        crop_y1 = secure_randint(max(0, y1 - crop_offset), y1)
        crop_y2 = min(crop_y1 + self.crop_size, H)

        # Apply same crop to all views
        cropped_views = aug_views[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_mask = aug_mask[:, crop_y1:crop_y2, crop_x1:crop_x2]

        return cropped_views, cropped_mask.float()


class MultiViewRandomRatioCrop(object):
    """
    Multi-view version of RandomRatioCrop that applies the SAME augmentation
    to all views simultaneously to maintain cross-view consistency.

    Crops a window centered on the mask centroid with a random zoom ratio
    and centroid jitter, mirroring the inference crop/paste mechanism.

    Works with views in [T, C, H, W] format by treating T as batch dimension.
    torchvision.transforms.functional operations (hflip, vflip, rotate)
    automatically broadcast over the first dimension, ensuring all views
    get the same transformation parameters.
    """
    def __init__(self, crop_size=512, ratio_range=None):
        self.final_crop_size = crop_size
        self.transform = Compose()
        self.ratio_range = tuple(ratio_range) if ratio_range is not None else (1.5, 16.0)

    def __call__(self, views, mask):
        """
        Apply random augmentation (flip + rotation + random ratio crop) to
        multi-view images.

        Input:
            * views: torch.tensor [T, C, H, W] - multiple views
            * mask:  torch.tensor [1, H, W] - shared mask across views
        Output:
            * cropped_views: torch.tensor [T, C, H', W']
            * cropped_mask:  torch.tensor [1, H', W']
        """
        T, C, H, W = views.shape

        # Apply rotation + flip to ALL views and mask simultaneously
        aug_views, aug_mask = self.transform(views, mask)

        # Find mask bounding box
        x = torch.nonzero(aug_mask)
        if x.numel() == 0:
            print(f"No masked pixels after augmentation. Use original images.")
            return aug_views, aug_mask.float()

        x1, x2 = x[:, 2].min().item(), x[:, 2].max().item()
        y1, y2 = x[:, 1].min().item(), x[:, 1].max().item()

        # Mask centroid
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        defect_size = max(x2 - x1, y2 - y1)

        # Random crop size proportional to defect size
        ratio_min, ratio_max = self.ratio_range
        ratio = ratio_min + secure_random() * (ratio_max - ratio_min)
        crop_size = max(int(defect_size * ratio), self.final_crop_size // 16)

        # Randomly shift centroid over the full valid range (defect stays fully visible)
        jitter = max((crop_size - defect_size) // 2, 0)
        if jitter > 0:
            cx += secure_randint(-jitter, jitter)
            cy += secure_randint(-jitter, jitter)

        # Crop boundaries centered on (cx, cy), clamped to image
        half = crop_size // 2
        crop_x1 = max(0, min(cx - half, W - crop_size))
        crop_y1 = max(0, min(cy - half, H - crop_size))
        crop_x2 = min(W, crop_x1 + crop_size)
        crop_y2 = min(H, crop_y1 + crop_size)

        # Apply same crop to all views
        cropped_views = aug_views[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_mask = aug_mask[:, crop_y1:crop_y2, crop_x1:crop_x2]
        return cropped_views, cropped_mask.float()