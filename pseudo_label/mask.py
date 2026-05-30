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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def cluster_mask(mask: Image.Image, eps=0.2, min_samples=5):
    mask_array = np.array(mask) > 127

    # Shortcut when the mask is already well-clustered.
    contours, _ = cv2.findContours(
        mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    elif len(contours) == 1:
        return [mask]

    # Cluster the mask by DBSCAN.
    height, width = mask.height, mask.width

    # Get all x, y coordinates of white pixels in the mask
    data = np.array(mask_array.nonzero()).T
    scaled = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(scaled)

    # Number of clusters
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    # Index the label color for each cluster
    clust_masks = np.zeros((n_clusters, height, width), dtype=np.uint8)
    # Loop over labels/clusters
    for z in range(0, len(db.labels_)):
        if not db.labels_[z] == -1:
            # Create a binary mask for each cluster
            clust_masks[db.labels_[z]][data[z][0], data[z][1]] = 255
    clust_masks = [Image.fromarray(m) for m in clust_masks]
    if len(clust_masks) == 0:
        # DBSCAN will fail when only 1 group of pixels is detected.
        clust_masks = [mask]
    return clust_masks


def binary_mask_to_rle(binary_mask: np.ndarray):
    """Convert a binary mask to RLE (Run-Length Encoding) format.

    Reference: https://stackoverflow.com/questions/49494337/encode-numpy-array-using-uncompressed-rle-for-coco-dataset
    """
    if binary_mask.dtype != np.bool_:
        raise ValueError("Input binary_mask must be of type np.bool_")
    if len(binary_mask.shape) != 2:
        raise ValueError("Input binary_mask must be a 2D array")

    rle = {"counts": [], "size": list(binary_mask.shape)}
    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))
    # note that the odd counts are always the numbers of zeros
    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))
    rle["counts"] = lengths.tolist()
    return rle


def coco_encode_rle(
    uncompressed_rle: typing.Dict[str, typing.Any],
) -> typing.Dict[str, typing.Any]:
    from pycocotools import mask as mask_utils

    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle


def coco_decode_rle(rle: typing.Dict[str, typing.Any]) -> np.ndarray:
    from pycocotools import mask as mask_utils

    return np.ascontiguousarray(mask_utils.decode(rle))


def compute_sam2_mask_prompt(
    mask: Image.Image, strength: float = 1.0, device=None
) -> typing.Optional[torch.Tensor]:
    if strength <= 0.0:
        return None

    binary_mask = (np.array(mask) > 127).astype(np.uint8)
    mask_prompt = torch.from_numpy(binary_mask.astype(np.float32)).to(device)
    
    # Simply applying the strength to the mask_prompt works better than the
    # distance map in the experiments.
    mask_prompt = torch.where(mask_prompt <= 0, -1.0 * strength, strength)

    # SAM2 only accepts 256x256 mask_input.
    mask_prompt = F.interpolate(
        mask_prompt.unsqueeze(0).unsqueeze(0),
        (256, 256),
        mode="bilinear",
        align_corners=False,
    )
    return mask_prompt


def post_process_sam2_mask(
    refined_mask: np.ndarray,
    original_mask: np.ndarray,
    image_height: int,
    image_width: int,
    cropped_bbox: typing.Tuple[int, int, int, int],
    fallback_ratio: float = 0.5,
) -> Image.Image:
    assert refined_mask.ndim == 2
    restored_mask = np.zeros((image_height, image_width), dtype=refined_mask.dtype)
    restored_mask[
        cropped_bbox[1] : cropped_bbox[3],
        cropped_bbox[0] : cropped_bbox[2],
    ] = refined_mask
    refined_mask = restored_mask
    instance_mask_array = (original_mask > 127).astype(refined_mask.dtype)
    intersection_array = refined_mask * instance_mask_array
    if np.sum(intersection_array) == 0:
        intersection_array = refined_mask
    refined_mask = intersection_array

    if fallback_ratio > 0.0:
        original_area = np.sum(instance_mask_array)
        refined_area = np.sum(refined_mask)
        if refined_area < fallback_ratio * original_area:
            refined_mask = instance_mask_array

    refined_mask = (refined_mask * 255).astype(np.uint8)
    return Image.fromarray(refined_mask)
