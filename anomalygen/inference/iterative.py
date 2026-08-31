# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Iterative multi-instance inpainting orchestration: split a full-image mask into
# per-instance masks, then inpaint each in turn, feeding each result into the next.

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from anomalygen.inference.crop_paste import annotate, best_crop, crop_grid_by_ratio, mask_bbox, paste_back

# (cropped_image, cropped_mask, anomaly_name) -> edited_crop
ModelInpaintFn = Callable[[Image.Image, Image.Image, str], Image.Image]


def split_mask_into_instances(mask: Image.Image, max_k: int = 5) -> List[Image.Image]:
    """Split a binary mask into up to ``max_k`` per-instance masks (CC + KMeans clustering)."""
    if max_k <= 0:
        raise ValueError(f"max_k must be positive, got {max_k}")
    if max_k == 1:
        return [mask]

    mask_np = np.array(mask)
    _, binary = cv2.threshold(mask_np, 0, 255, cv2.THRESH_BINARY)
    num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return []

    num_components = num_labels - 1
    instance_masks: List[Image.Image] = []
    if num_components <= max_k:
        for lbl in range(1, num_labels):
            instance_masks.append(Image.fromarray(np.where(labels == lbl, 255, 0).astype(np.uint8)))
    else:
        component_centroids = centroids[1:]
        cluster_labels = KMeans(n_clusters=max_k, n_init=10, random_state=0).fit(component_centroids).labels_
        for cluster_id in range(max_k):
            m = np.zeros_like(mask_np, dtype=np.uint8)
            for lbl in np.where(cluster_labels == cluster_id)[0] + 1:
                m[labels == lbl] = 255
            instance_masks.append(Image.fromarray(m))

    return instance_masks


def run_iterative_inpaint(
    image: Image.Image,
    mask: Image.Image,
    anomaly_name: str,
    model_inpaint: ModelInpaintFn,
    crop_grid: Tuple[int, int] = (512, 512),
    max_instances: int = 5,
    poisson_blend: bool = False,
    crop_and_paste: bool = True,
    crop_ratio: float | None = None,
    return_artifacts: bool = False,
):
    """Sequentially inpaint each defect instance, feeding each result into the next.

    Crop window per instance: the whole image when ``crop_and_paste=False``; a square
    window of ``max(bbox) * crop_ratio`` when ``crop_ratio`` is set; otherwise the fixed
    ``crop_grid``. With ``return_artifacts=True`` also returns a dict of per-instance PIL
    lists keyed ``cropped_image``, ``cropped_mask``, ``annotated_image``,
    ``mask_cropped_image``.

    ``crop_and_paste=False`` is supported but out of distribution for small defects: training
    always crops ``max(bbox) * ratio`` with ratio in [1.5, 8.0] before the 512 resize, so the
    defect reaches the model at 64-341px across. Taking the whole frame only stays in that range
    while the defect's bbox is >= ~1/8 of the image's longer side; below that it arrives smaller
    than anything trained on.
    """
    instance_masks = split_mask_into_instances(mask, max_k=max_instances)
    artifacts: Dict[str, List[Image.Image]] = {
        "cropped_image": [],
        "cropped_mask": [],
        "annotated_image": [],
        "mask_cropped_image": [],
    }
    if not instance_masks:
        out = image.copy()
        return (out, artifacts) if return_artifacts else out

    running = image.copy()
    grid_x, grid_y = crop_grid
    for inst_mask in instance_masks:
        if not crop_and_paste:
            gx, gy = running.size  # whole image
        elif crop_ratio is not None:
            gx = gy = crop_grid_by_ratio(inst_mask, crop_ratio)
        else:
            gx, gy = grid_x, grid_y

        cropped_image, cropped_mask, offset = best_crop(running, inst_mask, gx, gy)
        if return_artifacts:
            artifacts["annotated_image"].append(annotate(running, inst_mask, offset, cropped_image.size, anomaly_name))

        edited_crop = model_inpaint(cropped_image, cropped_mask, anomaly_name)
        running = paste_back(
            input_image=running,
            recon_image=edited_crop,
            cropped_image=cropped_image,
            cropped_mask=cropped_mask,
            offset=offset,
            poisson_blend=poisson_blend,
        )

        if return_artifacts:
            artifacts["cropped_image"].append(cropped_image)
            artifacts["cropped_mask"].append(cropped_mask)
            # mask_bbox is inclusive (xmax/ymax); PIL crop's right/lower are exclusive, so +1 —
            # otherwise a 1px-tall/wide instance mask yields a zero-area crop ("cannot write empty image").
            lx, uy, rx, by = mask_bbox(inst_mask)
            artifacts["mask_cropped_image"].append(running.crop((lx, uy, rx + 1, by + 1)))

    return (running, artifacts) if return_artifacts else running


def run_iterative_inpaint_batch(
    images: List[Image.Image],
    masks: List[Image.Image],
    anomaly_names: List[str],
    model_inpaint_batch,
    *,
    num_depth: int,
    seeds: List[int],
    crop_grids: List[Tuple[int, int]],
    crop_ratios: List[float | None],
    crop_and_pastes: List[bool],
    poisson_blends: List[bool],
    max_instances_list: List[int],
):
    """Batched iterative inpaint: process the j-th instance of every image in one forward.

    For each depth ``j`` in ``range(num_depth)``, gather the j-th instance crop of every image
    (dummy-padding images with fewer instances so the batch width stays constant), run one
    batched ``model_inpaint_batch(crops, masks, names, seeds)`` — each seed offset by ``j``, so a
    sample's instances don't share one noise draw — and paste each real result back into that
    image's running reconstruction. Crop windows derive from the (fixed) instance masks, so they
    are stable across depths; only the crop *content* updates from the running image.
    Returns per-image ``(composites, artifacts)`` where each artifact dict has per-instance lists
    keyed ``cropped_image``/``cropped_mask``/``annotated_image``/``mask_cropped_image``.
    """
    b = len(images)
    inst_masks = [split_mask_into_instances(masks[i], max_k=max_instances_list[i]) for i in range(b)]
    running = [images[i].copy() for i in range(b)]
    artifacts: List[Dict[str, List[Image.Image]]] = [
        {"cropped_image": [], "cropped_mask": [], "annotated_image": [], "mask_cropped_image": []} for _ in range(b)
    ]

    for j in range(num_depth):
        crops: List[Image.Image] = []
        cmasks: List[Image.Image] = []
        offsets: List[Tuple[int, int]] = []
        is_real: List[bool] = []
        for i in range(b):
            real = j < len(inst_masks[i])
            is_real.append(real)
            if real:
                inst = inst_masks[i][j]
                if not crop_and_pastes[i]:
                    gx, gy = running[i].size  # whole image
                elif crop_ratios[i] is not None:
                    gx = gy = crop_grid_by_ratio(inst, crop_ratios[i])
                else:
                    gx, gy = crop_grids[i]
                ci, cm, off = best_crop(running[i], inst, gx, gy)
                artifacts[i]["annotated_image"].append(annotate(running[i], inst, off, ci.size, anomaly_names[i]))
                artifacts[i]["cropped_image"].append(ci)
                artifacts[i]["cropped_mask"].append(cm)
                crops.append(ci)
                cmasks.append(cm)
                offsets.append(off)
            else:
                # Dummy slot keeps the batch width constant; its output is discarded.
                crops.append(images[i].copy())
                cmasks.append(Image.new("L", images[i].size))
                offsets.append((0, 0))

        # One noise draw per instance; mirrors build_inpaint_one's counter in the serial path.
        edited = model_inpaint_batch(crops, cmasks, anomaly_names, [int(s) + j for s in seeds])

        for i in range(b):
            if not is_real[i]:
                continue
            running[i] = paste_back(
                input_image=running[i],
                recon_image=edited[i],
                cropped_image=crops[i],
                cropped_mask=cmasks[i],
                offset=offsets[i],
                poisson_blend=poisson_blends[i],
            )
            # mask_bbox is inclusive; PIL crop right/lower are exclusive, so +1 (avoids zero-area crop).
            lx, uy, rx, by = mask_bbox(inst_masks[i][j])
            artifacts[i]["mask_cropped_image"].append(running[i].crop((lx, uy, rx + 1, by + 1)))

    return running, artifacts
