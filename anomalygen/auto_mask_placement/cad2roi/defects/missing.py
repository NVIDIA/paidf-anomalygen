# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Missing defect: ROI = component (capacitor/IC) + solder, no pad.
ROI itself IS the mask — no submask placement needed."""

from typing import List

import cv2
import numpy as np
from cosmos_framework.utils import log

from anomalygen.auto_mask_placement.cad2roi.mask_utils import mask_area, mask_bbox, merge_class_masks
from anomalygen.auto_mask_placement.cad2roi.morph_ops import close, dilate
from anomalygen.auto_mask_placement.cad2roi.parser import ROICandidate


def get_missing_candidates(components: dict, img_shape: tuple, min_area_abs: int = 5) -> List[ROICandidate]:
    """Each missing candidate = component (capacitor/IC) + solder, no pad.
    Found by: merge solder+component, dilate to connect, find connected units
    that contain a capacitor or IC."""
    h, w = img_shape

    comp_solder_mask = merge_class_masks(components, ("solder", "capacitor", "ic"), img_shape)
    connected = dilate(comp_solder_mask, 5)

    num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(connected)

    comp_mask = merge_class_masks(components, ("capacitor", "ic"), img_shape)

    candidates = []
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area_abs:
            continue

        unit_mask_dilated = (labels_map == i).astype(np.uint8) * 255

        overlap_comp = cv2.bitwise_and(unit_mask_dilated, comp_mask)
        if overlap_comp.sum() == 0:
            continue

        unit_mask = cv2.bitwise_and(comp_solder_mask, unit_mask_dilated)

        unit_mask = close(unit_mask, 7)

        bbox = mask_bbox(unit_mask)
        if bbox is None:
            continue
        ux, uy, uw, uh = bbox
        u_area = mask_area(unit_mask)

        candidates.append(
            ROICandidate(
                mask=unit_mask,
                bbox=(ux, uy, uw, uh),
                centroid=(centroids[i][0], centroids[i][1]),
                area=u_area,
                source_classes=["capacitor+solder", "ic+solder"],
                component_ids=[i],
            )
        )
    return candidates


class MissingMaskPlacer:
    """For missing defect: ROI = mask directly.
    Each seed randomly selects n_instances candidates and combines their masks."""

    def place_all(
        self,
        missing_candidates: List[ROICandidate],
        n_instances: int = -1,
        n_seeds: int = 5,
        base_seed: int = 42,
    ) -> dict:
        """For each seed, randomly pick n_instances candidates and OR their masks.

        Args:
            missing_candidates: list of missing ROI candidates
            n_instances: -1 = all, otherwise capped to len(candidates)
            n_seeds: number of seeds
            base_seed: base random seed

        Returns:
            dict mapping seed -> (n_selected, n_placed, combined_mask)
        """
        if not missing_candidates:
            return {}

        n_total = len(missing_candidates)

        results = {}
        for seed_i in range(n_seeds):
            rng = np.random.RandomState(base_seed + seed_i)
            if n_instances == -1:
                n_sel = rng.randint(1, n_total + 1)  # random from [1, n_total]
            else:
                n_sel = min(n_instances, n_total)
            indices = rng.choice(n_total, size=n_sel, replace=False)

            # Combine selected masks
            h, w = missing_candidates[0].mask.shape
            combined = np.zeros((h, w), dtype=np.uint8)
            for idx in indices:
                combined = cv2.bitwise_or(combined, missing_candidates[idx].mask)

            results[seed_i + 1] = (n_sel, n_sel, combined)
            selected_ids = [missing_candidates[i].component_ids[0] for i in indices]
            log.info(f"Missing seed {seed_i + 1}: selected {n_sel} ROIs {selected_ids}")
        return results
