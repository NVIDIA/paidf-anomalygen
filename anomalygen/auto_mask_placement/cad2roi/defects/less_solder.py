# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Less solder defect: ROI = pad component, AMP places submask within pad."""

import os
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np
from cosmos_framework.utils import log

from anomalygen.auto_mask_placement.cad2roi.mask_utils import (
    load_and_scale_submask,
    mask_area,
    read_amp_output,
    save_temp_mask,
)
from anomalygen.auto_mask_placement.cad2roi.parser import ROICandidate
from anomalygen.auto_mask_placement.config import AugmentationParams
from anomalygen.auto_mask_placement.core import AutoMaskPlacement
from anomalygen.auto_mask_placement.data_types import AlignmentPoint


def _check_min_area(mask: np.ndarray, roi_area: int, min_ratio: float = 0.5) -> bool:
    """Check that mask area is at least min_ratio of roi_area."""
    if roi_area <= 0:
        return True
    return mask_area(mask) >= roi_area * min_ratio


def _validate_placement(placed, roi_mask, roi_area, min_area_ratio):
    """Clip to ROI and check area ratio. Returns clipped mask or None."""
    clipped = cv2.bitwise_and(placed, roi_mask)
    if mask_area(clipped) == 0:
        return None
    if not _check_min_area(clipped, roi_area, min_area_ratio):
        return None
    return clipped


def get_less_solder_candidates(components: dict, img_shape: tuple) -> List[ROICandidate]:
    """Each pad component is a less_solder ROI candidate."""
    candidates = []
    for comp in components.get("pad", []):
        candidates.append(
            ROICandidate(
                mask=comp.mask,
                bbox=comp.bbox,
                centroid=comp.centroid,
                area=comp.area,
                source_classes=["pad"],
                component_ids=[comp.component_id],
            )
        )
    return candidates


class LessSolderMaskPlacer:
    """Place less_solder defect submask onto pad ROIs using AMP.

    Each pad ROI is processed individually with per-ROI submask scaling,
    similar to BridgeMaskPlacer. This ensures the submask is correctly
    sized for each specific pad.
    """

    def __init__(self, scale_factor: float = 0.8, min_area_ratio: float = 0.5, max_retries: int = 100):
        self.scale_factor = scale_factor
        self.min_area_ratio = min_area_ratio
        self.max_retries = max_retries

    def place_on_roi(
        self, submask_path: str, roi_candidate: ROICandidate, img_shape: tuple, seed: int = 42
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Place submask onto a single pad ROI.

        Returns:
            (mask, is_fallback) — clipped mask or None, and whether fallback was used.
        """
        h, w = img_shape
        cand = roi_candidate

        scaled_path = load_and_scale_submask(submask_path, cand.area, img_shape, self.scale_factor)
        if scaled_path is None:
            return None, True

        roi_path = save_temp_mask(cand.mask)
        rng = np.random.RandomState(seed)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                for attempt in range(self.max_retries):
                    s = rng.randint(0, 100000)
                    amp = AutoMaskPlacement(
                        image_width=w,
                        image_height=h,
                        augmentation_params=AugmentationParams(),
                        roi_alignment_point=AlignmentPoint.RANDOM,
                        strict_alignment=False,
                        random_seed=s,
                        max_retry_per_mask=5,
                        separate_rois=False,
                    )
                    amp.load_combined_rois(roi_image_paths=[roi_path])

                    try:
                        amp.process_submask(submask_path=scaled_path, n_instances=1, output_dir=tmp_dir)
                    except Exception:
                        continue

                    placed = read_amp_output(tmp_dir)
                    if placed is None:
                        continue

                    clipped = _validate_placement(placed, cand.mask, cand.area, self.min_area_ratio)
                    if clipped is None:
                        continue

                    log.info(f"LessSolder placement OK (attempt {attempt + 1})")
                    return clipped, False

                # Fallback: use the ROI mask itself
                log.warning(f"LessSolder fallback after {self.max_retries} retries")
                return cand.mask.copy(), True
        finally:
            os.unlink(scaled_path)
            os.unlink(roi_path)

    def place_all(
        self,
        submask_path: str,
        candidates: List[ROICandidate],
        img_shape: tuple,
        n_instances: int = -1,
        n_seeds: int = 5,
        base_seed: int = 42,
    ) -> dict:
        """For each seed, randomly pick n_instances pad ROIs and place submask on each.

        Returns:
            dict mapping seed -> (n_requested, n_placed, np.ndarray or None)
        """
        if not candidates:
            return {}

        n_total = len(candidates)
        h, w = img_shape
        results = {}

        for seed_i in range(n_seeds):
            seed = base_seed + seed_i
            rng = np.random.RandomState(seed)

            if n_instances == -1:
                n_sel = rng.randint(1, n_total + 1)
            else:
                n_sel = min(n_instances, n_total)

            roi_indices = rng.choice(n_total, size=n_sel, replace=False)

            combined = np.zeros((h, w), dtype=np.uint8)
            n_ok = 0
            n_fallback = 0
            for ri in roi_indices:
                mask, is_fallback = self.place_on_roi(
                    submask_path,
                    candidates[ri],
                    img_shape,
                    seed=seed + ri * 100,
                )
                if mask is not None:
                    combined = cv2.bitwise_or(combined, mask)
                    if is_fallback:
                        n_fallback += 1
                    else:
                        n_ok += 1

            n_placed = n_ok + n_fallback
            results[seed_i + 1] = (n_sel, n_placed, combined if combined.sum() > 0 else None)
            log.info(f"LessSolder seed {seed_i + 1}: requested={n_sel}, ok={n_ok}, fallback={n_fallback}")
        return results
