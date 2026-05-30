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

"""Bridge defect: ROI = pad + gap strip between adjacent groups, minus component and solder.
Most complex defect type with multi-step ROI generation and validated mask placement."""

import os
import tempfile
import cv2
import numpy as np
from typing import Optional, List

from automatic_mask_placement.core import AutomaticMaskPlacement
from automatic_mask_placement.config import AugmentationParams
from automatic_mask_placement.data_types import AlignmentPoint

from automatic_mask_placement.cad2roi.parser import ROICandidate
from automatic_mask_placement.cad2roi.morph_ops import dilate, erode, close
from automatic_mask_placement.cad2roi.mask_utils import (
    merge_class_masks, mask_area, mask_bbox,
    load_and_scale_submask, save_temp_mask, read_amp_output,
)
from automatic_mask_placement.cad2roi.placement_checks import clip_to_roi, check_touches_regions, check_single_component
from imaginaire.utils import log


def _validate_placement(placed, roi_mask, sub_regions):
    """Pre-clip group touch + clip + post-clip group touch + connectivity.
    Returns clipped mask or None."""
    if not check_touches_regions(placed, sub_regions, 2):
        return None
    clipped = clip_to_roi(placed, roi_mask)
    if clipped.sum() == 0:
        return None
    if not check_touches_regions(clipped, sub_regions, 2):
        return None
    if not check_single_component(clipped):
        return None
    return clipped


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def get_bridge_groups(components: dict, img_shape: tuple,
                      classes: tuple = ("pad", "solder"),
                      min_area_abs: int = 5) -> List[ROICandidate]:
    """Merge specified classes into connected groups for bridge detection."""
    h, w = img_shape
    combined = merge_class_masks(components, classes, img_shape)

    num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(combined)
    groups = []
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area_abs:
            continue
        group_mask = np.zeros((h, w), dtype=np.uint8)
        group_mask[labels_map == i] = 255
        groups.append(ROICandidate(
            mask=group_mask,
            bbox=(x, y, bw, bh),
            centroid=(centroids[i][0], centroids[i][1]),
            area=area,
            source_classes=list(classes),
            component_ids=[i],
        ))
    return groups


def _get_component_mask(components: dict, img_shape: tuple) -> np.ndarray:
    return merge_class_masks(components, ("capacitor", "ic"), img_shape)


def _get_solder_mask(components: dict, img_shape: tuple) -> np.ndarray:
    return merge_class_masks(components, ("solder",), img_shape)


def _fill_gap_between(merged_mask: np.ndarray, group_a: ROICandidate,
                      group_b: ROICandidate, img_shape: tuple) -> np.ndarray:
    """Fill gap between two groups using parallel line expansion from centroid-to-centroid."""
    h, w = img_shape
    pa = np.array(group_a.centroid, dtype=np.float64)
    pb = np.array(group_b.centroid, dtype=np.float64)

    direction = pb - pa
    length = np.linalg.norm(direction)
    if length < 1:
        return merged_mask

    perp = np.array([-direction[1], direction[0]], dtype=np.float64)
    perp = perp / np.linalg.norm(perp)

    mask_a = group_a.mask
    mask_b = group_b.mask
    max_offset = max(h, w)

    unit_dir = direction / length
    diag_a = np.sqrt(group_a.bbox[2]**2 + group_a.bbox[3]**2)
    diag_b = np.sqrt(group_b.bbox[2]**2 + group_b.bbox[3]**2)
    extend = length / 2 + max(diag_a, diag_b)

    def _line_connects_both(offset):
        shift = perp * offset
        mid = (pa + pb) / 2 + shift
        p1 = mid - unit_dir * extend
        p2 = mid + unit_dir * extend
        p1_int = (int(round(p1[0])), int(round(p1[1])))
        p2_int = (int(round(p2[0])), int(round(p2[1])))

        full_line = np.zeros((h, w), dtype=np.uint8)
        cv2.line(full_line, p1_int, p2_int, 255, 1)

        hits_a = cv2.bitwise_and(full_line, mask_a).sum() > 0
        hits_b = cv2.bitwise_and(full_line, mask_b).sum() > 0
        if not (hits_a and hits_b):
            return False, None

        temp = cv2.bitwise_and(full_line, cv2.bitwise_not(mask_a))
        temp = cv2.bitwise_and(temp, cv2.bitwise_not(mask_b))

        nz_a = cv2.findNonZero(cv2.bitwise_and(full_line, mask_a))
        nz_b = cv2.findNonZero(cv2.bitwise_and(full_line, mask_b))
        if nz_a is not None and nz_b is not None:
            proj_a = np.dot(nz_a.reshape(-1, 2).astype(np.float64), unit_dir)
            proj_b = np.dot(nz_b.reshape(-1, 2).astype(np.float64), unit_dir)
            bound_lo = min(proj_a.max(), proj_b.max())
            bound_hi = max(proj_a.min(), proj_b.min())
            if bound_lo > bound_hi:
                bound_lo, bound_hi = bound_hi, bound_lo
            temp_pts = cv2.findNonZero(temp)
            if temp_pts is not None:
                projs = np.dot(temp_pts.reshape(-1, 2).astype(np.float64), unit_dir)
                mask_keep = (projs >= bound_lo) & (projs <= bound_hi)
                filtered = np.zeros((h, w), dtype=np.uint8)
                for pt, keep in zip(temp_pts.reshape(-1, 2), mask_keep):
                    if keep:
                        filtered[pt[1], pt[0]] = 255
                temp = filtered

        return True, temp

    ok, temp = _line_connects_both(0)
    if ok:
        merged_mask = cv2.bitwise_or(merged_mask, temp)

    for offset in range(1, max_offset):
        ok, temp = _line_connects_both(offset)
        if not ok:
            break
        merged_mask = cv2.bitwise_or(merged_mask, temp)

    for offset in range(1, max_offset):
        ok, temp = _line_connects_both(-offset)
        if not ok:
            break
        merged_mask = cv2.bitwise_or(merged_mask, temp)

    return merged_mask


# ---------------------------------------------------------------------------
# ROI generation
# ---------------------------------------------------------------------------

def _try_bridge_roi(group_indices, groups, comp_mask, solder_mask,
                    img_shape, fill_gap, bridge_max_cut_ratio):
    """Try to create a bridge ROI for a set of groups."""
    h, w = img_shape

    groups_only = np.zeros((h, w), dtype=np.uint8)
    for idx in group_indices:
        groups_only = cv2.bitwise_or(groups_only, groups[idx].mask)

    gap_strip = np.zeros((h, w), dtype=np.uint8)
    if fill_gap:
        sorted_indices = sorted(group_indices, key=lambda k: groups[k].centroid)
        for a, b in zip(sorted_indices[:-1], sorted_indices[1:]):
            gap_strip = _fill_gap_between(gap_strip, groups[a], groups[b], img_shape)

    # Connectivity test
    sealed_comp = dilate(comp_mask, 5)
    test_mask = cv2.bitwise_or(groups_only, gap_strip)
    test_mask = cv2.bitwise_and(test_mask, cv2.bitwise_not(sealed_comp))

    num_labels, labels_map, _, _ = cv2.connectedComponentsWithStats(test_mask)
    group_labels = set()
    for idx in group_indices:
        cx, cy = int(groups[idx].centroid[0]), int(groups[idx].centroid[1])
        label = labels_map[cy, cx] if labels_map[cy, cx] > 0 else 0
        if label == 0:
            group_test = cv2.bitwise_and(test_mask, groups[idx].mask)
            pts = cv2.findNonZero(group_test)
            if pts is None:
                return None
            pt = pts[0][0]
            label = labels_map[pt[1], pt[0]]
        if label == 0:
            return None
        group_labels.add(label)
    if len(group_labels) != 1:
        return None

    # Cut ratio filter
    gap_only = cv2.bitwise_and(gap_strip, cv2.bitwise_not(groups_only))
    gap_only_area = mask_area(gap_only)
    if gap_only_area > 0:
        comp_in_gap = cv2.bitwise_and(gap_only, comp_mask)
        comp_in_gap_area = mask_area(comp_in_gap)
        if comp_in_gap_area / gap_only_area > bridge_max_cut_ratio:
            return None

    # Build final ROI: merge first, smooth, then subtract illegal regions last
    result = cv2.bitwise_or(groups_only, gap_strip)

    result = close(result, 11)
    result = cv2.GaussianBlur(result, (7, 7), 0)
    _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)

    result = cv2.bitwise_and(result, cv2.bitwise_not(comp_mask))
    result = cv2.bitwise_and(result, cv2.bitwise_not(solder_mask))

    nz = cv2.findNonZero(result)
    if nz is None:
        return None

    # Pad existence check
    pad_in_groups = cv2.bitwise_and(groups_only, cv2.bitwise_not(solder_mask))
    pad_in_groups = cv2.bitwise_and(pad_in_groups, cv2.bitwise_not(comp_mask))
    if pad_in_groups.sum() == 0:
        return None

    x, y, bw, bh = cv2.boundingRect(nz)
    area = mask_area(result)

    # Gap length filter
    max_group_dim = max(max(groups[idx].bbox[2], groups[idx].bbox[3]) for idx in group_indices)
    gap_only_result = cv2.bitwise_and(result, cv2.bitwise_not(groups_only))
    gap_nz = cv2.findNonZero(gap_only_result)
    if gap_nz is not None:
        _, _, gw, gh = cv2.boundingRect(gap_nz)
        if max(gw, gh) > 3 * max_group_dim:
            return None

    return ROICandidate(
        mask=result,
        bbox=(x, y, bw, bh),
        centroid=(x + bw / 2, y + bh / 2),
        area=area,
        source_classes=["pad", "solder"],
        component_ids=list(group_indices),
    )


def get_bridge_candidates(components: dict, img_shape: tuple,
                          fill_gap: bool = True, max_chain: int = 3,
                          bridge_classes: tuple = ("pad", "solder"),
                          bridge_max_cut_ratio: float = 0.25,
                          min_area_abs: int = 5) -> List[ROICandidate]:
    """Find all bridge ROI candidates."""
    groups = get_bridge_groups(components, img_shape, classes=bridge_classes,
                               min_area_abs=min_area_abs)
    if len(groups) < 2:
        return []

    comp_mask = _get_component_mask(components, img_shape)
    solder_mask = _get_solder_mask(components, img_shape)
    n = len(groups)
    candidates = []
    seen = set()

    valid_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            key = (i, j)
            cand = _try_bridge_roi(key, groups, comp_mask, solder_mask,
                                   img_shape, fill_gap, bridge_max_cut_ratio)
            if cand:
                valid_pairs.append(key)
                seen.add(key)
                candidates.append(cand)

    if max_chain >= 3:
        for (i, j) in valid_pairs:
            for k in range(n):
                if k == i or k == j:
                    continue
                triple = tuple(sorted([i, j, k]))
                if triple in seen:
                    continue
                cand = _try_bridge_roi(triple, groups, comp_mask, solder_mask,
                                       img_shape, fill_gap, bridge_max_cut_ratio)
                if cand:
                    seen.add(triple)
                    candidates.append(cand)

    return candidates


# ---------------------------------------------------------------------------
# Mask Placer
# ---------------------------------------------------------------------------

class BridgeMaskPlacer:
    """Place a bridge defect submask onto bridge ROI with 3-stage validation."""

    def __init__(self, max_retries: int = 100, scale_factor: float = 1.0,
                 expand_kernel_size: int = 31, expand_iterations: int = 2):
        self.max_retries = max_retries
        self.scale_factor = scale_factor
        self.expand_kernel_size = expand_kernel_size
        self.expand_iterations = expand_iterations

    def _get_aug_params(self):
        aug = AugmentationParams()
        aug.rotation_probability = 1.0
        aug.rotation_range = (-180, 180)
        aug.scale_x_probability = 1.0
        aug.scale_y_probability = 1.0
        aug.scale_x_range = (0.5, 2.0)
        aug.scale_y_range = (0.5, 2.0)
        aug.shear_x_probability = 1.0
        aug.shear_y_probability = 1.0
        aug.shear_x_range = (-20, 20)
        aug.shear_y_range = (-20, 20)
        return aug

    def _make_expanded_roi(self, roi_mask):
        expanded = dilate(roi_mask, self.expand_kernel_size,
                          iterations=self.expand_iterations)
        return save_temp_mask(expanded)

    def place_on_roi(self, submask_path, roi_candidate, groups, img_shape, seed=42):
        """Place bridge submask with 3-stage validation."""
        h, w = img_shape
        cand = roi_candidate
        sub_regions = [groups[i].mask for i in cand.component_ids]

        scaled_path = load_and_scale_submask(
            submask_path, cand.area, img_shape, self.scale_factor)
        if scaled_path is None:
            return None, True
        expanded_path = self._make_expanded_roi(cand.mask)
        aug_params = self._get_aug_params()

        rng = np.random.RandomState(seed)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                for attempt in range(self.max_retries):
                    s = rng.randint(0, 100000)
                    amp = AutomaticMaskPlacement(
                        image_width=w, image_height=h,
                        augmentation_params=aug_params,
                        roi_alignment_point=AlignmentPoint.CENTER,
                        strict_alignment=False, random_seed=s,
                        max_retry_per_mask=5, separate_rois=False,
                    )
                    amp.load_combined_rois(roi_image_paths=[expanded_path])

                    try:
                        amp.process_submask(submask_path=scaled_path,
                                            n_instances=1, output_dir=tmp_dir)
                    except Exception:
                        continue

                    placed = read_amp_output(tmp_dir)
                    if placed is None:
                        continue

                    clipped = _validate_placement(placed, cand.mask, sub_regions)
                    if clipped is None:
                        continue

                    log.info(f"Bridge placement OK (attempt {attempt+1})")
                    return clipped, False  # (mask, is_fallback)

                # Fallback
                log.warning(f"Bridge placement fallback after {self.max_retries} retries")
                fallback = cand.mask.copy()
                fallback = erode(fallback, 3)
                fallback = fallback if fallback.sum() > 0 else cand.mask.copy()
                return fallback, True  # (mask, is_fallback)
        finally:
            os.unlink(scaled_path)
            os.unlink(expanded_path)

    def place_all(self, submask_path, bridge_candidates, groups, img_shape,
                  n_instances=-1, n_seeds=5, base_seed=42):
        """For each seed, randomly pick n_instances bridge ROIs and place submask on each.

        Args:
            n_instances: -1 = random from [1, n_candidates], otherwise capped

        Returns:
            dict mapping seed -> (n_requested, n_placed, combined_mask_or_None)
        """
        if not bridge_candidates:
            return {}
        n_total = len(bridge_candidates)
        results = {}

        for seed_i in range(n_seeds):
            seed = base_seed + seed_i
            rng = np.random.RandomState(seed)

            if n_instances == -1:
                n_sel = rng.randint(1, n_total + 1)
            else:
                n_sel = min(n_instances, n_total)

            roi_indices = rng.choice(n_total, size=n_sel, replace=False)

            h, w = img_shape
            combined = np.zeros((h, w), dtype=np.uint8)
            n_ok = 0
            n_fallback = 0
            for ri in roi_indices:
                cand = bridge_candidates[ri]
                mask, is_fallback = self.place_on_roi(
                    submask_path, cand, groups, img_shape,
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
            log.info(f"Bridge seed {seed_i+1}: requested={n_sel}, "
                     f"ok={n_ok}, fallback={n_fallback}")
        return results
