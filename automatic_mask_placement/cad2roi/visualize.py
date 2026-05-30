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

"""Visualization helpers for CAD ROI candidates."""

import cv2
import numpy as np
from pathlib import Path

ROI_COLORS = {
    "missing":     (255, 0, 0),
    "less_solder": (0, 255, 0),
    "bridge":      (255, 255, 0),
}


def _blend_overlay(img, mask, color, alpha=0.6):
    """Blend color onto img where mask > 0."""
    out = img.copy()
    roi = mask > 0
    out[roi] = (
        out[roi].astype(np.float32) * (1 - alpha)
        + np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)
    return out


def _make_defect_panel(cad_img, cands, color, annotate=True):
    """Create a single overlay panel for one defect type's candidates."""
    overlay = cad_img.copy()
    for i, cand in enumerate(cands):
        overlay = _blend_overlay(overlay, cand.mask, color)
        x, y, bw, bh = cand.bbox
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 1)
        if annotate:
            cx, cy = int(cand.centroid[0]), int(cand.centroid[1])
            cv2.circle(overlay, (cx, cy), 3, (255, 255, 255), -1)
            cv2.putText(overlay, str(i), (cx + 4, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return overlay


def visualize_roi_candidates(cad_mask_path: str, candidates: dict,
                             output_dir: str, mask_name: str = "0000"):
    """Visualize all ROI candidates for each defect type."""
    from PIL import Image

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cad_img = np.array(Image.open(cad_mask_path).convert("RGB"))

    # Per-defect overview + individual masks
    for defect_type, cands in candidates.items():
        color = ROI_COLORS.get(defect_type, (255, 255, 255))
        overview = _make_defect_panel(cad_img, cands, color)
        Image.fromarray(overview).save(out_dir / f"{mask_name}_{defect_type}_overview.png")
        for i, cand in enumerate(cands):
            Image.fromarray(cand.mask).save(out_dir / f"{mask_name}_{defect_type}_roi_{i}.png")

    # Summary: side by side
    panels = []
    for defect_type in ("missing", "less_solder", "bridge"):
        cands = candidates.get(defect_type, [])
        color = ROI_COLORS.get(defect_type, (255, 255, 255))
        panel = _make_defect_panel(cad_img, cands, color, annotate=False)
        cv2.putText(panel, defect_type, (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        panels.append(panel)

    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    sep = np.ones((max_h, 2, 3), dtype=np.uint8) * 255
    combined = padded[0]
    for p in padded[1:]:
        combined = np.hstack([combined, sep, p])

    Image.fromarray(combined).save(out_dir / f"{mask_name}_summary.png")
