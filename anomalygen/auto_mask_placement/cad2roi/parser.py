# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CAD mask parser: classify pixels by color, extract per-class connected components."""

import ast
import json
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist


@dataclass
class CADComponent:
    """A single connected component extracted from the CAD mask."""

    class_name: str
    mask: np.ndarray  # binary mask (H, W), uint8 0/255
    bbox: tuple  # (x, y, w, h)
    area: int
    centroid: tuple  # (cx, cy)
    component_id: int = 0


@dataclass
class ROICandidate:
    """A candidate ROI region (may combine multiple components)."""

    mask: np.ndarray  # binary mask (H, W), uint8 0/255
    bbox: tuple  # (x, y, w, h)
    centroid: tuple  # (cx, cy)
    area: int
    source_classes: list = field(default_factory=list)
    component_ids: list = field(default_factory=list)


class CADParser:
    """Parse a color-coded CAD mask into per-class components."""

    def __init__(self, label_path: str, min_area_abs: int = 5, fragment_ratio: float = 0.1):
        self.min_area_abs = min_area_abs
        self.fragment_ratio = fragment_ratio

        with open(label_path) as f:
            raw = json.load(f)

        self.color_map = {}
        self.class_names = []
        for color_str, info in raw.items():
            try:
                rgba = ast.literal_eval(color_str)
            except (ValueError, SyntaxError):
                rgba = None
            # Must be an (r, g, b[, a]) sequence; a valid-but-wrong-type literal
            # (set/dict/scalar) would otherwise fail later at rgba[:3] with a
            # cryptic TypeError instead of a clear message.
            if not isinstance(rgba, (list, tuple)) or len(rgba) < 3:
                raise ValueError(f"Invalid color key in label file: {color_str!r}")
            cls = info["class"]
            if cls in ("BACKGROUND", "UNLABELLED"):
                continue
            self.color_map[cls] = np.array(rgba[:3], dtype=np.float32)
            self.class_names.append(cls)

        self._all_colors = {"bg": np.array([0, 0, 0], dtype=np.float32)}
        self._all_colors.update(self.color_map)

    def parse(self, mask_path: str):
        """Parse a CAD mask image.

        Returns:
            (dict[class_name -> list[CADComponent]], (H, W))
        """
        img = np.array(Image.open(mask_path).convert("RGB"))
        h, w = img.shape[:2]
        rgb = img.astype(np.float32)

        flat = rgb.reshape(-1, 3)
        names = list(self._all_colors.keys())
        color_array = np.array(list(self._all_colors.values()), dtype=np.float32)
        closest = cdist(flat, color_array).argmin(axis=1).reshape(h, w)

        result = {cls: [] for cls in self.class_names}
        for i, name in enumerate(names):
            if name == "bg":
                continue
            class_mask = (closest == i).astype(np.uint8) * 255

            num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(class_mask)

            all_ccs = []
            for j in range(1, num_labels):
                x, y, bw, bh, area = stats[j]
                all_ccs.append((j, x, y, bw, bh, area))

            max_area = max((a for _, _, _, _, _, a in all_ccs), default=0)

            comp_id = 0
            for j, x, y, bw, bh, area in all_ccs:
                if area < self.min_area_abs:
                    continue
                if max_area > 0 and area < max_area * self.fragment_ratio:
                    continue

                comp_mask = np.zeros((h, w), dtype=np.uint8)
                comp_mask[labels_map == j] = 255

                result[name].append(
                    CADComponent(
                        class_name=name,
                        mask=comp_mask,
                        bbox=(x, y, bw, bh),
                        area=area,
                        centroid=(centroids[j][0], centroids[j][1]),
                        component_id=comp_id,
                    )
                )
                comp_id += 1

        return result, (h, w)
