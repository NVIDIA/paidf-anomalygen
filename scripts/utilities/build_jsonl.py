#!/usr/bin/env python3
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
"""Build SDG JSONL by pairing AMP-placed masks with their clean images.

Allocation dictates N per defect_type (from allocate_samples.py). The AMP
output layout produced by `scripts/anomaly_gen/run_auto_roi_amp.py` with n_seeds=1 is:

    <amp-output-dir>/<clean_stem>__<submask_stem>/<TEXTURE>+<ANOMALY>/seed0.png

(Legacy layout `<amp-output-dir>/<clean_stem>/<TEXTURE>+<ANOMALY>/seed<N>.png`
is also accepted for backwards compatibility.)

build_amp_samples.py emits `ceil(allocation[t] / n_seeds)` records per defect;
AMP expands each by `n_seeds` placements (one per seed), so the total mask
count on disk is `>= allocation[t]`. This script finds exactly `allocation[t]`
masks. It errors if fewer exist (an AMP failure, not a planning error).

iteration_generation_max_instance is set automatically per defect when
--dataset-dir is provided: only if ALL training submasks have ≥2 disconnected
components is iterative disabled (iter=1). If even one single-instance mask
exists the model has seen that pattern and iterative remains on (iter=5).
"""
import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

from scripts.utilities.build_amp_samples import _images


def _texture(full):
    return full.split("+", 1)[0] if "+" in full else ""


def _defect_short(full):
    return full.split("+", 1)[1] if "+" in full else full


def _clean_image_for(stem, texture, clean_dir):
    for base in (clean_dir / texture / "clean_image", clean_dir / texture, clean_dir):
        for ext in ("jpg", "png", "jpeg", "bmp"):
            p = base / f"{stem}.{ext}"
            if p.exists():
                return p
    return None


def _all_multi_instance_masks(dataset_dir, full_type):
    """Return True only if EVERY training submask for full_type has ≥2 connected components.

    If even one single-instance mask exists the model has seen single-instance
    patterns, so iterative generation remains valid.
    """
    texture, anomaly = full_type.split("+", 1)
    mask_dir = pathlib.Path(dataset_dir) / texture / "mask" / anomaly
    imgs = _images(mask_dir)
    if not imgs:
        return False
    for p in imgs:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        n_labels, *_ = cv2.connectedComponentsWithStats(
            (binary > 0).astype(np.uint8), connectivity=8)
        if n_labels - 1 < 2:  # found a single-instance mask
            return False
    return True


def _enumerate_amp_masks(amp_output_dir, full_type):
    """Yield (clean_stem, mask_path) pairs for one defect_type.

    Walks <amp_output_dir>/<clean_stem>/<full_type>/<submask_stem>__seed<i>.png
    matching the layout produced by scripts/anomaly_gen/run_auto_roi_amp.py.
    """
    amp_output_dir = pathlib.Path(amp_output_dir)
    if not amp_output_dir.is_dir():
        return
    for name_dir in sorted(p for p in amp_output_dir.iterdir() if p.is_dir()):
        type_dir = name_dir / full_type
        if not type_dir.is_dir():
            continue
        clean_stem = name_dir.name
        for mask in sorted(type_dir.glob("*__seed*.png")):
            yield clean_stem, mask


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amp-output-dir", type=pathlib.Path, required=True)
    p.add_argument("--clean-dir", type=pathlib.Path, required=True)
    p.add_argument("--dataset-dir", type=pathlib.Path, default=None,
                   help="Training dataset root. When provided, iteration_generation_max_instance "
                        "is set to 1 for defects whose training masks contain ≥2 disconnected "
                        "components, and 5 otherwise.")
    p.add_argument("--allocation", type=pathlib.Path, required=True,
                   help="JSON: {anomaly_type: count}")
    p.add_argument("--defect-types", nargs="+", required=True,
                   help="Full TEXTURE+TYPE names matching allocation keys.")
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--crop-ratio", type=float, default=2.0)
    p.add_argument("--num-steps", type=int, default=35)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()

    allocation = json.loads(args.allocation.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w") as fp:
        for full_type in args.defect_types:
            n = allocation.get(full_type, 0)
            if n == 0:
                continue

            pairs = list(_enumerate_amp_masks(args.amp_output_dir, full_type))
            if not pairs:
                print(f"warn: no AMP masks for {full_type}", file=sys.stderr); continue

            if len(pairs) < n:
                print(
                    f"warn: allocation requested {n} masks for {full_type} "
                    f"but only {len(pairs)} AMP outputs exist. "
                    f"JSONL will be short by {n - len(pairs)} for this defect. "
                    f"Check run_auto_roi_amp.py logs for NO_DETECTION / FAILED statuses.",
                    file=sys.stderr,
                )
            pairs = pairs[:n]

            texture = _texture(full_type)

            # Auto-detect iteration mode from training mask topology.
            if args.dataset_dir is not None:
                all_multi = _all_multi_instance_masks(args.dataset_dir, full_type)
                iter_max = 1 if all_multi else 5
                mode_label = "no-iter: all masks are multi-instance" if all_multi else "iter: single-instance masks exist"
            else:
                iter_max = 5
                mode_label = "iter: default"
            print(f"  {full_type}: iteration_generation_max_instance={iter_max}  [{mode_label}]")

            for stem, mask_path in pairs:
                clean_img = _clean_image_for(stem, texture, args.clean_dir)
                if clean_img is None:
                    print(f"warn: no clean image for {texture}/{stem}", file=sys.stderr); continue
                fp.write(json.dumps({
                    "image_filename": str(clean_img),
                    "mask_filename":  str(mask_path),
                    "anomaly_type":   full_type,
                    "guidance":       args.guidance,
                    "num_steps":      args.num_steps,
                    "crop_and_paste": True,
                    "crop_ratio":     args.crop_ratio,
                    "crop_grid_X":    "none",
                    "crop_grid_Y":    "none",
                    "num_generated_images": 1,
                    "poisson_blend":  False,
                    "iteration_generation_max_instance": iter_max,
                }) + "\n")
                written += 1
    print(f"wrote {written} entries to {args.output}")
    if written == 0:
        print(
            "error: 0 entries written — AMP produced no usable output for any defect. "
            "Cannot run SDG inference with an empty JSONL. "
            "Check run_auto_roi_amp.py logs to diagnose dataset / prompt / cad_mask issues.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
