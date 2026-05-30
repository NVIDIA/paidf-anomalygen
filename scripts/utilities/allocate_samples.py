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
"""Distribute num_SDG across defect types.

Two modes:

  * --mode inference (default): uniform allocation across defect types.
    base = num_sdg // N; first (num_sdg % N) defects get +1. No floor on
    per-defect count (allows 0 if --per-defect-counts asks for it).

  * --mode validation: proportional to training mask counts (largest-
    remainder rounding). Enforces a KPI floor — every defect must have
    ≥1 sample so per-defect validation metrics stay defined.

Override:

  * --per-defect-counts <JSON> sets exact per-defect counts (e.g.,
    '{"IC+bridge": 5, "passive_component+missing": 10}'). Only valid
    with --mode inference. When sum != --num-sdg, a warning is printed
    to stderr and the override sum is used.

Usage:
    allocate_samples.py --num-sdg N --defect-types t1 t2 ... \\
        --mask-path <dataset_dir> --output alloc.json [--mode inference]
"""
import argparse
import json
import math
import pathlib
import sys

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def count_masks(mask_path, defect_types):
    counts = {}
    for full_type in defect_types:
        texture, anomaly = full_type.split("+", 1)
        d = pathlib.Path(mask_path) / texture / "mask" / anomaly
        if not d.is_dir():
            counts[full_type] = 0
            continue
        counts[full_type] = sum(1 for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)
    return counts


def _uniform(num_sdg, defect_types):
    """Uniform allocation: base each, first (num_sdg % N) defects get +1."""
    n = len(defect_types)
    base = num_sdg // n
    remainder = num_sdg % n
    alloc = {t: base for t in defect_types}
    for t in defect_types[:remainder]:
        alloc[t] += 1
    return alloc


def _proportional(num_sdg, defect_types, counts):
    """Proportional to mask counts via largest-remainder rounding."""
    total = float(sum(counts.get(t, 0) for t in defect_types))
    if total <= 0:
        raise ValueError(f"sum of mask counts must be > 0 (got {counts})")
    raw = {t: num_sdg * counts.get(t, 0) / total for t in defect_types}
    floors = {t: int(math.floor(r)) for t, r in raw.items()}
    remainder = num_sdg - sum(floors.values())
    order = sorted(defect_types, key=lambda t: raw[t] - floors[t], reverse=True)
    for t in order[:remainder]:
        floors[t] += 1
    return floors


def allocate(num_sdg, defect_types, counts, *, mode="inference", override=None):
    """Distribute num_sdg across defect_types.

    mode:
        - "inference": uniform; no per-defect floor.
        - "validation": proportional to mask counts; enforces ≥1 per defect.

    override:
        Dict of {defect_type: count}. Only valid with mode="inference".
        Defect types not listed get 0. If sum(override) != num_sdg, a
        warning is printed and the override sum is used as the effective
        num_sdg.
    """
    if mode not in ("inference", "validation"):
        raise ValueError(f"unknown --mode {mode!r} (expected 'inference' or 'validation')")
    if num_sdg < 0:
        raise ValueError(f"num_sdg must be >= 0 (got {num_sdg})")
    if not defect_types:
        raise ValueError("defect_types must be non-empty")

    if override is not None:
        if mode != "inference":
            raise ValueError(
                f"--per-defect-counts is only valid with --mode inference "
                f"(got --mode {mode})"
            )
        unknown = [k for k in override if k not in defect_types]
        if unknown:
            print(
                f"warn: --per-defect-counts has unknown defect type(s) {unknown} "
                f"not in --defect-types {list(defect_types)}; ignored",
                file=sys.stderr,
            )
        alloc = {t: int(override.get(t, 0)) for t in defect_types}
        override_sum = sum(alloc.values())
        if override_sum != num_sdg:
            print(
                f"warn: --per-defect-counts sum ({override_sum}) differs from "
                f"--num-sdg ({num_sdg}); using override sum as effective num_sdg",
                file=sys.stderr,
            )
        return alloc

    if mode == "inference":
        return _uniform(num_sdg, defect_types)

    # mode == "validation"
    alloc = _proportional(num_sdg, defect_types, counts)
    # KPI floor: training validation needs per-defect samples for
    # stable nn_score / mnn_score. Refuse to silently allocate 0.
    zero_types = [t for t in defect_types if alloc[t] == 0]
    if zero_types:
        n_total = int(sum(counts.get(t, 0) for t in defect_types))
        positive = [counts.get(t, 0) for t in defect_types if counts.get(t, 0) > 0]
        n_min = min(positive) if positive else 0
        suggested = max(num_sdg, math.ceil(3 * n_total / n_min)) if n_min > 0 else num_sdg
        raise ValueError(
            f"validation JSONL coverage broken: types {zero_types} got 0 entries "
            f"with num_sdg={num_sdg}. Increase num_sdg to >= {suggested} "
            f"or trim defect_spec."
        )
    return alloc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-sdg", type=int, required=True)
    p.add_argument("--defect-types", nargs="+", required=True)
    p.add_argument("--mask-path", type=pathlib.Path, required=True,
                   help="Dataset root — scans <mask_path>/<TEXTURE>/mask/<ANOMALY>/.")
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--mode", choices=["inference", "validation"], default="inference",
                   help="inference (default): uniform allocation, no floor. "
                        "validation: proportional to mask counts, ≥1 per defect.")
    p.add_argument("--per-defect-counts", default=None,
                   help='JSON dict of explicit per-defect counts, e.g. '
                        '\'{"IC+bridge": 5, "passive_component+missing": 10}\'. '
                        "Only valid with --mode inference.")
    args = p.parse_args()

    counts = count_masks(args.mask_path, args.defect_types)
    print(f"derived per-type mask counts: {counts}", file=sys.stderr)

    override = None
    if args.per_defect_counts is not None:
        override = json.loads(args.per_defect_counts)
        if not isinstance(override, dict):
            raise ValueError("--per-defect-counts must be a JSON object (dict)")

    alloc = allocate(args.num_sdg, args.defect_types, counts,
                     mode=args.mode, override=override)
    args.output.write_text(json.dumps(alloc, indent=2))
    print(f"wrote {args.output} (mode={args.mode}, sum={sum(alloc.values())})", file=sys.stderr)


if __name__ == "__main__":
    main()
