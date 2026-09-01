# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distribute num_sdg across defect types → allocation.json (roi_pair ``--allocation``).

Runs a fail-fast **preflight** over the AMP inputs (dataset_dir, defect_spec)
before allocating: every submask dir, clean-image pool (under dataset_dir), text
``roi_prompt_defect_location``, and cad_mask is checked and all problems are
reported at once (exit 1), so the downstream pair_build/place run can't crash
mid-AMP.

Two modes:

  * --mode inference (default): uniform allocation across defect types.
    base = num_sdg // N; first (num_sdg % N) defects get +1. No floor on
    per-defect count (allows 0 if --per_defect_counts asks for it).

  * --mode validation: proportional to training mask counts (largest-remainder
    rounding). Enforces a KPI floor — every defect must have ≥1 sample so
    per-defect validation metrics stay defined.

Override:

  * --per_defect_counts <JSON> sets exact per-defect counts (e.g.
    '{"IC+bridge": 5, "passive_component+missing": 10}'). Only valid with
    --mode inference. When the sum != --num_sdg, a warning is logged and the
    override sum is used.

Defect types are read from --defect_desc (the defect_spec JSONL), in file order.

Usage::

    python -m anomalygen.scripts.auto_mask_placement.roi_allocate \\
        --num_sdg N --defect_desc datasets/<ds>/defect_spec.jsonl \\
        --dataset_dir datasets/<ds> --output_allocation allocation.json [--mode inference]
"""

import argparse
import json
import math
import pathlib

from cosmos_framework.utils import log

from anomalygen.data.utils import validate_anomaly_type

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _list_images(d):
    d = pathlib.Path(d)
    if not d.is_dir():
        return []
    return [p for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS]


def _validate_amp_inputs(dataset_dir, defect_desc):
    """Cross-check the AMP inputs (dataset_dir, defect_desc); return error strings ([] == OK).

    Per defect in defect_desc:
      * <dataset_dir>/<TEXTURE>/mask/<ANOMALY>/ exists (submask source)
      * <dataset_dir>/<TEXTURE>/clean_image/ or <dataset_dir>/<TEXTURE>/ has >=1 image
        (flat <dataset_dir>/ as the final fallback)
      * spatial_dependency=="text": roi_prompt_defect_location present and non-empty
      * spatial_dependency=="cad": semantic_segmentation_labels.json exists,
        <TEXTURE>/cad_mask/ exists, and a cad_mask/<stem>.png exists per clean image
    """
    dataset_dir = pathlib.Path(dataset_dir)
    entries = [json.loads(line) for line in pathlib.Path(defect_desc).read_text().splitlines() if line.strip()]

    errors = []
    if any(e.get("spatial_dependency") == "cad" for e in entries):
        labels = dataset_dir / "semantic_segmentation_labels.json"
        if not labels.exists():
            errors.append(f"missing {labels} (required for spatial_dependency=cad)")

    for e in entries:
        full = e["defect_type"]
        sd = e.get("spatial_dependency")
        # Rejected here, at the dataset spec, because this value becomes a path segment below and a
        # directory name in the pseudo-label layout several stages later.
        try:
            validate_anomaly_type(full, field="defect_type")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        texture, anomaly = full.split("+", 1)

        submask_dir = dataset_dir / texture / "mask" / anomaly
        if not submask_dir.is_dir():
            errors.append(f"{full}: missing submask source {submask_dir}")

        clean_nested = dataset_dir / texture / "clean_image"
        clean_tex = dataset_dir / texture
        if clean_nested.is_dir():
            clean_pool = clean_nested
        elif clean_tex.is_dir():
            clean_pool = clean_tex
        else:
            clean_pool = dataset_dir
        clean_imgs = _list_images(clean_pool)
        if not clean_imgs:
            errors.append(f"{full}: no clean images under {clean_pool}")

        if sd == "text":
            prompt = (e.get("roi_prompt_defect_location") or "").strip()
            if not prompt:
                errors.append(f"{full}: spatial_dependency={sd} requires non-empty roi_prompt_defect_location")
        elif sd == "cad":
            cad_dir = dataset_dir / texture / "cad_mask"
            if not cad_dir.is_dir():
                errors.append(f"{full}: missing {cad_dir}")
                continue
            missing_cad = [p.stem for p in clean_imgs if not (cad_dir / f"{p.stem}.png").exists()]
            if missing_cad:
                errors.append(
                    f"{full}: {len(missing_cad)} clean image(s) without matching cad_mask (first: {missing_cad[0]}.png)"
                )
        elif sd in (None, "free"):
            pass
        else:
            errors.append(f"{full}: unknown spatial_dependency={sd!r}")

    return errors


def _defect_types_from_spec(defect_desc):
    """Ordered, de-duplicated defect_type list from the defect_spec JSONL."""
    types = []
    for line in pathlib.Path(defect_desc).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        dt = json.loads(line)["defect_type"]
        if dt not in types:
            types.append(dt)
    return types


def _count_masks(dataset_dir, defect_types):
    counts = {}
    for full_type in defect_types:
        texture, anomaly = full_type.split("+", 1)
        d = pathlib.Path(dataset_dir) / texture / "mask" / anomaly
        counts[full_type] = len(_list_images(d))
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


def _allocate(num_sdg, defect_types, counts, *, mode="inference", override=None):
    """Distribute num_sdg across defect_types.

    mode:
        - "inference": uniform; no per-defect floor.
        - "validation": proportional to mask counts; enforces ≥1 per defect.

    override:
        Dict of {defect_type: count}. Only valid with mode="inference". Defect
        types not listed get 0. If sum(override) != num_sdg, a warning is logged
        and the override sum is used as the effective num_sdg.
    """
    if mode not in ("inference", "validation"):
        raise ValueError(f"unknown --mode {mode!r} (expected 'inference' or 'validation')")
    if num_sdg < 0:
        raise ValueError(f"num_sdg must be >= 0 (got {num_sdg})")
    if not defect_types:
        raise ValueError("defect_types must be non-empty")

    if override is not None:
        if mode != "inference":
            raise ValueError(f"--per_defect_counts is only valid with --mode inference (got --mode {mode})")
        unknown = [k for k in override if k not in defect_types]
        if unknown:
            log.warning(
                f"--per_defect_counts has unknown defect type(s) {unknown} not in {list(defect_types)}; ignored"
            )
        alloc = {t: int(override.get(t, 0)) for t in defect_types}
        override_sum = sum(alloc.values())
        if override_sum != num_sdg:
            log.warning(
                f"--per_defect_counts sum ({override_sum}) differs from --num_sdg ({num_sdg}); "
                "using override sum as effective num_sdg"
            )
        return alloc

    if mode == "inference":
        return _uniform(num_sdg, defect_types)

    # mode == "validation"
    alloc = _proportional(num_sdg, defect_types, counts)
    # KPI floor: training validation needs per-defect samples for stable
    # nn_score / mnn_score. Refuse to silently allocate 0.
    zero_types = [t for t in defect_types if alloc[t] == 0]
    if zero_types:
        n_total = int(sum(counts.get(t, 0) for t in defect_types))
        positive = [counts.get(t, 0) for t in defect_types if counts.get(t, 0) > 0]
        n_min = min(positive) if positive else 0
        suggested = max(num_sdg, math.ceil(3 * n_total / n_min)) if n_min > 0 else num_sdg
        raise ValueError(
            f"validation JSONL coverage broken: types {zero_types} got 0 entries with num_sdg={num_sdg}. "
            f"Increase num_sdg to >= {suggested} or trim defect_spec."
        )
    return alloc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_sdg", type=int, required=True)
    parser.add_argument("--defect_desc", type=pathlib.Path, required=True, help="defect_spec JSONL")
    parser.add_argument(
        "--dataset_dir",
        type=pathlib.Path,
        required=True,
        help="Dataset root — scans <dataset_dir>/<TEXTURE>/mask/<ANOMALY>/.",
    )
    parser.add_argument("--output_allocation", type=pathlib.Path, required=True, help="Where to write allocation.json.")
    parser.add_argument(
        "--mode",
        choices=["inference", "validation"],
        default="inference",
        help="inference (default): uniform allocation, no floor. "
        "validation: proportional to mask counts, ≥1 per defect.",
    )
    parser.add_argument(
        "--per_defect_counts",
        default=None,
        help='JSON dict of explicit per-defect counts, e.g. \'{"IC+bridge": 5, "passive_component+missing": 10}\'. '
        "Only valid with --mode inference.",
    )
    args = parser.parse_args(argv)

    errors = _validate_amp_inputs(args.dataset_dir, args.defect_desc)
    if errors:
        log.error(f"AMP input validation failed ({len(errors)} problem(s)):")
        for msg in errors:
            log.error(f"  - {msg}")
        raise SystemExit(1)
    log.info("AMP inputs OK")

    defect_types = _defect_types_from_spec(args.defect_desc)
    counts = _count_masks(args.dataset_dir, defect_types)
    log.info(f"derived per-type mask counts: {counts}")

    override = None
    if args.per_defect_counts is not None:
        override = json.loads(args.per_defect_counts)
        if not isinstance(override, dict):
            raise ValueError("--per_defect_counts must be a JSON object (dict)")

    alloc = _allocate(args.num_sdg, defect_types, counts, mode=args.mode, override=override)
    args.output_allocation.parent.mkdir(parents=True, exist_ok=True)
    args.output_allocation.write_text(json.dumps(alloc, indent=2))
    log.info(f"wrote {args.output_allocation} (mode={args.mode}, sum={sum(alloc.values())})")


if __name__ == "__main__":
    main()
