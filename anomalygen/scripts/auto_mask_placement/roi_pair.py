# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the (clean, submask) pairing JSON consumed by ``roi_place`` (its ``--input_pair_path``).

Pair budget per defect = **every (submask, clean) combination**:
``num_submasks[d] × num_cleans[texture]``. ``n_seeds`` > 1 (AMP multi-placement on a repeated
record) only kicks in once the allocation exceeds that budget, so (submask, clean) diversity is
maximized before relying on AMP augmentation to make up the count.

``n_seeds`` is computed **per defect type** from the allocation::

    n_seeds[d] = ⌈allocation[d] / (num_submasks[d] × num_cleans[texture])⌉

(e.g. num_SDG=50 over 5 submasks × 10 cleans = 50 pair budget → n_seeds=1; num_SDG=200 → n_seeds=4.)
Per defect ``d`` it emits ``⌈allocation[d] / n_seeds[d]⌉`` records; each batch of S records pairs every
submask with a different clean, the next batch shifts the per-submask clean offset by one, and a
per-defect deterministic shuffle makes every (submask, clean) combination appear once before any repeat.

Per defect rather than one global maximum: a type with a thin pair budget (few clean images) would
otherwise raise ``n_seeds`` for *every* type, and since each type emits ``⌈allocation[d] / n_seeds⌉``
records worth ``n_seeds`` images each, that rounding inflates the run's total past ``num_sdg``.

Record schema (consumed by roi_place)::

    {"clean_image":     <path>,
     "defect_type":     "TEXTURE+ANOMALY",
     "submask":         <path to training mask>,
     "name":            "<clean_stem>__<submask_stem>",
     "cad_mask":        <path>  if spatial_dependency=="cad" else null,
     "cad_mask_label":  <path>  if spatial_dependency=="cad" else null,
     "n_seeds":         <int>   placements for THIS record}

``<output_pair_path>.n_seeds`` holds ``max_d n_seeds[d]`` — an upper bound for ``roi_place --n_seeds``,
which prefers each record's own value. The ``--allocation`` JSON ({defect_type: count}) comes from
``roi_allocate`` or can be written by hand.

Usage::

    python -m anomalygen.scripts.auto_mask_placement.roi_pair \\
        --dataset_dir datasets/<ds> \\
        --defect_desc datasets/<ds>/defect_spec.jsonl \\
        --allocation allocation.json \\
        --output_pair_path amp_samples.json [--seed 42]

Clean base images are read from ``<dataset_dir>/<TEXTURE>/clean_image/``.
"""

import argparse
import json
import math
import pathlib
import random

from cosmos_framework.utils import log

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _images(d):
    d = pathlib.Path(d)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)


def _clean_pool(clean_dir, texture):
    nested = clean_dir / texture / "clean_image"
    if nested.is_dir():
        return _images(nested)
    tex = clean_dir / texture
    return _images(tex) if tex.is_dir() else _images(clean_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True, type=pathlib.Path)
    parser.add_argument("--defect_desc", required=True, type=pathlib.Path, help="defect_spec JSONL")
    parser.add_argument("--allocation", required=True, type=pathlib.Path, help="JSON {defect_type: n}.")
    parser.add_argument(
        "--output_pair_path",
        required=True,
        type=pathlib.Path,
        help="Where to write the pairing JSON (roi_place --input_pair_path). "
        "n_seeds is written to <output_pair_path>.n_seeds alongside it.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for per-defect submask/clean shuffles.")
    args = parser.parse_args(argv)

    entries = [json.loads(line) for line in args.defect_desc.read_text().splitlines() if line.strip()]
    labels_path = args.dataset_dir / "semantic_segmentation_labels.json"
    allocation = json.loads(args.allocation.read_text())

    # Pass 1: gather per-defect pools and compute n_seeds.
    pools = {}  # defect_type -> (submasks, clean_imgs, sd, texture)
    for e in entries:
        full = e["defect_type"]
        sd = e.get("spatial_dependency", "free")
        texture, anomaly = full.split("+", 1)

        n = allocation.get(full, 0)
        if n == 0:
            continue

        submasks = _images(args.dataset_dir / texture / "mask" / anomaly)
        if not submasks:
            log.warning(f"no submasks for {full}")
            continue
        clean_imgs = _clean_pool(args.dataset_dir, texture)
        if not clean_imgs:
            log.warning(f"no clean images for {full}")
            continue
        pools[full] = (submasks, clean_imgs, sd, texture)

    # Per defect: only the types whose request outruns their own (submask, clean) budget need AMP
    # multi-placement. A single global max would make one starved type inflate every other type's
    # seed count, and with it their record counts — so a type with 1 clean image would push the whole
    # run's totals up by rounding.
    n_seeds_by_type = {}
    for full, (submasks, clean_imgs, _, _) in pools.items():
        pair_budget = len(submasks) * len(clean_imgs)
        n_seeds_by_type[full] = max(1, math.ceil(allocation[full] / pair_budget)) if pair_budget else 1

    # Pass 2: emit records — iterate every (submask, clean) pair.
    def _cad_fields(sd, texture, clean_img):
        if sd != "cad":
            return None, None
        cad_mask_path = args.dataset_dir / texture / "cad_mask" / f"{clean_img.stem}.png"
        if not cad_mask_path.exists():
            return "__SKIP__", None
        return str(cad_mask_path), str(labels_path)

    records = []
    for full, (submasks, clean_imgs, sd, texture) in pools.items():
        n_seeds = n_seeds_by_type[full]
        n_records = math.ceil(allocation[full] / n_seeds)

        rng = random.Random(f"{args.seed}:{full}")
        submasks = list(submasks)
        rng.shuffle(submasks)
        clean_imgs = list(clean_imgs)
        rng.shuffle(clean_imgs)

        S = len(submasks)
        C = len(clean_imgs)

        # ⌈alloc/n_seeds⌉ records of n_seeds placements each overshoots whenever n_seeds does not
        # divide the allocation, so the final record takes only the remainder. Without this a request
        # of 9 at n_seeds=2 yields 5 x 2 = 10 images — num_sdg would be a floor, not a target.
        remainder = allocation[full] - n_seeds * (n_records - 1)
        for i in range(n_records):
            submask_idx = i % S
            round_k = i // S
            clean_idx = (submask_idx + round_k) % C
            submask = submasks[submask_idx]
            clean_img = clean_imgs[clean_idx]
            cad_mask, cad_mask_label = _cad_fields(sd, texture, clean_img)
            if cad_mask == "__SKIP__":
                continue
            records.append(
                {
                    "clean_image": str(clean_img),
                    "defect_type": full,
                    "submask": str(submask),
                    "name": f"{clean_img.stem}__{submask.stem}",
                    "cad_mask": cad_mask,
                    "cad_mask_label": cad_mask_label,
                    "n_seeds": n_seeds if i < n_records - 1 else remainder,
                }
            )

    args.output_pair_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_pair_path.write_text(json.dumps(records, indent=2))
    # The sidecar carries the max so roi_place's scalar --n_seeds stays a safe upper bound for callers
    # that do not read the per-record value (and so the seed stride never overlaps). roi_place prefers
    # each record's own "n_seeds" when present.
    max_seeds = max(n_seeds_by_type.values(), default=1)
    sidecar = args.output_pair_path.with_suffix(args.output_pair_path.suffix + ".n_seeds")
    sidecar.write_text(str(max_seeds))
    log.info(
        f"wrote {len(records)} records to {args.output_pair_path} (n_seeds max={max_seeds}, per type={n_seeds_by_type})"
    )


if __name__ == "__main__":
    main()
