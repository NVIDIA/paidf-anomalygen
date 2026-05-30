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
"""Build the per-sample JSON consumed by scripts/anomaly_gen/run_auto_roi_amp.py.

Pair budget per defect = **every (submask, clean) combination**:
`num_submasks[d] × num_cleans[texture]`. n_seeds > 1 (AMP multi-placement
on a repeated record) only kicks in once allocation exceeds that budget.
This maximizes (submask, clean) diversity before relying on AMP
augmentation to make up the count.

n_seeds is computed globally from the allocation:

    n_seeds = max_d ⌈allocation[d] / (num_submasks[d] × num_cleans[texture])⌉

(e.g., num_SDG=50 over 5 submasks × 10 cleans = 50 pair budget → n_seeds=1;
num_SDG=200 over the same budget → n_seeds=4.)

Per defect d we emit `⌈allocation[d] / n_seeds⌉` records. Each batch of
S records pairs every submask with a different clean; the next batch
shifts the per-submask clean offset by one. Combined with a per-defect
deterministic shuffle, every (submask, clean) combination is used once
before any pair repeats.

Record schema (consumed by scripts/anomaly_gen/run_auto_roi_amp.py):

    {"clean_image":     <path>,
     "defect_type":     "TEXTURE+ANOMALY",
     "submask":         <path to training mask>,
     "name":            "<clean_stem>__<submask_stem>",
     "cad_mask":        <path>  if spatial_dependency=="cad" else null,
     "cad_mask_label":  <path>  if spatial_dependency=="cad" else null}

Note: AMP's preprocess_submask defaults submask_split_largest=True, so a
disconnected training submask is reduced to its largest connected
component before placement. We do not set the field here — the True
default applies.

n_seeds is written to a sidecar file next to the output JSON so
prep_testcase.sh can pass it to run_auto_roi_amp.py without re-deriving it.
"""
import argparse
import json
import math
import pathlib
import random
import sys

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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True, type=pathlib.Path)
    p.add_argument("--clean-dir", required=True, type=pathlib.Path)
    p.add_argument("--defect-spec", required=True, type=pathlib.Path)
    p.add_argument("--allocation", required=True, type=pathlib.Path,
                   help="JSON {defect_type: n}.")
    p.add_argument("--output", required=True, type=pathlib.Path,
                   help="Where to write the sample list JSON. n_seeds is "
                        "written to <output>.n_seeds alongside it.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for per-defect submask/clean shuffles.")
    args = p.parse_args()

    entries = [json.loads(l) for l in args.defect_spec.read_text().splitlines() if l.strip()]
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
            print(f"warn: no submasks for {full}", file=sys.stderr); continue
        clean_imgs = _clean_pool(args.clean_dir, texture)
        if not clean_imgs:
            print(f"warn: no clean images for {full}", file=sys.stderr); continue
        pools[full] = (submasks, clean_imgs, sd, texture)

    n_seeds = 1
    for full, (submasks, clean_imgs, _, _) in pools.items():
        pair_budget = len(submasks) * len(clean_imgs)
        need = math.ceil(allocation[full] / pair_budget)
        if need > n_seeds:
            n_seeds = need

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
        n_records = math.ceil(allocation[full] / n_seeds)

        rng = random.Random(f"{args.seed}:{full}")
        submasks = list(submasks); rng.shuffle(submasks)
        clean_imgs = list(clean_imgs); rng.shuffle(clean_imgs)

        S = len(submasks)
        C = len(clean_imgs)

        for i in range(n_records):
            submask_idx = i % S
            round_k = i // S
            clean_idx = (submask_idx + round_k) % C
            submask = submasks[submask_idx]
            clean_img = clean_imgs[clean_idx]
            cad_mask, cad_mask_label = _cad_fields(sd, texture, clean_img)
            if cad_mask == "__SKIP__":
                continue
            records.append({
                "clean_image":    str(clean_img),
                "defect_type":    full,
                "submask":        str(submask),
                "name":           f"{clean_img.stem}__{submask.stem}",
                "cad_mask":       cad_mask,
                "cad_mask_label": cad_mask_label,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2))
    sidecar = args.output.with_suffix(args.output.suffix + ".n_seeds")
    sidecar.write_text(str(n_seeds))
    print(f"wrote {len(records)} records to {args.output} (n_seeds={n_seeds})")


if __name__ == "__main__":
    main()
