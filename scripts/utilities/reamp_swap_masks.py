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
"""Swap mask_filename paths in a JSONL to point at a fresh AMP output.

Assumes `run_auto_roi_amp.py` was invoked with `--n_seeds <K> --seed <N>`
into --new-amp-dir, producing
<new-amp-dir>/<name>/<TEXTURE>+<ANOMALY>/<submask_stem>__seed{0..K-1}.png
for every record the original AMP pass emitted. For each JSONL row, derives
(name, full_type, seed_index) from the existing mask_filename and rewrites it
to the matching path under --new-amp-dir, **preserving the original seed
index**. This keeps the seed0 vs seed1 (etc.) split that prep-testcase
introduced for within-pair diversity intact under re-AMP.

This is used by sdg-refine's run_round.sh when --reamp-seed is set: the
(clean_image, submask) pairs stay the same, but the AMP augmentation is
re-rolled with a fresh base seed.
"""
import argparse
import json
import pathlib
import re
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-jsonl", required=True, type=pathlib.Path)
    p.add_argument("--new-amp-dir", required=True, type=pathlib.Path)
    p.add_argument("--output", required=True, type=pathlib.Path)
    p.add_argument("--seed-index", type=int, default=None,
                   help="If set, force every row to this seed index (overrides per-row preservation). "
                        "Default: preserve each row's original seed index from its mask_filename.")
    args = p.parse_args()

    rows = [json.loads(l) for l in args.base_jsonl.read_text().splitlines() if l.strip()]
    written, missing = 0, 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fp:
        for row in rows:
            old = pathlib.Path(row["mask_filename"])
            # old path is <amp>/<name>/<full_type>/<submask_stem>__seed<N>.png
            name, full_type = old.parent.parent.name, old.parent.name
            if args.seed_index is None:
                seed_stem = old.stem
            else:
                # Strip only a trailing __seed<N>: a submask stem may itself
                # contain "__seed" mid-name.
                base = re.sub(r"__seed\d+$", "", old.stem)
                seed_stem = f"{base}__seed{args.seed_index}"
            new = args.new_amp_dir / name / full_type / f"{seed_stem}.png"
            if not new.exists():
                print(f"warn: missing re-AMPed mask {new}", file=sys.stderr)
                missing += 1
                continue
            row["mask_filename"] = str(new)
            fp.write(json.dumps(row) + "\n")
            written += 1
    print(f"rewrote {written} rows, dropped {missing}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
