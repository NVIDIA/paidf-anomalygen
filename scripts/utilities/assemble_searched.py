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
"""Assemble the searched/ bucket from the original pass + N search rounds.

For each sample index, pick the attempt (original or any round) with the
highest nn_score — NaN treated as "no score" — and copy that attempt's
reconstructed_image / original_mask / overlay_image / original_image plus
its SDG_result.csv row into --searched-dir. Also writes
<rounds-dir>/search_summary.csv listing per-sample best_round / best params /
best_nn_score / attempts.

This phase is stitch-only: it does NOT re-run eval. searched/per_sample.csv
is built by carrying over each picked sample's row from its source round's
per_sample.csv (per-sample nn_score is bucket-independent — the same image
gets the same correspondence-to-real-set score regardless of which siblings
share its bucket). Phase 7's filter+regen runs the canonical post-pipeline
eval on the final searched/ bucket and refreshes per_sample.csv there.

Copied basenames are prefixed `idx<NNNNNN>_` (and `output_filename` in the
merged CSV is rewritten to match). The per-anomaly-name `_NNNNN` counter
in SDG output restarts per run, so without the prefix, winners from
different rounds would clobber on copy.

Expected layout on disk:
    <original-dir>/{reconstructed_image,…}/<name>_<NNNNN>.png
    <original-dir>/SDG_result.csv
    <original-csv>                        # per-sample mnn for the original pass
    <rounds-dir>/round_001/sdg/{…}/<name>_<NNNNN>.png
    <rounds-dir>/round_001/per_sample.csv
    <rounds-dir>/round_001/draws.json     # {idx_str: {guidance,crop_ratio}}
    ...
"""
import argparse
import csv
import json
import math
import pathlib
import shutil
import sys


# ---------- pure-logic helpers (unit-tested) ----------

def _is_better(candidate, current):
    c = candidate.get("nn_score")
    cur = current.get("nn_score") if current else None
    if cur is None or (isinstance(cur, float) and math.isnan(cur)):
        return True
    if c is None or (isinstance(c, float) and math.isnan(c)):
        return False
    return c > cur


def merge_best_seen(histories):
    """histories: list of {sample_index -> record}. Returns {idx -> best record}."""
    merged = {}
    for h in histories:
        for idx, rec in h.items():
            if _is_better(rec, merged.get(idx)):
                merged[idx] = dict(rec)
    return merged


# ---------- I/O glue ----------

def _load_per_sample_csv(csv_path, sdg_csv):
    """Map {sample_index -> per-sample record}, resolving index via sdg_csv.

    Captures anomaly_type + mnn_score in addition to nn_score so the caller
    can stitch a complete searched/per_sample.csv without re-running eval.
    """
    idx_by_path = {}
    with open(sdg_csv) as f:
        for row in csv.DictReader(f):
            idx_by_path[row["output_filename"]] = int(row["index"])
    rows = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["path"] not in idx_by_path:
                raise ValueError(f"{row['path']} not in {sdg_csv}")
            idx = idx_by_path[row["path"]]
            rows[idx] = {
                "anomaly_type": row.get("anomaly_type", ""),
                "path": row["path"],
                "nn_score":  float(row["nn_score"])  if row["nn_score"]  else float("nan"),
                "mnn_score": float(row["mnn_score"]) if row.get("mnn_score") else float("nan"),
            }
    return rows


def _copy_sample_outputs(src_dir, dst_dir, idx, basename):
    src_dir = pathlib.Path(src_dir); dst_dir = pathlib.Path(dst_dir)
    dst_name = f"idx{idx:06d}_{basename}"
    for kind in ("reconstructed_image", "original_mask", "overlay_image", "original_image"):
        src_f = src_dir / kind / basename
        if src_f.exists():
            (dst_dir / kind).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst_dir / kind / dst_name)
    return dst_name


def _merge_sdg_csv(src_csv, dst_rows, idx, searched_dir, dst_name):
    with open(src_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["index"]) == idx:
                row = dict(row)
                row["output_filename"] = str(searched_dir / "reconstructed_image" / dst_name)
                dst_rows.append(row)


def _tag(history, source_dir, round_id, draws=None):
    for idx, rec in history.items():
        rec["_source_dir"] = str(source_dir)
        rec["_round"] = round_id
        if draws is not None and str(idx) in draws:
            rec["guidance"] = draws[str(idx)]["guidance"]
            rec["crop_ratio"] = draws[str(idx)]["crop_ratio"]
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original-dir", required=True, type=pathlib.Path)
    p.add_argument("--original-csv", required=True, type=pathlib.Path)
    p.add_argument("--rounds-dir", required=True, type=pathlib.Path,
                   help="Contains round_001/, round_002/, ... each with sdg/, per_sample.csv, draws.json.")
    p.add_argument("--searched-dir", required=True, type=pathlib.Path)
    args = p.parse_args()

    # Baseline = original pass
    original = _tag(_load_per_sample_csv(args.original_csv, args.original_dir / "SDG_result.csv"),
                    args.original_dir, 0)
    histories = [original]

    # Each round under rounds_dir
    round_dirs = sorted(d for d in args.rounds_dir.iterdir() if d.is_dir() and d.name.startswith("round_"))
    for rd in round_dirs:
        csv_path = rd / "per_sample.csv"
        sdg_csv = rd / "sdg" / "SDG_result.csv"
        draws_path = rd / "draws.json"
        if not csv_path.exists():
            print(f"warn: skipping {rd.name} (no per_sample.csv)", file=sys.stderr); continue
        draws = json.loads(draws_path.read_text()) if draws_path.exists() else None
        round_id = int(rd.name.rsplit("_", 1)[1])
        histories.append(_tag(_load_per_sample_csv(csv_path, sdg_csv), rd / "sdg", round_id, draws))

    best = merge_best_seen(histories)

    # Copy winning outputs + merged SDG_result.csv + stitched per_sample.csv.
    args.searched_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = []
    per_sample_rows = []
    for idx, rec in best.items():
        src = pathlib.Path(rec["_source_dir"])
        basename = pathlib.Path(rec["path"]).name
        dst_name = _copy_sample_outputs(src, args.searched_dir, idx, basename)
        _merge_sdg_csv(src / "SDG_result.csv", combined_rows, idx, args.searched_dir, dst_name)

        def _fmt(x):
            return "" if x is None or (isinstance(x, float) and math.isnan(x)) else repr(x)
        per_sample_rows.append({
            "anomaly_type": rec.get("anomaly_type", ""),
            "path": str(args.searched_dir / "reconstructed_image" / dst_name),
            "nn_score":  _fmt(rec.get("nn_score")),
            "mnn_score": _fmt(rec.get("mnn_score")),
        })

    if combined_rows:
        header = list(combined_rows[0].keys())
        with (args.searched_dir / "SDG_result.csv").open("w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=header); w.writeheader(); w.writerows(combined_rows)

    # Stitched per_sample.csv — values pulled from each picked sample's
    # source round per_sample.csv. Phase 7 overwrites this with a fresh
    # bucket-evaluated per_sample.csv after the regen-aware rebuild.
    with (args.searched_dir / "per_sample.csv").open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["anomaly_type", "path", "nn_score", "mnn_score"])
        w.writeheader()
        w.writerows(per_sample_rows)

    # Summary
    with (args.rounds_dir / "search_summary.csv").open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["sample_index", "best_round", "best_guidance", "best_crop_ratio",
                    "best_nn_score", "attempts"])
        for idx in sorted(best):
            rec = best[idx]
            attempts = sum(1 for h in histories if idx in h)
            w.writerow([idx, rec["_round"], rec.get("guidance", ""), rec.get("crop_ratio", ""),
                        rec["nn_score"], attempts])

    print(f"searched bucket: {args.searched_dir}")
    print(f"summary: {args.rounds_dir/'search_summary.csv'}")


if __name__ == "__main__":
    main()
