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
"""Phase 7: filter searched/ by nn_threshold, then re-AMP + re-pair to
regenerate independent replacement samples (within the same defect type)
up to 5 attempts. Falls back to best-per-defect-type if regen
cannot top up to `num_sdg`. Overwrites --searched-dir in place.

What each phase varies:
  * Phase 5 rounds — same (clean, submask), vary (guidance, crop_ratio).
  * Phase 7 regen  — same defect_type, NEW (clean, submask) pairing via
                      build_amp_samples.py --seed=attempt_seed, then SDG
                      with default (guidance, crop_ratio).

Regen samples are independent of the dropped originals — they do NOT
preserve the dropped sample's base JSONL index. Per-defect allocation is
preserved: if Phase 5 left 3 dropped stains and 1 dropped scratch, regen
will attempt to produce 3 fresh stains and 1 fresh scratch (and retry up
to 5 attempts if any still fail threshold).

Workflow per attempt:
  1. Compute needed_per_defect = target_alloc[d] - kept_per_defect[d].
     Skip if all zero.
  2. Build attempt allocation.json + amp_samples.json (--seed varies the
     submask/clean pairings).
  3. Run AMP (--seed varies placement augmentation).
  4. Build attempt testcase.jsonl.
  5. SDG into regens/regen_NN/sdg/.
  6. Eval. Any new sample with nn_score >= threshold is admitted into the
     kept pool for its defect type.

Final assembly:
  * Passing originals from source bucket + admitted regens, up to
    target_alloc per defect.
  * If still short, top up from the best non-passing regen samples per
    defect, then from the dropped originals (last resort).

Outputs under --searched-dir (same layout as a normal SDG bucket):
    reconstructed_image/<defect>_<NNNNN>.png    (+ original_mask/, overlay_image/, original_image/)
    SDG_result.csv                              (rows merged from source + regens, with `source` column)

The `source` column tags each sample as:
    "original"  — from Phase 3 SDG that survived Phase 5 assemble
    "round_<N>" — Phase 5 search round N produced the best attempt
    "regen_<k>" — Phase 7 regen attempt k produced this sample (k = 1..5)

Plus --regens-dir/regen_summary.csv (richer audit; SDG_result.csv is the main
trace target for users):
    sample_index (-1 for regen), source, clean_image, mask_filename,
    prev_nn, nn_score, passed_threshold, output_filename
"""
import argparse
import csv
import json
import math
import pathlib
import shutil
import subprocess
import sys
from collections import defaultdict

MAX_REGEN_TRIES = 5
KINDS = ("reconstructed_image", "original_mask", "overlay_image", "original_image")
HERE = pathlib.Path(__file__).resolve().parent


def load_per_sample_nn(csv_path):
    out = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            m = r.get("nn_score", "")
            out[r["path"]] = float(m) if m else float("nan")
    return out


def load_sdg_rows(sdg_csv):
    with open(sdg_csv) as f:
        return list(csv.DictReader(f))


def run_subprocess(cmd, label):
    print(f"=== {label} ===", flush=True)
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        print(f"error: {label} failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--searched-dir", required=True, type=pathlib.Path,
                   help="Phase 5 final bucket. Overwritten in place.")
    p.add_argument("--per-sample-csv", required=True, type=pathlib.Path,
                   help="Per-sample CSV from eval on --searched-dir.")
    p.add_argument("--threshold", required=True, type=float,
                   help="Keep samples with nn_score >= threshold. Set 0 to keep everything (no regen).")
    p.add_argument("--num-sdg", required=True, type=int,
                   help="Target final sample count.")
    p.add_argument("--rounds-dir", required=True, type=pathlib.Path,
                   help="Phase 5 rounds dir (used to read search_summary.csv for `source` labels).")
    p.add_argument("--regens-dir", required=True, type=pathlib.Path,
                   help="Phase 7 dir. Regen attempts written under regens_dir/regen_NN/; "
                        "regen_summary.csv written at regens_dir/regen_summary.csv.")
    p.add_argument("--real-path", required=True, type=pathlib.Path,
                   help="Dataset dir for run_eval.sh.")
    p.add_argument("--anomaly-types", required=True, nargs="+",
                   help="TEXTURE+TYPE names (for eval).")
    p.add_argument("--checkpoint-dir", required=True, type=pathlib.Path)
    p.add_argument("--step", required=True, type=int)
    # re-AMP inputs
    p.add_argument("--dataset-dir", required=True, type=pathlib.Path,
                   help="Dataset dir for build_amp_samples + run_auto_roi_amp.")
    p.add_argument("--clean-dir", default=None, type=pathlib.Path,
                   help="Clean images dir. Defaults to --dataset-dir.")
    p.add_argument("--defect-spec", required=True, type=pathlib.Path,
                   help="JSONL of defect_type + spatial_dependency (for AMP routing).")
    p.add_argument("--guidance", default=7.0, type=float,
                   help="Guidance for regen SDG (default 7.0, same as Phase 3).")
    p.add_argument("--crop-ratio", default=2.0, type=float,
                   help="Crop ratio for regen SDG (default 2.0).")
    p.add_argument("--model-size", default="2b", choices=["2b", "14b"])
    p.add_argument("--num-gpus", default=1, type=int)
    p.add_argument("--base-seed", default=1000, type=int,
                   help="Regen seed base. Attempt k uses seed = base_seed * k + 1.")
    args = p.parse_args()

    if args.clean_dir is None:
        args.clean_dir = args.dataset_dir

    sdg_csv = args.searched_dir / "SDG_result.csv"
    if not sdg_csv.exists():
        print(f"error: {sdg_csv} not found", file=sys.stderr)
        sys.exit(1)

    src_sdg = load_sdg_rows(sdg_csv)
    src_nn = load_per_sample_nn(args.per_sample_csv)
    if not src_sdg:
        print(f"error: {sdg_csv} has no rows", file=sys.stderr)
        sys.exit(1)

    # Map sample_index -> best_round from Phase 5's search_summary.csv so
    # source-bucket samples carry the right "original" vs "round_N" label.
    # Missing file → all source samples labeled "original".
    best_round_by_idx = {}
    search_summary_path = args.rounds_dir / "search_summary.csv"
    if search_summary_path.exists():
        for r in csv.DictReader(open(search_summary_path)):
            best_round_by_idx[int(r["sample_index"])] = int(r.get("best_round") or 0)

    def label_source(source_attempt, base_idx):
        if source_attempt > 0:
            return f"regen_{source_attempt}"
        br = best_round_by_idx.get(base_idx, 0)
        return "original" if br == 0 else f"round_{br}"

    # Target allocation per defect (from current source bucket).
    target_alloc = defaultdict(int)
    for row in src_sdg:
        target_alloc[row["anomaly_type"]] += 1

    # Partition source samples into passing / dropped per defect.
    # Each entry: {row, nn, src_basename}
    passing_by_defect = defaultdict(list)
    dropped_by_defect = defaultdict(list)
    for row in src_sdg:
        defect = row["anomaly_type"]
        nn = src_nn.get(row["output_filename"], float("nan"))
        if math.isnan(nn):
            continue
        entry = {
            "row": row,
            "nn": nn,
            "src_dir": args.searched_dir,
            "src_basename": pathlib.Path(row["output_filename"]).name,
            "source_attempt": 0,
            "prev_nn": nn,  # for source samples, prev_nn == final nn
        }
        if nn >= args.threshold:
            passing_by_defect[defect].append(entry)
        else:
            dropped_by_defect[defect].append(entry)

    # Regen pool — admitted (passing) regens per defect type, plus a
    # parallel "all_regens" list per defect for fallback.
    admitted_regens_by_defect = defaultdict(list)
    all_regens_by_defect = defaultdict(list)

    args.regens_dir.mkdir(parents=True, exist_ok=True)

    attempt = 0
    while True:
        # How many more of each defect do we still need?
        needed = {}
        for d, target in target_alloc.items():
            have = len(passing_by_defect[d]) + len(admitted_regens_by_defect[d])
            shortfall = target - have
            if shortfall > 0:
                needed[d] = shortfall

        total_pass = sum(len(passing_by_defect[d]) + len(admitted_regens_by_defect[d])
                         for d in target_alloc)
        print(f"=== filter check (after attempt {attempt}): "
              f"{total_pass}/{args.num_sdg} pass threshold {args.threshold} "
              f"(needed_per_defect={needed}) ===", flush=True)

        if not needed:
            break
        if attempt >= MAX_REGEN_TRIES:
            print(f"=== max regen tries ({MAX_REGEN_TRIES}) exceeded; "
                  f"will fall back to best-per-defect ===", flush=True)
            break
        attempt += 1
        attempt_seed = args.base_seed * attempt + 1

        attempt_dir = args.regens_dir / f"regen_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write attempt allocation.json for the needed defects.
        alloc_path = attempt_dir / "allocation.json"
        alloc_path.write_text(json.dumps(needed, indent=2, sort_keys=True))

        # 2. Build subset amp_samples.json via build_amp_samples.py with
        # --seed=attempt_seed (shuffles submask/clean pools per defect).
        amp_samples_path = attempt_dir / "amp_samples.json"
        run_subprocess([
            "python3", HERE / "build_amp_samples.py",
            "--dataset-dir", args.dataset_dir,
            "--clean-dir", args.clean_dir,
            "--defect-spec", args.defect_spec,
            "--allocation", alloc_path,
            "--output", amp_samples_path,
            "--seed", attempt_seed,
        ], f"regen attempt {attempt}: build_amp_samples (re-pair seed={attempt_seed})")

        # 3. Run AMP (re-runs placement with new seed).
        amp_dir = attempt_dir / "amp"
        amp_dir.mkdir(parents=True, exist_ok=True)
        n_seeds_path = amp_samples_path.with_suffix(amp_samples_path.suffix + ".n_seeds")
        n_seeds = int(n_seeds_path.read_text().strip()) if n_seeds_path.exists() else 1
        run_subprocess([
            "python3", "-m", "scripts.anomaly_gen.run_auto_roi_amp",
            "--input", amp_samples_path,
            "--defect-desc", args.defect_spec,
            "--output", amp_dir,
            "--n_seeds", n_seeds,
            "--seed", attempt_seed,
            "--model-id", "checkpoints/Qwen/Qwen3-VL-4B-Instruct",
        ], f"regen attempt {attempt}: AMP (n_seeds={n_seeds})")

        # 4. Build the attempt testcase.jsonl.
        attempt_jsonl = attempt_dir / "testcase.jsonl"
        run_subprocess([
            "python3", "-m", "scripts.utilities.build_jsonl",
            "--amp-output-dir", amp_dir,
            "--clean-dir", args.clean_dir,
            "--allocation", alloc_path,
            "--defect-types", *needed.keys(),
            "--guidance", args.guidance,
            "--crop-ratio", args.crop_ratio,
            "--output", attempt_jsonl,
        ], f"regen attempt {attempt}: build_jsonl")

        # Override seed per row so diffusion noise also varies attempt-to-attempt
        # (else AnomalyInpaintCondition.seed defaults to 1).
        with attempt_jsonl.open() as f:
            jrows = [json.loads(line) for line in f if line.strip()]
        with attempt_jsonl.open("w") as f:
            for r in jrows:
                r["seed"] = attempt_seed
                f.write(json.dumps(r) + "\n")

        # 5. SDG.
        sdg_out = attempt_dir / "sdg"
        run_subprocess([
            HERE / "run_sdg.sh",
            "--checkpoint_dir", args.checkpoint_dir,
            "--step", args.step,
            "--input_jsonl", attempt_jsonl,
            "--output_dir", sdg_out,
            "--model_size", args.model_size,
            "--num_gpus", args.num_gpus,
            "--seed", attempt_seed,
        ], f"regen attempt {attempt}: SDG")

        # 6. Eval. Only pass anomaly types actually present this attempt.
        present_types = sorted(needed.keys())
        run_subprocess([
            HERE / "run_eval.sh",
            "--real-path", args.real_path,
            "--generated-path", sdg_out,
            "--anomaly-types", *present_types,
        ], f"regen attempt {attempt}: eval")

        # Read new sdg rows + per_sample.csv; admit passing ones (greedy:
        # highest score first per defect type, up to needed quota).
        attempt_sdg = load_sdg_rows(sdg_out / "SDG_result.csv")
        attempt_nn = load_per_sample_nn(sdg_out / "per_sample.csv")
        # Group new samples by defect type.
        new_by_defect = defaultdict(list)
        for row in attempt_sdg:
            defect = row["anomaly_type"]
            nn = attempt_nn.get(row["output_filename"], float("nan"))
            if math.isnan(nn):
                continue
            new_by_defect[defect].append({
                "row": row,
                "nn": nn,
                "src_dir": sdg_out,
                "src_basename": pathlib.Path(row["output_filename"]).name,
                "source_attempt": attempt,
                "prev_nn": float("nan"),
            })

        # Sort within each defect by nn descending; admit top-needed[d] passing.
        admitted_count = 0
        for defect, samples in new_by_defect.items():
            samples.sort(key=lambda s: -s["nn"])
            all_regens_by_defect[defect].extend(samples)
            quota_left = needed.get(defect, 0)
            for s in samples:
                if quota_left <= 0:
                    break
                if s["nn"] >= args.threshold:
                    admitted_regens_by_defect[defect].append(s)
                    quota_left -= 1
                    admitted_count += 1
        print(f"  regen attempt {attempt}: admitted {admitted_count} new samples "
              f"across {len(new_by_defect)} defect types", flush=True)

    # Final assembly: passing source + admitted regens + fallback fills.
    final_entries = []  # list of entry dicts
    for defect, target in target_alloc.items():
        kept = list(passing_by_defect[defect])  # source passing first
        kept.extend(admitted_regens_by_defect[defect])
        # If still short for this defect: fill from best non-admitted regens,
        # then from dropped originals (last resort).
        if len(kept) < target:
            non_admitted = [s for s in all_regens_by_defect[defect]
                            if s not in admitted_regens_by_defect[defect]]
            non_admitted.sort(key=lambda s: -s["nn"])
            for s in non_admitted:
                if len(kept) >= target:
                    break
                kept.append(s)
        if len(kept) < target:
            # Last resort: keep the highest-scoring dropped originals.
            dropped_sorted = sorted(dropped_by_defect[defect],
                                    key=lambda s: -s["nn"])
            for s in dropped_sorted:
                if len(kept) >= target:
                    break
                kept.append(s)
        # If somehow we have too many (shouldn't happen), trim by nn desc.
        if len(kept) > target:
            kept.sort(key=lambda s: -s["nn"])
            kept = kept[:target]
        final_entries.extend(kept)

    # Stage into a sibling dir, then atomic-ish swap into --searched-dir.
    staging = args.searched_dir.with_name(args.searched_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for k in KINDS:
        (staging / k).mkdir(parents=True)

    # Re-number outputs by defect type, sequentially, so basenames are
    # always <defect>_<NNNNN>.png in a stable order. The original source
    # bucket already used this scheme; regens get appended.
    seq_per_defect = defaultdict(int)
    out_rows = []
    summary_rows = []
    src_fields = list(src_sdg[0].keys())
    if "nn_score" not in src_fields:
        src_fields.append("nn_score")

    # Group final_entries by defect for re-numbering; original passing
    # samples first (keep their order), then regens.
    final_by_defect = defaultdict(list)
    for s in final_entries:
        final_by_defect[s["row"]["anomaly_type"]].append(s)

    flat_final = []
    for defect, samples in final_by_defect.items():
        # Stable: source samples first (source_attempt=0), regens after.
        samples.sort(key=lambda s: (s["source_attempt"], -s["nn"]))
        flat_final.extend(samples)

    # Make sure SDG_result.csv carries a `source` column for quick at-a-glance
    # tracing without having to cross-reference regen_summary.csv.
    if "source" not in src_fields:
        src_fields.append("source")

    for s in flat_final:
        defect = s["row"]["anomaly_type"]
        seq = seq_per_defect[defect]
        seq_per_defect[defect] += 1
        new_basename = f"{defect}_{seq:05d}.png"

        # Copy KINDS files from src_dir under src_basename to staging under new_basename.
        for k in KINDS:
            src_f = s["src_dir"] / k / s["src_basename"]
            if src_f.exists():
                shutil.copy2(src_f, staging / k / new_basename)

        # Compute source label: "original" | "round_<N>" | "regen_<k>".
        base_idx = int(s["row"].get("index", "-1"))
        source_label = label_source(s["source_attempt"], base_idx)

        # SDG_result.csv row: use the source row but with normalized fields.
        new_row = dict(s["row"])
        new_row["output_filename"] = str(args.searched_dir / "reconstructed_image" / new_basename)
        new_row["nn_score"] = f"{s['nn']:.6f}"
        new_row["source"] = source_label
        out_rows.append(new_row)

        sample_idx = base_idx
        if s["source_attempt"] > 0:
            sample_idx = -1  # regen samples don't map to base JSONL
        summary_rows.append({
            "sample_index": sample_idx,
            "source": source_label,
            "clean_image": s["row"].get("image_filename", ""),
            "mask_filename": s["row"].get("mask_filename", ""),
            "prev_nn": f"{s['prev_nn']:.6f}" if not math.isnan(s["prev_nn"]) else "",
            "nn_score": f"{s['nn']:.6f}",
            "passed_threshold": "1" if s["nn"] >= args.threshold else "0",
            "output_filename": new_basename,
        })

    # Write staging SDG_result.csv.
    with (staging / "SDG_result.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=src_fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in src_fields})

    # Write regens/regen_summary.csv.
    summary_path = args.regens_dir / "regen_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sample_index", "source", "clean_image", "mask_filename",
            "prev_nn", "nn_score", "passed_threshold", "output_filename"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    # Atomic-ish in-place swap.
    backup = args.searched_dir.with_name(args.searched_dir.name + ".old")
    if backup.exists():
        shutil.rmtree(backup)
    args.searched_dir.rename(backup)
    staging.rename(args.searched_dir)
    shutil.rmtree(backup, ignore_errors=True)

    # Phase 7 canonical eval on the final searched/ bucket — writes
    # searched/{per_sample.csv, eval.log} and re-merges nn_score into
    # searched/SDG_result.csv. Phase 6 (assemble) is stitch-only, so this
    # is the only eval that runs on the searched/ bucket. Per-sample
    # scores that were stitched into SDG_result.csv from source rounds
    # stay (run_eval.sh skips rows that already have nn_score), while
    # per_sample.csv + the aggregate log are produced fresh against the
    # final post-regen bucket contents.
    run_subprocess([
        HERE / "run_eval.sh",
        "--real-path", args.real_path,
        "--generated-path", args.searched_dir,
        "--anomaly-types", *args.anomaly_types,
        "--log-name", "eval.log",
    ], "Phase 7 final eval (searched/)")

    # Summary stats.
    pass_n = sum(1 for r in summary_rows if r["passed_threshold"] == "1")
    fallback_n = len(summary_rows) - pass_n
    by_source = defaultdict(int)
    for r in summary_rows:
        by_source[r["source"]] += 1
    print()
    print("=== filter+regen done ===")
    print(f"  total written: {len(summary_rows)} / target {args.num_sdg}")
    print(f"  passed threshold {args.threshold}: {pass_n}")
    print(f"  fallback (below threshold): {fallback_n}")
    print(f"  regen attempts used: {attempt} / max {MAX_REGEN_TRIES}")
    print(f"  source distribution: " +
          ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))
    print(f"  output: {args.searched_dir} (in-place update)")
    print(f"  audit: {summary_path}")


if __name__ == "__main__":
    main()
