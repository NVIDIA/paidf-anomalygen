#!/usr/bin/env bash
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
# Wrap scripts.anomaly_gen.evaluate. Always writes three files inside
# --generated-path:
#   per_sample.csv  — per-sample nn / mnn (path overridable via --per-sample-csv)
#   <log-name>      — aggregate KPI table (default eval.log)
#   SDG_result.csv  — same as before, but with nn_score column merged from
#                     per_sample.csv so the per-sample score is visible alongside
#                     the generation params for that row.
#
# Usage:
#   run_eval.sh \
#       --real-path <dir> \
#       --generated-path <dir> \
#       --anomaly-types <T+A> [<T+B> ...] \
#       [--per-sample-csv <path>] \
#       [--log-name <name>]              (default: eval.log)
#       [--backbone cradio_v3_base]
set -euo pipefail

backbone="cradio_v3_base"
per_sample_csv=""
log_name="eval.log"
anomaly_types=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --real-path)         real_path="$2";       shift 2;;
        --generated-path)    generated_path="$2";  shift 2;;
        --per-sample-csv)    per_sample_csv="$2";  shift 2;;
        --log-name)          log_name="$2";        shift 2;;
        --backbone)          backbone="$2";        shift 2;;
        --anomaly-types)     shift
            while [[ $# -gt 0 && "$1" != --* ]]; do anomaly_types+=("$1"); shift; done
            ;;
        -h|--help)           sed -n '2,18p' "$0"; exit 0;;
        *) echo "error: unknown arg $1" >&2; exit 2;;
    esac
done
: "${real_path:?--real-path required}"
: "${generated_path:?--generated-path required}"
[[ ${#anomaly_types[@]} -gt 0 ]] || { echo "error: --anomaly-types required" >&2; exit 2; }
: "${per_sample_csv:=${generated_path}/per_sample.csv}"

log_path="${generated_path}/${log_name}"

args=(-m scripts.anomaly_gen.evaluate
      --real_path "${real_path}"
      --generated_path "${generated_path}"
      --backbone "${backbone}"
      --anomaly_types "${anomaly_types[@]}"
      --per_sample_csv "${per_sample_csv}")

python3 "${args[@]}" 2>&1 | tee "${log_path}"
rc="${PIPESTATUS[0]}"
if [[ "${rc}" -ne 0 ]]; then
    exit "${rc}"
fi

# Upstream evaluate.py also writes <generated_path>/eval_stdout.log via loguru —
# same content as our tee'd log, just in loguru's format. Remove it to keep the
# bucket clean: one eval = one log file.
rm -f "${generated_path}/eval_stdout.log"

# Merge nn_score from per_sample.csv into SDG_result.csv so the per-sample
# score is visible at a glance in the row that produced it.
sdg_csv="${generated_path}/SDG_result.csv"
if [[ -f "${sdg_csv}" && -f "${per_sample_csv}" ]]; then
    python3 - "${sdg_csv}" "${per_sample_csv}" <<'PY'
import csv, sys
sdg_path, ps_path = sys.argv[1], sys.argv[2]
ps = {r["path"]: r.get("nn_score", "") for r in csv.DictReader(open(ps_path))}
with open(sdg_path) as f:
    reader = csv.DictReader(f)
    fields = list(reader.fieldnames or [])
    rows = list(reader)
if "nn_score" not in fields:
    fields.append("nn_score")
for r in rows:
    if r.get("nn_score"):
        continue  # don't clobber a richer value if already present (e.g. filter+regen)
    r["nn_score"] = ps.get(r["output_filename"], "")
with open(sdg_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
PY
fi
