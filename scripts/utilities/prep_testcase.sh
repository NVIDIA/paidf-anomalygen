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
# End-to-end prep-testcase: validate → allocate → build AMP sample list →
# run_auto_roi_amp.py (n_seeds=1) → build JSONL → verify.
#
# 1:1 invariant: num_sdg → allocation → N AMP records → N AMP masks → N
# JSONL rows. For validation, callers set num_sdg = total training mask
# count so every training mask appears once; for inference, num_sdg is the
# SDG target count.
#
# spatial_dependency routing (per defect_spec entry):
#   free → whole-image ROI (run_auto_roi_amp.py does it internally)
#   text  → text2roi (Qwen VL text2box + SAM2), needs roi_prompt_defect_location
#   cad  → cad2roi, needs <dataset>/<TEXTURE>/cad_mask/<stem>.png and
#          <dataset>/semantic_segmentation_labels.json
#
# Usage:
#   prep_testcase.sh \
#       --name <exp> \
#       --num-sdg N \
#       --dataset-dir <dir> \
#       --amp-output-dir <dir> \
#       --output-jsonl <path> \
#       --defect-spec <jsonl> \
#       [--clean-dir <dir>]  (default: --dataset-dir)
#       [--mode {inference, validation}]  (default: inference)
#       [--per-defect-counts <JSON>]  (inference only; e.g.
#           '{"IC+bridge":5,"passive_component+missing":10}')
#       [--guidance <F>] [--crop-ratio <F>]
#       [--seed <N>]   (default: 42, base seed for run_auto_roi_amp.py)
#
# Defect types are derived from --defect-spec.
set -euo pipefail

guidance=7.0
crop_ratio=2.0
base_seed=42
defect_spec=""
clean_dir=""
mode="inference"
per_defect_counts=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)                  name="$2";                  shift 2;;
        --num-sdg)               num_sdg="$2";               shift 2;;
        --dataset-dir)           dataset_dir="$2";           shift 2;;
        --clean-dir)             clean_dir="$2";             shift 2;;
        --amp-output-dir)        amp_output="$2";            shift 2;;
        --output-jsonl)          output_jsonl="$2";          shift 2;;
        --defect-spec)           defect_spec="$2";           shift 2;;
        --mode)                  mode="$2";                  shift 2;;
        --per-defect-counts)     per_defect_counts="$2";     shift 2;;
        --guidance)              guidance="$2";              shift 2;;
        --crop-ratio)            crop_ratio="$2";            shift 2;;
        --seed)                  base_seed="$2";             shift 2;;
        -h|--help)               sed -n '2,33p' "$0"; exit 0;;
        *) echo "error: unknown arg $1" >&2; exit 2;;
    esac
done

: "${name:?--name required}"
: "${num_sdg:?--num-sdg required}"
: "${dataset_dir:?--dataset-dir required}"
: "${amp_output:?--amp-output-dir required}"
: "${output_jsonl:?--output-jsonl required}"
: "${defect_spec:?--defect-spec required (see .agents/skills/anomalygen/assets/defect_spec_template.jsonl)}"
[[ -f "${defect_spec}" ]] || { echo "error: defect_spec not found: ${defect_spec}" >&2; exit 1; }
clean_dir="${clean_dir:-${dataset_dir}}"

readarray -t defect_types < <(python3 -c "
import json
for line in open('${defect_spec}'):
    line = line.strip()
    if line:
        print(json.loads(line)['defect_type'])
")
[[ ${#defect_types[@]} -gt 0 ]] || { echo "error: no defect_type entries in ${defect_spec}" >&2; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"

# 1. Fail-fast validation of the AMP input triple.
python3 "${HERE}/validate_amp_inputs.py" \
    --dataset-dir "${dataset_dir}" \
    --clean-dir "${clean_dir}" \
    --defect-spec "${defect_spec}"

# 2. Allocate num_sdg across defect types.
#    inference (default): uniform; per-defect-counts allowed.
#    validation: proportional to mask counts, ≥1 per defect.
alloc_json="$(dirname "${output_jsonl}")/allocation.json"
mkdir -p "$(dirname "${alloc_json}")"
alloc_args=(
    --num-sdg "${num_sdg}"
    --defect-types "${defect_types[@]}"
    --mask-path "${dataset_dir}"
    --output "${alloc_json}"
    --mode "${mode}"
)
[[ -n "${per_defect_counts}" ]] && alloc_args+=(--per-defect-counts "${per_defect_counts}")
python3 "${HERE}/allocate_samples.py" "${alloc_args[@]}"

# 3. Build the per-sample JSON for run_auto_roi_amp.py —
# ceil(allocation[defect] / n_seeds) records per defect, iterating every
# (submask, clean) combination once before any pair repeats.
amp_samples="$(dirname "${output_jsonl}")/amp_samples.json"
python3 "${HERE}/build_amp_samples.py" \
    --dataset-dir "${dataset_dir}" \
    --clean-dir "${clean_dir}" \
    --defect-spec "${defect_spec}" \
    --allocation "${alloc_json}" \
    --output "${amp_samples}"

# 4. Run AMP. n_seeds is auto-computed from allocation (max over defects of
# ceil(alloc[d] / (num_submasks[d] × num_cleans[texture]))) so every
# (submask, clean) combination is used before AMP multi-placement kicks in.
n_seeds="$(cat "${amp_samples}.n_seeds")"
mkdir -p "${amp_output}"
python3 -m scripts.anomaly_gen.run_auto_roi_amp \
    --input "${amp_samples}" \
    --defect-desc "${defect_spec}" \
    --output "${amp_output}" \
    --n_seeds "${n_seeds}" \
    --seed "${base_seed}" \
    --model-id checkpoints/Qwen/Qwen3-VL-4B-Instruct

# 5. Build SDG JSONL from AMP output — one row per AMP mask.
#    --dataset-dir lets build_jsonl.py auto-detect per-defect iteration mode.
python3 -m scripts.utilities.build_jsonl \
    --amp-output-dir "${amp_output}" \
    --clean-dir "${clean_dir}" \
    --dataset-dir "${dataset_dir}" \
    --allocation "${alloc_json}" \
    --defect-types "${defect_types[@]}" \
    --guidance "${guidance}" \
    --crop-ratio "${crop_ratio}" \
    --output "${output_jsonl}"

# 6. Verify + resize masks where needed.
python3 "${HERE}/verify_jsonl.py" \
    --jsonl "${output_jsonl}" \
    --cache-dir "$(dirname "${output_jsonl}")/resized_masks"

echo "=== prep-testcase done: ${output_jsonl} ==="
