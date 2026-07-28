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
# Download every pretrained checkpoint AnomalyGen needs end-to-end (FT, AMP,
# SDG, eval). Delegates to the in-repo `scripts.download_checkpoints` module,
# which handles Cosmos-Predict2, T5, NVDINOV2, C-RADIO-V3, dinov2-large,
# SAM2, and Qwen3-VL.
#
# Usage:
#   download_checkpoints.sh [--checkpoint-dir checkpoints]
#                           [--model-sizes "2B"] [--with-t5-11b]
#
# Options:
#   --checkpoint-dir DIR   Where to write checkpoints (default: checkpoints).
#   --model-sizes "LIST"   Space-separated Cosmos-Predict2 base sizes to fetch,
#                          from {2B, 14B} (default: 2B). Quote if more than one,
#                          e.g. --model-sizes "2B 14B".
#   --with-t5-11b          Also download google-t5/t5-11b (T5-XXL, ~45GB). Off by
#                          default: AnomalyGen uses t5-large; t5-11b is only
#                          needed when a config sets t5_model_name to it.
#
# Assumes the `cosmos-predict2` conda env is already active. Requires
# HF_TOKEN exported. Idempotent — the upstream module is invoked only when
# one of its artifacts is missing.
set -euo pipefail

ckpt_dir="checkpoints"
model_sizes=(2B)
with_t5_11b=0

usage() { sed -n '21,32p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) ckpt_dir="$2"; shift 2;;
        --model-sizes)    read -r -a model_sizes <<< "$2"; shift 2;;
        --with-t5-11b)    with_t5_11b=1; shift;;
        -h|--help)        usage; exit 0;;
        *) echo "error: unknown arg $1" >&2; exit 2;;
    esac
done

(( ${#model_sizes[@]} )) || { echo "error: --model-sizes cannot be empty" >&2; exit 2; }
for size in "${model_sizes[@]}"; do
    [[ "${size}" == "2B" || "${size}" == "14B" ]] \
        || { echo "error: --model-sizes must be from {2B, 14B}, got '${size}'" >&2; exit 2; }
done

: "${HF_TOKEN:?HF_TOKEN must be exported (Hugging Face access token)}"
command -v hf >/dev/null 2>&1 \
    || { echo "error: hf CLI not found in active env (huggingface_hub >= 1.x required)" >&2; exit 2; }
command -v python >/dev/null 2>&1 \
    || { echo "error: python not found in active env" >&2; exit 2; }

mkdir -p "${ckpt_dir}"

# Skip the upstream module call only if every artifact it would produce is
# already on disk — avoids the unconditional NVDINOV2 wget overwrite. The set
# depends on the requested --model-sizes / --with-t5-11b.
expect=()
for size in "${model_sizes[@]}"; do
    expect+=("${ckpt_dir}/nvidia/Cosmos-Predict2-${size}-Text2Image/model.pt")
done
expect+=(
    "${ckpt_dir}/google-t5/t5-large/config.json"
    "${ckpt_dir}/nvidia/Cosmos-Guardrail1/video_content_safety_filter/safety_filter.pt"
    "${ckpt_dir}/nvidia/Cosmos-Guardrail1/face_blur_filter/Resnet50_Final.pth"
    "${ckpt_dir}/NVDINOV2/nv_dinov2_classification_model.ckpt"
    "${ckpt_dir}/nvidia/C-RADIO-V3/model.safetensors"
    "${ckpt_dir}/facebook/dinov2-large/config.json"
    "${ckpt_dir}/sam2/sam2.1_hiera_large.pt"
    "${ckpt_dir}/Qwen/Qwen3-VL-4B-Instruct/model-00001-of-00002.safetensors"
    "${ckpt_dir}/Qwen/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors"
)
[[ "${with_t5_11b}" == "1" ]] && expect+=("${ckpt_dir}/google-t5/t5-11b/config.json")

upstream_present=1
for path in "${expect[@]}"; do
    [[ -f "${path}" ]] || { echo "[need] ${path}"; upstream_present=0; }
done

if [[ "${upstream_present}" == "0" ]]; then
    echo "[setup] HF auth"
    hf auth login --token "${HF_TOKEN}" --add-to-git-credential >/dev/null
    t5_flag=()
    [[ "${with_t5_11b}" == "1" ]] && t5_flag=(--with_t5_11b)
    echo "[fetch] python -m scripts.download_checkpoints --model_types text2image --model_sizes ${model_sizes[*]} ${t5_flag[*]}"
    CUDA_HOME="${CONDA_PREFIX:-}" python -m scripts.download_checkpoints \
        --model_types text2image \
        --model_sizes "${model_sizes[@]}" \
        "${t5_flag[@]}" \
        --checkpoint_dir "${ckpt_dir}"
else
    echo "[skip] all upstream-handled artifacts present"
fi

echo "[done] run scripts/utilities/check.sh to verify"
