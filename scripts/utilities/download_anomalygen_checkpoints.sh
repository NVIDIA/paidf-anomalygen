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
#
# Download the published finetuned AnomalyGen checkpoints from Hugging Face.
# These are the per-UC 2B Cosmos-Predict2 fine-tunes published under nvidia/:
#
#   --uc pcb    -> nvidia/Cosmos-AnomalyGen-PCB-2B    (iter_14000 + ag_config)
#   --uc metal  -> nvidia/Cosmos-AnomalyGen-Metal-2B  (iter_10000 + ag_config)
#   --uc glass  -> nvidia/Cosmos-AnomalyGen-Glass-2B  (iter_9000  + ag_config)
#   --uc all    -> all three
#
# These are end-of-finetune checkpoints. Pass the resulting directory to the
# AnomalyGen pipeline as `checkpoint_dir=` to run mode=inference_only without
# retraining. The companion `ag_config.yaml` next to the checkpoint lists the
# supported anomaly types and trained image_size.
#
# Usage:
#   download_anomalygen_checkpoints.sh --uc {pcb|metal|glass|all} [--checkpoint-dir checkpoints]
#
# Requires `HF_TOKEN` exported and the `hf` CLI (huggingface_hub >= 1.x)
# available on PATH. Idempotent — skips a UC whose target directory already
# contains an `ag_config.yaml`.
set -euo pipefail

ckpt_dir="checkpoints"
uc=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --uc)             uc="$2"; shift 2;;
        --checkpoint-dir) ckpt_dir="$2"; shift 2;;
        -h|--help)        sed -n '2,35p' "$0"; exit 0;;
        *) echo "error: unknown arg $1" >&2; exit 2;;
    esac
done

case "${uc}" in
    pcb|metal|glass|all) ;;
    "")  echo "error: --uc {pcb|metal|glass|all} is required" >&2; exit 2;;
    *)   echo "error: --uc must be one of pcb|metal|glass|all (got '${uc}')" >&2; exit 2;;
esac

: "${HF_TOKEN:?HF_TOKEN must be exported (Hugging Face access token)}"
command -v hf >/dev/null 2>&1 \
    || { echo "error: hf CLI not found in active env (huggingface_hub >= 1.x required)" >&2; exit 2; }

declare -A repo_for=(
    [pcb]="nvidia/Cosmos-AnomalyGen-PCB-2B"
    [metal]="nvidia/Cosmos-AnomalyGen-Metal-2B"
    [glass]="nvidia/Cosmos-AnomalyGen-Glass-2B"
)

if [[ "${uc}" == "all" ]]; then
    selected=(pcb metal glass)
else
    selected=("${uc}")
fi

echo "[setup] HF auth"
hf auth login --token "${HF_TOKEN}" --add-to-git-credential >/dev/null

mkdir -p "${ckpt_dir}"

for u in "${selected[@]}"; do
    repo="${repo_for[$u]}"
    target="${ckpt_dir}/${repo}"
    if [[ -f "${target}/ag_config.yaml" ]]; then
        echo "[skip] ${repo} already present at ${target}"
        continue
    fi
    echo "[fetch] hf download ${repo} --local-dir ${target}"
    hf download "${repo}" --local-dir "${target}"
done

echo "[done] checkpoints staged under ${ckpt_dir}/nvidia/"
echo "       point AnomalyGen at one of them via checkpoint_dir=<target> when running mode=inference_only"
