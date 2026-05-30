#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Validate filesystem and runtime mode permissions in a built AnomalyGen image.
set -euo pipefail

mode="product"

usage() {
    cat <<'EOF'
Usage:
  validate_image_permissions.sh [--mode product|develop] <image>

Product validation requires ANOMALYGEN_PRODUCT_MODE=1, a non-root runtime user,
read-only production code, and writable runtime artifact paths.

Develop validation requires ANOMALYGEN_PRODUCT_MODE to be unset and production
code paths to remain writable for agent-assisted development.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)    mode="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        --*)       echo "error: unknown arg $1" >&2; usage >&2; exit 2;;
        *)         image="${1}"; shift;;
    esac
done

case "${mode}" in
    product|develop) ;;
    *) echo "error: --mode must be product or develop (got ${mode})" >&2; exit 2;;
esac

if [[ -z "${image:-}" ]]; then
    echo "error: image is required" >&2
    usage >&2
    exit 2
fi

sudo docker image inspect "${image}" >/dev/null

sudo docker run --rm -e "VALIDATION_MODE=${mode}" "${image}" bash -lc '
set -euo pipefail

root="/workspace/paidf-anomalygen"
mode="${VALIDATION_MODE:?VALIDATION_MODE is required}"
errors=0

fail() {
    echo "FAIL: $*" >&2
    errors=$((errors + 1))
}

ok() {
    echo "OK: $*"
}

if [[ "${mode}" == "product" ]]; then
    if [[ "${ANOMALYGEN_PRODUCT_MODE:-}" == "1" ]]; then
        ok "ANOMALYGEN_PRODUCT_MODE=1"
    else
        fail "ANOMALYGEN_PRODUCT_MODE must be 1 in product images"
    fi
else
    if [[ -z "${ANOMALYGEN_PRODUCT_MODE:-}" ]]; then
        ok "ANOMALYGEN_PRODUCT_MODE is unset"
    else
        fail "ANOMALYGEN_PRODUCT_MODE must be unset in develop images"
    fi
fi

uid="$(id -u)"
if [[ "${mode}" == "product" ]]; then
    if [[ "${uid}" == "0" ]]; then
        fail "runtime user is root; product images must run as non-root"
    else
        ok "runtime user is non-root (uid=${uid})"
    fi
else
    ok "develop image runtime user uid=${uid}"
fi

production_paths=(
    "${root}/cosmos_predict2"
    "${root}/scripts/anomaly_gen"
    "${root}/scripts/utilities"
    "${root}/.agents/skills"
    "${root}/README.md"
    "${root}/CLAUDE.md"
    "${root}/cosmos-predict2-cuda128.yaml"
    "${root}"/docker/Dockerfile*
    "${root}"/requirements*.txt
)

for path in "${production_paths[@]}"; do
    [[ -e "${path}" ]] || { echo "WARN: expected production path not found: ${path}" >&2; continue; }
    if [[ "${mode}" == "product" ]]; then
        if [[ -w "${path}" ]]; then
            fail "production path is writable: ${path}"
        else
            ok "production path is non-writable: ${path}"
        fi
    else
        if [[ -w "${path}" ]]; then
            ok "develop path is writable: ${path}"
        else
            fail "develop path is not writable: ${path}"
        fi
    fi
done

writable_paths=(
    "${root}/results"
    "${root}/ag_configs"
    "${root}/ag_inference"
    "${root}/datasets"
    "${root}/checkpoints"
    "${root}/logs"
    "${root}/tmp"
    "/tmp"
    "${HOME}"
    "${HOME}/.cache"
)

for path in "${writable_paths[@]}"; do
    mkdir -p "${path}" 2>/dev/null || true
    if [[ -d "${path}" && -w "${path}" ]]; then
        ok "runtime path is writable: ${path}"
    else
        fail "runtime path is not writable: ${path}"
    fi
done

# Exercise the nested paths AnomalyGen actually writes during training, SDG,
# and refine. Checking only top-level directories can miss ownership issues on
# mounted volumes or pre-created subdirectories.
nested_write_targets=(
    "${root}/results/anomaly_gen/_permission_test/checkpoints/model"
    "${root}/results/_permission_test/original/reconstructed_image"
    "${root}/results/_permission_test/original/original_mask"
    "${root}/results/_permission_test/original/overlay_image"
    "${root}/results/_permission_test/searched/reconstructed_image"
    "${root}/results/_permission_test/rounds/round_001/sdg/reconstructed_image"
    "${root}/ag_inference/_permission_test/amp"
    "${root}/ag_inference/_permission_test/resized_masks"
    "${HOME}/.cache/paidf-anomalygen/torch_inductor"
    "${HOME}/.cache/paidf-anomalygen/triton"
    "${HOME}/.cache/huggingface"
)

for path in "${nested_write_targets[@]}"; do
    if mkdir -p "${path}" 2>/dev/null && touch "${path}/.write_test" 2>/dev/null; then
        ok "nested runtime path is writable: ${path}"
        rm -f "${path}/.write_test" 2>/dev/null || true
    else
        fail "nested runtime path is not writable: ${path}"
    fi
done

sdg_csv_targets=(
    "${root}/results/_permission_test/original/SDG_result.csv"
    "${root}/results/_permission_test/searched/SDG_result.csv"
    "${root}/results/_permission_test/rounds/round_001/sdg/SDG_result.csv"
)

for file in "${sdg_csv_targets[@]}"; do
    if mkdir -p "$(dirname "${file}")" 2>/dev/null && printf "index,output_filename\n" > "${file}" 2>/dev/null; then
        ok "SDG_result.csv target is writable: ${file}"
        rm -f "${file}" 2>/dev/null || true
    else
        fail "SDG_result.csv target is not writable: ${file}"
    fi
done

if (( errors > 0 )); then
    echo "BLOCKED: ${mode} image permission validation failed (${errors} issue(s))" >&2
    exit 1
fi

echo "READY: ${mode} image permission validation passed"
'
