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
# Build an AnomalyGen air-gapped Docker image (checkpoints baked in).
# Checks for required checkpoints and downloads any that are missing before
# building the airgapped-{product,develop} target of docker/Dockerfile.
# Checkpoints are passed via a named build context (`ckpts`) so they bypass
# the repo .dockerignore (which excludes checkpoints/ for thin builds).
#
# Usage:
#   build_airgapped_image.sh [--mode product|develop] [--tag TAG]
#                            [--checkpoint-dir checkpoints]
#                            [--dockerfile docker/Dockerfile]
#                            [--skip-download]
#
# Use --dockerfile docker/Dockerfile.arm.cuda130 to build the arm64 / CUDA-13
# air-gapped image instead of the default x86 / CUDA-12.8 one.
set -euo pipefail

mode="product"
tag="$(date -u +%Y%m%d)"
ckpt_dir="checkpoints"
skip_download=0
image_name=""
dockerfile="docker/Dockerfile"

usage() {
    cat <<'EOF'
Usage:
  build_airgapped_image.sh [--mode product|develop] [--tag TAG]
                           [--checkpoint-dir checkpoints]
                           [--dockerfile docker/Dockerfile]
                           [--skip-download]

Defaults:
  --mode product
  --tag current UTC date (YYYYMMDD)
  --checkpoint-dir checkpoints
  --dockerfile docker/Dockerfile  (use docker/Dockerfile.arm.cuda130 for arm64/CUDA-13)
  --skip-download  off (auto-downloads missing checkpoints)

Product images set ANOMALYGEN_PRODUCT_MODE=1 and lock production code.
Develop images leave ANOMALYGEN_PRODUCT_MODE unset and keep code writable.
Both image variants bake checkpoints into the image for air-gapped use.
EOF
}

if [[ "${ANOMALYGEN_PRODUCT_MODE:-}" == "1" ]]; then
    echo "error: refusing to build images inside ANOMALYGEN_PRODUCT_MODE=1 runtime" >&2
    echo "build product/develop containers from a normal clone or develop container" >&2
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)           mode="$2";         shift 2;;
        --tag)            tag="$2";          shift 2;;
        --checkpoint-dir) ckpt_dir="$2";     shift 2;;
        --dockerfile)     dockerfile="$2";   shift 2;;
        --skip-download)  skip_download=1;   shift;;
        --image-name)     image_name="$2";   shift 2;;
        -h|--help)        usage; exit 0;;
        *) echo "error: unknown arg $1" >&2; usage >&2; exit 2;;
    esac
done

case "${mode}" in
    product|develop) ;;
    *) echo "error: --mode must be product or develop (got ${mode})" >&2; exit 2;;
esac

[[ -f "${dockerfile}" ]] || {
    echo "error: ${dockerfile} not found — run from the repo root" >&2
    exit 1
}

if [[ -z "${image_name}" ]]; then
    if [[ "${mode}" == "product" ]]; then
        image_name="paidf-anomalygen-airgapped"
    else
        image_name="paidf-anomalygen-dev-airgapped"
    fi
fi

# ── Checkpoint preflight ───────────────────────────────────────────────────────
# The airgapped Dockerfile COPYs each of these paths into the image; all must
# be present before we start the (potentially multi-hour) build.

echo "=== checking required checkpoints in ${ckpt_dir}/ ==="
missing=0

check_file() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        printf "  [ok]      %s\n" "${path}"
    else
        printf "  [missing] %s\n" "${path}"
        missing=$((missing + 1))
    fi
}

check_nonempty_dir() {
    local path="$1"
    if [[ -d "${path}" ]] && [[ -n "$(ls -A "${path}" 2>/dev/null)" ]]; then
        printf "  [ok]      %s\n" "${path}"
    else
        printf "  [missing] %s\n" "${path}"
        missing=$((missing + 1))
    fi
}

check_file        "${ckpt_dir}/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt"
check_file        "${ckpt_dir}/nvidia/Cosmos-Predict2-14B-Text2Image/model.pt"
check_file        "${ckpt_dir}/NVDINOV2/nv_dinov2_classification_model.ckpt"
check_file        "${ckpt_dir}/nvidia/C-RADIO-V3/model.safetensors"
check_file        "${ckpt_dir}/sam2/sam2.1_hiera_large.pt"
check_nonempty_dir "${ckpt_dir}/facebook/dinov2-large"
check_nonempty_dir "${ckpt_dir}/facebook"
check_nonempty_dir "${ckpt_dir}/Qwen/Qwen3-VL-4B-Instruct"

# T5: at least one variant required (both are copied by the Dockerfile)
t5_ok=0
{ [[ -d "${ckpt_dir}/google-t5/t5-large" ]] && [[ -n "$(ls -A "${ckpt_dir}/google-t5/t5-large" 2>/dev/null)" ]]; } \
    && { t5_ok=1; printf "  [ok]      %s\n" "${ckpt_dir}/google-t5/t5-large"; }
{ [[ -d "${ckpt_dir}/google-t5/t5-11b" ]] && [[ -n "$(ls -A "${ckpt_dir}/google-t5/t5-11b" 2>/dev/null)" ]]; } \
    && { t5_ok=1; printf "  [ok]      %s\n" "${ckpt_dir}/google-t5/t5-11b"; }
if [[ "${t5_ok}" == "0" ]]; then
    printf "  [missing] %s\n" "${ckpt_dir}/google-t5/{t5-large,t5-11b} (need at least one)"
    missing=$((missing + 1))
fi

if [[ "${missing}" -gt 0 ]]; then
    echo
    echo "${missing} checkpoint(s) missing."
    if [[ "${skip_download}" == "1" ]]; then
        echo "error: pass without --skip-download to auto-download missing checkpoints" >&2
        exit 1
    fi
    echo
    echo "=== downloading missing checkpoints ==="
    echo "    (requires HF_TOKEN exported and huggingface-cli in PATH)"
    echo
    bash scripts/utilities/download_checkpoints.sh \
        --checkpoint-dir "${ckpt_dir}"
    echo
    echo "=== re-checking checkpoints after download ==="
    # Re-run the checks; abort if still missing (e.g. download failed).
    bash "$0" --mode "${mode}" --tag "${tag}" --checkpoint-dir "${ckpt_dir}" \
        --dockerfile "${dockerfile}" --image-name "${image_name}" --skip-download
    exit $?
fi

echo
echo "all required checkpoints present."

# ── Docker build ───────────────────────────────────────────────────────────────
# Use sudo for docker only when the daemon isn't reachable as the current user.
if [[ -n "${DOCKER_SUDO+x}" ]]; then
    SUDO="${DOCKER_SUDO}"
elif docker info >/dev/null 2>&1; then
    SUDO=""
else
    SUDO="sudo"
fi

image="${image_name}:${tag}"
echo
echo "=== building ${mode} air-gapped image: ${image} ==="
echo "    dockerfile: ${dockerfile}"
echo "    checkpoint size: $(du -sh "${ckpt_dir}" 2>/dev/null | cut -f1 || echo 'unknown')"
echo "    expected image size: ~75 GB+"
echo
echo "${SUDO:+${SUDO} }DOCKER_BUILDKIT=1 docker buildx build --load --target airgapped-${mode} --build-context ckpts=${ckpt_dir} -f ${dockerfile} -t ${image} ."

${SUDO} DOCKER_BUILDKIT=1 docker buildx build \
    --load \
    --target "airgapped-${mode}" \
    --build-context "ckpts=${ckpt_dir}" \
    -f "${dockerfile}" \
    -t "${image}" \
    .

echo
echo "=== built ${mode} air-gapped image: ${image} ==="
echo "    Run (no volume mounts needed):"
echo "    ${SUDO:+${SUDO} }docker run --gpus all -it --rm --shm-size=16g ${image} bash"
