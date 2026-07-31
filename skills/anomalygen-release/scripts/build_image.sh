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
# Build an AnomalyGen product or develop Docker image from docker/Dockerfile.
set -euo pipefail

mode="product"
image_name=""
tag="$(date -u +%Y%m%d)"
dockerfile="docker/Dockerfile"

usage() {
    cat <<'EOF'
Usage:
  build_image.sh [--mode product|develop] [--tag TAG] [--image-name NAME]
                 [--dockerfile docker/Dockerfile]

Defaults:
  --mode product
  --tag current UTC date (YYYYMMDD)
  --image-name paidf-anomalygen for product mode
  --image-name paidf-anomalygen-dev for develop mode

Product images set ANOMALYGEN_PRODUCT_MODE=1 and lock production code.
Develop images leave ANOMALYGEN_PRODUCT_MODE unset and keep code writable.
EOF
}

if [[ "${ANOMALYGEN_PRODUCT_MODE:-}" == "1" ]]; then
    echo "error: refusing to build images inside ANOMALYGEN_PRODUCT_MODE=1 runtime" >&2
    echo "build product/develop containers from a normal clone or develop container" >&2
    exit 1
fi

# Docker is required. We deliberately do NOT auto-install it: a build script
# silently running sudo to modify apt sources, add GPG keyrings, install system
# packages, and enable a system service is a significant, hard-to-reverse host
# change that a user running a "build image" command would not expect. If Docker
# is missing, install it yourself and re-run.
if ! command -v docker &>/dev/null; then
    echo "error: docker not found in PATH." >&2
    echo "       Install Docker Engine, then re-run this script:" >&2
    echo "         https://docs.docker.com/engine/install/" >&2
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       mode="$2";       shift 2;;
        --tag)        tag="$2";        shift 2;;
        --image-name) image_name="$2"; shift 2;;
        --dockerfile) dockerfile="$2"; shift 2;;
        -h|--help)    usage; exit 0;;
        *) echo "error: unknown arg $1" >&2; usage >&2; exit 2;;
    esac
done

case "${mode}" in
    product|develop) ;;
    *) echo "error: --mode must be product or develop (got ${mode})" >&2; exit 2;;
esac

if [[ -z "${image_name}" ]]; then
    if [[ "${mode}" == "product" ]]; then
        image_name="paidf-anomalygen"
    else
        image_name="paidf-anomalygen-dev"
    fi
fi

[[ -f "${dockerfile}" ]] || { echo "error: ${dockerfile} not found" >&2; exit 1; }

# Preflight the conda spec + requirements file the chosen Dockerfile needs.
# The arm64 / CUDA-13 Dockerfile pins the cuda130 inputs; everything else the
# cuda128 inputs.
case "${dockerfile}" in
    *arm.cuda130*) conda_yaml="cosmos-predict2-cuda130.yaml"; req_txt="requirements-conda-cuda130.txt";;
    *)             conda_yaml="cosmos-predict2-cuda128.yaml"; req_txt="requirements-conda-cuda128.txt";;
esac
[[ -f "${conda_yaml}" ]] || { echo "error: ${conda_yaml} not found" >&2; exit 1; }
[[ -f "${req_txt}" ]]    || { echo "error: ${req_txt} not found" >&2; exit 1; }

image="${image_name}:${tag}"

# Use sudo for docker only when the daemon isn't reachable as the current user
# (override with DOCKER_SUDO). Avoids running the build as root unnecessarily.
if [[ -n "${DOCKER_SUDO+x}" ]]; then
    SUDO="${DOCKER_SUDO}"
elif docker info >/dev/null 2>&1; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "=== building ${mode} image ${image} with ${dockerfile} ==="
echo "${SUDO:+${SUDO} }DOCKER_BUILDKIT=1 docker build --target ${mode} -f ${dockerfile} -t ${image} ."

${SUDO} DOCKER_BUILDKIT=1 docker build \
    --target "${mode}" \
    -f "${dockerfile}" \
    -t "${image}" \
    .

echo "=== built ${mode} image ${image} ==="
