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

# Ensure Docker is available; install if missing (Ubuntu/Debian only).
if ! command -v docker &>/dev/null; then
    echo "=== docker not found — installing Docker Engine (Ubuntu/Debian) ==="
    if ! command -v apt-get &>/dev/null; then
        echo "error: automatic Docker install only supports Ubuntu/Debian (apt-get not found)" >&2
        exit 1
    fi
    sudo apt-get update -qq
    sudo apt-get install -y ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    # shellcheck disable=SC1091
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y \
        docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin
    sudo systemctl enable --now docker
    echo "=== Docker $(docker --version) installed ==="
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
echo "=== building ${mode} image ${image} with ${dockerfile} ==="
echo "sudo DOCKER_BUILDKIT=1 docker build --target ${mode} -f ${dockerfile} -t ${image} ."

sudo DOCKER_BUILDKIT=1 docker build \
    --target "${mode}" \
    -f "${dockerfile}" \
    -t "${image}" \
    .

echo "=== built ${mode} image ${image} ==="
