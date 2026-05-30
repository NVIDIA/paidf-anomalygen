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
# =============================================================================
# Build script for PAIDF AnomalyGen Docker image
#
# Usage:
#   bash docker/build.sh              # default CUDA 12.8
#   bash docker/build.sh --push       # build and push to NGC
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

BASE_IMAGE="nvcr.io/nvidia/pytorch:25.02-py3"
VLLM_VER="0.10.2"
REQUIREMENTS="requirements-conda-cuda128.txt"
IMAGE_TAG="${IMAGE_TAG:-paidf-anomalygen:cuda12.8}"

PUSH=false
for arg in "$@"; do
    case $arg in
        --push) PUSH=true ;;
    esac
done

echo "============================================="
echo "  Building:    ${IMAGE_TAG}"
echo "  Base:        ${BASE_IMAGE}"
echo "  Context:     ${REPO_ROOT}"
echo "============================================="

docker build \
    -f "${DOCKERFILE}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg VLLM_VER="${VLLM_VER}" \
    --build-arg REQUIREMENTS="${REQUIREMENTS}" \
    -t "${IMAGE_TAG}" \
    "${REPO_ROOT}"

echo ""
echo "Done. Image tagged as: ${IMAGE_TAG}"

if [ "${PUSH}" = true ] && [ -n "${NGC_IMAGE_PATH:-}" ]; then
    echo "Pushing to ${NGC_IMAGE_PATH}..."
    docker tag "${IMAGE_TAG}" "${NGC_IMAGE_PATH}:latest"
    docker push "${NGC_IMAGE_PATH}:latest"
    echo "Push complete."
fi
