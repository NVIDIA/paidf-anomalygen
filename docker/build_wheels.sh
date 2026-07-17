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
# Build the heavy native wheels locally and upload them to the project's
# GitLab PyPI registry so subsequent Docker / CI builds can `pip install`
# instead of recompiling from source.
#
# Wheels produced (from docker/Dockerfile):
#   flash-attn                  2.8.3
#   transformer-engine          2.13.0   (wrapper)
#   transformer-engine-torch    2.13.0   (PyTorch backend, source-compiled)
#   transformer-engine-cu12     2.13.0   (CUDA-12 backend, mirrored from PyPI)
#   apex                        git 466e164b (--cpp_ext --cuda_ext)
#   opencv-python-headless      4.13.0.92  (GitHub tag `92`, WITH_FFMPEG=OFF)
#
# Usage:
#   GITLAB_TOKEN=<token-with-api-scope> ./docker/build_wheels.sh
#
# Required env:
#   GITLAB_TOKEN     project or personal access token with `api` scope
# Optional env:
#   GITLAB_PROJECT   project path or numeric id on gitlab-master.nvidia.com
#                    default: metropolis-perf/sdg/cosmos-anomalygen
#   MAX_JOBS         parallel jobs for native compiles
#                    (also used as MAKEFLAGS=-j$MAX_JOBS).
#                    default: 4 (matches docker/Dockerfile).

set -euo pipefail

case "${1:-}" in
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    "") ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
esac

: "${GITLAB_TOKEN:?set GITLAB_TOKEN (token with api scope)}"
GITLAB_PROJECT="${GITLAB_PROJECT:-metropolis-perf/sdg/cosmos-anomalygen}"
MAX_JOBS="${MAX_JOBS:-4}"
# GitLab's API accepts either a numeric id or a URL-encoded path.
PROJECT_REF="${GITLAB_PROJECT//\//%2F}"
REPO_URL="https://gitlab-master.nvidia.com/api/v4/projects/${PROJECT_REF}/packages/pypi"

BUILD_IMAGE="nvidia/cuda:12.8.2-devel-ubuntu24.04"
WHEELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/wheels"
mkdir -p "$WHEELS_DIR"

# Pins (keep in sync with docker/Dockerfile)
export TORCH_VERSION="2.10.0"
export TORCHVISION_VERSION="0.25.0"
export EINOPS_VERSION="0.8.2"
export FLASH_ATTN_VERSION="2.8.3"
export TE_VERSION="2.13.0"
export OPENCV_TAG="92"   # = opencv-python-headless 4.13.0.92
export APEX_SHA="466e164bfd9548b3026c1f3a1c296bed8ef55c43"

# ---- build ------------------------------------------------------------------
echo "==> Building wheels in $BUILD_IMAGE → $WHEELS_DIR"
export MAX_JOBS
docker run --rm \
    -v "$WHEELS_DIR:/wheels" \
    -e TORCH_VERSION -e TORCHVISION_VERSION -e EINOPS_VERSION \
    -e FLASH_ATTN_VERSION -e TE_VERSION -e OPENCV_TAG -e APEX_SHA \
    -e MAX_JOBS \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    "$BUILD_IMAGE" bash -euxc '
        apt-get update
        apt-get install -y --no-install-recommends \
            python3.12 python3.12-dev python3-pip git build-essential \
            cmake ninja-build libssl-dev libffi-dev
        ln -sf /usr/bin/python3.12 /usr/local/bin/python3
        ln -sf /usr/bin/python3.12 /usr/local/bin/python

        # Bypass PEP 668 + strip Debian wheel/setuptools/packaging
        # (no RECORD files so pip refuses to uninstall them).
        rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED
        rm -rf /usr/lib/python3/dist-packages/wheel \
               /usr/lib/python3/dist-packages/wheel-*.dist-info \
               /usr/lib/python3/dist-packages/wheel-*.egg-info \
               /usr/lib/python3/dist-packages/setuptools \
               /usr/lib/python3/dist-packages/setuptools-*.dist-info \
               /usr/lib/python3/dist-packages/setuptools-*.egg-info \
               /usr/lib/python3/dist-packages/packaging \
               /usr/lib/python3/dist-packages/packaging-*.dist-info \
               /usr/lib/python3/dist-packages/packaging-*.egg-info
        # scikit-build is required by the opencv-python PEP 517 build.
        # --no-build-isolation means pip will not auto-install it.
        pip install --no-cache-dir setuptools wheel packaging scikit-build

        pip install --no-cache-dir \
            --extra-index-url https://download.pytorch.org/whl/cu128 \
            torch=="$TORCH_VERSION" torchvision=="$TORCHVISION_VERSION" einops=="$EINOPS_VERSION"
        ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/*/include/* /usr/include/ || true
        ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/*/include/* /usr/include/python3.12/ || true

        # flash-attn
        MAX_JOBS="$MAX_JOBS" NVCC_THREADS=2 pip wheel --no-build-isolation --no-deps \
            -w /wheels flash-attn=="$FLASH_ATTN_VERSION"

        # transformer-engine wrapper
        MAX_JOBS="$MAX_JOBS" NVCC_THREADS=2 pip wheel --no-build-isolation --no-deps \
            -w /wheels transformer-engine=="$TE_VERSION"

        # transformer-engine-cu12 (prebuilt binary on PyPI, just mirror)
        pip download --no-deps --only-binary=:all: \
            --platform manylinux_2_28_x86_64 --python-version 312 \
            -d /wheels transformer-engine-cu12=="$TE_VERSION"

        # transformer-engine-torch (source-only on PyPI — must compile;
        # needs wrapper + cu12 backend installed for its build)
        pip install --no-cache-dir --no-deps \
            transformer-engine=="$TE_VERSION" transformer-engine-cu12=="$TE_VERSION"
        MAX_JOBS="$MAX_JOBS" NVCC_THREADS=2 pip wheel --no-build-isolation --no-deps \
            -w /wheels transformer-engine-torch=="$TE_VERSION"

        # apex (pinned SHA, cpp_ext + cuda_ext)
        MAX_JOBS="$MAX_JOBS" NVCC_THREADS=2 pip wheel -v --no-build-isolation --no-deps \
            --disable-pip-version-check --no-cache-dir \
            --config-settings "--build-option=--cpp_ext --cuda_ext" \
            -w /wheels "apex @ git+https://github.com/NVIDIA/apex.git@${APEX_SHA}"

        # opencv-python-headless (GitHub tag, WITH_FFMPEG=OFF)
        git clone --recursive --shallow-submodules --depth 1 --branch "$OPENCV_TAG" \
            https://github.com/opencv/opencv-python.git /tmp/opencv-python
        cd /tmp/opencv-python
        ENABLE_HEADLESS=1 CMAKE_ARGS="-DWITH_FFMPEG=OFF" MAKEFLAGS="-j$MAX_JOBS" \
            pip wheel --no-build-isolation --no-deps -w /wheels .

        chown -R "$HOST_UID":"$HOST_GID" /wheels
    '

echo "==> Built wheels:"
ls -lh "$WHEELS_DIR"/*.whl

# ---- upload -----------------------------------------------------------------
echo "==> Uploading to ${REPO_URL}"
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e USER="$(id -un)" -e HOME=/tmp \
    -v "$WHEELS_DIR:/wheels:ro" \
    -e REPO_URL="$REPO_URL" -e GITLAB_TOKEN="$GITLAB_TOKEN" \
    python:3.12-slim bash -euxc '
        pip install --no-cache-dir twine
        python -m twine upload \
            --repository-url "$REPO_URL" \
            --username gitlab-ci-token \
            --password "$GITLAB_TOKEN" \
            /wheels/*.whl
    '
echo "==> Uploaded. Consume via:"
echo "    pip install --extra-index-url '${REPO_URL}/simple' \\"
echo "      flash-attn==${FLASH_ATTN_VERSION} \\"
echo "      transformer-engine==${TE_VERSION} \\"
echo "      apex \\"
echo "      opencv-python-headless==4.13.0.92"
