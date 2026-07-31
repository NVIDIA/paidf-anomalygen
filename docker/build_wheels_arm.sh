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
# arm64 / CUDA-13 counterpart of build_wheels.sh. Builds the heavy native wheels
# and uploads them to the project's GitLab PyPI registry so subsequent Docker /
# CI builds (docker/Dockerfile.arm.cuda130 with GITLAB_PYPI_INDEX_URL set) can
# `pip install` instead of recompiling. The wheels are aarch64-tagged, so they
# coexist with the x86_64 wheels from build_wheels.sh in the same registry.
#
# MUST run on an aarch64 host (the wheels are arch-specific; the CUDA-13 base
# image is pulled for the host arch).
#
# Wheels produced (from docker/Dockerfile.arm.cuda130):
#   flash-attn                2.8.3   PATCHED multi-arch sm_90/100/103/120
#   transformer-engine        2.14.0  (wrapper, py3-none-any)
#   transformer-engine-torch  2.14.0  (PyTorch backend, source-compiled)
#   transformer-engine-cu13   2.14.0  (CUDA-13 backend, mirrored from PyPI)
#   apex                      git 466e164b (--cpp_ext --cuda_ext, multi-arch)
#   opencv-python-headless    4.13.0.92  (GitHub tag `92`, WITH_FFMPEG=OFF)
#
# Usage:
#   GITLAB_TOKEN=<token-with-api-scope> ./docker/build_wheels_arm.sh
#
# Required env:
#   GITLAB_TOKEN     project or personal access token with `api` scope
# Optional env:
#   GITLAB_PROJECT   default: metropolis-perf/sdg/cosmos-anomalygen
#   MAX_JOBS         parallel native-compile jobs (default 12; ~6 GB RAM each)

set -euo pipefail

case "${1:-}" in
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    "") ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
esac

[ "$(uname -m)" = "aarch64" ] || { echo "error: must run on an aarch64 host (got $(uname -m))" >&2; exit 1; }
: "${GITLAB_TOKEN:?set GITLAB_TOKEN (token with api scope)}"
GITLAB_PROJECT="${GITLAB_PROJECT:-metropolis-perf/sdg/cosmos-anomalygen}"
MAX_JOBS="${MAX_JOBS:-12}"
PROJECT_REF="${GITLAB_PROJECT//\//%2F}"
REPO_URL="https://gitlab-master.nvidia.com/api/v4/projects/${PROJECT_REF}/packages/pypi"

BUILD_IMAGE="nvidia/cuda:13.0.3-devel-ubuntu24.04"
WHEELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/wheels"
mkdir -p "$WHEELS_DIR"

# Pins (keep in sync with docker/Dockerfile.arm.cuda130)
export TORCH_VERSION="2.10.0"
export TORCHVISION_VERSION="0.25.0"
export EINOPS_VERSION="0.8.2"
export FLASH_ATTN_VERSION="2.8.3"
export TE_VERSION="2.14.0"
export OPENCV_TAG="92"   # = opencv-python-headless 4.13.0.92
export APEX_SHA="466e164bfd9548b3026c1f3a1c296bed8ef55c43"
export PLATFORM="manylinux_2_28_aarch64"
export TORCH_CUDA_ARCH_LIST="9.0 10.0 10.3 12.0"
export FLASH_ATTN_CUDA_ARCHS="90;100;103;120"

# ---- build ------------------------------------------------------------------
echo "==> Building aarch64/cu13 wheels in $BUILD_IMAGE → $WHEELS_DIR"
export MAX_JOBS
docker run --rm \
    -v "$WHEELS_DIR:/wheels" \
    -e TORCH_VERSION -e TORCHVISION_VERSION -e EINOPS_VERSION \
    -e FLASH_ATTN_VERSION -e TE_VERSION -e OPENCV_TAG -e APEX_SHA \
    -e PLATFORM -e TORCH_CUDA_ARCH_LIST -e FLASH_ATTN_CUDA_ARCHS \
    -e MAX_JOBS -e NVCC_THREADS=1 \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    "$BUILD_IMAGE" bash -euxc '
        apt-get update
        apt-get install -y --no-install-recommends \
            python3.12 python3.12-dev python3-pip git build-essential \
            cmake ninja-build libssl-dev libffi-dev
        ln -sf /usr/bin/python3.12 /usr/local/bin/python3
        ln -sf /usr/bin/python3.12 /usr/local/bin/python

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
        pip install --no-cache-dir setuptools wheel packaging scikit-build

        pip install --no-cache-dir \
            --extra-index-url https://download.pytorch.org/whl/cu130 \
            torch=="$TORCH_VERSION" torchvision=="$TORCHVISION_VERSION" einops=="$EINOPS_VERSION"

        # cp -rsf, not ln -sf: the CUDA-13 base ships /usr/include/nccl_device as
        # a real dir and ln -sf cannot overwrite a directory.
        for inc in /usr/local/lib/python3.12/dist-packages/nvidia/*/include; do
            cp -rsf "$inc"/* /usr/include/ 2>/dev/null || true
            cp -rsf "$inc"/* /usr/include/python3.12/ 2>/dev/null || true
        done

        # flash-attn — PATCHED multi-arch (2.8.3 has no sm_103 branch). Download
        # source (--no-build-isolation so the metadata hook sees torch), inject
        # the sm_103 gencode, then build the wheel for FLASH_ATTN_CUDA_ARCHS.
        FA_SRC=/tmp/fa
        pip download --no-deps --no-binary :all: --no-build-isolation \
            flash-attn=="$FLASH_ATTN_VERSION" -d "$FA_SRC"
        tar -xf "$FA_SRC"/flash_attn-"$FLASH_ATTN_VERSION".tar.gz -C "$FA_SRC"
        python3 - "$FA_SRC/flash_attn-$FLASH_ATTN_VERSION/setup.py" <<PYEOF
import sys
p = sys.argv[1]; s = open(p).read()
anchor = "cc_flag.append(\"arch=compute_120,code=sm_120\")"
add = ("\n        if bare_metal_version >= Version(\"12.8\") and \"103\" in cuda_archs():"
       "\n            cc_flag.append(\"-gencode\")"
       "\n            cc_flag.append(\"arch=compute_103,code=sm_103\")")
assert anchor in s, "flash-attn setup.py layout changed"
open(p, "w").write(s.replace(anchor, anchor + add, 1))
PYEOF
        MAX_JOBS="$MAX_JOBS" pip wheel --no-build-isolation --no-deps \
            -w /wheels "$FA_SRC/flash_attn-$FLASH_ATTN_VERSION"

        # transformer-engine wrapper (arch-agnostic)
        MAX_JOBS="$MAX_JOBS" pip wheel --no-build-isolation --no-deps \
            -w /wheels transformer-engine=="$TE_VERSION"

        # transformer-engine-cu13 (prebuilt aarch64 binary on PyPI — mirror)
        pip download --no-deps --only-binary=:all: \
            --platform "$PLATFORM" --python-version 312 \
            -d /wheels transformer-engine-cu13=="$TE_VERSION"

        # transformer-engine-torch (source-only — compile; needs wrapper + cu13 backend)
        pip install --no-cache-dir --no-deps \
            transformer-engine=="$TE_VERSION" transformer-engine-cu13=="$TE_VERSION"
        MAX_JOBS="$MAX_JOBS" pip wheel --no-build-isolation --no-deps \
            -w /wheels transformer-engine-torch=="$TE_VERSION"

        # apex (pinned SHA, cpp_ext + cuda_ext, multi-arch)
        TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
        MAX_JOBS="$MAX_JOBS" pip wheel -v --no-build-isolation --no-deps \
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
