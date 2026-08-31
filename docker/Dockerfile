# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# =============================================================================
# PAIDF AnomalyGen (paidf-anomalygen) Docker image — multi-stage build.
#
# STAGE GRAPH — three build stages, then four selectable targets:
#
#   wheelbuilder ─┐
#    (throwaway)  ├─> runtime ──┬─> product
#   wandbcore ────┘   (the image)
#    (throwaway)                 ├─> airgapped-base ─┬─> airgapped-product
#                                 │                   └─> airgapped-develop
#                                 └─> develop  (DEFAULT)
#
#   The air-gapped payload is their shared ANCESTOR, not a copy in each: a stage gets one parent, so
#   sharing means putting it above the split. The multi-GB checkpoint layers are built once.
#
#   Stage 1  wheelbuilder  (nvidia/cuda:...-cudnn-devel)
#       WHERE THE COMPILING HAPPENS — and the only stage that needs nvcc.
#       Delegates entirely to `docker/build_wheels.sh --in-container`, the same
#       file (and the same mode) used to populate the GitLab registry from a
#       developer host, so both routes build byte-comparable wheels. Produces
#       /wheels; nothing else from this stage ends up in the final image.
#       The COMPILE RECIPE lives in one place, but the version pins do NOT: the
#       PINS block below and build_wheels.sh's config block both carry them, and
#       a bump has to touch both or stage 1 builds a version stage 2 never
#       downloads.
#       SKIPPED when the `pypi_index` secret is given: stage 2 then downloads the
#       prebuilt wheels instead, turning a >1h compile into a short pull. With
#       it empty the image still builds standalone, just slowly.
#
#   Stage 2  runtime  (nvidia/cuda:...-base)  — this is the shipped image
#       Slim: no CUDA toolkit, no compiler. Creates the py3.13 venv and only
#       *installs* things, in dependency order so a code edit rebuilds one layer:
#         torch -> heavy wheels (registry, else stage 1's /wheels)
#           -> requirements.txt -> cosmos-framework (pinned in
#           requirements-nodeps.txt) -> app source + the two in-repo packages.
#       Also bakes the OSS source obligation into /usr/share/oss-sources. That
#       collector lives in `docker/oss_bake.sh`, for the same reason the compile
#       recipe lives in build_wheels.sh: it is a program, not image assembly.
#       Works without nvcc because torch's +cu132 wheel bundles the CUDA
#       userspace libs into the venv and every heavy dep arrives prebuilt, so
#       the base needs only the CUDA *runtime* libraries.
#
#   Targets (--target, default `develop`) — see the stage definitions at the
#   bottom of this file:
#       product            runtime + app code locked read-only
#       develop            runtime + pytest/ruff/pre-commit, interactive bash
#       airgapped-product  product hardening + model checkpoints baked in
#       airgapped-develop  dev tooling + the same baked checkpoints
#       The airgapped-* targets need `--build-context ckpts=./checkpoints`. Both build on
#       airgapped-base, which warms a uv cache and flips UV_NO_CACHE back off because the framework
#       resolves the Edge caption processor through a `uv run`-ed HF CLI.
#
# Build and run commands — including the prebuilt-wheel index, the air-gapped
# `ckpts` context, and the `docker run` invocation — live in docker/README.md.
# This file documents only how the image is assembled.
# =============================================================================

# =============================================================================
# PINS — versions stage 2 downloads from the registry
#
# These are the versions stage 2 DOWNLOADS from the registry; the versions
# stage 1 BUILDS are pinned in docker/build_wheels.sh, which owns the compile
# recipe. Bump both together when a wheel is rebuilt.
# =============================================================================
ARG CUDA_DEVEL_BASE=nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04
ARG CUDA_RUNTIME_BASE=nvidia/cuda:13.2.1-base-ubuntu24.04
ARG PYTHON_VERSION=3.13.15
ARG PY_VER=${PYTHON_VERSION%.*}
ARG UV_VERSION=0.12.5
ARG TORCH_VER=2.13.0
ARG TORCHVISION_VER=0.28.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132
# Exact versions of the prebuilt wheels in the registry (from build_wheels.sh
# output). The registry path downloads these exactly, so they must be pinned.
ARG FA_VER=2.8.3
ARG TE_VER=2.15.0
ARG APEX_VER=0.1
ARG APEX_SHA=becbb77cea4cb54f2929f7c938a0a6f7dd1fdc39
ARG FA3_PKG=flash-attn-3-nv
ARG FA3_VER=1.0.3
ARG FA3_SHA=c603389369b8662a5cf94e5f0096ba739b99ffdd
# sam-2 installs from a pinned git commit, so its source is the archive at that commit.
ARG SAM2_SHA=2b90b9f5ceec907a1c18123530e92e794ad901a4
ARG NATTEN_VER=0.21.7
ARG OPENCV_VER=4.14.0.94
# libquadmath is LGPL-2.1-or-later and outside the Runtime Library Exception that
# covers libgfortran, so the copy bundled into the numpy and scipy wheels ships
# with source. 8.5.0 is the manylinux_2_28 (AlmaLinux 8) GCC that auditwheel
# recorded in them; bump it if they move to a different manylinux image.
ARG LIBGFORTRAN_LIBQUADMATH_GCC_VER=8.5.0
# wandb-core is a Go binary the wheel ships prebuilt, so its toolchain and module versions are frozen
# at whatever upstream built that release with. Container scans read that build info and attribute
# the Go advisories to this one file, and no wheel we can pin is guaranteed to carry a fixed
# toolchain — upstream rebuilds on its own schedule. Rebuilding from source is the only way to
# choose. The toolchain covers the standard library; a vulnerable module dependency needs its own
# pin, which is what WANDB_GO_MODULES is. Keep WANDB_SRC_SHA in step with the wandb pin in
# requirements.txt.
ARG GO_VERSION=1.26.6
# Pinned by tag AND digest, like uv and the CUDA bases. The tag is derived from GO_VERSION so there
# is one place to bump; if the digest is not bumped with it the pull resolves to the old image, and
# the toolchain assertion in the wandbcore stage fails rather than shipping it.
ARG GO_BASE=golang:${GO_VERSION}@sha256:0d1d3a794be25f809dd2cb3160d8c73276c4056a9f8242a138e908ddeee7b6b6
ARG WANDB_SRC_SHA=bc2cf24b1bd9fa3a1b6cbeb920087b832afc526f
ARG WANDB_GO_MODULES="golang.org/x/net@v0.58.0 github.com/go-git/go-git/v5@v5.19.2 google.golang.org/grpc@v1.83.0 golang.org/x/image@v0.45.0"
ARG MAX_JOBS=16
ARG NVCC_THREADS=1
# Bump for an Ubuntu security refresh, or to rebuild the OSS bake against the current archive.
ARG APT_CACHE_BUST=2026-08-20
# SHA-256 of the four OSS-bake artefacts that are NOT PyPI sdists (those carry a
# digest from the PyPI JSON API and are checked inline). Recompute after bumping a
# pin:  curl -sfL <url> | sha256sum
# Caveat: GitHub does not guarantee byte-stable archive tarballs. If one of the
# three GitHub ones starts failing with no pin change, that is why — re-verify the
# content before updating the digest rather than just refreshing it.
ARG APEX_TARBALL_SHA256=51a3db0ae700dc9f843e5418aec3379c6cc227f4d7e02c59852038b8f02cb1bc
ARG FA3_TARBALL_SHA256=ee3693af43ed54a614ecc5c3dd0f78e17d0637a0d44ad16aaabd3d493f0ce127
ARG UV_TARBALL_SHA256=e349c9eb85876921895330f6fee5f01f109d5ec06dcd3b475fb7b4b8de75eac6
ARG PYTHON_TARBALL_SHA256=1e66a7945a48390ee4c2a4268a0e4185884059a13c4aab6d148aa208deea4a76
ARG SAM2_TARBALL_SHA256=1f2fbfad3ffa38110368abac76c6ef9df9c282a66d5c2807bc94abf4d2fb30f8
ARG LIBGFORTRAN_LIBQUADMATH_GCC_TARBALL_SHA256=d308841a511bb830a6100397b0042db24ce11f642dab6ea6ee44842e5325ed50

# GPU compute-capability list for the source-compile path (stage 1). Ignored on
# the registry path, where stage 1 is skipped entirely.
#
# Defaulted, NOT empty, so `docker build -f docker/Dockerfile .` works with no
# arguments and no registry access: a docker build gets no GPU, so nothing can be
# autodetected and an empty value aborts stage 1. MUST stay quoted — an unquoted
# ARG default stops at the first space, silently yielding "8.0" alone.
#
# The default is the amd64 set. These two lists are what we publish per CPU arch —
# note the GPU arch list is independent of the CPU arch; they differ only because
# of which machines exist:
#   amd64  "8.0 8.6 9.0 10.0 12.0"  A100 / A10,A40,RTX30 / H100 / B100,B200 /
#                                   RTX PRO,RTX50. 8.6 also covers Ada (8.9) by
#                                   minor-version cubin compatibility, and 10.0
#                                   likewise covers 10.3.
#   arm64  "9.0 10.0 10.3 12.0"     GH200 / GB200 / GB300 / workstation Blackwell.
#                                   No Ampere entries: NVIDIA's Grace-based systems
#                                   pair with Hopper and Blackwell only. 10.3 is
#                                   listed explicitly because GB300 is a Grace part.
# Override with --build-arg for a narrower set or a different machine.
ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 10.0 12.0"
ARG APP_UID=10000
ARG APP_GID=10000

# =============================================================================
# STAGE 1 — wheelbuilder: compile the CUDA-extension wheels into /wheels
#
# Throwaway stage. Only /wheels is carried forward (stage 2 COPY --from). The
# recipe and its pins are NOT here: they live in docker/build_wheels.sh, which
# this stage runs with --in-container. That is the same file, in the same mode,
# that `./docker/build_wheels.sh` runs on a developer host to populate the
# registry — so the compile recipe is defined once. The version pins are still
# duplicated in the PINS block above; keep them in step.
#
# Needs only the four ARGs below; the version pins come from the script.
# =============================================================================
FROM ${CUDA_DEVEL_BASE} AS wheelbuilder

ARG MAX_JOBS
ARG NVCC_THREADS
ARG TORCH_CUDA_ARCH_LIST

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=${MAX_JOBS}
ENV NVCC_THREADS=${NVCC_THREADS}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV UV_LINK_MODE=copy

# uv pinned by tag AND digest. The digest makes the binary reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 /uv /uvx /usr/local/bin/
ENV PATH="/usr/local/bin:$PATH"

COPY docker/build_wheels.sh /tmp/build_wheels.sh

# Compile only when no registry URL was given. With a token-bearing index this
# short-circuits to an empty /wheels in ~0.2s and stage 2 downloads instead.
RUN --mount=type=secret,id=pypi_index <<'EOF'
set -eu
mkdir -p /wheels
if [ -s /run/secrets/pypi_index ]; then
  echo "pypi_index secret provided -> heavy wheels come from the registry; skipping local build."
  exit 0
fi
bash /tmp/build_wheels.sh --in-container
EOF

# =============================================================================
# wandbcore — rebuild wandb-core on the pinned Go toolchain
#
# Its own stage so the toolchain, the clone and the module cache never touch the shipped image, and
# so this only re-runs when its own pins change. Building it inside `runtime` meant any edit to
# requirements.txt re-ran the whole compile.
# =============================================================================
FROM ${GO_BASE} AS wandbcore
ARG GO_VERSION
ARG WANDB_SRC_SHA
ARG WANDB_GO_MODULES
ARG TARGETARCH
RUN <<'EOF'
set -eu
# By commit, not the tag the wheel was built from: a tag can move.
git init -q /src
git -C /src remote add origin https://github.com/wandb/wandb.git
git -C /src fetch -q --depth 1 origin "${WANDB_SRC_SHA}"
git -C /src checkout -q FETCH_HEAD
cd /src/core
# wandb vendors its Go dependencies, and Go picks vendor mode automatically when vendor/modules.txt
# is present — which makes the bump below conflict with the vendored tree. Resolve from go.mod
# instead; go.sum still pins every module, and nothing here consumes vendor/.
export GOFLAGS=-mod=mod
# Unquoted on purpose: the list is several module@version pairs.
go get ${WANDB_GO_MODULES}
go mod tidy
CGO_ENABLED=0 GOOS=linux GOARCH="${TARGETARCH}" go build -trimpath -ldflags="-s -w" -o /wandb-core ./cmd/wandb-core/
# Assert what will ship, so a toolchain or module that did not move fails here rather than shipping.
go version -m /wandb-core | grep -q "go${GO_VERSION}\b"
for m in ${WANDB_GO_MODULES}; do
  go version -m /wandb-core \
    | awk -v p="${m%@*}" -v v="${m#*@}" '$2==p && $3==v {ok=1} END{exit !ok}' \
    || { echo "wandb-core: ${m} is not in the built binary" >&2; exit 1; }
done
echo "wandb-core built on go${GO_VERSION} with ${WANDB_GO_MODULES}"
EOF

# =============================================================================
# STAGE 2 — runtime: the image that actually ships
#
# No CUDA toolkit, no compiler: everything heavy arrives prebuilt, either from
# the registry or from stage 1's /wheels. Ordered cheapest-to-invalidate last,
# so editing app code rebuilds only the final layers:
#   apt -> uv/venv -> torch -> heavy wheels -> requirements.txt
#     -> cosmos-framework -> app source
# =============================================================================
FROM ${CUDA_RUNTIME_BASE} AS runtime

ARG PYTHON_VERSION
ARG PY_VER
ARG UV_VERSION
ARG APT_CACHE_BUST
ARG TORCH_VER
ARG TORCHVISION_VER
ARG TORCH_INDEX_URL
ARG FA_VER
ARG TE_VER
ARG APEX_VER
ARG APEX_SHA
ARG FA3_PKG
ARG FA3_VER
ARG FA3_SHA
ARG SAM2_SHA
ARG NATTEN_VER
ARG OPENCV_VER
ARG LIBGFORTRAN_LIBQUADMATH_GCC_VER
ARG APEX_TARBALL_SHA256
ARG FA3_TARBALL_SHA256
ARG UV_TARBALL_SHA256
ARG PYTHON_TARBALL_SHA256
ARG SAM2_TARBALL_SHA256
ARG LIBGFORTRAN_LIBQUADMATH_GCC_TARBALL_SHA256
ARG APP_UID
ARG APP_GID

# /usr/local/bin holds the pinned uv (COPYed below); /opt/venv/bin holds python
# and pip. Ordered explicitly in one place — uv first, then the venv — so
# nothing depends on the base image's default PATH.
ENV PATH="/usr/local/bin:/opt/venv/bin:$PATH"
ENV UV_PYTHON=/opt/venv/bin/python
# Install uv's managed Python under /opt (world-readable) instead of the default
# ~/.local/share/uv (~ = /root, mode 700) so the non-root runtime user can reach
# the interpreter the venv symlinks to.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_LINK_MODE=copy
ENV UV_NO_CACHE=1
ENV PIP_NO_CACHE_DIR=1
# SAM2's optional `sam2._C` connected-components ext never builds here.
# anomalygen doesn't use the ext.
ENV SAM2_BUILD_ALLOW_ERRORS=1

# The -base image ships almost no CUDA libs — torch bundles its own into the
# venv. libgomp1 is torch's OpenMP runtime; libgl1/libglib2.0-0 are for opencv.
# gcc/g++ are needed at runtime by torch.compile / Triton inductor to JIT the
# kernel + C++ wrapper (ptxas is already bundled inside the triton wheel).
# The dpkg snapshot taken first scopes the OSS source bake further down: it records
# what the base image already carries, so the bake fetches source for what WE add
# on top. It has to happen here — afterwards the two sets are indistinguishable.
RUN echo "apt-cache-bust: ${APT_CACHE_BUST}" \
     && export DEBIAN_FRONTEND=noninteractive \
     && dpkg-query -W -f='${source:Package}\n' | sort -u > /var/lib/oss-pkgs-before.txt \
     && apt-get update && apt-get install -y --no-install-recommends \
     ca-certificates curl git gcc g++ jq libgomp1 libgl1 libglib2.0-0 \
     # Hold the CUDA/NVIDIA stack before upgrading. The `wc -l` check is
     # load-bearing: a pattern that matches nothing leaves grep's non-zero status
     # invisible (RUN uses /bin/sh, which has no `pipefail`), so without it the
     # hold is a no-op and apt-get upgrade silently drags the CUDA stack past what
     # the prebuilt wheels were built against.
     && dpkg-query -W -f='${binary:Package}\n' \
        | grep -E '^(cuda|libcublas|libcudnn|libcufft|libcufile|libcurand|libcusolver|libcusparse|libcusparselt|libnccl|libnpp|libnv|nvidia)' \
        > /tmp/_hold.txt \
     && test "$(wc -l < /tmp/_hold.txt)" -gt 0 \
     && xargs -r apt-mark hold < /tmp/_hold.txt \
     && rm -f /tmp/_hold.txt \
     && apt-get upgrade -y --no-install-recommends \
     && rm -rf /var/lib/apt/lists/*

# uv pinned by tag AND digest. The digest makes the binary reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 /uv /uvx /usr/local/bin/
RUN uv venv --python ${PYTHON_VERSION} /opt/venv && uv pip install pip

# torch runtime (bundles the CUDA userspace libs into the venv)
RUN uv pip install torch==${TORCH_VER} torchvision==${TORCHVISION_VER} \
     --index-url ${TORCH_INDEX_URL}

# Heavy deps: from the GitLab registry if a URL was given, else from the wheels
# compiled in stage 1.
#
# Stage 1's /wheels is BIND-MOUNTED, not COPYed. A COPY is its own layer, which no
# later `rm -rf` can reclaim — on the source-compile path (no `pypi_index` secret)
# that layer would carry ~2 GB of freshly compiled wheels in the shipped image
# forever. A bind mount exists only for this RUN. The single
# FFmpeg-free opencv wheel is copied out (~33 MB, original filename kept — uv rejects
# a wheel whose name has no Python tag) because a later layer reinstalls
# it over the FFmpeg-bundling one that retinaface-py drags in.
RUN --mount=type=bind,from=wheelbuilder,source=/wheels,target=/wheels \
    --mount=type=secret,id=pypi_index <<'EOF'
set -eu
# Read the index URL from the secret mount (tmpfs, gone when this RUN ends).
# Never `set -x` in this block: that would echo the token into the build log.
GITLAB_PYPI_INDEX_URL="$(cat /run/secrets/pypi_index 2>/dev/null || true)"
if [ -n "$GITLAB_PYPI_INDEX_URL" ]; then
  # Download the exact prebuilt wheels from the registry ONLY (--index-url, not
  # --extra-index-url) so PyPI is never consulted — otherwise uv could pick a
  # PyPI sdist for transformer-engine / natten (which would need nvcc, absent
  # here) or the unrelated PyPI "apex". Then install the files (deps resolve
  # from PyPI normally).
  mkdir -p /tmp/regwheels
  python -m pip download --no-deps --index-url "$GITLAB_PYPI_INDEX_URL" -d /tmp/regwheels \
    flash-attn=="$FA_VER" \
    transformer-engine=="$TE_VER" \
    transformer-engine-cu13=="$TE_VER" \
    transformer-engine-torch=="$TE_VER" \
    apex=="$APEX_VER" \
    "$FA3_PKG"=="$FA3_VER" \
    natten=="$NATTEN_VER" \
    opencv-python-headless=="$OPENCV_VER"
  uv pip install /tmp/regwheels/*.whl
  # Keep the FFmpeg-free headless wheel for the reinstall in the next layer.
  mkdir -p /tmp/cvwheel && cp /tmp/regwheels/opencv_python_headless-*.whl /tmp/cvwheel/
  rm -rf /tmp/regwheels
else
  uv pip install /wheels/*.whl
  mkdir -p /tmp/cvwheel && cp /wheels/opencv_python_headless-*.whl /tmp/cvwheel/
fi
EOF

# pip requirements. retinaface-py hard-requires the FFmpeg-bundling
# `opencv-python`, which installs into the same cv2/ directory as the
# FFmpeg-free opencv-python-headless above — so uninstalling it leaving an empty
# cv2 namespace that still imports but fails at the first real call. Reinstall
# the headless wheel last so cv2 is whole.
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt \
     && (uv pip uninstall av opencv-python 2>/dev/null || true) \
     && uv pip install --no-deps --force-reinstall /tmp/cvwheel/*.whl \
     && rm -rf /tmp/cvwheel \
     && python -c "import cv2; cv2.connectedComponentsWithStats; print('cv2 OK', cv2.__version__)" \
     # Drop setuptools' vendored autocommand (LGPL-3.0-only).
     # Verified: setuptools, build_meta and jaraco.text all still import.
     && rm -rf /opt/venv/lib/python${PY_VER}/site-packages/setuptools/_vendor/autocommand \
               /opt/venv/lib/python${PY_VER}/site-packages/setuptools/_vendor/autocommand-*.dist-info \
     && test ! -e /opt/venv/lib/python${PY_VER}/site-packages/setuptools/_vendor/autocommand \
     && python -c "import setuptools, setuptools.build_meta; import setuptools._vendor.jaraco.text; print('setuptools OK', setuptools.__version__)"

# The rebuilt wandb-core, from the stage above. Copied rather than built here so the Go toolchain
# never enters this image and an unrelated requirements.txt edit does not re-run the compile.
COPY --from=wandbcore /wandb-core /opt/venv/lib/python${PY_VER}/site-packages/wandb/bin/wandb-core
# The COPY destination is a literal path; this fails the build if wandb ever moves the binary, rather
# than leaving the wheel's own copy in place beside a stray file.
RUN test "$(python -c 'import pathlib, wandb; print(pathlib.Path(wandb.__file__).parent / "bin" / "wandb-core")')" \
     = "/opt/venv/lib/python${PY_VER}/site-packages/wandb/bin/wandb-core"

# cosmos-framework, from the git commit pinned in requirements-nodeps.txt.
COPY requirements-nodeps.txt /tmp/requirements-nodeps.txt
RUN CF_SITE=/opt/venv/lib/python${PY_VER}/site-packages/cosmos_framework \
     && uv pip install -r /tmp/requirements-nodeps.txt --no-deps \
     && rm -f /tmp/requirements-nodeps.txt \
     # Drop cosmos-framework's vendored TaylorSeer helper.
     && test -e "${CF_SITE}/model/generator/utils/taylorseer.py" \
     && rm -f "${CF_SITE}/model/generator/utils/taylorseer.py" \
               "${CF_SITE}/model/generator/utils/__pycache__/taylorseer."*.pyc \
     && test ! -e "${CF_SITE}/model/generator/utils/taylorseer.py" \
     # unified_mot is the module carrying the init_taylorseer flag-setting code.
     && python -c "import cosmos_framework.model.generator.mot.unified_mot; print('cosmos_framework OK')"

# Register the venv's bundled nvidia lib dirs with the dynamic linker.
ENV NV_SITE=/opt/venv/lib/python${PY_VER}/site-packages/nvidia
ENV LD_LIBRARY_PATH=${NV_SITE}/cudnn/lib:${NV_SITE}/nccl/lib:${NV_SITE}/cusparselt/lib
RUN { for d in ${NV_SITE}/*/lib; do echo "$d"; done; } > /etc/ld.so.conf.d/cosmos.conf \
     && ldconfig

# =============================================================================
# OSS SOURCE-CODE BAKE — /usr/share/oss-sources
#
# Source for the third-party OSS in the image, travelling with it rather than refetchable later.
# The collector is docker/oss_bake.sh; it documents the layout. Only what we ADD: the base image is a
# published NVIDIA artifact with its own source obligation, NVIDIA's wheels are not third-party, and
# paidf-anomalygen is ours. Known gap — `apt-get upgrade` moves base packages without renaming them,
# so those stay on the base side of the diff.
#
# PLACEMENT: after the installs, so editing the bake does not re-run the >1 h wheel compile; before
# `COPY . /workspace`, so a code change does not re-run the bake. The develop family runs the pip
# half again once requirements-dev.txt is in, since develop is the target CI publishes.
# =============================================================================
COPY --chmod=0755 docker/oss_bake.sh /usr/local/bin/oss-bake
RUN oss-bake archives && oss-bake apt && oss-bake pip && oss-bake manifest

# ---- Non-root user + app source (last, so code changes don't bust deps) -----
RUN groupadd --gid ${APP_GID} nvidia \
     && useradd --create-home --shell /bin/bash --uid ${APP_UID} --gid ${APP_GID} nvidia

COPY --chown=nvidia:nvidia . /workspace/paidf-anomalygen
WORKDIR /workspace/paidf-anomalygen

# Must stay BELOW the COPY above — this is a source install, not a dependency
# install, so it cannot be hoisted into the install section.
RUN uv pip install -e assets/lerobot_stub --no-deps \
     && uv pip install -e . --no-deps

# The nvidia/cuda base ships the NGC Deep Learning Container licence at /, which
# governs this image's base layers. Surface it in the workdir beside LICENSE,
# NOTICE and LICENSE-3rd-party.txt so it is discoverable where a user actually
# looks — at / it is easy to miss.
RUN cp /NGC-DL-CONTAINER-LICENSE /workspace/paidf-anomalygen/NGC-DL-CONTAINER-LICENSE \
     && chown nvidia:nvidia /workspace/paidf-anomalygen/NGC-DL-CONTAINER-LICENSE

ENV PYTHONPATH=/workspace/paidf-anomalygen

USER nvidia

# =============================================================================
# SELECTABLE TARGETS — thin layers on top of stage 2; none of them install deps
#
#   runtime ──┬─> product
#             ├─> airgapped-base ─┬─> airgapped-product
#             │                   └─> airgapped-develop
#             └─> develop  (DEFAULT: last stage in the file)
#
# Air-gapped targets bake checkpoints/ in via a named `ckpts` build context, which
# bypasses the repo .dockerignore that excludes checkpoints/. That context is
# required for those targets — see docker/README.md for the invocation.
# =============================================================================

# ---- product: app code locked read-only + product-mode flag -----------------
FROM runtime AS product
USER root
ENV ANOMALYGEN_PRODUCT_MODE=1
RUN for d in anomalygen assets ag_config scripts shmpatch; do \
     [ ! -e "$d" ] || { chown -R root:root "$d" && chmod -R a-w "$d"; }; \
     done \
     && for f in README.md pyproject.toml requirements.txt requirements-nodeps.txt docker/Dockerfile docker/build_wheels.sh docker/oss_bake.sh; do \
     [ ! -e "$f" ] || { chown root:root "$f" && chmod a-w "$f"; }; \
     done
USER nvidia
CMD [ "/bin/bash" ]

# ---- airgapped-base: runtime + everything a disconnected node needs ---------
# Sits beside product and develop, not below them: a stage gets one parent, so this is how the two
# air-gapped variants share the checkpoint tree instead of each materialising a copy.
# Requires `--build-context ckpts=./checkpoints` — .dockerignore excludes checkpoints/ by design.
FROM runtime AS airgapped-base
# No --chown: the weights land root-owned, so the uid running the model can read them and nothing
# else. checkpoint/utils.py verifies their digests; a writable copy leaves that the only defence.
# The two parents stay writable — HF locks under checkpoints/hf — and are made first, because a
# COPY only owns what it writes.
USER root
RUN mkdir -p /workspace/paidf-anomalygen/checkpoints/hf \
     && chown nvidia:nvidia /workspace/paidf-anomalygen/checkpoints \
                            /workspace/paidf-anomalygen/checkpoints/hf
COPY --from=ckpts Cosmos3-Nano           /workspace/paidf-anomalygen/checkpoints/Cosmos3-Nano
COPY --from=ckpts Cosmos3-Edge           /workspace/paidf-anomalygen/checkpoints/Cosmos3-Edge
COPY --from=ckpts wan2pt2/Wan2.2_VAE.pth /workspace/paidf-anomalygen/checkpoints/wan2pt2/Wan2.2_VAE.pth
COPY --from=ckpts facebook/dinov2-large  /workspace/paidf-anomalygen/checkpoints/facebook/dinov2-large
COPY --from=ckpts nvidia/C-RADIO-V3      /workspace/paidf-anomalygen/checkpoints/nvidia/C-RADIO-V3
COPY --from=ckpts hf/models--nvidia--Cosmos-Guardrail1 \
     /workspace/paidf-anomalygen/checkpoints/hf/models--nvidia--Cosmos-Guardrail1
COPY --from=ckpts hf/models--Qwen--Qwen3Guard-Gen-0.6B \
     /workspace/paidf-anomalygen/checkpoints/hf/models--Qwen--Qwen3Guard-Gen-0.6B
# Caption tokenizer resolved from the HF cache.
COPY --from=ckpts hf/models--Qwen--Qwen3-VL-8B-Instruct \
     /workspace/paidf-anomalygen/checkpoints/hf/models--Qwen--Qwen3-VL-8B-Instruct
COPY --from=ckpts hf/models--nvidia--Cosmos3-Edge \
     /workspace/paidf-anomalygen/checkpoints/hf/models--nvidia--Cosmos3-Edge
USER root
ENV UV_CACHE_DIR=/opt/uv/cache
# "0", never "": uv parses this as a boolean and rejects an empty value, breaking every uv call.
ENV UV_NO_CACHE=0
# One RUN on purpose: a chmod in a later layer would rewrite all 23 MB and ship it twice.
RUN <<'EOF'
set -eu
# Ask the interpreter for the path — ${PY_VER} is not re-declared in this stage.
HF_CLI="$(python -c 'import importlib.util as iu, pathlib; print(pathlib.Path(iu.find_spec("cosmos_framework").origin).parent / "utils" / "hf_cli")')"
test -f "$HF_CLI/uv.lock"
mkdir -p "$UV_CACHE_DIR"
# One run populates all 16 packages (~23 MB).
uv run --isolated --project "$HF_CLI" --locked --no-default-groups hf --version
# World-writable: uv rewrites the cache every run and `docker run --user` fixes no owner.
# Never sticky — uv renames into place, and sticky stops a non-root uid replacing root's files.
chmod -R a+rwX "$UV_CACHE_DIR"
# Prove it offline as the runtime user, so a broken cache fails the build rather than a customer's
# disconnected node. setpriv keeps the environment; `su` would reset PATH.
# Drop the interpreter entry so the test hits the rename, like every runtime start does.
rm -rf "$UV_CACHE_DIR"/interpreter-v*
setpriv --reuid=nvidia --regid=nvidia --clear-groups \
     env HOME=/tmp uv run --offline --isolated --project "$HF_CLI" --locked --no-default-groups hf --version
# The self-test just wrote new entries under nvidia's umask; open those up too.
chmod -R a+rwX "$UV_CACHE_DIR"
echo "uv hf_cli cache: $(du -sh "$UV_CACHE_DIR" | cut -f1)"
EOF
# Make the baked checkpoints the only source a model can load from.
#
# Every checkpoint is already pinned by revision when download_checkpoints.sh fetches it, but some
# framework code calls from_pretrained() with no revision. Without these, a cache miss in that code
# silently resolves a mutable branch over the network and loads whatever it finds; with them, the
# same miss fails loudly and the baked, pinned copy is authoritative.
#
# Set on the air-gapped targets only: this image is built for a disconnected node, so refusing to
# reach the network costs nothing here. Product and develop keep network access — a develop user
# may legitimately run download_checkpoints.sh inside the container — and can opt in per run with
# `docker run -e HF_HUB_OFFLINE=1 ...` (see docker/README.md).
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
USER nvidia

# ---- airgapped-product: airgapped-base + the product hardening --------------
FROM airgapped-base AS airgapped-product
USER root
ENV ANOMALYGEN_PRODUCT_MODE=1
RUN for d in anomalygen assets ag_config scripts shmpatch; do \
     [ ! -e "$d" ] || { chown -R root:root "$d" && chmod -R a-w "$d"; }; \
     done \
     && for f in README.md pyproject.toml requirements.txt requirements-nodeps.txt docker/Dockerfile docker/build_wheels.sh docker/oss_bake.sh; do \
     [ ! -e "$f" ] || { chown root:root "$f" && chmod a-w "$f"; }; \
     done
USER nvidia
CMD [ "/bin/bash" ]

# ---- airgapped-develop: airgapped-base + the dev/test tooling ---------------
FROM airgapped-base AS airgapped-develop
USER root
ARG TORCH_INDEX_URL
RUN uv pip install --no-cache -r requirements-dev.txt \
     && python -c "import pathlib, setuptools; p = pathlib.Path(setuptools.__file__).parent / '_vendor' / 'autocommand'; assert not p.exists(), f'LGPL-3.0-only autocommand restored by requirements-dev.txt at {p}'; print('autocommand absent:', p)"
RUN oss-bake pip && oss-bake manifest
USER nvidia
CMD [ "/bin/bash" ]

# ---- develop (DEFAULT target): runtime + dev/test tooling, interactive bash -
# Last stage in the file, which is what makes it the default `--target`.
FROM runtime AS develop
USER root
ARG TORCH_INDEX_URL
RUN uv pip install --no-cache -r requirements-dev.txt \
     && python -c "import pathlib, setuptools; p = pathlib.Path(setuptools.__file__).parent / '_vendor' / 'autocommand'; assert not p.exists(), f'LGPL-3.0-only autocommand restored by requirements-dev.txt at {p}'; print('autocommand absent:', p)"
RUN oss-bake pip && oss-bake manifest
USER nvidia
CMD [ "/bin/bash" ]
