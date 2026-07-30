# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# =============================================================================
# PAIDF AnomalyGen Docker Image
#
# Single-stage build: combines the compile-heavy dependencies (flash-attn,
# transformer-engine, apex, opencv-python-headless, requirements) and
# the application layer in one Dockerfile. The pre-built CUDA-extension
# wheels live in the project's GitLab PyPI registry; CI passes a token-bearing
# index URL via --build-arg GITLAB_PYPI_INDEX_URL (see .gitlab-ci.yml). For
# local builds without that token, each compile-heavy RUN falls back to a
# source compile.
#
# RUNNING THIS IMAGE:
#
#   IMAGE=paidf-anomalygen:<tag>
#   HF_TOKEN=<your_token>
#   docker run --rm -it --gpus all --shm-size=16g \
#       --user "$(id -u):$(id -g)" \
#       -e USER="$(id -un)" \
#       -e HF_TOKEN \
#       -e HOME=/tmp \
#       -v "$PWD/checkpoints":/workspace/paidf-anomalygen/checkpoints \
#       -v "$PWD/datasets":/workspace/paidf-anomalygen/datasets \
#       -v "$PWD/ag_configs":/workspace/paidf-anomalygen/ag_configs \
#       -v "$PWD/ag_inference":/workspace/paidf-anomalygen/ag_inference \
#       -v "$PWD/results":/workspace/paidf-anomalygen/results \
#       -v /etc/passwd:/etc/passwd:ro \
#       -v /etc/group:/etc/group:ro \
#       -w /workspace/paidf-anomalygen \
#       $IMAGE bash
#
# =============================================================================

ARG PYTORCH_BASE=nvidia/cuda:12.8.2-devel-ubuntu24.04
FROM ${PYTORCH_BASE} AS base

ARG TORCH_VER=2.10.0
ARG TORCHVISION_VER=0.25.0
ARG EINOPS_VER=0.8.2
ARG FA_VER=2.8.3
ARG TE_VER=2.13.0
ARG OPENCV_VER=4.13.0.92
ARG OPENCV_GH_BRANCH=92
ARG APEX_VER=0.1
ARG APEX_GH_COMMIT=466e164bfd9548b3026c1f3a1c296bed8ef55c43
ARG REQUIREMENTS=requirements-conda-cuda128.txt

# Pre-compiled wheels for flash-attn / transformer-engine / apex /
# opencv-python-headless live in this project's GitLab PyPI registry. CI
# passes a token-bearing index URL via --build-arg. Leave unset for local
# builds — each compile-heavy step falls back to source compile.
ARG GITLAB_PYPI_INDEX_URL=

# Bump this string when Ubuntu packages need a security refresh without changing
# requirements.
ARG APT_CACHE_BUST=2026-05-11

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "apt-cache-bust: $APT_CACHE_BUST" \
    && dpkg-query -W -f='${binary:Package}\n' | sort -u > /tmp/_pkgs_before \
    && apt-get update && apt-get install -y --no-install-recommends \
    wget git build-essential \
    libglib2.0-0 libgl1 \
    libssl-dev libffi-dev \
    python3.12 python3.12-dev python3-pip python-is-python3 \
    cmake ninja-build unzip \
    && apt-get upgrade -y --no-install-recommends \
    && dpkg-query -W -f='${binary:Package}\t${Source}\n' \
       | awk -F'\t' '{print $1 "\t" ($2=="" ? $1 : $2)}' \
       | sed 's/ (.*//' > /tmp/_pkgs_after \
    && rm -rf /var/lib/apt/lists/*

# Bypass PEP 668 so pip can install to the system Python (the container itself
# is the isolation boundary).
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && \
    rm -rf /usr/lib/python3/dist-packages/wheel \
           /usr/lib/python3/dist-packages/wheel-*.dist-info \
           /usr/lib/python3/dist-packages/wheel-*.egg-info \
           /usr/lib/python3/dist-packages/setuptools \
           /usr/lib/python3/dist-packages/setuptools-*.dist-info \
           /usr/lib/python3/dist-packages/setuptools-*.egg-info \
           /usr/lib/python3/dist-packages/packaging \
           /usr/lib/python3/dist-packages/packaging-*.dist-info \
           /usr/lib/python3/dist-packages/packaging-*.egg-info && \
    pip install --no-cache-dir --ignore-installed setuptools packaging wheel "pip>=26.1"

# Install build-time deps (torch + einops). Versions must match the pins in
# requirements-conda-cuda128.txt.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==${TORCH_VER} torchvision==${TORCHVISION_VER} einops==${EINOPS_VER} \
 && pip cache purge

# Symlink nvidia/*/include into the standard include paths (needed by native
# compiles below to find CUDA headers shipped by torch's nvidia-* deps).
RUN ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/*/include/* /usr/include/ && \
    ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/*/include/* /usr/include/python3.12/

# flash-attn: pre-built wheel from the GitLab PyPI registry.
# Falls back to source compile if the index URL is unset.
RUN MAX_JOBS=16 NVCC_THREADS=1 \
    pip install --no-cache-dir --no-build-isolation \
      --extra-index-url "${GITLAB_PYPI_INDEX_URL}" \
      flash-attn==${FA_VER}

# transformer-engine: pre-built wheels from the GitLab PyPI registry
# (transformer-engine meta + transformer-engine-cu12 + transformer-engine-torch).
# Falls back to source compile if the index URL is unset.
RUN MAX_JOBS=16 NVCC_THREADS=1 \
    pip install --no-cache-dir --no-build-isolation \
      --extra-index-url "${GITLAB_PYPI_INDEX_URL}" \
      "transformer-engine[pytorch]==${TE_VER}"

# apex: prefer the pre-built wheel from the GitLab PyPI registry. If the
# registry URL is unset (local builds without a token), fall back to a
# git+source build with --cpp_ext --cuda_ext.
RUN if [ -n "$GITLAB_PYPI_INDEX_URL" ]; then \
      MAX_JOBS=16 NVCC_THREADS=1 \
      pip install --no-cache-dir --no-build-isolation \
        --extra-index-url "${GITLAB_PYPI_INDEX_URL}" \
        apex==${APEX_VER}; \
    else \
      MAX_JOBS=16 NVCC_THREADS=1 pip install --force-reinstall -v \
        --disable-pip-version-check --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext --cuda_ext" \
        git+https://github.com/NVIDIA/apex.git@${APEX_GH_COMMIT}; \
    fi

# opencv-python-headless: prefer the pre-built wheel from the GitLab PyPI
# registry (compiled with WITH_FFMPEG=OFF to scrub the FFmpeg CVEs). If the
# registry URL is unset (local builds without a token), fall back to a
# from-source build that disables FFmpeg — never install the PyPI wheel,
# which bundles vulnerable libav*.so / libsw*.so.
RUN if [ -n "$GITLAB_PYPI_INDEX_URL" ]; then \
      pip install --no-cache-dir --no-deps \
        --extra-index-url "${GITLAB_PYPI_INDEX_URL}" \
        opencv-python-headless==${OPENCV_VER}; \
    else \
      git clone --recursive --shallow-submodules --depth 1 --branch ${OPENCV_GH_BRANCH} \
        https://github.com/opencv/opencv-python.git /tmp/opencv-python && \
      cd /tmp/opencv-python && \
      ENABLE_HEADLESS=1 CMAKE_ARGS="-DWITH_FFMPEG=OFF" MAKEFLAGS="-j16" \
        pip install --no-cache-dir --no-deps . && \
      cd / && rm -rf /tmp/opencv-python; \
    fi

# Install full pip requirements (cheap once the heavy compiles above are cached)
COPY ${REQUIREMENTS} /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt \
 && pip cache purge

# Drop PyAV (from qwen-vl-utils): its bundled FFmpeg .so files flag the same CVEs.
RUN pip uninstall -y av

# Register pip-installed nvidia/*/lib paths with the dynamic linker so non-torch
# CUDA consumers (flash-attn, TE) can find cuBLAS / cuDNN / cuSPARSE etc.
# without LD_LIBRARY_PATH gymnastics. (torch itself has RPATH and doesn't need this.)
RUN { for d in /usr/local/lib/python3.12/dist-packages/nvidia/*/lib; do echo "$d"; done; \
    } > /etc/ld.so.conf.d/cosmos.conf && ldconfig

# OSRB / container scan: drop paths that aren't needed at runtime.
RUN rm -rf /opt/pytorch /root/.cache/pip || true

# OSS source-code bake — fetch upstream source for every apt package that
# WE added on top of the base CUDA image (computed via dpkg-state diff
# stashed at apt-install time), and store under /usr/share/oss-sources/.
RUN sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources && \
    apt-get update && \
    mkdir -p /usr/share/oss-sources && cd /usr/share/oss-sources && \
    # Compute the set of NEW packages (binary names) added on top of the base
    awk 'NR==FNR{base[$0]=1; next} !($1 in base){print $2}' \
        /tmp/_pkgs_before /tmp/_pkgs_after \
      | sort -u \
      | grep -vE '^(cuda-|libcublas-|libcusparse-|libnccl|libnv|nvidia-)' \
      > /tmp/_src_pkgs && \
    echo "Source bake: $(wc -l < /tmp/_src_pkgs) source packages added on top of the base image:" && \
    cat /tmp/_src_pkgs && \
    : > /tmp/_src_failed.log && \
    while read pkg; do \
        apt-get source --download-only "$pkg" >/dev/null 2>&1 || \
            echo "$pkg" >> /tmp/_src_failed.log; \
    done < /tmp/_src_pkgs && \
    echo "Source download done. $(ls *.orig.* 2>/dev/null | wc -l) upstream tarballs, $(ls *.dsc 2>/dev/null | wc -l) .dsc files." && \
    if [ -s /tmp/_src_failed.log ]; then \
        echo "Packages with no source available (skipped):"; \
        cat /tmp/_src_failed.log; \
    fi && \
    # Generate a manifest of (file, sha256) for integrity audit
    (cd /usr/share/oss-sources && sha256sum * 2>/dev/null > MANIFEST.sha256) && \
    # Restore deb822 sources (deb-only) and clear apt cache
    sed -i 's/^Types: deb deb-src$/Types: deb/' /etc/apt/sources.list.d/ubuntu.sources && \
    rm -rf /var/lib/apt/lists/* /tmp/_pkgs_before /tmp/_pkgs_after /tmp/_src_pkgs /tmp/_src_failed.log

# ---------------------------------------------------------------------------
# Non-root runtime user. The baked-in UID/GID can be overridden at run time
# with `docker run --user $UID:$GID` to align with the host user when bind-
# mounting writable directories.
# ---------------------------------------------------------------------------
ARG APP_UID=10000
ARG APP_GID=10000
RUN groupadd --gid ${APP_GID} anomalygen \
 && useradd --create-home --shell /bin/bash \
            --uid ${APP_UID} --gid ${APP_GID} anomalygen

# ---------------------------------------------------------------------------
# Bake source code into the image. Placed last so source-only changes only
# invalidate this layer (the heavy install layers above stay cached).
# ---------------------------------------------------------------------------
COPY --chown=anomalygen:anomalygen . /workspace/paidf-anomalygen

RUN cp /NGC-DL-CONTAINER-LICENSE /workspace/paidf-anomalygen/NGC-DL-CONTAINER-LICENSE \
 && chown anomalygen:anomalygen /workspace/paidf-anomalygen/NGC-DL-CONTAINER-LICENSE

USER anomalygen

WORKDIR /workspace/paidf-anomalygen

ENV PYTHONPATH=/workspace/paidf-anomalygen
ENV ANOMALYGEN_SCRIPTS=/workspace/paidf-anomalygen/scripts/utilities

# ---------------------------------------------------------------------------
# product: same image as base, plus ANOMALYGEN_PRODUCT_MODE=1 and production
# code locked read-only. Build with `--target product`.
# ---------------------------------------------------------------------------
FROM base AS product

USER root
ENV ANOMALYGEN_PRODUCT_MODE=1

# Lock production code + key files read-only (runtime/data dirs stay writable).
RUN for d in automatic_mask_placement cosmos_predict2 pseudo_label roi_generate imaginaire \
             scripts/anomaly_gen scripts/utilities skills; do \
        [ ! -e "$d" ] || { chown -R root:root "$d" && chmod -R a-w "$d"; }; \
    done && \
    for f in README.md CLAUDE.md docker/Dockerfile* requirements*.txt cosmos-predict2-cuda*.yaml; do \
        [ ! -e "$f" ] || { chown root:root "$f" && chmod a-w "$f"; }; \
    done

USER anomalygen

CMD [ "/bin/bash" ]

# ---------------------------------------------------------------------------
# Air-gapped variants: same image as product/develop, plus checkpoints baked
# into the layers so the container runs with no network and no volume mounts.
#
# Checkpoints are supplied via a named build context `ckpts` rather than the
# main build context, so they bypass the repo `.dockerignore`.
#
#   docker buildx build --target airgapped-product \
#       --build-context ckpts=./checkpoints \
#       -f docker/Dockerfile -t paidf-anomalygen-airgapped:<tag> .
# ---------------------------------------------------------------------------
FROM product AS airgapped-product

COPY --from=ckpts --chown=anomalygen:anomalygen README.md                             /workspace/paidf-anomalygen/checkpoints/README.md
COPY --from=ckpts --chown=anomalygen:anomalygen sam2                                  /workspace/paidf-anomalygen/checkpoints/sam2
COPY --from=ckpts --chown=anomalygen:anomalygen NVDINOV2                              /workspace/paidf-anomalygen/checkpoints/NVDINOV2
COPY --from=ckpts --chown=anomalygen:anomalygen google-t5                             /workspace/paidf-anomalygen/checkpoints/google-t5
COPY --from=ckpts --chown=anomalygen:anomalygen facebook                              /workspace/paidf-anomalygen/checkpoints/facebook
COPY --from=ckpts --chown=anomalygen:anomalygen nvidia/C-RADIO-V3                     /workspace/paidf-anomalygen/checkpoints/nvidia/C-RADIO-V3
COPY --from=ckpts --chown=anomalygen:anomalygen nvidia/Cosmos-Predict2-2B-Text2Image  /workspace/paidf-anomalygen/checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image
COPY --from=ckpts --chown=anomalygen:anomalygen nvidia/Cosmos-Predict2-14B-Text2Image /workspace/paidf-anomalygen/checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image
COPY --from=ckpts --chown=anomalygen:anomalygen nvidia/Cosmos-Guardrail1              /workspace/paidf-anomalygen/checkpoints/nvidia/Cosmos-Guardrail1
COPY --from=ckpts --chown=anomalygen:anomalygen Qwen                                  /workspace/paidf-anomalygen/checkpoints/Qwen

CMD [ "/bin/bash" ]

# airgapped-develop: develop (== base) + checkpoints baked in.
FROM base AS airgapped-develop

COPY --from=ckpts --chown=anomalygen:anomalygen README.md                             /workspace/paidf-anomalygen/checkpoints/README.md
COPY --from=ckpts --chown=anomalygen:anomalygen sam2                                  /workspace/paidf-anomalygen/checkpoints/sam2
COPY --from=ckpts --chown=anomalygen:anomalygen NVDINOV2                              /workspace/paidf-anomalygen/checkpoints/NVDINOV2
COPY --from=ckpts --chown=anomalygen:anomalygen google-t5                             /workspace/paidf-anomalygen/checkpoints/google-t5
COPY --from=ckpts --chown=anomalygen:anomalygen facebook                              /workspace/paidf-anomalygen/checkpoints/facebook
COPY --from=ckpts --chown=anomalygen:anomalygen nvidia/C-RADIO-V3                     /workspace/paidf-anomalygen/checkpoints/nvidia/C-RADIO-V3
COPY --from=ckpts --chown=anomalygen:anomalygen nvidia/Cosmos-Predict2-2B-Text2Image  /workspace/paidf-anomalygen/checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image
COPY --from=ckpts --chown=anomalygen:anomalygen nvidia/Cosmos-Predict2-14B-Text2Image /workspace/paidf-anomalygen/checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image
COPY --from=ckpts --chown=anomalygen:anomalygen nvidia/Cosmos-Guardrail1              /workspace/paidf-anomalygen/checkpoints/nvidia/Cosmos-Guardrail1
COPY --from=ckpts --chown=anomalygen:anomalygen Qwen                                  /workspace/paidf-anomalygen/checkpoints/Qwen

CMD [ "/bin/bash" ]

# ---------------------------------------------------------------------------
# develop (DEFAULT target): the base image as-is (non-root anomalygen),
# interactive bash. Last stage, so a plain `docker build` builds this.
# ---------------------------------------------------------------------------
FROM base AS develop

CMD [ "/bin/bash" ]
