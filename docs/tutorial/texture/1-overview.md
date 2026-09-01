<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 1 · Overview & Environment Setup

This is the first page of the PAIDF AnomalyGen texture tutorial series:

1. **Overview & Setup** ←
2. [Dataset Preparation](2-dataset-preparation.md)
3. [Auto Mask Placement](3-auto-mask-placement.md)
4. [Fine-tuning](4-fine-tuning.md)
5. [Generation](5-generation.md)
6. [Evaluation, Refinement & Pseudo-labeling](6-evaluation-and-refinement.md)

## Overview

### What is PAIDF AnomalyGen for texture-based defects?

**Physical AI Data Factory (PAIDF) AnomalyGen** is a diffusion-based pipeline for **Synthetic Data
Generation (SDG)** of anomaly images in a few-shot scenario. It adapts the base Cosmos generator into
an **anomaly-generation** model for *your* defect types: given a clean product image and a mask
marking where a defect should go, the fine-tuned model paints a realistic defect of the requested
type into that region.

Because only a small parameter subset — a per-defect-type LoRA — is trained on top of a frozen base
network, a handful of real examples per defect is enough to fine-tune, runs are cheap, and the
resulting checkpoints are tiny: a scalable source of synthetic anomaly data for training downstream
defect detectors and segmenters.

### The workflow

The series walks the full pipeline end to end. The seven stages listed in the
[README](../../../README.md) map onto six pages — evaluation, quality refinement and
pseudo-labeling all live on the last one:

1. **Overview & Environment Setup** (this page) — what you'll build and how the pieces fit together;
   install (native `uv` venv or Docker), download the checkpoints, and preflight the environment.
2. **[Dataset Preparation](2-dataset-preparation.md)** — turn an upstream source into the
   `anomaly_image/` + `mask/` + `clean_image/` layout the trainer expects (worked example:
   `phone_screen`).
3. **[Auto Mask Placement](3-auto-mask-placement.md)** — decide where each defect goes and place its
   mask on the clean images, building the `testcase.jsonl` that fine-tuning and generation consume.
4. **[Fine-tuning](4-fine-tuning.md)** — train the per-defect-type LoRAs from a recipe; produces a
   run dir with checkpoints and validation samples.
5. **[Generation](5-generation.md)** — run the fine-tuned checkpoint over a batch of testcases to
   synthesize defect images (SDG).
6. **[Evaluation, Refinement & Pseudo-labeling](6-evaluation-and-refinement.md)** — score the batch
   with the correspondence and anomaly-quality metrics and optionally split it into a clean `keep/`
   set, refine per-sample generation params by searching for a better `(guidance, crop_ratio)`, then
   pseudo-label the result into a ready-to-train COCO / classification dataset (with optional
   captions).

Each stage feeds the next: the dataset from stage 2 is placed into testcases in stage 3, which train
the model in stage 4, whose checkpoint drives generation in stage 5 — whose output is scored, refined
and pseudo-labeled in stage 6.

## Environment setup

The rest of this page sets up your environment: by the end you will have a working environment
(Python 3.13, torch 2.12.1+cu132, CUDA 13.2), the model checkpoints downloaded, and a passing
preflight check.

### Choose an install path

There are two supported ways to get a working environment — pick **one**:

- **Path A — Native (uv):** build the venv directly in the repo (compiles the heavy extensions from source, **> 1 h**) and activate it.
- **Path B — Docker:** pull a prebuilt image (fastest) and open an **interactive shell inside the container**.

Either path leaves you with a shell at the repo root where the environment is ready. **Every command from
[Download checkpoints & preflight](#download-checkpoints--preflight) onward — and in all later tutorial
pages — is run the same way in that shell**, so the commands are shown once with no per-path variants.

### Prerequisites

Complete these before either path.

1. **NVIDIA driver / CUDA 13.2.** Confirm your driver supports CUDA 13.2:

   ```shell
   nvidia-smi
   ```

2. **Hugging Face token.** Create an `HF_TOKEN` and **accept both the `nvidia/Cosmos3-Nano` and
   `nvidia/Cosmos3-Edge` licenses** on Hugging Face — `scripts/download_checkpoints.sh` converts
   both base sizes, and a gated-repo `403` on either one aborts the whole script, including the
   DINOv2 / C-RADIO / SAM2 / guardrail models it pulls afterwards. Budget ≈ 85 GB of disk for the
   full checkpoint set.

3. **Native path (Path A) only:** a host C/C++ toolchain (`gcc`/`g++`) and `git` — several
   dependencies are compiled from source or installed straight from a git URL — and a **visible
   GPU at build time**, since some of the compiled CUDA extensions autodetect the target arch and
   silently build without GPU kernels if they cannot see one.

4. **Docker path (Path B) only:** install **Docker 23+** and the
   **nvidia-container-toolkit**. The image is published publicly on NGC, so no registry
   login is required to pull it.

### Get the source

Clone the AnomalyGen source repo.

```shell
git clone https://github.com/NVIDIA/paidf-anomalygen.git
cd paidf-anomalygen
```

> **⚠ All commands below are run from the repo root (`paidf-anomalygen`).**

### Path A — Native (uv)

1. **Install uv and build the venv.** This compiles torch, the CUDA extensions,
   and the in-repo packages (**> 1 hour**).

   ```shell
   curl -LsSf https://astral.sh/uv/install.sh | sh
   bash scripts/env_setup.sh
   ```

   **Note:** The `env_setup.sh` script takes two **optional** positional args controlling build
   parallelism — `MAX_JOBS` (parallel compile jobs, default `16`) and
   `NVCC_THREADS` (threads per `nvcc`, default `1`). Lower `MAX_JOBS` if the
   build runs out of memory, e.g. `bash scripts/env_setup.sh 8 1`.

   If you're developing the repo, also run:

   ```shell
   uv pip install -r requirements-dev.txt && pre-commit install
   ```

2. **Activate and authenticate.**

   ```shell
   source .venv/bin/activate
   hf auth login  # or run `export HF_TOKEN=<your-hf-token>` for env-only.
   ```

You now have an **activated shell at the repo root**. Skip to
[Download checkpoints & preflight](#download-checkpoints--preflight).

### Path B — Docker

Uses the prebuilt image published on
[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/paidf-anomalygen) — a public
container, so no registry login is needed.

1. **Pull the image.**

   ```shell
   export IMAGE=nvcr.io/nvidia/paidf-anomalygen:1.1.0
   docker pull "$IMAGE"
   ```

   The tag tracks this repo's `VERSION` (`1.1.0`); the
   [tag list](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/paidf-anomalygen/tags)
   shows what is published. If the pull fails with a *manifest unknown* error, that version
   has **not been released yet — build it yourself (step 2)**. Do not fall back to an older
   tag: earlier tags were built from a different source revision and will not match the code
   you just cloned.

2. **Build it yourself.** Needed whenever the tag above is unavailable, and the way to get a
   modified image. It compiles the heavy CUDA extensions from source and takes **> 1 hour**;
   see [docker/README.md](../../../docker/README.md) for the build targets and options.

3. **Open an interactive shell in the container.** Define a run helper that mounts the repo and
   forwards your `HF_TOKEN`, then enter the container with `-it`:

   ```shell
   export HF_TOKEN=<your-hf-token>
   DOCKER="docker run --rm --gpus all --shm-size=16g \
     --user $(id -u):$(id -g) -e USER=$(id -un) -e HOME=/tmp -e HF_TOKEN \
     -v $PWD:/workspace/paidf-anomalygen \
     -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
     -w /workspace/paidf-anomalygen $IMAGE"

   $DOCKER -it bash      # drops you into a shell inside the container, at the repo root
   ```

You're now in an **interactive container shell** at the repo root, with `HF_TOKEN` set. Run every
command below (and in the later tutorial pages) here — they're identical to the native path.

### Download checkpoints & preflight

Run these in your shell (native venv **or** container — same commands either way).

1. **Download checkpoints.** The script is idempotent — safe to re-run.

   ```shell
   scripts/download_checkpoints.sh
   ```

2. **Verify the env.** Print the installed torch version and whether it can see a GPU:

   ```shell
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   # expected output: 2.12.1+cu132 True
   ```

   - `2.12.1+cu132` — torch 2.12.1 built against CUDA 13.2 (the version this repo needs).
   - `True` — torch can see a GPU. If it prints `False`, torch cannot reach a CUDA
     device: recheck your driver with `nvidia-smi` and confirm you are on a GPU node.

3. **Run the preflight check.** This is the smoke test for your environment: it checks the GPU,
   Python, CUDA/torch, every in-repo and compiled dependency, Hugging Face auth, and that each model
   checkpoint actually resolved on disk.

   ```shell
   scripts/preflight_env_ckpt.sh
   ```

   It ends with `All checks passed.` or a per-group summary of what failed. **Read which group
   failed before reacting** — only one of them is fixed by downloading:

   - *missing checkpoints* → re-run `scripts/download_checkpoints.sh` (authenticate first).
   - *env / Python / deps* → rebuild the environment (`bash scripts/env_setup.sh`, then activate it).

## Next step

Continue to [2 · Dataset Preparation](2-dataset-preparation.md).
