---
name: anomalygen-release
description: >-
  Build and validate PAIDF AnomalyGen product and develop Docker containers
  from docker/Dockerfile. Use when the user asks to build an anomalygen
  product container, build an anomalygen develop container, validate container
  runtime permissions, or produce release summaries.
license: Apache-2.0
metadata:
  author: Wenny Lo <wennyl@nvidia.com>
  tags: [release, docker, container, airgapped, paidf]
---

# AnomalyGen Release

Use this skill to build and validate AnomalyGen CUDA 12.8 containers. This is a
release/build workflow, not a training or inference workflow.

Run every command from the repo root.

**Docker privilege.** All `docker` commands below assume your user is in
the `docker` group (or the daemon is rootless). If your environment
requires elevated privileges, prefix `sudo` to each `docker` invocation.

## Container Modes

There are two image modes:

- **Product container**: protected runtime for users operating AnomalyGen through
  an agent. The image sets `ANOMALYGEN_PRODUCT_MODE=1`, runs as a non-root user,
  locks production code read-only, and keeps runtime artifacts writable.
- **Develop container**: writable development environment for developers using
  an agent to edit code. The image leaves `ANOMALYGEN_PRODUCT_MODE` unset and
  keeps production code paths writable.

Each mode is available in two variants:

- **Standard**: checkpoints are mounted at runtime (thin image, ~10 GB).
- **Air-gapped**: checkpoints are baked into the image (fat image, ~75 GB+).
  Use when the target environment cannot reach the network at runtime.

Builds should happen from a normal cloned repo or develop container where
`ANOMALYGEN_PRODUCT_MODE` is unset. Do not build nested images from a product
runtime container.

User-facing prompts:

```text
Build anomalygen product container
Build anomalygen develop container
Generate airgapped docker image
Build airgapped product image
Build airgapped develop image
```

## Target Architecture

The repo ships two Dockerfiles, one per host architecture:

| Architecture | Dockerfile | CUDA | GPUs |
|---|---|---|---|
| **x86_64** (default) | `docker/Dockerfile` | 12.8 | Blackwell-class, e.g. RTX PRO 6000 |
| **arm64** | `docker/Dockerfile.arm.cuda130` | 13.0 | GB10 / GB200 / GB300 |

A Docker image is built for the **host's** architecture — this skill does not
cross-build. **Auto-detect the host arch and confirm the matching Dockerfile
with the user** before building (don't ask open-endedly — detect, then confirm):

```bash
uname -m   # x86_64 -> docker/Dockerfile ; aarch64/arm64 -> docker/Dockerfile.arm.cuda130
```

State the detected choice and let the user confirm or override, e.g. *"Detected
aarch64 → building with `docker/Dockerfile.arm.cuda130` (arm64 / CUDA 13).
Proceed?"* If the user already named an architecture, skip the confirmation and
use it. Only deviate from the host arch if the user explicitly asks.

Pass the chosen file via `--dockerfile` to `build_image.sh` /
`build_airgapped_image.sh` (the x86 `docker/Dockerfile` is the default, so it
can be omitted for x86). The scripts preflight the matching conda spec and
requirements file (`cuda130` for arm, `cuda128` otherwise) automatically.

For arm builds, give the image an arch-distinct `--tag` or `--image-name` (e.g.
`--tag cuda13-arm-$(date -u +%Y%m%d)`) so it does not clobber the x86 image.

## Scope

Allowed:

- Inspect `docker/Dockerfile` (all targets, including the airgapped ones)
  and release inputs.
- Build product or develop images (standard or air-gapped) with the helper scripts.
- Auto-download missing checkpoints when building an air-gapped image.
- Validate product image runtime guardrails.
- Validate develop image writability for code development.
- Report image tag, image id, mode, and validation results.

Do not:

- Run AnomalyGen training or SDG generation as part of release.
- Bake secrets, private datasets, or user experiment outputs into the image.
- Build images inside `ANOMALYGEN_PRODUCT_MODE=1`.
- Use arbitrary Docker commands beyond the build and validation steps unless
  the user explicitly asks.

## Canonical Build Commands

Product container:

```bash
bash skills/anomalygen-release/scripts/build_image.sh --mode product
```

Equivalent Docker command:

```bash
DATE_TAG="$(date -u +%Y%m%d)"
DOCKER_BUILDKIT=1 docker build \
    --target product \
    -f docker/Dockerfile \
    -t "paidf-anomalygen:${DATE_TAG}" \
    .
```

Develop container:

```bash
bash skills/anomalygen-release/scripts/build_image.sh --mode develop
```

Use a minute-level tag when multiple builds may happen in one day:

```bash
bash skills/anomalygen-release/scripts/build_image.sh \
    --mode product \
    --tag "$(date -u +%Y%m%d-%H%M)"
```

arm64 / CUDA-13 build (on an aarch64 host) — pass the arm Dockerfile and an
arch-distinct tag:

```bash
bash skills/anomalygen-release/scripts/build_image.sh \
    --mode product \
    --dockerfile docker/Dockerfile.arm.cuda130 \
    --tag "cuda13-arm-$(date -u +%Y%m%d)"
```

## Air-Gapped Image

Use when the target environment has no network access and cannot pull
checkpoints at runtime. The airgapped build uses the `airgapped-product` /
`airgapped-develop` targets of `docker/Dockerfile`, which bake all checkpoints
into the image layers via a named build context (`--build-context
ckpts=checkpoints`). The result is self-contained (~75 GB+).

### Canonical command

```bash
bash skills/anomalygen-release/scripts/build_airgapped_image.sh \
    --mode product
```

The script:

1. **Checks all required checkpoints** in `checkpoints/` (the paths the
   Dockerfile COPYs in):
   - `checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt`
   - `checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image/model.pt`
   - `checkpoints/NVDINOV2/nv_dinov2_classification_model.ckpt`
   - `checkpoints/nvidia/C-RADIO-V3/model.safetensors`
   - `checkpoints/nvidia/Cosmos-Guardrail1/` (image guardrail — three
     components: `video_content_safety_filter/safety_filter.pt`, the SigLIP
     encoder snapshot, and `face_blur_filter/Resnet50_Final.pth`)
   - `checkpoints/sam2/sam2.1_hiera_large.pt`
   - `checkpoints/Qwen/Qwen3-VL-4B-Instruct/` (non-empty)
   - `checkpoints/facebook/dinov2-large/` (non-empty)
   - `checkpoints/google-t5/t5-large/` **and** `checkpoints/google-t5/t5-11b/`
     (both baked in — t5-large is the default encoder, t5-11b/T5-XXL is for
     configs that select it via `ag_config.t5_model_name`)

2. **Auto-downloads** any missing checkpoints via
   `scripts/utilities/download_checkpoints.sh --model-sizes "2B 14B" --with-t5-11b`
   (both base sizes and both T5 variants are baked into the image; t5-large +
   guardrail come by default, `--with-t5-11b` adds T5-XXL ~45 GB).
   This requires
   `HF_TOKEN` to be exported and the `hf` CLI (`huggingface_hub >= 1.x`)
   in `PATH`. If you do
   not want auto-download, pass `--skip-download` and download manually
   with the setup skill first.

3. **Builds** the airgapped image after all checkpoints are confirmed present.

Default image names:

| Mode | Tag |
|---|---|
| `product` | `paidf-anomalygen-airgapped:<date>` |
| `develop` | `paidf-anomalygen-dev-airgapped:<date>` |

Options:

```bash
bash skills/anomalygen-release/scripts/build_airgapped_image.sh \
    --mode product|develop \
    --tag YYYYMMDD \
    --checkpoint-dir checkpoints \
    --dockerfile docker/Dockerfile \
    --skip-download
```

Pass `--dockerfile docker/Dockerfile.arm.cuda130` to build the arm64 / CUDA-13
air-gapped image instead of the default x86 / CUDA-12.8 one (build on an aarch64
host). Give it a distinct `--tag` or `--image-name` to avoid clobbering the x86
image's tag.

### Running the air-gapped image

No volume mounts required — checkpoints live inside the image:

```bash
docker run --gpus all -it --rm --shm-size=16g \
    paidf-anomalygen-airgapped:<tag> bash
```

### Delivering to an air-gapped host

```bash
# on the build host
docker save paidf-anomalygen-airgapped:<tag> | gzip \
    > paidf-anomalygen-airgapped-<tag>.tar.gz

# transfer the .tar.gz to the air-gapped host, then:
docker load < paidf-anomalygen-airgapped-<tag>.tar.gz
docker run --gpus all -it --rm --shm-size=16g \
    paidf-anomalygen-airgapped:<tag> bash
```

## Product Filesystem Policy

The product image should run as a non-root runtime user. Only production
implementation paths should be non-writable. Runtime data, artifacts,
checkpoints, caches, and logs should remain writable.

Production paths expected to be non-writable in product containers and writable
in develop containers:

```text
/workspace/paidf-anomalygen/cosmos_predict2/
/workspace/paidf-anomalygen/imaginaire/
/workspace/paidf-anomalygen/automatic_mask_placement/
/workspace/paidf-anomalygen/pseudo_label/
/workspace/paidf-anomalygen/roi_generate/
/workspace/paidf-anomalygen/scripts/anomaly_gen/
/workspace/paidf-anomalygen/scripts/utilities/
/workspace/paidf-anomalygen/skills/
/workspace/paidf-anomalygen/README.md
/workspace/paidf-anomalygen/docker/Dockerfile*
/workspace/paidf-anomalygen/requirements*.txt
/workspace/paidf-anomalygen/cosmos-predict2-cuda128.yaml
```

Runtime paths expected to be writable in both modes:

```text
/workspace/paidf-anomalygen/results/
/workspace/paidf-anomalygen/ag_configs/
/workspace/paidf-anomalygen/ag_inference/
/workspace/paidf-anomalygen/datasets/
/workspace/paidf-anomalygen/checkpoints/
/workspace/paidf-anomalygen/logs/
/workspace/paidf-anomalygen/tmp/
/tmp/
the runtime user's `$HOME` (`/home/anomalygen/` in product images)
the runtime user's `$HOME/.cache` (used by HF, torch.compile, and triton)
```

In addition to the top-level directories above, the validator probes a set of
nested paths that AnomalyGen actually writes to during training, SDG, and
refine. Top-level writability alone can miss ownership issues on mounted
volumes or pre-created subdirectories, so each of these must also be writable:

```text
# Training output
/workspace/paidf-anomalygen/results/anomaly_gen/_permission_test/checkpoints/model

# SDG output (original and searched buckets, plus per-round refine)
/workspace/paidf-anomalygen/results/_permission_test/original/reconstructed_image
/workspace/paidf-anomalygen/results/_permission_test/original/original_mask
/workspace/paidf-anomalygen/results/_permission_test/original/annotated_image
/workspace/paidf-anomalygen/results/_permission_test/searched/reconstructed_image
/workspace/paidf-anomalygen/results/_permission_test/rounds/round_001/sdg/reconstructed_image

# AMP cache
/workspace/paidf-anomalygen/ag_inference/_permission_test/amp
/workspace/paidf-anomalygen/ag_inference/_permission_test/resized_masks

# torch.compile / triton cache
$HOME/.cache/paidf-anomalygen/torch_inductor
$HOME/.cache/paidf-anomalygen/triton

# Hugging Face cache
$HOME/.cache/huggingface
```

The validator also probes that `SDG_result.csv` can be created and written
under each of the SDG bucket roots:

```text
/workspace/paidf-anomalygen/results/_permission_test/original/SDG_result.csv
/workspace/paidf-anomalygen/results/_permission_test/searched/SDG_result.csv
/workspace/paidf-anomalygen/results/_permission_test/rounds/round_001/sdg/SDG_result.csv
```

This `SDG_result.csv` probe exists so the validator can fail fast on
read-only mounts where SDG would later be unable to record its index. It does
not mean this release skill runs SDG itself; SDG generation remains
out-of-scope for the release workflow (see Scope above) and is handled by the
`sdg-inference` skill.

If a product image does not set `ANOMALYGEN_PRODUCT_MODE=1`, runs as root, or
leaves production code writable, treat the image as not productized.

## Pre-Build Checklist

Before building:

1. Verify these files exist:
   - `docker/Dockerfile`
   - `cosmos-predict2-cuda128.yaml`
   - `requirements-conda-cuda128.txt`
2. Verify the Docker build context does not intentionally include secrets:
   - `.env`
   - token files
   - SSH keys
   - private datasets
   - user result folders
3. Decide checkpoint strategy for product images:
   - Thin image: checkpoints mounted or downloaded at runtime.
   - Fat image: checkpoints copied into image. This is large and should be
     explicitly requested.

## Validation

After building a product image:

```bash
bash skills/anomalygen-release/scripts/validate_image_permissions.sh \
    --mode product \
    "paidf-anomalygen:${DATE_TAG}"
```

After building a develop image:

```bash
bash skills/anomalygen-release/scripts/validate_image_permissions.sh \
    --mode develop \
    "paidf-anomalygen-dev:${DATE_TAG}"
```

If validation fails, do not call the image ready for its intended mode.

## Running the Container

> **Warning:** always pass `--shm-size` (minimum `16g`). PyTorch DataLoader
> uses `/dev/shm` for multiprocessing shared memory; the Docker default of
> 64 MB causes "Bus error" crashes or silent hangs during training and
> inference. Remind the user of this flag whenever reporting a completed build.

Product container:

```bash
TAG="paidf-anomalygen:$(date -u +%Y%m%d)"
REPO="$PWD"
docker run --rm -it --gpus all --shm-size=16g \
    -v "${REPO}/checkpoints:/workspace/paidf-anomalygen/checkpoints" \
    -v "${REPO}/datasets:/workspace/paidf-anomalygen/datasets" \
    -v "${REPO}/results:/workspace/paidf-anomalygen/results" \
    "${TAG}" \
    bash
```

Develop container:

```bash
TAG="paidf-anomalygen-dev:$(date -u +%Y%m%d)"
REPO="$PWD"
docker run --rm -it --gpus all --shm-size=16g \
    -v "${REPO}:/workspace/paidf-anomalygen" \
    "${TAG}" \
    bash
```

## Release Summary

Report:

```text
Image: paidf-anomalygen:<tag>
Mode: product | develop
Image ID: <docker image id>
Dockerfile: docker/Dockerfile
ANOMALYGEN_PRODUCT_MODE: set | unset | failed validation
Production code: non-writable | writable | failed validation
Runtime paths: writable | failed validation
Checkpoint strategy: thin | fat | unknown
Notes: <warnings, if any>
```
