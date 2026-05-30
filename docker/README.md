# AnomalyGen Docker Image

Docker build for **PAIDF AnomalyGen** on **CUDA 12.8** (Blackwell-class
GPUs, e.g. RTX PRO 6000). The image ships system Python 3.12 with PyTorch 2.10,
flash-attn 2.8.3, transformer-engine 2.13.0, Apex, vLLM 0.19.1, and the project
source.

| Item | Value |
| --- | --- |
| Dockerfile | [`Dockerfile.cuda128`](Dockerfile.cuda128) |
| Base image | `nvidia/cuda:12.8.2-devel-ubuntu24.04` |
| Python deps | [`requirements-conda-cuda128.txt`](../requirements-conda-cuda128.txt) |

## Prerequisites

- Docker Engine (any reasonably recent version).
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  for `--gpus` support.
- Host driver compatible with CUDA 12.8 (R555+).
- ~40 GB free disk for the image build (PyTorch + CUDA + flash-attn + apex +
  vLLM + transformer-engine). The final `develop` image is ~16 GB.

## Build context

The Dockerfile copies the repo into the image with
`COPY --exclude=checkpoints . /workspace/paidf-anomalygen/`
(BuildKit `dockerfile:1.7-labs` syntax). The repo root
[`.dockerignore`](../.dockerignore) additionally strips out `datasets/`,
`results/`, `.git`, build caches, and virtualenvs so the build context stays
small. `checkpoints/` is intentionally left in the build context so the
air-gapped variant can bake it in; the base build excludes it via the COPY
flag and expects it mounted at runtime (see [Run](#run)).

## Build

The build context must be the **repo root** (the Dockerfile copies
`requirements-conda-cuda128.txt` from there).

The Dockerfile is multi-stage with two targets sharing the same `base` build:

- `develop` (writable runtime, root user, `ANOMALYGEN_PRODUCT_MODE` unset) —
  for code iteration with the agent.
- `product` (production code locked read-only, non-root `anomalygen` user,
  `ANOMALYGEN_PRODUCT_MODE=1`) — for runtime delivery.

Build the `develop` target:

```shell
# from the repo root
docker build --target develop -f docker/Dockerfile.cuda128 \
    -t paidf-anomalygen-dev:cuda12.8 .
```

Build the `product` target:

```shell
# from the repo root
docker build --target product -f docker/Dockerfile.cuda128 \
    -t paidf-anomalygen:cuda12.8 .
```

The [`anomalygen-release`](../.agents/skills/anomalygen-release/SKILL.md) skill
wraps both targets with permission validation and is the recommended path for
release builds.

The build takes ~60-120 minutes on a fast machine; the `flash-attn`,
`transformer-engine`, and `apex` steps each compile CUDA extensions. Build
parallelism is capped at `MAX_JOBS=4 NVCC_THREADS=2` to keep peak memory
manageable — raise it if you have RAM to spare.

> **Heads-up:** with the default `MAX_JOBS=4 NVCC_THREADS=2`, the `flash-attn`
> compile peaks at **~43 GiB** of host RAM. Make sure your build host has at
> least **~48 GiB free**, or lower `MAX_JOBS` to halve the peak.

## Run

```shell
LOCAL_PROJECT_DIR="$HOME/workspace/paidf-anomalygen"

docker run --gpus all -it --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "${LOCAL_PROJECT_DIR}/checkpoints:/workspace/paidf-anomalygen/checkpoints" \
    -v "${LOCAL_PROJECT_DIR}/datasets:/workspace/paidf-anomalygen/datasets" \
    -v "${LOCAL_PROJECT_DIR}/results:/workspace/paidf-anomalygen/results" \
    paidf-anomalygen:cuda12.8 bash
```

To iterate on code without rebuilding, build the `develop` target and
bind-mount the whole repo over the baked-in copy:

```shell
docker run --gpus all -it --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "${LOCAL_PROJECT_DIR}:/workspace/paidf-anomalygen" \
    paidf-anomalygen-dev:cuda12.8 bash
```

> Bind-mounting the whole repo over a `product` image cancels the read-only
> filesystem policy. Use the `develop` image for whole-repo bind-mounts.

### Quick training / generation example

Requires checkpoints and a dataset; see the project tutorial for downloads.

```shell
export IMAGINAIRE_OUTPUT_ROOT=./results

torchrun --nproc_per_node=1 --master_port=12341 -m scripts.anomaly_gen.ag_train \
    --config=cosmos_predict2/configs/base/ag_config.py \
    --ag_config=ag_configs/MeiweiPCB_NVDINOV2_2B_512.yaml \
    -- experiment=predict2_anomaly_gen_fsdp_2b

torchrun --nproc_per_node=1 -m scripts.anomaly_gen.synthetic_dataset_generation \
    --config=cosmos_predict2/configs/base/ag_config.py \
    --ag_checkpoint_dir=results/anomaly_gen/MeiweiPCB/MeiweiPCB_training_exp_FP32_lr0.02_bs=2_larger_guided_mask_maskconf=0.85_2B_512x512 \
    --step=200 \
    --input_data_path=ag_inference/example.jsonl \
    --output_image_path=results/MeiweiPCB/example_output \
    --seed=0 \
    -- experiment=predict2_anomaly_gen_fsdp_2b
```

## Push

```shell
DOCKER_REPO="YOUR_REGISTRY/YOUR_NAMESPACE"
DATE_TAG="cuda12.8_$(date +%Y%m%d)"

docker tag paidf-anomalygen:cuda12.8 \
    "${DOCKER_REPO}/paidf-anomalygen:${DATE_TAG}"
docker push "${DOCKER_REPO}/paidf-anomalygen:${DATE_TAG}"
```

## Air-gapped variant (checkpoints baked in)

[`Dockerfile.cuda128.airgapped`](Dockerfile.cuda128.airgapped) is a standalone
build that mirrors `Dockerfile.cuda128` but bakes `./checkpoints/` into the
image (the base build excludes them via `--exclude=checkpoints`; the air-gapped
build does not). The result is a self-contained image that runs with no network
or volume mounts. Use this when delivering to environments that can't pull
model weights at runtime.

**Image size will be ~75 GB+** (whatever `du -sh ./checkpoints` reports, plus
the ~10 GB framework base). Make sure your registry and host disk can handle it.

### Requirements

- **BuildKit is required** (same as the base build — the `--exclude` flag in
  `COPY` needs `dockerfile:1.7-labs` syntax). Enable with
  `export DOCKER_BUILDKIT=1` if it's not already the default.
- Populated `./checkpoints` directory at the repo root (sam2, NVDINOV2,
  facebook, google-t5/{t5-large,t5-11b}, nvidia/{C-RADIO-V3,Cosmos-Predict2-2B-Text2Image}).
  Refer to the project tutorial for download instructions.

### Build the air-gapped image

The Dockerfile exposes the same `develop` and `product` targets as the base
build. Pick the target that matches the runtime contract you need.

```shell
# from the repo root
export DOCKER_BUILDKIT=1

docker build --target product \
    -f docker/Dockerfile.cuda128.airgapped \
    -t paidf-anomalygen:cuda12.8-airgapped \
    .
```

This is a fresh build (not a thin overlay), so it takes the same ~60-120 minutes
as the base image plus checkpoint copy time.

### Run the air-gapped image

No volume mounts needed — checkpoints live inside the image:

```shell
docker run --gpus all -it --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    paidf-anomalygen:cuda12.8-airgapped bash
```

## Troubleshooting

- **`flash-attn` / `transformer-engine` / `apex` build OOMs** — lower
  `MAX_JOBS` in the Dockerfile (currently 4) or give Docker more memory.
- **Driver / CUDA mismatch** — the image needs a host driver that supports
  CUDA 12.8. Run `nvidia-smi` on the host to confirm.
- **`failed to find stage "base"`** — the multi-stage build needs the
  BuildKit/Buildx engine. Enable with `export DOCKER_BUILDKIT=1`, or use
  `docker buildx build`.
