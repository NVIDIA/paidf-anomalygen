# AnomalyGen Docker Image

Docker build for **PAIDF AnomalyGen**. The default image targets **CUDA 12.8**
(Blackwell-class GPUs, e.g. RTX PRO 6000); an **arm64 / CUDA 13** variant for
the GB10/GB200/GB300 line is also provided.

| Variant | Dockerfile | Base image | Python deps |
| --- | --- | --- | --- |
| CUDA 12.8 (x86_64) | [`Dockerfile`](Dockerfile) | `nvidia/cuda:12.8.2-devel-ubuntu24.04` | [`requirements-conda-cuda128.txt`](../requirements-conda-cuda128.txt) |
| CUDA 13 (arm64) | [`Dockerfile.arm.cuda130`](Dockerfile.arm.cuda130) | `nvidia/cuda:13.0.3-devel-ubuntu24.04` | [`requirements-conda-cuda130.txt`](../requirements-conda-cuda130.txt) |

Both Dockerfiles also expose **air-gapped** targets (`airgapped-product` /
`airgapped-develop`) that bake `checkpoints/` into the image — see
[Air-gapped variant](#air-gapped-variant-checkpoints-baked-in).

## Prerequisites

- Docker Engine (any reasonably recent version).
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  for `--gpus` support.
- Host driver compatible with CUDA 12.8 (R555+).
- ~40 GB free disk for the image build (PyTorch + CUDA + flash-attn + apex +
  vLLM + transformer-engine). The final `develop` image is ~16 GB.

## Targets

Both Dockerfiles are multi-stage, sharing the same `base` build, and each
exposes the same four targets:

- **`develop`** — writable runtime, non-root `anomalygen` user,
  `ANOMALYGEN_PRODUCT_MODE` unset. For code iteration with the agent. **This is
  the default target** (last stage), so a plain `docker build` (no `--target`)
  builds it.
- **`product`** — production code locked read-only, non-root `anomalygen` user,
  `ANOMALYGEN_PRODUCT_MODE=1`. For runtime delivery. Build with `--target product`.
- **`airgapped-develop`** / **`airgapped-product`** — same as `develop` /
  `product`, plus `checkpoints/` baked into the image so the container runs with
  no network and no volume mounts. See [Air-gapped variant](#air-gapped-variant-checkpoints-baked-in).

## Build context

The Dockerfile copies the repo with `COPY . /workspace/paidf-anomalygen`; the
repo-root [`.dockerignore`](../.dockerignore) keeps the context small by
excluding `checkpoints/`, `datasets/`, `results/`, `ag_inference/`, `.git`,
build caches, and virtualenvs. Mount those dirs at runtime (see [Run](#run)).
The air-gapped variant instead bakes `checkpoints/` into the image.

## Build

The build context must be the **repo root** (the Dockerfile copies
`requirements-conda-cuda128.txt` from there).

```shell
# from the repo root — develop (default)
docker build -f docker/Dockerfile -t paidf-anomalygen-dev:cuda12.8 .

# product
docker build --target product -f docker/Dockerfile -t paidf-anomalygen:cuda12.8 .
```

arm64 / CUDA 13 (build on an aarch64 host):

```shell
docker build -f docker/Dockerfile.arm.cuda130 -t paidf-anomalygen-dev:cuda13-arm .
docker build --target product -f docker/Dockerfile.arm.cuda130 -t paidf-anomalygen:cuda13-arm .
```

The [`anomalygen-release`](../skills/anomalygen-release/SKILL.md) skill
wraps both targets with permission validation and is the recommended path for
release builds.

The build takes ~60-120 minutes on a fast machine; the `flash-attn`,
`transformer-engine`, and `apex` steps each compile CUDA extensions.

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

docker tag paidf-anomalygen:cuda12.8 "${DOCKER_REPO}/paidf-anomalygen:${DATE_TAG}"
docker push "${DOCKER_REPO}/paidf-anomalygen:${DATE_TAG}"
```

## Air-gapped variant (checkpoints baked in)

The `airgapped-product` / `airgapped-develop` targets bake `./checkpoints/` into
the image so the result runs with no network or volume mounts. Use this when
delivering to environments that can't pull model weights at runtime. Both
[`Dockerfile`](Dockerfile) and [`Dockerfile.arm.cuda130`](Dockerfile.arm.cuda130)
support them — swap the `-f` argument below for the arm64 / CUDA-13 build.

**Image size will be ~75 GB+** (whatever `du -sh ./checkpoints` reports, plus
the ~10 GB framework base). Make sure your registry and host disk can handle it.

### Requirements

- **buildx is required** — the `--build-context` flag is a buildx feature.
  Modern Docker aliases `docker build` to buildx; use `docker buildx build`
  explicitly to be safe.
- Populated `./checkpoints` directory at the repo root (sam2, NVDINOV2,
  facebook, google-t5/{t5-large,t5-11b}, nvidia/{C-RADIO-V3,Cosmos-Predict2-2B-Text2Image,Cosmos-Predict2-14B-Text2Image}, Qwen).
  Refer to the project tutorial for download instructions.

### Build the air-gapped image

```shell
# from the repo root
docker buildx build --target airgapped-product \
    --build-context ckpts=./checkpoints \
    -f docker/Dockerfile \
    -t paidf-anomalygen:cuda12.8-airgapped .
```

### Run the air-gapped image

No volume mounts needed — checkpoints live inside the image:

```shell
docker run --gpus all -it --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    paidf-anomalygen:cuda12.8-airgapped bash
```

## Troubleshooting

- **`flash-attn` / `transformer-engine` / `apex` build OOMs** — lower
  `--build-arg MAX_JOBS=...`, or supply `GITLAB_PYPI_INDEX_URL` to skip the
  compiles entirely.
- **Driver / CUDA mismatch** — the image needs a host driver supporting its
  CUDA version. Run `nvidia-smi` on the host to confirm.
- **`failed to find stage "base"`** — the multi-stage build needs BuildKit.
  Enable with `export DOCKER_BUILDKIT=1`, or use `docker buildx build`.
