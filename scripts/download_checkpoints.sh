#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download the checkpoints anomalygen needs into ./checkpoints :
#   - base models -> checkpoints/Cosmos3-Nano, checkpoints/Cosmos3-Edge (DCP, framework converter)
#   - DINOv2      -> checkpoints/facebook/dinov2-large    (KPI correspondence backbone, NN/MNN)
#   - C-RADIO v3  -> checkpoints/nvidia/C-RADIO-V3        (KPI FID backbone)
#   - SAM2.1      -> checkpoints/facebook/sam2.1-hiera-large  (ROI-generation segmentation)
#   - Cosmos3-Nano-> checkpoints/nvidia/Cosmos3-Nano          (text2roi grounding VLM, default)
#   - Cosmos-Guardrail1 -> checkpoints/hf         (generate.py guardrail: blocklist + face-blur)
#   - Qwen3Guard-Gen-0.6B -> checkpoints/hf       (generate.py guardrail: text safety LLM)
#   - Qwen3-VL-8B-Instruct tokenizer -> checkpoints/hf   (nano caption tokenizer)
#   - Cosmos3-Edge processor      -> checkpoints/hf   (edge caption processor)
#
# Prerequisites: the uv environment is active (see README.md "Installation" / "Usage")
# and you are authenticated to Hugging Face (HF_TOKEN exported or `uvx hf auth login`),
# with both base model licenses accepted on their HF pages.
#
# Idempotent: existing outputs are skipped.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
BASE_CHECKPOINT_NAMES=(Cosmos3-Nano Cosmos3-Edge)
# The downloader is pinned too, not just what it downloads: `hf@latest` would resolve and run the
# newest PyPI release on every invocation, with HF_TOKEN in the environment and write access to the
# manifests below — an unverified executable fronting every pin in this file. 1.26.x matches the
# huggingface-hub pin in requirements.txt and the Edge trees/ removal below; bump them together.
HF_CLI="hf==${HF_CLI_VERSION:-1.26.0}"
DINOV2_REPO="${DINOV2_REPO:-facebook/dinov2-large}"
# Pinned like every other repo below: an unpinned "main" means a later run can fetch different
# bytes for the same command, and the digest recorded in assets/checkpoint_manifest.sha256 would
# then be the only thing that noticed. Bump the revision and the manifest together.
DINOV2_REV="${DINOV2_REV:-47b73eefe95e8d44ec3623f8890bd894b6ea2d6c}"
# C-RADIO v3 ViT-B FID backbone. The HF repo is nvidia/C-RADIOv3-B but the code resolves the
# checkpoint under checkpoints/nvidia/C-RADIO-V3 (matching upstream), so download just the weight
# file into that fixed directory.
CRADIO_REPO="${CRADIO_REPO:-nvidia/C-RADIOv3-B}"
CRADIO_FILE="${CRADIO_FILE:-model.safetensors}"
CRADIO_REV="${CRADIO_REV:-44653a0482cf460bb4f12595fc3cc3dfecc403d1}"
# Wan2.2 VAE the tokenizer encodes/decodes with. Shipped inside the Wan2.2-TI2V-5B repo.
WAN_VAE_REPO="${WAN_VAE_REPO:-Wan-AI/Wan2.2-TI2V-5B}"
WAN_VAE_FILE="${WAN_VAE_FILE:-Wan2.2_VAE.pth}"
WAN_VAE_REV="${WAN_VAE_REV:-921dbaf3f1674a56f47e83fb80a34bac8a8f203e}"
# SAM2.1 hiera-large weights for ROI-generation segmentation. The code resolves the checkpoint
# under checkpoints/facebook/sam2.1-hiera-large, so download just the weight file into that dir.
SAM2_REPO="${SAM2_REPO:-facebook/sam2.1-hiera-large}"
SAM2_FILE="${SAM2_FILE:-sam2.1_hiera_large.pt}"
SAM2_REV="${SAM2_REV:-665f8e2ad61cf5f53d65644ff27c8ee525124610}"
# text2roi grounding VLMs (HF transformers format), resolved from checkpoints/<org>/<name> by
# anomalygen.auto_mask_placement.text2roi.Text2BoxDetector. Cosmos3-Nano is the default.
# For Cosmos3-Nano the reasoner shares the transformer/ + vision_encoder/ weights, so the
# generation-only vae/sound_tokenizer/scheduler dirs are skipped.
COSMOS_VLM_REPO="${COSMOS_VLM_REPO:-nvidia/Cosmos3-Nano}"
# Pinned like every other repo here, and doubly so: the Hub manifest below records this model's
# shards, so an unpinned fetch would move the snapshot out from under the recorded digests.
COSMOS_VLM_REV="${COSMOS_VLM_REV:-411f42a8fdfb8c5b2583cb8786e0938f49796eaa}"
# Content-safety guardrail models used by anomalygen/scripts/texture/generate.py.
# The framework resolves these from the HF cache at runtime, so we only warm the cache below rather
# than writing into checkpoints/.
GUARDRAIL_REPO="${GUARDRAIL_REPO:-nvidia/Cosmos-Guardrail1}"
QWEN_GUARD_REPO="${QWEN_GUARD_REPO:-Qwen/Qwen3Guard-Gen-0.6B}"
# The framework loads Qwen3Guard unpinned via from_pretrained() at runtime, so pinning here makes only
# the *download* deterministic; full runtime reproducibility also needs an upstream pin. Pinned to the
# repo HEAD at authoring time; override QWEN_GUARD_REV to bump.
QWEN_GUARD_REV="${QWEN_GUARD_REV:-fada3b2f655b89601929198343c94cd2f64d93cc}"
# Caption tokenizer for training. NANO_MODEL_CONFIG wires vlm_config.tokenizer to
# create_qwen2_tokenizer_with_download(pretrained_model_name="Qwen/Qwen3-VL-8B-Instruct",
# config_variant="hf"), which resolves straight from the HF cache at model-construction time.
# Without it a run with no Hub access dies before iteration 1 (TypeError: expected str, bytes or
# os.PathLike object, not NoneType — transformers' cached_file returns None when offline), so the
# air-gapped images need it baked in. Only *.json/*.txt: the 8B weights (~17 GB) are never loaded
# here, the model itself comes from checkpoints/Cosmos3-Nano.
QWEN_VLM_REPO="${QWEN_VLM_REPO:-Qwen/Qwen3-VL-8B-Instruct}"
# Caption processor for model_size: edge — the Edge counterpart of the Qwen tokenizer above.
# EDGE_MODEL_CONFIG resolves it through CheckpointDirHf at model-build time, so without this
# prefetch every rank pulls 8.6 GB, and an air-gapped run fails inside load_for_inference.
EDGE_VLM_REPO="${EDGE_VLM_REPO:-nvidia/Cosmos3-Edge}"

DINOV2_OUT="${CKPT_DIR}/${DINOV2_REPO}"
CRADIO_OUT="${CKPT_DIR}/nvidia/C-RADIO-V3"
WAN_VAE_OUT="${CKPT_DIR}/wan2pt2/${WAN_VAE_FILE}"
SAM2_OUT="${CKPT_DIR}/facebook/sam2.1-hiera-large"
COSMOS_VLM_OUT="${CKPT_DIR}/${COSMOS_VLM_REPO}"

mkdir -p "${CKPT_DIR}"

# ---- base models (download + convert to DCP) ---------------------------------
if ! python -c "import cosmos_framework" 2>/dev/null; then
  echo "ERROR: cosmos_framework not importable. Activate the venv first:" >&2
  echo "  source ${REPO_ROOT}/.venv/bin/activate && export LD_LIBRARY_PATH=" >&2
  exit 1
fi

# The converter pulls each repo from the Hub before writing the DCP, deliberately BEFORE the
# HF_HUB_CACHE export below, so those one-time source snapshots land in the user's default cache
# rather than bloating checkpoints/hf (the runtime cache the air-gapped images bake in). The DCP
# holds tensors only — no tokenizer — and this loop is skipped when it already exists, so the
# processor prefetch at the end cannot rely on this pull.
#
# Both sizes are required — a checkout that can only run one is half-installed, and a
# `model_size: edge` recipe would otherwise fail deep inside load_for_inference instead of here.
# A failure is collected rather than fatal so one unaccepted licence does not also take out the
# ungated DINOv2 / C-RADIO / SAM2 / guardrail pulls below and read as eight broken downloads.
failed_sizes=()
for base_name in "${BASE_CHECKPOINT_NAMES[@]}"; do
  base_out="${CKPT_DIR}/${base_name}"
  if [[ -d "${base_out}" ]] && [[ -n "$(ls -A "${base_out}" 2>/dev/null)" ]]; then
    echo "[skip] base checkpoint already present: ${base_out}"
  else
    echo "[download+convert] ${base_name} -> ${base_out}"
    if ! python -m cosmos_framework.scripts.convert_model_to_dcp \
      -o "${base_out}" \
      --checkpoint-path "${base_name}"; then
      echo "[FAIL] ${base_name} — continuing with the remaining downloads" >&2
      failed_sizes+=("${base_name}")
      rm -rf "${base_out}"   # a half-written DCP would be "present" on the next run
    fi
  fi
done

# ---- DINOv2 (KPI backbone) ---------------------------------------------------
if [[ -d "${DINOV2_OUT}" ]] && [[ -n "$(ls -A "${DINOV2_OUT}" 2>/dev/null)" ]]; then
  echo "[skip] DINOv2 already present: ${DINOV2_OUT}"
else
  echo "[download] ${DINOV2_REPO} -> ${DINOV2_OUT}"
  uvx "${HF_CLI}" download --repo-type model "${DINOV2_REPO}" --revision "${DINOV2_REV}" --local-dir "${DINOV2_OUT}"
fi

# ---- C-RADIO v3 (KPI FID backbone) -------------------------------------------
if [[ -f "${CRADIO_OUT}/${CRADIO_FILE}" ]]; then
  echo "[skip] C-RADIO v3 already present: ${CRADIO_OUT}/${CRADIO_FILE}"
else
  echo "[download] ${CRADIO_REPO}:${CRADIO_FILE} -> ${CRADIO_OUT}"
  mkdir -p "${CRADIO_OUT}"
  uvx "${HF_CLI}" download --repo-type model "${CRADIO_REPO}" "${CRADIO_FILE}" --revision "${CRADIO_REV}" \
    --local-dir "${CRADIO_OUT}"
fi

# ---- Wan2.2 VAE (tokenizer) --------------------------------------------------
if [[ -f "${WAN_VAE_OUT}" ]]; then
  echo "[skip] Wan2.2 VAE already present: ${WAN_VAE_OUT}"
else
  echo "[download] ${WAN_VAE_REPO}:${WAN_VAE_FILE} -> ${WAN_VAE_OUT}"
  mkdir -p "$(dirname "${WAN_VAE_OUT}")"
  uvx "${HF_CLI}" download --repo-type model "${WAN_VAE_REPO}" "${WAN_VAE_FILE}" --revision "${WAN_VAE_REV}" \
    --local-dir "$(dirname "${WAN_VAE_OUT}")"
fi

# ---- SAM2.1 hiera-large (ROI-generation segmentation) ------------------------
if [[ -f "${SAM2_OUT}/${SAM2_FILE}" ]]; then
  echo "[skip] SAM2.1 already present: ${SAM2_OUT}/${SAM2_FILE}"
else
  echo "[download] ${SAM2_REPO}:${SAM2_FILE} -> ${SAM2_OUT}"
  mkdir -p "${SAM2_OUT}"
  uvx "${HF_CLI}" download --repo-type model "${SAM2_REPO}" "${SAM2_FILE}" --revision "${SAM2_REV}" \
    --local-dir "${SAM2_OUT}"
fi

# ---- Cosmos3-Nano (reasoner only) ---
if [[ -f "${COSMOS_VLM_OUT}/config.json" ]]; then
  echo "[skip] Cosmos3-Nano VLM already present: ${COSMOS_VLM_OUT}"
else
  echo "[download] ${COSMOS_VLM_REPO}@${COSMOS_VLM_REV} -> ${COSMOS_VLM_OUT} (~30GB; excludes generation-only vae/sound_tokenizer/scheduler)"
  uvx "${HF_CLI}" download --repo-type model "${COSMOS_VLM_REPO}" --revision "${COSMOS_VLM_REV}" \
    --local-dir "${COSMOS_VLM_OUT}" \
    --exclude "sound_tokenizer/*" --exclude "vae/*" --exclude "scheduler/*" --exclude "assets/*" --exclude "images/*"
fi

# ---- Cosmos-Guardrail1 + Qwen3Guard (content-safety guardrail for generate.py) ----
# These are resolved from the HF hub cache at runtime (the classes give no local-dir option), so we
# point that cache under ${CKPT_DIR}/hf to keep the checkout self-contained.
export HF_HUB_CACHE="${CKPT_DIR}/hf"

# Skip guard for the four hub-cache entries. The six --local-dir entries above test an output path;
# these land as content-addressed blobs, so the equivalent test is "does this revision's snapshot
# hold a file the loader actually reads" — the same probe preflight_env_ckpt.sh checks for.
#
# Guards the download ONLY. The refs/main writes and the trees/ removal below stay unconditional:
# they are what make an offline resolve work, they are what preflight asserts, and skipping them on
# a warm cache would silently re-break air-gapped runs. They cost a 40-byte write and an rm.
# The revision may be a sha (snapshots/<sha>) or a branch name, which hf records as refs/<branch>
# -> sha rather than a directory. Resolve the ref when the direct path is absent, like preflight.
# Sets HF_SNAPSHOT_DIR to the snapshot it matched, so the skip message names the directory that was
# actually checked rather than the shared cache root — all four entries live under the same root, so
# printing that alone says nothing about which one was found.
hf_snapshot_has() {  # hf_snapshot_has <repo id> <revision> <probe path in snapshot>
  local dir="${HF_HUB_CACHE}/models--${1//\//--}" sha="$2"
  HF_SNAPSHOT_DIR=""
  if [[ ! -e "${dir}/snapshots/${sha}/${3}" ]]; then
    [[ -s "${dir}/refs/${2}" ]] || return 1
    sha="$(cat "${dir}/refs/${2}")"
    [[ -e "${dir}/snapshots/${sha}/${3}" ]] || return 1
  fi
  HF_SNAPSHOT_DIR="${dir}/snapshots/${sha}"
}
# Read the framework's pinned revision (single source of truth — no hardcoded hash, no drift), then
# download with hf like every other entry above.
GUARDRAIL_REV="${GUARDRAIL_REV:-$(python -c "from cosmos_framework.auxiliary.guardrail.common.core import GUARDRAIL1_CHECKPOINT as c; print(c.revision)")}"
if hf_snapshot_has "${GUARDRAIL_REPO}" "${GUARDRAIL_REV}" blocklist; then
  echo "[skip] Cosmos-Guardrail1 already present: ${HF_SNAPSHOT_DIR}"
else
  echo "[download] ${GUARDRAIL_REPO}@${GUARDRAIL_REV} -> ${HF_HUB_CACHE} (blocklist word-lists + RetinaFace face-blur weights)"
  uvx "${HF_CLI}" download --repo-type model "${GUARDRAIL_REPO}" --revision "${GUARDRAIL_REV}"
fi

if hf_snapshot_has "${QWEN_GUARD_REPO}" "${QWEN_GUARD_REV}" model.safetensors; then
  echo "[skip] Qwen3Guard already present: ${HF_SNAPSHOT_DIR}"
else
  echo "[download] ${QWEN_GUARD_REPO}@${QWEN_GUARD_REV} -> ${HF_HUB_CACHE} (text-safety LLM)"
  uvx "${HF_CLI}" download --repo-type model "${QWEN_GUARD_REPO}" --revision "${QWEN_GUARD_REV}"
fi
# Qwen3Guard.__init__ calls from_pretrained() with no revision, so an offline resolve goes through
# refs/main — which `hf download --revision <sha>` does not write. Without this the guardrail is
# unusable air-gapped despite the snapshot being present. Online runs rewrite it anyway.
QWEN_GUARD_CACHE_DIR="${HF_HUB_CACHE}/models--${QWEN_GUARD_REPO//\//--}"
mkdir -p "${QWEN_GUARD_CACHE_DIR}/refs"
printf '%s' "${QWEN_GUARD_REV}" > "${QWEN_GUARD_CACHE_DIR}/refs/main"

# ---- Qwen3-VL-8B-Instruct tokenizer (training caption tokenizer) ----
# Revision comes from the framework's own registry rather than a hardcoded hash, so this cannot
# drift from the pin create_qwen2_tokenizer_with_download resolves. _CHECKPOINTS is private; if an
# upstream refactor moves it this fails loudly here instead of silently fetching a different commit.
QWEN_VLM_REV="${QWEN_VLM_REV:-$(python -c "
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.utils.checkpoint_db import _CHECKPOINTS
register_checkpoints()
print(next(c.hf.revision for c in _CHECKPOINTS.values()
           if getattr(c, 'hf', None) and c.hf.repository == '${QWEN_VLM_REPO}'))")}"
if hf_snapshot_has "${QWEN_VLM_REPO}" "${QWEN_VLM_REV}" tokenizer.json; then
  echo "[skip] Qwen3-VL tokenizer already present: ${HF_SNAPSHOT_DIR}"
else
  echo "[download] ${QWEN_VLM_REPO}@${QWEN_VLM_REV} -> ${HF_HUB_CACHE} (training caption tokenizer, *.json/*.txt only)"
  uvx "${HF_CLI}" download --repo-type model "${QWEN_VLM_REPO}" --revision "${QWEN_VLM_REV}" \
    --include "*.json" --include "*.txt"
fi
# create_qwen2_tokenizer_with_download calls Qwen2Tokenizer.from_pretrained(<repo id>) with no
# revision, so an offline resolve maps "main" -> sha through refs/main. `hf download --revision
# <sha>` writes the snapshot but never that ref, so without this line the cache is unusable offline:
# every file resolves to None and tokenization_qwen2.py does open(None). Pin + ref, not one or the
# other. 40 bytes, no trailing newline — matches what huggingface_hub itself writes.
QWEN_VLM_CACHE_DIR="${HF_HUB_CACHE}/models--${QWEN_VLM_REPO//\//--}"
mkdir -p "${QWEN_VLM_CACHE_DIR}/refs"
printf '%s' "${QWEN_VLM_REV}" > "${QWEN_VLM_CACHE_DIR}/refs/main"

# ---- Cosmos3-Edge processor (model_size: edge caption processor) ----
# Repo id and revision are read from the model config, not hardcoded, so this cannot drift from what
# build_processor_lazy asks for; an upstream move fails loudly here instead of warming the wrong entry.
EDGE_VLM_REV="${EDGE_VLM_REV:-$(python -c "
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
tok = EDGE_MODEL_CONFIG['vlm_config']['tokenizer']
assert tok['repository'] == '${EDGE_VLM_REPO}', tok['repository']
print(tok['revision'])")}"
# ~28 MB of an 8.6 GB repo: the processor reads only the config / tokenizer / chat-template files.
# transformer/, vae/ and vision_encoder/ hold the generation weights, which are never read here —
# the model loads those from the DCP in checkpoints/Cosmos3-Edge. No refs/main needed, unlike the
# Qwen block above: the revision IS a branch name, so hf records the mapping itself.
#
# CAVEAT: this warms the cache but cannot cap it. The tokenizer node passes no include patterns, so
# on a networked node the first edge run still pulls the remaining ~8.5 GB. What the restriction
# buys is a small air-gapped image and a working offline resolve.
if hf_snapshot_has "${EDGE_VLM_REPO}" "${EDGE_VLM_REV}" processor_config.json; then
  echo "[skip] Cosmos3-Edge processor already present: ${HF_SNAPSHOT_DIR}"
else
  echo "[download] ${EDGE_VLM_REPO}@${EDGE_VLM_REV} -> ${HF_HUB_CACHE} (edge caption processor, no generation weights)"
  uvx "${HF_CLI}" download --repo-type model "${EDGE_VLM_REPO}" --revision "${EDGE_VLM_REV}" \
    --exclude "transformer/*" --exclude "vae/*" --exclude "vision_encoder/*"
fi
# Drop hf's repo-tree manifest. NOT cosmetic: hf 1.26 (the pinned HF_CLI) writes trees/<sha>.json, and the
# framework's own pinned hf 1.25.1 then compares it against the snapshot, sees the weight dirs we
# skipped, and dies with "Incomplete snapshot available". Absent, it returns the snapshot as-is.
# Verified both ways offline. Only the offline path is affected; preflight asserts it stays gone.
EDGE_VLM_CACHE_DIR="${HF_HUB_CACHE}/models--${EDGE_VLM_REPO//\//--}"
rm -rf "${EDGE_VLM_CACHE_DIR}/trees"

echo ""
echo "Done. Checkpoints in ${CKPT_DIR}:"
for base_name in "${BASE_CHECKPOINT_NAMES[@]}"; do
  echo "  base       : ${CKPT_DIR}/${base_name}"
done
echo "  dinov2     : ${DINOV2_OUT}"
echo "  cradio     : ${CRADIO_OUT}/${CRADIO_FILE}"
echo "  wan vae    : ${WAN_VAE_OUT}"
echo "  sam2       : ${SAM2_OUT}/${SAM2_FILE}"
echo "  cosmos vlm : ${COSMOS_VLM_OUT}"
echo "  guardrail  : ${GUARDRAIL_REPO} (${HF_HUB_CACHE})"
echo "  qwen guard : ${QWEN_GUARD_REPO} (${HF_HUB_CACHE})"
echo "  qwen vlm tk: ${QWEN_VLM_REPO} (${HF_HUB_CACHE})"
echo "  edge proc  : ${EDGE_VLM_REPO} (${HF_HUB_CACHE})"
echo ""
echo "These paths are hardcoded in the anomalygen code; no env vars to set."

# --- checkpoint integrity -----------------------------------------------------
# Pinning a revision fixes which commit is fetched, not which bytes land on disk, and says nothing
# about the tree afterwards: checkpoints/ is bind-mounted into the container, so anything with write
# access to the host path can swap a weight file between download and load.
#
# So: record a digest the first time, and check against it on every later run. Skipped when a
# download failed, because a partial tree would record partial digests as the reference.
#
# The weights that carry model behaviour, not the whole tree: these are ~90% of the bytes, and the
# rest is config and tokenizer JSON a tamper would have to change these to exploit. All seven Nano
# shards, though — a checkpoint loads every one, so recording a single shard leaves six unverified.
#
# `sha256sum -c` does the work, so there is no manifest format of our own and a reviewer can re-run
# the same command by hand. Paths are relative to CKPT_DIR so the manifest is portable.
#
# Two manifests, because the two groups of files have different truth conditions and therefore
# different remedies when they mismatch:
#
#   hub       Downloaded, and every repo is revision-pinned above. HuggingFace publishes the same
#             sha256 as each file's LFS oid, so these digests are upstream's bytes, independently
#             checkable, and only change if a pin changed. A mismatch here is an integrity problem.
#   converted Produced locally by convert_model_to_dcp, so the bytes depend on the torch version,
#             the cosmos-framework pin, and the shard layout of the machine that ran the conversion.
#             A bump to any of those changes them while the upstream model is identical, so a
#             mismatch here is usually a stale record rather than tampering.
#
# Re-recording is "delete the manifest and re-run", so keeping them apart is what makes that safe: a
# torch bump discards the converted digests and leaves the verifiable Hub ones in place. One file
# would make the only remedy throw away both groups, turning the gate into a habit of deleting it.
#
# The Hub manifest is committed, so a fresh clone verifies rather than records. Its digests are
# HuggingFace's published LFS oids, re-derivable from the Hub without trusting this checkout.
HUB_MANIFEST="${REPO_ROOT}/assets/checkpoint_manifest.sha256"
HUB_WEIGHTS=(
  "nvidia/Cosmos3-Nano/transformer/diffusion_pytorch_model-00001-of-00007.safetensors"
  "nvidia/Cosmos3-Nano/transformer/diffusion_pytorch_model-00002-of-00007.safetensors"
  "nvidia/Cosmos3-Nano/transformer/diffusion_pytorch_model-00003-of-00007.safetensors"
  "nvidia/Cosmos3-Nano/transformer/diffusion_pytorch_model-00004-of-00007.safetensors"
  "nvidia/Cosmos3-Nano/transformer/diffusion_pytorch_model-00005-of-00007.safetensors"
  "nvidia/Cosmos3-Nano/transformer/diffusion_pytorch_model-00006-of-00007.safetensors"
  "nvidia/Cosmos3-Nano/transformer/diffusion_pytorch_model-00007-of-00007.safetensors"
  "facebook/dinov2-large/pytorch_model.bin"
  "nvidia/C-RADIO-V3/model.safetensors"
  "wan2pt2/Wan2.2_VAE.pth"
  "facebook/sam2.1-hiera-large/sam2.1_hiera_large.pt"
)
# Edge is here and not in HUB_WEIGHTS because the base models are never downloaded in Hub weight
# format — convert_model_to_dcp writes a DCP, so these shards are their only local copy. Nano is
# in both lists because text2roi and the captioner load a second, Hub-format copy of it.
# All shards, not one: a DCP loads every shard, and hashing the 8 takes ~30 s.
CONVERTED_MANIFEST="${REPO_ROOT}/assets/checkpoint_manifest_converted.sha256"
CONVERTED_WEIGHTS=(
  "Cosmos3-Nano/model/__0_0.distcp"
  "Cosmos3-Nano/model/__0_1.distcp"
  "Cosmos3-Nano/model/__0_2.distcp"
  "Cosmos3-Nano/model/__0_3.distcp"
  "Cosmos3-Nano/model/__0_4.distcp"
  "Cosmos3-Nano/model/__0_5.distcp"
  "Cosmos3-Edge/model/__0_0.distcp"
  "Cosmos3-Edge/model/__0_1.distcp"
)

verify_or_record() {  # verify_or_record <manifest path> <label> <mismatch guidance> <files...>
  local manifest="$1" label="$2" guidance="$3"
  shift 3
  local rel="${manifest#"${REPO_ROOT}/"}"
  if [[ -f "${manifest}" ]]; then
    echo "Verifying ${label} weights against ${rel} ..."
    # `sha256sum -c` only checks the lines it is given, so a weight added to the list above but
    # absent from an existing manifest would be silently unverified — the check would still pass
    # and report OK. Fail on under-coverage instead, naming the remedy, so extending the list is
    # never a no-op.
    local uncovered=()
    local want
    for want in "$@"; do
      grep -qF -- "  ${want}" "${manifest}" || uncovered+=("${want}")
    done
    if [[ ${#uncovered[@]} -gt 0 ]]; then
      echo "" >&2
      echo "ERROR: ${rel} does not cover ${#uncovered[@]} requested ${label} weight(s):" >&2
      printf '  %s\n' "${uncovered[@]}" >&2
      echo "The list in this script grew without the manifest growing with it. Re-record with:" >&2
      echo "  rm ${rel} && bash scripts/download_checkpoints.sh" >&2
      echo "then confirm the new digests against the Hub before committing them." >&2
      exit 1
    fi
    if ( cd "${CKPT_DIR}" && sha256sum -c --quiet "${manifest}" ); then
      echo "${label} weights OK ($(grep -c . "${manifest}") files)."
    else
      echo "" >&2
      echo "ERROR: ${label} weights do not match ${rel}." >&2
      echo "Each line above is a weight whose bytes differ from what was recorded, or that is gone." >&2
      echo "${guidance}" >&2
      exit 1
    fi
  else
    mkdir -p "$(dirname "${manifest}")"
    # Write-then-rename: a redirect truncates first, so a kill between truncate and write would
    # leave a half-written manifest behind. rename(2) is atomic, so the file is either the old
    # content or the complete new one.
    ( cd "${CKPT_DIR}" && sha256sum "$@" ) > "${manifest}.tmp"
    mv "${manifest}.tmp" "${manifest}"
    echo "Recorded ${label} weights -> ${rel} ($(grep -c . "${manifest}") files)."
  fi
}

if [[ ${#failed_sizes[@]} -eq 0 ]]; then
  echo ""
  verify_or_record "${HUB_MANIFEST}" "Hub-sourced" \
    "These are revision-pinned downloads whose digests match HuggingFace's published LFS oids, so
they should never change unless a pin above did. Treat a mismatch as an integrity problem, not a
stale manifest — re-record only after confirming the new bytes are the ones upstream publishes." \
    "${HUB_WEIGHTS[@]}"
  verify_or_record "${CONVERTED_MANIFEST}" "Locally-converted" \
    "These are produced by convert_model_to_dcp on this machine, so a torch or cosmos-framework bump
legitimately changes them. Re-record by deleting ${CONVERTED_MANIFEST#"${REPO_ROOT}/"} and re-running;
the Hub-sourced manifest is separate and stays in place." \
    "${CONVERTED_WEIGHTS[@]}"
fi

# Reported last so it is the final thing on screen, and after the summary so the user can see which
# downloads DID succeed — the point of not aborting on the first gated 403.
if [[ ${#failed_sizes[@]} -gt 0 ]]; then
  echo "" >&2
  echo "ERROR: base model size(s) not downloaded: ${failed_sizes[*]}" >&2
  echo "Everything else above succeeded. This is almost always an unaccepted model licence:" >&2
  for base_name in "${failed_sizes[@]}"; do
    echo "  https://huggingface.co/nvidia/${base_name}   <- open, accept, then re-run this script" >&2
  done
  echo "Both sizes are required: a recipe's model_size selects the base checkpoint, so a" >&2
  echo "missing size fails at model load instead of here. Re-runs skip what is already present." >&2
  exit 1
fi
