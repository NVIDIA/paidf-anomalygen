#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# preflight_env_ckpt.sh — preflight the anomalygen runtime before running the pipeline. Checks, in order:
#   1. GPU / driver       (nvidia-smi lists the NVIDIA driver + visible GPUs)
#   2. Python             (>= 3.10)
#   3. CUDA / torch       (torch imports and can use CUDA)
#   4. Python deps        (the core packages the pipeline imports)
#   5. Hugging Face auth  (only a failure when checkpoints turn out to be missing — it is needed to
#                          fetch them, not to run once they are on disk)
#   6. Model checkpoints  (under ./checkpoints: both base sizes, KPI/ROI models, HF cache entries)
#
# Read-only; exits non-zero if any check fails (doubles as a CI/preflight gate) and prints a fix step
# for each failing group. Read WHICH group failed before reacting: it exits 1 on any failure, so
# chaining the downloader off that exit code would fire a checkpoint download for a torch/CUDA
# problem. Run scripts/download_checkpoints.sh only when the checkpoint group is what failed.
#
# Run from anywhere (with the uv env active — source .venv/bin/activate):
#   bash scripts/preflight_env_ckpt.sh
#
# Env overrides:
#   CKPT_DIR   checkpoint root (default: <repo>/checkpoints) — same override as download_checkpoints.sh
#   PYTHON     python interpreter to probe (default: python)
#
# No BASE_CHECKPOINT_NAME override — keep the list below in step with download_checkpoints.sh's.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
BASE_CHECKPOINT_NAMES=(Cosmos3-Nano Cosmos3-Edge)
PY="${PYTHON:-python}"

fail=0
ckpt_missing=0
ok()  { printf '[pass] %s\n' "$1"; }
bad() { printf '[fail] %s\n' "$1"; fail=$((fail + 1)); }
hdr() { printf '\n=== %s ===\n' "$1"; }

# --- 1. GPU / NVIDIA driver (nvidia-smi) --------------------------------------
hdr "GPU / NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  bad "nvidia-smi not on PATH — no NVIDIA driver / GPU visible"
else
  gpus="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
  if [[ -z "$gpus" ]]; then
    bad "nvidia-smi found but reported no GPUs"
  else
    drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    ndev="$(printf '%s\n' "$gpus" | grep -c .)"
    ok "NVIDIA driver ${drv}, ${ndev} GPU(s)"
    i=0
    while IFS= read -r g; do
      ok "GPU${i}: ${g}"
      i=$((i + 1))
    done <<< "$gpus"
  fi
fi

# --- 2. Python ----------------------------------------------------------------
hdr "Python (need >= 3.10)"
if ! command -v "$PY" >/dev/null 2>&1; then
  bad "no '$PY' on PATH — activate the env: source ${REPO_ROOT}/.venv/bin/activate"
else
  pyver="$("$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)"
  if "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    ok "Python ${pyver} ($(command -v "$PY"))"
  else
    bad "Python ${pyver:-unknown} — need >= 3.10"
  fi
fi

# --- 3 + 4. CUDA / torch + Python deps (one interpreter, so torch loads once) --
hdr "CUDA / torch + Python deps"
"$PY" - <<'PY'
import importlib, sys

try:
    import importlib.metadata as _md
    _pkgdist = _md.packages_distributions()  # top-level import name -> [dist names]
except Exception:  # noqa: BLE001
    _md, _pkgdist = None, {}

n_fail = 0
def line(passed, msg):
    print(("[pass] " if passed else "[fail] ") + msg)

def version(name, mod):
    v = getattr(mod, "__version__", None)
    if v:
        return str(v)
    if _md is not None:
        try:
            return _md.version(name)  # dist name usually matches the import name
        except Exception:  # noqa: BLE001
            pass
        for dist in _pkgdist.get(name, []):  # fall back to the import-name -> dist map
            try:
                return _md.version(dist)
            except Exception:  # noqa: BLE001
                pass
    return "unknown"

try:
    import torch
    avail = torch.cuda.is_available()
    ndev = torch.cuda.device_count() if avail else 0
    line(True, f"torch {torch.__version__}")
    line(avail, f"CUDA available={avail}, torch sees {ndev} device(s)")
    if not avail:
        n_fail += 1
except Exception as exc:  # noqa: BLE001
    line(False, f"torch import failed: {exc}")
    n_fail += 1

# Core packages the pipeline imports: framework + this package + transformers, and the
# compiled CUDA extensions built by scripts/env_setup.sh (attention / TE / apex / natten).
for name in (
    "cosmos_framework", "anomalygen", "transformers",
    "flash_attn", "flash_attn_3_nv", "transformer_engine", "apex", "natten",
):
    try:
        mod = importlib.import_module(name)
        line(True, f"{name} {version(name, mod)}")
    except Exception as exc:  # noqa: BLE001
        line(False, f"{name}: {exc}")
        n_fail += 1

sys.exit(min(n_fail, 250))  # exit code carries the failure count
PY
fail=$((fail + $?))

# --- 5. Hugging Face auth -----------------------------------------------------
# Checked offline (env var, else the token the CLI stores at login) so the preflight stays fast and
# usable air-gapped — `hf auth whoami` would need network. The *verdict* is deferred to section 6:
# auth is only needed to fetch checkpoints, so failing here when they are already on disk would make
# the documented `preflight || download_checkpoints.sh` chain start a download nothing was waiting for.
hdr "Hugging Face auth (needed only to download checkpoints)"
hf_auth_ok=0
hf_token_file="${HF_TOKEN_PATH:-${HF_HOME:-${HOME:-}/.cache/huggingface}/token}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  ok "HF_TOKEN is set"; hf_auth_ok=1
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  ok "HUGGING_FACE_HUB_TOKEN is set"; hf_auth_ok=1
elif [[ -s "${hf_token_file}" ]]; then
  ok "stored login token: ${hf_token_file}"; hf_auth_ok=1
else
  printf '[warn] %s\n' "not authenticated — harmless if every checkpoint below is already present;"
  printf '[warn] %s\n' "  a failure only if any must be downloaded (see the next section)."
fi

# --- 6. Model checkpoints -----------------------------------------------------
hdr "Model checkpoints (under ${CKPT_DIR})"
# Weight files are size-checked, not just presence-checked: an interrupted HF pull commonly writes the
# small metadata (config.json) and dies before the weights, and a presence-only gate green-lights that
# — then the model load fails much later instead of here. `min_mb` is a floor well under the real size,
# enough to catch a truncated or zero-byte file without pinning an exact version.
check_dir() {  # non-empty directory
  if [[ -d "$1" ]] && [[ -n "$(ls -A "$1" 2>/dev/null)" ]]; then ok "$2"; else bad "$2 — missing: $1"; ckpt_missing=$((ckpt_missing + 1)); fi
}
check_file() {  # check_file <path> <label> [min_mb]
  local min_mb="${3:-0}" size_mb
  if [[ ! -f "$1" ]]; then bad "$2 — missing: $1"; ckpt_missing=$((ckpt_missing + 1)); return; fi
  size_mb=$(( $(wc -c < "$1" 2>/dev/null || echo 0) / 1048576 ))
  if [[ "$size_mb" -lt "$min_mb" ]]; then
    bad "$2 — truncated: $1 is ${size_mb}MB, expected >= ${min_mb}MB (re-download)"
    ckpt_missing=$((ckpt_missing + 1))
  else
    ok "$2"
  fi
}
# A weights dir is only as good as its largest file — catches a config-only partial pull.
check_weights_dir() {  # check_weights_dir <dir> <label> <min_mb>
  local biggest
  if [[ ! -d "$1" ]]; then bad "$2 — missing: $1"; ckpt_missing=$((ckpt_missing + 1)); return; fi
  biggest=$(find "$1" -type f -printf '%s\n' 2>/dev/null | sort -rn | head -1)
  biggest=$(( ${biggest:-0} / 1048576 ))
  if [[ "$biggest" -lt "$3" ]]; then
    bad "$2 — no weight file >= $3MB under $1 (largest ${biggest}MB; partial download?)"
    ckpt_missing=$((ckpt_missing + 1))
  else
    ok "$2"
  fi
}

# HF hub-cache entries are resolved by repo id, not path, so a snapshot alone is not enough:
# refs/<rev> must map branch -> sha (a sha-pinned `hf download` never writes it), and <probe> must
# exist in that snapshot. Pass an empty <ref> for entries the framework resolves by pinned sha.
check_hf_cache() {  # check_hf_cache <repo id> <ref name, or "" to skip> <probe path in snapshot> <label>
  local dir="${CKPT_DIR}/hf/models--${1//\//--}" sha
  if [[ ! -d "$dir" ]]; then bad "$4 — missing: $dir"; ckpt_missing=$((ckpt_missing + 1)); return; fi
  if [[ -n "$2" ]]; then
    if [[ ! -s "${dir}/refs/$2" ]]; then
      bad "$4 — no refs/$2 under $dir; an offline resolve of revision '$2' cannot find the snapshot"
      ckpt_missing=$((ckpt_missing + 1)); return
    fi
    sha="$(cat "${dir}/refs/$2")"
  else
    sha="$(ls -1 "${dir}/snapshots" 2>/dev/null | head -1)"
  fi
  if [[ -z "$sha" ]] || [[ ! -e "${dir}/snapshots/${sha}/$3" ]]; then
    bad "$4 — $3 missing from snapshot ${sha:-<none>} under $dir (partial download?)"
    ckpt_missing=$((ckpt_missing + 1)); return
  fi
  ok "$4"
}

# Both sizes: model_size picks the DCP, so checking nano alone let an edge run die at model load.
for base_name in "${BASE_CHECKPOINT_NAMES[@]}"; do
  check_weights_dir "${CKPT_DIR}/${base_name}" "${base_name} (Base model)" 100
done
check_weights_dir "${CKPT_DIR}/facebook/dinov2-large" "DINOv2 (KPI model)" 100
check_file "${CKPT_DIR}/nvidia/C-RADIO-V3/model.safetensors" "C-RADIO v3 (KPI model)" 100
check_file "${CKPT_DIR}/wan2pt2/Wan2.2_VAE.pth" "Wan2.2 VAE (tokenizer)" 100
check_file "${CKPT_DIR}/facebook/sam2.1-hiera-large/sam2.1_hiera_large.pt" "SAM2.1 (ROI segmentation)" 100
check_file "${CKPT_DIR}/nvidia/Cosmos3-Nano/config.json" "Cosmos3-Nano HF config" 0
check_weights_dir "${CKPT_DIR}/nvidia/Cosmos3-Nano" "Cosmos3-Nano HF weights (Text2ROI, Caption)" 100

# --- HF hub cache (${CKPT_DIR}/hf) --------------------------------------------
# Guardrail: resolved at the sha the framework itself pins (GUARDRAIL1_CHECKPOINT.revision), so no ref.
check_hf_cache nvidia/Cosmos-Guardrail1 "" blocklist "Cosmos-Guardrail1 (generate.py guardrail)"
# Qwen3Guard: the framework calls from_pretrained() unpinned, i.e. by branch, so refs/main is required.
check_hf_cache Qwen/Qwen3Guard-Gen-0.6B main model.safetensors "Qwen3Guard-Gen-0.6B (text safety)"
# Caption tokenizer for model_size: nano. Weights are deliberately absent — probe a tokenizer file.
check_hf_cache Qwen/Qwen3-VL-8B-Instruct main tokenizer.json "Qwen3-VL tokenizer (nano caption)"
# Caption processor for model_size: edge, resolved from this cache while the model is being built.
check_hf_cache nvidia/Cosmos3-Edge main processor_config.json "Cosmos3-Edge processor (edge caption)"
# That entry is partial (no generation weights), which only resolves offline while no tree manifest
# claims more files than are on disk. hf >= 1.26 writes one; the downloader removes it, a hand-run
# `hf download` puts it back and silently re-breaks air-gapped edge.
if [[ -d "${CKPT_DIR}/hf/models--nvidia--Cosmos3-Edge/trees" ]]; then
  bad "Cosmos3-Edge processor — repo tree manifest present; an offline resolve will report an incomplete snapshot. Re-run scripts/download_checkpoints.sh (or: rm -rf ${CKPT_DIR}/hf/models--nvidia--Cosmos3-Edge/trees)"
  ckpt_missing=$((ckpt_missing + 1))
else
  ok "Cosmos3-Edge processor — no tree manifest (partial snapshot resolves offline)"
fi

# --- 7. checkpoint integrity --------------------------------------------------
# Pinning a revision fixes which commit was fetched, not which bytes are on disk now: checkpoints/ is
# a bind mount, so anything with write access can swap a weight between download and load. The two
# manifests were previously only compared inside download_checkpoints.sh, i.e. at download time and
# nowhere else. Checked here so the documented pre-run gate is also the integrity gate.
#
# Only when the weights are present — a missing checkpoint is already reported above, and reporting
# it twice as an integrity failure would point at the wrong fix step.
hdr "Checkpoint integrity"
if [[ "$ckpt_missing" -gt 0 ]]; then
  note_skip="${ckpt_missing} checkpoint(s) missing above — integrity check skipped until they are present"
  printf '[skip] %s\n' "$note_skip"
else
  for manifest_rel in assets/checkpoint_manifest.sha256 assets/checkpoint_manifest_converted.sha256; do
    manifest="${REPO_ROOT}/${manifest_rel}"
    if [[ ! -f "$manifest" ]]; then
      bad "${manifest_rel} — absent; run scripts/download_checkpoints.sh to record it"
      ckpt_missing=$((ckpt_missing + 1))
      continue
    fi
    # One implementation, shared with the opt-in load-time check in anomalygen/inference/inpaint.py,
    # so the two cannot drift. Failures are printed one per line by the helper.
    if out=$("$PY" -c '
import sys
from anomalygen.checkpoint.utils import verify_manifest
failures = verify_manifest(sys.argv[1], sys.argv[2])
for rel, why in failures:
    print(f"{rel}: {why}")
sys.exit(1 if failures else 0)
' "$manifest" "$CKPT_DIR" 2>&1); then
      ok "${manifest_rel} — all recorded weights match"
    else
      bad "${manifest_rel} — recorded weights do not match:"
      printf '        %s\n' "$out"
      ckpt_missing=$((ckpt_missing + 1))
    fi
  done
fi

# The section-5 auth verdict, resolvable only now that we know whether anything must be fetched.
hf_auth_missing=0
if [[ "$ckpt_missing" -gt 0 ]] && [[ "$hf_auth_ok" -eq 0 ]]; then
  bad "Hugging Face auth — required to download the ${ckpt_missing} missing checkpoint(s) above"
  hf_auth_missing=1
fi

# --- summary ------------------------------------------------------------------
if [[ "$fail" -eq 0 ]]; then
  printf '\nAll checks passed.\n'
  exit 0
fi

# Auth is subtracted alongside the checkpoints: it can only fail when they are missing, and the
# checkpoint fix step below already tells you to authenticate first. Counting it as an env failure
# would wrongly advise rebuilding the venv.
env_fail=$((fail - ckpt_missing - hf_auth_missing))
printf '\n%d check(s) failed. Fix steps:\n' "$fail"
if [[ "$env_fail" -gt 0 ]]; then
  printf '  - env / Python / deps (%d): rebuild the venv, then activate it —\n' "$env_fail"
  printf '      bash scripts/env_setup.sh\n'
  printf '      source .venv/bin/activate\n'
fi
if [[ "$ckpt_missing" -gt 0 ]]; then
  printf '  - missing checkpoints (%d): authenticate to Hugging Face, then fetch —\n' "$ckpt_missing"
  printf '      export HF_TOKEN=<token>       # or: hf auth login  (accept the model HF license)\n'
  printf '      bash scripts/download_checkpoints.sh\n'
fi
exit 1
