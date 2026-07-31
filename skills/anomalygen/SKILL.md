---
name: anomalygen
description: >-
  PAIDF AnomalyGen pipeline — fine-tune, generate synthetic anomaly images (SDG),
  evaluate quality (nn_score), and per-sample search. Modes: full (train +
  generate), finetune_only, inference_only (from checkpoint). Invoke for any
  AnomalyGen / SDG / fine-tune / synthetic-anomaly task — even if the user only
  mentions one phase.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA
  tags: [anomaly-generation, synthetic-data, fine-tuning, diffusion, paidf]
---

# PAIDF AnomalyGen

## TL;DR — Mode Selection

Lock the mode first, then collect the required params for that mode:

| User says | `mode` | Required | Phases that run |
|---|---|---|---|
| supplies a checkpoint dir + step | `inference_only` | `checkpoint_dir`, `step`, `num_SDG`, `dataset_dir` | 0, 2-7 |
| has a dataset, wants to **train only** | `finetune_only` | `dataset_dir`, `defect_spec` | 0-1 |
| has a dataset, wants **train + generate** | `full` | `dataset_dir`, `defect_spec`, `num_SDG` | 0-7 |

In `mode=full`, `CKPT` and `STEP` are auto-derived after Phase 1 — do NOT
ask the user for them. In `mode=inference_only`, both must be supplied.

Multi-phase pipeline (0–7). The `mode` flag selects which phases run.

| Phase | What runs | Mode(s) |
|---|---|---|
| 0 | Verify / download pretrained checkpoints | all |
| 1 | Fine-tune on `dataset_dir` | `full`, `finetune_only` |
| 2 | Prepare inference JSONL (AMP routing) | `full`, `inference_only` |
| 3 | SDG → `original/` (targets `num_SDG`) | `full`, `inference_only` |
| 4 | Eval `original/` → per-sample nn + `eval.log` | `full`, `inference_only` |
| 5 | Per-sample `(guidance, crop_ratio)` search → `rounds/round_NN/` | `full`, `inference_only` |
| 6 | Assemble best-of-rounds → `searched/` (stitch only) | `full`, `inference_only` |
| 7 | Filter `searched/` by `nn_threshold`; regen dropped via re-AMP; canonical eval | `full`, `inference_only` |

Run every phase through to completion without mid-run pauses. Collect all
required parameters up front. Run every command from the repo root.

**Shell setup.** `${ANOMALYGEN_SCRIPTS}` is preset in the container; on
host, export once per shell:

```bash
export ANOMALYGEN_SCRIPTS="$(git rev-parse --show-toplevel)/scripts/utilities"
```

`python3 -m scripts.utilities.<name>` works from any CWD in the container
(PYTHONPATH preset) and from the repo root on host.

When inside a product container (`ANOMALYGEN_PRODUCT_MODE=1`), invoke
`anomalygen-guard` before any GPU work. If it reports `BLOCKED`, fix the
listed issues before continuing.

## Reference files

| File | When to read |
|---|---|
| `references/finetune.md` | Before Phase 0 / Phase 1 (env check, dataset validation, config, training, best-checkpoint selection) |
| `references/inference.md` | Before any of Phases 2–7 (AMP routing, JSONL, SDG, eval, search, filtering navigator) |
| `references/setup.md` | Checkpoint download fails; first-time setup; HF_TOKEN / disk issues |
| `references/datasets.md` | Preparing or obtaining a UC1 / UC2 / UC3 dataset; `dataset_dir` doesn't exist yet |
| `references/prep-testcase.md` | AMP fails; need full param table, helper descriptions, allocation invariant |
| `references/sdg-inference.md` | NCCL hang; checkpoint validation error; multi-GPU VRAM question; full step list |
| `references/eval.md` | Unexpected scores; FID column order; eval output format reference |
| `references/sdg-refine.md` | draws.json alignment; re-AMP heuristics; search output layout |

For `mode=full`: read both `finetune.md` and `inference.md` before starting.

---

## Required parameters

`num_SDG` allocation depends on `prep_testcase.sh --mode`:
- **`inference` (default, used by Phase 2)** — uniform across defect types;
  override per-defect via `--per-defect-counts`.
- **`validation` (used by Phase 1's validation JSONL)** — proportional to
  training mask counts (largest-remainder rounding); enforces ≥1 per defect.

See `references/prep-testcase.md` for the full mode table.

| Parameter | Description |
|---|---|
| `mode` | `full` (Phase 0→7), `inference_only` (skip Phase 1), or `finetune_only` (Phase 0→1 only). |
| `name` | Experiment label. |
| `dataset_dir` | Training/reference dataset root. Drives mask-count allocation, AMP submask templates, and holds `semantic_segmentation_labels.json` for `cad` defects. |
| `defect_spec` | JSONL tagging each defect `spatial_dependency` as `free`/`text`/`cad`. `text` entries need `roi_prompt_defect_location`. Template: `assets/defect_spec_template.jsonl`. |
| `num_SDG` | Total output samples per bucket. *(Ignored when `mode=finetune_only`.)* |

## Conditionally required

| Parameter | Required when | Description |
|---|---|---|
| `checkpoint_dir` / `step` | `mode=inference_only` | Pre-existing fine-tuned model. In `mode=full` these are auto-derived after Phase 1; passing them is an error. In `mode=finetune_only` silently ignored — Phase 1 always trains from scratch (no resume-from-checkpoint support). Both must be present together — supplying only one is an error. |

## Optional parameters

| Parameter | Default | Description |
|---|---|---|
| `clean_dir` | `dataset_dir` | Clean images. Set only when they live outside the training dataset. Forwarded as `--clean-dir` to prep-testcase and `--clean-image-path` to finetune. |
| `validation_jsonl` | auto-generated | Pre-built validation JSONL for Phase 1. When supplied, preflight verifies every `defect_spec` type appears and paths exist. |
| `num_search_run` | `3` | Per-sample search budget for Phase 5. Set `0` to skip search (only `original/` produced). *(Ignored when `mode=finetune_only`.)* |
| `nn_threshold` | `0.4` | `nn_score` cutoff for Phase 7 (DINOv2 correspondence to real defects — key KPI). Samples below the threshold are regenerated; final `searched/` always has `num_SDG`. Set to `0` to disable filtering. |
| `max_iter` | `75000` | Phase 1 only. Total fine-tune iterations. |
| `save_iter` | `5000` | Phase 1 only. Checkpoint save interval. |
| `validation_iter` | `5000` | Phase 1 only. Validation (`nn_score`) logging interval. |
| `num_gpus` | `1` | Forwarded to Phase 1 (finetune) and Phase 3 (SDG). Eval and search rounds stay single-GPU. |
| `model_size` | `2b` | `2b` or `14b`. Used by finetune and SDG. On-disk checkpoint path encodes in upper-case (`2b`→`2B`, `14b`→`14B`). |
| `lr` | `0.02` | Phase 1 only. Learning rate. |
| `batch_size` | `2` | Phase 1 only. Per-GPU batch size. |
| `image_size` | `512` | Phase 1 only. Training resolution (square). |
| `guidance_range` | `1.5 10.0` | Phase 5 search draw range for guidance. |
| `crop_ratio_range` | `1.5 10.0` | Phase 5 search draw range for crop_ratio. |

---

## Mode validation (fail fast before any phase)

- `mode` unset → halt: *"`mode` is required (`full` | `inference_only` | `finetune_only`)."*
- `mode=inference_only` missing either `checkpoint_dir` or `step` → halt: *"inference_only requires both `checkpoint_dir` and `step`."*
- `mode=full` with `checkpoint_dir` or `step` supplied → halt: *"full mode runs finetune; use `mode=inference_only` to reuse an existing checkpoint."*

## Shared variables

Set once before Phase 0:

```bash
MODE=<full|inference_only|finetune_only>
NAME=<exp>
DATASET_DIR=<dataset_dir>
CLEAN_DIR=${clean_dir:-${DATASET_DIR}}
CKPT=<checkpoint_dir>      # required iff MODE=inference_only; auto-derived after Phase 1 when MODE=full
STEP=<iter>                # required iff MODE=inference_only; auto-derived after Phase 1 when MODE=full
NUM_SDG=<N>
DEFECT_DESC=<defect_spec.jsonl>
DEFECTS=(T+A T+B)          # TEXTURE+TYPE. inference_only: from ${CKPT}/ag_config.yaml → dataloader_train.dataset.anomaly_types (or validate_checkpoint.py). full: from DEFECT_DESC. See references/inference.md §Pre-flight checkpoint validation.
NUM_SEARCH_RUN=${num_search_run:-3}
NN_THRESHOLD=${nn_threshold:-0.4}
MODEL_SIZE=<2b|14b>
NUM_GPUS=${num_gpus:-1}
MAX_ITER=${max_iter:-75000}
SAVE_ITER=${save_iter:-5000}
VALIDATION_ITER=${validation_iter:-5000}
LR=${lr:-0.02}
BATCH_SIZE=${batch_size:-2}
IMAGE_SIZE=${image_size:-512}
VALIDATION_JSONL=${validation_jsonl:-}  # optional; set by Phase 1 Step 2 if not user-supplied

BASE=results/${NAME}
JSONL=ag_inference/${NAME}/testcase.jsonl
ORIGINAL=${BASE}/original
SEARCHED=${BASE}/searched
ROUNDS=${BASE}/rounds
REGENS=${BASE}/regens
```

## Guard preflight (product mode only)

```bash
if [[ "${ANOMALYGEN_PRODUCT_MODE:-}" == "1" ]]; then
    python3 .agents/skills/anomalygen-guard/scripts/preflight.py \
        --mode ${MODE} \
        --name ${NAME} \
        --dataset-dir ${DATASET_DIR} \
        --defect-spec ${DEFECT_DESC} \
        --num-search-run ${NUM_SEARCH_RUN} \
        --model-size ${MODEL_SIZE} \
        ${CLEAN_DIR:+--clean-dir ${CLEAN_DIR}} \
        ${NUM_SDG:+--num-sdg ${NUM_SDG}} \
        ${CKPT:+--checkpoint-dir ${CKPT}} \
        ${STEP:+--step ${STEP}} \
        ${VALIDATION_JSONL:+--validation-jsonl ${VALIDATION_JSONL}}
fi
```

`--validation-jsonl` is forwarded only when user-supplied; preflight then
verifies every `defect_spec` type appears and its `image_filename` /
`mask_filename` paths exist (auto-generated JSONLs are caught upstream by
`allocate_samples.py`, which refuses 0 entries for any defect). For
`MODE=finetune_only`, omit `--num-sdg` if not supplied.

---

## Phase 0 — checkpoints

Read `references/finetune.md §Phase 0` for HF_TOKEN requirements and what
gets downloaded. Both scripts default to the **2B** base + t5-large; pass
`--model-sizes ${MODEL_SIZE^^}` so the chain checks and fetches the base size
this run actually uses (`2b`→`2B`, `14b`→`14B`) — otherwise a `14b` run
silently passes the 2B-only check and never downloads its checkpoint. Verify
first; download only what is missing.

```bash
${ANOMALYGEN_SCRIPTS}/check.sh --model-sizes ${MODEL_SIZE^^} \
    || ${ANOMALYGEN_SCRIPTS}/download_checkpoints.sh --model-sizes ${MODEL_SIZE^^}
```

---

## Phase 1 — fine-tune (skip when `MODE=inference_only`)

Read `references/finetune.md §Phase 1` for dataset structure, config
template details, and best-checkpoint selection.

```bash
# Step 1: Validate dataset structure — derive anomaly types
python3 -m scripts.utilities.validate_dataset ${DATASET_DIR}

# Step 2: Generate validation JSONL (skip if user provided VALIDATION_JSONL)
# num_SDG = total training mask count from Step 1 output
# --mode validation is required (prep_testcase.sh default is inference).
${ANOMALYGEN_SCRIPTS}/prep_testcase.sh \
    --name validation_${NAME} \
    --num-sdg <total_mask_count> \
    --dataset-dir ${DATASET_DIR} \
    --clean-dir ${CLEAN_DIR} \
    --defect-spec ${DEFECT_DESC} \
    --amp-output-dir ag_inference/validation_${NAME}/amp \
    --output-jsonl ag_inference/validation_${NAME}/testcase.jsonl \
    --mode validation
VALIDATION_JSONL=ag_inference/validation_${NAME}/testcase.jsonl

# Step 3: Generate training config — show to user and confirm before writing
python3 -m scripts.utilities.generate_config \
    --name ${NAME} --dataset-dir ${DATASET_DIR} \
    --defect-spec ${DEFECT_DESC} --validation-jsonl ${VALIDATION_JSONL} \
    --output ag_configs/${NAME}.yaml \
    --model-size ${MODEL_SIZE} --max-iter ${MAX_ITER} \
    --save-iter ${SAVE_ITER} --validation-iter ${VALIDATION_ITER} \
    --lr ${LR} --batch-size ${BATCH_SIZE} \
    --image-size ${IMAGE_SIZE}

# Step 4: Launch training (run in background)
${ANOMALYGEN_SCRIPTS}/launch_training.sh \
    --ag-config ag_configs/${NAME}.yaml \
    --num-gpus ${NUM_GPUS} \
    --model-size ${MODEL_SIZE}
```

After training, derive `CKPT` and `STEP` (uppercase model_size in path):

```bash
MODEL_SIZE_UPPER="${MODEL_SIZE^^}"
CKPT="./results/anomaly_gen/${NAME}/${NAME}_training_FP32_lr${LR}_bs=${BATCH_SIZE}_${MODEL_SIZE_UPPER}_${IMAGE_SIZE}x${IMAGE_SIZE}"
# STEP = highest nn_score step from validation logs (see references/finetune.md)
```

If `MODE=finetune_only`: stop here.

---

## Phase 2 — prep-testcase (skip when `MODE=finetune_only`)

Read `references/prep-testcase.md` (AMP routing, full param table) and
`references/inference.md §Phase 2` (pairing strategy, n_seeds sizing).
Do NOT pass `--seeds` — it is auto-computed and is not a recognized flag.

`prep_testcase.sh` defaults to `--mode inference` (uniform allocation across
defect types, no KPI floor). Phase 2 always uses inference mode here.

```bash
${ANOMALYGEN_SCRIPTS}/prep_testcase.sh \
    --name ${NAME} --num-sdg ${NUM_SDG} \
    --dataset-dir ${DATASET_DIR} \
    --clean-dir ${CLEAN_DIR} \
    --defect-spec ${DEFECT_DESC} \
    --amp-output-dir ag_inference/${NAME}/amp \
    --output-jsonl ${JSONL}
```

**Custom per-defect counts.** When the user specifies counts per defect
type (e.g. "5 IC+bridge, 10 passive_component+missing"), add
`--per-defect-counts`:

```bash
${ANOMALYGEN_SCRIPTS}/prep_testcase.sh \
    --name ${NAME} --num-sdg 15 \
    --dataset-dir ${DATASET_DIR} \
    --clean-dir ${CLEAN_DIR} \
    --defect-spec ${DEFECT_DESC} \
    --amp-output-dir ag_inference/${NAME}/amp \
    --output-jsonl ${JSONL} \
    --per-defect-counts '{"IC+bridge":5,"passive_component+missing":10}'
# Defects not in the dict get 0.
```

Sum should equal `--num-sdg` (script warns + uses override sum if not).
Confirm the allocation when the user's intent is ambiguous (e.g. "each
defect 10" + "total only 1").

---

## Phase 3 — SDG → `original/`

Read `references/inference.md §Phase 3` (JSONL validation, multi-GPU
caveats) and `references/sdg-inference.md` (full step list, output
verification, NCCL controls).

```bash
python3 -m scripts.utilities.validate_checkpoint ${CKPT} --step ${STEP}
python3 -m scripts.utilities.validate_jsonl ${CKPT} ${JSONL}

${ANOMALYGEN_SCRIPTS}/run_sdg.sh \
    --checkpoint_dir ${CKPT} --step ${STEP} \
    --input_jsonl ${JSONL} --output_dir ${ORIGINAL} \
    --model_size ${MODEL_SIZE} --num_gpus ${NUM_GPUS}

${ANOMALYGEN_SCRIPTS}/verify_output.sh ${JSONL} ${ORIGINAL}
```

---

## Phase 4 — eval `original/`

Read `references/eval.md` for score interpretation, feature counts, and
error cases. `run_eval.sh` writes three files inside `original/`:
`per_sample.csv`, `eval.log`, and merges `nn_score` into `SDG_result.csv`.

```bash
${ANOMALYGEN_SCRIPTS}/run_eval.sh \
    --real-path ${DATASET_DIR} --generated-path ${ORIGINAL} \
    --anomaly-types ${DEFECTS[@]}
```

---

## Phase 5 — per-sample search rounds

Read `references/inference.md §Phase 5` (draw strategy) and
`references/sdg-refine.md` (draws.json format, ranges, re-AMP).
For `r` in `1..NUM_SEARCH_RUN`:

1. Read prior round's `per_sample.csv` (or `${ORIGINAL}/per_sample.csv` for `r=1`).
2. Write `${ROUNDS}/round_${r}/draws.json` with Claude-chosen `(guidance, crop_ratio)` per sample.
3. Run round (SDG + eval; the round dir gets its own `sdg/{SDG_result.csv, per_sample.csv, eval.log}`):

```bash
${ANOMALYGEN_SCRIPTS}/run_round.sh \
    --base-jsonl ${JSONL} \
    --draws ${ROUNDS}/round_${r}/draws.json \
    --output-dir ${ROUNDS}/round_${r} \
    --real-path ${DATASET_DIR} --anomaly-types ${DEFECTS[@]} \
    --checkpoint-dir ${CKPT} --step ${STEP} \
    [--model-size ${MODEL_SIZE}]
```

`NUM_SEARCH_RUN=0` is valid — skip this phase entirely and let Phase 6
clone `original/` into `searched/`.

---

## Phase 6 — assemble `searched/` (stitch only)

Always run assemble (works with 0 rounds — `searched/` clones
`original/`, so downstream always reads `searched/` regardless of
`num_search_run`). Stitch-only: copies winning images per sample-index
into `searched/` and carries over per-sample `nn_score` / `mnn_score`
from each pick's source-round per_sample.csv. No eval — Phase 7 emits
the canonical `searched/eval.log`.

```bash
mkdir -p ${ROUNDS}
python3 -m scripts.utilities.assemble_searched \
    --original-dir ${ORIGINAL} --original-csv ${ORIGINAL}/per_sample.csv \
    --rounds-dir ${ROUNDS} --searched-dir ${SEARCHED}
```

---

## Phase 7 — filter + regen + eval (default `nn_threshold=0.4`)

Phase 7 **runs by default** (`nn_threshold=0.4`) on every `mode=full` and
`mode=inference_only` invocation. Pass `nn_threshold=0` to skip Phase 7.

Filter `searched/` by `nn_threshold`; dropped samples are regenerated via
re-AMP (fresh `(clean, submask)` pairing, same defect type) up to 5×, then
fall back to best non-passing regens, then dropped originals — final bucket
always equals `num_SDG`. `--allocation` (prep-testcase's `allocation.json`)
makes regen target the intended per-defect counts; without it a bucket left
short (e.g. by an interrupted SDG) can't be topped up. `filter_with_regen.py`
runs the final `run_eval.sh` internally — the only eval against `searched/`.
Read `references/inference.md §Phase 7` for regen mechanics, source-column
tracing, and the `regens/regen_summary.csv` schema.

```bash
python3 -m scripts.utilities.filter_with_regen \
    --searched-dir ${SEARCHED} \
    --per-sample-csv ${SEARCHED}/per_sample.csv \
    --threshold ${NN_THRESHOLD} \
    --num-sdg ${NUM_SDG} \
    --allocation ag_inference/${NAME}/allocation.json \
    --rounds-dir ${ROUNDS} \
    --regens-dir ${REGENS} \
    --dataset-dir ${DATASET_DIR} \
    --clean-dir ${CLEAN_DIR} \
    --defect-spec ${DEFECT_DESC} \
    --real-path ${DATASET_DIR} \
    --anomaly-types ${DEFECTS[@]} \
    --checkpoint-dir ${CKPT} --step ${STEP} \
    --model-size ${MODEL_SIZE} --num-gpus ${NUM_GPUS}
```

---

## Output layout

Every eval'd bucket carries the same triad: `SDG_result.csv` (gen params +
`nn_score`), `per_sample.csv` (per-sample nn + mnn), `eval.log` (FID /
per-defect avg). Buckets under `results/<name>/`: `original/` (Phase 3+4);
`searched/` (final — Phase 6 stitch + Phase 7 filter+regen+eval, its
`SDG_result.csv` adds a `source` column); `rounds/round_NN/` +
`rounds/search_summary.csv` (Phase 5); `regens/regen_NN/` +
`regens/regen_summary.csv` (Phase 7). See `references/inference.md` for the
full tree and per-file schemas.

## Verification

1. `${ORIGINAL}/reconstructed_image/` has up to `num_SDG` images.
2. `${SEARCHED}/reconstructed_image/` count == `num_SDG` (Phase 7 fills via regen + best-per-defect fallback).
3. `${ROUNDS}/search_summary.csv` has one row per sample.
4. `eval.log` in `original/`, each `rounds/round_NN/sdg/`, and `searched/` has per-type `nn_score`, `mnn_score`, `fid`.
5. `${REGENS}/regen_summary.csv` exists when Phase 7 ran (`passed_threshold` = per-sample status; `prev_nn` vs `nn_score` shows which regens improved).

## Error handling

- `dataset_dir` missing per-type mask dir → allocation scans zero and errors.
- AMP output short of allocation → `build_jsonl.py` warns and writes what's available (check `run_auto_roi_amp.py` logs for `NO_DETECTION` / `FAILED`); a defect with **zero** outputs is dropped warn-only, but if **every** defect is zero it halts with `error: 0 entries written`.
- SDG failure mid-round in Phase 5 → halts; re-run resumes from the next round (append-only).
- `mode=inference_only` with a `step` not on a `save_iter` boundary → `torch.load` FileNotFoundError; `ls ${CKPT}/checkpoints/model/iter_*.pt` to find valid steps.
- See `references/finetune.md` and `references/inference.md` for phase-specific error handling.
