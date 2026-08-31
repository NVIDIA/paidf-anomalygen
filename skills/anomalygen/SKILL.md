---
name: anomalygen
description: >-
  Use when running the PAIDF AnomalyGen pipeline over a defect dataset — mask placement,
  fine-tuning, synthetic defect-image generation (SDG), evaluation, quality refinement, and
  pseudo-labeling — even when the user names only one stage.
license: Apache-2.0
compatibility: >-
  Requires a CUDA GPU and the built env (native .venv or the Docker image); Hugging Face auth +
  network access to download the model checkpoints.
metadata:
  owner: NVIDIA
  service: anomaly generation
  version: 1.1.0
  reviewed: '2026-08-06'
  author: NVIDIA <nvidia@nvidia.com>
  tags:
      - physical-ai
      - anomaly-generation
      - cosmos
      - fine-tuning
  languages: [python]
  frameworks: [pytorch, cosmos, diffusion]
---

# Skill: anomalygen

## Purpose

AnomalyGen fine-tunes the frozen **Cosmos** base into an **anomaly-generation** model: given a clean
image + defect mask, it paints a realistic defect into that region (synthetic data for detectors).
Pick a mode, collect its params, run the steps — **all commands from the repo root.**

## When to Use This Skill

Use for the PAIDF AnomalyGen pipeline over a defect dataset — any single stage (mask placement,
fine-tuning, generation, evaluation, refinement, or pseudo-labeling) or the full run.

- "Run the AnomalyGen pipeline on phone_screen"
- "Fine-tune an anomaly-generation model for PCB defects"
- "Generate 25 synthetic PCB anomaly images"
- "Place defect masks", "evaluate quality", or "pseudo-label the generated defects"

Do not use this skill for generic image generation, or for synthetic data that isn't defect images
built from the `{texture}/{defect}` clean-image + defect-mask contract.

## Prerequisites

- A **CUDA GPU** and the project's built env (native `.venv` or the Docker image).
- **Hugging Face auth** — run `hf auth login` — plus network access; the downloader fetches
  the model checkpoints.
- A **`dataset_dir`** in the Step 1 contract layout (clean images + `{defect}` masks + `defect_spec.jsonl`).

Preflight the env + checkpoints (GPU, Python, CUDA/torch, deps, HF auth, model checkpoints):

```shell
scripts/preflight_env_ckpt.sh
```

**Read which group failed before reacting** — the summary separates them, and only one is fixed by
downloading:

- *missing checkpoints* → `scripts/download_checkpoints.sh` (run `hf auth login` first).
- *env / Python / deps* → rebuild the environment (`bash scripts/env_setup.sh`, then activate it).

## Inputs

Collect the run's inputs from the user before starting. **If the `mode` or any input it requires is
missing, ask the user first — never guess a required input or silently fall back to its default.**

**Lock the `mode` first** — it fixes which steps run and which inputs are required (ask for any not
provided):

- **`full`** — train + generate, Steps **1 → 7**. Required: `name`, `dataset_dir`, `recipe`, `num_sdg`.
- **`finetune_only`** — train only, Steps **1 → 3**. Required: `name`, `dataset_dir`, `recipe`.
- **`generation_only`** (a.k.a. `inference_only`) — generate from an existing checkpoint, Steps
  **2, 4 → 7**. Required: `name`, `checkpoint`, `recipe`, `dataset_dir`, `num_sdg`.

`name` is required in every mode — it names the dataset dirs, run dir, recipe and log.

**Optional in every mode** — all have working defaults, so ask only if the user raises them:
`num_gpus` (`1`), `num_search_run` (`3`), `task` (`texture_ft`),
`base_seed` (`1`). Anything not listed as **Required** above is optional.

Parameter tables (each mapped to its CLI arg) and the step map: [`inputs.md`](references/inputs.md).
Confirm the inputs, set the shared variables below, then run straight through — in `full`,
`checkpoint` is derived after Step 3, so don't pass it.

## Instructions

Run the steps your `mode` selects (see **Inputs**), in order — each step's header notes when to skip it.

> **A step is done only when its output exists on disk — not when its command is *launched*.** Every
> step ends with a named gate (**dataset**, **amp**, **train**, …); run it before moving on. Never
> treat starting a long job as a stopping point, nor proceed on a partial artifact. Gate commands and
> the fix for each failure: [`references/verification.md`](references/verification.md).

Set the shared variables every step below refers to. **Re-source this in each new shell** — variables
do not survive one — and read the values it echoes back (`${DATASET_DIR}`, `${VAL_DIR}`, `${GEN_DIR}`,
`${OUT}`, `${RUN_DIR}`, `${LOG_DIR}`, …):

```shell
source scripts/skill_utility/set_pipeline_vars.sh --name <name> --num_sdg <num_sdg> \
    [--num_gpus 1] [--num_search_run 3] [--task texture_ft]
```

**Every step's log goes under `${LOG_DIR}` (`results/`), never the repo root.** Use
`${LOG_DIR}/<step>_${NAME}.log`. The same goes for PID files — the launcher writes its own under `results/`;
do not add one.

The steps below pipe through `tee` to keep the log and the console. **Run `set -o pipefail` first** —
without it the pipeline's exit status is `tee`'s, so a step that died reports success:

```shell
set -o pipefail
```

`set_pipeline_vars.sh` **rejects** a `name`/`task` outside `[A-Za-z0-9][A-Za-z0-9._-]*` and any non-integer count rather
than sanitizing them — both reach command lines below. Apply the same care to user-supplied paths
(`dataset_dir`, `recipe`, `checkpoint`): quote every expansion. `texture_ft` is the only `task_type`
today; a new task family adds one `${TASK}`→`${SCRIPTS}` line to that script plus a matching
`task_type:` in the recipe.

Step 3 sets `CKPT` and repoints `RECIPE` at the run-dir copy for Steps 4–7. **`generation_only` skips
Step 3**, so set both yourself before Step 4 (use the user's `checkpoint` verbatim if given; `CKPT` and
`RECIPE` must come from one run or class ids drift):

```shell
RUN_DIR=<existing run dir>                      # only if it is not the ${RUN_DIR} already set above
CKPT="${RUN_DIR}/checkpoints/model/$(cat "${RUN_DIR}/checkpoints/best_checkpoint.txt")"
RECIPE="${RUN_DIR}/exp_${TASK}_${NAME}.yaml"    # run-dir copy — MUST match the checkpoint
```

### Step 1 — dataset (contract layout)

All modes need `dataset_dir` on disk in this layout — training reads the `anomaly_image/`+`mask/`
pairs; AMP and eval read `clean_image/` and `defect_spec.jsonl`:

```text
{dataset_dir}/{texture}/anomaly_image/{defect}/<name>.png   # defect image
{dataset_dir}/{texture}/mask/{defect}/<name>_mask.png       # binary mask (white=defect); same stem + _mask
{dataset_dir}/{texture}/clean_image/<name>.png              # clean images (AMP inputs)
{dataset_dir}/{texture}/cad_mask/<clean_stem>.png           # cad only: one per clean image, no {defect} level
{dataset_dir}/semantic_segmentation_labels.json             # cad only: label map for those CAD masks
{dataset_dir}/defect_spec.jsonl                             # one row per {texture}+{defect}; sets spatial_dependency (free/text/cad)
```

Ready-made subjects:

```shell
python -m anomalygen.scripts.datasets.prepare_phone_screen_defect --output_dir "${DATASET_DIR}" --zip <roboflow.zip>
# also: prepare_pcb_defect, prepare_magnetic_tile_defect  (see anomalygen/scripts/datasets/README.md)
```

For custom data, emit the same layout and list every `[texture, defect]` pair in the recipe's
`anomaly_types`. Confirm the layout before proceeding — gate **dataset**.

### Step 2 — auto mask placement (AMP)

**Reference** (allocation modes, KPI floor, `--per_defect_counts`, ROI caching, `free/text/cad`):
[`mask-placement.md`](references/mask-placement.md).

AMP builds a `testcase.jsonl` (clean image + placed mask + gen params) by running allocate → pair →
place. Build the **`validation`** set (fine-tune) and/or the **`generation`** set (SDG), each with a
distinct `--seed` — **full** → both, **finetune_only** → validation, **generation_only** → generation:

```shell
AMP_MODE=validation; AMP_DIR="${VAL_DIR}"; SEED=42    # or: AMP_MODE=inference AMP_DIR="${GEN_DIR}" SEED=43

python -m anomalygen.scripts.auto_mask_placement.roi_place_pipeline \
    --num_sdg "${NUM_SDG}" --mode "${AMP_MODE}" --defect_desc "${DEFECT_SPEC}" \
    --dataset_dir "${DATASET_DIR}" --output_dir "${AMP_DIR}" --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/amp_${AMP_MODE}_${NAME}.log"
# flags for roi_place go after a bare --, e.g.:  -- --refresh_roi
```

It derives every intermediate path from `--output_dir`, threads `roi_pair`'s `n_seeds` into
`roi_place`, and stops with a named stage if one produces nothing. Output is placed masks +
**`${AMP_DIR}/testcase.jsonl`**. The **generation** set must have exactly as many rows as the
allocation asked for — `num_sdg`, or the `--per_defect_counts` sum when you passed one — verify before
proceeding: gate **amp**.

### Step 3 — fine-tuning (skip when generation_only)

**Reference** (recipe knobs, launch mechanics, early stopping, checkpoint selection):
[`fine-tuning.md`](references/fine-tuning.md).

**Before training:** copy the template recipe and edit its dataset block so `testcase_jsonl` points at
the validation AMP output from Step 2 (other keys ship at tested defaults):

```shell
cp ag_config/exp_texture_ft_phone_screen.yaml "${RECIPE}"   # the worked example doubles as the template
```

```yaml
# edit the dataset block in ${RECIPE}
dataset_name: ${NAME}
anomaly_types: [[<texture>, <defect>], ...]   # every [texture, defect] pair on disk
dataset_path: datasets/${NAME}
testcase_jsonl: datasets/validation_${NAME}/testcase.jsonl        # ← Step 2 validation set
```

A fine-tune runs hours at the `15000` default, so it is launched detached (own session, PID file, log).
**Check `${RECIPE}` is the one you meant before launching** — it then runs unattended with your
credentials:

```shell
scripts/skill_utility/train_control.sh start --name "${NAME}" --recipe "${RECIPE}" \
    --num_gpus "${NUM_GPUS}" --scripts "${SCRIPTS}" --task "${TASK}"
```

**This launcher is the only sanctioned way to start a fine-tune. Never hand-roll a `torchrun` for
training.** It is what writes `results/train_${NAME}.log` and `results/train_${NAME}.pid`, records the
run dir, and makes `wait`/`stop` work; a direct `torchrun` bypasses all of it, and a literal
`--nproc_per_node=<n>` typed at the prompt silently discards `${NUM_GPUS}` and the user's GPU budget
with it. Pass `--num_gpus ${NUM_GPUS}` — never a literal.

Writes `${RUN_DIR}` (recipe copied in). `train_control.sh stop --name ${NAME}` ends a run;
`start` again resumes from `latest_checkpoint.txt`.

**Waiting is an action.** Call `wait` in the foreground until it stops returning **2** — arming a
background watcher and ending your turn leaves nobody to run Steps 4–7:

```shell
scripts/skill_utility/train_control.sh wait --name "${NAME}"   # 0 = finished · 1 = failed · 2 = still running, call again
```

**Each call is a ~9.5-minute slice, not the whole wait.** The `15000` default ≈ **~25 calls**.
Only exit **0** clears gate **train** — not a slice expiring, and not a checkpoint appearing
(the trainer writes an intermediate `iter_<N>.pt` every `save_iter`). On **1**, read the log
and fix the run; do not proceed.

**`finetune_only` stops here.** For `full`, take the **best** iteration from the pointer the trainer
writes at train end (peak `nn_score` on the `Average` column) — runs routinely peak mid-training:

```shell
CKPT="${RUN_DIR}/checkpoints/model/$(cat "${RUN_DIR}/checkpoints/best_checkpoint.txt")"  # iter_<best>.pt
RECIPE="${RUN_DIR}/exp_${TASK}_${NAME}.yaml"  # run-dir copy — MUST match the checkpoint
```

Never substitute `latest_checkpoint.txt` — it names the *last* iteration. `${RUN_DIR}` also resolves
correctly (best pointer first, then latest, then highest `iter_*.pt`), but pass the explicit path so
the record names the exact checkpoint. A missing pointer is a **train** gate failure — fix the run,
don't let it fall back to the latest.

### Step 4 — generation / SDG (skip when finetune_only)

Reads the **generation** AMP `testcase.jsonl` (Step 2, `GEN_DIR`). `--recipe` **must be the recipe the
checkpoint was trained with** (use the run-dir copy) so `anomaly_type`→class-id lines up.

```shell
torchrun --nproc_per_node="${NUM_GPUS}" "${SCRIPTS}/generate.py" \
    --checkpoint "${CKPT}" \
    --recipe "${RECIPE}" \
    --input_data_path "${GEN_DIR}/testcase.jsonl" \
    --output_dir "${OUT}" \
    2>&1 | tee "${LOG_DIR}/generation_${NAME}.log"
```

`--nproc_per_node=${NUM_GPUS}` — expand the variable, never a literal count.

`--base_seed` (default `1`) seeds the per-testcase noise. **Don't change it mid-pipeline** — Step 6
compares KPIs across re-generated rounds, which a different seed makes incomparable.

Output tree: `reconstructed_image/` (the synthetic defects), `original_image/`, `original_mask/`, and
`${TASK}_generation_result.csv`. `generate.py` emits **one image per testcase row** (no quality drop),
except for samples the content-safety guardrail blocks — those write no image and land in
`guardrail_blocked.csv`, which is expected, not a failure. Verify images + blocked = rows, and the
canonical CSV header, before Step 5 (gate **generate**); **never hand-edit the CSV** — a non-canonical
CSV crashes Step 6.

### Step 5 — evaluation (skip when finetune_only)

Scores the batch against the real dataset with the same correspondence KPI as training validation.

```shell
python "${SCRIPTS}/evaluate.py" \
    --gen_root "${OUT}" --real_root "${DATASET_DIR}" \
    --recipe "${RECIPE}" \
    --output_file "${OUT}/${NAME}_kpi.json" \
    2>&1 | tee "${LOG_DIR}/evaluation_${NAME}.log"
```

Writes per-type + `Average` blocks (`nn_score`, `mnn_score`, `fid`, `per_sample[]`) — verify it exists
(gate **eval**). **`nn_score` is the primary KPI** (higher is better); `mnn_score`/`fid` are secondary.
It also scores the anomaly-quality axes and `aq_nn`, the same metrics validation records
([`fine-tuning.md`](references/fine-tuning.md)). `--anomaly_types Phone+oil ...` overrides
recipe-derived types.

### Step 6 — quality refinement (skip when finetune_only)

**Reference** (draw strategy, ranges/seeding, `select`):
[`quality-refinement.md`](references/quality-refinement.md).

`quality_refine.py run` does the whole loop: `(draw → generate → evaluate) ×${NUM_SEARCH_RUN}`, then `select` and a final
score. Each round's `(guidance, crop_ratio)` is drawn per sample — **round 1 uniform**, **rounds 2+
per-sample Bayesian optimization** (see the reference):

```shell
python "${SCRIPTS}/quality_refine.py" run \
    --base_testcase "${GEN_DIR}/testcase.jsonl" \
    --original "${OUT}" --original_kpi "${OUT}/${NAME}_kpi.json" \
    --rounds_dir "${OUT}/rounds" --output "${OUT}/searched" \
    --final_kpi "${OUT}/searched/${NAME}_kpi.json" \
    --checkpoint "${CKPT}" --recipe "${RECIPE}" --real_root "${DATASET_DIR}" \
    --num_search_run "${NUM_SEARCH_RUN}" --num_gpus "${NUM_GPUS}"
```

It refuses to continue past a round that did not finish — one image per row *and* its `kpi.json`, both
of which `select` keys off, so an unfinished round would otherwise be dropped silently and its GPU time
wasted. **`select` always runs**: `num_search_run=0` reduces it to cloning `${OUT}` into `searched/`, so
`searched/` always exists for Step 7. Verify the round dirs before Step 7: gate **refine**.

The `Average.nn_score` delta between `${OUT}/searched/${NAME}_kpi.json` and the Step 5
`${OUT}/${NAME}_kpi.json` is what refinement bought. Expect the refined batch to beat *every* individual
round (selection is per sample); ~0 delta means the search added nothing — raise `num_search_run` or
widen `guidance_range`.

### Step 7 — pseudo-labeling (skip when finetune_only)

Turns the refined `${OUT}/searched` tree into a labeled dataset: COCO instance annotations, a
classification layout, visualizations, and (unless `--no_caption`) VLM captions.

```shell
python "${SCRIPTS}/pseudo_label.py" \
    --gen_root "${OUT}/searched" \
    --output_dir "${OUT}/searched/pseudo_labels"
# add --no_caption to skip the VLM captioning (fast, no GPU model load)
```

`--gen_root` must contain `reconstructed_image/`, `original_image/`, `original_mask/`, and
`${TASK}_generation_result.csv` (Step 6 always produces this at `${OUT}/searched`). Verify the labels
landed — gate **label**.

## Verification

The artifact each gate requires. Runnable commands + the fix for each failure:
[`verification.md`](references/verification.md).

| #   | Gate       | Step | Requires                                                                                   |
| --- | ---------- | ---- | ------------------------------------------------------------------------------------------ |
| 1   | `dataset`  | 1    | `defect_spec.jsonl` + per-pair `clean_image/`, `anomaly_image/{defect}/`, `mask/{defect}/` |
| 2   | `amp`      | 2    | `${AMP_DIR}/testcase.jsonl`; generation row count = the allocation                         |
| 3   | `train`    | 3    | `wait` exit **0** **and** `checkpoints/best_checkpoint.txt`                                |
| 4   | `generate` | 4    | canonical CSV header **and** images + `guardrail_blocked.csv` rows = testcase rows         |
| 5   | `eval`     | 5    | `${OUT}/${NAME}_kpi.json` with per-type + `Average`                                        |
| 6   | `refine`   | 6    | `${NUM_SEARCH_RUN}` round dirs, **each complete** (images *and* `kpi.json`)                |
| 7   | `label`    | 7    | `pseudo_labels/coco_annotations.json` + `classification/classes.txt`                       |

Gate 6 counts *complete* rounds — existing dirs aren't enough. An all-`original` `searched/` `source`
column is a valid outcome (no round beat Step 4), not a failure; missing round dirs are.

## Examples

Natural invocations — the `key=value` params map to the **Inputs** table (recipe knobs like `max_iter`
edit the recipe; see [`references/inputs.md`](references/inputs.md)). `pcb` is `cad`-routed,
`magnetic_tile` is `free`-routed and `phone_screen` is `text`-routed, so their Step 2 AMP differs
(see [`references/mask-placement.md`](references/mask-placement.md)).

Full pipeline (Steps 1→7) on the text-grounded phone-screen set:

```text
Use AnomalyGen skill with
name=phone_screen
mode=full
dataset_dir=datasets/phone_screen
num_sdg=25
max_iter=15000
validation_iter=1000
save_iter=1000
```

To generate more from an existing checkpoint instead, switch to `mode=generation_only` and add
`recipe=${RUN_DIR}/exp_${TASK}_<name>.yaml` + `checkpoint=${RUN_DIR}/checkpoints/model/iter_<best>.pt`
(from `best_checkpoint.txt`), dropping the training knobs (`max_iter`/`validation_iter`/`save_iter`).

## Troubleshooting

Each fails *quietly* — plausible output, wrong result. Detail in
[`fine-tuning.md`](references/fine-tuning.md) and [`mask-placement.md`](references/mask-placement.md).

| Symptom                                     | Cause / fix                                                                 |
| ------------------------------------------- | --------------------------------------------------------------------------- |
| Class ids misaligned in the labels          | Different `anomaly_types` order — use the run-dir recipe copy               |
| Validation scores nothing                   | `testcase_jsonl` not wired to the Step 2 validation set                     |
| A fresh `max_iter` run exits immediately    | Resumed an old `iter_*.pt` — clear it or set a new `job_name`               |
| Fewer rows than the allocation              | A defect yielded zero placements — check `roi_place` logs                   |
| A worse checkpoint ships, numbers look fine | Read column 1 of `valid_kpi.csv`, not `Average` — use `best_checkpoint.txt` |
