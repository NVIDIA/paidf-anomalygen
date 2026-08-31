# Inputs reference

Progressive-disclosure detail for the **Inputs** section in the parent `SKILL.md` — the mode table, the
parameter reference (required + optional), and the step map. If this file conflicts with `SKILL.md`,
`SKILL.md` wins.

## Modes

Which steps run depends on the mode:

| User intent                | `mode`            | Must collect up front                            | Steps    |
| -------------------------- | ----------------- | ------------------------------------------------ | -------- |
| Train + generate           | `full`            | `dataset_dir`, `recipe`, `num_sdg`               | 1 → 7    |
| Train only                 | `finetune_only`   | `dataset_dir`, `recipe`                          | 1 → 3    |
| Generate from a checkpoint | `generation_only` | `checkpoint`, `recipe`, `dataset_dir`, `num_sdg` | 2, 4 → 7 |

`generation_only` (a.k.a. `inference_only`) still needs `dataset_dir` + `defect_spec` — to build the
generation batch and as the evaluation real-reference root. In `full`, `checkpoint` is derived after
Step 3 — don't pass it. Collect params up front; run straight through.

Each step's section header lists the modes it runs in (e.g. `### Step 6 — … (skip when finetune_only)`).

## Parameters

### Required

| Param         | Maps to                         | Notes                                                       |
| ------------- | ------------------------------- | ----------------------------------------------------------- |
| `mode`        | —                               | `full` \| `finetune_only` \| `generation_only`.             |
| `name`        | —                               | Run label; names the run dir + output paths.                |
| `dataset_dir` | `--dataset_dir` / `--real_root` | Step 1 contract layout; also the eval real-reference.       |
| `defect_spec` | `--defect_desc`                 | `defect_spec.jsonl` — one row per `{texture}+{defect}`.     |
| `recipe`      | `--recipe`                      | Central config — copy the template, edit the dataset block. |
| `num_sdg`     | `--num_sdg`                     | Samples across defect types; final `searched/` count.       |
| `checkpoint`  | `--checkpoint`                  | **generation_only** — `iter_<best>.pt`; `full` derives it.  |

### Optional (defaults shown)

| Param                       | Maps to               | Default       | Notes                            |
| --------------------------- | --------------------- | ------------- | -------------------------------- |
| `num_search_run`            | — (Step 6 rounds)     | `3`           | Search rounds; `0` = clone only. |
| `num_gpus`                  | `--nproc_per_node`    | `1`           | GPUs (fine-tune + generation).   |
| `per_defect_counts`         | `roi_allocate`        | uniform       | Exact per-defect gen counts.     |
| `no_caption`                | `pseudo_label`        | off           | Skip Step 7 captioning.          |
| `max_iter`                  | `recipe`              | `15000`       | Total fine-tune iterations.      |
| `save_iter`                 | `recipe`              | `1000`        | Checkpoint interval.             |
| `validation_iter`           | `recipe`              | `1000`        | Validation interval.             |
| `lr`                        | `recipe`              | `0.001`       | AdamW learning rate.             |
| `batch_size`                | `recipe`              | `4`           | Samples per iteration.           |
| `image_size`                | `recipe`              | `[512,512]`   | Training resolution.             |
| `ratio_range`               | `recipe`              | `[1.5,8.0]`   | Anomaly-crop area ratio.         |
| `per_class_lora_rank/alpha` | `recipe`              | `8` / `8`     | Per-defect LoRA capacity.        |
| `early_stop_enabled`        | `recipe`              | `false`       | Stop when `nn` plateaus.         |
| `guidance_range`            | `quality_refine draw` | `1.5 10.0`    | Step 6 guidance draw range.      |
| `crop_ratio_range`          | `quality_refine draw` | `1.5 8.0`     | Step 6 crop-ratio draw range.    |
| `bo_min_obs`                | `quality_refine draw` | `2`           | Min obs before per-sample BO.    |

`recipe` knobs live in `ag_config/exp_${TASK}_<name>.yaml` (full key list + defaults in
`fine-tuning.md`); `quality_refine draw` search flags in `quality-refinement.md`;
`--per_defect_counts` in `mask-placement.md`. `SKILL.md` links all three.

## Step map

| Step | What runs                                      | Entry point                              |
| ---- | ---------------------------------------------- | ---------------------------------------- |
| 1    | Dataset in contract layout                     | `prepare_*` or custom                    |
| 2    | Mask placement → `testcase.jsonl`              | `roi_place_pipeline`                     |
| 3    | Fine-tuning (LoRA adapters)                    | `scripts/skill_utility/train_control.sh` |
| 4    | Generation (SDG)                               | `${SCRIPTS}/generate.py`                 |
| 5    | Evaluation (quality KPI)                       | `${SCRIPTS}/evaluate.py`                 |
| 6    | Quality refinement → `searched/`               | `${SCRIPTS}/quality_refine.py run`       |
| 7    | Pseudo-labeling → COCO/classification/captions | `${SCRIPTS}/pseudo_label.py`             |
