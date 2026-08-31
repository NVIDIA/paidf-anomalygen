<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 3 · Auto Mask Placement

This is the third page of the PAIDF AnomalyGen texture tutorial series:

1. [Overview & Setup](1-overview.md)
2. [Dataset Preparation](2-dataset-preparation.md)
3. **Auto Mask Placement** ←
4. [Fine-tuning](4-fine-tuning.md)
5. [Generation](5-generation.md)
6. [Evaluation, Refinement & Pseudo-labeling](6-evaluation-and-refinement.md)

It assumes you have finished [`2-dataset-preparation.md`](2-dataset-preparation.md): a `dataset_dir`
on disk in the [directory contract](2-dataset-preparation.md#the-directory-contract), with a
`defect_spec.jsonl` alongside it.

By the end you will have the two `testcase.jsonl` files the rest of the pipeline consumes — a
**validation** set for fine-tuning and a **generation** set for SDG.

All commands are run from the **repo root**.

______________________________________________________________________

## Prerequisites

Work from the environment shell you set up in [`1-overview.md`](1-overview.md) — either the activated
native venv (`source .venv/bin/activate`) or an interactive container shell (`$DOCKER -it bash`). All
commands below run from the repo root and are identical either way. Ensure Hugging Face auth is
configured (`export HF_TOKEN=<your-token>` or `hf auth login`) — the `text` route downloads a VLM and
SAM2.

______________________________________________________________________

## What AMP does

Fine-tuning ([`4-fine-tuning.md`](4-fine-tuning.md)) trains on the `anomaly_image/` + `mask/` pairs,
but both later stages also consume a **testcase** list — one row per synthetic image, pairing a
**clean** image with a **placed defect mask** (a defect region positioned somewhere valid on that
clean image) plus its gen params:

- **Fine-tuning** reads a validation `testcase.jsonl` (the recipe's `testcase_jsonl` field) for its
  training-time validation passes — synthesizing samples and scoring them per type.
- **Generation** ([`5-generation.md`](5-generation.md)) reads it via `--input_data_path` as the batch
  to synthesize.

**Auto Mask Placement (AMP)** builds that `testcase.jsonl` from the dataset: it decides *where* each
defect goes and places the mask on each clean image.

AMP is a three-stage pipeline of `python -m` CLIs under `anomalygen/scripts/auto_mask_placement/`, run
in order:

```text
roi_allocate  →  roi_pair  →  roi_place
 how many         which pairs    where + place → testcase.jsonl
```

`roi_place_pipeline` runs all three as one command and is how you invoke AMP — it derives every
intermediate path from `--output_dir`, threads `roi_pair`'s seed count into `roi_place`, and stops
with a named stage if one produces nothing.

`roi_allocate` takes a `--mode`:

- **`validation`** — for a fine-tuning validation set: allocates proportionally to the training mask
  counts and guarantees ≥1 sample per defect, so every per-type validation KPI stays defined.
- **`inference`** — uniform counts across defect types, for a generation batch.

`roi_place` routes each defect by its `spatial_dependency` in `defect_spec.jsonl`:

- **`free`** — uses the whole image.
- **`text`** (used by `phone_screen`) — grounds the region from `roi_prompt_defect_location` with a
  VLM + SAM2.
- **`cad`** — uses a PCB CAD mask.

______________________________________________________________________

## Worked example: phone_screen

`phone_screen` has 20 clean images and 5 masks per defect (`oil` / `scratch` / `stain`, all
`spatial_dependency: text`). Build the fine-tuning validation set — 15 placed masks, 5 per defect:

```shell
python -m anomalygen.scripts.auto_mask_placement.roi_place_pipeline \
    --num_sdg 15 --mode validation \
    --defect_desc datasets/phone_screen/defect_spec.jsonl \
    --dataset_dir datasets/phone_screen \
    --output_dir datasets/validation_phone_screen \
    --seed 42
# flags for roi_place go after a bare --, e.g.:  -- --refresh_roi
# --dry_run prints each stage's argv without running it
```

This writes the placed masks, **`testcase.jsonl`**, and `allocation.json`.
`testcase.jsonl` is what your fine-tuning recipe's `testcase_jsonl` points to
([`4-fine-tuning.md`](4-fine-tuning.md)):

```text
datasets/validation_phone_screen/
├── <clean_stem>/Phone+<defect>/
│   ├── assets/                             # roi_mask.png, bbox.png, roi_overlay.png, …
│   └── <submask>_largest__seed0.png        # the placed mask
├── testcase.jsonl                          # one row per placed mask ↔ clean image + gen params
├── allocation.json                         # per-defect counts
├── amp_samples.json                        # submask ↔ clean pairing from roi_pair (+ .n_seeds sidecar)
├── resized_masks/                          # placed masks resized to match their clean image, when needed
└── summary.json                            # per-sample ROI + placement status
```

Each `testcase.jsonl` row is `{image_filename, mask_filename, anomaly_type, guidance, crop_ratio, num_steps, …}`.
Re-running `roi_place` reuses the cached ROIs (the costly VLM + SAM2 step), so adding seeds
or tweaking params is cheap; pass `--refresh_roi` to force regeneration. To place into an ROI you
already have (no models), use the lower-level `place` CLI
(`python -m anomalygen.scripts.auto_mask_placement.place -h`).

For a separate **generation** batch, re-run with `--mode inference` and a different `--seed`, writing
to `datasets/generation_phone_screen/` — the dir generation's `--input_data_path` reads
([`5-generation.md`](5-generation.md)).

```shell
python -m anomalygen.scripts.auto_mask_placement.roi_place_pipeline \
    --num_sdg 15 --mode inference \
    --defect_desc datasets/phone_screen/defect_spec.jsonl \
    --dataset_dir datasets/phone_screen \
    --output_dir datasets/generation_phone_screen \
    --seed 43
```

Use a **different seed** for the two sets, or the generation batch repeats the validation placements.
`--mode inference` splits `--num_sdg` uniformly across defect types; pass
`--per_defect_counts '{"Phone+oil": 5, "Phone+scratch": 5, "Phone+stain": 5}'` instead for explicit
counts (that flag is `inference`-only).

______________________________________________________________________

## A custom dataset

The same pipeline works on any dataset built to the
[directory contract](2-dataset-preparation.md#the-directory-contract): point `--dataset_dir` and
`--defect_desc` at your dataset root and its `defect_spec.jsonl`, and write to your own output dirs
(e.g. `datasets/validation_<name>/` for the recipe's `testcase_jsonl` and
`datasets/generation_<name>/` for a generation batch). Each defect's routing follows its
`spatial_dependency` in `defect_spec.jsonl` (`free` / `text` / `cad`), so no code changes are needed —
just set `--num_sdg` to the batch size you want.

______________________________________________________________________

## Next step

Continue to [4 · Fine-tuning](4-fine-tuning.md).
