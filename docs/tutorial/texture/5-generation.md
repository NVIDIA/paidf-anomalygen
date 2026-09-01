<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 5 · Generation (SDG)

This is the fifth page of the PAIDF AnomalyGen texture tutorial series:

1. [Overview & Setup](1-overview.md)
2. [Dataset Preparation](2-dataset-preparation.md)
3. [Auto Mask Placement](3-auto-mask-placement.md)
4. [Fine-tuning](4-fine-tuning.md)
5. **Generation** ←
6. [Evaluation, Refinement & Pseudo-labeling](6-evaluation-and-refinement.md)

It assumes you finished [`4-fine-tuning.md`](4-fine-tuning.md) and have a fine-tuned run directory
under `results/anomalygen/phone_screen/<job_name>` (e.g.
`results/anomalygen/phone_screen/anomalygen_texture_ft`).

In this stage you run the fine-tuned model over a batch of **testcases** to synthesize defect images.
Scoring, filtering, refinement and pseudo-labeling all follow on
[`6-evaluation-and-refinement.md`](6-evaluation-and-refinement.md). All commands run from the repo
root.

## Prerequisites

Work from the environment shell you set up in [`1-overview.md`](1-overview.md) — either the activated
native venv (`source .venv/bin/activate`) or an interactive container shell (`$DOCKER -it bash`). All
commands below run from the repo root and are identical either way.

## Generation

### Inputs

Generation takes three inputs:

1. **The fine-tuned checkpoint** (`--checkpoint`) — the run directory produced in
   [`4-fine-tuning.md`](4-fine-tuning.md) (or a specific `checkpoints/model/iter_<N>.pt` inside it).
2. **The fine-tuning recipe** (`--recipe`) — the recipe the checkpoint was trained with.
3. **The testcase JSONL** (`--input_data_path`) — a list of what to generate.

> **⚠ `nn_score` fluctuates early in training.** An early iteration can post a high score before the
> model has settled, and `checkpoints/best_checkpoint.txt` — which `--checkpoint <run_dir>` resolves
> through — is otherwise a plain best-of across every validated iteration. So on any run with
> `max_iter` above **7500**, iterations below that are excluded from the pick.
>
> Two cases to know about. A **short run** (`max_iter <= 7500`, e.g. the dry run) keeps every
> iteration eligible, so its pick may well be an unsettled early one. And a long run that stopped
> before 7500 — early stopping, or a crash — falls back to the full set and logs a warning. In both,
> check the iteration named by `cat .../checkpoints/best_checkpoint.txt` against
> `training_curves.png`, and compare against a later `iter_<N>.pt` before committing to a full batch.

### The testcase JSONL

The testcase file has **one JSON object per line**. Each object describes a single generation
request: which clean image to inpaint, which mask defines the defect region, and which defect type
to synthesize. It is built by Auto Mask Placement ([`3-auto-mask-placement.md`](3-auto-mask-placement.md))
as a **generation** batch (`--mode inference`).

Each line has three required fields plus optional per-row knobs:

| Field            | Meaning                                                          |
| ---------------- | ---------------------------------------------------------------- |
| `anomaly_type`   | `"{texture}+{defect}"`, e.g. `Phone+oil` — mapped to a class id. |
| `image_filename` | clean source image to inpaint the defect into.                   |
| `mask_filename`  | binary mask marking where the defect goes.                       |

Optional per-row knobs override the default for that one line (each falls back to a default when
omitted): `guidance`, `num_steps`, `seed`, `num_generated_images`, `crop_and_paste`, `crop_ratio`,
`poisson_blend`, and `iteration_generation_max_instance`.

> **⚠ `crop_and_paste: false` is supported, but expect worse results on small defects.** With it
> off, the whole image is resized to the model's 512×512 input instead of a defect-centred crop.
> Training never shows that framing when the defect is small: it always crops a window of
> `max(bbox) × ratio` with `ratio` in `[1.5, 8.0]`. Prefer leaving `crop_and_paste: true`
> (with `crop_ratio`) unless you specifically need the whole frame.

`seed` is the one whose default is not a constant: when a line omits it, the noise seed is derived
per testcase from `--base_seed` (default `1`) and the line's position in the file, so each testcase
gets its own noise. An explicit `seed` on the line always wins; bump `--base_seed` to re-roll the
noise for a whole batch.

A real example line (from `datasets/generation_phone_screen/testcase.jsonl`):

```json
{"image_filename": "datasets/phone_screen/Phone/clean_image/0001.png", "mask_filename": "datasets/generation_phone_screen/0001/Phone+oil/Oil_0001_mask_largest__seed0.png", "anomaly_type": "Phone+oil", "guidance": 6.0, "num_steps": 35, "crop_and_paste": true, "crop_ratio": 2.0, "num_generated_images": 1, "poisson_blend": false, "iteration_generation_max_instance": 5}
```

### Run generation

```shell
torchrun --nproc_per_node=1 anomalygen/scripts/texture/generate.py \
    --checkpoint results/anomalygen/phone_screen/anomalygen_texture_ft/checkpoints/model/iter_000015000.pt \
    --recipe results/anomalygen/phone_screen/anomalygen_texture_ft/exp_texture_ft_phone_screen.yaml \
    --input_data_path datasets/generation_phone_screen/testcase.jsonl \
    --output_dir results/generation_phone_screen
```

### Checkpoint & recipe semantics

- **`--checkpoint`** is the fine-tuned run dir (or a specific
  `checkpoints/model/iter_<N>.pt`). The base network is loaded automatically from the recipe's
  `model_size` (`nano` → `checkpoints/Cosmos3-Nano`, `edge` → `checkpoints/Cosmos3-Edge`; a recipe
  that omits `model_size` falls back to `nano`, and `edge` is **experimental** — see
  [4-fine-tuning](4-fine-tuning.md)), then the trained **per-defect-type LoRA adapters**
  are overlaid on top of it — the fine-tuned checkpoint only stores those adapters, not the full
  network. `--base_checkpoint <dcp_dir>` overrides that choice; only use it when the base you point
  at is the same size the adapters were trained on.
- **`--recipe`** registers the experiment config and maps each `anomaly_type` (`texture+defect`) to
  its class id, derived from the fine-tuning recipe's `anomaly_types` list. **Use the same fine-tuning
  recipe the checkpoint was trained with** — the commands above use the copy saved in the run dir
  (`.../<JOB_NAME>/exp_texture_ft_phone_screen.yaml`), which always matches. Otherwise the class ids will
  not line up with the trained adapters.

### Output folder structure

Everything lands under `--output_dir` (here `results/generation_phone_screen`):

```text
results/generation_phone_screen/
├── reconstructed_image/            # the generated defect images (the actual output)
│   ├── Phone+oil_00000.png
│   ├── Phone+oil_00001.png
│   └── Phone+scratch_00000.png
├── original_image/                 # the clean input images (image_filename)
│   └── Phone+oil_00000.png
├── original_mask/                  # the input defect masks (mask_filename)
│   └── Phone+oil_00000.png
├── cropped_image/                  # crop-and-paste debugging aid
├── cropped_mask/                   # crop-and-paste debugging aid
├── mask_cropped_image/             # crop-and-paste debugging aid
├── annotated_image/                # mask overlaid on the result, for inspection
├── texture_ft_generation_result.csv   # one row per generated image + its params
├── guardrail_blocked.csv              # blocked samples; written whenever the guardrail ran (header-only if none)
└── timing_summary.json                # setup / model-init / generation wall-times
```

In `reconstructed_image/`, `original_image/` and `original_mask/`, files are named
`{anomaly_type}_{idx:05d}.png`, with `idx` a stable per-`anomaly_type` counter — so the same base name
in those three subdirs refers to the same sample. The four debug views (`cropped_image/`,
`cropped_mask/`, `mask_cropped_image/`, `annotated_image/`) are written **per defect instance**, with a
second index appended: `{anomaly_type}_{idx:05d}_{inst:05d}.png`. One sample contributes as many files
there as its mask had instances, so those subdirs can hold more files than `reconstructed_image/`.

- **`reconstructed_image/`** — the synthesized defect image; this is what you actually want.
- **`original_image/`** and **`original_mask/`** — the inputs (clean image and mask) for that
  sample, copied out for easy side-by-side comparison.
- **`cropped_image/`**, **`cropped_mask/`**, **`mask_cropped_image/`**, **`annotated_image/`** —
  debugging / inspection aids (the crop-and-paste region and a mask overlay).
- **`texture_ft_generation_result.csv`** — one row per generated image recording the resolved
  `guidance`, `num_steps`, `seed`, crop/blend settings, and PSNR.
- **`guardrail_blocked.csv`** — one row per sample the content-safety guardrail rejected (see below).
- **`timing_summary.json`** — aggregate timing across ranks.

### The content-safety guardrail

**The guardrail is on by default**, so a batch can legitimately produce **fewer images than your
testcase file has rows**. Each sample's caption is screened by the text guardrail before generation,
and every generated composite goes through the image guardrail before it is saved.

A blocked sample is **skipped entirely** and recorded in `guardrail_blocked.csv`. That manifest is
kept separate from `texture_ft_generation_result.csv` on purpose: filtering and pseudo-labeling read
the latter by `output_filename`, so they never see rows for files that were not written.

So the count to check after generation is **images + `guardrail_blocked.csv` rows = testcase rows**.

Only the final `reconstructed_image/` composite is screened. The `original_image/` / `original_mask/`
inputs are your own passthroughs, and the `annotated_image/` / `cropped_*` / `mask_cropped_image/`
artifacts are internal debug views — all are written unguarded.

Two flags control it:

- **`--no-guardrail`** — disable the screening entirely (it is `--guardrail` / on by default).
- **`--offload_guardrail_models`** — offload the guardrail models to CPU between calls to save VRAM.

## Notes

- **The example checkpoint is from a full run.** `iter_000015000.pt` is what a complete 15000-iter
  fine-tune produces (the recipe's `max_iter`). If you ran the quick smoke test from
  [`4-fine-tuning.md`](4-fine-tuning.md) instead, point `--checkpoint` at the checkpoint your run
  actually produced (e.g. `iter_000000020.pt`).
- **Multi-GPU generation.** `--nproc_per_node=1` runs on a single GPU; set it to your GPU count
  (e.g. `--nproc_per_node=8`) to go faster. The output is identical to a single-GPU run — work is
  split round-robin across ranks, not duplicated.
- **Images can be fewer than testcase rows.** That is the content-safety guardrail, not a failure —
  check `guardrail_blocked.csv` before treating a shortfall as a bug.
- **⚠ Small `/dev/shm`.** If the run aborts with a `/torch_*` shared-memory error, prepend
  `PYTHONPATH="$PWD/shmpatch"` to the command. It stops the dataloader creating `/dev/shm/torch_*`
  files at all; passing `--num_workers 1` is a further fallback.

## Next step

Continue to [6 · Evaluation, Refinement & Pseudo-labeling](6-evaluation-and-refinement.md).
