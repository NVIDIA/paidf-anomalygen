# Inference Reference — Phases 2–7

Navigator for `anomalygen` Phases 2–7. Each phase has a canonical reference;
this file holds the cross-phase glue and routes to the right place.

| Phase | Canonical reference |
|---|---|
| Pre-flight env check | `references/finetune.md` §Environment check |
| Phase 2 (prep-testcase) | `references/prep-testcase.md` |
| Phase 3 (SDG) | `references/sdg-inference.md` |
| Phase 4 / 6 eval (`run_eval.sh`) | `references/eval.md` |
| Phase 5 search (draws.json, re-AMP) | `references/sdg-refine.md` |
| Phase 6 assemble | this file |
| Phase 7 filter + regen | this file |

## Contents
- [Pre-flight checkpoint validation](#pre-flight-checkpoint-validation)
- [Phase 2: cross-phase notes](#phase-2-cross-phase-notes)
- [Phase 3: cross-phase notes](#phase-3-cross-phase-notes)
- [Eval pre-flight (Phases 4 / round / 7)](#eval-pre-flight-phases-4--round--7)
- [Phase 5: search strategy](#phase-5-search-strategy)
- [Phase 6: Assemble (stitch only)](#phase-6-assemble-searched-stitch-only)
- [Phase 7: Filter + Regen + Eval](#phase-7-filter--regen--eval)

---

## Pre-flight checkpoint validation

Run before Phase 3. `validate_checkpoint.py` ensures `ag_config.yaml` exists
and is well-formed, and prints the supported `TEXTURE+ANOMALY` set — use it
as `DEFECTS` if not already set from `defect_spec`.

```python
# What validate_checkpoint.py reads:
cfg["dataloader_train"]["dataset"]["anomaly_types"]
# Each entry is [texture, anomaly] → rendered as "texture+anomaly"
```

Stop if `ag_config.yaml` is missing — ask the user to verify `checkpoint_dir`.

For the GPU / CUDA environment check, see `references/finetune.md`
§Environment check (same check used across all phases).

---

## Phase 2: cross-phase notes

For the full parameter table, helper script descriptions, two calling
conventions (validation vs inference), AMP routing table, submask handling,
post-run verification, and error diagnosis, read **`references/prep-testcase.md`**.

Cross-phase notes specific to inference-mode invocation:

### Clean image discovery

`prep_testcase.sh` probes these layouts in order:
1. `<clean_dir>/<TEXTURE>/clean_image/*`
2. `<clean_dir>/<TEXTURE>/*`
3. flat `<clean_dir>/*`

When clean images live at `<dataset_dir>/<TEXTURE>/clean_image/`, omit
`--clean-dir` (defaults to `dataset_dir`).

### Pairing strategy

Budget per defect = `num_submasks[d] × num_cleans[texture]` (every submask
× clean combination). Pairing iterates every combination once (deterministic
shuffle per defect) before any pair repeats.

JSONL defaults: `guidance=7.0`, `crop_ratio=2.0`, `seed=42` — overridden
by Phase 5 search.

### n_seeds sizing

Auto-computed from allocation — do NOT pass `--seeds`:

```
n_seeds = max_d ⌈allocation[d] / (num_submasks[d] × num_cleans[texture])⌉
```

n_seeds > 1 only when allocation exceeds the pair budget.

### Phase 2 errors (cross-phase)

- Validator failure → stop with itemised report (missing submask/clean/cad/prompt).
- AMP output count < allocation → `build_jsonl.py` errors (check `run_auto_roi_amp.py` logs for NO_DETECTION / FAILED).
- Mask-size mismatch → `verify_jsonl.py` auto-resizes into `resized_masks/` cache.

---

## Phase 3: cross-phase notes

For the full parameter table, step-by-step detail, `run_sdg.sh` flags, NCCL
controls, output verification, and phase-3-specific error handling, read
**`references/sdg-inference.md`**.

### JSONL validation against checkpoint

`validate_jsonl.py` cross-checks anomaly types in the JSONL against the
checkpoint's supported set. If any are unsupported, stop and show the mismatch:
```
Supported:  TEXTURE+TYPE_A, TEXTURE+TYPE_B
JSONL has:  TEXTURE+TYPE_C  ← unsupported
```
Options: retrain with extended defect set, use an isolated model for the new
defect, or trim `defect_spec` to supported types.

If many image/mask paths are missing → stop; ask the user to verify paths
before burning GPU time.

### Multi-GPU caveats

| Config | Behavior |
|---|---|
| 14B + `num_gpus > 1` | FSDP auto-disabled at inference; each rank holds the full 14B model (~80 GB VRAM per GPU) |
| 14B + single GPU | FSDP enabled; fits on smaller GPUs |
| 2B + any num_gpus | Standard DDP; no special VRAM constraint |

### SDG output structure

```
<output_dir>/
├── reconstructed_image/   # final synthetic anomaly images
├── annotated_image/
├── cropped_image/
├── cropped_mask/
├── original_image/
├── original_mask/
└── SDG_result.csv
```

---

## Eval pre-flight (Phases 4 / round / 7)

For `run_eval.sh` flags, score interpretation (`nn_score`, `mnn_score`, FID),
feature-count semantics, and eval-specific error handling, read
**`references/eval.md`**.

Check before invoking `run_eval.sh`:
1. `reconstructed_image/` exists and is non-empty.
2. No SDG process is still writing to the output directory.
3. Auto-detect anomaly types from generated filenames if not supplied:
   ```bash
   ls <generated_path>/reconstructed_image/ | sed 's/_[0-9]*\.\(png\|jpg\)$//' | sort -u
   ```

---

## Phase 5: search strategy

For the full inputs table, `draws.json` format and alignment to
`SDG_result.csv`, re-AMP mechanics, and `num_search_run = 0` semantics,
read **`references/sdg-refine.md`**.

### Claude's draw strategy

For each round `r`:
1. Read `per_sample.csv` from round `r-1` (or `original/per_sample.csv` for `r=1`).
2. For each sample, pick new `(guidance, crop_ratio)`. Focus search budget on
   low-scoring samples. Skip samples already scoring well to save inference time.
3. Write `draws.json` to `${ROUNDS}/round_${r}/draws.json`.

Ranges to consider: `guidance ∈ [1.5, 10.0]`, `crop_ratio ∈ [1.5, 10.0]`.
Narrow or widen based on prior-round feedback.

`run_round.sh` produces `rounds/round_r/testcase.jsonl`, `sdg/`, and
`per_sample.csv`. The `sdg/SDG_result.csv.index` column carries the
base-JSONL `sample_index`, keeping `assemble_searched.py` aligned across rounds.

### search_summary.csv

After Phase 6 assemble: `rounds_dir/search_summary.csv` has one row per
sample with `best_round`, `best_guidance`, `best_crop_ratio`, `best_nn_score`,
`attempts`.

---

## Phase 6: Assemble `searched/` (stitch only)

Runs `assemble_searched.py` to pick best-of-rounds (or clone `original/`
when `num_search_run=0`) into `searched/`. **No eval runs here.** The
script copies each picked sample's images into `searched/` and stitches
`searched/per_sample.csv` by carrying over per-sample `nn_score` and
`mnn_score` from the picked sample's source round `per_sample.csv` —
correspondence-to-real-set scoring is per-sample-independent (one
generated image against the real set, no sibling-generation coupling),
so the stitched values are exact, not approximate. The same `nn_score`
gets merged into `searched/SDG_result.csv`. Phase 7 owns the canonical
post-pipeline eval against the final regen-aware bucket.

---

## Phase 7: Filter + Regen + Eval

Runs by default (`nn_threshold=0.4`). Set `nn_threshold=0` to disable.

Updates `searched/` **in place** — downstream always reads `searched/`
as the final SDG bucket regardless of whether Phase 7 ran or whether
`num_search_run` was 0.

`filter_with_regen.py` orchestrates a **re-AMP + re-pair** regen flow:

1. **Initial filter** — partition source bucket into `passing_per_defect`
   and `dropped_per_defect`. Read target allocation (per-defect count)
   from the source bucket.
2. **Regen loop** — for each attempt up to **5**:
   * Compute `needed_per_defect = target_alloc - kept_per_defect`. If
     zero everywhere, stop.
   * Write a subset `allocation.json` for the still-needed defects.
   * Run `build_amp_samples.py --seed=attempt_seed` — per-defect
     `(clean, submask)` lists get shuffled before pairing, so each
     attempt's pairings are distinct.
   * Run `run_auto_roi_amp.py --seed=attempt_seed` (new placement).
   * Run `build_jsonl.py`, then overwrite each row's `seed` field with
     `attempt_seed` (so diffusion noise via `AnomalyInpaintCondition.seed`
     → `misc.arch_invariant_rand` also varies).
   * Run SDG into `regens/regen_NN/sdg/`, eval.
   * Admit new samples scoring ≥ `threshold` into `admitted_regens_by_defect`,
     greedy by nn descending up to the defect's quota.
3. **Per-defect fallback fill** — if a defect is still short of its
   target, top up with the best non-admitted regens for that defect,
   then with the highest-scoring dropped originals (last resort).
4. **Atomic-ish in-place swap** — stage to `searched.staging/`, rename
   over `searched/`. Atomic on same filesystem.

`regens/` (sibling of `rounds/`) holds Phase 7 artifacts:
`regens/regen_NN/{allocation.json, amp_samples.json, amp/, testcase.jsonl,
sdg/}` per attempt, plus `regens/regen_summary.csv` with columns
`sample_index` (`-1` for regen), `source`, `clean_image`, `mask_filename`,
`prev_nn`, `nn_score`, `passed_threshold`, `output_filename`.

For at-a-glance tracing, `searched/SDG_result.csv` carries a `source`
column:

- `original` — survived Phase 5 assemble straight from Phase 3 SDG
- `round_<N>` — Phase 5 search round `N` produced this sample's best attempt
- `regen_<k>` — Phase 7 regen attempt `k` produced this sample

`searched/SDG_result.csv` alone suffices for most tracing (it has
`image_filename`, `mask_filename`, `nn_score`, `source`). `regen_summary.csv`
adds `prev_nn` and `passed_threshold` for deeper audit.

Per-attempt eval is invoked with only the anomaly types present in that
attempt's subset, suppressing harmless "No generated defect patches"
warnings for types absent from the subset.
