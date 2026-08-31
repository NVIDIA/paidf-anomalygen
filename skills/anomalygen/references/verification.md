# Verification reference — per-step gates

Progressive-disclosure detail for the **Verification** section in the parent `SKILL.md`: one gate per
step, named after the step (`dataset`, `amp`, `train`, `generate`, `eval`, `refine`, `label`) — run it
before treating that step as done. If this file conflicts with `SKILL.md`, `SKILL.md` wins.

Uses the shared vars set at the top of the SKILL.md step list (`${DATASET_DIR}`, `${GEN_DIR}`, `${OUT}`,
`${NAME}`, `${TASK}`, `${NUM_SDG}`, `${NUM_SEARCH_RUN}`, `${AMP_DIR}`, `${RUN_DIR}`).

## `dataset` — contract layout on disk (Step 1)

Before anything else, `${DATASET_DIR}` must hold `defect_spec.jsonl` plus, per `[texture, defect]` pair,
`{texture}/clean_image/`, `{texture}/anomaly_image/{defect}/`, and `{texture}/mask/{defect}/` (each mask
is the anomaly's stem + `_mask`). `cad` defects also need `{texture}/cad_mask/<clean_stem>.png` — one
per clean image, no `{defect}` level — plus `semantic_segmentation_labels.json`.

```shell
test -f "${DATASET_DIR}/defect_spec.jsonl" \
  && ls -d "${DATASET_DIR}"/*/clean_image "${DATASET_DIR}"/*/anomaly_image/*/ "${DATASET_DIR}"/*/mask/*/ >/dev/null 2>&1 \
  && echo OK || echo "dataset not in contract layout"
```

**The directories existing is not enough — check the pairing too.** The loader pairs each anomaly image
with `{stem}_mask` *in the image's own extension, falling back to `.png`*, and an image whose mask is
missing is **silently skipped**, not reported. So a typo'd or absent mask quietly shrinks the training
and reference sets, and you only get a hard error when *every* mask for a defect is missing:

```shell
miss=0; seen=0
for img in "${DATASET_DIR}"/*/anomaly_image/*/*; do
  # Also drops the pattern itself when nothing matched: it ends in `*`, so it fails this test.
  case "${img,,}" in *.png|*.jpg|*.jpeg) ;; *) continue ;; esac
  seen=$((seen + 1))
  tex="${img%/anomaly_image/*}"; defect="${img#*/anomaly_image/}"; defect="${defect%%/*}"
  base="$(basename "${img}")"; ext="${base##*.}"; stem="${base%.*}"
  [ -f "${tex}/mask/${defect}/${stem}_mask.${ext}" ] || [ -f "${tex}/mask/${defect}/${stem}_mask.png" ] \
    || { echo "unpaired: ${img}"; miss=$((miss + 1)); }
done
# `seen` is load-bearing: without it a wrong DATASET_DIR sweeps nothing and still reports OK.
if [ "${seen}" = 0 ]; then echo "no anomaly images under ${DATASET_DIR} — wrong path or empty dataset"
elif [ "${miss}" = 0 ]; then echo "pairing OK (${seen} images)"
else echo "${miss}/${seen} anomaly image(s) with no mask — fix before training"; fi
```

A missing dir or an unpaired `_mask` will otherwise surface as empty AMP placements or a
training/scoring error later — fix the layout now.

## `amp` — testcase built (Step 2)

`roi_allocate`+`roi_pair` alone (just `allocation.json`+`amp_samples.json`) are **not** a finished AMP —
`roi_place` must run and write `testcase.jsonl`; `roi_place_pipeline` stops with the offending stage
named if one does not. The **generation** set must then match the row count the allocation asked for:

```shell
EXPECT="${NUM_SDG}"   # with --per_defect_counts: the SUM of that dict instead, which is authoritative
[ "$(wc -l < "${GEN_DIR}/testcase.jsonl")" = "${EXPECT}" ] && echo OK || echo "AMP count != ${EXPECT}"
```

- **Plain `--num_sdg`** → uniform allocation sums to *exactly* `num_sdg`, so `EXPECT=${NUM_SDG}`.
- **With `--per_defect_counts`** → the override sum wins by design and a differing total is **correct,
  not a failure** (see `mask-placement.md`); set `EXPECT` to that sum before
  judging the gate.
- **Fewer rows than `EXPECT`** → a defect lost placements (see *AMP short of allocation* in Common
  Pitfalls); check the `roi_place` logs and fix before generating.
- **More rows than `EXPECT`** → most often **stale placed masks from an earlier AMP run** in a reused
  `--output_dir`: `roi_place` globs `*__seed*.png` back up, so leftovers from a previous call are
  counted as this one's. Clear `${AMP_DIR}` and re-run rather than adjusting `EXPECT`. Note rows =
  the sum of each record's `n_seeds` (one row per placed mask), which is exactly what
  `allocation.json` sums to — so a mismatch is a stale-file or wrong-call problem, not an arithmetic
  one (see `mask-placement.md`).

## `train` — training finished (Step 3)

Reached `trainer.max_iter` (not just an early `save_iter` checkpoint), and the best-checkpoint pointer
written at train end exists — it names the iteration Steps 4–7 generate from. Gate Step 4 on
`train_control.sh wait` returning **0**, not on the first checkpoint appearing — see
`fine-tuning.md`.

`Done with training` is logged only at `max_iter`, so it already implies the final checkpoint:

```shell
grep -q 'Done with training' "results/train_${NAME}.log" \
  && [ -s "${RUN_DIR}/checkpoints/best_checkpoint.txt" ] \
  && echo "train gate PASS: best = $(cat "${RUN_DIR}/checkpoints/best_checkpoint.txt")" \
  || echo "train gate FAIL"
```

A missing pointer after a completed run means validation never scored (check `testcase_jsonl`) or the
callback warned — see the log. Fix the run; do **not** fall back to `latest_checkpoint.txt`, which names
the last iteration rather than the best.

## `generate` — generation complete and well-formed (Step 4)

`generate.py` emits **one image per testcase row** and never drops by quality (`PSNR` is recorded in the
CSV, never a filter). The **one** legitimate shortfall is the content-safety guardrail, on by default:
a blocked caption or composite writes no image and is logged to `guardrail_blocked.csv` instead. So
images + blocked must equal the rows, and the CSV must carry the canonical header:

```shell
BLOCKED=$([ -f "${OUT}/guardrail_blocked.csv" ] && echo $(( $(wc -l < "${OUT}/guardrail_blocked.csv") - 1 )) || echo 0)
head -1 "${OUT}/${TASK}_generation_result.csv" | grep -q '^output_filename,image_filename,mask_filename,anomaly_type' \
  && [ "$(( $(ls "${OUT}"/reconstructed_image/*.png | wc -l) + BLOCKED ))" = "$(wc -l < "${GEN_DIR}/testcase.jsonl")" ] \
  && echo "OK (${BLOCKED} guardrail-blocked)" || echo "generation partial/malformed — re-run Step 4"
```

`guardrail_blocked.csv` is written whenever the guardrail ran, header-only when it blocked nothing, and
not at all under `--no-guardrail` — so an absent file is zero either way. A **nonzero** count is not a
failure: it is the guardrail doing its job, and those rows are simply absent from the batch. Read the
`message` column before re-running, since a re-run reproduces the same block.

A shortfall the blocked count does *not* explain, or a non-canonical header, means Step 4 didn't finish
— **re-run it; never hand-write or patch the CSV** (a malformed `${TASK}_generation_result.csv` crashes
Step 6 `quality_refine draw` with `KeyError: 'output_filename'`).

## `eval` — KPI computed (Step 5)

`${OUT}/${NAME}_kpi.json` exists with a per-type block and an `Average` block carrying
`nn_score`/`mnn_score`(/`fid`). `nn_score` is the primary KPI.

## `refine` — every round actually finished (Step 6)

Unless `num_search_run=0`, Step 6 runs the draw→generate→evaluate loop before `select`. Catch two
failures: no rounds at all (clone-only), and — subtler — a round that started but didn't finish.

**Existing round directories are not enough.** A round dir appears with `generate.py`'s first image, so
an interrupted round looks fine; `select` then **silently ignores** it for want of a `kpi.json`. A round
can also be complete on files *and* counts yet still be unrankable — see the `${SCORE}` check below.
Run this after each round, and again before `select`:

```shell
SCORE=nn        # must match --score on the Step 6 command; nn is the default
fail=0
rankable() {    # rankable <kpi.json> <key> -> True when any sample carries a non-NaN value
  python3 -c '
import json, math, sys
def ok(v):
    try: return not math.isnan(float(v))
    except (TypeError, ValueError): return False
d = json.load(open(sys.argv[1]))
print(any(ok(row.get(sys.argv[2])) for k, v in d.items()
          if k != "Average" and isinstance(v, dict) for row in v.get("per_sample", [])))
' "$1" "$2" 2>/dev/null || echo False
}

# The ORIGINAL bucket competes against the rounds, so it must carry ${SCORE}_score too. Unranked, it
# forfeits every pick — a round wins each sample even when strictly worse, and the run still exits 0.
[ "$(rankable "${OUT}/${NAME}_kpi.json" "${SCORE}_score")" = True ] || {
  echo "FAIL original: no usable ${SCORE}_score in ${OUT}/${NAME}_kpi.json"; fail=1; }

n=$(ls -d "${OUT}"/rounds/round_* 2>/dev/null | wc -l)
[ "$n" = "${NUM_SEARCH_RUN}" ] || { echo "only ${n}/${NUM_SEARCH_RUN} round dirs"; fail=1; }
for r in "${OUT}"/rounds/round_*/; do
  img=$(ls "$r"reconstructed_image/*.png 2>/dev/null | wc -l)
  row=$(wc -l < "$r"testcase.jsonl 2>/dev/null || echo 0)
  # Rounds run the same guardrail-enabled generate.py as Step 4, so a blocked sample writes no image
  # by design. Counting images against rows alone fails a correct round, and the fix below (re-run
  # generate.py) reproduces the same block — the loop would never terminate.
  blk=$([ -f "$r"guardrail_blocked.csv ] && echo $(( $(wc -l < "$r"guardrail_blocked.csv) - 1 )) || echo 0)
  # Can this round actually be ranked? evaluate.py degrades to NN/MNN(+FID) whenever anomaly_quality
  # raises, so with an aq_* metric the rows arrive complete but carry no ${SCORE}_score at all.
  scored=$(rankable "$r"kpi.json "${SCORE}_score")
  if [ -f "$r"kpi.json ] && [ "$(( img + blk ))" = "$row" ] && [ "$img" != 0 ] && [ "$scored" = True ]; then
    echo "OK   $(basename $r): ${img}+${blk}/${row} images, kpi.json present, ${SCORE}_score usable"
  else
    echo "FAIL $(basename $r): ${img}+${blk}/${row} images, kpi.json $([ -f "$r"kpi.json ] && echo present || echo MISSING), ${SCORE}_score usable=${scored}"
    fail=1
  fi
done
[ "$fail" = 0 ] && echo "refine gate PASS" || echo "refine gate FAIL — fix every FAIL line above before select"
```

- **Short image count (beyond what `guardrail_blocked.csv` explains) or missing `kpi.json`** → re-run that round's `generate.py` (it overwrites) and its
  `evaluate.py`, then re-run `select` and Step 7. Do **not** accept the existing `searched/`: it was
  assembled without that round.
- **Fewer round dirs than `${NUM_SEARCH_RUN}`** → rounds were skipped entirely.
- **`${SCORE}_score usable=False`** → the round rendered fine but cannot be ranked, so re-running
  `generate.py` fixes nothing. Only an `aq_*` metric hits this: `evaluate.py` degrades to
  NN/MNN(/FID) whenever `anomaly_quality` raises, and a **missing SAM2 checkpoint** at
  `checkpoints/facebook/sam2.1-hiera-large/` is the usual cause. Grep that round's evaluate log for
  `anomaly_quality computation failed`, fix the checkpoint, and re-run **`evaluate.py` only**. Left
  unchecked, every sample scores NaN, the original wins each pick, and the search burns its rounds
  to reproduce Step 4 exactly. `Step 6` itself now stops on this rather than exiting 0.
- **`FAIL original: no usable ${SCORE}_score`** → the *original* bucket cannot be ranked, so it is
  dropped from the competition and the best round wins every sample **with no magnitude comparison** —
  the run exits 0 reporting `improved over original by search: N/N` and `search_summary.csv` carries an
  empty `original_score` on every row. Same causes as above, plus a `${NAME}_kpi.json` scored before
  `--score` accepted the `aq_*` metrics. Re-run **Step 5's `evaluate.py`** so the KPI carries
  `${SCORE}_score`, then re-run Step 6. `quality_refine.py` now refuses this before the first round
  rather than after the last, so no GPU time is spent on a search that was already decided.

Also confirm `${OUT}/searched/${NAME}_kpi.json` exists (the closing evaluation).

An all-`original` `source` column in `${OUT}/searched/${TASK}_generation_result.csv` is **not** a
failure on its own — it is the honest result when no round beat the Step 4 render. Read it together
with the round check above: round dirs present *and* all-`original` means the search ran and found
nothing (accept it, and consider wider ranges or more rounds); **no round dirs** at a nonzero
`${NUM_SEARCH_RUN}` is the actual failure, and that is what the loop above catches.

## `label` — pseudo-labels written (Step 7)

`${OUT}/searched/pseudo_labels/coco_annotations.json` + `classification/classes.txt` exist.
