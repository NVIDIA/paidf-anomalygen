# Quality-refinement reference — Step 6 (quality_refine)

Progressive-disclosure detail for **Step 6 — quality refinement** in the parent `SKILL.md`.
Read this for the draw strategy (uniform vs. Bayesian optimization), the ranges/seeding, and `select`.
If this file conflicts with `SKILL.md`, `SKILL.md` wins.

## What the step does

For each generation sample, search a better `(guidance, crop_ratio)` than the Step 4 default. A round =
**draw** a candidate per sample → **generate** → **evaluate**; after `num_search_run` rounds, **select**
keeps each sample's best-scoring render across the original bucket + every round into `searched/`.

`quality_refine` has three subcommands. **`run`** is the one `SKILL.md` Step 6 calls — it drives the
whole loop and gates each round. `draw` (write one round's `testcase.jsonl`) and `select` (assemble
`searched/`) are the pieces it composes, and stay separately callable for re-running a single round by
hand. **`select` always runs** — with 0 rounds it clones the original bucket, so `searched/` always
exists for Step 7.

> **"select ran" ≠ "refinement ran."** The default is `num_search_run=3`: run the
> draw→generate→evaluate loop **3 times before** `select`. Running only `select` (clone-only) is correct
> **only** if you deliberately chose `num_search_run=0` — otherwise you skipped refinement and gain no
> per-sample quality lift. Tell-tale of a skipped Step 6: no `${OUT}/rounds/round_*/`, and `searched/`'s
> `source` column is entirely `original`.

## `draw` — how each round is proposed

Per sample:

- **Round 1 (or no history): uniform.** With only the original observation so far, `draw` samples
  `(guidance, crop_ratio)` uniformly from the ranges.
- **Rounds 2+: per-sample Bayesian optimization.** When `--original`/`--original_kpi` (+ `--rounds_dir`)
  supply history, `draw` fits an **independent GP** on *that sample's own* `(guidance, crop_ratio) →
  score` observations (inputs scaled to `[0,1]²` by the ranges; Matérn + white-noise kernel) and
  **Thompson-samples** one point from `--bo_candidates` (default 1024) candidates.
- A sample falls back to **uniform** if it has fewer than `--bo_min_obs` (default 2) observations, or
  fewer than 2 *distinct* `(guidance, crop_ratio)` points (a GP needs spread to fit).

Key flags:

| Flag                            | Default    | Note                                                                                  |
| ------------------------------- | ---------- | ------------------------------------------------------------------------------------- |
| `--base_testcase`               | —          | the generation `testcase.jsonl` to redraw                                             |
| `--output`                      | —          | this round's `testcase.jsonl`                                                         |
| `--seed`                        | `42`       | vary per round (pass the round number) for reproducible draws                         |
| `--guidance_range`              | `1.5 10.0` | draw range for guidance                                                               |
| `--crop_ratio_range`            | `1.5 8.0`  | draw range for crop_ratio                                                             |
| `--original` / `--original_kpi` | off        | original bucket + its `kpi.json` — **enables BO**                                     |
| `--rounds_dir`                  | off        | prior `round_<r>/` buckets (each with a `kpi.json`) pooled into each sample's history |
| `--score`                       | `nn`       | per-sample metric to optimize (see the choices below)                                 |
| `--bo_min_obs`                  | `2`        | min observations before a sample uses BO instead of uniform                           |

Without `--original`/`--rounds_dir`, every sample draws uniformly (a plain random round).

`--score` accepts `nn`, `mnn`, `completeness`, `precision`, `boundary_iou` or `aq_nn` — every
per-sample metric `evaluate.py` writes, so each round's `kpi.json` already carries the chosen one.

> **`aq_rank` is `filter.py`-only.** It is a rank-relative composite computed across a whole bucket,
> so a sample's value moves when its neighbours change — not a fixed target a per-sample optimizer
> can climb. `quality_refine` rejects it at argparse.

## `run` — the round loop

For `r` in `1..num_search_run` (default 3): `draw` → `generate.py` (same `--checkpoint`/`--recipe` as
Step 4) → `evaluate.py` writing `round_<r>/kpi.json`, then `select` and a final `evaluate.py` on
`searched/`. Each round's `kpi.json` feeds the next round's BO history via `--rounds_dir`, and each
round launches its own `torchrun --standalone` so back-to-back rounds never contend for a fixed
rendezvous port.

> **A round counts only with one image per testcase row AND its `kpi.json`.** Both consumers key off
> `kpi.json` — `select` scores buckets from it, `draw` pools it into the next round's BO history — so a
> round that stopped early would be **silently dropped from both**: GPU time wasted, and later rounds
> searching with less history. `run` checks this after every round and **exits 1** rather than
> assembling a stale `searched/`; it names what was short. Repair that round in place — re-run its
> `generate.py` (it overwrites) and `evaluate.py` — then finish with `select` plus the final
> `evaluate.py`. Do **not** re-run `run`: it has no resume, so it would redraw and regenerate every
> round from 1, discarding the ones that already succeeded.

`run` takes `draw`'s tuning flags (`--guidance_range`, `--crop_ratio_range`, `--score`, `--bo_min_obs`,
`--bo_candidates`) and forwards them to every round; the per-round `--seed` is the round number. Use
`--dry_run` to print each stage's command without running it.

## `select` — assembling `searched/`

`select` reads the original bucket + all `round_<r>/` buckets and, per sample index, keeps the render
with the best `--score` (`nn` by default) into `searched/`. `searched/` is what Step 7 pseudo-labels.

`${TASK}_generation_result.csv` gains four columns: `nn_score` and `mnn_score` (the winner's *actual*
metrics, always under their own names regardless of `--score`), `selected_by` (which metric decided it —
`nn_score` or `mnn_score`), and `source` (which bucket the winner came from). `search_summary.csv` is
the per-sample audit trail: `source`, `selected_by`, `score`, `original_score`, `improved`.

The `source` distribution is the readout on the search — how many samples came from `original` (no
improvement found) vs. each round. All-`original` means refinement bought nothing.

## Scoring the result

`select` does not evaluate what it assembled, so run `evaluate.py` on `searched/` (Step 6's closing
command) and compare `Average.nn_score` against the Step 5 `${OUT}/${NAME}_kpi.json`. That delta — not
any individual round's KPI — is what refinement was worth. Expect the refined batch to beat **every**
round: per-sample selection composes the best render across buckets, so `searched/` can exceed rounds
that each scored below the Step 5 batch. Judging the step by whether some round beat the original
average understates it.

## Notes

- BO is **per sample and independent** — there is no shared/global surrogate; each sample explores its
  own response surface, so a hard sample keeps searching while an easy one settles.
- Round 1's log line reads `mode=uniform (no sample had enough history)` even with `--original` passed:
  one observation per sample isn't enough to fit a GP, so round 1 explores blind by design. From round 2
  the line reports the real split (`bayesopt <n>/<total> samples`).
- `select` only ever *keeps* the best render across the original + rounds, so a sample never lands below
  its Step 4 result — **with one exception**: if a sample is unscored in the original bucket (a `NaN`
  score in `kpi.json`), any scored round wins it by default, since NaN sorts as worst. Check for empty
  `original_score` cells in `search_summary.csv` if a `source` distribution looks surprising.
