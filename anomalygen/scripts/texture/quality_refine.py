# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-sample ``(guidance, crop_ratio)`` search — three subcommands.

``draw``   writes one search round's ``testcase.jsonl`` by re-drawing ``(guidance, crop_ratio)`` for
           **every** sample of the base testcase, keeping each mask placement fixed. Round 1 draws
           uniformly; given prior buckets (``--original`` / ``--rounds_dir``), rounds 2+ use per-sample
           Bayesian optimization — an independent GP over each sample's own observed
           ``(guidance, crop_ratio) -> score`` history, Thompson-sampled. Draws are seeded (``--seed``)
           so a round is reproducible.

``select`` picks the best-scoring render per sample across the original bucket and all rounds into a
           ``searched/`` bucket. Always runnable — with zero rounds every winner is the original, so
           ``searched/`` simply clones it and is guaranteed to exist for the downstream
           filter/regeneration + pseudo-label steps.

``run``    drives the whole loop: ``(draw -> generate -> evaluate) x --num_search_run``, then
           ``select`` and a closing ``evaluate`` on ``searched/``. It gates each round (one image per
           testcase row *and* its ``kpi.json``) and exits non-zero rather than assembling a stale
           ``searched/``. The scoring knobs it is given are forwarded to every round's ``evaluate``,
           so each round is scored the same way as the batch it is selected against.

``draw``'s Bayesian optimization is the only part that *computes* with numpy / scipy / scikit-learn —
``select`` and the uniform draw path are pure stdlib logic — but all three are imported at module load,
so every subcommand needs them installed. Only scikit-learn is pinned in ``requirements.txt``; scipy
arrives as its dependency. ``anomalygen.eval.correspondence`` is imported at module load too, for the
``add_nn_scoring_args`` declaration ``run`` forwards to each round's ``evaluate.py``; it pulls in torch
and the DINOv2 backbone spec, so every subcommand needs those on the path as well. ``draw`` reads the
base ``testcase.jsonl`` (the AMP output that produced the generation bucket); ``select`` reads
precomputed ``kpi.json`` files plus each bucket's generation CSV and copies the winning renders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from cosmos_framework.utils import log
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from anomalygen.eval.correspondence import add_nn_scoring_args

_GEN_CSV = "texture_ft_generation_result.csv"

# Per-sample metrics the search can optimize and select on — every one of them is on the per_sample
# rows ``evaluate.py`` writes for each round. ``aq_rank`` is deliberately absent: it is a rank-relative
# composite computed inside filter.py across a whole bucket, so a sample's value moves when its
# neighbours change — not a fixed target a per-sample optimizer can climb.
_SCORE_CHOICES = ["nn", "mnn", "completeness", "precision", "boundary_iou", "aq_nn"]

# Scoring knobs forwarded verbatim to every round's evaluate.py. They all change the computed numbers,
# so a round scored without them is not comparable to the Step 5 KPI it is selected against. The nn/mnn
# knobs are declared by ``add_nn_scoring_args`` itself, so their names and defaults live in one place.
_EVAL_SCORING_FLAGS = ("top_k", "model_input_size", "nn_layer", "nn_readout", "nn_region_policy", "nn_inst_agg")


_SINGLE_KINDS = ("reconstructed_image", "original_image", "original_mask")
_MULTI_KINDS = ("annotated_image",)
_ROUND_DIGITS = 2  # decimal places for emitted guidance / crop_ratio (e.g. 1.23456 -> 1.23)


# --- draw: write one search round's testcase --------------------------------------------------


def _load_base_rows(base_testcase):
    """All rows from the base ``testcase.jsonl`` (blank lines skipped)."""
    rows = []
    with open(base_testcase) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _draw_rows(rows, rng, guidance_range, crop_ratio_range):
    """Copy every row and overwrite ``(guidance, crop_ratio)`` with a fresh uniform draw; every other
    field (the ``(clean, mask)`` pairing, seed, num_steps, ...) is preserved verbatim."""
    gl, gh = guidance_range
    cl, ch = crop_ratio_range
    drawn = []
    for row in rows:
        new = dict(row)
        new["guidance"] = round(rng.uniform(gl, gh), _ROUND_DIGITS)
        new["crop_ratio"] = round(rng.uniform(cl, ch), _ROUND_DIGITS)
        drawn.append(new)
    return drawn


# --- draw (BO): reconstruct prior observations ------------------------------------------------


def _numeric(v):
    """Float value, or None for missing / "none" / NaN (so invalid rows drop out of the surrogate)."""
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "none")):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _round_dirs(rounds_dir):
    """``round_<r>/`` sub-buckets ordered by round *number*, not name.

    Plain lexicographic sorting puts round_10 before round_2, which silently reorders the
    tie-break priority in ``_pick_best`` (and the select log) once a search exceeds 9 rounds."""
    if not rounds_dir or not Path(rounds_dir).is_dir():
        return []

    def _key(p):
        suffix = p.name[len("round_") :]
        return (0, int(suffix)) if suffix.isdigit() else (1, 0, p.name)

    return sorted((p for p in Path(rounds_dir).glob("round_*") if p.is_dir()), key=_key)


def _bucket_observations(bucket_dir, kpi_path, score_key):
    """One (index, guidance, crop_ratio, score) per sample index in a single bucket.

    Joins the bucket CSV to its kpi.json by output basename; keeps the best (max) score per index
    (a sample with num_generated_images>1 has several same-(g,c) outputs); drops rows whose score is
    missing/NaN or whose guidance/crop_ratio is non-numeric (e.g. a fixed-grid crop_ratio=="none")."""
    scores = _load_scores(kpi_path, score_key)  # {basename: score}
    best = {}  # index -> (g, c, score)
    with (Path(bucket_dir) / _GEN_CSV).open(newline="") as f:
        for row in csv.DictReader(f):
            y = _numeric(scores.get(os.path.basename(row["output_filename"])))
            g = _numeric(row.get("guidance"))
            c = _numeric(row.get("crop_ratio"))
            if y is None or g is None or c is None:
                continue
            idx = int(row["index"])
            if idx not in best or y > best[idx][2]:
                best[idx] = (g, c, y)
    return [(idx, g, c, y) for idx, (g, c, y) in best.items()]


def _load_observations(original, original_kpi, rounds_dir, score_key, kpi_name):
    """{sample_index: [(guidance, crop_ratio, score), ...]} over the original bucket and every existing
    round_* bucket (each carrying both a CSV and a kpi.json). Keyed by sample index so every sample gets
    its own surrogate."""
    buckets = [(original, original_kpi)]
    for round_dir in _round_dirs(rounds_dir):
        kpi = round_dir / kpi_name
        if (round_dir / _GEN_CSV).is_file() and kpi.is_file():
            buckets.append((round_dir, kpi))
    obs = defaultdict(list)
    for bucket_dir, kpi_path in buckets:
        for idx, g, c, y in _bucket_observations(bucket_dir, kpi_path, score_key):
            obs[idx].append((g, c, y))
    return obs


# --- draw (BO): per-sample Gaussian-process proposal ------------------------------------------


def _distinct_xy(observations):
    """Number of distinct (guidance, crop_ratio) locations among observations, at emitted precision."""
    return len({(round(g, _ROUND_DIGITS), round(c, _ROUND_DIGITS)) for g, c, _ in observations})


def _bo_propose(observations, n, rng_seed, guidance_range, crop_ratio_range, n_candidates):
    """Thompson-sample n (guidance, crop_ratio) proposals from a GP fit on ``observations``.

    Fits a GP on the observations ([(g, c, y), ...], inputs scaled to [0,1]^2 by the ranges), then draws
    n independent posterior realizations over an LHS candidate pool and returns each realization's
    argmax. Deterministic for a fixed rng_seed."""
    gl, gh = guidance_range
    cl, ch = crop_ratio_range

    def _scale(g, c):
        return [(g - gl) / (gh - gl), (c - cl) / (ch - cl)]

    X = np.array([_scale(g, c) for g, c, _ in observations], dtype=float)
    y = np.array([yy for *_, yy in observations], dtype=float)

    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=[0.2, 0.2], length_scale_bounds=(1e-2, 1e1), nu=2.5
    ) + WhiteKernel(1e-6, (1e-8, 1e-2))
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, alpha=1e-6, n_restarts_optimizer=2, random_state=rng_seed
    )
    gp.fit(X, y)

    cand = LatinHypercube(d=2, seed=rng_seed).random(n_candidates)  # (n_candidates, 2) in [0,1)
    draws = gp.sample_y(cand, n_samples=n, random_state=rng_seed)  # (n_candidates, n)
    picks = cand[np.argmax(draws, axis=0)]  # (n, 2)
    return [
        (round(gl + px[0] * (gh - gl), _ROUND_DIGITS), round(cl + px[1] * (ch - cl), _ROUND_DIGITS)) for px in picks
    ]


def _sample_index(row, position):
    """The row's sample index — the key BO history is stored under.

    ``InpaintInferenceDataset`` does ``rec.setdefault("index", i)``, i.e. it honors an index already in
    the JSONL and only falls back to line position. Mirror that exactly: assuming position here would
    hand every sample another sample's history, silently, on any testcase that carries its own indices."""
    try:
        return int(row["index"])
    except (KeyError, TypeError, ValueError):
        return position


def _draw_rows_bo(rows, observations, seed, guidance_range, crop_ratio_range, bo_min_obs, n_candidates):
    """Overwrite (guidance, crop_ratio) on copies of rows via per-sample Bayesian optimization: fit an
    independent GP on each sample's own (guidance, crop_ratio) -> score history (``observations`` keyed
    by sample index) and Thompson-sample one point. A sample with < bo_min_obs observations over < 2
    distinct locations falls back to uniform. Row order and every other field are preserved. Fully
    seeded (per-sample seed derived from `seed`).

    Returns ``(drawn_rows, n_bo)`` — n_bo is how many samples actually used the GP, which is what the
    caller reports (an invocation with BO *enabled* can still be 100% uniform; see ``_run_draw``)."""
    gl, gh = guidance_range
    cl, ch = crop_ratio_range
    drawn, n_bo = [], 0
    for position, row in enumerate(rows):
        idx = _sample_index(row, position)
        obs = observations.get(idx, [])
        sample_seed = (seed * 100003 + idx) & 0x7FFFFFFF
        if len(obs) >= bo_min_obs and _distinct_xy(obs) >= 2:
            g, c = _bo_propose(obs, 1, sample_seed, guidance_range, crop_ratio_range, n_candidates)[0]
            n_bo += 1
        else:
            rng = random.Random(sample_seed)
            g = round(rng.uniform(gl, gh), _ROUND_DIGITS)
            c = round(rng.uniform(cl, ch), _ROUND_DIGITS)
        new = dict(row)
        new["guidance"] = g
        new["crop_ratio"] = c
        drawn.append(new)
    return drawn, n_bo


def _run_draw(args) -> None:
    rows = _load_base_rows(args.base_testcase)
    if args.original and args.original_kpi:
        observations = _load_observations(
            args.original, args.original_kpi, args.rounds_dir, f"{args.score}_score", args.kpi_name
        )
        drawn, n_bo = _draw_rows_bo(
            rows,
            observations,
            args.seed,
            args.guidance_range,
            args.crop_ratio_range,
            args.bo_min_obs,
            args.bo_candidates,
        )
        # Report the actual per-sample split, not just "BO was enabled". Round 1 supplies exactly one
        # observation per sample, so every sample falls back to uniform — logging "bayesopt" there
        # reads as if the search were already steering when it is still exploring blind.
        mode = (
            f"bayesopt {n_bo}/{len(drawn)} samples (rest uniform)" if n_bo else "uniform (no sample had enough history)"
        )
    else:
        drawn = _draw_rows(rows, random.Random(args.seed), args.guidance_range, args.crop_ratio_range)
        mode = "uniform"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in drawn:
            f.write(json.dumps(row) + "\n")
    log.info(
        f"wrote {len(drawn)} search-round samples -> {out} "
        f"(mode={mode}, guidance {tuple(args.guidance_range)}, crop_ratio {tuple(args.crop_ratio_range)}, seed {args.seed})"
    )


# --- select: pick the best render per sample across rounds ------------------------------------


def _load_scores(kpi_path, score_key):
    """``{output_basename: score}`` flattened across every anomaly type in a ``kpi.json`` (skips ``Average``)."""
    data = json.loads(Path(kpi_path).read_text())
    scores = {}
    for name, block in data.items():
        if name == "Average" or not isinstance(block, dict):
            continue
        for row in block.get("per_sample", []):
            scores[os.path.basename(row["path"])] = row.get(score_key, float("nan"))
    return scores


def _has_usable_score(kpi_path, score_key) -> bool:
    """True when at least one sample in ``kpi_path`` carries a non-NaN ``score_key``.

    A bucket whose kpi.json lacks the metric entirely does not merely rank low — it is removed from
    the competition. ``_load_scores`` defaults a missing key to NaN and ``_pick_best`` replaces a NaN
    incumbent with *any* scored candidate without comparing magnitudes, so an all-NaN bucket loses
    every pick regardless of how good its renders are.
    """
    return any(not math.isnan(v) for v in _load_scores(kpi_path, score_key).values())


def _unrankable_original(kpi_path, bucket_dir, score_key) -> str:
    """The error for an ``--original_kpi`` that cannot rank the original bucket on ``score_key``."""
    return (
        f"{kpi_path} carries no usable {score_key}: every value is missing or NaN. select ranks the "
        "original bucket against the rounds, so it would be dropped from the competition and a round "
        "would win every sample even when strictly worse — while the run exited 0 and reported "
        f"'improved over original by search: N/N'. Re-run evaluate.py on {bucket_dir} so its KPI "
        f"carries {score_key} (an anomaly_quality metric needs the SAM2 checkpoint), then re-run."
    )


def _load_bucket(bucket_dir, kpi_path, score_key, source):
    """``{output_basename: {row, score, nn, mnn, src_dir, source}}`` for one generation bucket.

    ``score`` is the selection metric (``score_key``); ``nn``/``mnn`` are carried for the stitched
    per_sample.csv. All three are per-sample-independent (one render vs the real set), so copying them
    from the source bucket's kpi.json is exact — no re-eval needed."""
    nn = _load_scores(kpi_path, "nn_score")
    mnn = _load_scores(kpi_path, "mnn_score")
    # The anomaly-quality keys live on the same per_sample rows (evaluate.py folds them in), so they
    # load through the same path; nn/mnn are reused rather than re-read since they are always carried.
    if score_key == "nn_score":
        sel = nn
    elif score_key == "mnn_score":
        sel = mnn
    else:
        sel = _load_scores(kpi_path, score_key)
    out = {}
    with (Path(bucket_dir) / _GEN_CSV).open(newline="") as f:
        for row in csv.DictReader(f):
            basename = os.path.basename(row["output_filename"])
            out[basename] = {
                "row": row,
                "score": sel.get(basename, float("nan")),
                "nn": nn.get(basename, float("nan")),
                "mnn": mnn.get(basename, float("nan")),
                "src_dir": str(bucket_dir),
                "source": source,
            }
    return out


def _pick_best(candidates):
    """Highest-scoring candidate (NaN counts as worst; ties keep the earliest source — original before
    round_1 before round_2 ...), given ``candidates`` already in that priority order."""
    best = None
    for cand in candidates:
        if best is None:
            best = cand
            continue
        if not math.isnan(cand["score"]) and (math.isnan(best["score"]) or cand["score"] > best["score"]):
            best = cand
    return best


def _copy_sample_kinds(src_dir, basename, dst_dir) -> None:
    """Copy every KIND for one sample from ``src_dir`` into ``dst_dir`` under the same name. Single-file
    kinds copy 1:1; multi-instance kinds (annotated_image, ``<stem>_<NNNNN>.png``) are globbed per instance."""
    src_dir = Path(src_dir)
    stem = Path(basename).stem
    for k in _SINGLE_KINDS:
        src_f = src_dir / k / basename
        if src_f.exists():
            (dst_dir / k).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst_dir / k / basename)
    for k in _MULTI_KINDS:
        for src_f in sorted((src_dir / k).glob(f"{stem}_*.png")):
            suffix = src_f.name[len(stem) :]  # e.g. "_00000.png"
            if not suffix[1:-4].isdigit():  # require _<digits>.png so a stem can't over-match a sibling
                continue
            (dst_dir / k).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst_dir / k / src_f.name)


def _run_select(args) -> None:
    if Path(args.output).resolve() == Path(args.original).resolve():
        log.error("--output must differ from --original (assembling in place would delete the source bucket)")
        sys.exit(1)
    score_key = f"{args.score}_score"

    original = _load_bucket(args.original, args.original_kpi, score_key, "original")
    buckets = [original]
    for round_dir in _round_dirs(args.rounds_dir):
        kpi = round_dir / args.kpi_name
        if not (round_dir / _GEN_CSV).is_file() or not kpi.is_file():
            log.warning(f"skipping {round_dir}: missing {_GEN_CSV} or {args.kpi_name}")
            continue
        buckets.append(_load_bucket(round_dir, kpi, score_key, round_dir.name))

    # The rounds are gated on carrying the metric; the original must clear the same bar or it silently
    # forfeits every pick (see _has_usable_score). Only when there is something to lose to: with no
    # round buckets this is the clone-only path, where an unscored original still wins by default and
    # searched/ must still be produced for Step 7.
    if len(buckets) > 1 and original and not any(not math.isnan(c["score"]) for c in original.values()):
        log.error(_unrankable_original(args.original_kpi, args.original, score_key))
        sys.exit(1)

    out_dir = Path(args.output)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for k in _SINGLE_KINDS:
        (out_dir / k).mkdir(parents=True, exist_ok=True)

    # Always emit BOTH metrics under their own names, plus which one selection ran on. Writing the
    # winning ``--score`` into a fixed "nn_score" column would silently put mnn values in a column
    # named nn_score under --score mnn; both are already loaded per bucket, so there is no reason to
    # collapse them.
    fieldnames = list(next(iter(original.values()))["row"].keys()) if original else []
    for extra in ("nn_score", "mnn_score", "selected_by", "source"):
        if extra not in fieldnames:
            fieldnames.append(extra)

    def _fmt(x):
        return f"{x:.6f}" if not math.isnan(x) else ""

    out_rows, summary_rows, per_sample_rows = [], [], []
    by_source = defaultdict(int)
    improved = 0
    # Iterate the original bucket's basenames in CSV order so searched/ mirrors the original layout.
    for basename, orig in original.items():
        candidates = [b[basename] for b in buckets if basename in b]
        winner = _pick_best(candidates)
        _copy_sample_kinds(winner["src_dir"], basename, out_dir)

        by_source[winner["source"]] += 1
        if winner["source"] != "original":
            improved += 1
        new_row = dict(winner["row"])
        new_row["output_filename"] = basename
        new_row["nn_score"] = _fmt(winner["nn"])
        new_row["mnn_score"] = _fmt(winner["mnn"])
        new_row["selected_by"] = score_key
        new_row["source"] = winner["source"]
        out_rows.append(new_row)
        summary_rows.append(
            {
                "output_filename": basename,
                "source": winner["source"],
                "selected_by": score_key,
                "score": _fmt(winner["score"]),
                "original_score": _fmt(orig["score"]),
                "improved": "1" if winner["source"] != "original" else "0",
            }
        )
        per_sample_rows.append(
            {
                "anomaly_type": winner["row"].get("anomaly_type", ""),
                "path": str(out_dir / "reconstructed_image" / basename),
                "nn_score": _fmt(winner["nn"]),
                "mnn_score": _fmt(winner["mnn"]),
            }
        )

    with (out_dir / _GEN_CSV).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    with (out_dir / "search_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["output_filename", "source", "selected_by", "score", "original_score", "improved"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    # Per-sample eval table (stitched nn + mnn), mirroring the layout eval produces elsewhere.
    with (out_dir / "per_sample.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["anomaly_type", "path", "nn_score", "mnn_score"])
        writer.writeheader()
        writer.writerows(per_sample_rows)

    n_rounds = len(buckets) - 1
    log.info(
        f"selected {len(out_rows)} best-per-sample renders into {out_dir} "
        f"({n_rounds} round(s), selected by {score_key})"
    )
    log.info("picked-from distribution: " + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))
    log.info(f"improved over original by search: {improved}/{len(out_rows)}")


# --- run: the whole Step 6 loop (draw -> generate -> evaluate) x N, then select ------------------


def _round_dir(rounds_dir, r):
    return Path(rounds_dir) / f"round_{r}"


def _count_rows(testcase):
    return sum(1 for line in open(testcase) if line.strip())


def _blocked_count(round_dir):
    """Outputs the content-safety guardrail suppressed in this bucket.

    ``generate.py`` writes ``guardrail_blocked.csv`` whenever the guardrail ran — header-only when it
    blocked nothing — and not at all under ``--no-guardrail``, so an absent file means zero either way.
    """
    path = Path(round_dir) / "guardrail_blocked.csv"
    if not path.is_file():
        return 0
    with path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _round_incomplete_reason(round_dir, expected_rows, kpi_name, score_key=None):
    """Why this round does not count, or None if it is complete.

    Both consumers key off ``kpi.json``: ``select`` scores buckets from it and the next round's BO
    pools it into history. A round that stopped early is otherwise dropped silently — the run still
    exits 0 — so the loop refuses to continue instead of wasting the remaining GPU time.

    A guardrail-blocked sample writes no image *by design*, so it counts toward the round rather than
    against it. Comparing images to rows alone would abort the whole search over one blocked caption.

    ``score_key`` is the metric the search optimises, and a present-but-unscored ``kpi.json`` is
    checked separately because it fails *silently* rather than loudly. ``evaluate.py`` degrades to
    NN/MNN(+FID) on any anomaly_quality failure — a missing SAM2 checkpoint is enough — so with an
    ``aq_*`` metric the rows can arrive complete and correctly counted but carry no ``score_key``.
    ``_load_scores`` then defaults every sample to NaN, ``_pick_best`` ranks NaN worst and keeps the
    earliest source on ties, and the original wins every sample: N rounds of generation burned for a
    byte-identical ``searched/``, exit 0. Requiring one real value turns that into a named failure.
    """
    round_dir = Path(round_dir)
    kpi_path = round_dir / kpi_name
    if not kpi_path.is_file():
        return f"missing {kpi_name}"
    images = len(list((round_dir / "reconstructed_image").glob("*.png")))
    blocked = _blocked_count(round_dir)
    if images + blocked != expected_rows:
        return f"{images} image(s) + {blocked} guardrail-blocked for {expected_rows} testcase row(s)"
    if score_key and not _has_usable_score(kpi_path, score_key):
        return (
            f"no sample in {kpi_name} carries a usable {score_key} — every value is missing or NaN, so "
            "the search cannot rank this round (an anomaly_quality metric needs the SAM2 checkpoint; "
            "check the evaluate step's log for an 'anomaly_quality computation failed' warning)"
        )
    return None


def _generate_argv(args, testcase, output_dir):
    return [
        "--checkpoint",
        str(args.checkpoint),
        "--recipe",
        str(args.recipe),
        "--input_data_path",
        str(testcase),
        "--output_dir",
        str(output_dir),
    ]


def _evaluate_argv(args, gen_root, output_file):
    argv = [
        "--gen_root",
        str(gen_root),
        "--real_root",
        str(args.real_root),
        "--recipe",
        str(args.recipe),
        "--output_file",
        str(output_file),
    ]
    # Forward only what the caller set, so evaluate.py's own defaults stay authoritative. Without this
    # a run that scored Step 5 with non-default knobs would search against differently-computed
    # numbers and select winners on a metric that is not the one being reported.
    for name in _EVAL_SCORING_FLAGS:
        value = getattr(args, name, None)
        if value is not None:
            argv += [f"--{name}", str(value)]
    return argv


# Whole namespaces torchrun owns. ``PET_<ARG>`` is torchrun's env fallback for its own CLI args
# (torch.distributed.argparse_util.env): an explicit flag beats it, so the three we pass — nproc-per-node,
# rdzv-backend, rdzv-endpoint — are already immune. The exposure is the ones we *don't* pass, above all
# PET_NNODES, which would leave the child waiting on a multi-node rendezvous that never arrives.
# Stripping by prefix also covers whatever a future torch adds.
_TORCHRUN_OWNED_PREFIXES = ("PET_", "TORCHELASTIC_")

# Set by torchrun for its children. Inheriting them from *our* environment — e.g. when this command is
# itself run inside an outer torchrun — makes the child see two conflicting rendezvous configurations,
# which generate.py surfaces as a world-size mismatch rather than anything obvious.
_TORCHRUN_OWNED = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def _generate_env(output_root, environ=None):
    """The environment for the torchrun child: ours, minus torchrun's own vars, plus an explicit
    output root.

    ``IMAGINAIRE_OUTPUT_ROOT`` is worth setting rather than inheriting because the framework falls
    back to ``/tmp/imaginaire4-output`` when it is unset (``utils/config.py``) — so an unset value does
    not fail, it quietly resolves the config's job paths somewhere else entirely.
    """
    source = os.environ if environ is None else environ
    env = {k: v for k, v in source.items() if k not in _TORCHRUN_OWNED and not k.startswith(_TORCHRUN_OWNED_PREFIXES)}
    env["IMAGINAIRE_OUTPUT_ROOT"] = str(output_root)
    return env


def _generate_cmd(args, testcase, output_dir):
    """The torchrun command for one round's generation.

    Every round launches a *fresh* torchrun, so each needs its own rendezvous: the static backend pins
    ``MASTER_PORT=29500``, and a round starting before the previous one released that port fails to
    bind — a flake that appears mid-search on a busy machine, not on the first round. ``--standalone``
    asks for a c10d rendezvous on an ephemeral port plus a unique run id (``run.py``: ``rdzv_backend =
    "c10d"``, ``rdzv_endpoint = "localhost:0"``, ``rdzv_id = uuid4()``).

    Requesting it explicitly rather than relying on the default: torch >=2.6 already applies the same
    three settings when ``--nnodes`` is 1 and no endpoint or master port is given, but that is an
    implicit fallback guarded by conditions a future caller could break simply by passing one more
    flag. Saying ``--standalone`` states the requirement instead of inheriting it.
    """
    return [
        "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        "--standalone",
        str(Path(__file__).with_name("generate.py")),
        *_generate_argv(args, testcase, output_dir),
    ]


def _generate(args, testcase, output_dir):
    """generate.py needs torch.distributed, so it is launched through torchrun rather than called."""
    cmd = _generate_cmd(args, testcase, output_dir)
    log.info(f"$ IMAGINAIRE_OUTPUT_ROOT={args.output_root} " + " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, env=_generate_env(args.output_root)).returncode


def _evaluate(args, gen_root, output_file):
    argv = _evaluate_argv(args, gen_root, output_file)
    log.info("$ evaluate.py " + " ".join(argv))
    if args.dry_run:
        return 0
    from anomalygen.scripts.texture import evaluate

    evaluate.main(argv)
    return 0


def _draw_round(args, out_testcase, seed):
    argv = [
        "draw",
        "--base_testcase",
        str(args.base_testcase),
        "--output",
        str(out_testcase),
        "--seed",
        str(seed),
        "--original",
        str(args.original),
        "--original_kpi",
        str(args.original_kpi),
        "--rounds_dir",
        str(args.rounds_dir),
        "--score",
        args.score,
        "--kpi_name",
        args.kpi_name,
        "--bo_min_obs",
        str(args.bo_min_obs),
        "--bo_candidates",
        str(args.bo_candidates),
        "--guidance_range",
        str(args.guidance_range[0]),
        str(args.guidance_range[1]),
        "--crop_ratio_range",
        str(args.crop_ratio_range[0]),
        str(args.crop_ratio_range[1]),
    ]
    log.info("$ quality_refine.py " + " ".join(argv))
    if args.dry_run:
        return
    main(argv)


def _run_run(args) -> None:
    """draw -> generate -> evaluate for each round, gate each round, then select and score."""
    expected_rows = 0 if args.dry_run else _count_rows(args.base_testcase)
    score_key = f"{args.score}_score"
    original_kpi = Path(args.original_kpi)
    # Fail before the first round rather than after the last: select ranks the original bucket against
    # the rounds, so an original that cannot be ranked forfeits every pick and the whole search is
    # decided before it starts. Skipped with no rounds (select then just clones the original), and
    # guarded on the file existing so a missing --original_kpi still surfaces at select, where it
    # always has.
    if (
        not args.dry_run
        and args.num_search_run > 0
        and original_kpi.is_file()
        and not _has_usable_score(original_kpi, score_key)
    ):
        log.error(_unrankable_original(original_kpi, args.original, score_key))
        sys.exit(1)

    for r in range(1, args.num_search_run + 1):
        rd = _round_dir(args.rounds_dir, r)
        log.info(f"--- refinement round {r}/{args.num_search_run} -> {rd} ---")
        testcase = rd / "testcase.jsonl"
        _draw_round(args, testcase, seed=r)
        rc = _generate(args, testcase, rd)
        if rc != 0:
            log.error(f"round {r}: generate.py exited {rc} — stopping before select.")
            sys.exit(rc)
        _evaluate(args, rd, rd / args.kpi_name)
        if args.dry_run:
            continue
        reason = _round_incomplete_reason(rd, expected_rows, args.kpi_name, score_key)
        if reason:
            log.error(
                f"round {r} is incomplete ({reason}) — stopping. Re-run this round's generate.py "
                "(it overwrites) and evaluate.py; a searched/ assembled without it is stale."
            )
            sys.exit(1)

    log.info(f"--- select best-per-sample -> {args.output} ---")
    if not args.dry_run:
        _run_select(args)
    _evaluate(args, args.output, args.final_kpi)
    log.info(f"refinement complete: {args.output} scored into {args.final_kpi}")


# --- CLI ---------------------------------------------------------------------------------------


def _get_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-sample (guidance, crop_ratio) search.")
    sub = parser.add_subparsers(dest="command", required=True)

    draw = sub.add_parser("draw", help="write one search round's testcase.jsonl (redraw every sample)")
    draw.add_argument(
        "--base_testcase", required=True, help="base testcase.jsonl (the AMP output for the generation batch)"
    )
    draw.add_argument("--output", required=True, help="round testcase.jsonl to write")
    draw.add_argument(
        "--guidance_range",
        type=float,
        nargs=2,
        default=(1.5, 10.0),
        metavar=("LOW", "HIGH"),
        help="uniform draw range for guidance",
    )
    draw.add_argument(
        "--crop_ratio_range",
        type=float,
        nargs=2,
        default=(1.5, 8.0),
        metavar=("LOW", "HIGH"),
        help="uniform draw range for crop_ratio (default matches the recipe's ratio_range)",
    )
    draw.add_argument(
        "--seed", type=int, default=42, help="RNG seed for reproducible draws (vary per round, e.g. the round number)"
    )
    draw.add_argument(
        "--original", default=None, help="original generation bucket (enables BO together with --original_kpi)"
    )
    draw.add_argument("--original_kpi", default=None, help="kpi.json scored on --original")
    draw.add_argument(
        "--rounds_dir", default=None, help="dir of prior round_<r>/ buckets (each with a kpi.json) to pool into BO"
    )
    draw.add_argument("--score", choices=_SCORE_CHOICES, default="nn", help="per-sample metric to optimize")
    draw.add_argument("--kpi_name", default="kpi.json", help="kpi filename inside each round dir")
    draw.add_argument(
        "--bo_min_obs",
        type=int,
        default=2,
        help="min observations per sample to use BO (else uniform); per-sample history is thin, keep low",
    )
    draw.add_argument("--bo_candidates", type=int, default=1024, help="Thompson-sampling candidate pool size")
    draw.set_defaults(func=_run_draw)

    select = sub.add_parser("select", help="select best-per-sample renders across rounds into searched/")
    select.add_argument("--original", required=True, help="original generation bucket (${OUT})")
    select.add_argument("--original_kpi", required=True, help="kpi.json scored on --original")
    select.add_argument(
        "--rounds_dir",
        default=None,
        help="dir of round_<r>/ buckets each with a kpi.json; omit/empty to clone --original",
    )
    select.add_argument("--output", required=True, help="searched/ bucket to write")
    select.add_argument("--score", choices=_SCORE_CHOICES, default="nn", help="per-sample metric to compare")
    select.add_argument("--kpi_name", default="kpi.json", help="kpi filename inside each round dir")
    select.set_defaults(func=_run_select)

    run = sub.add_parser("run", help="the whole Step 6 loop: (draw -> generate -> evaluate) xN, select, score")
    run.add_argument("--base_testcase", required=True, help="generation testcase.jsonl (the AMP output)")
    run.add_argument("--original", required=True, help="original generation bucket (${OUT})")
    run.add_argument("--original_kpi", required=True, help="kpi.json scored on --original")
    run.add_argument("--rounds_dir", required=True, help="dir the round_<r>/ buckets are written under")
    run.add_argument("--output", required=True, help="searched/ bucket to assemble")
    run.add_argument("--final_kpi", required=True, help="where to score the assembled searched/ bucket")
    run.add_argument("--checkpoint", required=True, help="fine-tuned checkpoint for generate.py")
    run.add_argument("--recipe", required=True, help="run-dir recipe copy — must match --checkpoint")
    run.add_argument("--real_root", required=True, help="dataset root, the evaluation real-reference")
    run.add_argument("--num_search_run", type=int, default=3, help="search rounds; 0 = select-only (clone)")
    run.add_argument("--num_gpus", type=int, default=1, help="torchrun --nproc_per_node for generate.py")
    run.add_argument(
        "--output_root",
        default="results",
        help="IMAGINAIRE_OUTPUT_ROOT for generate.py; set explicitly because the framework otherwise "
        "falls back to /tmp/imaginaire4-output without complaining.",
    )
    run.add_argument("--score", choices=_SCORE_CHOICES, default="nn", help="per-sample metric to optimize")
    run.add_argument("--kpi_name", default="kpi.json", help="kpi filename inside each round dir")
    run.add_argument("--guidance_range", type=float, nargs=2, default=(1.5, 10.0), metavar=("LOW", "HIGH"))
    run.add_argument("--crop_ratio_range", type=float, nargs=2, default=(1.5, 8.0), metavar=("LOW", "HIGH"))
    run.add_argument("--bo_min_obs", type=int, default=2, help="min observations per sample before BO")
    run.add_argument("--bo_candidates", type=int, default=1024, help="Thompson-sampling candidate pool size")
    # The scoring knobs each round's evaluate.py is run with. add_nn_scoring_args is evaluate's own
    # declaration, so the nn/mnn flag names, choices and defaults cannot drift apart from it.
    add_nn_scoring_args(run)
    run.add_argument("--top_k", type=int, default=None, help="evaluate.py --top_k for every round")
    run.add_argument("--model_input_size", type=int, default=None, help="evaluate.py --model_input_size")
    run.add_argument("--dry_run", action="store_true", help="print each stage's command without running it")
    run.set_defaults(func=_run_run)

    args = parser.parse_args(argv)
    if args.command in ("draw", "run"):
        # A degenerate range would divide by zero when scaling to the GP's unit box, producing
        # nan/inf proposals that argparse would otherwise let straight through into the testcase.
        for name in ("guidance_range", "crop_ratio_range"):
            low, high = getattr(args, name)
            if not high > low:
                parser.error(f"--{name} must be LOW < HIGH (got {low} {high})")
    return args


def main(argv=None) -> None:
    args = _get_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
