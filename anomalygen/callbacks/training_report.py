# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-report callback (rank 0, non-fatal on failure): snapshots the recipe at train start,
refreshes ``training_loss.png`` (per ``loss_window``-step mean) and ``training_curves.png``
(validation-metric trajectory, loss curve as fallback) at every validation step and at train end, and
at train end records the best-scoring iteration in ``checkpoints/best_checkpoint.txt``."""

from __future__ import annotations

import csv
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before pyplot is imported
import matplotlib.pyplot as plt
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback

from anomalygen.callbacks.training_curve_plot import plot_early_stop
from anomalygen.configs.texture.constants import CKPT_WARMUP_ITER
from anomalygen.eval.metric_specs import METRIC_SPECS

# valid_kpi.csv row key (e.g. "nn_score") -> plot metric name (e.g. "nn").
_KPI_TO_METRIC = {kpi_key: metric for metric, (kpi_key, _) in METRIC_SPECS.items()}
# Stable plot/legend order.
_METRIC_ORDER = ["nn", "mnn", "fid", "aq_nn", "completeness", "precision", "boundary_iou"]


class TrainingReport(Callback):
    def __init__(
        self,
        recipe_path: Optional[str] = None,
        valid_subdir: str = "valid",
        loss_window: int = 10,
        best_metric: str = "nn",
    ) -> None:
        super().__init__()
        # Absolute path to the recipe module; copied verbatim into the run dir.
        self.recipe_path = recipe_path
        # Subdir ValidationKPI writes valid_kpi.csv into; must match its output_subdir.
        self.valid_subdir = valid_subdir
        # Metric best_checkpoint.txt selects on (key of METRIC_SPECS, which supplies the direction).
        # The recipe builder passes the run's own early_stop_metric, so the checkpoint we keep is
        # scored on the same metric training was monitored (and possibly stopped) on; "nn" — the
        # primary KPI and EarlyStop's own default — applies when a caller constructs this bare.
        self.best_metric = best_metric
        # Number of consecutive training steps averaged into one loss-curve data point.
        self.loss_window = max(1, int(loss_window))
        # Per-step losses buffered, then flushed to one (iteration, mean) point every ``loss_window`` steps.
        self._loss_buffer: List[float] = []
        self._loss_iters: List[int] = []
        self._loss_means: List[float] = []
        # Captured before early stop can shrink max_iter; the plot needs the original planned horizon.
        self._max_iter_planned: Optional[int] = None

    def _run_dir(self) -> str:
        job = getattr(getattr(self, "config", None), "job", None)
        return getattr(job, "path_local", None) or os.getcwd()

    def on_train_start(self, model, iteration: int = 0) -> None:
        trainer = getattr(getattr(self, "config", None), "trainer", None)
        self._max_iter_planned = getattr(trainer, "max_iter", None)

        if distributed.is_rank0():
            self._copy_recipe()
            # Reload the persisted loss history so a resumed run keeps the pre-resume curve
            # instead of overwriting training_loss.png with only the post-resume segment.
            self._load_loss_history(iteration)

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        if not distributed.is_rank0():
            return
        self._loss_buffer.append(float(loss.item()))
        if len(self._loss_buffer) >= self.loss_window:
            mean = sum(self._loss_buffer) / len(self._loss_buffer)
            self._loss_iters.append(iteration)
            self._loss_means.append(mean)
            self._loss_buffer.clear()
            self._append_loss_point(iteration, mean)

    def on_validation_end(self, model, iteration: int = 0) -> None:
        # Refresh mid-run. Relies on ValidationKPI running first so this iteration's CSV is already written.
        if distributed.is_rank0():
            self._refresh_reports()

    def on_train_end(self, model, iteration: int = 0) -> None:
        if distributed.is_rank0():
            self._refresh_reports()
            self._write_best_checkpoint()

    def _write_best_checkpoint(self) -> None:
        """Record the best-scoring iteration in ``checkpoints/best_checkpoint.txt``.

        The trainer's own ``latest_checkpoint.txt`` names the *last* iteration, which is frequently not
        the best — small datasets peak early and then drift, so generating from the latest silently uses
        a worse model. This writes a sibling pointer in the same bare-filename format
        (``iter_<N>.pt``), leaving ``latest_checkpoint.txt`` untouched for resume.
        """
        try:
            spec = METRIC_SPECS.get(self.best_metric)
            if spec is None:
                log.warning(f"TrainingReport: unknown best_metric {self.best_metric!r} — skipping best_checkpoint.txt.")
                return

            iters, metrics = self._read_valid_metrics()
            values = metrics.get(self.best_metric) if metrics else None
            # Both branches below leave no pointer, which is not inert: inference resolves a run dir
            # through best_checkpoint.txt *then* latest_checkpoint.txt, so a missing pointer silently
            # generates from the last iteration — usually the worse model. Say which cause it was.
            if not iters:
                log.warning(
                    f"TrainingReport: no validation results under {self.valid_subdir}/ — skipping "
                    "best_checkpoint.txt, so inference will fall back to latest_checkpoint.txt. "
                    "Check that validation_iter is not larger than max_iter."
                )
                return
            if not values:
                log.warning(
                    f"TrainingReport: {len(iters)} validated iteration(s) but none recorded "
                    f"{self.best_metric!r} — skipping best_checkpoint.txt, so inference will fall "
                    "back to latest_checkpoint.txt. Check that best_metric names a metric this run "
                    "actually scores."
                )
                return

            model_dir = os.path.join(self._run_dir(), "checkpoints", "model")
            # Only iterations with a checkpoint on disk are eligible: run_validation_on_start scores
            # iteration 0, which is never checkpointed, and a metric is NaN when a defect type dropped
            # out of that validation pass. The pointer must never dangle.
            scored = [
                (it, v)
                for it, v in zip(iters, values)
                if v == v and os.path.isfile(os.path.join(model_dir, f"iter_{it:09}.pt"))
            ]
            # Warm-up guard, on runs long enough for it to leave anything behind.
            trainer = getattr(getattr(self, "config", None), "trainer", None)
            max_iter = self._max_iter_planned or getattr(trainer, "max_iter", None) or 0
            if scored and max_iter > CKPT_WARMUP_ITER:
                settled = [t for t in scored if t[0] >= CKPT_WARMUP_ITER]
                if settled:
                    scored = settled
                else:
                    # Early stopping or a crash ended the run before the warm-up. Writing nothing
                    # would silently demote inference to latest_checkpoint.txt, which is worse than a
                    # possibly-unsettled pick — so keep every iteration, and say the pick is early.
                    log.warning(
                        f"TrainingReport: no validated checkpoint at or after the warm-up "
                        f"({CKPT_WARMUP_ITER}) though max_iter was {max_iter} — the run ended "
                        f"early. Selecting from iterations below it; {self.best_metric} may not have "
                        "settled, so compare against a longer run before trusting this checkpoint."
                    )

            if not scored:
                # Silence here is indistinguishable from "training never validated" — say why, since
                # the usual cause (validation_iter not a multiple of save_iter, so no validated
                # iteration was ever checkpointed) is a recipe fix, not a transient.
                log.warning(
                    f"TrainingReport: {len(iters)} validated iteration(s) but none has both a finite "
                    f"{self.best_metric} and a checkpoint under {model_dir} — skipping "
                    "best_checkpoint.txt. Check that validation_iter is a multiple of save_iter."
                )
                return

            best_iter, best_val = (max if spec[1] == "max" else min)(scored, key=lambda t: t[1])
            name = f"iter_{best_iter:09}.pt"

            # Write atomically: a plain open("w") truncates first, so an interrupt mid-write leaves an
            # empty or partial pointer. Inference degrades gracefully either way (it falls back to
            # latest_checkpoint.txt), but that silently loses the best-checkpoint pick — os.replace is
            # atomic within a filesystem, so readers see either the old pointer or the complete new one.
            pointer = os.path.join(self._run_dir(), "checkpoints", "best_checkpoint.txt")
            os.makedirs(os.path.dirname(pointer), exist_ok=True)
            tmp = f"{pointer}.tmp"
            with open(tmp, "w") as f:
                f.write(f"{name}\n")
            os.replace(tmp, pointer)

            log.info(
                f"Best checkpoint by {self.best_metric} ({spec[1]}): iteration {best_iter} "
                f"= {best_val:.4f} -> {os.path.join(model_dir, name)}"
            )
        except Exception as e:  # noqa: BLE001
            log.warning(f"TrainingReport: failed to write best_checkpoint.txt: {e}")

    def _refresh_reports(self) -> None:
        self._plot_curves()
        self._plot_loss_curve(os.path.join(self._run_dir(), "training_loss.png"))

    def _copy_recipe(self) -> None:
        if not self.recipe_path:
            log.warning("TrainingReport: recipe_path not set — skipping recipe copy.")
            return
        if not os.path.isfile(self.recipe_path):
            log.warning(f"TrainingReport: recipe_path {self.recipe_path!r} not found — skipping recipe copy.")
            return

        try:
            dest = os.path.join(self._run_dir(), os.path.basename(self.recipe_path))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(self.recipe_path, dest)
        except Exception as e:  # noqa: BLE001
            log.warning(f"TrainingReport: failed to copy recipe: {e}")

    def _plot_curves(self) -> None:
        path = os.path.join(self._run_dir(), "training_curves.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Prefer the validation-metric trajectory; fall back to the loss curve when no KPI data exists.
        iters, metrics = self._read_valid_metrics()
        if iters and metrics:
            if self._plot_metric_curves(path, iters, metrics):
                return

        self._plot_loss_curve(path)

    def _read_valid_metrics(self) -> Tuple[List[int], Dict[str, List[float]]]:
        """Read per-iteration validation metrics (the ``Average`` column of each ``<valid_subdir>/<iter>/valid_kpi.csv``)."""
        base = os.path.join(self._run_dir(), self.valid_subdir)
        if not os.path.isdir(base):
            return [], {}

        records: Dict[int, Dict[str, float]] = {}
        for name in os.listdir(base):
            csv_path = os.path.join(base, name, "valid_kpi.csv")
            if not name.isdigit() or not os.path.isfile(csv_path):
                continue
            try:
                with open(csv_path, newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header or header[-1] != "Average":
                        continue
                    row_vals: Dict[str, float] = {}
                    for row in reader:
                        if not row:
                            break  # blank line ends the averages matrix (per-sample section follows)
                        metric = _KPI_TO_METRIC.get(row[0])
                        if metric is None:
                            continue
                        try:
                            row_vals[metric] = float(row[-1])
                        except (ValueError, IndexError):
                            continue
                if row_vals:
                    records[int(name)] = row_vals
            except Exception as e:  # noqa: BLE001
                log.warning(f"TrainingReport: failed to read {csv_path}: {e}")

        if not records:
            return [], {}

        iters = sorted(records)
        present = {m for vals in records.values() for m in vals}
        metrics = {m: [records[it].get(m, float("nan")) for it in iters] for m in _METRIC_ORDER if m in present}
        return iters, metrics

    def _read_early_stop_state(self) -> Optional[dict]:
        """State ValidationKPI writes to ``early_stop.json`` when early stop fires; present enables the triggered-mode plot."""
        es_path = os.path.join(self._run_dir(), "early_stop.json")
        if not os.path.isfile(es_path):
            return None

        try:
            with open(es_path) as f:
                state = json.load(f)
            return state if state.get("triggered") else None
        except Exception as e:  # noqa: BLE001
            log.warning(f"TrainingReport: failed to read early_stop.json: {e}")
            return None

    def _plot_metric_curves(self, path: str, iters: List[int], metrics: Dict[str, List[float]]) -> bool:
        """Plot the validation-metric trajectory; returns True on success, False (logged) on failure."""
        try:
            # Original planned horizon; fall back to live config or last recorded iteration.
            trainer = getattr(getattr(self, "config", None), "trainer", None)
            max_iter = self._max_iter_planned or getattr(trainer, "max_iter", None) or max(iters)

            kwargs = dict(
                iters=iters,
                metrics=metrics,
                max_iter_planned=int(max_iter),
                out_path=path,
                triggered=False,
            )
            # Use triggered mode only if early_stop.json is consistent with the CSV data.
            state = self._read_early_stop_state()
            if (
                state
                and state.get("criteria") in metrics
                and state.get("best_iteration") in iters
                and state.get("stop_iteration") in iters
            ):
                kwargs.update(
                    triggered=True,
                    criteria=state["criteria"],
                    patience=state.get("patience"),
                    best_iteration=state["best_iteration"],
                    stop_iteration=state["stop_iteration"],
                    last_improved_iteration=state.get("last_improved_iteration"),
                    min_delta=state.get("min_delta"),
                    min_delta_mode=state.get("min_delta_mode"),
                    cumulative_delta=state.get("cumulative_delta"),
                )
            elif state:
                log.warning(
                    "TrainingReport: early_stop.json present but inconsistent with valid_kpi data — "
                    "drawing plain metric curves."
                )

            plot_early_stop(**kwargs)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning(f"TrainingReport: failed to write validation-metric training_curves.png: {e}")
            return False

    def _loss_csv_path(self) -> str:
        return os.path.join(self._run_dir(), "training_loss.csv")

    def _load_loss_history(self, resume_iteration: int) -> None:
        """Seed the in-memory loss series from ``training_loss.csv`` so the curve survives resume.

        Keeps only points at or before ``resume_iteration`` — any written past the checkpoint the
        run resumes from will be recomputed, so we drop them and rewrite the CSV to stay consistent
        with the appends that follow. ``resume_iteration == 0`` (fresh start) keeps nothing."""
        path = self._loss_csv_path()
        if not os.path.isfile(path):
            return

        iters: List[int] = []
        means: List[float] = []
        try:
            with open(path, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # header
                for row in reader:
                    if len(row) < 2:
                        continue
                    try:
                        it, mean = int(row[0]), float(row[1])
                    except ValueError:
                        continue
                    if resume_iteration and it > resume_iteration:
                        continue
                    iters.append(it)
                    means.append(mean)
        except Exception as e:  # noqa: BLE001
            log.warning(f"TrainingReport: failed to read training_loss.csv: {e}")
            return

        self._loss_iters = iters
        self._loss_means = means
        self._rewrite_loss_csv()

    def _rewrite_loss_csv(self) -> None:
        path = self._loss_csv_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "mean_loss"])
                writer.writerows(zip(self._loss_iters, self._loss_means))
        except Exception as e:  # noqa: BLE001
            log.warning(f"TrainingReport: failed to rewrite training_loss.csv: {e}")

    def _append_loss_point(self, iteration: int, mean: float) -> None:
        path = self._loss_csv_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_header = not os.path.isfile(path)
            with open(path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["iteration", "mean_loss"])
                writer.writerow([iteration, mean])
        except Exception as e:  # noqa: BLE001
            log.warning(f"TrainingReport: failed to append to training_loss.csv: {e}")

    def _plot_loss_curve(self, path: str) -> None:
        if not self._loss_iters:
            return

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig, ax = plt.subplots()
            ax.plot(self._loss_iters, self._loss_means, marker="o", markersize=3)
            ax.set_xlabel("iteration")
            ax.set_ylabel(f"total loss ({self.loss_window}-iter mean)")
            ax.set_title("training loss")
            ax.grid(True, alpha=0.3)

            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            log.warning(f"TrainingReport: failed to write {os.path.basename(path)}: {e}")
