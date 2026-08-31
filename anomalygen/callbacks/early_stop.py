# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early-stopping callback.

Tracks a monitored validation metric (read from the ``valid_kpi.csv`` ``ValidationKPI`` writes)
and, when it stops improving for ``patience`` consecutive validations, shrinks ``trainer.max_iter``
so the trainer exits. Rank 0 decides; the decision is broadcast so every rank stops together.
Disabled by default.
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, Optional

import torch
import torch.distributed as dist
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback

from anomalygen.configs.texture.constants import CKPT_WARMUP_ITER
from anomalygen.eval.metric_specs import METRIC_SPECS


class EarlyStop(Callback):
    def __init__(
        self,
        enabled: bool = False,
        metric: str = "nn",
        patience: int = 5,
        scope: str = "Average",
        min_delta: float = 0.0,
        min_delta_mode: str = "rel",
        cumulative_delta: bool = False,
        valid_subdir: str = "valid",
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.valid_subdir = valid_subdir
        self._run_dir: Optional[str] = None
        self.metric = metric
        self.patience = patience
        self.scope = scope
        self.min_delta = min_delta
        self.min_delta_mode = min_delta_mode
        self.cumulative_delta = cumulative_delta
        self._kpi_key: Optional[str] = None
        self.mode: Optional[str] = None
        self.best: float = math.inf
        self.best_iteration: Optional[int] = None
        self.last_improved_iteration: Optional[int] = None
        self.wait = 0
        self._max_iter_planned: Optional[int] = None

        if enabled:
            if metric not in METRIC_SPECS:
                raise ValueError(f"Unknown early-stop metric '{metric}'. Choose one of {list(METRIC_SPECS)}.")
            if patience < 1:
                raise ValueError(f"patience must be ≥ 1, got {patience}")
            if min_delta < 0.0:
                raise ValueError(f"min_delta must be non-negative, got {min_delta}")
            if min_delta_mode not in ("abs", "rel"):
                raise ValueError(f"min_delta_mode must be 'abs' or 'rel', got '{min_delta_mode}'")

            self._kpi_key, self.mode = METRIC_SPECS[metric]
            self.best = -math.inf if self.mode == "max" else math.inf

    def on_train_start(self, model, iteration: int = 0) -> None:
        job = getattr(getattr(self, "config", None), "job", None)
        self._run_dir = getattr(job, "path_local", None) or os.getcwd()
        # Captured here, before _trigger can shrink it — the warm-up floor keys off what the run was
        # *planned* to reach, exactly as TrainingReport._write_best_checkpoint does.
        trainer = getattr(getattr(self, "config", None), "trainer", None)
        self._max_iter_planned = getattr(trainer, "max_iter", None)

    def on_validation_end(self, model, iteration: int = 0) -> None:
        if not self.enabled:
            return
        # The KPI is read on rank 0 only; _should_stop broadcasts the decision to every rank, so
        # all ranks must call it (non-rank-0 passes None).
        valid_kpi = self._read_kpi(iteration) if distributed.is_rank0() else None
        if self._should_stop(valid_kpi, iteration):
            self._trigger(iteration)

    def _read_kpi(self, iteration: int) -> Optional[dict]:
        """Reconstruct ``{scope: {kpi_key: value}}`` from the valid_kpi.csv ValidationKPI wrote."""
        csv_path = os.path.join(self._run_dir or os.getcwd(), self.valid_subdir, str(iteration), "valid_kpi.csv")
        if not os.path.isfile(csv_path):
            log.warning(f"EarlyStop: {csv_path} not found — skipping this validation.")
            return None

        try:
            with open(csv_path, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return None
                scopes = header[1:]  # per-anomaly columns + "Average"
                kpi: Dict[str, Dict[str, float]] = {s: {} for s in scopes}
                for row in reader:
                    if not row:
                        break  # blank line ends the averages matrix (per-sample section follows)
                    kpi_key = row[0]
                    for scope, value in zip(scopes, row[1:]):
                        try:
                            kpi[scope][kpi_key] = float(value)
                        except (ValueError, TypeError):
                            pass  # missing/blank cell — leave that (scope, metric) absent

            return kpi
        except Exception as e:  # noqa: BLE001
            log.warning(f"EarlyStop: failed to read {csv_path}: {e}")
            return None

    def _improvement_threshold(self) -> float:
        """Score threshold a new value must cross to count as improvement."""
        signed = -self.min_delta if self.mode == "min" else self.min_delta
        if self.min_delta_mode == "abs":
            return self.best + signed
        return self.best * (1 + signed)

    def _should_stop(self, valid_kpi: Optional[dict], iteration: int) -> bool:
        """Update the monitored-metric state on rank 0, then broadcast the stop decision to all ranks."""
        stop = 0
        try:
            if distributed.get_rank() == 0 and valid_kpi is not None:
                stop = self._rank0_decision(valid_kpi, iteration)
        finally:
            # The broadcast is collective: every rank must reach it, even if rank-0 scoring raised,
            # or the other ranks block here forever. A rank-0 exception still propagates afterward,
            # failing the job loudly instead of hanging it.
            if distributed.get_world_size() > 1:
                stop_tensor = torch.tensor([stop], device="cuda")
                dist.broadcast(stop_tensor, src=0)
                stop = int(stop_tensor.item())
        return bool(stop)

    def _below_warmup(self, iteration: int) -> bool:
        """Whether this validation is too early to count toward patience."""
        planned = self._max_iter_planned or 0
        return planned > CKPT_WARMUP_ITER and iteration < CKPT_WARMUP_ITER

    def _rank0_decision(self, valid_kpi: dict, iteration: int) -> int:
        """Update best/wait state from the monitored metric; return 1 when patience is exhausted."""
        stop = 0
        if self._below_warmup(iteration):
            log.info(
                f"EarlyStop: iter={iteration} is below the warm-up ({CKPT_WARMUP_ITER}) on a run planned "
                f"to {self._max_iter_planned} — not counted toward patience ({self.metric} has not settled)."
            )
            return stop
        score = valid_kpi.get(self.scope, {}).get(self._kpi_key)

        if score is None or (isinstance(score, float) and math.isnan(score)):
            log.warning(f"EarlyStop: {self.scope}/{self._kpi_key} not available at iter={iteration}; skipping.")
        elif math.isinf(self.best):
            self.best = score
            self.best_iteration = iteration
            self.last_improved_iteration = iteration
            log.info(f"EarlyStop: iter={iteration} {self.metric}={score:.4f} (initial)")
        else:
            threshold = self._improvement_threshold()
            improved = score > threshold if self.mode == "max" else score < threshold
            if improved:
                log.info(f"EarlyStop: iter={iteration} {self.metric}={score:.4f} improved over best={self.best:.4f}")
                self.best = score
                self.best_iteration = iteration
                self.last_improved_iteration = iteration
                self.wait = 0
            else:
                # When not cumulative_delta, creep the threshold along with
                # the actual best ever seen (stricter).
                if not self.cumulative_delta:
                    is_strict_better = (self.mode == "max" and score > self.best) or (
                        self.mode == "min" and score < self.best
                    )
                    if is_strict_better:
                        self.best = score
                        self.best_iteration = iteration
                self.wait += 1
                log.info(
                    f"EarlyStop: iter={iteration} {self.metric}={score:.4f} "
                    f"no improve ({self.wait}/{self.patience}, "
                    f"best={self.best:.4f} at iter={self.best_iteration})"
                )
                if self.wait >= self.patience:
                    stop = 1
                    log.success(
                        f"EarlyStop: triggered at iter={iteration}, "
                        f"best {self.metric}={self.best:.4f} "
                        f"at iter={self.best_iteration}"
                    )

        return stop

    def _trigger(self, iteration: int) -> None:
        """Shrink ``config.trainer.max_iter`` to the current iteration so the trainer exits after this step."""
        trainer = getattr(getattr(self, "config", None), "trainer", None)

        if trainer is not None:
            was_frozen = getattr(trainer, "_is_frozen", False)
            if was_frozen:
                trainer._is_frozen = False
            try:
                trainer.max_iter = int(iteration)
            finally:
                if was_frozen:
                    trainer._is_frozen = True
            log.success(f"EarlyStop: training will exit at iter={iteration} (max_iter set to {iteration}).")

        if distributed.is_rank0():
            self._write_state(iteration)

    def _write_state(self, iteration: int) -> None:
        """Write ``early_stop.json`` — the state TrainingReport reads for its triggered-mode plot."""
        state = {
            "triggered": True,
            "criteria": self.metric,
            "patience": self.patience,
            "best_iteration": self.best_iteration,
            "stop_iteration": iteration,
            "last_improved_iteration": self.last_improved_iteration,
            "min_delta": self.min_delta,
            "min_delta_mode": self.min_delta_mode,
            "cumulative_delta": self.cumulative_delta,
        }

        try:
            with open(os.path.join(self._run_dir or os.getcwd(), "early_stop.json"), "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:  # noqa: BLE001
            log.warning(f"EarlyStop: failed to write early_stop.json: {e}")
