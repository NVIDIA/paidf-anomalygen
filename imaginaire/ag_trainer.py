# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict, List, Optional

import torch

from cosmos_predict2.utils.early_stop_plot import plot_early_stop
from cosmos_predict2.utils.early_stopper import EarlyStopper
from cosmos_predict2.utils.metric_specs import METRIC_SPECS
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.model import ImaginaireModel
from imaginaire.utils import distributed, log


# Inverse of METRIC_SPECS: valid_kpi[scope] dict-key → plot metric key.
# Derived so a new metric only needs to be added to metric_specs.py.
_KPI_TO_METRIC = {kpi_key: metric for metric, (kpi_key, _) in METRIC_SPECS.items()}


class AnomalyGenTrainer(ImaginaireTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.early_stopper = None
        # History feeds the diagnostic plot. Kept whether or not early stop is
        # enabled so the plain end-of-run plot still has a full trajectory.
        self._valid_iters: List[int] = []
        self._valid_metrics: Dict[str, List[float]] = {"fid": [], "nn": [], "mnn": []}
        self._max_iter_planned: int = int(config.trainer.max_iter)
        if config.trainer.early_stop.enabled:
            self.early_stopper = EarlyStopper(
                metric=config.trainer.early_stop.metric,
                patience=config.trainer.early_stop.patience,
                scope=config.trainer.early_stop.scope,
                min_delta=config.trainer.early_stop.min_delta,
                min_delta_mode=config.trainer.early_stop.min_delta_mode,
                cumulative_delta=config.trainer.early_stop.cumulative_delta,
            )

    @torch.no_grad()
    def validate(self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0) -> None:
        save_dir = os.path.join(self.config.job.path_local, "valid", str(iteration))
        os.makedirs(save_dir, exist_ok=True)

        model.on_validation_start(
            self.config.dataloader_train.dataset.dataset_dir,
            self.config.dataloader_train.dataset.anomaly_types,
            self.config.dataloader_train.dataset.image_size
        )
        for val_iter, data_batch in enumerate(dataloader_val):
            model.validation_step(data_batch)
        valid_kpi = model.on_validation_end(save_dir)

        self._after_validation(valid_kpi, iteration)

    def _after_validation(self, valid_kpi: Optional[dict], iteration: int) -> None:
        """Post-validation: record history, run early-stop, render plot.

        Plot timing (at most once per run):
          * early-stop triggered → triggered-mode plot at the trigger moment;
          * else, last scheduled validation → plain-mode plot at end of run.

        "Last validation" is detected as `iteration + validation_iter >
        max_iter_planned` so it works even when ``validation_iter`` does not
        divide ``max_iter`` evenly (e.g. max_iter=10300, validation_iter=500
        → final val at iter 10000, never == max_iter).
        """
        is_rank0 = distributed.get_rank() == 0
        if is_rank0 and valid_kpi is not None:
            self._record_valid_metrics(valid_kpi, iteration)

        # should_stop broadcasts across ranks — must be called on all ranks.
        triggered = (
            self.early_stopper is not None
            and self.early_stopper.should_stop(valid_kpi, iteration)
        )

        is_last_validation = (
            iteration + self.config.trainer.validation_iter > self._max_iter_planned
        )
        if is_rank0 and (triggered or is_last_validation):
            self._save_diagnostic_plot()

        if triggered:
            self._request_early_stop(iteration)

    def _record_valid_metrics(self, valid_kpi: dict, iteration: int) -> None:
        avg = valid_kpi.get("Average", {}) or {}
        self._valid_iters.append(iteration)
        for kpi_key, metric_key in _KPI_TO_METRIC.items():
            val = avg.get(kpi_key)
            try:
                val = float(val) if val is not None else float("nan")
            except (TypeError, ValueError):
                val = float("nan")
            self._valid_metrics[metric_key].append(val)

    def _save_diagnostic_plot(self) -> None:
        if not self._valid_iters:
            return
        es = self.early_stopper
        triggered = bool(es and es.triggered)
        kwargs = dict(
            iters=list(self._valid_iters),
            metrics={k: list(v) for k, v in self._valid_metrics.items()},
            max_iter_planned=self._max_iter_planned,
            out_path=os.path.join(self.config.job.path_local, "training_curves.png"),
            triggered=triggered,
        )
        if triggered:
            kwargs.update(
                criteria=es.metric,
                patience=es.patience,
                best_iteration=es.best_iteration,
                stop_iteration=self._valid_iters[-1],
                last_improved_iteration=es.last_improved_iteration,
                min_delta=es.min_delta,
                min_delta_mode=es.min_delta_mode,
                cumulative_delta=es.cumulative_delta,
            )
        try:
            plot_early_stop(**kwargs)
        except Exception as exc:
            # A plotting failure should never crash training.
            log.warning(f"[EarlyStop] failed to write diagnostic plot: {exc}")

    def _request_early_stop(self, iteration: int) -> None:
        """Signal the base trainer to exit after this iteration.

        Base trainer only checks `iteration >= config.trainer.max_iter` to decide
        when to stop, so we reuse that path: shrink `max_iter` to the current
        iteration and the next loop check triggers a clean shutdown with final
        checkpoint save and on_train_end callbacks.

        NOTE: Mutating `config.trainer.max_iter` is intentional — do not "fix"
        by removing or guarding the assignment below. The `_is_frozen` toggle
        bypasses `make_freezable`'s write guard for this single update.
        """
        log.success(
            f"[EarlyStop] Triggering training exit at iter={iteration} "
            f"(original max_iter={self.config.trainer.max_iter})."
        )
        self.config.trainer._is_frozen = False
        self.config.trainer.max_iter = iteration
        self.config.trainer._is_frozen = True
