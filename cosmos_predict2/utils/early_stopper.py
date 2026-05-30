# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
from typing import Optional

import torch
import torch.distributed as dist

from cosmos_predict2.utils.metric_specs import METRIC_SPECS
from imaginaire.utils import distributed, log


class EarlyStopper:
    """Track a monitored validation KPI and decide when training should stop.

    Only rank 0 typically computes the KPI dict (others pass None). The
    decision is made on rank 0 and broadcast as a single int tensor so every
    rank returns the same bool — required under DDP/FSDP to avoid hanging on
    later collectives.

    State consistency under DDP:
        After ``should_stop`` returns, the *only* rank-consistent piece of
        state is its return value (the broadcast bool). Internal attributes
        (``best``, ``best_iteration``, ``wait``, ``triggered``) are updated
        on rank 0 only — non-rank-0 instances keep their default values
        (``best=±inf``, ``wait=0``, ``triggered=False``). Read those fields
        only on rank 0 (e.g. for diagnostics, plotting); never branch on
        them across ranks without going through another collective.
    """

    def __init__(
        self,
        metric: str,
        patience: int,
        scope: str = "Average",
        min_delta: float = 0.0,
        min_delta_mode: str = "rel",
        cumulative_delta: bool = False,
    ) -> None:
        if metric not in METRIC_SPECS:
            raise ValueError(
                f"Unknown early-stop metric '{metric}'. "
                f"Choose one of {list(METRIC_SPECS)}."
            )
        if patience < 1:
            raise ValueError(f"patience must be ≥ 1, got {patience}")
        if min_delta < 0.0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        if min_delta_mode not in ("abs", "rel"):
            raise ValueError(
                f"min_delta_mode must be 'abs' or 'rel', got '{min_delta_mode}'"
            )
        self.metric = metric
        self._kpi_key, self.mode = METRIC_SPECS[metric]
        self.patience = patience
        self.scope = scope
        self.min_delta = min_delta
        self.min_delta_mode = min_delta_mode
        self.cumulative_delta = cumulative_delta
        self.best = -math.inf if self.mode == "max" else math.inf
        self.best_iteration: Optional[int] = None
        self.last_improved_iteration: Optional[int] = None
        self.wait = 0
        self.triggered = False

    def _improvement_threshold(self) -> float:
        """Score threshold a new value must cross to count as improvement.

        Mirrors the convention from PyTorch Ignite's EarlyStopping handler:
        the sign of `min_delta` is flipped for `mode=='min'` so the same
        formula handles both directions.
        """
        signed = -self.min_delta if self.mode == "min" else self.min_delta
        if self.min_delta_mode == "abs":
            return self.best + signed
        return self.best * (1 + signed)

    def should_stop(self, valid_kpi: Optional[dict], iteration: int) -> bool:
        stop = 0
        if distributed.get_rank() == 0 and valid_kpi is not None:
            score = valid_kpi.get(self.scope, {}).get(self._kpi_key)
            if score is None or (isinstance(score, float) and math.isnan(score)):
                log.warning(
                    f"[EarlyStop] {self.scope}/{self._kpi_key} "
                    f"not available at iter={iteration}; skipping."
                )
            elif math.isinf(self.best):
                # First valid score: nothing to compare against; bootstrap.
                self.best = score
                self.best_iteration = iteration
                self.last_improved_iteration = iteration
                log.info(
                    f"[EarlyStop] iter={iteration} {self.metric}={score:.4f} (initial)"
                )
            else:
                threshold = self._improvement_threshold()
                improved = (
                    score > threshold if self.mode == "max" else score < threshold
                )
                if improved:
                    log.info(
                        f"[EarlyStop] iter={iteration} {self.metric}={score:.4f} "
                        f"improved over best={self.best:.4f}"
                    )
                    self.best = score
                    self.best_iteration = iteration
                    self.last_improved_iteration = iteration
                    self.wait = 0
                else:
                    # Track the absolute best when not cumulative_delta — so
                    # the threshold creeps along with the actual best ever
                    # seen (stricter, matches Ignite's default).
                    if not self.cumulative_delta:
                        is_strict_better = (
                            (self.mode == "max" and score > self.best)
                            or (self.mode == "min" and score < self.best)
                        )
                        if is_strict_better:
                            self.best = score
                            self.best_iteration = iteration
                    self.wait += 1
                    log.info(
                        f"[EarlyStop] iter={iteration} {self.metric}={score:.4f} "
                        f"no improve ({self.wait}/{self.patience}, "
                        f"best={self.best:.4f} at iter={self.best_iteration})"
                    )
                    if self.wait >= self.patience:
                        stop = 1
                        self.triggered = True
                        log.success(
                            f"[EarlyStop] triggered at iter={iteration}, "
                            f"best {self.metric}={self.best:.4f} "
                            f"at iter={self.best_iteration}"
                        )

        if distributed.get_world_size() > 1:
            stop_tensor = torch.tensor([stop], device="cuda")
            dist.broadcast(stop_tensor, src=0)
            stop = int(stop_tensor.item())
        return bool(stop)
