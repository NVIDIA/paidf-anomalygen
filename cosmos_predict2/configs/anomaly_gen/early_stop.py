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

import attrs
from hydra.core.config_store import ConfigStore

from imaginaire.config import TrainerConfig, make_freezable


@attrs.define(slots=False)
class EarlyStopConfig:
    """Config for early stopping, consulted by AnomalyGenTrainer.validate.

    Resolved at runtime by `cosmos_predict2.utils.early_stopper.EarlyStopper`,
    which maps `metric` to the concrete KPI key and optimisation direction
    (FID: lower-is-better; MNN/NN: higher-is-better).
    """

    enabled: bool = False
    # One of: "fid", "mnn", "nn".
    metric: str = "nn"
    # Consecutive validations without improvement before stopping.
    patience: int = 5
    # KPI dict scope to read. "Average" aggregates across anomaly types;
    # set to a specific anomaly name to monitor that type only.
    scope: str = "Average"
    # Minimum change in the score to qualify as an improvement. Interpretation
    # depends on `min_delta_mode`.
    min_delta: float = 0.0
    # How to interpret `min_delta` (mirrors PyTorch Ignite EarlyStopping):
    #   "rel" (default) — relative (fraction of |best|). Threshold = best × (1 ± min_delta).
    #                     e.g. min_delta=0.05 means "improve by ≥5%". Scale-invariant
    #                     across FID (~30) and NN/MNN (~0.7) so the same value works for any.
    #                     Caveat: when |best| is near 0 (e.g. an untrained model's
    #                     initial NN/MNN ≈ 0), the threshold also collapses to ~0
    #                     and any non-zero score crosses it — effectively disabling
    #                     min_delta until best moves away from 0. Use "abs" if your
    #                     metric is expected to live near zero in early training.
    #   "abs"           — absolute units. Threshold = best ± min_delta.
    #                     e.g. min_delta=0.5 with FID means "improve by 0.5 FID points".
    min_delta_mode: str = "rel"
    # If True, `min_delta` defines the change since the last patience reset
    # (the bar stays fixed at the last qualifying best). If False (default),
    # `best` is also updated on no-improvement steps, so the bar creeps with
    # the actual best ever seen — stricter behaviour, same as Ignite default.
    cumulative_delta: bool = False


@make_freezable
@attrs.define(slots=False)
class EarlyStopTrainerConfig(TrainerConfig):
    """TrainerConfig extended with an `early_stop` field.

    Exists so that `cs.store(package="trainer.early_stop", ...)` has a valid
    target and early stopping stays nested under `trainer` (where training-
    control concerns belong).
    """

    early_stop: EarlyStopConfig = attrs.field(factory=EarlyStopConfig)


DefaultEarlyStopConfig: EarlyStopConfig = EarlyStopConfig()


def register_early_stop() -> None:
    cs = ConfigStore.instance()
    cs.store(group="early_stop", package="trainer.early_stop", name="default", node=DefaultEarlyStopConfig)
