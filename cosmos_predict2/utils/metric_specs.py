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

"""Single source of truth for metrics this project early-stops / reports on.

Maps the user-facing metric name to ``(valid_kpi dict key, optimisation
direction)``. Keep the dict-keys in sync with what
``cosmos_predict2.metrics.utils.compute_kpi`` writes into
``valid_kpi["Average"]``. Adding a new metric is a one-place change here;
``EarlyStopper``, ``AnomalyGenTrainer`` and ``plot_early_stop`` derive their
own lookups from this constant.

Kept torch-free so the diagnostic plot can be exercised standalone.
"""
from __future__ import annotations


METRIC_SPECS: dict[str, tuple[str, str]] = {
    "mnn": ("mnn_score", "max"),
    "nn":  ("nn_score",  "max"),
    "fid": ("cradio_v3_base_fid", "min"),
}
