# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation metric registry: metric name -> (valid_kpi dict key, optimisation direction)."""

from __future__ import annotations

METRIC_SPECS: dict[str, tuple[str, str]] = {
    "mnn": ("mnn_score", "max"),
    "nn": ("nn_score", "max"),
    "fid": ("fid", "min"),
    # Composite defect-quality score (see anomalygen.eval.anomaly_quality); higher = better.
    "aq_nn": ("aq_nn", "max"),
    # Individual defect-quality axes (raw per-type macro). The direction here is only used for
    # early-stop selection; watch each axis's actual trend on the training curve.
    "completeness": ("completeness", "max"),
    "precision": ("precision", "max"),
    "boundary_iou": ("boundary_iou", "max"),
}
