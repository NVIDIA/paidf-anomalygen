# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Texture anomaly-inpainting CLIs: ``train.py`` (fine-tune), ``generate.py`` (inference), ``evaluate.py`` (KPI).

Run them by file path (e.g. ``torchrun ... anomalygen/scripts/texture/train.py``), not ``python -m``:
``train.py`` must call ``init_script()`` before ``import anomalygen``, which a ``-m`` launch
(importing the package first) would break.
"""
