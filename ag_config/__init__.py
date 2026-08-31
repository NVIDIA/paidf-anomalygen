# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example AnomalyGen experiment configs, kept at the repo root rather than inside the
``anomalygen`` package because they are dataset-specific samples, not library code. They are
still included in the editable install (see ``pyproject.toml`` ``packages.find``), so
``--recipe=ag_config.<recipe>`` imports without needing ``PYTHONPATH=.``.
"""
