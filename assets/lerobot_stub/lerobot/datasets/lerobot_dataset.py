# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stub of ``lerobot.datasets.lerobot_dataset``.

cosmos-framework's action datasets import ``LeRobotDataset`` / ``LeRobotDatasetMetadata`` at module
load (``cosmos3_action_lerobot``), but only *instantiate* them inside methods at runtime. These
placeholders satisfy the import; constructing one raises, since assets/lerobot_stub is import-only.
"""


class LeRobotDataset:
    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN204
        raise RuntimeError(
            "lerobot stub: LeRobotDataset was constructed, but assets/lerobot_stub is an import-only stub."
        )


class LeRobotDatasetMetadata:
    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN204
        raise RuntimeError(
            "lerobot stub: LeRobotDatasetMetadata was constructed, but assets/lerobot_stub is an import-only stub."
        )
