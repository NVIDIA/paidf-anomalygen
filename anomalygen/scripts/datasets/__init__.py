# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset preparation CLIs for AnomalyGen fine-tuning.

Each module is a standalone entry point (stdlib + huggingface_hub), run as
``python -m anomalygen.scripts.datasets.<module> <output_dir>`` to produce a ready ``dataset_dir``:

- ``prepare_pcb_defect``           — PCB
- ``prepare_magnetic_tile_defect`` — Magnetic Tile surface
- ``prepare_phone_screen_defect``  — Mobile Phone Screen
"""
