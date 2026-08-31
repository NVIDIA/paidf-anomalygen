# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default per-sample generation params — the single source of truth for the texture SDG pipeline.

Kept dependency-free (no anomalygen/framework imports) so any layer — the inference dataset, the
generation core, the SDG CLI, the model config, and the experiment builder — can import these
without pulling in the heavier ``exp_config`` module or risking an import cycle.
"""

DEFAULT_GUIDANCE = 6.0
DEFAULT_CROP_RATIO = 2.0
DEFAULT_NUM_STEPS = 35
DEFAULT_MAX_INSTANCES = 5
DEFAULT_SHIFT = 5.0

# Base-model size: "nano" (Qwen3-VL-8B reasoner) | "edge" (Nemotron-3 Dense VL 2B). Selects the
# frozen backbone, so the two sizes' checkpoints are not interchangeable.
DEFAULT_MODEL_SIZE = "nano"
MODEL_SIZES = ("nano", "edge")

# Frozen base DCP per size, relative to the repo root; see scripts/download_checkpoints.sh.
BASE_CHECKPOINT_PATHS = {
    "nano": "checkpoints/Cosmos3-Nano",
    "edge": "checkpoints/Cosmos3-Edge",
}

# Iterations below this are not eligible for best_checkpoint.txt. The pick is a plain best-of,
# and nn swings widely before the adapters settle, so an early spike would otherwise beat a
# genuinely better late checkpoint. Only applied when the run was *planned* to reach past it
# — a dry run or a short fine-tune must still produce a pointer, and for those every iteration
# stays eligible.
CKPT_WARMUP_ITER = 7500
