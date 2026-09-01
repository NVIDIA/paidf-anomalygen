# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for the fine-tuned trainable parameter subset.

These prefixes name the parameters AnomalyGen trains on top of the frozen base network: the LoRA
adapters, the per-defect-type inpaint embedding, the learnable text prompt, and the learnable vision
prompt. The optimizer, the checkpointer, and the inference loader all read the subset from here so
their lists can't drift apart (the original bug was ``save_keys_filter`` missing ``text_prompt_emb``).
"""

from __future__ import annotations

# The trained subset, as module-path-segment prefixes.
TRAINED_KEY_PREFIXES = ("lora_", "inpaint_class_emb", "text_prompt_emb", "vision_prompt_emb")

# The trained subset plus the EMA shadow: keys absent from the base checkpoint, skipped on warm-start.
WARM_START_SKIP_PREFIXES = ("net_ema.", *TRAINED_KEY_PREFIXES)


def match_keys(keys, prefixes):
    """Keys with a dotted-path segment starting with any prefix.

    Matches whole segments, not substrings: ``lora_`` hits ``...lora_A.weight`` but not
    ``...foolora_x.weight``.
    """
    return {k for k in keys if any(seg.startswith(p) for seg in k.split(".") for p in prefixes)}
