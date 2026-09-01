# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared data helpers for the anomaly pipeline: caption building, tokenizer/prompt utilities,
image/mask directory pairing, and source (conditioning) item construction."""

from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence, Tuple

import torch

# Single editing-instruction caption shared by training and inference.
CAPTION_TEMPLATE = (
    "Add {defect} to a close-up look of a {texture}. "
    "Defective, Non-conforming, Substandard, Irregular, Damaged, Misaligned, "
    "Incomplete, Malformed, Contaminated, Under-processed, Faulty, Blemished, Expired."
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg")
MASK_FG_THRESHOLD = 128  # Foreground threshold for binarizing a uint8 mask (0–255).


def build_caption(defect: str, texture: str) -> str:
    return CAPTION_TEMPLATE.format(
        defect=defect.replace("-", " ").replace("_", " "),
        texture=texture.replace("-", " ").replace("_", " "),
    )


# Each half must be one safe path segment: an anomaly type becomes a directory name downstream.
# Rejected, never sanitized — as scripts/skill_utility/set_pipeline_vars.sh does for --name/--task.
_ANOMALY_TYPE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\+[A-Za-z0-9][A-Za-z0-9._-]*")


def validate_anomaly_type(anomaly_type: str, *, field: str = "anomaly_type") -> str:
    """Return ``anomaly_type`` if it is one safe ``{texture}+{defect}`` pair, else raise ``ValueError``.

    Values come from a dataset spec or a generation manifest and are joined onto an output directory.
    ``pathlib`` drops the base for an absolute segment and follows ``..`` for a relative one, so the
    check must precede the join. ``field`` names the source key in the error.
    """
    if not isinstance(anomaly_type, str) or not _ANOMALY_TYPE_RE.fullmatch(anomaly_type):
        raise ValueError(
            f"invalid anomaly type in {field}: {anomaly_type!r}. Expected '{{texture}}+{{defect}}', "
            "each half alphanumeric plus '._-'. It becomes a directory name, so a separator, a "
            "leading '.', or an absolute path would write outside the run's output directory."
        )
    return anomaly_type


def caption_for_anomaly_type(anomaly_type: str) -> str:
    """Editing-instruction caption for an ``anomaly_type`` (``"{texture}+{defect}"`` or a bare
    texture). Single source of truth shared by inpaint generation and the guardrail text check,
    so the guarded text is exactly the text the model conditions on. A bare token (no ``"+"``)
    is used as both texture and defect."""
    texture, _, defect = anomaly_type.partition("+")
    return build_caption(defect=defect or texture, texture=texture)


def resolve_word_token_id(tokenizer, word: str) -> int:
    """Resolve a single representative token id for ``word`` (the prompt pad/init word).

    Tries a direct vocabulary lookup first; if that misses (the word is not a standalone
    vocab entry, e.g. a multi-piece BPE word), falls back to the last piece of the encoded
    word. Used to pad the learnable text-prompt region with the ``"anomaly"`` token id so
    the dataset's padded captions and the model-side template init agree.
    """
    tid = tokenizer.convert_tokens_to_ids(word)
    unk = getattr(tokenizer, "unk_token_id", None)
    if isinstance(tid, int) and tid >= 0 and tid != unk:
        return tid
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Cannot resolve a token id for pad/init word {word!r}.")
    return int(ids[-1])


def pad_or_truncate(ids: Sequence[int], length: int, pad_token_id: int) -> List[int]:
    """Right-pad (with ``pad_token_id``) or truncate ``ids`` to exactly ``length`` tokens."""
    out = list(ids[:length])
    if len(out) < length:
        out = out + [int(pad_token_id)] * (length - len(out))
    return out


def build_source_item(
    base: torch.Tensor,
    m3: torch.Tensor,
    background_dropout: bool = False,
    *,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Build the source item from ``base`` (clean/target image) and ``m3`` (defect mask, 1 inside).

    The defect region is ALWAYS uniform [-1, 1] noise — that is the single source encoding shared by
    training and inference. ``background_dropout`` only changes what surrounds it: the clean
    background (False) or a constant -1, i.e. black (True), a training-time augmentation that
    occasionally hides the background so the model cannot lean on it as a shortcut. Inference always
    passes False.

    ``seed`` draws the noise from a private generator, so it is reproducible and independent of
    rank, batch composition and guardrail skips. ``None`` (the training path) uses the global RNG,
    i.e. fresh noise every epoch. The draw is on CPU because CUDA's RNG is not arch-invariant.

    Returns a NEW tensor; ``base`` is never mutated.
    """
    if seed is None:
        noise = torch.rand_like(base) * 2.0 - 1.0
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.rand(base.shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0
        noise = noise.to(device=base.device, dtype=base.dtype)
    background = torch.full_like(base, -1.0) if background_dropout else base
    return background * (1.0 - m3) + noise * m3


def list_image_mask_pairs(img_dir: str, mask_dir: str, mask_suffix: str = "") -> List[Tuple[str, str]]:
    """List ``(image_path, mask_path)`` pairs for images in ``img_dir``.

    The mask name is the image stem plus ``mask_suffix`` and the image extension, with a
    ``.png`` fallback. The mask path is returned even if it does not exist. Returns an
    empty list when ``img_dir`` is not a directory.
    """
    pairs: List[Tuple[str, str]] = []
    if not os.path.isdir(img_dir):
        return pairs
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith(IMAGE_EXTS):
            continue
        stem, ext = os.path.splitext(fn)
        mask_path = os.path.join(mask_dir, f"{stem}{mask_suffix}{ext}")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, f"{stem}{mask_suffix}.png")

        pairs.append((os.path.join(img_dir, fn), mask_path))

    return pairs
