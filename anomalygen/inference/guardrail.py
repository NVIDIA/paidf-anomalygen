# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PIL glue for Cosmos 3's content-safety guardrail, for the standalone texture SDG script.

The framework's ``GuardrailRunner`` (``cosmos_framework.auxiliary.guardrail.common.core``) already
is the guardrail abstraction: ``generate.py`` builds the text and image runners directly from
``presets.create_{text,video}_guardrail_runner`` and calls ``run_safety_check`` on them. This module
only adds what the framework does not provide for this repo: the ``[C,T,H,W]``-float API is
video-tensor oriented, whereas our backend produces PIL images — so :func:`guard_image` converts a
PIL image to the ``[1,H,W,C]`` uint8 frames the runner expects, runs it, and converts back. It also
holds the blocked-sample manifest helpers, and :func:`create_guardrail_runners`. ``cosmos_framework``
is imported only inside that function, so the module stays importable on CPU without downloading any
models; everything else takes the runner as a duck-typed argument.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import nltk
import numpy as np
import PIL.Image
from huggingface_hub import constants as hf_constants

# Columns of the separate blocked-sample manifest (guardrail_blocked.csv). Kept out of
# texture_ft_generation_result.csv because downstream (pseudo_label/filter) reads that manifest
# by output_filename and would choke on rows pointing at files that were never saved.
BLOCKED_CSV_HEADER = [
    "index",
    "output_idx",
    "anomaly_type",
    "image_filename",
    "mask_filename",
    "guardrail",
    "message",
]


GUARDRAIL_REPO_DIR = "models--nvidia--Cosmos-Guardrail1"


def _read_guardrail_corpora_for_nltk() -> str | None:
    """Let nltk read the guardrail's corpora out of the Hugging Face cache.

    nltk >= 3.10 fixed CVE-2026-54293 (path traversal in ``nltk.data.load()``) by restricting
    loads to an allowlist built from ``NLTK_DATA`` and ``nltk.data.path``. Cosmos-Guardrail1 ships
    its corpora inside the HF cache, which is not on that list, so the guardrail dies with
    ``PermissionError: Security Violation [pathsec.open]``. Add just that one model directory —
    not the whole cache — so the traversal fix keeps applying everywhere else.

    The cache location is read from the environment on every call, not from
    ``huggingface_hub.constants``: those are frozen when that module is first imported, and
    generate.py does ``os.environ.setdefault("HF_HUB_CACHE", <repo>/checkpoints/hf)`` at import,
    which can land afterwards. Reading the constant would then point at the wrong cache.

    Idempotent: returns the model directory whenever it exists, whether this call appended it or a
    previous one already did. Returns None only when the directory is absent.
    """
    cache = os.environ.get("HF_HUB_CACHE")
    if not cache:
        home = os.environ.get("HF_HOME")
        cache = os.path.join(home, "hub") if home else hf_constants.HF_HUB_CACHE
    model_dir = pathlib.Path(cache) / GUARDRAIL_REPO_DIR
    if not model_dir.is_dir():
        return None
    entry = str(model_dir)
    if entry not in nltk.data.path:
        nltk.data.path.append(entry)
    return entry


def create_guardrail_runners(*, offload_model_to_cpu: bool = False) -> tuple[Any, Any]:
    """Build the framework's ``(text, video)`` guardrail runners, nltk allowlist widened first.

    ``cosmos_framework`` is imported here rather than at module scope so this module stays
    importable on CPU without pulling the framework in.
    """
    # Imported here, not at module scope, for the reason in the module docstring: this module stays
    # importable on CPU without pulling the framework in. ``log`` comes from the framework too.
    from cosmos_framework.auxiliary.guardrail.common import presets
    from cosmos_framework.utils import log

    _read_guardrail_corpora_for_nltk()
    text_guardrail = presets.create_text_guardrail_runner(offload_model_to_cpu=offload_model_to_cpu)
    video_guardrail = presets.create_video_guardrail_runner(offload_model_to_cpu=offload_model_to_cpu)
    if not is_enforcing(video_guardrail):
        log.warning(
            "[guardrail] image content-safety screening is NOT enforcing: the framework preset "
            "supplies no safety model, so no generated image can be blocked. Face blurring still "
            "runs. Text screening is unaffected. Recorded as image_guardrail_enforcing=false in the "
            "run summary."
        )
    return text_guardrail, video_guardrail


def is_enforcing(runner: Any) -> bool:
    """Whether ``runner`` can actually reach a *deny* decision.

    A ``GuardrailRunner`` with an empty ``safety_models`` list always returns ``is_safe=True`` — its
    postprocessors (face blur) still run, but nothing can block. That is the current state of the
    framework's video preset, where the content-safety filter is commented out upstream.

    This matters beyond a log line: ``timing_summary.json`` records ``guardrail_enabled``, and a
    reader would reasonably take that as evidence screening occurred. Reporting a flag the operator
    set, rather than a capability the run actually had, is the part worth fixing here — whether the
    upstream model comes back is not ours to decide.
    """
    return bool(getattr(runner, "safety_models", None))


def _pil_to_frames(img: PIL.Image.Image) -> np.ndarray:
    """PIL image -> a single-frame ``[1, H, W, C]`` uint8 RGB array for the guardrail runner."""
    return np.array(img.convert("RGB"), dtype=np.uint8)[None, ...]


def _frames_to_pil(frames: np.ndarray) -> PIL.Image.Image:
    """``[1, H, W, C]`` uint8 array -> PIL RGB image (inverse of :func:`_pil_to_frames`)."""
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(f"Expected [1, H, W, C] frames, got shape {arr.shape}")
    return PIL.Image.fromarray(arr[0].astype(np.uint8), mode="RGB")


def guard_image(video_guardrail: Any, img: PIL.Image.Image) -> tuple[PIL.Image.Image | None, str]:
    """Run the image guardrail (content-safety check + face-blur) on a PIL image.

    ``video_guardrail`` is a framework ``GuardrailRunner`` (from
    ``presets.create_video_guardrail_runner``). Returns ``(image, message)``: the (possibly
    face-blurred) image, or ``None`` if the content-safety check blocks it. Blocking is unreachable
    with the current preset (its ``safety_models`` list is empty — only the face-blur postprocessor
    runs), but handled for parity with the framework's video guardrail.
    """
    frames = _pil_to_frames(img)
    is_safe, message = video_guardrail.run_safety_check(frames)
    if not is_safe:
        return None, message
    return _frames_to_pil(video_guardrail.postprocess(frames)), message


def blocked_row(
    index: Any,
    output_idx: Any,
    anomaly_type: str,
    image_filename: str,
    mask_filename: str,
    guardrail: str,
    message: str,
) -> dict:
    """Build a ``{"sort_key", "row"}`` payload for the blocked-sample manifest, matching the merge
    shape used for the main CSV so ``_merge_rank_rows`` can gather and order it across ranks.

    ``output_idx`` is the 0-based generated-output index for a per-output image block, or ``-1`` for a
    whole-sample text block (which skips every output of the sample). It is part of the sort key so a
    sample with several blocked outputs yields distinct, individually-identifiable rows.
    """
    return {
        "sort_key": [int(index), int(output_idx)],
        "row": {
            "index": index,
            "output_idx": output_idx,
            "anomaly_type": anomaly_type,
            "image_filename": image_filename,
            "mask_filename": mask_filename,
            "guardrail": guardrail,
            "message": message,
        },
    }
