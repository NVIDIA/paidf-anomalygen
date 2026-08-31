# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the generate.py content-safety guardrail adapter.

Pure plumbing (caption, PIL<->numpy, block-record) is tested directly on CPU with real
inputs and no mocks. Behavior that needs the real guardrail models (Qwen3Guard, RetinaFace)
is tagged ``@pytest.mark.gpu`` and auto-skipped when no CUDA device is present.
"""

import nltk
import numpy as np
import pytest
from huggingface_hub import constants as hf_constants
from PIL import Image

import anomalygen.inference.guardrail as guardrail
from anomalygen.data.utils import build_caption, caption_for_anomaly_type
from anomalygen.inference.guardrail import (
    BLOCKED_CSV_HEADER,
    GUARDRAIL_REPO_DIR,
    _frames_to_pil,
    _pil_to_frames,
    _read_guardrail_corpora_for_nltk,
    blocked_row,
    create_guardrail_runners,
    guard_image,
)


def test_caption_for_anomaly_type_texture_plus_defect():
    assert caption_for_anomaly_type("phone_screen+scratch") == build_caption(defect="scratch", texture="phone_screen")


def test_caption_for_anomaly_type_bare_texture():
    # No "+": the bare token is used as both texture and defect (matches inpaint.py).
    assert caption_for_anomaly_type("phone_screen") == build_caption(defect="phone_screen", texture="phone_screen")


def test_pil_to_frames_shape_and_dtype():
    img = Image.new("RGB", (7, 5), (10, 20, 30))  # size = (W=7, H=5)
    frames = _pil_to_frames(img)
    assert frames.shape == (1, 5, 7, 3)  # [1, H, W, C]
    assert frames.dtype == np.uint8


def test_pil_to_frames_returns_writable_array():
    # RetinaFace face-blur writes into the frame in place, so the buffer must be writable.
    frames = _pil_to_frames(Image.new("RGB", (7, 5), (10, 20, 30)))
    assert frames.flags.writeable is True


def test_pil_frames_roundtrip_is_pixel_exact():
    arr = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)  # H=5, W=7
    img = Image.fromarray(arr, mode="RGB")
    back = _frames_to_pil(_pil_to_frames(img))
    assert back.size == img.size
    assert np.array_equal(np.asarray(back), arr)


def test_frames_to_pil_rejects_bad_shape():
    with pytest.raises(ValueError):
        _frames_to_pil(np.zeros((5, 7, 3), dtype=np.uint8))  # missing frame axis (ndim != 4)
    with pytest.raises(ValueError):
        _frames_to_pil(np.zeros((2, 5, 7, 3), dtype=np.uint8))  # multi-frame (shape[0] != 1)


def test_blocked_row_shape_and_fields():
    row = blocked_row(5, 3, "phone_screen+scratch", "img.png", "mask.png", "image", "bad words")
    assert row["sort_key"] == [5, 3]  # (index, output_idx) — keeps per-output rows distinct
    assert set(row["row"]) == set(BLOCKED_CSV_HEADER)
    assert row["row"]["index"] == 5
    assert row["row"]["output_idx"] == 3
    assert row["row"]["guardrail"] == "image"
    assert row["row"]["message"] == "bad words"


def test_nltk_allowlist_gains_the_guardrail_model_dir(tmp_path, monkeypatch):
    """The guardrail's corpora live in the HF cache, which nltk >= 3.10 refuses to read.

    CPU-only on purpose: the end-to-end guardrail test below needs a GPU and is skipped in CI,
    so without this the nltk fix would have no automated coverage at all — and a future nltk
    bump would silently reintroduce the PermissionError it exists to prevent.
    """

    model_dir = tmp_path / GUARDRAIL_REPO_DIR
    model_dir.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(nltk.data, "path", list(nltk.data.path))

    assert _read_guardrail_corpora_for_nltk() == str(model_dir)
    assert str(model_dir) in nltk.data.path

    # Idempotent: a second call must not append a duplicate.
    before = list(nltk.data.path)
    assert _read_guardrail_corpora_for_nltk() == str(model_dir)
    assert nltk.data.path == before


def test_nltk_allowlist_is_a_noop_when_the_model_is_absent(tmp_path, monkeypatch):

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))  # no model dir inside
    monkeypatch.setattr(nltk.data, "path", list(nltk.data.path))
    before = list(nltk.data.path)

    assert _read_guardrail_corpora_for_nltk() is None
    assert nltk.data.path == before


def test_nltk_allowlist_reads_the_env_not_the_frozen_constant(tmp_path, monkeypatch):
    """HF_HUB_CACHE must win over huggingface_hub.constants.

    generate.py sets HF_HUB_CACHE at import time, which can land after huggingface_hub has
    already frozen its constants — reading the constant then points at the wrong cache.
    """

    (tmp_path / GUARDRAIL_REPO_DIR).mkdir()
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", "/nonexistent/frozen/cache")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(nltk.data, "path", list(nltk.data.path))

    assert _read_guardrail_corpora_for_nltk() == str(tmp_path / GUARDRAIL_REPO_DIR)


@pytest.mark.gpu
def test_guardrail_runners_check_and_guard():
    # Exercise the same factory generate.py uses — it also widens nltk's allowlist.
    text_guardrail, video_guardrail = create_guardrail_runners(offload_model_to_cpu=False)

    # A benign templated caption must pass the text guardrail.
    is_safe, _message = text_guardrail.run_safety_check(caption_for_anomaly_type("phone_screen+scratch"))
    assert is_safe is True

    # A plain, faceless image passes through the face-blur postprocessor unchanged in size.
    out, _msg = guard_image(video_guardrail, Image.new("RGB", (64, 48), (127, 127, 127)))
    assert isinstance(out, Image.Image)
    assert out.size == (64, 48)


# --- enforcement capability vs the operator's flag ------------------------------------------------
# A GuardrailRunner with an empty safety_models list always answers "safe": its postprocessors (face
# blur) still run, but nothing can be blocked. timing_summary.json records guardrail_enabled, which
# a reader would take as evidence screening happened, so capability is reported separately.


class _Runner:
    def __init__(self, safety_models):
        self.safety_models = safety_models


def test_runner_with_no_safety_model_is_not_enforcing():
    """The current state of the framework's video preset — face blur only, nothing can deny."""
    assert guardrail.is_enforcing(_Runner([])) is False


def test_runner_with_a_safety_model_is_enforcing():
    assert guardrail.is_enforcing(_Runner([object()])) is True


def test_runner_without_the_attribute_is_not_enforcing():
    """Absence of evidence is reported as not-enforcing, never as enforcing."""

    class _Bare:
        pass

    assert guardrail.is_enforcing(_Bare()) is False
