# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the post-generation image content-safety guardrail.

These tests exercise the guardrail wiring without loading the real SigLIP
classifier or requiring a GPU. A fake runner stands in for the real
``GuardrailRunner``: it implements ``run_safety_check`` and flags
predominantly-red images as unsafe so behaviour is deterministic.

Verified behaviour (per the feature spec):
  * an unsafe image is replaced in place with a black image, and
  * its per-image verdict is recorded as ``False`` (-> ``guardrail_pass=0``).
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

presets = importlib.import_module("cosmos_predict2.auxiliary.guardrail.common.presets")
inference_utils = importlib.import_module(
    "cosmos_predict2.inference.anomaly_gen.inference_anomaly_diffusion_utils"
)
multiview_utils = importlib.import_module(
    "cosmos_predict2.inference.anomaly_gen.multiview_inference_utils"
)
vision_encoder = importlib.import_module(
    "cosmos_predict2.auxiliary.guardrail.video_content_safety_filter.vision_encoder"
)


class FakeRunner:
    """Stand-in for GuardrailRunner: flags predominantly-red frames as unsafe."""

    def __init__(self):
        self.num_calls = 0

    def run_safety_check(self, frames):
        self.num_calls += 1
        frame = np.asarray(frames[0])
        is_unsafe = frame[..., 0].mean() > 200 and frame[..., 1].mean() < 50
        return (not is_unsafe), ("unsafe content detected" if is_unsafe else "safe")


def _green():
    return Image.new("RGB", (8, 8), (0, 255, 0))


def _red():
    return Image.new("RGB", (8, 8), (255, 0, 0))


def _blue():
    return Image.new("RGB", (8, 8), (0, 0, 255))


def _is_black(image):
    return int(np.asarray(image).sum()) == 0


def _model_with_runner(runner):
    return SimpleNamespace(pipe=SimpleNamespace(image_guardrail_runner=runner))


# --------------------------------------------------------------------------- #
# presets.run_image_guardrail
# --------------------------------------------------------------------------- #
def test_run_image_guardrail_safe_pil():
    assert presets.run_image_guardrail(_green(), FakeRunner()) is True


def test_run_image_guardrail_unsafe_pil():
    assert presets.run_image_guardrail(_red(), FakeRunner()) is False


def test_run_image_guardrail_accepts_ndarray():
    arr = np.zeros((8, 8, 3), dtype=np.uint8)  # black -> safe
    assert presets.run_image_guardrail(arr, FakeRunner()) is True


# --------------------------------------------------------------------------- #
# single-view: _apply_image_guardrail
# --------------------------------------------------------------------------- #
def test_apply_guardrail_replaces_unsafe_with_black_and_flags():
    images = [_green(), _red(), _blue()]
    flags = inference_utils._apply_image_guardrail(_model_with_runner(FakeRunner()), images)

    assert flags == [True, False, True]
    # Unsafe image (index 1) replaced with a black image of the same size.
    assert _is_black(images[1])
    assert images[1].size == (8, 8)
    # Safe images are left untouched.
    assert not _is_black(images[0])
    assert not _is_black(images[2])


def test_apply_guardrail_disabled_when_runner_is_none():
    images = [_red(), _green()]
    flags = inference_utils._apply_image_guardrail(_model_with_runner(None), images)

    assert flags == [True, True]
    # Nothing is rewritten when the guardrail is disabled.
    assert not _is_black(images[0])


def test_apply_guardrail_when_model_has_no_pipe():
    images = [_red()]
    flags = inference_utils._apply_image_guardrail(SimpleNamespace(), images)

    assert flags == [True]
    assert not _is_black(images[0])


def test_apply_guardrail_all_safe_keeps_everything():
    runner = FakeRunner()
    images = [_green(), _blue()]
    flags = inference_utils._apply_image_guardrail(_model_with_runner(runner), images)

    assert flags == [True, True]
    assert runner.num_calls == 2


# --------------------------------------------------------------------------- #
# multi-view: _apply_multiview_image_guardrail
# --------------------------------------------------------------------------- #
def test_apply_multiview_guardrail_nested_flags_and_black():
    views = [[_green(), _red()], [_red(), _green()]]
    flags = multiview_utils._apply_multiview_image_guardrail(_model_with_runner(FakeRunner()), views)

    assert flags == [[True, False], [False, True]]
    assert _is_black(views[0][1])
    assert _is_black(views[1][0])
    assert not _is_black(views[0][0])
    assert not _is_black(views[1][1])


def test_apply_multiview_guardrail_disabled_when_runner_is_none():
    views = [[_red()], [_green()]]
    flags = multiview_utils._apply_multiview_image_guardrail(_model_with_runner(None), views)

    assert flags == [[True], [True]]
    assert not _is_black(views[0][0])


# --------------------------------------------------------------------------- #
# CSV schema: guardrail_pass column is recorded
# --------------------------------------------------------------------------- #
def test_sdg_result_csv_header_has_guardrail_pass():
    sdg = importlib.import_module("scripts.anomaly_gen.synthetic_dataset_generation")
    assert sdg.CSV_HEADER[-1] == "guardrail_pass"


# --------------------------------------------------------------------------- #
# SigLIPEncoder.encode_image — regression guard for the transformers>=5 fix
# --------------------------------------------------------------------------- #
# Without the `.pooler_output` handling, `get_image_features` returning a
# BaseModelOutputWithPooling (transformers >= 5) makes `encode_image` raise; the
# exception is then swallowed by `is_safe_frames` and the guardrail silently
# passes every image. These tests pin both return shapes with no GPU / no weights.
class _ProcessorOutput(dict):
    """Dict-like stand-in for a transformers BatchEncoding (supports **unpack)."""

    def to(self, *args, **kwargs):
        return self


class _FakeProcessor:
    def __call__(self, images=None, return_tensors=None):
        return _ProcessorOutput(pixel_values=torch.zeros(1, 3, 4, 4))


class _FakeModel:
    """Stands in for SiglipModel; get_image_features returns a preset value."""

    def __init__(self, image_features):
        self._image_features = image_features

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def get_image_features(self, **inputs):
        return self._image_features


def _make_encoder(monkeypatch, image_features):
    monkeypatch.setattr(
        vision_encoder, "SiglipModel",
        SimpleNamespace(from_pretrained=lambda *a, **k: _FakeModel(image_features)),
    )
    monkeypatch.setattr(
        vision_encoder, "SiglipProcessor",
        SimpleNamespace(from_pretrained=lambda *a, **k: _FakeProcessor()),
    )
    return vision_encoder.SigLIPEncoder(checkpoint_dir="unused", device="cpu")


def test_encode_image_uses_pooler_output_for_transformers5(monkeypatch):
    # transformers >= 5: get_image_features returns an object with .pooler_output.
    pooled = torch.tensor([[3.0, 4.0]])
    output = SimpleNamespace(pooler_output=pooled)
    encoder = _make_encoder(monkeypatch, output)

    feats = encoder.encode_image(_green())

    # Pooled embedding, L2-normalized: [3, 4] / 5 == [0.6, 0.8].
    assert isinstance(feats, torch.Tensor)
    assert torch.allclose(feats, torch.tensor([[0.6, 0.8]]), atol=1e-6)


def test_encode_image_accepts_plain_tensor_for_transformers4(monkeypatch):
    # transformers < 5: get_image_features returns a plain tensor (isinstance guard
    # keeps this path a no-op). Both paths must yield the same normalized vector.
    encoder = _make_encoder(monkeypatch, torch.tensor([[3.0, 4.0]]))

    feats = encoder.encode_image(_green())

    assert isinstance(feats, torch.Tensor)
    assert torch.allclose(feats, torch.tensor([[0.6, 0.8]]), atol=1e-6)
