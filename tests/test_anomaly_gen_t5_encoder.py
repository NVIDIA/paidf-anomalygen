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
"""Regression tests for the t5_model_name text-encoder selection.

Locks in the fix for the eager t5-11b load: the AnomalyGen pipeline must load
the encoder selected by ag_config.t5_model_name (e.g. t5-large) directly, and
must NOT eagerly load the ~45 GB t5-11b default only to discard it. Covers
from_config (records the loaded path), the from_anomaly_gen_config swap guard,
and that both the single- and multi-view model init sites forward the choice.

These exercise the real methods with the heavy model loads mocked out, so they
run on CPU without any checkpoints.
"""
import contextlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cosmos_predict2.pipelines.anomaly_gen as ag  # noqa: E402


def _mock_from_config_internals(monkeypatch):
    """Stub every heavy call in from_config so it runs on CPU. Returns a list
    that records each CosmosT5TextEncoder(**kwargs) construction."""
    te_calls = []

    def fake_te(*args, **kwargs):
        te_calls.append(kwargs)
        return MagicMock()

    STATE_CH = 16
    tokenizer = MagicMock()
    tokenizer.latent_ch = STATE_CH
    conditioner = MagicMock()
    conditioner.parameters.return_value = []
    conditioner.embedders = {"text": MagicMock()}
    dit = MagicMock()
    dit.eval.return_value = dit
    dit.to.return_value = dit

    monkeypatch.setattr(ag, "CosmosT5TextEncoder", MagicMock(side_effect=fake_te))
    monkeypatch.setattr(ag, "instantiate",
                        MagicMock(side_effect=[tokenizer, conditioner, dit]))
    monkeypatch.setattr(ag, "load_state_dict", MagicMock(return_value={}))
    monkeypatch.setattr(ag, "init_weights_on_device",
                        lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(ag.torch.cuda, "empty_cache", lambda: None)
    return te_calls, STATE_CH


def _make_config(state_ch):
    cfg = MagicMock()
    cfg.precision = "bfloat16"
    cfg.sigma_data = 1.0  # RectifiedFlowScaling asserts sigma_data == 1.0
    cfg.timestamps.t_min = 0.0
    cfg.timestamps.t_max = 1.0
    cfg.timestamps.order = 2.0
    cfg.rectified_flow_t_scaling_factor = 1.0
    cfg.state_ch = state_ch
    cfg.guardrail_config.enabled = False
    cfg.guardrail_config.image_enabled = False
    return cfg


def test_from_config_loads_selected_encoder_and_records_path(monkeypatch):
    te_calls, state_ch = _mock_from_config_internals(monkeypatch)
    cfg = _make_config(state_ch)

    pipe = ag.AnomalyGenPipeline.from_config(
        cfg, dit_path="dummy/dit.pt",
        text_encoder_path="checkpoints/google-t5/t5-large",
    )

    # The encoder is loaded from the requested path, and the path is recorded.
    assert len(te_calls) == 1
    assert te_calls[0]["cache_dir"] == "checkpoints/google-t5/t5-large"
    assert pipe.text_encoder_path == "checkpoints/google-t5/t5-large"


def test_from_config_default_is_t5_11b(monkeypatch):
    # Guards the fallback: with no text_encoder_path the historical t5-11b
    # default still applies (only reached when a config wants t5-11b).
    te_calls, state_ch = _mock_from_config_internals(monkeypatch)
    cfg = _make_config(state_ch)

    pipe = ag.AnomalyGenPipeline.from_config(cfg, dit_path="dummy/dit.pt")

    assert te_calls[0]["cache_dir"] == "checkpoints/google-t5/t5-11b"
    assert pipe.text_encoder_path == "checkpoints/google-t5/t5-11b"


def _build_pipe(monkeypatch, text_encoder_path):
    _mock_from_config_internals(monkeypatch)
    cfg = _make_config(16)
    return ag.AnomalyGenPipeline.from_config(
        cfg, dit_path="dummy/dit.pt", text_encoder_path=text_encoder_path)


def _stub_ag_component_inits(monkeypatch, pipe):
    for name in ("_initialize_mask_encoder", "_initialize_adapter",
                 "_initialize_anomaly_embedding"):
        monkeypatch.setattr(pipe, name, MagicMock(return_value=MagicMock()))


def _ag_config(t5_model_name):
    c = MagicMock()
    c.ad_precision = "bfloat16"
    c.t5_model_name = t5_model_name
    return c


def _recording_encoder(monkeypatch):
    calls = []

    def rec(*args, **kwargs):
        calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(ag, "CosmosT5TextEncoder", MagicMock(side_effect=rec))
    return calls


def test_swap_skipped_when_encoder_already_selected(monkeypatch):
    # from_config already loaded t5-large (recorded), and ag_config selects the
    # same — the swap must NOT reload it (no redundant t5-large load).
    pipe = _build_pipe(monkeypatch, "checkpoints/google-t5/t5-large")
    _stub_ag_component_inits(monkeypatch, pipe)
    swap_calls = _recording_encoder(monkeypatch)

    pipe.from_anomaly_gen_config(_ag_config("checkpoints/google-t5/t5-large"))

    assert swap_calls == []
    assert pipe.text_encoder_path == "checkpoints/google-t5/t5-large"


def test_swap_reloads_when_encoder_differs(monkeypatch):
    # from_config loaded the t5-11b default, ag_config selects t5-large — the
    # swap must reload to the selected encoder (backward-compatible behavior).
    pipe = _build_pipe(monkeypatch, "checkpoints/google-t5/t5-11b")
    _stub_ag_component_inits(monkeypatch, pipe)
    swap_calls = _recording_encoder(monkeypatch)

    pipe.from_anomaly_gen_config(_ag_config("checkpoints/google-t5/t5-large"))

    assert len(swap_calls) == 1
    assert swap_calls[0]["cache_dir"] == "checkpoints/google-t5/t5-large"
    assert pipe.text_encoder_path == "checkpoints/google-t5/t5-large"


class _StopInit(Exception):
    pass


def _capture_from_config_arg(monkeypatch, model_module, pipeline_cls_name):
    """Short-circuit the pipeline's from_config so we can read the
    text_encoder_path the model forwarded, without building the real pipeline."""
    captured = {}

    def _cap(cls, config, dit_path=None,
             text_encoder_path="checkpoints/google-t5/t5-11b", **kw):
        captured["text_encoder_path"] = text_encoder_path
        raise _StopInit()

    monkeypatch.setattr(getattr(model_module, pipeline_cls_name),
                        "from_config", classmethod(_cap))
    return captured


def _model_config(t5_model_name):
    config = MagicMock()
    config.precision = "bfloat16"
    config.loss_reduce = "mean"
    config.loss_scale = 1.0
    config.adjust_video_noise = False
    if t5_model_name is None:
        # A config that genuinely wants the t5-11b default omits t5_model_name;
        # a plain object makes getattr(config.ag_config, "t5_model_name", None) None.
        config.ag_config = object()
    else:
        config.ag_config.t5_model_name = t5_model_name
    return config


def test_single_view_model_forwards_t5_model_name(monkeypatch):
    import cosmos_predict2.models.anomaly_gen_model as m
    captured = _capture_from_config_arg(monkeypatch, m, "AnomalyGenPipeline")
    try:
        m.Predict2AnomalyGenModel(_model_config("checkpoints/google-t5/t5-large"))
    except _StopInit:
        pass
    assert captured.get("text_encoder_path") == "checkpoints/google-t5/t5-large"


def test_multiview_model_forwards_t5_model_name(monkeypatch):
    import cosmos_predict2.models.anomaly_gen_multiview_model as m
    captured = _capture_from_config_arg(monkeypatch, m, "AnomalyGenMultiViewPipeline")
    try:
        m.Predict2AnomalyGenMultiViewModel(_model_config("checkpoints/google-t5/t5-large"))
    except _StopInit:
        pass
    assert captured.get("text_encoder_path") == "checkpoints/google-t5/t5-large"


def test_model_omits_text_encoder_path_when_t5_unset(monkeypatch):
    # No t5_model_name -> from_config falls back to its t5-11b default (the
    # historical behavior for configs that actually want t5-11b).
    import cosmos_predict2.models.anomaly_gen_model as m
    captured = _capture_from_config_arg(monkeypatch, m, "AnomalyGenPipeline")
    try:
        m.Predict2AnomalyGenModel(_model_config(None))
    except _StopInit:
        pass
    assert captured.get("text_encoder_path") == "checkpoints/google-t5/t5-11b"
