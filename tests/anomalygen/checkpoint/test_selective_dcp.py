# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SelectiveCheckpointer's restricted checkpoint loading.

``load()`` reads all four components — ``model``, ``optim``, ``scheduler``, ``trainer`` — with
``weights_only=True``. Model state is plain tensors and loads under that restriction unmodified;
the other three are why the module registers a numpy allowlist at import. A missing entry in that
allowlist does not fail at save time. It fails on the way back, on a resume, hours into a run.

So the check that matters is a real training checkpoint, which is gated on one existing. The CPU
round trip below covers the same shape in CI, and deliberately carries numpy scalars: a bare
``AdamW`` state dict contains none, so a round trip built only from stock torch objects would pass
whether or not the allowlist were there, and prove nothing.
"""

import glob
import os

import numpy as np
import pytest
import torch

import anomalygen.checkpoint.selective_dcp as selective_dcp
from anomalygen.checkpoint.selective_dcp import _COMPONENTS

_RESULTS = os.path.join(os.path.dirname(os.path.dirname(selective_dcp.__file__)), "..", "results")


def _real_component(name):
    """Newest saved <component>/iter_*.pt from any completed training run, or None."""
    found = sorted(glob.glob(os.path.join(_RESULTS, "**", "checkpoints", name, "iter_*.pt"), recursive=True))
    return found[-1] if found else None


def test_module_registers_the_numpy_allowlist_on_import():
    """Importing the module is what makes the restricted load work — the loads rely on that."""
    allowed = {getattr(g, "__name__", str(g)) for g in torch.serialization.get_safe_globals()}
    assert "scalar" in allowed, "numpy scalar reconstructor must be allowlisted"
    assert any(name.endswith("DType") for name in allowed), "numpy dtype classes must be allowlisted"


@pytest.mark.parametrize("component", _COMPONENTS)
def test_real_training_checkpoint_component_loads_under_the_restriction(component):
    """The check that answers the question: a component this pipeline actually wrote.

    Skips where no training run is present (CI); runs anywhere `results/` has one.
    """
    path = _real_component(component)
    if path is None:
        pytest.skip(f"no saved {component} checkpoint under results/")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded, f"{component} checkpoint loaded empty"


def test_saved_component_shape_round_trips_under_the_restriction(tmp_path):
    """The four-key shape ``save()`` writes, carrying the numpy scalars real optimizer state has."""
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
    # enable_grad explicitly: importing the pipeline scripts runs init_script(training=False),
    # which disables grad process-wide, so this backward fails when the full suite runs it after
    # them and passes when this module runs alone.
    with torch.enable_grad():
        model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    scheduler.step()

    optim_state = optimizer.state_dict()
    # Stock torch keeps these as python floats; the framework's optimizer carries numpy scalars, and
    # those are what needed allowlisting. Inject them so this exercises the same path.
    optim_state["_numpy_scalars"] = {"lr": np.float64(1e-3), "beta": np.float32(0.9), "step": np.int64(1)}
    saved = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optim": optim_state,
        "scheduler": scheduler.state_dict(),
        "trainer": {"grad_scaler": grad_scaler.state_dict(), "iteration": 7},
    }
    assert set(saved) == set(_COMPONENTS), "test shape must track _COMPONENTS"

    for component, state in saved.items():
        path = tmp_path / f"{component}.pt"
        torch.save(state, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert set(loaded) == set(state), f"{component} lost keys through a restricted load"

    reloaded = torch.load(tmp_path / "optim.pt", map_location="cpu", weights_only=True)
    assert float(reloaded["_numpy_scalars"]["lr"]) == pytest.approx(1e-3)
    assert int(reloaded["_numpy_scalars"]["step"]) == 1
