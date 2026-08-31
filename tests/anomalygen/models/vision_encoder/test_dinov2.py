# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DINOv2 correspondence backbone loader.

The spec registry is checked purely in memory; loading the real model needs the checkpoint (and
runs on CUDA when available, else CPU — ``get_dinov2_model`` picks the device), so that test skips
only when the checkpoint is absent.
"""

import os

import pytest
import torch

from anomalygen.models.vision_encoder.dinov2 import DEFAULT_BACKBONE, BackboneSpec, get_dinov2_model

_DINOV2_WEIGHTS = os.path.join(DEFAULT_BACKBONE, "model.safetensors")


def test_backbone_spec_registry():
    # Both the local checkpoint dir and the HF id resolve to the same ViT-L/14 spec.
    assert DEFAULT_BACKBONE in BackboneSpec
    assert "facebook/dinov2-large" in BackboneSpec
    spec = BackboneSpec[DEFAULT_BACKBONE]
    assert spec["patch_size"] == 14
    assert spec["mean"] == (0.485, 0.456, 0.406)
    assert spec["std"] == (0.229, 0.224, 0.225)


@pytest.mark.skipif(not os.path.exists(_DINOV2_WEIGHTS), reason="DINOv2 checkpoint not present")
def test_get_dinov2_model_loads_and_caches():
    model = get_dinov2_model()
    assert isinstance(model, torch.nn.Module)
    assert not model.training  # loaded in eval mode
    # Cached per model id: a second call returns the exact same object.
    assert get_dinov2_model() is model
