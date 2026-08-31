# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the C-RADIO ViT backbone.

``radio.py`` is pure timm/torch (no CUDA-only ops), so the model can be built and run forward on
CPU. Construction + every forward path is covered with a random-init model at a tiny resolution;
the real-weight load path is covered separately and skipped when the checkpoint is absent (e.g. CI).
"""

import inspect
import io
import os
import pickle
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

import anomalygen
from anomalygen.models.vision_encoder.cradio import c_radio_v3_vit_base_patch16_reg4_dinov2
from anomalygen.models.vision_encoder.cradio.ptm_util import load_pretrained_weights
from anomalygen.models.vision_encoder.cradio.radio import (
    RADIO,
    ClsToken,
    Im2Patches,
    remove_state_dict_prefix,
    replace_state_dict_key,
)

RES = 64  # multiple of the patch size (16); tiny so CPU forward is fast
_N_PATCHES = (RES // 16) ** 2  # 16
_SUMMARY_DIM = 768 * 3  # embed_dim * len(summary_idxs=[0,1,2])

_CKPT = str(
    Path(anomalygen.__file__).resolve().parent.parent / "checkpoints" / "nvidia" / "C-RADIO-V3" / "model.safetensors"
)


@pytest.fixture(scope="module")
def model():
    m = c_radio_v3_vit_base_patch16_reg4_dinov2(resolution=(RES, RES))
    m.eval()
    return m


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(0)
    return torch.randn(1, 3, RES, RES)


@pytest.fixture(scope="module")
def pretrained():
    """C-RADIO-V3 with real weights loaded (module-scoped). Skips if the checkpoint is absent."""
    if not os.path.exists(_CKPT):
        pytest.skip("C-RADIO-V3 checkpoint not present")
    m = c_radio_v3_vit_base_patch16_reg4_dinov2(resolution=(RES, RES))
    load_result = m.load_state_dict(load_pretrained_weights(_CKPT), strict=False)
    m.eval()
    return m, load_result


# --- construction --------------------------------------------------------------------------------


def test_build_config(model):
    assert model.radio_version == "CRADIOV2"  # reg4_dinov2 is not the mlpnorm V1 variant
    assert model.patch_size == 16
    assert model.num_features == _SUMMARY_DIM


def test_unsupported_backbone_raises():
    with pytest.raises(ValueError):
        RADIO(backbone="does_not_exist", summary_idxs=[0, 1, 2])


def test_non_multiple_resolution_rejected():
    # RADIOWrapper validates the resolution is a multiple of the patch size.
    with pytest.raises(ValueError):
        c_radio_v3_vit_base_patch16_reg4_dinov2(resolution=(70, 70))


# --- forward paths (random-init; no checkpoint needed) -------------------------------------------


def test_forward_summary(model, x):
    with torch.no_grad():
        summary = model(x)  # head is Identity (num_classes=0)
    assert summary.shape == (1, _SUMMARY_DIM)


def test_forward_pre_logits(model, x):
    with torch.no_grad():
        summary, feat = model.forward_pre_logits(x)
    assert summary.shape == (1, _SUMMARY_DIM)
    assert feat.shape == (1, _N_PATCHES, 768)


def test_forward_feature_pyramid(model, x):
    with torch.no_grad():
        fp = model.forward_feature_pyramid(x)
    # [B, C, H, W] with H=W=RES/patch_size.
    assert fp.shape == (1, 768, RES // 16, RES // 16)


def test_classification_head_path():
    head_model = RADIO(
        backbone="vit_base_patch16_reg4_dinov2",
        summary_idxs=[0, 1, 2],
        num_teacher=4,
        register_multiple=8,
        resolution=(RES, RES),
        num_classes=5,
    ).eval()
    assert not isinstance(head_model.get_classifier(), torch.nn.Identity)
    with torch.no_grad():
        out = head_model(torch.randn(1, 3, RES, RES))
    assert out.shape == (1, 5)

    head_model.reset_classifier(3)
    with torch.no_grad():
        out = head_model(torch.randn(1, 3, RES, RES))
    assert out.shape == (1, 3)


# --- state-dict remapping ------------------------------------------------------------------------


def test_remove_state_dict_prefix():
    sd = {"base_model.a": 1, "keep.b": 2}
    out = remove_state_dict_prefix(sd, "base_model.")
    assert out == {"a": 1, "keep.b": 2}  # only the prefixed key is stripped


def test_replace_state_dict_key():
    sd = {"x.grandma.w": 1, "y.z": 2}
    out = replace_state_dict_key(sd, old_key="grandma", new_key="gamma")
    assert out == {"x.gamma.w": 1, "y.z": 2}


def test_load_state_dict_roundtrip(model):
    # Feeding the model its own state dict exercises the prefix-stripping + CRADIOV2 load path.
    result = model.load_state_dict(model.state_dict(), strict=False)
    assert result.missing_keys == []


def test_load_real_checkpoint_interpolates_pos_embed(pretrained):
    # Loading the real checkpoint exercises the bicubic pos-embed interpolation in
    # ViTPatchGenerator._load_embed (checkpoint grid != cpe_max_size grid); all keys must land.
    _model, load_result = pretrained
    assert load_result.missing_keys == []


# A brown bear from COCO val2017. Golden summary fingerprint below was captured from the
# pretrained C-RADIO-V3 on this image, resized to RES and ImageNet-normalized. Regenerate the
# constants if the checkpoint or preprocessing changes.
_COCO_URL = "http://images.cocodataset.org/val2017/000000000285.jpg"
_GOLDEN_FIRST8 = torch.tensor([-0.52137, -0.04276, 0.22433, -1.72400, -1.31361, -0.54156, 2.56846, -1.11526])
_GOLDEN_NORM = 50.26108
_GOLDEN_MEAN = -0.023905


@pytest.fixture(scope="module")
def coco_bear_input():
    """COCO bear image, downloaded on the fly and preprocessed to a normalized [1,3,RES,RES] tensor."""
    try:
        raw = urllib.request.urlopen(_COCO_URL, timeout=30).read()  # noqa: S310 (fixed https-less COCO URL)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip(f"could not download COCO sample image: {exc}")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((RES, RES)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(img).unsqueeze(0)


def test_pretrained_matches_golden_on_coco_image(pretrained, coco_bear_input):
    """Golden-value regression check: real COCO image -> pretrained backbone -> known summary."""
    model, _ = pretrained
    with torch.no_grad():
        summary = model(coco_bear_input)[0]  # [2304]
    assert summary.shape == (_SUMMARY_DIM,)
    assert torch.isfinite(summary).all()
    # Fingerprint (leading values + L2 norm + mean) must match the golden values captured from this
    # checkpoint and preprocessing; tolerances absorb cross-platform CPU float noise while still
    # catching any real change in the model wiring or preprocessing.
    assert torch.allclose(summary[:8], _GOLDEN_FIRST8, atol=1e-2)
    assert summary.norm().item() == pytest.approx(_GOLDEN_NORM, abs=0.2)
    assert summary.mean().item() == pytest.approx(_GOLDEN_MEAN, abs=1e-2)


# --- small submodules ----------------------------------------------------------------------------


def test_im2patches_unit_and_block():
    x = torch.randn(1, 3, 4, 4)
    # patch_size==1: flatten to [B, H*W, C].
    assert Im2Patches(1)(x).shape == (1, 16, 3)
    # patch_size==2: [B, num_patches, C*ph*pw].
    assert Im2Patches(2)(x).shape == (1, 4, 12)


def test_cls_token_concat_and_disable():
    cls = ClsToken(ndim=8, num_tokens=1, enabled=True, register_multiple=8)
    assert cls.num_registers == 7  # register_multiple - (num_tokens % register_multiple)
    assert cls.no_weight_decay() == ["token"]

    feats = torch.randn(2, 5, 8)
    out = cls(feats)
    assert out.shape == (2, 5 + 1 + 7, 8)  # tokens + registers prepended

    cls.disable()
    assert cls.token is None
    assert torch.equal(cls(feats), feats)  # disabled -> passthrough


# --- safe checkpoint loading ----------------------------------------------------------------------
# A ``.pt`` file is a pickle, so loading one without ``weights_only=True`` runs whatever
# ``__reduce__`` it contains. ``load_pretrained_weights`` is the shared entry point for the KPI
# backbones, so its default is what every caller inherits.


class _ExecOnUnpickle:
    """Stands in for a crafted checkpoint: unpickling this calls a global of the file's choosing.

    The payload is deliberately inert (``print``) rather than something destructive. What is being
    tested is that the unpickler refuses to call *any* global that is not allowlisted, so a harmless
    one proves it exactly as well — and if this test ever regresses it prints instead of running
    whatever a real attacker would have put here.
    """

    def __reduce__(self):
        return (print, ("unpickling executed a global",))


def test_load_pretrained_weights_restricts_the_unpickler_by_default():
    assert inspect.signature(load_pretrained_weights).parameters["weights_only"].default is True


def test_load_pretrained_weights_rejects_a_crafted_checkpoint(tmp_path):
    path = tmp_path / "evil.pt"
    with open(path, "wb") as handle:
        pickle.dump({"state_dict": _ExecOnUnpickle()}, handle)
    with pytest.raises(Exception):
        load_pretrained_weights(str(path))


@pytest.mark.skipif(not os.path.exists(_CKPT), reason="C-RADIO-V3 checkpoint not present")
def test_real_checkpoint_loads_under_the_restriction():
    """The shipped C-RADIO weights must load with the restricted unpickler, not just a synthetic file."""
    state = load_pretrained_weights(_CKPT)
    assert state, "expected a non-empty state dict"
    assert all(isinstance(v, torch.Tensor) for v in state.values())
