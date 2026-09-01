# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CPU-testable parts of the inference-loading public API.

Covers checkpoint resolution, the fine-tuned-overlay guard, the inpaint-closure's
unknown-anomaly guard, and the recipe-name error path — none of which build a model.
The full model-loading / generation path needs a fine-tuned checkpoint and a GPU and is
not exercised here.
"""

import glob
import hashlib
import pathlib
import pickle

import numpy as np
import pytest
import torch
from hydra.core.config_store import ConfigStore
from PIL import Image

import anomalygen
import anomalygen.checkpoint.utils as ckpt_utils
import anomalygen.inference.inpaint as inpaint_module
from anomalygen.configs.texture.constants import BASE_CHECKPOINT_PATHS, DEFAULT_MODEL_SIZE, MODEL_SIZES
from anomalygen.inference.inpaint import (
    _base_checkpoint_for_experiment,
    _build_inpaint_batch_multi,
    _resolve_ft_model_pt,
    _verify_overlay,
    build_inpaint_batch_fn,
    build_inpaint_one,
    load_finetuned_model,
    load_for_inference,
)

# --- _resolve_ft_model_pt ------------------------------------------------------------------------


def test_resolve_direct_pt_file(tmp_path):
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    assert _resolve_ft_model_pt(str(pt)) == str(pt)


def test_resolve_picks_highest_iter(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "iter_000001.pt").write_bytes(b"x")
    (model_dir / "iter_000002.pt").write_bytes(b"x")
    assert _resolve_ft_model_pt(str(tmp_path)) == str(model_dir / "iter_000002.pt")


def test_resolve_honours_latest_pointer(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "iter_000001.pt").write_bytes(b"x")
    (model_dir / "iter_000002.pt").write_bytes(b"x")
    (tmp_path / "latest_checkpoint.txt").write_text("iter_000001")  # no .pt suffix -> appended
    assert _resolve_ft_model_pt(str(tmp_path)) == str(model_dir / "iter_000001.pt")


def test_resolve_prefers_best_over_latest(tmp_path):
    """best_checkpoint.txt names the peak-scoring iteration; latest_checkpoint.txt names the last one,
    which is frequently worse. Generating from a run dir must pick the best without the caller
    having to read the pointer itself."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for it in ("iter_000001.pt", "iter_000002.pt", "iter_000003.pt"):
        (model_dir / it).write_bytes(b"x")
    (tmp_path / "latest_checkpoint.txt").write_text("iter_000003.pt")
    (tmp_path / "best_checkpoint.txt").write_text("iter_000002.pt\n")  # trailing newline is stripped
    assert _resolve_ft_model_pt(str(tmp_path)) == str(model_dir / "iter_000002.pt")


def test_resolve_falls_back_to_latest_when_best_is_absent_or_dangling(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "iter_000001.pt").write_bytes(b"x")
    (model_dir / "iter_000009.pt").write_bytes(b"x")
    (tmp_path / "latest_checkpoint.txt").write_text("iter_000001.pt")
    # A pointer naming a checkpoint that was rotated away must not win, and must not raise.
    (tmp_path / "best_checkpoint.txt").write_text("iter_000005.pt")
    assert _resolve_ft_model_pt(str(tmp_path)) == str(model_dir / "iter_000001.pt")


def test_resolve_checkpoints_subdir_layout(tmp_path):
    model_dir = tmp_path / "checkpoints" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "iter_000005.pt").write_bytes(b"x")
    assert _resolve_ft_model_pt(str(tmp_path)) == str(model_dir / "iter_000005.pt")


def test_resolve_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_ft_model_pt(str(tmp_path))


# --- _verify_overlay -----------------------------------------------------------------------------

# A model state with one base key plus the two trained keys the overlay must supply.
_FULL_STATE = {
    "net.blocks.0.weight": 0,
    "net.blocks.0.lora_A.weight": 0,
    "net.inpaint_class_emb": 0,
}


def test_verify_overlay_accepts_valid_subset():
    ckpt = {"net.blocks.0.lora_A.weight": 1, "net.inpaint_class_emb": 1}
    assert _verify_overlay(ckpt, _FULL_STATE) is None  # no raise


def test_verify_overlay_rejects_unexpected_key():
    ckpt = {"net.blocks.0.lora_A.weight": 1, "net.does_not_exist": 1}
    with pytest.raises(ValueError):
        _verify_overlay(ckpt, _FULL_STATE)


def test_verify_overlay_rejects_missing_trained_key():
    # Subset of the model (no unexpected keys) but missing a trained key -> would stay at base value.
    ckpt = {"net.inpaint_class_emb": 1}  # lora_A missing
    with pytest.raises(ValueError):
        _verify_overlay(ckpt, _FULL_STATE)


# --- _base_checkpoint_for_experiment --------------------------------------------------------------


@pytest.fixture
def register_experiment():
    """Register throwaway ``experiment`` nodes in the (global) ConfigStore, and clean them up.

    ``_base_checkpoint_for_experiment`` reads the BUILT node rather than the recipe file, so a
    hand-registered node is enough to exercise it — no model build, no checkpoint, no GPU.
    """
    names = []

    def _register(name, model_size=...):
        expert_config = {} if model_size is ... else {"model_size": model_size}
        ConfigStore.instance().store(
            group="experiment",
            package="_global_",
            name=name,
            node={"model": {"config": {"diffusion_expert_config": expert_config}}},
        )
        names.append(f"{name}.yaml")
        return name

    yield _register

    repo = ConfigStore.instance().repo.get("experiment", {})
    for key in names:
        repo.pop(key, None)


def test_base_checkpoint_follows_the_edge_size(register_experiment):
    # The sizes use different frozen backbones, so loading the other size's DCP would silently be
    # the wrong weights; the base checkpoint must track the experiment's model_size.
    name = register_experiment("unittest_size_edge", model_size="edge")
    assert _base_checkpoint_for_experiment(name) == "checkpoints/Cosmos3-Edge"


def test_base_checkpoint_follows_the_nano_size(register_experiment):
    name = register_experiment("unittest_size_nano", model_size="nano")
    assert _base_checkpoint_for_experiment(name) == "checkpoints/Cosmos3-Nano"


def test_base_checkpoint_rejects_unknown_size(register_experiment):
    name = register_experiment("unittest_size_bogus", model_size="giant")
    with pytest.raises(ValueError):
        _base_checkpoint_for_experiment(name)


@pytest.mark.parametrize("label,model_size", [("absent", ...), ("null", None), ("empty", "")])
def test_base_checkpoint_falls_back_to_the_default_size(register_experiment, label, model_size):
    # DELIBERATE, not an accident: a recipe that never mentions model_size (every recipe predating
    # the Edge size) resolves to the default size instead of failing. Pinned so the silent fallback
    # stays a decision — an unset size must never become "whatever size ran last".
    name = register_experiment(f"unittest_size_unset_{label}", model_size=model_size)
    assert _base_checkpoint_for_experiment(name) == BASE_CHECKPOINT_PATHS[DEFAULT_MODEL_SIZE]
    assert BASE_CHECKPOINT_PATHS[DEFAULT_MODEL_SIZE] == "checkpoints/Cosmos3-Nano"


def test_size_table_and_default_agree():
    # Drift guard: a size added to one table and not the other resolves to a KeyError/ValueError at
    # load time, long after the recipe was written.
    assert set(BASE_CHECKPOINT_PATHS) == set(MODEL_SIZES)
    assert DEFAULT_MODEL_SIZE in MODEL_SIZES


# --- build_inpaint_one --------------------------------------------------------------------------


def test_build_inpaint_one_rejects_unknown_anomaly():
    # The closure validates anomaly_name before touching the model, so a dummy model is fine.
    inpaint_one = build_inpaint_one(model=object(), class_ids={"metal+scratch": 0}, num_steps=1, guidance=1.0)
    img = Image.new("RGB", (16, 16))
    mask = Image.new("L", (16, 16))
    with pytest.raises(KeyError):
        inpaint_one(img, mask, "unknown+type")


def test_build_inpaint_batch_fn_rejects_unknown_anomaly():
    # Same guard, batched: unknown name fails before any model call.
    fn = build_inpaint_batch_fn(model=object(), class_ids={"metal+scratch": 0}, num_steps=1, guidance=1.0, shift=5.0)
    img = Image.new("RGB", (16, 16))
    mask = Image.new("L", (16, 16))
    with pytest.raises(KeyError):
        fn([img], [mask], ["unknown+type"], [1])


# --- _build_inpaint_batch_multi ------------------------------------------------------------------


class _StubNet:
    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))  # CPU param -> device = cpu


class _StubModel:
    """Minimal surface build_inpaint_batch(_multi) reads; no real forward."""

    net = _StubNet()
    tensor_kwargs = {"dtype": torch.float32}
    input_caption_key = "ai_caption"


def test_build_inpaint_batch_multi_merges_samples():
    model = _StubModel()
    imgs = [Image.fromarray(np.full((64, 64, 3), v, np.uint8), "RGB") for v in (100, 200)]
    m = np.zeros((64, 64), np.uint8)
    m[20:40, 20:40] = 255
    masks = [Image.fromarray(m, "L"), Image.fromarray(m, "L")]

    batch = _build_inpaint_batch_multi(model, imgs, masks, [0, 1], ["a b", "c d"])

    assert batch["dataset_name"] == "anomaly" and batch["is_preprocessed"] is True
    assert len(batch["images"]) == 4  # 2 vision items * 2 samples, flattened
    assert all(t.shape == (1, 3, 1, 64, 64) for t in batch["images"])
    assert batch["num_vision_items_per_sample"] == [2, 2]
    assert len(batch["image_size"]) == 4
    assert batch["edit_mask"].shape == (2, 1, 64, 64)
    assert batch["anomaly_class_id"].tolist() == [0, 1]
    assert batch[model.input_caption_key] == ["a b", "c d"]


# --- load_for_inference error path ---------------------------------------------------------------


def test_load_for_inference_requires_experiment_name_for_module_recipe():
    # A Python-module recipe yields no experiment name; without an explicit experiment= it must fail
    # loudly (before any model load). "json" is importable and returns None from register_recipe.
    with pytest.raises(ValueError):
        load_for_inference("json", "unused_checkpoint")


# --- per-instance seed wiring (serial path) ------------------------------------------------------
# The batched path is covered in test_iterative.py; this is the half generate.py actually runs, plus
# the build_inpaint_batch(seed=) -> build_source_item(seed=) hand-off that IS the seeding fix. A
# regression that dropped `seed=` on the way in would leave test_utils.py's build_source_item suite
# green while silently restoring the global-CUDA-RNG behaviour.


class _RecordingModel(_StubModel):
    """``_StubModel`` plus the two hooks ``_generate_inpaint`` drives, recording what it saw."""

    def __init__(self):
        self.seeds = []
        self.sources = []

    def generate_samples_from_batch(self, batch, guidance=None, num_steps=None, shift=None, seed=None):
        self.seeds.append(seed)
        self.sources.append(batch["images"][0].clone())  # [source, cond] -> source item
        return {"vision": [torch.zeros(1, 3, 1, 8, 8)]}

    def decode(self, latent):
        return torch.zeros(1, 3, 1, 8, 8)


def _crop_pair(size=64, fill=120):
    img = Image.fromarray(np.full((size, size, 3), fill, np.uint8), "RGB")
    m = np.zeros((size, size), np.uint8)
    m[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
    return img, Image.fromarray(m, "L")


def test_build_inpaint_one_advances_the_seed_per_instance():
    model = _RecordingModel()
    img, mask = _crop_pair()
    fn = build_inpaint_one(model, {"t+d": 0}, num_steps=1, guidance=1.0, model_input_size=64, seed=500)

    for _ in range(3):
        fn(img, mask, "t+d")

    # n-th call is seeded seed+n, so a sample's instances never share a noise draw.
    assert [s[0] for s in model.seeds] == [500, 501, 502]

    # ... and that seed is what actually builds the source item: call n must match a fresh closure
    # started at 500+n. Asserting merely that consecutive sources DIFFER would pass on the global
    # RNG too, i.e. it would still pass with the seeding dropped — verified by mutation.
    for n in (1, 2):
        ref = _RecordingModel()
        build_inpaint_one(ref, {"t+d": 0}, num_steps=1, guidance=1.0, model_input_size=64, seed=500 + n)(
            img, mask, "t+d"
        )
        assert torch.equal(model.sources[n], ref.sources[0])


def test_build_inpaint_one_is_reproducible_across_closures():
    """Two closures built with the same seed produce identical source items — the --base_seed contract."""
    img, mask = _crop_pair()
    runs = []
    for _ in range(2):
        model = _RecordingModel()
        build_inpaint_one(model, {"t+d": 0}, num_steps=1, guidance=1.0, model_input_size=64, seed=77)(img, mask, "t+d")
        runs.append(model.sources[0])
    assert torch.equal(runs[0], runs[1])


def test_batch_source_items_are_invariant_to_batch_position():
    """A sample's conditioning noise depends only on its own seed, not on where it sits in the batch."""
    model = _StubModel()
    img_a, mask_a = _crop_pair(fill=100)
    img_b, mask_b = _crop_pair(fill=200)

    both = _build_inpaint_batch_multi(model, [img_a, img_b], [mask_a, mask_b], [0, 1], ["a", "b"], seeds=[11, 22])
    swapped = _build_inpaint_batch_multi(model, [img_b, img_a], [mask_b, mask_a], [1, 0], ["b", "a"], seeds=[22, 11])

    # images is flattened [src0, cond0, src1, cond1]; b is at index 2 in one and index 0 in the other.
    assert torch.equal(both["images"][2], swapped["images"][0]), "sample b's source moved with its position"
    assert not torch.equal(both["images"][0], both["images"][2]), "different seeds must give different noise"


# --- safe checkpoint loading ----------------------------------------------------------------------
# ``load_for_inference`` reads a fine-tuned ``.pt`` with ``torch.load``. That file is the one most
# likely to arrive from elsewhere — copied between machines, shared, or read from a mounted directory
# — so the unpickler stays restricted, and the numpy state real checkpoints carry has to keep loading
# under that restriction.

_PACKAGE_ROOT = pathlib.Path(anomalygen.__file__).resolve().parent
_FT_CKPTS = sorted(glob.glob(str(_PACKAGE_ROOT.parent / "results" / "**" / "iter_*.pt"), recursive=True))


class _ExecOnUnpickle:
    """Unpickling this calls a global; the payload is inert on purpose (see test_cradio.py)."""

    def __reduce__(self):
        return (print, ("unpickling executed a global",))


def test_torch_load_rejects_a_crafted_checkpoint(tmp_path):
    """Without the restriction this pickle would run a shell command instead of returning weights."""
    path = tmp_path / "evil.pt"
    with open(path, "wb") as handle:
        pickle.dump({"state_dict": _ExecOnUnpickle()}, handle)
    with pytest.raises(Exception):
        torch.load(path, map_location="cpu", weights_only=True)


@pytest.mark.parametrize("scalar", [np.float64(0.5), np.float32(0.9), np.int64(7)])
def test_numpy_scalar_state_loads_under_the_restriction(tmp_path, scalar):
    """Optimizer and scheduler state carry numpy scalars at whatever precision training used.

    Importing ``anomalygen`` allowlists the numpy scalar and dtype constructors; without it this
    raises ``UnpicklingError``, which is what made real optim and scheduler checkpoints unloadable
    when the restriction was first switched on.
    """
    path = tmp_path / "optim.pt"
    torch.save({"step": scalar, "dtype": np.dtype("float64"), "tensor": torch.zeros(2)}, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert float(loaded["step"]) == pytest.approx(float(scalar))
    assert loaded["dtype"] == np.dtype("float64")


@pytest.mark.skipif(not _FT_CKPTS, reason="no fine-tuned checkpoint under results/")
def test_real_finetuned_checkpoint_loads_under_the_restriction():
    """A checkpoint this pipeline actually produced, not just a synthetic state dict."""
    state = torch.load(_FT_CKPTS[0], map_location="cpu", weights_only=True)
    assert state, "expected a non-empty state dict"
    assert all(isinstance(v, torch.Tensor) for v in state.values())


def test_no_call_site_disables_the_unpickler_restriction():
    """``weights_only=False`` must not reappear — including in a module no test imports."""
    offenders = [
        f"{source.relative_to(_PACKAGE_ROOT.parent)}:{lineno}"
        for source in _PACKAGE_ROOT.rglob("*.py")
        for lineno, line in enumerate(source.read_text().splitlines(), 1)
        if "weights_only=False" in line.split("#", 1)[0].replace(" ", "")
    ]
    assert offenders == [], "checkpoint loads must keep weights_only=True; found: " + ", ".join(offenders)


# --- checkpoint integrity at the inference load site ----------------------------------------------


def test_load_finetuned_model_refuses_a_tampered_checkpoint(tmp_path):
    """Inference reads the .pt that travelled, so the digest is checked here and not only on resume.

    The refusal lands before the base network loads, so no model fixture is needed — reaching
    load_model at all would mean the check moved back after it.
    """
    model_pt = tmp_path / "iter_000000100.pt"
    torch.save({"w": torch.zeros(2)}, model_pt)
    ckpt_utils.write_digest(str(model_pt))
    torch.save({"w": torch.ones(2)}, model_pt)  # same path, different bytes

    with pytest.raises(ValueError, match="does not match its recorded digest"):
        load_finetuned_model(str(model_pt), base_checkpoint="unused")


def test_load_finetuned_model_accepts_an_untampered_checkpoint(tmp_path, monkeypatch):
    """The digest must not be what stops a legitimate checkpoint: it gets past the check."""
    model_pt = tmp_path / "iter_000000100.pt"
    torch.save({"w": torch.zeros(2)}, model_pt)
    ckpt_utils.write_digest(str(model_pt))

    def _stop_after_the_check(*_args, **_kwargs):
        raise RuntimeError("reached load_model")

    monkeypatch.setattr(inpaint_module, "load_model", _stop_after_the_check)
    with pytest.raises(RuntimeError, match="reached load_model"):
        load_finetuned_model(str(model_pt), base_checkpoint="unused")


# --- opt-in base-weight verification --------------------------------------------------------------
# verify_digest covers the fine-tuned .pt. The frozen base DCP had no load-time check at all: its
# digests are recorded by download_checkpoints.sh and were only compared on a later download run,
# while checkpoints/ is a bind mount anything on the host can rewrite in between.


def _base_tree(tmp_path, payload=b"shard"):
    """A checkpoints/ root holding one base DCP shard, plus a manifest recording it."""
    ckpt_dir = tmp_path / "checkpoints"
    shard = ckpt_dir / "Cosmos3-Nano" / "model" / "__0_0.distcp"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(payload)
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text(f"{hashlib.sha256(payload).hexdigest()}  Cosmos3-Nano/model/__0_0.distcp\n")
    return ckpt_dir, shard, manifest


def _ft_checkpoint(tmp_path):
    model_pt = tmp_path / "iter_000000100.pt"
    torch.save({"w": torch.zeros(2)}, model_pt)
    ckpt_utils.write_digest(str(model_pt))
    return str(model_pt)


def test_base_weights_are_not_verified_unless_opted_in(tmp_path, monkeypatch):
    """Hashing multi-GB shards on every load is not the default; the check is opt-in."""
    ckpt_dir, shard, manifest = _base_tree(tmp_path)
    shard.write_bytes(b"tampered")
    monkeypatch.setattr(inpaint_module, "CONVERTED_MANIFEST", manifest)
    monkeypatch.delenv("ANOMALYGEN_VERIFY_BASE_WEIGHTS", raising=False)
    monkeypatch.setattr(
        inpaint_module, "load_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached load_model"))
    )

    with pytest.raises(RuntimeError, match="reached load_model"):
        load_finetuned_model(_ft_checkpoint(tmp_path), base_checkpoint=str(ckpt_dir / "Cosmos3-Nano"))


def test_opted_in_base_weight_verification_refuses_a_tampered_shard(tmp_path, monkeypatch):
    ckpt_dir, shard, manifest = _base_tree(tmp_path)
    shard.write_bytes(b"tampered")
    monkeypatch.setattr(inpaint_module, "CONVERTED_MANIFEST", manifest)
    monkeypatch.setenv("ANOMALYGEN_VERIFY_BASE_WEIGHTS", "1")

    with pytest.raises(ValueError, match="does not match the recorded digest"):
        load_finetuned_model(_ft_checkpoint(tmp_path), base_checkpoint=str(ckpt_dir / "Cosmos3-Nano"))


def test_opted_in_base_weight_verification_passes_an_untouched_shard(tmp_path, monkeypatch):
    """The check must not be what stops a legitimate load: it gets past to load_model."""
    ckpt_dir, _, manifest = _base_tree(tmp_path)
    monkeypatch.setattr(inpaint_module, "CONVERTED_MANIFEST", manifest)
    monkeypatch.setenv("ANOMALYGEN_VERIFY_BASE_WEIGHTS", "1")
    monkeypatch.setattr(
        inpaint_module, "load_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached load_model"))
    )

    with pytest.raises(RuntimeError, match="reached load_model"):
        load_finetuned_model(_ft_checkpoint(tmp_path), base_checkpoint=str(ckpt_dir / "Cosmos3-Nano"))
