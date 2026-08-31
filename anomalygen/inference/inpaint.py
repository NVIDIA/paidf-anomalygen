# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-to-image anomaly-inpainting model ops.

The reusable inference building blocks shared by the ``anomalygen/scripts/texture/generate.py`` CLI and the
``ValidationKPI`` callback: load the trained model, build a single-crop inpaint batch,
generate, decode, and the ``build_inpaint_one`` closure the iterative orchestrator drives.
"""

from __future__ import annotations

import contextlib
import glob
import itertools
import os
import pathlib
from typing import Dict, List, Optional

import numpy as np
import torch
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.common.config import ROOT_DIR
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.model_loader import load_model_from_checkpoint
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from PIL import Image
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

import anomalygen  # noqa: F401  (registration side effects)
from anomalygen.checkpoint.trained_keys import TRAINED_KEY_PREFIXES, WARM_START_SKIP_PREFIXES, match_keys
from anomalygen.checkpoint.utils import verify_digest, verify_manifest
from anomalygen.configs.loader import register_recipe
from anomalygen.configs.texture.constants import (
    BASE_CHECKPOINT_PATHS,
    DEFAULT_GUIDANCE,
    DEFAULT_MODEL_SIZE,
    DEFAULT_NUM_STEPS,
    DEFAULT_SHIFT,
    MODEL_SIZES,
)
from anomalygen.data.utils import MASK_FG_THRESHOLD, build_source_item, caption_for_anomaly_type

# Trained subset the fine-tuned .pt must contain, and the keys to skip when warm-starting the base
# network. Shared with the optimizer / checkpointer (see trained_keys).
_FT_TRAINED_KEYS = TRAINED_KEY_PREFIXES
_BASE_SKIP_KEYS = WARM_START_SKIP_PREFIXES


def load_model(
    checkpoint_path: str,
    experiment: str = "anomalygen_texture_ft",
    keys_to_skip_loading: Optional[List[str]] = None,
):
    """Load a model from a full DCP/safetensors checkpoint using its experiment config.

    Internal base-network loader; for an anomalygen fine-tuned run (a filtered ``.pt`` from
    ``SelectiveCheckpointer``) use :func:`load_finetuned_model` / :func:`load_for_inference`.
    The framework's own default config (``cosmos_framework/configs/base/config.py``) is used.
    """
    kwargs = {}
    if keys_to_skip_loading is not None:
        kwargs["keys_to_skip_loading"] = keys_to_skip_loading
    # Resolve the checkpoint to an absolute path against the caller's cwd before the chdir below,
    # so a relative --checkpoint / --base_checkpoint keeps pointing at the repo, not the framework tree.
    checkpoint_path = os.path.abspath(checkpoint_path)
    # Build with cwd at cosmos_framework's parent dir so the model reads its packaged resources
    # by "cosmos_framework/..."-prefixed paths relative to it (e.g. the Qwen3-VL config JSON,
    # the tokenizers/ tree), mirroring anomalygen/scripts/texture/train.py. ROOT_DIR is the
    # package dir; its parent is site-packages for the pip-installed framework.
    with contextlib.chdir(ROOT_DIR.parent):
        model, _config = load_model_from_checkpoint(
            experiment_name=experiment,
            checkpoint_path=checkpoint_path,
            experiment_opts=["model.config.rectified_flow_inference_config.scheduler_type=unipc"],
            **kwargs,
        )
    model.eval()

    return model


# Checkpoint pointers, in preference order. ``best_checkpoint.txt`` (written by the TrainingReport
# callback at train end) names the peak-scoring iteration; ``latest_checkpoint.txt`` (written by the
# checkpointer for resume) names the *last* one, which is frequently worse — small datasets peak early
# and then drift. Preferring "best" here means generating from a run dir does the right thing without
# the caller having to read the pointer themselves.
_CKPT_POINTERS = ("best_checkpoint.txt", "latest_checkpoint.txt")


def _resolve_ft_model_pt(path: str) -> str:
    """Resolve the fine-tuned model component ``.pt`` from ``path``.

    Accepts the ``.pt`` file directly, or a directory (run dir or its ``checkpoints/``) under which
    the SelectiveCheckpointer layout ``model/iter_<N>.pt`` lives. Honors ``best_checkpoint.txt``,
    then ``latest_checkpoint.txt``, else picks the highest iteration (filenames are zero-padded, so
    lexicographic). Logs which pointer decided it, since best vs. latest changes the output quality.
    """
    if path.endswith(".pt") and os.path.isfile(path):
        return path

    for root in (path, os.path.join(path, "checkpoints")):
        model_dir = os.path.join(root, "model")
        if not os.path.isdir(model_dir):
            continue
        for pointer_name in _CKPT_POINTERS:
            pointer = os.path.join(root, pointer_name)
            if not os.path.isfile(pointer):
                continue
            with open(pointer) as f:
                name = f.read().strip()
            if name:
                candidate = os.path.join(model_dir, name if name.endswith(".pt") else f"{name}.pt")
                if os.path.isfile(candidate):
                    log.info(f"Resolved fine-tuned checkpoint from {pointer_name}: {candidate}")
                    return candidate
        pts = sorted(glob.glob(os.path.join(model_dir, "iter_*.pt")))
        if pts:
            log.warning(
                f"No usable checkpoint pointer under {root!r}; falling back to the highest iteration "
                f"{pts[-1]} — this is the LAST checkpoint, not necessarily the best-scoring one."
            )
            return pts[-1]

    raise FileNotFoundError(
        f"No fine-tuned model checkpoint found under {path!r}. Expected a 'model/iter_<N>.pt' "
        "(SelectiveCheckpointer layout) or a direct .pt file."
    )


def _verify_overlay(ckpt_model: dict, full_state: dict) -> None:
    """Guard the strict=False overlay: reject unknown checkpoint keys (silently dropped otherwise)
    and a checkpoint missing trained keys (left at warm-started base values). Mirrors
    ``SelectiveCheckpointer._verify_load_filtered``."""
    unexpected = set(ckpt_model) - set(full_state)
    if unexpected:
        raise ValueError(
            f"Fine-tuned checkpoint has {len(unexpected)} key(s) absent from the model; refusing to load. "
            f"Examples: {sorted(unexpected)[:10]}. The checkpoint likely predates a model/key-layout change."
        )
    missing = match_keys(full_state, _FT_TRAINED_KEYS) - set(ckpt_model)
    if missing:
        raise ValueError(
            f"Fine-tuned checkpoint is missing {len(missing)} trained key(s) {_FT_TRAINED_KEYS} present in the "
            f"model — they would stay at base values. Examples: {sorted(missing)[:10]}."
        )


# Digests for the locally-converted base DCP shards, recorded by scripts/download_checkpoints.sh.
CONVERTED_MANIFEST = pathlib.Path(__file__).resolve().parents[2] / "assets" / "checkpoint_manifest_converted.sha256"


def _verify_base_weights(base_checkpoint: str) -> None:
    """Check the base DCP's shards against the recorded manifest — opt-in, off by default.

    Hashing multi-GB shards costs tens of seconds, which is too much for every load but cheap next to
    a run that silently used substituted weights. Only the size being loaded is hashed. Skipped
    silently when the manifest is absent, so a checkout that never recorded one still runs.
    """
    if os.environ.get("ANOMALYGEN_VERIFY_BASE_WEIGHTS") != "1":
        return
    manifest = pathlib.Path(CONVERTED_MANIFEST)
    if not manifest.is_file():
        log.warning(f"Base-weight verification requested but {manifest} is absent; skipping.")
        return
    base = pathlib.Path(base_checkpoint).resolve()
    failures = verify_manifest(manifest, base.parent, only_under=base.name)
    if failures:
        detail = "\n".join(f"  {rel}: {why}" for rel, why in failures)
        raise ValueError(f"Base checkpoint integrity: {len(failures)} shard(s) under {base} failed.\n{detail}")
    log.info(f"Base checkpoint integrity: {base.name} shards match {manifest.name}.")


def load_finetuned_model(
    ft_checkpoint: str,
    base_checkpoint: Optional[str] = None,
    experiment: str = "anomalygen_texture_ft",
):
    """Load an anomalygen fine-tuned model saved by ``SelectiveCheckpointer``.

    Those checkpoints hold only the trained subset (LoRA + ``inpaint_class_emb``), so this warm-starts
    the full base network from ``base_checkpoint`` (skipping the keys absent from it), then overlays
    the trained subset from ``ft_checkpoint``'s flat ``.pt`` — the model-only half of
    ``SelectiveCheckpointer._load_filtered``. ``ft_checkpoint`` may be the run dir, its ``checkpoints/``,
    or the ``iter_<N>.pt`` directly. The experiment node must already be registered (see
    :func:`load_for_inference`, which registers it from a recipe and calls this).
    ``base_checkpoint`` defaults to the DCP matching that node's ``model_size``.
    """
    if base_checkpoint is None:
        base_checkpoint = _base_checkpoint_for_experiment(experiment)
    model_pt = _resolve_ft_model_pt(ft_checkpoint)
    # Before the base network loads, so a checkpoint that fails costs seconds rather than a multi-GB
    # warm start. verify_digest says these are the bytes we trained; weights_only=True below says an
    # unpickle cannot execute code. This is the load that reads a .pt handed over from a training
    # host or a mounted directory, so both matter here more than on a resume.
    verify_digest(model_pt)
    _verify_base_weights(base_checkpoint)
    model = load_model(base_checkpoint, experiment=experiment, keys_to_skip_loading=list(_BASE_SKIP_KEYS))

    log.info(f"Overlaying fine-tuned weights from {model_pt} onto base {base_checkpoint}.")
    ckpt_model = torch.load(model_pt, map_location="cpu", weights_only=True)
    full_state = get_model_state_dict(model)
    _verify_overlay(ckpt_model, full_state)
    full_state.update(ckpt_model)
    set_model_state_dict(model, model_state_dict=full_state, options=StateDictOptions(strict=False))
    model.eval()

    return model


def _base_checkpoint_for_experiment(experiment: str) -> str:
    """Frozen base DCP matching the registered ``experiment`` node's ``model_size``.

    Read off the built node, not the recipe file, so it covers both recipe forms the loader accepts
    (YAML/JSON file or Python module). The other size's DCP would load silently as wrong weights.
    """
    node = ConfigStore.instance().repo["experiment"][f"{experiment}.yaml"].node
    model_size = str(OmegaConf.select(node, "model.config.diffusion_expert_config.model_size") or DEFAULT_MODEL_SIZE)
    if model_size not in BASE_CHECKPOINT_PATHS:
        raise ValueError(f"Experiment {experiment!r} sets model_size={model_size!r}, not one of {MODEL_SIZES}.")
    return BASE_CHECKPOINT_PATHS[model_size]


def load_for_inference(
    recipe: str,
    ft_checkpoint: str,
    base_checkpoint: Optional[str] = None,
    experiment: Optional[str] = None,
):
    """Single entry point for inference scripts: register the experiment recipe, then load the model.

    Importing ``anomalygen`` only registers the reusable groups (model / ckpt_type / callbacks); the
    experiment node is registered here from ``recipe`` before the loader hydra-composes it. The
    experiment name comes from the recipe; pass ``experiment`` only to override it (e.g. for a
    Python-module recipe whose name :func:`register_recipe` can't surface). ``base_checkpoint``
    defaults to the DCP matching the recipe's ``model_size``.
    """
    name = register_recipe(recipe) or experiment
    if name is None:
        raise ValueError(
            f"Recipe {recipe!r} did not yield an experiment name (a Python-module recipe?); "
            "pass experiment=... explicitly."
        )
    if base_checkpoint is None:
        base_checkpoint = _base_checkpoint_for_experiment(name)
    return load_finetuned_model(ft_checkpoint, base_checkpoint=base_checkpoint, experiment=name)


def _load_conditioning(image: Image.Image, h: int, w: int, device, dtype) -> torch.Tensor:
    """PIL RGB -> [3, 1, H, W] tensor in [-1, 1]."""
    img = image.convert("RGB").resize((w, h), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]
    return t.unsqueeze(1).to(device=device, dtype=dtype)  # [3,1,H,W]


def build_inpaint_batch(
    model,
    image: Image.Image,
    mask: Image.Image,
    class_id: int,
    caption: str,
    seed: Optional[int] = None,
) -> Dict:
    """Build a single-sample two-vision-item batch for inpainting one crop.

    ``seed`` seeds the source item's conditioning noise; ``None`` uses the global RNG.
    """
    device = next(model.net.parameters()).device
    dtype = model.tensor_kwargs["dtype"]
    w, h = image.size
    cond = _load_conditioning(image, h, w, device, dtype)  # [3,1,H,W] clean (guide target)
    m = (
        torch.from_numpy(np.asarray(mask.convert("L").resize((w, h), Image.Resampling.NEAREST), dtype=np.float32))
        >= MASK_FG_THRESHOLD
    ).float()
    m = m.to(device=device)  # [H,W]

    m3 = m[None, None].expand(cond.shape)  # [3,1,H,W]
    # Defect region -> noise, background kept: the one source encoding, matching training with
    # background dropout off (the dropout is a training-only augmentation).
    source = build_source_item(cond, m3, seed=seed)

    image_size = torch.tensor([[h, w, h, w]], dtype=torch.float32, device=device)
    batch = {
        "dataset_name": "anomaly",
        "images": [source.unsqueeze(0), cond.unsqueeze(0)],  # each [1,3,1,H,W]
        "image_size": [image_size, image_size],
        "num_frames": [torch.tensor([2], dtype=torch.int64, device=device)],
        "num_vision_items_per_sample": [2],
        "is_preprocessed": True,
        "sequence_plan": [SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[])],
        "edit_mask": m[None, None],  # [1,1,H,W]
        "anomaly_class_id": torch.tensor([int(class_id)], dtype=torch.long, device=device),
        model.input_caption_key: [caption],
    }

    return batch


def _build_inpaint_batch_multi(
    model,
    images: List[Image.Image],
    masks: List[Image.Image],
    class_ids: List[int],
    captions: List[str],
    seeds: Optional[List[int]] = None,
) -> Dict:
    """Merge B single-sample inpaint batches into one B-sample batch.

    Reuses :func:`build_inpaint_batch` per sample, then concatenates exactly as the framework's
    collate flattens per-sample list-valued keys (``images``/``image_size``/``num_frames``/
    ``sequence_plan`` flattened across samples; ``edit_mask``/``anomaly_class_id`` stacked on dim 0).
    ``seeds`` is one source-noise seed per sample, so noise never depends on batch position or size.
    """
    if seeds is None:
        seeds = [None] * len(images)
    singles = [
        build_inpaint_batch(model, img, msk, cid, cap, seed=sd)
        for img, msk, cid, cap, sd in zip(images, masks, class_ids, captions, seeds)
    ]
    return {
        "dataset_name": "anomaly",
        "is_preprocessed": True,
        "images": [vi for s in singles for vi in s["images"]],
        "image_size": [sz for s in singles for sz in s["image_size"]],
        "num_frames": [nf for s in singles for nf in s["num_frames"]],
        "num_vision_items_per_sample": [n for s in singles for n in s["num_vision_items_per_sample"]],
        "sequence_plan": [sp for s in singles for sp in s["sequence_plan"]],
        "edit_mask": torch.cat([s["edit_mask"] for s in singles], dim=0),
        "anomaly_class_id": torch.cat([s["anomaly_class_id"] for s in singles], dim=0),
        model.input_caption_key: [c for s in singles for c in s[model.input_caption_key]],
    }


def _decode_to_pil(model, vision_latent: torch.Tensor) -> Image.Image:
    """Decode a single sample's vision latent to a PIL RGB image.

    Normalizes the latent to 5D ``[B, C, T, H, W]`` as the VAE's conv3d requires.
    """
    latent = vision_latent
    while latent.dim() > 5 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)
    pix = model.decode(latent)  # [B,C,T,H,W] in [-1,1]

    pix = ((1.0 + pix) / 2.0).clamp(0, 1)[0, :, 0]  # [C,H,W]
    arr = (pix.permute(1, 2, 0).float().cpu().numpy() * 255.0).round().astype(np.uint8)

    return Image.fromarray(arr)


def _generate_inpaint(
    model,
    image: Image.Image,
    mask: Image.Image,
    class_id: int,
    caption: str,
    num_steps: int = DEFAULT_NUM_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    shift: float = DEFAULT_SHIFT,
    seed: int = 1,
) -> Image.Image:
    """Inpaint a single crop and return the edited crop as a PIL image.

    ``seed`` drives both noise streams: source conditioning and sampler latent.
    """
    batch = build_inpaint_batch(model, image, mask, class_id, caption, seed=seed)
    out = model.generate_samples_from_batch(batch, guidance=guidance, num_steps=num_steps, shift=shift, seed=[seed])
    return _decode_to_pil(model, out["vision"][0])


@torch.inference_mode()
def build_inpaint_one(
    model,
    class_ids: Dict[str, int],
    num_steps: int,
    guidance: float,
    model_input_size: int = 512,
    shift: float = DEFAULT_SHIFT,
    seed: int = 1,
):
    """Build the ``(crop, crop_mask, name) -> edited_crop`` closure for the iterative
    orchestrator. The crop is resized to ``model_input_size`` before generation;
    ``paste_back`` resizes the result back to the crop's native size.

    Seeds its n-th call with ``seed + n`` so a sample's instances don't share one noise draw; that
    seed drives both noise streams, so output depends only on the call's own seed.
    Stateful: build one closure per sample generation, never share it across samples.
    """
    instance_counter = itertools.count()

    def inpaint_one(cropped_image: Image.Image, cropped_mask: Image.Image, anomaly_name: str) -> Image.Image:
        # anomaly_name is data-driven (testcase JSONL "anomaly_type"); a typo or "{texture}+{defect}"
        # format mismatch must fail loudly instead of silently generating with class 0.
        if anomaly_name not in class_ids:
            raise KeyError(f"Unknown anomaly_name {anomaly_name!r}; known: {sorted(class_ids)}")
        cid = class_ids[anomaly_name]
        caption = caption_for_anomaly_type(anomaly_name)

        size = (model_input_size, model_input_size)
        if cropped_image.size != size:
            cropped_image = cropped_image.resize(size, Image.Resampling.BICUBIC)
            cropped_mask = cropped_mask.resize(size, Image.Resampling.NEAREST)

        return _generate_inpaint(
            model,
            cropped_image,
            cropped_mask,
            cid,
            caption,
            num_steps=num_steps,
            guidance=guidance,
            shift=shift,
            seed=seed + next(instance_counter),
        )

    return inpaint_one


@torch.inference_mode()
def build_inpaint_batch_fn(
    model,
    class_ids: Dict[str, int],
    num_steps: int,
    guidance: float,
    shift: float,
    model_input_size: int = 512,
):
    """Build a ``(crops, masks, names, seeds) -> edited_crops`` batched closure for the iterative driver.

    Crops are resized to ``model_input_size`` before generation. ``guidance``/``num_steps``/``shift``
    are batch-level scalars (the caller guarantees the batch is homogeneous in ``(guidance, shift)``);
    ``seeds`` is one seed per crop, driving both its noise streams, so results are invariant to
    batch composition. Returns one edited crop (PIL) per input crop.
    """
    size = (model_input_size, model_input_size)

    def fn(
        crops: List[Image.Image],
        masks: List[Image.Image],
        names: List[str],
        seeds: List[int],
    ) -> List[Image.Image]:
        cids: List[int] = []
        caps: List[str] = []
        rc: List[Image.Image] = []
        rm: List[Image.Image] = []
        for img, msk, name in zip(crops, masks, names):
            if name not in class_ids:
                raise KeyError(f"Unknown anomaly_name {name!r}; known: {sorted(class_ids)}")
            cids.append(class_ids[name])
            caps.append(caption_for_anomaly_type(name))
            if img.size != size:
                img = img.resize(size, Image.Resampling.BICUBIC)
                msk = msk.resize(size, Image.Resampling.NEAREST)
            rc.append(img)
            rm.append(msk)
        batch = _build_inpaint_batch_multi(model, rc, rm, cids, caps, seeds=[int(s) for s in seeds])
        out = model.generate_samples_from_batch(
            batch, guidance=guidance, num_steps=num_steps, shift=shift, seed=list(seeds)
        )
        return [_decode_to_pil(model, out["vision"][i]) for i in range(len(crops))]

    return fn
