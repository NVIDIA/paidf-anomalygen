# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable config builders for the texture I2I anomaly-inpainting experiment.

A recipe (e.g. ``ag_config/exp_texture_ft_phone_screen.yaml``) only needs to supply the
dataset-specific values (paths, ``anomaly_types``, ``dataset_name``) and call
``build_anomalygen_texture_ft_experiment`` to get the full experiment ``LazyDict`` to ``cs.store``.
Architecture knobs that define *what the texture model is* (any-touch latent mask, replacement
trick, mask-emphasis weight, LoRA target modules) live here, not in the recipe; the commonly
tuned values are ``build_anomalygen_texture_ft_experiment`` parameters with sensible defaults — paths,
anomaly types, schedule (``max_iter``/``validation_iter``/``cycle_lengths``/``warm_up_steps``),
optimizer (``lr``/``betas``/``eps``/``weight_decay``), data (``batch_size``/``num_workers``/
``ratio_range``/``image_size``), early stop, ``logging_iter``, ``seed``, etc.

``ANOMALYGEN_PARALLELISM`` (ddp/fsdp) and ``ANOMALYGEN_COMPILE`` (0/1) are read from the environment
here so the documented launch command is unchanged.
"""

from __future__ import annotations

import copy
import inspect
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from anomalygen.checkpoint.trained_keys import TRAINED_KEY_PREFIXES, WARM_START_SKIP_PREFIXES
from anomalygen.configs.texture.constants import (
    BASE_CHECKPOINT_PATHS,
    DEFAULT_MODEL_SIZE,
    DEFAULT_SHIFT,
    MODEL_SIZES,
)
from anomalygen.data.inpaint_inference_dataset import get_inpaint_val_dataloader
from anomalygen.data.inpainting_dataset import get_inpainting_dataset

# Frozen base ``model.config`` per size. Framework constants: deep-copy before mutating, which
# ``_base_model_config`` is the only place to do.
_SIZE_MODEL_CONFIGS = {
    "nano": NANO_MODEL_CONFIG,
    "edge": EDGE_MODEL_CONFIG,
}


def _base_model_config(model_size: str) -> dict:
    """Deep-copy the frozen base ``model.config`` for ``model_size``."""
    if model_size not in _SIZE_MODEL_CONFIGS:
        raise ValueError(f"model_size={model_size!r} is not one of {MODEL_SIZES}.")
    return copy.deepcopy(_SIZE_MODEL_CONFIGS[model_size])


# Repo root, derived from this module's location (anomalygen/configs/texture/exp_config.py). Used to
# absolutize recipe-supplied relative paths: training chdir's to the framework checkout root for the
# model build, so dataset / checkpoint / VAE paths must be absolute to survive that.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _abs(path: str) -> str:
    """Absolutize ``path`` against the repo root (no-op if already absolute)."""
    return path if os.path.isabs(path) else str(_REPO_ROOT / path)


def build_anomalygen_texture_ft_model_config(
    num_classes: int,
    vae_path: str,
    *,
    anomaly_types: Sequence[Sequence[str]] = (),
    lora_enabled: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    resolution: str = "480",
    vision_prompt_item_enabled: bool = False,
    vision_prompt_num_tokens: int = 256,
    vision_prompt_freeze: bool = False,
    per_class_lora_enabled: bool = True,
    per_class_lora_rank: int = 8,
    per_class_lora_alpha: int = 8,
    inpaint_class_hidden_embed_enabled: bool = False,
    learnable_text_prompt_enabled: bool = False,
    text_prompt_num_tokens: int = 512,
    text_prompt_init_word: str = "anomaly",
    text_prompt_freeze: bool = False,
    model_input_size: int = 512,
    shift: float = DEFAULT_SHIFT,
    crop: int = 512,
    parallelism: str = "ddp",
    model_size: str = DEFAULT_MODEL_SIZE,
) -> dict:
    """Base model config for the size + the texture anomaly-inpainting overrides.

    Deep-copies the ``model_size`` baseline (``nano`` -> Qwen3-VL-8B, ``edge`` -> Nemotron-3 Dense
    VL 2B) so the vlm_config / tokenizer / net wiring is inherited rather than restated; everything
    overridden below is size-independent. ``resolution`` selects the sampler time-shift
    (480 -> 5.0; 720 -> 10.0).

    The four per-defect-type conditioning mechanisms are independent and stack — any subset may be
    enabled at once: the learnable vision prompt item (``vision_prompt_item_enabled``), the
    per-defect-type LoRA (``per_class_lora_enabled``), the additive hidden embedding
    (``inpaint_class_hidden_embed_enabled``), and the learnable text prompt
    (``learnable_text_prompt_enabled``).
    """
    cfg = _base_model_config(model_size)
    cfg["lora_enabled"] = lora_enabled
    cfg["lora_rank"] = lora_rank
    cfg["lora_alpha"] = lora_alpha
    cfg["lora_target_modules"] = "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"
    cfg["resolution"] = resolution  # selects the sampler time-shift

    _dec = cfg.setdefault("diffusion_expert_config", {})
    # Recorded so inference can read the size off a registered node (see inference/inpaint.py).
    _dec["model_size"] = model_size
    _dec["anomaly_num_classes"] = num_classes
    _dec["anomaly_types"] = [list(p) for p in anomaly_types]
    _dec["inpaint_latent_mask_mode"] = "any"
    _dec["inpaint_loss_mask_emphasis"] = True
    _dec["inpaint_loss_mask_max_adaptive_weight"] = 100.0
    _dec["inpaint_replacement_trick"] = True
    # Per-defect-type conditioning — four independent mechanisms (any subset stacks).
    # 1) Per-defect-type learnable vision prompt item (a clean square-grid item prepended before source).
    _dec["vision_prompt_item_enabled"] = vision_prompt_item_enabled
    _dec["vision_prompt_num_tokens"] = vision_prompt_num_tokens
    _dec["vision_prompt_freeze"] = vision_prompt_freeze
    # 2) Per-defect-type LoRA on the gen-expert attention projections (routed per gen token).
    _dec["per_class_lora_enabled"] = per_class_lora_enabled
    _dec["per_class_lora_rank"] = per_class_lora_rank
    _dec["per_class_lora_alpha"] = per_class_lora_alpha
    # 3) Post-VAE per-defect-type additive hidden embedding.
    _dec["inpaint_class_hidden_embed_enabled"] = inpaint_class_hidden_embed_enabled
    # 4) Per-defect-type learnable text prompt (whole text sequence tunable, template-initialized).
    _dec["learnable_text_prompt_enabled"] = learnable_text_prompt_enabled
    _dec["text_prompt_num_tokens"] = text_prompt_num_tokens
    _dec["text_prompt_init_word"] = text_prompt_init_word
    _dec["text_prompt_freeze"] = text_prompt_freeze
    # Distributed parallelism mode ("ddp"/"fsdp"), read by set_up_parallelism: "ddp" skips
    # fully_shard so each rank keeps the full model for the DDP wrapper to broadcast.
    _dec["parallelism"] = parallelism
    # Generation params read by validation_step (class_ids is derived from anomaly_types).
    # model_input_size is the model's one input size (not val-specific); val_shift/val_crop are
    # per-testcase fallbacks.
    _dec["model_input_size"] = model_input_size
    _dec["val_shift"] = shift
    _dec["val_crop"] = crop

    # Point the VAE tokenizer at the local checkpoint (overrides the framework's relative default).
    cfg["tokenizer"]["vae_path"] = vae_path
    # Disable the model-level EMA. The nano base enables it, which makes the model build a second,
    # float32 copy of the network (net_ema) — ~2x the bf16 net, the bulk of single-GPU memory. We
    # don't use EMA here (trainer `/ema` is overridden to None and net_ema. is skipped at load),
    # so turning it off keeps only the bf16 net resident and lets the model fit one GPU under DDP.
    cfg["ema"]["enabled"] = False
    # torch.compile is env-gated: ON by default, set ANOMALYGEN_COMPILE=0 to disable.
    # The nano base ships enabled=True with compiled_region="language", which compiles ONLY the MoT
    # transformer blocks (parallelize_unified_mot.apply_compile) — the heavy compute. Those blocks take
    # the packed-sequence tensor as input and never read the live inpaint context, so the side-channel
    # (network.set_anomaly_context) does NOT force recompiles there. The recompile / stale-context
    # hazard only arises at compiled_region="all", which additionally compiles the encode/decode heads,
    # including our _encode_vision injection. We keep "language", so the head (and its per-defect-type additive
    # embed) stays eager and enabling compile is safe. The USE_TORCH_COMPILE env var is not read
    # anywhere; this flag is the switch.
    cfg["compile"]["enabled"] = os.environ.get("ANOMALYGEN_COMPILE", "1").strip() == "1"
    # Pin the region to "language" (inherited from nano) so the recipe is self-documenting: compile the
    # transformer blocks, never the encode/decode heads where the eager inpaint-context injection lives.
    cfg["compile"]["compiled_region"] = "language"
    if cfg["compile"].get("compiled_region") not in (None, "language"):
        raise ValueError("AnomalyGen Texture FT requires compiled_region in {None, 'language'}")
    # Auto-select the FSDP shard degree from the torchrun world size so one recipe scales to any
    # GPU count (ignored under DDP). The nano base pins it to 8.
    cfg["parallelism"]["data_parallel_shard_degree"] = -1
    return cfg


def build_anomalygen_texture_ft_experiment(
    *,
    dataset_name: str,
    job_name: str = "anomalygen_texture_ft",
    anomaly_types: Sequence[Sequence[str]],
    dataset_path: str,
    testcase_jsonl: str,
    recipe_path: Optional[str] = None,
    # Frozen backbone size ("nano" | "edge"); also picks base_checkpoint_path when it is omitted.
    model_size: str = DEFAULT_MODEL_SIZE,
    base_checkpoint_path: Optional[str] = None,
    vae_path: Optional[str] = None,
    image_size: Tuple[int, int] = (512, 512),
    max_iter: int = 15000,
    validation_iter: int = 1000,
    run_validation_on_start: bool = True,
    save_iter: int = 1000,
    lr: float = 1.0e-03,
    model_input_size: int = 512,
    shift: float = DEFAULT_SHIFT,
    # model / LoRA
    lora_enabled: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    # per-defect-type conditioning methods (independent; any subset stacks)
    vision_prompt_item_enabled: bool = False,
    vision_prompt_num_tokens: int = 256,
    vision_prompt_freeze: bool = False,
    # per-defect-type LoRA (routed per gen token; trains all classes in one mixed-batch run)
    per_class_lora_enabled: bool = True,
    per_class_lora_rank: int = 8,
    per_class_lora_alpha: int = 8,
    inpaint_class_hidden_embed_enabled: bool = False,
    learnable_text_prompt_enabled: bool = False,
    text_prompt_num_tokens: int = 512,
    text_prompt_init_word: str = "anomaly",
    text_prompt_freeze: bool = False,
    # optimizer (AdamW)
    betas: Sequence[float] = (0.9, 0.95),
    eps: float = 1.0e-06,
    weight_decay: float = 0.0,
    # Per-key LR multipliers: name-substring -> factor on the base ``lr`` (empty => all trained
    # params share the base LR). Applied by the framework optimizer to any param whose name
    # contains the substring; useful to e.g. give the learnable text prompt a larger step.
    lr_multipliers: Optional[Dict[str, float]] = None,
    # LambdaCosine scheduler (single warmup -> cosine-decay cycle)
    cycle_lengths: int = 15000,
    warm_up_steps: int = 500,
    # trainer
    logging_iter: int = 10,
    seed: int = 42,
    # validation metrics: anomaly-quality axes (completeness/precision/boundary_iou) + the aq_nn
    # composite (= completeness + nn_score) are on by default; set False to skip their per-validation
    # SAM2 load (see anomalygen.eval.anomaly_quality).
    compute_anomaly_quality: bool = True,
    # early stopping (disabled by default)
    early_stop_enabled: bool = False,
    early_stop_metric: str = "nn",
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 0.0,
    early_stop_min_delta_mode: str = "rel",
    # data
    batch_size: int = 4,
    validation_batch_size: int = 16,
    num_workers: int = 4,
    ratio_range: Sequence[float] = (1.5, 8.0),
    # Per-sample prob the training source item's background is dropped to -1 (black; defect region is noise
    # either way); training-only, inference always keeps the background.
    background_dropout_prob: float = 0.5,
    # Per-sample prob of the keep-one-instance augmentation (drop all but one defect instance,
    # blacking the dropped ones out in the image). 0 = off.
    inst_aug_prob: float = 0.5,
    # Per-sample prob of the ring colour jitter (recolour a band around the defect; the band width
    # itself stays on $ANOMALYGEN_RING_JITTER_PX). 0 = off.
    ring_jitter_prob: float = 0.5,
) -> LazyDict:
    """Assemble the full ``anomalygen_texture_ft`` experiment ``LazyDict`` for a texture dataset.

    The caller (a recipe) supplies the dataset-specific values and then ``cs.store``s the result.
    ``anomaly_types`` is a list of ``[texture, defect]`` pairs; ``num_classes`` and the
    ``"{texture}+{defect}" -> id`` map are derived from it. ``recipe_path`` is the caller's own
    file path, copied into the run dir by the TrainingReport callback.
    """
    # Absolutize all paths against the repo root (recipes may pass repo-root-relative paths, but the
    # model build runs under a chdir to the framework root). base_checkpoint_path / vae_path default
    # to the repo-root checkpoints/ tree when a recipe omits them.
    dataset_path = _abs(dataset_path)
    testcase_jsonl = _abs(testcase_jsonl)
    # recipe_path defaults to the calling recipe module's own file (so TrainingReport snapshots it);
    # the YAML/JSON loader passes the recipe file explicitly.
    recipe_path = _abs(recipe_path or inspect.stack()[1].filename)
    if model_size not in MODEL_SIZES:
        raise ValueError(f"model_size={model_size!r} is not one of {MODEL_SIZES}.")
    base_checkpoint_path = _abs(base_checkpoint_path or BASE_CHECKPOINT_PATHS[model_size])
    vae_path = _abs(vae_path or "checkpoints/wan2pt2/Wan2.2_VAE.pth")

    anomaly_types = [list(p) for p in anomaly_types]
    num_classes = len(anomaly_types)
    image_size_list: List[int] = list(image_size)

    # Parallelism is env-switchable: ANOMALYGEN_PARALLELISM=ddp (default, each rank holds the full,
    # DDP-replicated model) or =fsdp (shard params/optimizer across all launched GPUs). This single
    # value drives three things that must agree: the model group node, trainer.distributed_parallelism,
    # and the model's own ``_dec.parallelism`` (read by set_up_parallelism to skip fully_shard under DDP).
    parallelism = os.environ.get("ANOMALYGEN_PARALLELISM", "ddp").strip().lower()
    if parallelism not in ("ddp", "fsdp"):
        raise ValueError(f"ANOMALYGEN_PARALLELISM must be 'ddp' or 'fsdp', got {parallelism!r}")
    model_group = "anomalygen_texture_ft_mot_fsdp" if parallelism == "fsdp" else "anomalygen_texture_ft_mot_ddp"

    model_config = build_anomalygen_texture_ft_model_config(
        num_classes,
        vae_path,
        anomaly_types=anomaly_types,
        model_size=model_size,
        lora_enabled=lora_enabled,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        vision_prompt_item_enabled=vision_prompt_item_enabled,
        vision_prompt_num_tokens=vision_prompt_num_tokens,
        vision_prompt_freeze=vision_prompt_freeze,
        per_class_lora_enabled=per_class_lora_enabled,
        per_class_lora_rank=per_class_lora_rank,
        per_class_lora_alpha=per_class_lora_alpha,
        inpaint_class_hidden_embed_enabled=inpaint_class_hidden_embed_enabled,
        learnable_text_prompt_enabled=learnable_text_prompt_enabled,
        text_prompt_num_tokens=text_prompt_num_tokens,
        text_prompt_init_word=text_prompt_init_word,
        text_prompt_freeze=text_prompt_freeze,
        model_input_size=model_input_size,
        shift=shift,
        parallelism=parallelism,
    )

    return LazyDict(
        dict(
            defaults=[
                {"override /model": model_group},
                {"override /data_train": None},
                {"override /data_val": None},
                {"override /optimizer": "adamw"},
                {"override /scheduler": "lambdacosine"},
                {"override /checkpoint": "local"},
                {"override /callbacks": ["basic", "optimization", "anomalygen_texture_ft"]},
                {"override /ema": None},
                {"override /tokenizer": "wan2pt2_tokenizer"},
                {"override /sound_tokenizer": None},
                {"override /vlm_config": None},
                {"override /ckpt_type": "selective_dcp_anomalygen_texture_ft"},  # tiny filtered checkpoints
                "_self_",
            ],
            job=dict(project="anomalygen", group=dataset_name, name=job_name, wandb_mode="disabled"),
            model=dict(config=model_config),
            optimizer=dict(
                betas=list(betas),
                eps=eps,
                fused=True,
                # Train only the LoRA adapters, the per-defect-type embedding, and the text prompt.
                keys_to_select=list(TRAINED_KEY_PREFIXES),
                # Per-key LR multipliers (name-substring -> factor on base LR). Empty by default
                # => all trained params share the base LR.
                lr_multipliers=dict(lr_multipliers or {}),
                lr=lr,
                optimizer_type="AdamW",
                weight_decay=weight_decay,
            ),
            scheduler=dict(
                # Single-cycle warmup -> cosine decay: the scheduler takes parallel per-cycle lists,
                # so the scalar knobs are wrapped in one-element lists here.
                lr_scheduler_type="LambdaCosine",
                cycle_lengths=[cycle_lengths],
                f_max=[1.0],
                f_min=[0.1],
                f_start=[0.0],
                warm_up_steps=[warm_up_steps],
            ),
            trainer=dict(
                distributed_parallelism=parallelism,
                grad_accum_iter=1,
                logging_iter=logging_iter,
                max_iter=max_iter,
                run_validation=True,
                validation_iter=validation_iter,
                run_validation_on_start=run_validation_on_start,
                seed=seed,
                grad_scaler_args=dict(enabled=False),
                # Hand this recipe's own values to the callbacks registered (arg-free) by
                # anomalygen.register; merges onto the `anomalygen_texture_ft` callbacks group.
                callbacks=dict(
                    # Override the optimization group's default clip_norm (1.0) → tighter 0.1.
                    grad_clip=dict(clip_norm=0.1),
                    # ValidationKPI writes valid_kpi.csv each validation_iter (nn/mnn/fid always;
                    # axes + aq_nn when compute_anomaly_quality=True — these need SAM2).
                    validation_kpi=dict(compute_anomaly_quality=compute_anomaly_quality),
                    # Early stopping on the monitored validation metric (the EarlyStop callback reads
                    # the valid_kpi.csv ValidationKPI writes each validation_iter). metric ∈ {nn, mnn,
                    # fid, aq_nn, completeness, precision, boundary_iou}; stops when it fails to improve for `patience` consecutive
                    # validations. Disabled by default; set enabled=True to use it.
                    early_stop=dict(
                        enabled=early_stop_enabled,
                        metric=early_stop_metric,
                        patience=early_stop_patience,
                        scope="Average",
                        min_delta=early_stop_min_delta,
                        min_delta_mode=early_stop_min_delta_mode,
                        cumulative_delta=False,
                    ),
                    # best_metric mirrors early_stop_metric so best_checkpoint.txt is selected on the
                    # same metric the run was monitored on (see the EarlyStop block above); it is
                    # honored whether or not early stopping is enabled.
                    training_report=dict(recipe_path=recipe_path, best_metric=early_stop_metric),
                ),
            ),
            checkpoint=dict(
                dcp_async_mode_enabled=False,  # required for the save-keys filter (sync save path)
                # The trained subset (+ EMA) is absent from the base DCP checkpoint → skip loading
                # it (trained from scratch / template-initialized).
                keys_to_skip_loading=list(WARM_START_SKIP_PREFIXES),
                load_path=base_checkpoint_path,
                save_iter=save_iter,
                strict_resume=False,
                verbose=True,
            ),
            dataloader_train=L(PackingDataLoader)(
                dataset_name="anomaly",
                patch_spatial=2,
                tokenizer_spatial_compression_factor=16,
                tokenizer_temporal_compression_factor=4,
                # Batch by sample count, not token budget. The nano recipe packs to
                # max_sequence_length=45056 because video samples are huge; our 512x512 image
                # samples are ~560 tokens each, so a token budget would cram ~80 samples into
                # every step and the packing loop would dominate wall-clock. Pack 2 samples per
                # step instead (mirrors the reference anomaly recipe's batch_size=2). Exactly one
                # of max_sequence_length / max_samples_per_batch must be set.
                max_sequence_length=None,
                max_samples_per_batch=batch_size,  # THE training batch size: samples packed per step.
                dataloader=L(RankPartitionedDataLoader)(
                    # Inner per-pull size. Kept equal to max_samples_per_batch only to avoid the
                    # misleading look of "batch_size=1"; it does NOT multiply the batch — the packed
                    # batch is capped at max_samples_per_batch regardless of this value.
                    batch_size=batch_size,
                    num_workers=num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                    prefetch_factor=4,
                    datasets=dict(
                        image=dict(
                            ratio=1,
                            dataset=L(get_inpainting_dataset)(
                                dataset_dir=dataset_path,
                                anomaly_types=anomaly_types,
                                image_size=image_size_list,
                                ratio_range=list(ratio_range),
                                # Same seed as trainer.seed: drives the per-worker augmentation rng so
                                # the augmentation stream is reproducible from the experiment seed.
                                seed=seed,
                                # Cycle the (few-shot) sample set to a large virtual length so the
                                # PackingDataLoader's one-shot iterator never exhausts mid-run (it
                                # busy-spins on StopIteration otherwise). 100000x covers any max_iter.
                                repeat=100000,
                                # Per-sample prob the source item's background is dropped to -1 (black).
                                background_dropout_prob=background_dropout_prob,
                                # Per-sample probs for the two read-time defect augmentations.
                                inst_aug_prob=inst_aug_prob,
                                ring_jitter_prob=ring_jitter_prob,
                                # Pad every caption to the learnable-prompt length so each sample
                                # emits a fixed text-token count (None when the feature is off).
                                text_prompt_pad_to=(text_prompt_num_tokens if learnable_text_prompt_enabled else None),
                                text_prompt_pad_word=text_prompt_init_word,
                                tokenizer_config="${model.config.vlm_config.tokenizer}",
                            ),
                        ),
                    ),
                ),
            ),
            # Real sharded validation loader: each rank generates a disjoint, (guidance,shift)-
            # homogeneous shard in model.validation_step; ValidationKPI.on_validation_end gathers +
            # scores.
            dataloader_val=L(get_inpaint_val_dataloader)(
                input_data_path=testcase_jsonl,
                val_batch_size=validation_batch_size,
                shift=shift,
                num_workers=num_workers,
                # Training instantiates dataloaders under a chdir to the framework checkout, so
                # repo-root-relative testcase image/mask paths must resolve against the repo root.
                base_dir=str(_REPO_ROOT),
            ),
        ),
        flags={"allow_objects": True},
    )
