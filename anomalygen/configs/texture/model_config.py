# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``DiffusionExpertConfig`` extended with the I2I anomaly-inpainting knobs.

With every flag at its default (all off), it behaves identically to the base config.
"""

from __future__ import annotations

from typing import List

import attrs
from cosmos_framework.configs.base.defaults.model_config import DiffusionExpertConfig

from anomalygen.configs.texture.constants import DEFAULT_MODEL_SIZE, DEFAULT_SHIFT


@attrs.define(slots=False)
class AnomalyGenTextureDiffusionExpertConfig(DiffusionExpertConfig):
    """``DiffusionExpertConfig`` + the I2I anomaly-inpainting knobs."""

    # Model size this config was built for; recorded so inference can pick the matching base DCP off a
    # registered node instead of re-parsing the recipe.
    model_size: str = DEFAULT_MODEL_SIZE

    # Anomaly class metadata (shared): `anomaly_num_classes` sizes `inpaint_class_emb` and
    # `text_prompt_emb`; `anomaly_types` are the ``[texture, defect]`` pairs in class-id order,
    # used to rebuild each class's caption for the learnable text-prompt template init.
    anomaly_num_classes: int = 0
    anomaly_types: List[List[str]] = attrs.field(factory=list)

    # --- per-defect-type conditioning: four independent mechanisms; any subset stacks -------------

    # 1) Per-defect-type learnable vision prompt item (`vision_prompt_emb`): a synthetic vision item
    # prepended before the source, whose K tokens are a per-defect-type learnable set. Injected as a
    # clean (no-noise, no-loss) conditioning item that both source and target attend to
    # bidirectionally. Laid out on a square (1, side, side) token grid (K = side*side).
    vision_prompt_item_enabled: bool = False
    # Number of learnable vision-prompt tokens K; must be a perfect square (default 256 = 16x16).
    vision_prompt_num_tokens: int = 256
    # Freeze the vision prompt (keep it at its init; train only the other enabled mechanisms).
    vision_prompt_freeze: bool = False

    # 2) Per-defect-type LoRA on the gen-attention projections: each class gets its own stacked
    # adapter, gen tokens are routed per forward, so all classes train in one mixed-batch run.
    # Mutually exclusive with the shared ``lora_enabled`` (same targets). Needs
    # ``use_cuda_graphs=False`` (the default) for compile-safe buffer routing.
    per_class_lora_enabled: bool = False
    per_class_lora_rank: int = 8
    per_class_lora_alpha: int = 8

    # 3) Post-VAE per-defect-type additive hidden embedding (`inpaint_class_emb`).
    inpaint_class_hidden_embed_enabled: bool = False

    # 4) Per-defect-type learnable text prompt (`text_prompt_emb`): the whole text-token sequence becomes
    # tunable, initialized from the caption template (padded to `text_prompt_num_tokens` with the
    # `text_prompt_init_word` token).
    learnable_text_prompt_enabled: bool = False
    # Fixed learnable prompt length (= caption tokens emitted per sample). Default 512.
    text_prompt_num_tokens: int = 512
    # Word whose token embedding initializes the padded tail of each class's prompt.
    text_prompt_init_word: str = "anomaly"
    # Freeze the learnable prompt (keep it at its template init; train only the other mechanisms).
    text_prompt_freeze: bool = False

    # Latent edit-mask downsampling: "any" (max-pool any-touch, keeps thin defects) | "nearest".
    inpaint_latent_mask_mode: str = "any"

    # Mask-emphasised flow-matching loss: extra supervision inside the defect region.
    inpaint_loss_mask_emphasis: bool = False
    inpaint_loss_mask_max_adaptive_weight: float = 100.0

    # Replacement trick at inference: latent hard-composite outside the mask each denoising step.
    inpaint_replacement_trick: bool = False

    # Distributed parallelism mode, mirroring ``trainer.distributed_parallelism``: "ddp" (default,
    # each rank keeps the full model) or "fsdp" (shard params/optimizer). Read by
    # ``AnomalyGenTextureMoTModel.set_up_parallelism``: under "ddp" it skips fully_shard so the DDP
    # wrapper can broadcast plain-tensor params.
    parallelism: str = "ddp"

    # Square pixel size crops are resized to before generation — the model's one input size, shared by
    # standalone inference (generate.py) and validation.
    model_input_size: int = 512

    # Validation-generation fallbacks, read by ``AnomalyGenTextureMoTModel.validation_step`` only when
    # a testcase omits the value: ``val_shift`` is the sampler time-shift; ``val_crop`` is the
    # fixed crop-grid used when a testcase omits ``crop_ratio``.
    val_shift: float = DEFAULT_SHIFT
    val_crop: int = 512
