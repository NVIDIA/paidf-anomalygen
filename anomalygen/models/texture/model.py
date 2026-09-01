# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``OmniMoTModel`` wired for I2I anomaly-inpainting generation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.parallelize_vfm_network import parallelize_vfm_network
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.utils import distributed, log, misc
from cosmos_framework.utils.flags import DEVICE, Device
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate

from anomalygen.configs.texture.constants import DEFAULT_SHIFT
from anomalygen.data.utils import build_caption, pad_or_truncate, resolve_word_token_id
from anomalygen.inference.inpaint import build_inpaint_batch_fn
from anomalygen.inference.iterative import run_iterative_inpaint_batch, split_mask_into_instances
from anomalygen.models.texture.network import AnomalyVFMNetwork
from anomalygen.models.texture.per_class_lora import (
    PerClassLoraLinear,
    init_per_class_lora_weights,
    inject_per_class_lora,
    per_class_lora_buffer_names,
    per_class_lora_target_modules,
)


def downsample_edit_mask(
    mask: torch.Tensor,
    t_lat: int,
    h_lat: int,
    w_lat: int,
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Downsample a pixel-space binary mask to a ``[1, T_lat, H_lat, W_lat]`` latent mask.

    ``mode`` is ``"any"`` (max-pool any-touch) or ``"nearest"``.
    """
    m = mask
    if m.dim() == 2:  # [H, W]
        m_in = m.unsqueeze(0).unsqueeze(0).float()
    elif m.dim() == 3:  # [1, H, W]
        m_in = m.unsqueeze(0).float()
    elif m.dim() == 4:  # [1, 1, H, W]
        m_in = m.float()
    else:
        raise ValueError(f"Unexpected edit_mask shape: {tuple(m.shape)}")

    m_in = m_in.to(device=device)

    if mode == "any":
        h_pix, w_pix = m_in.shape[-2:]
        kh, kw = max(1, h_pix // h_lat), max(1, w_pix // w_lat)
        m_down = F.max_pool2d(m_in, kernel_size=(kh, kw), stride=(kh, kw))  # [1,1,H_lat,W_lat]
    else:
        m_down = F.interpolate(m_in, size=(h_lat, w_lat), mode="nearest")

    m_down = (m_down > 0.5).to(dtype=dtype)
    return m_down.squeeze(0).expand(1, t_lat, h_lat, w_lat).contiguous()


def to_4d_chtw(t: torch.Tensor) -> torch.Tensor:
    """Drop leading singleton dims so a ``[1,C,T,H,W]`` prediction becomes ``[C,T,H,W]``."""
    while t.dim() > 4 and t.shape[0] == 1:
        t = t[0]
    return t


def mask_emphasis_term(
    pred: List[torch.Tensor],
    target: List[torch.Tensor],
    condition_mask: List[torch.Tensor],
    edit_masks: Sequence[Optional[torch.Tensor]],
    max_adaptive_weight: float,
) -> Optional[torch.Tensor]:
    """Additive mask-region emphasis for the vision flow-matching loss.

    Per supervised (noised) vision item computes ``adaptive_i * mean(sqerr * edit_mask * noisy_mask)``
    where ``adaptive = clamp(total_pixels / mask_area, 1, max)``, then averages those items. Pure
    conditioning items (source / prompt) are excluded so they don't dilute the mean. Returns
    ``None`` when no item qualifies.
    """
    if not edit_masks or max_adaptive_weight <= 0.0:
        return None

    terms: List[torch.Tensor] = []
    for i in range(len(pred)):
        if i >= len(edit_masks) or edit_masks[i] is None:
            continue
        sqerr_i = (to_4d_chtw(pred[i]) - to_4d_chtw(target[i])) ** 2  # [C,T,H,W]
        if sqerr_i.dim() != 4:
            continue

        em_i = edit_masks[i].to(dtype=sqerr_i.dtype, device=sqerr_i.device)  # [1,T,H,W]
        if em_i.shape[-3:] != sqerr_i.shape[-3:]:
            continue

        # noisy_mask: 1 on generated frames, 0 on clean/source frames; broadcast over [C,T,H,W].
        t_dim = sqerr_i.shape[-3]
        cm = condition_mask[i].to(dtype=sqerr_i.dtype, device=sqerr_i.device).reshape(-1)
        if cm.numel() == t_dim:
            noisy_mask_i = (1.0 - cm).view(1, t_dim, 1, 1)
        else:
            noisy_mask_i = (1.0 - cm).mean().reshape(1, 1, 1, 1)

        # Skip pure conditioning items (source / prompt): their noisy_mask is all-zero, so the term
        # is structurally 0 and would only drag down the mean over the supervised (target) items —
        # halving the effective emphasis for the standard [source, target] pair.
        if float(noisy_mask_i.max()) <= 0.0:
            continue

        mask_area = em_i.sum().clamp(min=1.0)
        adaptive_i = torch.clamp(float(em_i.numel()) / mask_area, min=1.0, max=max_adaptive_weight)
        terms.append(adaptive_i * (sqerr_i * em_i * noisy_mask_i).mean())

    if not terms:
        return None

    return torch.stack(terms).mean()


def replacement_velocity(
    flat_velocity: torch.Tensor,
    flat_latent: torch.Tensor,
    guide_latent: torch.Tensor,
    edit_mask_latent: torch.Tensor,
    offset: int,
    sigma: float,
) -> torch.Tensor:
    """Apply the inference replacement trick to one sample's flat velocity.

    Overwrites the background (non-mask) latent positions of the target vision item with the
    clean guide at each step, then converts back to velocity:
    ``x0 = x_t - sigma*v`` → ``x0_new = keep*guide + (1-keep)*x0`` → ``v_new = (x_t - x0_new)/sigma``,
    where ``keep = 1 - edit_mask`` (1 = anchor to guide). ``offset`` is the target item's start
    index within the flat ``[D]`` vectors; ``sigma`` is the noise level in ``[0, 1]``.
    """
    shape = tuple(guide_latent.shape)
    length = int(guide_latent.numel())
    lat_target = flat_latent[offset : offset + length].reshape(shape)
    vel_target = flat_velocity[offset : offset + length].reshape(shape)
    keep = (1.0 - edit_mask_latent).to(dtype=lat_target.dtype, device=lat_target.device)  # 1 = guide
    guide = guide_latent.to(dtype=lat_target.dtype, device=lat_target.device)

    x0_pred = lat_target - sigma * vel_target
    x0_new = keep * guide + (1.0 - keep) * x0_pred
    vel_new = (lat_target - x0_new) / sigma

    out = flat_velocity.clone()
    out[offset : offset + length] = vel_new.reshape(-1)
    return out


class AnomalyGenTextureMoTModel(OmniMoTModel):
    def __init__(self, config):
        super().__init__(config)
        # Transient per-batch latent edit masks, read by the loss and the replacement trick.
        # (Class ids also live on the network via set_anomaly_context.)
        self._anomaly_edit_masks: Optional[List[Optional[torch.Tensor]]] = None
        # Vision items per sample for the current batch: 2 ([source, target]) or 3 when the
        # learnable vision prompt item is prepended ([prompt, source, target]).
        self._anomaly_items_per_sample: int = 2
        try:
            self._device = next(self.net.parameters()).device
        except (StopIteration, AttributeError):
            self._device = getattr(self, "device", torch.device("cpu"))

    def set_up_parallelism(self) -> None:
        """Skip FSDP under DDP so each rank keeps the full, unsharded model.

        For any multi-GPU VFM run, ``ParallelDims.dp_enabled`` is always True, so the base
        ``set_up_parallelism`` builds a shard mesh and ``build_net`` applies ``fully_shard`` — the
        params become DTensors. The trainer then *also* wraps the model with DDP
        (``distributed_parallelism="ddp"``), and DDP's init-time param broadcast cannot operate on
        DTensors (``c10d.broadcast_`` / uneven-flatten errors). To run true DDP data-parallel
        (each rank holds the full model; grads all-reduced by the DDP wrapper), we must not shard:
        leaving ``parallel_dims=None`` makes ``build_net`` skip FSDP and behave like the single-GPU
        path (compile/activation-checkpointing still apply). The ``parallelism`` mode is set by the
        experiment recipe from the same ``ANOMALYGEN_PARALLELISM`` switch that picks the ddp/fsdp
        model group + trainer ``distributed_parallelism``, so the three stay consistent. Under
        ``parallelism="fsdp"``, defer to the base sharded setup.
        """
        if getattr(self._dec, "parallelism", "ddp") == "ddp":
            self.parallel_dims = None
            return
        super().set_up_parallelism()

    @torch.inference_mode()
    def validation_step(self, data_batch: Dict, iteration: int = 0):
        """Batched, multi-GPU validation generation. Accumulates onto the model; scoring happens in
        the ValidationKPI callback's ``on_validation_end``. No-op (harmless tuple) when no KPI pass
        is active — ``validation_generated_images_by_anomaly`` is initialized by the callback's
        on_validation_start, so its absence means the empty val loader / no ValidationKPI callback."""
        zero = torch.zeros((), device=self._device)
        if getattr(self, "validation_generated_images_by_anomaly", None) is None:
            return {}, zero
        self._generate_validation_batch(data_batch)
        return {}, zero

    def _generate_validation_batch(self, data_batch: Dict) -> None:
        """Generate one sharded batch of testcases and accumulate results onto ``self``.

        Reads per-record generation params from ``data_batch`` and the fixed generation config from
        ``self._dec`` (``model_input_size`` / ``val_shift`` / ``val_crop``; ``class_ids`` derived
        from ``anomaly_types``), builds the batched ``(crops, masks, names, seeds) -> edited_crops``
        inpaint callable via :func:`build_inpaint_batch_fn`, batches generation across images at each
        iterative instance-depth, and appends to ``self.validation_generated_images_by_anomaly`` /
        ``self.validation_sample_indices_by_anomaly`` (the callback gathers + scores these in
        on_validation_end).
        """

        def _to_float(img):
            return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

        def _mask_to_float(img):
            return np.asarray(img.convert("L"), dtype=np.float32) / 255.0

        dec = self._dec  # fixed generation config lives on the model config, not the callback
        class_ids = {f"{t}+{d}": i for i, (t, d) in enumerate(dec.anomaly_types)}
        names: List[str] = list(data_batch["anomaly_type"])
        images = list(data_batch["image"])
        masks = list(data_batch["mask"])
        indices = [int(x) for x in data_batch["index"]]
        seeds = [int(s) for s in data_batch["seed"]]
        b = len(names)

        # (guidance, shift, num_steps) are homogeneous within the batch (guaranteed by
        # _build_val_batch_indices for guidance/shift; num_steps is globally consistent).
        guidance = float(data_batch["guidance"][0])
        num_steps = int(data_batch["num_steps"][0])
        shift = (
            float(data_batch["shift"][0]) if "shift" in data_batch else float(getattr(dec, "val_shift", DEFAULT_SHIFT))
        )

        max_inst = [int(x) for x in data_batch["iteration_generation_max_instance"]]
        crop_and_pastes = [bool(x) for x in data_batch["crop_and_paste"]]
        # crop_ratio: "none"/None/"" -> None (use fixed grid), else float.
        crop_ratios = [None if x in (None, "none", "None", "") else float(x) for x in data_batch["crop_ratio"]]
        poisson = [bool(x) for x in data_batch["poisson_blend"]]
        # Fixed crop grid (used when crop_ratio is off); no per-testcase override.
        default_grid = int(getattr(dec, "val_crop", 512))
        crop_grids = [(default_grid, default_grid) for _ in crop_ratios]

        # FSDP-consistent instance depth: the max instance count across ranks for THIS batch, so every
        # rank runs the same number of forwards (guarded on world_size to stay single-GPU-safe).
        local_depth = max((len(split_mask_into_instances(masks[i], max_k=max_inst[i])) for i in range(b)), default=0)
        num_depth = local_depth
        if distributed.get_world_size() > 1:
            t = torch.tensor([local_depth], device="cuda")
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            num_depth = int(t.item())

        inpaint_fn = build_inpaint_batch_fn(
            self,
            class_ids=class_ids,
            num_steps=num_steps,
            guidance=guidance,
            shift=shift,
            model_input_size=int(getattr(dec, "model_input_size", 512)),
        )
        composites, artifacts = run_iterative_inpaint_batch(
            images,
            masks,
            names,
            inpaint_fn,
            num_depth=num_depth,
            seeds=seeds,
            crop_grids=crop_grids,
            crop_ratios=crop_ratios,
            crop_and_pastes=crop_and_pastes,
            poisson_blends=poisson,
            max_instances_list=max_inst,
        )

        gd = self.validation_generated_images_by_anomaly
        si = self.validation_sample_indices_by_anomaly
        for i in range(b):
            name = names[i]
            bucket = gd.setdefault(
                name,
                {
                    "reconstructed_image": [],
                    "original_image": [],
                    "original_mask": [],
                    "cropped_image": [],
                    "cropped_mask": [],
                    "annotated_image": [],
                    "mask_cropped_image": [],
                },
            )
            bucket["reconstructed_image"].append(_to_float(composites[i]))
            bucket["original_image"].append(_to_float(images[i]))
            bucket["original_mask"].append(_mask_to_float(masks[i]))
            bucket["cropped_image"].append([_to_float(a) for a in artifacts[i]["cropped_image"]])
            bucket["cropped_mask"].append([_mask_to_float(a) for a in artifacts[i]["cropped_mask"]])
            bucket["annotated_image"].append([_to_float(a) for a in artifacts[i]["annotated_image"]])
            bucket["mask_cropped_image"].append([_to_float(a) for a in artifacts[i]["mask_cropped_image"]])
            si.setdefault(name, []).append(indices[i])

    @property
    def _dec(self):
        return self.config.diffusion_expert_config

    def build_net(self, dtype: torch.dtype):
        with torch.device("meta"):
            if self.vlm_config.model_instance is None:
                raise ValueError("Model instance should be specified")
            language_model = lazy_instantiate(self.vlm_config.model_instance)
            num_train_timesteps = self.config.rectified_flow_inference_config.num_train_timesteps

            network_config = Cosmos3VFMNetworkConfig(
                vlm_config=language_model.config,
                latent_patch_size=self.config.diffusion_expert_config.patch_spatial,
                latent_downsample_factor=self.config.latent_downsample_factor,
                latent_channel_size=self.config.state_ch,
                max_latent_h=self.config.diffusion_expert_config.max_vae_latent_side_after_patchify,
                max_latent_w=self.config.diffusion_expert_config.max_vae_latent_side_after_patchify,
                max_latent_t=self.config.state_t,
                enable_fps_modulation=self.config.diffusion_expert_config.enable_fps_modulation,
                base_fps=self.config.diffusion_expert_config.base_fps,
                vision_gen=self.config.vision_gen,
                action_gen=self.config.action_gen,
                sound_gen=self.config.sound_gen,
                joint_attn_implementation=self.config.joint_attn_implementation,
                timestep_scale=1.0 / float(num_train_timesteps) * self.config.diffusion_expert_config.timestep_range,
                action_dim=self.config.max_action_dim,
                num_embodiment_domains=self.config.num_embodiment_domains,
                temporal_compression_factor_vision=self.tokenizer_vision_gen.temporal_compression_factor,
                natten_parameter_list=self.config.natten_parameter_list,
                video_temporal_causal=self.config.video_temporal_causal,
                sound_dim=self.config.sound_dim,
                sound_latent_fps=self.config.sound_latent_fps,
            )
            network_config._attn_implementation_internal = "eager"
            # Read by AnomalyVFMNetwork.__init__.
            network_config.inpaint_class_hidden_embed_enabled = bool(
                getattr(self._dec, "inpaint_class_hidden_embed_enabled", False)
            )
            network_config.anomaly_num_classes = int(getattr(self._dec, "anomaly_num_classes", 0) or 0)
            network_config.learnable_text_prompt_enabled = bool(
                getattr(self._dec, "learnable_text_prompt_enabled", False)
            )
            network_config.text_prompt_num_tokens = int(getattr(self._dec, "text_prompt_num_tokens", 512) or 512)
            network_config.vision_prompt_item_enabled = bool(getattr(self._dec, "vision_prompt_item_enabled", False))
            network_config.vision_prompt_num_tokens = int(getattr(self._dec, "vision_prompt_num_tokens", 256) or 256)

            net = AnomalyVFMNetwork(language_model=language_model, config=network_config)
            net.pad_for_cuda_graphs = self.config.compile.use_cuda_graphs

            # Per-defect-type LoRA and the shared framework LoRA adapt the same gen-attention
            # projections, so they are mutually exclusive; per-defect-type takes priority when enabled.
            if bool(getattr(self._dec, "per_class_lora_enabled", False)):
                if getattr(self.config, "lora_enabled", False):
                    raise ValueError(
                        "per_class_lora_enabled and lora_enabled are both True: they target the same "
                        "gen-attention projections and cannot both be injected. Set lora_enabled=False "
                        "when using per-defect-type LoRA."
                    )
                targets = per_class_lora_target_modules()
                replaced = inject_per_class_lora(
                    net,
                    targets=targets,
                    num_classes=int(getattr(self._dec, "anomaly_num_classes", 0) or 0),
                    rank=int(getattr(self._dec, "per_class_lora_rank", 16)),
                    alpha=int(getattr(self._dec, "per_class_lora_alpha", 16)),
                )
                log.info(
                    f"AnomalyGenTextureMoTModel: injected per-defect-type LoRA into {replaced} modules ({targets})."
                )
            elif getattr(self.config, "lora_enabled", False):
                net = self.add_lora(
                    net,
                    lora_rank=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_target_modules=self.config.lora_target_modules,
                )

        self.install_attention_dispatch(net)

        net = parallelize_vfm_network(
            net,
            parallel_dims=self.parallel_dims,
            compile_config=self.config.compile,
            ac_config=self.config.activation_checkpointing,
        )

        with misc.timer("meta to cuda and broadcast model states"):
            net = net.to(dtype=dtype)
            net.to_empty(device=DEVICE)
            if DEVICE == Device.CUDA:
                net.init_weights(buffer_device=DEVICE)
                if bool(getattr(self._dec, "per_class_lora_enabled", False)):
                    init_per_class_lora_weights(net)  # lora_A ~ kaiming, lora_B = 0 (delta starts at 0)
                elif getattr(self.config, "lora_enabled", False):
                    self._init_lora_weights_post_materialization(net)
                # Zero the learnable text prompt after to_empty (uninitialised storage); the
                # template values are filled lazily on the first forward (embed_tokens loaded).
                if getattr(net, "text_prompt_emb", None) is not None:
                    net._init_text_prompt_post_materialization()
                # Zero the learnable vision prompt after to_empty (uninitialised storage).
                if getattr(net, "vision_prompt_emb", None) is not None:
                    net._init_vision_prompt_post_materialization()

        return net

    def set_up_model(self):
        super().set_up_model()
        # LoRA freezes every non-LoRA param; re-enable the class embedding so it trains too.
        emb = getattr(self.net, "inpaint_class_emb", None)
        if emb is not None:
            for p in emb.parameters():
                p.requires_grad_(True)
            log.info("AnomalyGenTextureMoTModel: re-enabled grad on inpaint_class_emb after LoRA freeze.")

        # Learnable text prompt: re-enable grad (unless frozen) and hand the per-defect-type template
        # token ids to the network for the one-shot lazy init on the first forward.
        prompt = getattr(self.net, "text_prompt_emb", None)
        if prompt is not None:
            freeze = bool(getattr(self._dec, "text_prompt_freeze", False))
            prompt.requires_grad_(not freeze)
            self.net.set_text_prompt_init_ids(self._build_text_prompt_init_ids())
            log.info(
                f"AnomalyGenTextureMoTModel: text_prompt_emb {tuple(prompt.shape)} "
                f"(trainable={not freeze}); template init ids staged."
            )

        # Learnable vision prompt: re-enable grad (unless frozen) after the LoRA freeze.
        vprompt = getattr(self.net, "vision_prompt_emb", None)
        if vprompt is not None:
            vfreeze = bool(getattr(self._dec, "vision_prompt_freeze", False))
            vprompt.requires_grad_(not vfreeze)
            log.info(f"AnomalyGenTextureMoTModel: vision_prompt_emb {tuple(vprompt.shape)} (trainable={not vfreeze}).")

        # Per-defect-type LoRA: our stacked adapters are named lora_A/lora_B, so the base LoRA-freeze
        # in super().set_up_model() leaves them trainable — re-assert grad defensively (as above).
        if bool(getattr(self._dec, "per_class_lora_enabled", False)):
            n_mod = 0
            for m in self.net.modules():
                if isinstance(m, PerClassLoraLinear):
                    m.lora_A.requires_grad_(True)
                    m.lora_B.requires_grad_(True)
                    n_mod += 1
            log.info(f"AnomalyGenTextureMoTModel: re-enabled grad on per-defect-type LoRA in {n_mod} modules.")

            # Exclude the per-batch routing buffers from DDP buffer sync. broadcast_buffers
            # (framework default) would broadcast gen_class_ids from rank 0 each forward, but its
            # length differs per rank, so the collective mismatches. DDP reads this attribute on the
            # module it wraps (this model) at construction time in trainer.train(), after set_up_model.
            ignore = list(getattr(self, "_ddp_params_and_buffers_to_ignore", []))
            ignore += per_class_lora_buffer_names(self)  # names relative to the model (the DDP module)
            self._ddp_params_and_buffers_to_ignore = ignore
            log.info(
                f"AnomalyGenTextureMoTModel: excluded {len(ignore)} per-defect-type routing buffers "
                "from DDP buffer sync (_ddp_params_and_buffers_to_ignore)."
            )

    def _build_text_prompt_init_ids(self) -> Optional[torch.Tensor]:
        """Build the per-defect-type ``[num_classes, P]`` template token ids for the prompt init.

        Mirrors the dataset exactly: each class's caption is tokenized and padded/truncated to
        ``text_prompt_num_tokens`` with the ``text_prompt_init_word`` token, so the template init
        equals the base model's embedding of the captions the dataset actually feeds.
        """
        anomaly_types = list(getattr(self._dec, "anomaly_types", []) or [])
        if not anomaly_types:
            log.warning("AnomalyGenTextureMoTModel: learnable text prompt enabled but anomaly_types is empty.")
            return None
        p = int(getattr(self._dec, "text_prompt_num_tokens", 512) or 512)
        pad_id = resolve_word_token_id(self.vlm_tokenizer, str(getattr(self._dec, "text_prompt_init_word", "anomaly")))
        rows: List[torch.Tensor] = []
        for texture, defect in anomaly_types:
            caption = build_caption(defect=defect, texture=texture)
            ids = tokenize_caption(caption, self.vlm_tokenizer, is_video=False, use_system_prompt=False)
            ids = pad_or_truncate(ids, p, pad_id)
            rows.append(torch.tensor(ids, dtype=torch.long))
        return torch.stack(rows)  # [num_classes, P]

    def _get_inference_text_tokens(self, data_batch: dict, has_negative_prompt: bool):
        """Pad the conditional caption tokens to ``P`` so inference matches training.

        Training emits exactly ``P`` caption tokens per sample; the framework's inference
        tokenizer does not pad, so without this the conditional text block would be shorter than
        the learnable prompt and the per-defect-type scatter would not fire. The unconditional (empty)
        branch is left short on purpose — its block length stays below ``P`` and is skipped by the
        prompt gate, keeping the trained prompt out of the CFG unconditional pass.
        """
        cond_tokens, uncond_tokens = super()._get_inference_text_tokens(data_batch, has_negative_prompt)
        if getattr(self._dec, "learnable_text_prompt_enabled", False):
            p = int(getattr(self._dec, "text_prompt_num_tokens", 512) or 512)
            pad_id = resolve_word_token_id(
                self.vlm_tokenizer, str(getattr(self._dec, "text_prompt_init_word", "anomaly"))
            )
            cond_tokens = [pad_or_truncate(ids, p, pad_id) for ids in cond_tokens]
        return cond_tokens, uncond_tokens

    def get_data_and_condition(
        self,
        data_batch: Dict[str, torch.Tensor],
        iteration: int = 1,
        vision_condition_indexes: Optional[List[List[int]]] = None,
        retain_raw_state_vision: bool = True,
    ):
        # ``retain_raw_state_vision`` is accepted only so the base's keyword call sites bind.
        # It gates the per-camera multiview VAE path, which anomalygen never takes.
        gen_data_clean = super().get_data_and_condition(
            data_batch,
            iteration=iteration,
            vision_condition_indexes=vision_condition_indexes,
            retain_raw_state_vision=retain_raw_state_vision,
        )

        x0 = gen_data_clean.x0_tokens_vision  # flat list of [1, C, T_lat, H_lat, W_lat]
        num_vis = gen_data_clean.num_vision_items_per_sample
        class_ids = _harvest_class_ids(data_batch, gen_data_clean.batch_size)

        # Dataloader always emits [source, target]: 2 vision items per sample.
        if not (num_vis and set(num_vis) == {2}):
            raise ValueError(f"expected 2 vision items per sample, got num_vis={num_vis}")

        # Build a per-vision-item list of latent masks aligned 1:1 with x0_tokens_vision, computed
        # over the original [source, target] items (before any prompt-item injection).
        latent_masks: Optional[List[Optional[torch.Tensor]]] = None
        pixel_masks = _per_item_pixel_masks(data_batch.get("edit_mask"), num_vis, len(x0))
        if pixel_masks is not None:
            mode = getattr(self._dec, "inpaint_latent_mask_mode", "any")
            latent_masks = []
            for mask, latent in zip(pixel_masks, x0):
                if mask is None:
                    latent_masks.append(None)
                    continue

                _, _, t_lat, h_lat, w_lat = latent.shape
                latent_masks.append(downsample_edit_mask(mask, t_lat, h_lat, w_lat, mode, latent.dtype, latent.device))

        # Optionally prepend a per-defect-type learnable vision prompt item -> [prompt, source, target].
        # The framework auto-marks all-but-last vision items as clean, so prompt + source stay
        # conditioning (no noise, no loss) and only the target is supervised.
        if self._vision_prompt_enabled():
            latent_masks, num_vis = self._inject_vision_prompt_items(gen_data_clean, latent_masks, num_vis)

        items_per_sample = num_vis[0]
        self._anomaly_edit_masks = latent_masks
        self._anomaly_items_per_sample = items_per_sample

        # Pass to the network for vision encoding (read live each forward; needs compile off).
        net = getattr(self, "net", None)
        if isinstance(net, AnomalyVFMNetwork):
            net.set_anomaly_context(latent_masks, class_ids, items_per_sample)

        return gen_data_clean

    def _vision_prompt_enabled(self) -> bool:
        """True when the learnable vision prompt item is configured and its parameter exists."""
        if not bool(getattr(self._dec, "vision_prompt_item_enabled", False)):
            return False
        net = getattr(self, "net", None)
        return getattr(net, "vision_prompt_emb", None) is not None

    def _vision_prompt_side(self, k: int) -> int:
        """Side length of the square ``(side, side)`` token grid holding ``K`` vision-prompt tokens.

        The K learnable tokens are laid out on a square 2-D grid, so ``K`` must be a perfect square.
        """
        side = int(math.isqrt(k))
        if side * side != k:
            raise ValueError(f"vision_prompt_num_tokens={k} must be a perfect square (e.g. 256 = 16x16).")
        return side

    def _inject_vision_prompt_items(self, gen_data_clean, latent_masks, num_vis):
        """Prepend a synthetic clean vision item to each sample, in-place on ``gen_data_clean``.

        Keeps the three parallel per-item lists (``x0_tokens_vision``, ``raw_state_vision``,
        ``temporal_positions_vision``) and ``num_vision_items_per_sample`` mutually consistent, and
        prepends a ``None`` mask for the prompt item. The prompt latent has shape
        ``(1, C, 1, side*patch, side*patch)`` -> a ``(1, side, side)`` square token grid of
        ``K = side*side`` tokens; its contents are overwritten in the network, so the placeholder
        values are irrelevant. Returns the updated ``(latent_masks, num_vis)``.
        """
        k = int(getattr(self._dec, "vision_prompt_num_tokens", 256) or 256)
        patch = int(self._dec.patch_spatial)
        hw_pix = self._vision_prompt_side(k) * patch
        x0 = gen_data_clean.x0_tokens_vision
        raw = gen_data_clean.raw_state_vision
        tpos = gen_data_clean.temporal_positions_vision

        new_x0: List[torch.Tensor] = []
        new_raw: List[torch.Tensor] = []
        new_tpos: List[torch.Tensor] = []
        new_masks: List[Optional[torch.Tensor]] = []
        new_num_vis: List[int] = []

        idx = 0
        for n in num_vis:
            src = x0[idx]  # first (source) item of this sample
            c_lat = int(src.shape[1])
            new_x0.append(torch.zeros((1, c_lat, 1, hw_pix, hw_pix), dtype=src.dtype, device=src.device))
            new_x0.extend(x0[idx : idx + n])

            if raw is not None:
                rsrc = raw[idx]
                new_raw.append(
                    torch.zeros((1, int(rsrc.shape[1]), 1, hw_pix, hw_pix), dtype=rsrc.dtype, device=rsrc.device)
                )
                new_raw.extend(raw[idx : idx + n])

            if tpos is not None:
                tsrc = tpos[idx]
                new_tpos.append(torch.zeros(1, dtype=tsrc.dtype, device=tsrc.device))
                new_tpos.extend(tpos[idx : idx + n])

            if latent_masks is not None:
                new_masks.append(None)
                new_masks.extend(latent_masks[idx : idx + n])

            new_num_vis.append(n + 1)
            idx += n

        gen_data_clean.x0_tokens_vision = new_x0
        if raw is not None:
            gen_data_clean.raw_state_vision = new_raw
        if tpos is not None:
            gen_data_clean.temporal_positions_vision = new_tpos
        gen_data_clean.num_vision_items_per_sample = new_num_vis

        return (new_masks if latent_masks is not None else None), new_num_vis

    def _extract_condition_images_for_visualization(self, gen_data_clean, sequence_plans, n_samples):
        """Pick the source (condition) image as the second-to-last vision item per sample.

        The base method assumes item 0 is the source, but the prompt item shifts the source to
        ``num_items - 2`` (target stays last at ``num_items - 1``). Reduces to the base behavior
        when no prompt item is present (source at index 0 for the 2-item case).
        """
        num_items = gen_data_clean.num_vision_items_per_sample
        if num_items is None or gen_data_clean.raw_state_vision is None:
            return super()._extract_condition_images_for_visualization(gen_data_clean, sequence_plans, n_samples)

        condition_images: List[Optional[torch.Tensor]] = []
        vision_offset = 0
        for i in range(n_samples):
            n = int(num_items[i])
            if n >= 2:
                cond_frame = gen_data_clean.raw_state_vision[vision_offset + n - 2]  # source
                target_frame = gen_data_clean.raw_state_vision[vision_offset + n - 1]  # target
                if cond_frame.shape[-2:] != target_frame.shape[-2:]:
                    cond_frame = F.interpolate(
                        cond_frame.squeeze(2), size=target_frame.shape[-2:], mode="bilinear", align_corners=False
                    ).unsqueeze(2)
                condition_images.append(cond_frame)
            else:
                condition_images.append(None)
            vision_offset += n
        return condition_images

    def _compute_flow_matching_loss(self, pred, target, condition_mask, timesteps, *args, **kwargs):
        """Vision flow-matching loss restricted to the defect region.

        The mask-emphasis term REPLACES the base whole-image loss rather than being added to it, so
        the background contributes no gradient. Read loss curves with that in mind: values are not
        comparable with runs from before this became a replacement, and a falling loss says nothing
        about background reconstruction because the background is not in the objective.

        ``per_instance`` is the base value, passed through untouched — telemetry, not the objective.
        When the term is unavailable (emphasis off, non-vision modality, or no supervised item
        qualified) this falls through to the base loss, so those paths keep training normally.

        KNOWN LIMITATION — ``inpaint_loss_mask_max_adaptive_weight`` (100.0) was calibrated when the
        term was additive and the base loss carried a consistent gradient scale. As the sole
        objective the clamp is load-bearing: ``mask_emphasis_term`` is
        ``clamp(numel / mask_area, 1, max_w) * mean(sqerr * em * nm)``, which equals the in-mask MSE
        only while ``numel / mask_area <= max_w``; past that it scales with mask area instead. At
        512 crops (64x64 latent, numel 4096) it binds below ~41 latent cells, which a thin defect
        reaches at high zoom because ``RandomRatioCrop`` sizes the crop from the *bounding box* — a
        3x200px crack at ratio 8.0 trains at ~0.2x the gradient it should. Compact defects never
        trip it. Left as-is deliberately: every validated result was produced under this clamp.
        Revisit with the planned loss rewrite, not piecemeal.
        """
        loss, per_instance = super()._compute_flow_matching_loss(
            pred, target, condition_mask, timesteps, *args, **kwargs
        )
        if (
            getattr(self._dec, "inpaint_loss_mask_emphasis", False)
            and self._anomaly_edit_masks is not None
            and len(pred) > 0
            and pred[0].dim() in (4, 5)  # vision modality only (action/sound are 2-D); 5-D = [1,C,T,H,W]
        ):
            max_w = float(getattr(self._dec, "inpaint_loss_mask_max_adaptive_weight", 100.0))
            # `pred`/`target`/`condition_mask` are one-per-vision-item (see unpatchify_and_unpack_latents,
            # which appends an entry for every token_shape), and `_anomaly_edit_masks` is the aligned
            # per-item mask list (None at the prompt item, the sample's mask at source/target). Clean
            # items are zeroed out inside mask_emphasis_term via `1 - condition_mask`, so passing the
            # full per-item list is correct for both the 2-item and 3-item (prompt) layouts.
            term = mask_emphasis_term(
                pred=pred,
                target=target,
                condition_mask=condition_mask,
                edit_masks=self._anomaly_edit_masks,
                max_adaptive_weight=max_w,
            )
            if term is not None:
                # Defect-only loss: the mask term REPLACES the whole-image one, so the background
                # contributes no loss. Falls through to the base loss when the term is unavailable.
                loss = term.to(loss.dtype)
        return loss, per_instance

    def _get_velocity(
        self,
        *,
        net=None,
        noise_x,
        timestep,
        text_tokens,
        sequence_plans,
        gen_data_clean,
        skip_text_tokens=False,
        **kwargs,
    ):
        # `**kwargs` passes through sampler arguments this override doesn't use (has_noisy_actions /
        # packed_sequence_template / memory); the base always calls by keyword.
        vel = super()._get_velocity(
            net=net,
            noise_x=noise_x,
            timestep=timestep,
            text_tokens=text_tokens,
            sequence_plans=sequence_plans,
            gen_data_clean=gen_data_clean,
            skip_text_tokens=skip_text_tokens,
            **kwargs,
        )
        if not getattr(self._dec, "inpaint_replacement_trick", False):
            return vel

        masks = self._anomaly_edit_masks
        guides = gen_data_clean.x0_tokens_vision
        if masks is None or guides is None:
            return vel

        ips = int(self._anomaly_items_per_sample)
        if not (len(guides) == len(masks) == ips * len(noise_x)):
            raise ValueError(f"expected {ips} vision items per sample, got {len(guides)} for {len(noise_x)} samples")

        num_train_ts = float(self.config.rectified_flow_inference_config.num_train_timesteps)
        sigma = float(timestep.flatten()[0].item()) / num_train_ts
        if sigma <= 1e-6:
            return vel

        out: List[torch.Tensor] = []
        for i in range(len(noise_x)):
            # Flat vision items are [(prompt,) source, target] per sample; the target is the last
            # item and carries the defect mask. All preceding (clean) items sit ahead of it in the
            # per-sample flat latent, so the target starts at the sum of their element counts.
            base = ips * i
            items = guides[base : base + ips]
            target = items[-1]
            target_mask = masks[base + ips - 1]
            if target_mask is None:
                out.append(vel[i])
                continue

            offset = int(sum(int(it.numel()) for it in items[:-1]))
            out.append(
                replacement_velocity(
                    flat_velocity=vel[i],
                    flat_latent=noise_x[i],
                    guide_latent=target.to(**self.tensor_kwargs),
                    edit_mask_latent=target_mask.to(**self.tensor_kwargs),
                    offset=offset,
                    sigma=sigma,
                )
            )
        return out


def _harvest_class_ids(data_batch: Dict, batch_size: int) -> List[int]:
    """Pull per-sample anomaly class ids from the batch; all-zeros when absent."""
    raw = data_batch.get("anomaly_class_id")
    if raw is None:
        return [0] * int(batch_size)
    if isinstance(raw, torch.Tensor):
        return [int(x) for x in raw.flatten().tolist()]
    if isinstance(raw, (list, tuple)):
        out: List[int] = []
        for x in raw:
            out.append(int(x.flatten()[0].item()) if isinstance(x, torch.Tensor) else int(x))
        return out
    return [int(raw)] * int(batch_size)


def _squeeze_mask(m: torch.Tensor) -> torch.Tensor:
    """Reduce a mask to ``[1, H, W]`` from ``[1,1,H,W]`` / ``[1,H,W]`` / ``[H,W]``."""
    if m.dim() == 4:
        return m[0]
    if m.dim() == 2:
        return m.unsqueeze(0)
    return m


def _per_item_pixel_masks(edit, num_vis: Optional[List[int]], x0_len: int) -> Optional[List[Optional[torch.Tensor]]]:
    """Expand edit masks into a flat per-vision-item list of length ``x0_len`` (or ``None``)."""
    if edit is None:
        return None

    per_sample: List[torch.Tensor] = []
    if isinstance(edit, (list, tuple)):
        per_sample = [_squeeze_mask(m) for m in edit if isinstance(m, torch.Tensor)]
    elif isinstance(edit, torch.Tensor):
        if edit.dim() == 4:
            per_sample = [edit[b] for b in range(edit.shape[0])]
        else:
            per_sample = [_squeeze_mask(edit)]
    if not per_sample:
        return None

    if num_vis:
        flat: List[Optional[torch.Tensor]] = []
        for b, n in enumerate(num_vis):
            mb = per_sample[b] if b < len(per_sample) else None
            flat.extend([mb] * int(n))
    else:
        flat = list(per_sample)

    if len(flat) != x0_len:
        if len(per_sample) == 1:
            flat = [per_sample[0]] * x0_len  # broadcast single mask to all items
        else:
            return None
    return flat
