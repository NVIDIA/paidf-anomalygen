# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Cosmos3VFMNetwork`` extended with a post-VAE per-defect-type additive embedding.

A zero-initialised ``nn.Embedding`` whose per-defect-type hidden vector is added to vision
tokens at masked latent patches. Latent masks and class ids are supplied per-forward via
:meth:`set_anomaly_context` (read live, so requires ``compile.enabled=False``).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import (
    Cosmos3VFMNetwork,
    Cosmos3VFMNetworkConfig,
)
from torch import nn

from anomalygen.models.texture.per_class_lora import build_gen_class_ids, has_per_class_lora, set_gen_class_ids


def _patch_any_touch_mask(latent_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Max-pool a latent mask ``[1, T_lat, H_lat, W_lat]`` by ``patch_size`` to a flat ``[t*h*w]`` bool tensor."""
    lm = latent_mask.float().squeeze(0).unsqueeze(1)  # [T_lat, 1, H_lat, W_lat]
    patch_mask = F.max_pool2d(lm, kernel_size=patch_size, stride=patch_size) > 0.5  # [T_lat,1,h,w]
    return patch_mask.reshape(-1)  # [t*h*w]


def class_embed_addition(
    token_shapes: Sequence[Tuple[int, int, int]],
    latent_masks: Sequence[Optional[torch.Tensor]],
    class_ids: Sequence[int],
    items_per_sample: int,
    patch_size: int,
    emb_lookup: Callable[[int], torch.Tensor],
    num_classes: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build the additive per-defect-type hidden vector for masked vision patches.

    Every sample contributes the same number of vision items (``items_per_sample``), so a flat
    vision item's owning sample is ``item_idx // items_per_sample``. The caller
    (:meth:`AnomalyGenTextureMoTModel.get_data_and_condition`) asserts that uniformity holds.
    Returns a tensor aligned 1:1 with the projected vision tokens (``gate * class_vec`` at masked
    patch rows), or ``None`` when there is nothing to add.
    """
    n_items = len(token_shapes)
    if n_items == 0 or len(latent_masks) != n_items or items_per_sample <= 0:
        return None

    total_patches = sum(int(t) * int(h) * int(w) for (t, h, w) in token_shapes)
    addition: Optional[torch.Tensor] = None  # lazily allocated [total_patches, hidden]
    row_offset = 0  # start row of the current item in the flattened patch buffer
    for item_idx, (shape, latent_mask) in enumerate(zip(token_shapes, latent_masks)):
        t, h, w = int(shape[0]), int(shape[1]), int(shape[2])
        n = t * h * w  # patch tokens for this item

        # Recover the owning sample to look up its anomaly class.
        sample_i = item_idx // items_per_sample
        ci = int(class_ids[sample_i])
        if 0 <= ci < num_classes and latent_mask is not None:
            patch_mask = _patch_any_touch_mask(latent_mask.to(device), patch_size)  # [t*h*w] bool
            if patch_mask.numel() == n and bool(patch_mask.any()):
                class_vec = emb_lookup(ci).to(dtype=dtype, device=device)  # [hidden]
                if addition is None:
                    addition = torch.zeros(total_patches, class_vec.shape[-1], dtype=dtype, device=device)
                # Add the class vector only at masked patch rows.
                gate = patch_mask[:, None].to(dtype=dtype)  # [n,1]
                addition[row_offset : row_offset + n] = gate * class_vec
        row_offset += n
    return addition


class AnomalyVFMNetwork(Cosmos3VFMNetwork):
    def __init__(self, language_model, config: Cosmos3VFMNetworkConfig):
        super().__init__(language_model, config)

        self.inpaint_class_emb: Optional[nn.Embedding] = None
        num_classes = int(getattr(config, "anomaly_num_classes", 0) or 0)
        if getattr(config, "inpaint_class_hidden_embed_enabled", False) and num_classes > 0:
            # Zero-initialised so the additive term is 0 until gradients flow.
            self.inpaint_class_emb = nn.Embedding(num_classes, self.hidden_size)
            with torch.no_grad():
                self.inpaint_class_emb.weight.zero_()

        # Per-defect-type learnable text prompt: [num_classes, P, hidden]. The whole text-token
        # sequence (P positions) is learnable per defect type, initialized from the caption
        # template (see _maybe_init_text_prompt). Materialized empty here (meta device);
        # _init_text_prompt_post_materialization zeros it so all-zero == "not yet template-inited".
        self.text_prompt_emb: Optional[nn.Parameter] = None
        self.text_prompt_num_tokens: int = int(getattr(config, "text_prompt_num_tokens", 512) or 512)
        if getattr(config, "learnable_text_prompt_enabled", False) and num_classes > 0:
            self.text_prompt_emb = nn.Parameter(torch.zeros(num_classes, self.text_prompt_num_tokens, self.hidden_size))
        # Per-defect-type [num_classes, P] init token ids, stashed by the model (which owns the
        # tokenizer + anomaly types); consumed once for the lazy template init.
        self._text_prompt_init_ids: Optional[torch.Tensor] = None
        self._text_prompt_init_done: bool = False

        # Per-defect-type learnable vision prompt: [num_classes, K, hidden]. The model prepends a synthetic
        # clean vision item (a square (1, side, side) grid, K = side*side) before the source; here we
        # overwrite that item's projected vision tokens with the per-defect-type learnable set (see
        # _apply_vision_prompt), so the K tokens live in hidden space and bypass the VAE projection.
        self.vision_prompt_emb: Optional[nn.Parameter] = None
        self.vision_prompt_num_tokens: int = int(getattr(config, "vision_prompt_num_tokens", 256) or 256)
        if getattr(config, "vision_prompt_item_enabled", False) and num_classes > 0:
            self.vision_prompt_emb = nn.Parameter(
                torch.zeros(num_classes, self.vision_prompt_num_tokens, self.hidden_size)
            )

        # Transient per-forward context.
        self._anomaly_latent_masks: Optional[Sequence[Optional[torch.Tensor]]] = None
        self._anomaly_class_ids: Optional[Sequence[int]] = None
        self._anomaly_items_per_sample: int = 1

    def set_anomaly_context(
        self,
        latent_masks: Optional[Sequence[Optional[torch.Tensor]]],
        class_ids: Optional[Sequence[int]],
        items_per_sample: int,
    ) -> None:
        """Stash per-forward anomaly context."""
        self._anomaly_latent_masks = latent_masks
        self._anomaly_class_ids = class_ids
        self._anomaly_items_per_sample = int(items_per_sample)

    # ------------------------ Learnable text prompt ------------------------
    def set_text_prompt_init_ids(self, init_ids: Optional[torch.Tensor]) -> None:
        """Stash the per-defect-type [num_classes, P] template token ids for the lazy prompt init.

        Called by the model (which owns the tokenizer and anomaly types). The actual embedding
        lookup is deferred to the first forward, where ``embed_tokens`` is guaranteed loaded.
        """
        self._text_prompt_init_ids = init_ids

    def _init_text_prompt_post_materialization(self) -> None:
        """Zero ``text_prompt_emb`` after ``to_empty`` (which leaves uninitialised storage).

        All-zero is the sentinel for "not yet template-initialised": on a fresh run the base
        checkpoint skips this param so it stays zero and the lazy template init fires; on resume
        the trained (non-zero) values are loaded and the lazy init is skipped.
        """
        if self.text_prompt_emb is not None:
            with torch.no_grad():
                self.text_prompt_emb.zero_()

    def _init_vision_prompt_post_materialization(self) -> None:
        """Zero ``vision_prompt_emb`` after ``to_empty`` (which leaves uninitialised storage).

        Zero init is deterministic across ranks (no broadcast needed) and trainable: unlike
        ``inpaint_class_emb`` it is not a no-op (the K tokens are always attended to), but their
        zero embeddings project to near-zero keys/values, so the item starts as neutral context
        and gradients flow into it from the first step. A resume checkpoint overwrites these zeros.
        """
        if self.vision_prompt_emb is not None:
            with torch.no_grad():
                self.vision_prompt_emb.zero_()

    def _apply_vision_prompt(self, packed_seq, packed_sequence: torch.Tensor) -> None:
        """Overwrite each sample's prompt vision item (item 0) with its class's learnable tokens.

        The model prepends the prompt item, so within each sample's ``items_per_sample`` vision
        items it sits at local position 0. ``vision.sequence_indexes`` is aligned 1:1 with the
        concatenation of all items' tokens in flat order, so we walk ``token_shapes`` accumulating
        a row offset and scatter ``vision_prompt_emb[class_id]`` into the prompt item's rows.
        """
        if self.vision_prompt_emb is None or self._anomaly_class_ids is None:
            return
        vision = getattr(packed_seq, "vision", None)
        if vision is None or vision.tokens is None or vision.token_shapes is None:
            return
        seq_idx = vision.sequence_indexes
        if not isinstance(seq_idx, torch.Tensor) or seq_idx.numel() == 0:
            return

        items_per_sample = int(self._anomaly_items_per_sample)
        class_ids = list(self._anomaly_class_ids)
        k = self.vision_prompt_num_tokens
        num_classes = int(self.vision_prompt_emb.shape[0])
        row_offset = 0
        for item_idx, shape in enumerate(vision.token_shapes):
            n = int(shape[0]) * int(shape[1]) * int(shape[2])
            if items_per_sample > 0 and item_idx % items_per_sample == 0:  # prompt item
                sample_i = item_idx // items_per_sample
                ci = int(class_ids[sample_i]) if sample_i < len(class_ids) else -1
                if 0 <= ci < num_classes and n == k:
                    rows = seq_idx[row_offset : row_offset + n]
                    packed_sequence[rows] = self.vision_prompt_emb[ci].to(
                        dtype=packed_sequence.dtype, device=packed_sequence.device
                    )
            row_offset += n

    def _maybe_init_text_prompt(self) -> None:
        """One-shot template init of ``text_prompt_emb`` from the stashed token ids.

        Runs on the first forward (so ``embed_tokens`` weights are loaded). Skips when the param
        was already populated from a resume checkpoint (non-zero) or no init ids were provided.
        """
        if self._text_prompt_init_done or self.text_prompt_emb is None:
            return
        self._text_prompt_init_done = True
        init_ids = self._text_prompt_init_ids
        self._text_prompt_init_ids = None
        if init_ids is None:
            return
        # Resume checkpoint already filled the prompt -> keep it, don't overwrite with the template.
        if torch.count_nonzero(self.text_prompt_emb) != 0:
            return
        with torch.no_grad():
            ids = init_ids.to(device=self.text_prompt_emb.device)
            emb = self.language_model.model.embed_tokens(ids)  # [num_classes, P, hidden]
            self.text_prompt_emb.copy_(emb.to(self.text_prompt_emb.dtype))

    def _apply_text_prompt(self, packed_seq, packed_sequence: torch.Tensor) -> None:
        """Overwrite each sample's first-P text positions with its class's learnable prompt.

        The packed text block for a sample is ``[caption(P), EOS, start_of_generation]`` (no BOS
        for this tokenizer), contiguous and in sample order, so reshaping ``text_indexes`` to
        ``[n_samples, block_len]`` and scattering ``text_prompt_emb[class_id]`` into the leading P
        rows is exact. The block-length gate also skips the CFG unconditional pass (empty caption
        -> block far shorter than P).
        """
        if self.text_prompt_emb is None or self._anomaly_class_ids is None:
            return
        idx = getattr(packed_seq, "text_indexes", None)
        if not isinstance(idx, torch.Tensor) or idx.numel() == 0:
            return

        class_ids = list(self._anomaly_class_ids)
        n_samples = len(class_ids)
        p = self.text_prompt_num_tokens
        total = int(idx.numel())
        if n_samples <= 0 or total % n_samples != 0:
            return
        block_len = total // n_samples
        if block_len < p:  # unconditional / non-padded captions: nothing to inject
            return

        idx2 = idx.view(n_samples, block_len)
        num_classes = int(self.text_prompt_emb.shape[0])
        for i, ci in enumerate(class_ids):
            if 0 <= int(ci) < num_classes:
                rows = idx2[i, :p]
                packed_sequence[rows] = self.text_prompt_emb[int(ci)].to(
                    dtype=packed_sequence.dtype, device=packed_sequence.device
                )

    def _encode_text(self, packed_seq):
        packed_sequence, target_dtype = super()._encode_text(packed_seq)
        self._maybe_init_text_prompt()
        self._apply_text_prompt(packed_seq, packed_sequence)
        return packed_sequence, target_dtype

    def _encode_vision(
        self,
        packed_seq,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> List[Tuple[int, int, int]] | None:
        original_latent_shapes = super()._encode_vision(packed_seq, packed_sequence, target_dtype)

        if (
            self.inpaint_class_emb is not None
            and self._anomaly_latent_masks is not None
            and self._anomaly_class_ids is not None
            and packed_seq.vision is not None
            and packed_seq.vision.tokens is not None
        ):
            vision = packed_seq.vision

            addition = class_embed_addition(
                token_shapes=vision.token_shapes,
                latent_masks=self._anomaly_latent_masks,
                class_ids=list(self._anomaly_class_ids),
                items_per_sample=self._anomaly_items_per_sample,
                patch_size=self.latent_patch_size,
                emb_lookup=lambda ci: self.inpaint_class_emb.weight[ci],
                num_classes=int(self.inpaint_class_emb.num_embeddings),
                dtype=packed_sequence.dtype,
                device=packed_sequence.device,
            )
            if addition is not None:
                idx = vision.sequence_indexes
                packed_sequence[idx] = packed_sequence[idx] + addition

        # Applied last so the learnable per-defect-type tokens are the final content of the prompt item,
        # regardless of what the VAE projection / class-emb addition produced for those rows.
        self._apply_vision_prompt(packed_seq, packed_sequence)

        # Route each gen token to its class's LoRA adapter. Runs here (eager encode head), before the
        # compiled transformer blocks, so the class ids reach them as a tensor buffer — never a Python
        # attribute, which fullgraph=True would bake as a stale constant inside the compiled region.
        self._set_per_class_routing(packed_seq, packed_sequence)

        return original_latent_shapes

    def _set_per_class_routing(self, packed_seq, packed_sequence: torch.Tensor) -> None:
        """Refresh every :class:`PerClassLoraLinear`'s ``gen_class_ids`` buffer for this forward.

        Builds a per-gen-token class-id vector in ``full_only_seq`` order (identical at every gen
        layer) when the class ids and vision tokens are available; otherwise clears the buffer so the
        adapters fall back to the frozen base rather than reuse the previous batch's routing.
        """
        if not has_per_class_lora(self):
            return
        vision = getattr(packed_seq, "vision", None)
        gen_class_ids = None
        if self._anomaly_class_ids is not None and vision is not None and vision.token_shapes is not None:
            gen_class_ids = build_gen_class_ids(
                packed_sequence=packed_sequence,
                token_shapes=vision.token_shapes,
                sequence_indexes=vision.sequence_indexes,
                class_ids=list(self._anomaly_class_ids),
                items_per_sample=int(self._anomaly_items_per_sample),
            )
        set_gen_class_ids(self, gen_class_ids)  # None -> clears the buffer (frozen base this forward)
