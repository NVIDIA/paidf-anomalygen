# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Multi-view Anomaly Generation Pipeline.

This pipeline extends AnomalyGenPipeline to handle multiple views as video frames,
with per-view anomaly embeddings and per-view cross-attention via MultiViewDiT.
"""

from typing import Any, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.transforms import Resize
from PIL import Image

from cosmos_predict2.conditioner import DataType, T2VCondition
from cosmos_predict2.pipelines.anomaly_gen import AnomalyGenPipeline
from imaginaire.lazy_config import LazyDict
from imaginaire.utils import log, misc


IS_PREPROCESSED_KEY = "is_preprocessed"
NEG_PROMPT = "The video captures a series of frames showing a product without any anomalies. " \
            "Flawless, Precise, Pure, Reliable, Durable, Uniform, Aesthetic, Seamless, " \
            "Functional, Intact, High-quality, Polished, Accurate, Consistent, Efficient. " \
            "Overall, the product is of high quality."


class AnomalyGenMultiViewPipeline(AnomalyGenPipeline):
    """
    Multi-view Anomaly Generation Pipeline based on AnomalyGenPipeline.
    
    This pipeline extends single-view anomaly generation to handle multiple views
    as video frames, with per-view anomaly embeddings and batched text inversion.
    
    Key features:
    - Inherits all anomaly-specific logic from AnomalyGenPipeline
    - Per-view anomaly embeddings for better view-specific representation
    - Uses MultiViewDiT (loaded via config.net) for per-view cross-attention
    - Overrides only data processing methods to handle multi-view inputs
    """
    
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        # Call parent AnomalyGenPipeline's __init__, which will initialize all anomaly-specific components
        # and call Text2ImagePipeline's __init__
        super().__init__(device=device, torch_dtype=torch_dtype)
        
        # Multi-view specific data keys
        self.input_data_key = 'video'  # Multi-view uses 'video' key
        self.input_image_key = 'images'  # Fallback for single-view compatibility
        
    def from_anomaly_gen_config(self, ag_config: LazyDict) -> None:
        """Initialize anomaly generation specific components with per-view embeddings."""
        # Precision
        super().from_anomaly_gen_config(ag_config)
        self.perview_perturbation_std  = ag_config.anomaly_embedding.perview_perturbation_std
        # Get number of views from MultiViewDiT (single source of truth)
        # This is set via pipe_config.net.num_views in YAML and passed to MultiViewDiT.__init__
        if hasattr(self.dit, 'num_views'):
            self.num_views = self.dit.num_views
            log.info(f"Multi-View Config: num_views={self.num_views} (from MultiViewDiT)")
        else:
            raise ValueError("MultiViewDiT does not have num_views attribute. Please ensure pipe_config.net.num_views is set in YAML.")
        
        # Replace single anomaly_embedding with per-view embeddings
        log.info(f"Replacing single anomaly embedding with per-view embeddings for {self.num_views} views")
        init_embedding = self._get_initial_embedding()
        self.anomaly_embedding = self._initialize_per_view_anomaly_embeddings(
            ag_config.anomaly_embedding.num_tokens,
            ag_config.anomaly_embedding.anomaly_types,
            self.anomaly_embedding_token_dim,
            init_embedding,
            ag_config.anomaly_embedding.freeze,
            num_views=self.num_views,
        ).to(**self.ad_tensor_kwargs)

    def _get_initial_embedding(self):
        """Get initial embedding for anomaly tokens using text encoder."""
        init_word = self.anomaly_embedding_init_word
        log.info(f"Use T5 Encoder to convert {init_word} as anomaly embedding's initial value.")
        initial_token = self.text_tokenizer.encode(init_word)
        assert len(initial_token) == 2, f"Initial word {init_word} is tokenized into >1 tokens. Please change your init_word"
        initial_token = torch.tensor(initial_token[0]).cuda()  # Remove <EOS> token
        init_embedding = self.text_encoder.text_encoder.shared(initial_token)
        return init_embedding
    
    def _initialize_per_view_anomaly_embeddings(
        self, 
        num_tokens: int, 
        anomaly_types: List[Tuple[str, str]], 
        token_dim: int, 
        init_embedding: torch.Tensor,
        freeze: bool,
        num_views: int,
    ) -> nn.ModuleDict:
        """
        Initialize per-view anomaly embeddings.
        
        Similar to parent's build_anomaly_LUT but creates num_views separate embeddings 
        for each anomaly type with small perturbations.
        
        Returns:
            nn.ModuleDict mapping anomaly_type_key to nn.ParameterList[view_idx]
        """
        log.info(f"Building per-view anomaly LUT: {num_views} views, {num_tokens} tokens per view")
        
        # Initialize per-view LUT
        per_view_embeddings = nn.ModuleDict()
        perturbation_std = self.perview_perturbation_std # Small perturbation for view diversity
        
        for sample_name, anomaly_name in anomaly_types:
            type_key = f"{sample_name}+{anomaly_name}"
            view_embs = nn.ParameterList()
            
            for view_idx in range(num_views):
                # Initialize with init_embedding + small perturbation
                base_emb = init_embedding.unsqueeze(0).repeat(num_tokens, 1)
                if not freeze and view_idx > 0:
                    # Add perturbation for views > 0 to create diversity
                    perturbation = torch.randn_like(base_emb) * perturbation_std
                    view_emb = nn.Parameter(base_emb + perturbation, requires_grad=True)
                else:
                    view_emb = nn.Parameter(base_emb, requires_grad=not freeze)
                
                view_embs.append(view_emb)
            
            per_view_embeddings[type_key] = view_embs
        
        # Freeze / Unfreeze embeddings
        if freeze:
            for type_key in per_view_embeddings.keys():
                for view_emb in per_view_embeddings[type_key]:
                    view_emb.requires_grad = False
            log.info(f"Per-view Anomaly LUT is frozen.")
        
        return per_view_embeddings

    def _get_text_embedding(self, text: List):
        """Get text embedding from text encoder."""
        tokenized_output = self.text_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            max_length=self.text_tokenizer_max_length  # Reduced from 512 to save T5 memory (text ~50 + spatial 57 = ~107)
        )
        input_ids = tokenized_output['input_ids'].cuda()
        attn_mask = tokenized_output['attention_mask'].cuda()
        text_embedding = self.text_encoder.text_encoder.shared(input_ids)
        return input_ids, attn_mask, text_embedding

    def _text_inversion(self, data_batch):
        """
        Per-view text inversion for multi-view anomaly generation.
        
        Strategy: Sequential processing (loop over views) to avoid OOM.
        All masks are in [B, T, H, W] format (unified).
        
        Returns:
            condition: [B, V*512, D] where V=num_views
            attn_mask: [B, V*512]
            neg_condition: [B, V*512, D]
            neg_attn_mask: [B, V*512]
        """
        # Determine batch size and number of views
        B = len(data_batch['name'])
        num_views = self.num_views
        
        # ============ Step 1: Batch Encode Masks (all are [B, T, H, W]) ============
        mask = data_batch['mask']  # [B, T, H, W] or [B, C, T, H, W]
        
        # Handle different input dimensions
        if mask.dim() == 5:
            # [B, C, T, H, W] -> [B, T, H, W]
            mask = mask[:, 0, :, :, :]  # Take first channel
        
        # Split into per-view masks: List of [B, 1, H, W]
        masks_per_view = [mask[:, t:t+1, :, :] for t in range(num_views)]
        
        # Stack and batch encode
        masks_stacked = torch.stack(masks_per_view, dim=0)  # [V, B, 1, H, W]
        V, B_dim, C, H, W = masks_stacked.shape
        masks_flat = masks_stacked.reshape(V * B_dim, C, H, W)  # [V*B, 1, H, W]
        
        # Batch encode all masks at once (V*B masks in one forward pass)
        mask_embeddings_flat = self.mask_encoder(masks_flat.to(self.ad_precision))  # [V*B, M, D]
        
        # Reshape back to per-view format
        _, M, D = mask_embeddings_flat.shape
        mask_embeddings_reshaped = mask_embeddings_flat.reshape(V, B_dim, M, D)  # [V, B, M, D]
        mask_embeddings = [mask_embeddings_reshaped[i] for i in range(V)]  # List of [B, M, D]
        
        num_mask_tokens = M
        
        # ============ Step 2: Get Per-View Anomaly Embeddings ============
        anomaly_embeddings = []
        for view_idx in range(num_views):
            # Collect anomaly embeddings for this view across the batch
            view_anomaly_embs = torch.stack([
                self.anomaly_embedding[anomaly_name][view_idx]  # [num_anomaly_tokens, C]
                for anomaly_name in data_batch['name']
            ])  # [B, num_anomaly_tokens, C]
            anomaly_embeddings.append(view_anomaly_embs)
        
        # ============ Step 3: Concatenate Mask + Anomaly for Each View ============
        # Both mask and anomaly embeddings are PER-VIEW
        spatial_embeddings = []
        for view_idx in range(num_views):
            spatial_emb = torch.cat([
                mask_embeddings[view_idx],      # [B, num_mask_tokens, C] - PER-VIEW
                anomaly_embeddings[view_idx],   # [B, num_anomaly_tokens, C] - PER-VIEW
            ], dim=1)  # [B, total_tokens_per_view, C]
            spatial_embeddings.append(spatial_emb)
        
        # ============ Step 4: Batched Text Inversion (all views at once) ============
        # Stack all views: [V, B, M, C] -> [V*B, M, C]
        spatial_embeddings_stacked = torch.stack(spatial_embeddings, dim=0)  # [V, B, M, C]
        V, B, M, C = spatial_embeddings_stacked.shape
        spatial_embeddings_flat = spatial_embeddings_stacked.reshape(V * B, M, C)  # [V*B, M, C]
        
        # Repeat captions V times
        captions_repeated = data_batch['caption'] * num_views  # List[str] × (V*B)
        
        # Batch process all views
        conditions_flat, attn_masks_flat = self._batch_text_inversion_t5(
            spatial_embeddings_flat,  # [V*B, M, C]
            captions_repeated,         # List[str] × (V*B)
        )
        
        # Reshape back to [B, V*seq_len, D]
        seq_len = conditions_flat.shape[1]  # Should be max_length (317)
        D = conditions_flat.shape[2]
        conditions_reshaped = conditions_flat.reshape(V, B, seq_len, D)  # [V, B, seq_len, D]
        conditions = conditions_reshaped.permute(1, 0, 2, 3).reshape(B, V * seq_len, D)  # [B, V*seq_len, D]
        
        attn_masks_reshaped = attn_masks_flat.reshape(V, B, seq_len)  # [V, B, seq_len]
        attn_masks = attn_masks_reshaped.permute(1, 0, 2).reshape(B, V * seq_len)  # [B, V*seq_len]
        
        # ============ Step 6: Prepare Negative Condition ============
        if self.neg_condition is None:
            neg_input_ids, neg_attn_mask, neg_text_embedding = self._get_text_embedding([self.neg_prompt])
            neg_condition_single = self.text_encoder.text_encoder(
                inputs_embeds=neg_text_embedding,
                attention_mask=neg_attn_mask
            ).last_hidden_state
            neg_condition_single = neg_condition_single.to(torch.bfloat16)
            neg_condition_single[0][neg_attn_mask[0] == 0] = 0
            self.neg_condition = neg_condition_single[0]  # [seq_len, D]
            self.neg_attn_mask = neg_attn_mask[0]         # [seq_len]
        
        # Expand negative condition to match [B, V*seq_len, D]
        neg_condition = self.neg_condition.unsqueeze(0).repeat(B, num_views, 1)  # [B, V*seq_len, D]
        neg_attn_mask = self.neg_attn_mask.unsqueeze(0).repeat(B, num_views)      # [B, V*seq_len]
        
        return conditions, attn_masks, neg_condition, neg_attn_mask
    
    def _batch_text_inversion_t5(
        self,
        spatial_embeddings: torch.Tensor,  # [V*B, M, C]
        captions: List[str],               # List of V*B captions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform text inversion through T5 encoder for multiple views in a batch.
        
        Args:
            spatial_embeddings: [V*B, M, C] where V is num_views
            captions: List of V*B captions (repeated V times)
        
        Returns:
            conditions: [V*B, max_length, D]
            attn_masks: [V*B, max_length]
        """
        VB, M, C = spatial_embeddings.shape
        
        # Tokenize captions
        input_ids, attn_mask, text_embedding = self._get_text_embedding(captions)
        L = input_ids.shape[1]  # Max length (317)
        
        # Check placeholder token
        occurrence = (input_ids == self.placeholder_token_id).sum(dim=1)
        assert (occurrence == 1).all().item(), \
            f"Placeholder token id {self.placeholder_token_id} does not exist exactly once in every prompt!"
        
        # Find placeholder positions
        placeholder_rows, placeholder_cols = torch.where(input_ids == self.placeholder_token_id)
        sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
        sorted_rows = placeholder_rows[sort_idx]
        
        # Insert spatial embeddings
        for idx in range(VB):
            row, col = sorted_rows[idx], sorted_cols[idx]
            num_text_tokens = torch.where(input_ids[row] != 0)[0].max()
            final_length = num_text_tokens + M - 1
            
            # Create new token sequence
            new_token_row = torch.cat([
                input_ids[row][:col],
                torch.tensor(self.placeholder_token_id).repeat(M).to('cuda'),
                input_ids[row][col + 1:]
            ], dim=0)[:L]
            input_ids[row] = new_token_row
            
            # Update attention mask
            attn_mask[row][:final_length] = True
            
            # Insert spatial embeddings into text_embedding
            new_embedding_row = torch.cat([
                text_embedding[row][:col],
                spatial_embeddings[idx],  # [M, C]
                text_embedding[row][col + 1:]
            ], dim=0)[:L]
            text_embedding[row] = new_embedding_row
        
        # Apply adapter to entire text_embedding
        text_embedding = self.adapter(text_embedding)
        
        # Forward through T5 encoder (batched: [V*B, L, C])
        conditions = self.text_encoder.text_encoder(
            inputs_embeds=text_embedding,
            attention_mask=attn_mask
        ).last_hidden_state
        conditions = conditions.to(torch.bfloat16)
        
        # Clear masked regions (same as single-view)
        for idx in range(VB):
            conditions[idx][attn_mask[idx] == 0] = 0
        
        # Convert attn_mask to bfloat16 for consistency
        attn_mask = attn_mask.to(torch.bfloat16)
        
        return conditions, attn_mask

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, T2VCondition]:
        """
        Get data and condition for multi-view training.
        """
        # Multi-view uses 'video' key with shape (B, C, T, H, W)
        if 'video' in data_batch:
            raw_state = data_batch['video']
        else:
            raise ValueError(f"'video' key is not found in data_batch")
        
        # Encode to latent space
        latent_state = self.encode(raw_state).contiguous().float()
        
        # Text inversion
        condition, attn_mask, neg_condition, neg_attn_mask = self._text_inversion(data_batch)
        B = condition.shape[0]
        total_seq_len = condition.shape[1]  # V * 512
        data_batch['t5_text_embeddings'] = condition
        data_batch['t5_text_mask'] = torch.ones(B, total_seq_len, dtype=torch.bfloat16).cuda()
        data_batch['neg_t5_text_embeddings'] = neg_condition
        data_batch['neg_t5_text_mask'] = torch.ones(B, total_seq_len, dtype=torch.bfloat16).cuda()
        
        # Condition - use VIDEO type for multi-view
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE)
        
        return raw_state, latent_state, condition

    def _get_guided_mask_and_weight(self, x0_gt, masks, max_adaptive_mask_weight=100):
        """
        Retain only un-masked regions via masks for multi-view.
        All masks are in [B, T, H, W] format (unified).
        """
        B, C, T, H, W = x0_gt.shape
        
        # Handle mask shape - always [B, T, H, W]
        if masks.dim() == 5:
            # [B, C, T, H, W] -> [B, T, H, W]
            masks = masks[:, 0, :, :, :]  # Take first channel
        elif masks.dim() == 4:
            # [B, T, H, W] - already correct
            pass
        else:
            raise ValueError(f"Unexpected mask shape: {masks.shape}")
        
        # Build per-view guided masks [B, 1, T, H, W]
        guided_masks_per_view = []
        for t in range(T):
            view_masks = masks[:, t:t+1, :, :]  # [B, 1, H, W]
            view_guided = torch.zeros((B, 1, H, W))
            for i in range(B):
                latent_mask = 1 - Resize((H, W), interpolation=Image.Resampling.BICUBIC)(view_masks[i])
                latent_mask[latent_mask <= 0.85] = 0
                latent_mask[latent_mask > 0.85] = 1
                view_guided[i] = 1 - latent_mask
            guided_masks_per_view.append(view_guided)
        
        # Stack: [B, 1, T, H, W]
        guided_masks = torch.stack(guided_masks_per_view, dim=2).contiguous().to("cuda")
        
        # Compute per-view adaptive weights [B, T]
        # Each view should have its own adaptive weight based on its mask
        adaptive_weights_per_view = []
        for t in range(T):
            view_masked_region = guided_masks_per_view[t].sum(dim=(1, 2, 3))  # [B]
            view_adaptive_weight = (H * W) / view_masked_region  # [B]
            view_adaptive_weight = torch.clamp(view_adaptive_weight, min=1.0, max=max_adaptive_mask_weight)
            adaptive_weights_per_view.append(view_adaptive_weight)
        
        # Stack to [B, T]
        adaptive_weight = torch.stack(adaptive_weights_per_view, dim=1).to("cuda")
        
        return guided_masks, adaptive_weight

    @torch.inference_mode()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode multi-view frames.
        
        For multi-view with few frames (< 9), we encode each frame independently
        as images and then combine them into a video tensor.
        
        Args:
            state: Input tensor of shape [B, C, T, H, W]
            
        Returns:
            Latent tensor of shape [B, C_latent, T_latent, H_latent, W_latent]
        """
        B, C, T, H, W = state.shape

        latents = []
        for t in range(T):
            # Extract frame [B, C, H, W] and add temporal dim [B, C, 1, H, W]
            frame = state[:, :, t:t+1, :, :]
            # Encode single frame
            latent = self.tokenizer.encode(frame) * self.sigma_data
            latents.append(latent)
        # Concatenate along temporal dimension [B, C, T, H, W]
        return torch.cat(latents, dim=2)

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode multi-view latents.
        
        For multi-view with few frames, we decode each frame independently.
        """
        B, C, T, H, W = latent.shape
        
        frames = []
        for t in range(T):
            frame_latent = latent[:, :, t:t+1, :, :]
            frame = self.tokenizer.decode(frame_latent / self.sigma_data)
            frames.append(frame)
        return torch.cat(frames, dim=2)

    @torch.no_grad()
    def __call__(self, data_batch: dict, seed: int = 0, guidance: float = 4.0, 
                 num_steps: int = 35, is_negative_prompt: bool = True, n_sample: int = 2,
                 use_cuda_graphs: bool = False):
        """Run inference to generate multi-view images from data_batch.
        
        Args:
            data_batch: Dictionary containing 'images' (or 'video') and 'mask' tensors
            seed: Random seed for generation
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion steps
            is_negative_prompt: Whether to use negative prompts
            n_sample: Number of samples (not used, kept for API compatibility)
            use_cuda_graphs: Whether to use CUDA graphs
            
        Returns:
            Generated images tensor of shape [B, C, T, H, W]
        """        
        # Use 'video' key for multi-view if 'images' not present
        input_key = 'video' if 'video' in data_batch else 'images'
        
        log.info(f"Begin multi-view diffusion inference. Guidance = {guidance}, Num steps = {num_steps}, seed = {seed}.")
        
        # Prepare guided image & mask
        data_batch['guided_image'] = self.encode(data_batch[input_key]).contiguous()
        data_batch['guided_mask'], _ = self._get_guided_mask_and_weight(
            x0_gt=data_batch['guided_image'],
            masks=data_batch['mask']
        )
        data_batch['guided_mask'] = 1 - data_batch['guided_mask']  # 1 for using gt, 0 for using denoised result

        # Multi-view data shape: [B, C, T, H, W] - already has temporal dimension
        if data_batch[input_key].dim() == 4:
            raise ValueError(f"Multi-view data should have 5 dimensions, but got {data_batch[input_key].shape}")
        
        n_sample = data_batch[input_key].shape[0]
        _T, _H, _W = data_batch[input_key].shape[-3:]
        state_shape = [
            self.config.state_ch,
            _T,  # Multi-view: T frames -> T latent frames (since we encode independently)
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]

        # Text inversion
        _, latent_state, _ = self.get_data_and_condition(data_batch)

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        # Use VIDEO data type for multi-view
        condition = condition.edit_data_type(DataType.IMAGE)
        uncondition = uncondition.edit_data_type(DataType.IMAGE)

        # Context parallelism
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(latent_state, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(latent_state, uncondition, None, None)

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            * self.scheduler.config.sigma_max
        )

        # Sampling loop
        scheduler = self.scheduler
        scheduler.set_timesteps(num_steps, device=x_sigma_max.device)
        sample = x_sigma_max.to(dtype=torch.float32)

        x0_prev = None

        for i, _ in enumerate(tqdm(scheduler.timesteps, desc="Generating multi-view", leave=False)):
            sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)
            sigma_in = sigma_t.repeat(sample.shape[0])

            # x0 prediction with CFG
            cond_x0 = self.denoise(sample, sigma_in, condition).x0
            uncond_x0 = self.denoise(sample, sigma_in, uncondition).x0
            x0_pred = cond_x0 + guidance * (cond_x0 - uncond_x0)

            # Apply guidance mask
            x0_pred = self._apply_guided_mask(x0_pred, data_batch)

            # AB2 step - use step index i, not sigma value
            # scheduler.step returns (x_next, x0_t) tuple
            sample, x0_prev = scheduler.step(
                x0_pred=x0_pred,
                i=i,
                sample=sample,
                x0_prev=x0_prev,
            )

        # Final clean pass at sigma_min (consistent with single-view pipeline)
        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])
        
        cond_x0 = self.denoise(sample, sigma_in, condition).x0
        uncond_x0 = self.denoise(sample, sigma_in, uncondition).x0
        samples = cond_x0 + guidance * (cond_x0 - uncond_x0)
        
        # Apply guidance mask one final time
        samples = self._apply_guided_mask(samples, data_batch)

        # Decode to pixel space
        denoised_result = self.decode(samples)
        
        log.info(f"Multi-view inference complete. Output shape: {denoised_result.shape}")
        return denoised_result

    def _apply_guided_mask(self, x0_pred, data_batch):
        """Apply guided mask to blend GT and predicted latents."""
        x0_pred = data_batch['guided_mask'] * data_batch['guided_image'] + (1 - data_batch['guided_mask']) * x0_pred
        return x0_pred

