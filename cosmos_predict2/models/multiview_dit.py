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
Multi-View DiT with Per-View Cross-Attention.
Inherits from MiniTrainDIT but overrides cross-attention behavior.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, Callable

from cosmos_predict2.models.text2image_dit import Block, MiniTrainDIT
from loguru import logger as log


class MultiViewBlock(Block):
    """
    Multi-View Block with per-view cross-attention.
    Inherits from Block but overrides the cross-attention mechanism.
    
    Key difference: Each view attends to its own condition tokens instead of
    all patches attending to the same shared condition.
    """
    
    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        backend: str = "transformer_engine",
        num_views: int = 6,
    ):
        # Initialize parent (includes self_attn, cross_attn, mlp)
        super().__init__(
            x_dim=x_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            backend=backend,
        )
        
        self.num_views = num_views
        log.debug(f"MultiViewBlock: num_views={num_views}")
    
    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with per-view cross-attention.
        
        Args:
            x_B_T_H_W_D: [B, T, H, W, D] where T = num_views
            emb_B_T_D: [B, T, D] timestep embeddings
            crossattn_emb: [B, V×M, D] where V=num_views, M auto-inferred from shape
            rope_emb_L_1_1_D: Optional rope embeddings
            adaln_lora_B_T_3D: Optional AdaLN-LoRA embeddings
            extra_per_block_pos_emb: Optional extra positional embeddings
            
        Returns:
            x_B_T_H_W_D: [B, T, H, W, D]
        """
        B, T, H, W, D = x_B_T_H_W_D.shape
        
        # Verify T matches num_views
        assert T == self.num_views, f"Expected T={self.num_views}, got T={T}"
        
        # Auto-infer tokens_per_view from crossattn_emb shape
        _, total_tokens, _ = crossattn_emb.shape
        tokens_per_view = total_tokens // self.num_views
        
        # Verify crossattn_emb has correct shape
        expected_context_tokens = self.num_views * tokens_per_view
        assert total_tokens == expected_context_tokens, \
            f"Expected context tokens={expected_context_tokens}, got {total_tokens}"
        
        # Add extra positional embeddings if provided
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb
        
        # === AdaLN Modulation ===
        if self.use_adaln_lora:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = \
                self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = \
                self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = \
                self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)
        
        # Reshape for broadcasting
        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")
        
        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        
        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")
        
        # Helper function for layer norm with AdaLN
        def _fn(
            _x_B_T_H_W_D: torch.Tensor,
            layer_norm: Callable,
            _scale_B_T_1_1_D: torch.Tensor,
            _shift_B_T_1_1_D: torch.Tensor,
        ) -> torch.Tensor:
            return layer_norm(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D
        
        # === Self-Attention (same as parent) ===
        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )
        
        result_B_T_H_W_D = rearrange(
            self.self_attn(
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D
        
        # === Per-View Cross-Attention ⭐⭐⭐ ===
        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_1_1_D,
            shift_cross_attn_B_T_1_1_D,
        )
        
        # Reshape query: [B, T, H, W, D] -> [B×T, H×W, D]
        query_BT_HW_D = rearrange(normalized_x_B_T_H_W_D, "b t h w d -> (b t) (h w) d")
        # [B×T, H×W, D] = [B×6, 1024, 2048]
        
        # Reshape context: [B, V×M, D] -> [B×V, M, D]
        context_BV_M_D = rearrange(
            crossattn_emb,
            "b (v m) d -> (b v) m d",
            v=self.num_views,
            m=tokens_per_view,
        )
        # [B×V, M, D] = [B×6, 512, 1024]
        
        # Per-view cross-attention: each view attends to its own condition
        result_BT_HW_D = self.cross_attn(
            query_BT_HW_D,      # [B×6, 1024, 2048]
            context_BV_M_D,     # [B×6, 512, 1024]
            rope_emb=rope_emb_L_1_1_D,
        )
        
        # Reshape back: [B×T, H×W, D] -> [B, T, H, W, D]
        result_B_T_H_W_D = rearrange(
            result_BT_HW_D,
            "(b t) (h w) d -> b t h w d",
            b=B, t=T, h=H, w=W,
        )
        
        x_B_T_H_W_D = x_B_T_H_W_D + gate_cross_attn_B_T_1_1_D * result_B_T_H_W_D
        
        # === MLP (same as parent) ===
        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_mlp,
            scale_mlp_B_T_1_1_D,
            shift_mlp_B_T_1_1_D,
        )
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D
        
        return x_B_T_H_W_D


class MultiViewDiT(MiniTrainDIT):
    """
    Multi-View DiT that uses MultiViewBlock for per-view cross-attention.
    Inherits everything from MiniTrainDIT except the blocks.
    
    Args:
        num_views: Number of camera views
        *args, **kwargs: Same as MiniTrainDIT
    """
    
    def __init__(
        self,
        *args,
        num_views: int,
        **kwargs
    ):
        # Store multi-view config before calling super().__init__
        if not isinstance(num_views, int):
            raise ValueError("num_views must be an integer greater than 0")
        self.num_views = num_views
        
        log.info(f"Initializing MultiViewDiT with num_views={num_views}")
        
        # Call parent init (will create standard blocks)
        super().__init__(*args, **kwargs)
        
        log.info(f"Replacing {len(self.blocks)} blocks with MultiViewBlock")
        # Replace blocks with MultiViewBlocks
        self._replace_blocks_with_multiview()
    
    def _replace_blocks_with_multiview(self):
        """Replace standard Block with MultiViewBlock."""
        new_blocks = nn.ModuleList()
        
        for idx, block in enumerate(self.blocks):
            # Create MultiViewBlock with same config as original block
            multiview_block = MultiViewBlock(
                x_dim=block.x_dim,
                context_dim=block.cross_attn.context_dim,
                num_heads=block.self_attn.n_heads,
                mlp_ratio=4.0,  # Assuming default
                use_adaln_lora=block.use_adaln_lora,
                adaln_lora_dim=256 if block.use_adaln_lora else 0,
                backend="transformer_engine",
                num_views=self.num_views,
            )
            
            # Copy weights from original block
            multiview_block.load_state_dict(block.state_dict(), strict=False)
            
            new_blocks.append(multiview_block)
        
        self.blocks = new_blocks
        
        log.info(f"✅ Replaced {len(new_blocks)} blocks with MultiViewBlock "
                f"(num_views={self.num_views})")

