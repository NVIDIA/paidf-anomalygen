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

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr
from sam2.modeling.backbones.hieradet import (
    Hiera,
    MultiScaleBlock,
    PatchEmbed,
    do_pool,
    window_partition,
    window_unpartition,
)
from sam2.modeling.sam2_base import SAM2Base


class InfoSAMAdapter(nn.Module):
    def __init__(
        self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class InfoSAMMultiScaleBlock(MultiScaleBlock):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        adapter_type: int = -1,
        adapter_mlp_ratio=[0.25, 0.25],
    ):
        super().__init__(
            dim,
            dim_out,
            num_heads,
            mlp_ratio,
            drop_path,
            norm_layer,
            q_stride,
            act_layer,
            window_size,
        )

        # InfoSAM adapter.
        self.adapter_type = adapter_type
        if self.adapter_type == 0:
            self.space_adapter_series = InfoSAMAdapter(
                dim_out, mlp_ratio=adapter_mlp_ratio[0]
            )
            self.mlp_adapter_series = InfoSAMAdapter(
                dim_out, mlp_ratio=adapter_mlp_ratio[1]
            )
            self.adapters_list = [self.space_adapter_series, self.mlp_adapter_series]
        else:
            self.adapters_list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        if self.adapter_type == 0:
            x = self.space_adapter_series(x)

        x = shortcut + self.drop_path(x)

        # MLP
        if self.adapter_type == 0:
            x = x + self.mlp_adapter_series(self.drop_path(self.mlp(self.norm2(x))))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class InfoSAMHiera(Hiera):
    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        weights_path=None,
        return_interm_layers=True,  # return feats from every stage
        # InfoSAM adapter.
        adapter_config=None,
        adapter_mlp_ratio=None,
    ):
        nn.Module.__init__(self)

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            # HY: Replace MultiScaleBlock with InfoSAMMultiScaleBlock.
            block = InfoSAMMultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                adapter_type=adapter_config[i],
                adapter_mlp_ratio=adapter_mlp_ratio[i * 2 : (i + 1) * 2],
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            self.load_state_dict(chkpt, strict=False)


class InfoSAM2(SAM2Base):
    def get_image_embedding(self, x: torch.Tensor):
        # Pre-defined parameters.
        _bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]

        batch_size = x.shape[0]
        backbone_out = self.forward_image(x)
        _, vision_feats, _, _ = self._prepare_backbone_features(backbone_out)
        if self.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], _bb_feat_sizes[::-1])
        ][::-1]
        attn_features = feats[-1]
        high_res_features = feats[:-1]
        return attn_features, high_res_features

    def infosam_decode(self, image_embeddings, high_res_features, input_points=None):
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=input_points,
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions, mask_tokens_out, _ = self.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        infosam_mask_tokens_out = mask_tokens_out[:, 0:1, :]
        return low_res_masks, iou_predictions, infosam_mask_tokens_out


class RelationModel(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)  # Linear projection for Query
        self.key_proj = nn.Linear(dim, dim)  # Linear projection for Key
        self.scale = dim**-0.5  # Scaled factor for attention scores
        self.layer_norm_feat = nn.LayerNorm(dim)  # LayerNorm for feat
        self.layer_norm_mask = nn.LayerNorm(dim)  # LayerNorm for mask

    def forward(self, feat, mask):
        # Apply LayerNorm to feat and mask (pre-norm)
        feat = self.layer_norm_feat(feat)  # (bsz, h*w, dim)
        mask = self.layer_norm_mask(mask)  # (bsz, 1, dim)

        # Linear projections for Query and Key
        query = self.query_proj(mask)  # (bsz, 1, dim)
        key = self.key_proj(feat)  # (bsz, h*w, dim)

        # Compute scaled attention scores
        linear_attn_scores = (
            torch.matmul(query, key.transpose(-1, -2)) * self.scale
        )  # (bsz, 1, h*w)

        # Residual attention scores (direct dot product)
        residual_attn_scores = torch.matmul(
            mask, feat.transpose(-1, -2)
        )  # (bsz, 1, h*w)

        # Combine linear and residual attention scores
        attn_scores = linear_attn_scores + residual_attn_scores

        attn_scores = attn_scores.squeeze(1)
        batch_size = attn_scores.shape[0]
        attn_score_norm = F.normalize(attn_scores.view(batch_size, -1))
        return attn_score_norm
