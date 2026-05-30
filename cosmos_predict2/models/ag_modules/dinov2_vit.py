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

""" DINOv2 ViT Model Module """

import math
from functools import partial
import copy

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard

from timm.layers import PatchEmbed, SwiGLUPacked
from timm.models.vision_transformer import VisionTransformer
import logging
import safetensors



class DinoV2ViT(VisionTransformer):
    """
    This class extends the VisionTransformer class from timm library so that we can
    handle different image sizes.
    """

    def __init__(self, *args, **kwargs):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""
        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.
        init_cfg = kwargs.pop('init_cfg', None)
        self.init_cfg = None
        self._is_init = False
        if init_cfg is not None:
            self.init_cfg = copy.deepcopy(init_cfg)
        register_tokens = kwargs.pop('register_tokens', 0)
        self.num_register_tokens = register_tokens
        super(DinoV2ViT, self).__init__(*args, **kwargs)

        if register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, register_tokens, self.embed_dim)
            )

    @property
    def is_init(self):
        """Return is_init."""
        return self._is_init

    @is_init.setter
    def is_init(self, value):
        self._is_init = value

    def __interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8

        w0, h0 = w0 + 0.1, h0 + 0.1

        # We need fp32 for the interpolation
        reshaped_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(N)), int(math.sqrt(N)), dim
        ).permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            reshaped_pos_embed,
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1], "The interpolated value does not match the positional embedding size."
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def _pos_embed(self, x):
        B, S, _ = x.shape
        w = h = int(math.sqrt(S)) * self.patch_embed.patch_size[0]

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.__interpolate_pos_encoding(x, w, h)

        # add register tokens
        if self.num_register_tokens > 0:
            x = torch.cat((x, self.register_tokens.expand(B, -1, -1)), dim=1)

        return self.pos_drop(x)

    def forward(self, *args, **kwargs):
        """
        Forward function and return the flatten output (cls_token)
        """

        if self.freeze:
            self.eval()

        input_mask = args[0]
        B, C, H, W = input_mask.shape
        if H != 518 or W != 518:
            input_mask = torch.nn.functional.interpolate(input_mask, size=(518, 518), mode="bilinear")

        if input_mask.size(1)==1: # Channel = 1
            input_mask = input_mask.repeat(1,3,1,1)

        x = super().forward(input_mask, **kwargs)
        return x

    def fully_shard(self, mesh: DeviceMesh) -> None:
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(
                block, mesh=mesh, reshard_after_forward=reshard_after_forward
            )
        fully_shard(self.patch_embed, mesh=mesh, reshard_after_forward=True)
        # nn.Parameter cannot be fully sharded. Skip `cls_token`, `reg_token`
        # and `pos_embed`.


class VitLargePatch14Dinov2Swiglu(DinoV2ViT):
    """
    DINOV2 ViT Large model with SwiGLU activation
    """

    def __init__(self, *args, freeze=False, init_cfg=None, pool_kernel=3, learnable_head=False, **kwargs):
        """Initialize"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        model_kwargs = dict(
            patch_size=14,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            init_values=1e-5,
            img_size=518,
            mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU,
            mlp_ratio=5472 / 1024,
            embed_layer=partial(PatchEmbed, strict_img_size=False),
            global_pool='',
            num_classes=0,
            init_cfg=init_cfg,
            **kwargs
        )
        super(VitLargePatch14Dinov2Swiglu, self).__init__(*args, **model_kwargs)

        self.freeze = freeze
        pretrained = None
        if init_cfg:
            pretrained = init_cfg["checkpoint"]
            self.logger.info(f"[NVDINOV2] Loading model checkpoint: {pretrained}")
            state_dict = torch.load(pretrained, weights_only=True)
            _tmp_st_output = self.load_state_dict(state_dict, strict=False)

        if freeze:
            assert pretrained is not None, "You shouldn't freeze a model without specifying pretrained"
            self.logger.info(f"[NVDINOV2] Freezing backbone:")
            self.eval()

            for p in self.parameters():
                p.requires_grad = False

        # Pooling
        self.pool_layer = torch.nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_kernel)

    def forward(self, *args, **kwargs):
        """
        Forward function and return the flatten output (cls_token)
        """
        if self.freeze:
            self.eval()

        # Resize input
        x = super().forward(*args, **kwargs) # [B, 1370 = (37 * 37 + 1), 1024]
        B, L, C = x.shape # [B, 1370, 1024]
        x = x[:, 1:, :] # Drop class token, [B, 1369, 1024]
        x = x.transpose(1, 2).reshape(-1, self.embed_dim, 37, 37) # [B, 1024, 37, 37]

        x = self.pool_layer(x) # [B, 1024, out_H, out_W]

        B, C, H, W = x.shape

        x = x.reshape(B, C, H*W).transpose(1, 2) # [B, H*W, C]

        return x


class VitGiantPatch14Reg4Dinov2Swiglu(DinoV2ViT):
    """
    DINOV2 ViT Giant model with SwiGLU activation
    """

    def __init__(self, *args, freeze=False, init_cfg=None, **kwargs):
        """Initialize"""
        model_kwargs = dict(
            patch_size=14,
            embed_dim=1536,
            depth=40,
            num_heads=24,
            init_values=1e-5,
            img_size=518,
            mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU,
            mlp_ratio=8192 / 1536,
            embed_layer=partial(PatchEmbed, strict_img_size=False),
            global_pool="token",
            num_classes=0,
            register_tokens=4,
            init_cfg=init_cfg,
            **kwargs
        )
        super(VitGiantPatch14Reg4Dinov2Swiglu, self).__init__(*args, **model_kwargs)

        self.freeze = freeze
        pretrained = None
        if init_cfg:
            pretrained = init_cfg["checkpoint"]
            state_dict = torch.load(pretrained, weights_only=True)
            _tmp_st_output = self.load_state_dict(state_dict, strict=False)

        if freeze:
            assert pretrained is not None, "You shouldn't freeze a model without specifying pretrained"
            self.eval()

            for p in self.parameters():
                p.requires_grad = False
