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

import torch
from sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.memory_encoder import CXBlock, Fuser, MaskDownSampler, MemoryEncoder
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.sam.transformer import RoPEAttention

from imaginaire.utils import log
from pseudo_label.infosam.infosam2_module import InfoSAM2, InfoSAMHiera


def build_infosam2(
    pretrained_checkpoint,
    checkpoint=None,
    adapter_config=None,
    adapter_mlp_ratio=None,
    use_pretrained_sam2=False,
    device="cuda",
    apply_postprocessing=True,
):
    stages = [2, 6, 36, 4]
    if adapter_config is None:
        adapter_config = [0 for _ in range(sum(stages))]
        for i in range(stages[0]):
            adapter_config[i] = -1
    if adapter_mlp_ratio is None:
        adapter_mlp_ratio = [0.25 for _ in range(sum(stages) * 2)]
    if apply_postprocessing:
        sam_mask_decoder_extra_args = {
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        }
    else:
        sam_mask_decoder_extra_args = None

    # Use the pretrained SAM2.
    if use_pretrained_sam2:
        adapter_config = [-1 for _ in range(len(adapter_config))]

    model = InfoSAM2(
        image_encoder=ImageEncoder(
            scalp=1,
            trunk=InfoSAMHiera(
                embed_dim=144,
                num_heads=2,
                stages=stages,
                global_att_blocks=[23, 33, 43],
                window_pos_embed_bkg_spatial_size=[7, 7],
                window_spec=[8, 4, 16, 8],
                adapter_config=adapter_config,
                adapter_mlp_ratio=adapter_mlp_ratio,
            ),
            neck=FpnNeck(
                position_encoding=PositionEmbeddingSine(
                    num_pos_feats=256,
                    normalize=True,
                    scale=None,
                    temperature=10000,
                ),
                d_model=256,
                backbone_channel_list=[1152, 576, 288, 144],
                fpn_top_down_levels=[2, 3],
                fpn_interp_model="nearest",
            ),
        ),
        memory_attention=MemoryAttention(
            d_model=256,
            pos_enc_at_input=True,
            layer=MemoryAttentionLayer(
                activation="relu",
                dim_feedforward=2048,
                dropout=0.1,
                pos_enc_at_attn=False,
                self_attention=RoPEAttention(
                    rope_theta=10000.0,
                    feat_sizes=[64, 64],
                    embedding_dim=256,
                    num_heads=1,
                    downsample_rate=1,
                    dropout=0.1,
                ),
                d_model=256,
                pos_enc_at_cross_attn_keys=True,
                pos_enc_at_cross_attn_queries=False,
                cross_attention=RoPEAttention(
                    rope_theta=10000.0,
                    feat_sizes=[64, 64],
                    embedding_dim=256,
                    num_heads=1,
                    downsample_rate=1,
                    dropout=0.1,
                    kv_in_dim=64,
                ),
            ),
            num_layers=4,
        ),
        memory_encoder=MemoryEncoder(
            out_dim=64,
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=64,
                normalize=True,
                scale=None,
                temperature=10000,
            ),
            mask_downsampler=MaskDownSampler(
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            fuser=Fuser(
                layer=CXBlock(
                    dim=256,
                    kernel_size=7,
                    padding=3,
                    layer_scale_init_value=1e-6,
                    use_dwconv=True,
                ),
                num_layers=2,
            ),
        ),
        num_maskmem=7,
        image_size=1024,
        # apply scaled sigmoid on mask logits for memory encoder, and directly feed input mask as output mask
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True,
        # Memory
        directly_add_no_mem_embed=True,
        no_obj_embed_spatial=True,
        # use high-resolution feature map in the SAM mask decoder
        use_high_res_features_in_sam=True,
        # output 3 masks on the first click on initial conditioning frames
        multimask_output_in_sam=True,
        # SAM heads
        iou_prediction_use_sigmoid=True,
        # cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=True,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_signed_tpos_enc_to_obj_ptrs=True,
        only_obj_ptrs_in_the_past_for_eval=True,
        # object occlusion prediction
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        # multimask tracking settings
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
        sam_mask_decoder_extra_args=sam_mask_decoder_extra_args,
        # Compilation flag
        compile_image_encoder=False,
    )

    # Load the weights. Normalize the pretrained dict to its inner "model" (or
    # top level) once, so merging a fine-tune checkpoint works even when the
    # pretrained dict has no "model" key. Previously the merge indexed
    # ["model"] unconditionally, raising KeyError for a raw pretrained dict.
    pretrained_state_dict = torch.load(pretrained_checkpoint, weights_only=True)
    if "model" in pretrained_state_dict:
        model_state_dict = pretrained_state_dict["model"]
    else:
        model_state_dict = pretrained_state_dict
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model_state_dict.update(state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(
        model_state_dict, strict=False
    )
    if len(unexpected_keys) > 0:
        raise ValueError(f"Unexpected keys in SAM2 state_dict: {unexpected_keys}")
    if len(missing_keys) > 0:
        if checkpoint is None:
            msg = (
                f"Number of missing keys in InfoSAM2 state_dict: {len(missing_keys)}. "
                "It is acceptable for keys to be missing during fine-tuning."
            )
            log.info(msg)
        else:
            msg = f"Number of missing keys in InfoSAM2 state_dict: {len(missing_keys)}"
            raise ValueError(msg)
    model = model.to(device)
    model.eval()
    return model
