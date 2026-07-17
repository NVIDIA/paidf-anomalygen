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

from hydra.core.config_store import ConfigStore

from cosmos_predict2.configs.anomaly_gen.config_text2image import (
    PREDICT2_TEXT2IMAGE_PIPELINE_2B,
    PREDICT2_TEXT2IMAGE_PIPELINE_14B,
)
from cosmos_predict2.configs.anomaly_gen.config_video2world import (
    PREDICT2_ANOMALY_GEN_MULTIVIEW_PIPELINE_2B,
    PREDICT2_ANOMALY_GEN_MULTIVIEW_PIPELINE_14B,
)
from cosmos_predict2.models.anomaly_gen_model import (
    Predict2ModelManagerConfig,
    Predict2AnomalyGenModel,
    Predict2AnomalyGenModelConfig,
)
from cosmos_predict2.models.anomaly_gen_multiview_model import (
    Predict2AnomalyGenMultiViewModel,
    Predict2AnomalyGenMultiViewModelConfig,
    Predict2MultiViewModelManagerConfig,
)
from imaginaire.lazy_config import LazyCall as L, LazyDict

PREDICT2_ANOMALY_GEN_FSDP_2B = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2AnomalyGenModel)(
        config=Predict2AnomalyGenModelConfig(
            pipe_config=PREDICT2_TEXT2IMAGE_PIPELINE_2B,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            ag_config=L(LazyDict)(
                ad_precision="float32",
                t5_model_name=None,
                mask_encoder=L(LazyDict)(
                    encoder_type="nvdinov2",
                    encoder_config=L(LazyDict)(
                        pool_kernel=7,
                        init_cfg=L(LazyDict)(
                            checkpoint=None
                        )
                    ),
                    freeze=True,
                ),
                adapter=L(LazyDict)(
                    adapter_type="mlp_gelu",
                    adapter_config=L(LazyDict)(
                        num_layers=2,
                        input_hidden_size=1024,
                        final_hidden_size=1024
                    ),
                    freeze=False,
                ),
                anomaly_embedding=L(LazyDict)(
                    num_tokens=256,
                    token_dim=1024,
                    init_word="abnormal",
                    anomaly_types=None,
                    freeze=False,
                ),
                text_tokenizer=L(LazyDict)(
                    max_length=512,
                ),
            ),
        ),
        _recursive_=False,
    ),
)

PREDICT2_ANOMALY_GEN_DDP_2B = dict(
    trainer=dict(
        distributed_parallelism="ddp",
        ddp=dict(
            find_unused_parameters=True,
            static_graph=False,
            broadcast_buffers=True,
        ),
    ),
    model=L(Predict2AnomalyGenModel)(
        config=Predict2AnomalyGenModelConfig(
            pipe_config=PREDICT2_TEXT2IMAGE_PIPELINE_2B,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=0,
            use_cuda_graphs_for_dit=True,
            compile_vae_encoder=True,
            compile_text_encoder=True,
            compile_mask_encoder=True,
            ag_config=L(LazyDict)(
                ad_precision="float32",
                t5_model_name=None,
                mask_encoder=L(LazyDict)(
                    encoder_type="nvdinov2",
                    encoder_config=L(LazyDict)(
                        pool_kernel=7,
                        init_cfg=L(LazyDict)(
                            checkpoint=None
                        )
                    ),
                    freeze=True,
                ),
                adapter=L(LazyDict)(
                    adapter_type="mlp_gelu",
                    adapter_config=L(LazyDict)(
                        num_layers=2,
                        input_hidden_size=1024,
                        final_hidden_size=1024
                    ),
                    freeze=False,
                ),
                anomaly_embedding=L(LazyDict)(
                    num_tokens=256,
                    token_dim=1024,
                    init_word="abnormal",
                    anomaly_types=None,
                    freeze=False,
                ),
                text_tokenizer=L(LazyDict)(
                    max_length=512,
                ),
            ),
        ),
        _recursive_=False,
    ),
)

PREDICT2_ANOMALY_GEN_FSDP_14B = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2AnomalyGenModel)(
        config=Predict2AnomalyGenModelConfig(
            pipe_config=PREDICT2_TEXT2IMAGE_PIPELINE_14B,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image/model.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            ag_config=L(LazyDict)(
                ad_precision="float32",
                t5_model_name=None,
                mask_encoder=L(LazyDict)(
                    encoder_type="nvdinov2",
                    encoder_config=L(LazyDict)(
                        pool_kernel=7,
                        init_cfg=L(LazyDict)(
                            checkpoint=None
                        )
                    ),
                    freeze=True,
                ),
                adapter=L(LazyDict)(
                    adapter_type="mlp_gelu",
                    adapter_config=L(LazyDict)(
                        num_layers=2,
                        input_hidden_size=1024,
                        final_hidden_size=1024
                    ),
                    freeze=False,
                ),
                anomaly_embedding=L(LazyDict)(
                    num_tokens=256,
                    anomaly_types=None,
                    token_dim=1024,
                    init_word="abnormal",
                    freeze=False,
                ),
                text_tokenizer=L(LazyDict)(
                    max_length=512,
                ),
            ),
        ),
        _recursive_=False,
    ),
)

PREDICT2_ANOMALY_GEN_DDP_14B = dict(
    trainer=dict(
        distributed_parallelism="ddp",
        ddp=dict(
            find_unused_parameters=True,
            static_graph=False,
            broadcast_buffers=True,
        ),
    ),
    model=L(Predict2AnomalyGenModel)(
        config=Predict2AnomalyGenModelConfig(
            pipe_config=PREDICT2_TEXT2IMAGE_PIPELINE_14B,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image/model.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=0,
            use_cuda_graphs_for_dit=True,
            compile_vae_encoder=True,
            compile_text_encoder=True,
            compile_mask_encoder=True,
            ag_config=L(LazyDict)(
                ad_precision="float32",
                t5_model_name=None,
                mask_encoder=L(LazyDict)(
                    encoder_type="nvdinov2",
                    encoder_config=L(LazyDict)(
                        pool_kernel=7,
                        init_cfg=L(LazyDict)(
                            checkpoint=None
                        )
                    ),
                    freeze=True,
                ),
                adapter=L(LazyDict)(
                    adapter_type="mlp_gelu",
                    adapter_config=L(LazyDict)(
                        num_layers=2,
                        input_hidden_size=1024,
                        final_hidden_size=1024
                    ),
                    freeze=False,
                ),
                anomaly_embedding=L(LazyDict)(
                    num_tokens=256,
                    token_dim=1024,
                    init_word="abnormal",
                    anomaly_types=None,
                    freeze=False,
                ),
                text_tokenizer=L(LazyDict)(
                    max_length=512,
                ),
            ),
        ),
        _recursive_=False,
    ),
)


# =====================================================
# Multi-view Anomaly Generation Models (Video2World)
# =====================================================

PREDICT2_ANOMALY_GEN_MULTIVIEW_FSDP_2B = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2AnomalyGenMultiViewModel)(
        config=Predict2AnomalyGenMultiViewModelConfig(
            pipe_config=PREDICT2_ANOMALY_GEN_MULTIVIEW_PIPELINE_2B,
            model_manager_config=L(Predict2MultiViewModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt",  # Use Text2Image checkpoint
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            ag_config=L(LazyDict)(
                ad_precision="float32",
                t5_model_name=None,
                mask_encoder=L(LazyDict)(
                    encoder_type="nvdinov2",
                    encoder_config=L(LazyDict)(
                        pool_kernel=7,
                        init_cfg=L(LazyDict)(
                            checkpoint=None
                        )
                    ),
                    freeze=True,
                ),
                adapter=L(LazyDict)(
                    adapter_type="mlp_gelu",
                    adapter_config=L(LazyDict)(
                        num_layers=2,
                        input_hidden_size=1024,
                        final_hidden_size=1024
                    ),
                    freeze=False,
                ),
                anomaly_embedding=L(LazyDict)(
                    num_tokens=256,
                    token_dim=1024,
                    init_word="abnormal",
                    anomaly_types=None,
                    freeze=False,
                    perview_perturbation_std=0.01,
                ),
                text_tokenizer=L(LazyDict)(
                    max_length=512,
                ),
            ),
        ),
        _recursive_=False,
    ),
)

PREDICT2_ANOMALY_GEN_MULTIVIEW_DDP_2B = dict(
    trainer=dict(
        distributed_parallelism="ddp",
        ddp=dict(
            find_unused_parameters=True,
            static_graph=False,
            broadcast_buffers=True,
        ),
    ),
    model=L(Predict2AnomalyGenMultiViewModel)(
        config=Predict2AnomalyGenMultiViewModelConfig(
            pipe_config=PREDICT2_ANOMALY_GEN_MULTIVIEW_PIPELINE_2B,
            model_manager_config=L(Predict2MultiViewModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt",  # Use Text2Image checkpoint
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=0,
            use_cuda_graphs_for_dit=True,
            compile_vae_encoder=True,
            compile_text_encoder=True,
            compile_mask_encoder=True,
            ag_config=L(LazyDict)(
                ad_precision="float32",
                t5_model_name=None,
                mask_encoder=L(LazyDict)(
                    encoder_type="nvdinov2",
                    encoder_config=L(LazyDict)(
                        pool_kernel=7,
                        init_cfg=L(LazyDict)(
                            checkpoint=None
                        )
                    ),
                    freeze=True,
                ),
                adapter=L(LazyDict)(
                    adapter_type="mlp_gelu",
                    adapter_config=L(LazyDict)(
                        num_layers=2,
                        input_hidden_size=1024,
                        final_hidden_size=1024
                    ),
                    freeze=False,
                ),
                anomaly_embedding=L(LazyDict)(
                    num_tokens=256,
                    token_dim=1024,
                    init_word="abnormal",
                    anomaly_types=None,
                    freeze=False,
                    perview_perturbation_std=0.01,
                ),
                text_tokenizer=L(LazyDict)(
                    max_length=512,
                ),
            ),
        ),
        _recursive_=False,
    ),
)

PREDICT2_ANOMALY_GEN_MULTIVIEW_FSDP_14B = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2AnomalyGenMultiViewModel)(
        config=Predict2AnomalyGenMultiViewModelConfig(
            pipe_config=PREDICT2_ANOMALY_GEN_MULTIVIEW_PIPELINE_14B,
            model_manager_config=L(Predict2MultiViewModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image/model.pt",  # Use Text2Image checkpoint
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            ag_config=L(LazyDict)(
                ad_precision="float32",
                t5_model_name=None,
                mask_encoder=L(LazyDict)(
                    encoder_type="nvdinov2",
                    encoder_config=L(LazyDict)(
                        pool_kernel=7,
                        init_cfg=L(LazyDict)(
                            checkpoint=None
                        )
                    ),
                    freeze=True,
                ),
                adapter=L(LazyDict)(
                    adapter_type="mlp_gelu",
                    adapter_config=L(LazyDict)(
                        num_layers=2,
                        input_hidden_size=1024,
                        final_hidden_size=1024
                    ),
                    freeze=False,
                ),
                anomaly_embedding=L(LazyDict)(
                    num_tokens=256,
                    anomaly_types=None,
                    token_dim=1024,
                    init_word="abnormal",
                    freeze=False,
                    perview_perturbation_std=0.01,
                ),
                text_tokenizer=L(LazyDict)(
                    max_length=512,
                ),
            ),
        ),
        _recursive_=False,
    ),
)


PREDICT2_ANOMALY_GEN_MULTIVIEW_DDP_14B = dict(
    trainer=dict(
        distributed_parallelism="ddp",
        ddp=dict(
            find_unused_parameters=True,
            static_graph=False,
            broadcast_buffers=True,
        ),
    ),
    model=L(Predict2AnomalyGenMultiViewModel)(
        config=Predict2AnomalyGenMultiViewModelConfig(
            pipe_config=PREDICT2_ANOMALY_GEN_MULTIVIEW_PIPELINE_14B,
            model_manager_config=L(Predict2MultiViewModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image/model.pt",  # Use Text2Image checkpoint
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=0,
            use_cuda_graphs_for_dit=True,
            compile_vae_encoder=True,
            compile_text_encoder=True,
            compile_mask_encoder=True,
            ag_config=L(LazyDict)(
                ad_precision="float32",
                t5_model_name=None,
                mask_encoder=L(LazyDict)(
                    encoder_type="nvdinov2",
                    encoder_config=L(LazyDict)(
                        pool_kernel=7,
                        init_cfg=L(LazyDict)(
                            checkpoint=None
                        )
                    ),
                    freeze=True,
                ),
                adapter=L(LazyDict)(
                    adapter_type="mlp_gelu",
                    adapter_config=L(LazyDict)(
                        num_layers=2,
                        input_hidden_size=1024,
                        final_hidden_size=1024
                    ),
                    freeze=False,
                ),
                anomaly_embedding=L(LazyDict)(
                    num_tokens=256,
                    anomaly_types=None,
                    token_dim=1024,
                    init_word="abnormal",
                    freeze=False,
                    perview_perturbation_std=0.01,
                ),
                text_tokenizer=L(LazyDict)(
                    max_length=512,
                ),
            ),
        ),
        _recursive_=False,
    ),
)


def register_model() -> None:
    cs = ConfigStore.instance()
    # predict2 anomaly gen 2b model (single image - Text2Image based)
    cs.store(
        group="model", package="_global_", name="predict2_anomaly_gen_fsdp_2b", node=PREDICT2_ANOMALY_GEN_FSDP_2B
    )
    # predict2 anomaly gen 2b model with DDP (single image - Text2Image based)
    cs.store(
        group="model", package="_global_", name="predict2_anomaly_gen_ddp_2b", node=PREDICT2_ANOMALY_GEN_DDP_2B
    )
    # predict2 anomaly gen 14b model (single image - Text2Image based)
    cs.store(
        group="model", package="_global_", name="predict2_anomaly_gen_fsdp_14b", node=PREDICT2_ANOMALY_GEN_FSDP_14B
    )
    # predict2 anomaly gen 14b model with DDP (single image - Text2Image based)
    cs.store(
        group="model", package="_global_", name="predict2_anomaly_gen_ddp_14b", node=PREDICT2_ANOMALY_GEN_DDP_14B
    )
    # predict2 multi-view anomaly gen 2b model (Video2World based)
    cs.store(
        group="model", package="_global_", name="predict2_anomaly_gen_multiview_fsdp_2b", node=PREDICT2_ANOMALY_GEN_MULTIVIEW_FSDP_2B
    )
    # predict2 multi-view anomaly gen 2b model with DDP + CUDA graphs (Video2World based)
    cs.store(
        group="model", package="_global_", name="predict2_anomaly_gen_multiview_ddp_2b", node=PREDICT2_ANOMALY_GEN_MULTIVIEW_DDP_2B
    )
    # predict2 multi-view anomaly gen 14b model (Video2World based)
    cs.store(
        group="model", package="_global_", name="predict2_anomaly_gen_multiview_fsdp_14b", node=PREDICT2_ANOMALY_GEN_MULTIVIEW_FSDP_14B
    )
    # predict2 multi-view anomaly gen 14b model with DDP + CUDA graphs (Video2World based)
    cs.store(
        group="model", package="_global_", name="predict2_anomaly_gen_multiview_ddp_14b", node=PREDICT2_ANOMALY_GEN_MULTIVIEW_DDP_14B
    )
