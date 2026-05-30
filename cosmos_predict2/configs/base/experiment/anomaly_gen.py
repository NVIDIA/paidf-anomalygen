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

import math

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.anomaly_gen.anomaly_dataset import Dataset, AnomalyInpaintValidationDataset
from imaginaire.lazy_config import LazyCall as L
from cosmos_predict2.callbacks.log_image import LogImage


class BatchAlignedDistributedSampler(DistributedSampler):
    """DistributedSampler that pads each rank's indices to a multiple of batch_size.

    This prevents the last batch from being smaller than expected, which is
    required when CUDA graphs are used (they are captured for a fixed shape
    and cannot replay with a different batch size).
    """

    def __init__(self, *args, batch_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_samples = math.ceil(self.num_samples / batch_size) * batch_size
        self.total_size = self.num_samples * self.num_replicas


def get_sampler(dataset, shuffle=True, batch_size=1):
    return BatchAlignedDistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=shuffle,
        seed=0,
        batch_size=batch_size,
    )

cs = ConfigStore.instance()

### Dataset & DataLoader
train_anomaly_dataset = L(Dataset)(
    dataset_dir="",
    num_frames=1,
    image_size=(512, 512),
    anomaly_types=[],
    data_augprob=0.5,
    seed=1,
    # Multi-view support (optional)
    view_types=None,  # List of view type suffixes, e.g., ['LowAngleLight', 'SolderLight', ...]
)
valid_anomaly_dataset = L(AnomalyInpaintValidationDataset)(
    input_data_path="",
)

dataloader_train_anomaly = L(DataLoader)(
    dataset=train_anomaly_dataset,
    sampler=L(get_sampler)(dataset=train_anomaly_dataset, batch_size=2),
    batch_size=2,
    drop_last=False, # Don't turn on this on few-shot scenario
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
dataloader_val_anomaly = L(DataLoader)(
    dataset=valid_anomaly_dataset,
    sampler=L(get_sampler)(dataset=valid_anomaly_dataset, shuffle=False, batch_size=2),
    batch_size=2,
    drop_last=False, # Don't turn on this on few-shot scenario
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

predict2_anomaly_gen_fsdp_2b = dict(
    defaults=[
        {"override /model": "predict2_anomaly_gen_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=8,
            pipe_config=dict(guardrail_config=dict(enabled=False)),
        )
    ),
    optimizer=dict(
        lr=1e-2,  # Suitable for anomaly diffusion training
        weight_decay=1e-6,
        betas=[0.9, 0.99],
        eps=1e-10,
    ),
    scheduler=dict(
        f_max=[0.2],
        f_min=[0.1],
        warm_up_steps=[1_000],
        cycle_lengths=[100_000],
    ),
    job=dict(
        project="posttraining",
        group="anomaly_gen",
        name="2b_anomaly_gen",
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train_anomaly,
    dataloader_val=dataloader_val_anomaly,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=200),
            # LogImage is expensive. Disable it for now.
            # log_image = L(LogImage)(
            #     exp_path = "results/MeiweiPCB/trialrun",
            #     every_n = 200,     # Configurable
            #     num_steps = 35,  
            #     guidance = 1.5,    # Configurable
            #     seed = 1,               # Configurable
            #     is_negative_prompt = True,
            #     n_sample = 2       # Configurable
            # )
        ),
    ),
    checkpoint=dict(
        save_iter=200,
    ),
)

predict2_anomaly_gen_ddp_2b = dict(
    defaults=[
        {"override /model": "predict2_anomaly_gen_ddp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=0,
            pipe_config=dict(guardrail_config=dict(enabled=False)),
        )
    ),
    optimizer=dict(
        lr=1e-2,
        weight_decay=1e-6,
        betas=[0.9, 0.99],
        eps=1e-10,
    ),
    scheduler=dict(
        f_max=[0.2],
        f_min=[0.1],
        warm_up_steps=[1_000],
        cycle_lengths=[100_000],
    ),
    job=dict(
        project="posttraining",
        group="anomaly_gen",
        name="2b_anomaly_gen_ddp",
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train_anomaly,
    dataloader_val=dataloader_val_anomaly,
    trainer=dict(
        distributed_parallelism="ddp",
        ddp=dict(
            find_unused_parameters=True,
            static_graph=False,
            broadcast_buffers=True,
        ),
        callbacks=dict(
            iter_speed=dict(hit_thres=200),
            # LogImage is expensive. Disable it for now.
            # log_image = L(LogImage)(
            #     exp_path = "results/MeiweiPCB/trialrun",
            #     every_n = 200,
            #     num_steps = 35,
            #     guidance = 1.5,
            #     seed = 1,
            #     is_negative_prompt = True,
            #     n_sample = 2
            # )
        ),
    ),
    checkpoint=dict(
        save_iter=200,
    ),
)

predict2_anomaly_gen_fsdp_14b = dict(
    defaults=[
        {"override /model": "predict2_anomaly_gen_fsdp_14b"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=32,
            pipe_config=dict(guardrail_config=dict(enabled=False)),
        )
    ),
    optimizer=dict(
        lr=1e-2,  # Suitable for anomaly diffusion training
        weight_decay=1e-6,
        betas=[0.9, 0.99],
        eps=1e-10,
    ),
    scheduler=dict(
        f_max=[0.2],
        f_min=[0.1],
        warm_up_steps=[1_000],
        cycle_lengths=[100_000],
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    job=dict(
        project="posttraining",
        group="anomaly_gen",
        name="14b_anomaly_gen",
    ),
    dataloader_train=dataloader_train_anomaly,
    dataloader_val=dataloader_val_anomaly,
    trainer=dict(
        max_iter=200000, # Configurable
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=200),
            # LogImage is expensive. Disable it for now.
            # log_image = L(LogImage)(
            #     exp_path = "results/MeiweiPCB/trialrun",
            #     every_n = 200,     # Configurable
            #     num_steps = 35,  
            #     guidance = 1.5,    # Configurable
            #     seed = 1,               # Configurable
            #     is_negative_prompt = True,
            #     n_sample = 2       # Configurable
            # )
        ),
    ),
    checkpoint=dict(
        save_iter=200,
    ),
)

predict2_anomaly_gen_ddp_14b = dict(
    defaults=[
        {"override /model": "predict2_anomaly_gen_ddp_14b"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=0,
            pipe_config=dict(guardrail_config=dict(enabled=False)),
        )
    ),
    optimizer=dict(
        lr=1e-2,  # Suitable for anomaly diffusion training
        weight_decay=1e-6,
        betas=[0.9, 0.99],
        eps=1e-10,
    ),
    scheduler=dict(
        f_max=[0.2],
        f_min=[0.1],
        warm_up_steps=[1_000],
        cycle_lengths=[100_000],
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    job=dict(
        project="posttraining",
        group="anomaly_gen",
        name="14b_anomaly_gen_ddp",
    ),
    dataloader_train=dataloader_train_anomaly,
    dataloader_val=dataloader_val_anomaly,
    trainer=dict(
        distributed_parallelism="ddp",
        ddp=dict(
            find_unused_parameters=True,
            static_graph=False,
            broadcast_buffers=True,
        ),
        callbacks=dict(
            iter_speed=dict(hit_thres=200),
            # LogImage is expensive. Disable it for now.
            # log_image = L(LogImage)(
            #     exp_path = "results/MeiweiPCB/trialrun",
            #     every_n = 200,     # Configurable
            #     num_steps = 35,  
            #     guidance = 1.5,    # Configurable
            #     seed = 1,               # Configurable
            #     is_negative_prompt = True,
            #     n_sample = 2       # Configurable
            # )
        ),
    ),
    checkpoint=dict(
        save_iter=200,
    ),
)

# =====================================================
# Multi-view Anomaly Generation Experiments (Video2World)
# =====================================================

# Multi-view anomaly gen 2B (Video2World based)
predict2_anomaly_gen_multiview_fsdp_2b = dict(
    defaults=[
        {"override /model": "predict2_anomaly_gen_multiview_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=8,
            pipe_config=dict(guardrail_config=dict(enabled=False)),
        )
    ),
    optimizer=dict(
        lr=1e-2,
        weight_decay=1e-6,
        betas=[0.9, 0.99],
        eps=1e-10,
    ),
    scheduler=dict(
        f_max=[0.2],
        f_min=[0.1],
        warm_up_steps=[1_000],
        cycle_lengths=[100_000],
    ),
    job=dict(
        project="posttraining",
        group="anomaly_gen_multiview",
        name="2b_anomaly_gen_multiview",
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train_anomaly,
    dataloader_val=dataloader_val_anomaly,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=200),
            # LogImage is expensive. Disable it for now.
            # log_image=L(LogImage)(
            #     exp_path="results/multiview/trialrun",
            #     every_n=200,
            #     num_steps=35,
            #     guidance=1.5,
            #     seed=1,
            #     is_negative_prompt=True,
            #     n_sample=2
            # )
        ),
    ),
    checkpoint=dict(
        save_iter=200,
    ),
)

# Multi-view anomaly gen 14B (Video2World based)
predict2_anomaly_gen_multiview_fsdp_14b = dict(
    defaults=[
        {"override /model": "predict2_anomaly_gen_multiview_fsdp_14b"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=32,
            pipe_config=dict(guardrail_config=dict(enabled=False)),
        )
    ),
    optimizer=dict(
        lr=1e-2,
        weight_decay=1e-6,
        betas=[0.9, 0.99],
        eps=1e-10,
    ),
    scheduler=dict(
        f_max=[0.2],
        f_min=[0.1],
        warm_up_steps=[1_000],
        cycle_lengths=[100_000],
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    job=dict(
        project="posttraining",
        group="anomaly_gen_multiview",
        name="14b_anomaly_gen_multiview",
    ),
    dataloader_train=dataloader_train_anomaly,
    dataloader_val=dataloader_val_anomaly,
    trainer=dict(
        max_iter=200000,
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=200),
            # LogImage is expensive. Disable it for now.
            # log_image=L(LogImage)(
            #     exp_path="results/multiview/trialrun",
            #     every_n=200,
            #     num_steps=35,
            #     guidance=1.5,
            #     seed=1,
            #     is_negative_prompt=True,
            #     n_sample=2
            # )
        ),
    ),
    checkpoint=dict(
        save_iter=200,
    ),
)


for _item in [
    # 2b, anomaly generation (single-view)
    predict2_anomaly_gen_fsdp_2b,
    # 2b, anomaly generation with DDP (single-view)
    predict2_anomaly_gen_ddp_2b,
    # 14b, anomaly generation (single-view)
    predict2_anomaly_gen_fsdp_14b,
    # 14b, anomaly generation with DDP (single-view)
    predict2_anomaly_gen_ddp_14b,
    # 2b, multi-view anomaly generation (Video2World)
    predict2_anomaly_gen_multiview_fsdp_2b,
    # 14b, multi-view anomaly generation (Video2World)
    predict2_anomaly_gen_multiview_fsdp_14b,
]:
    # Get the experiment name from the global variable, e.g. exp01_wan_lora -> experiment_name = "exp01_wan_lora"
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
