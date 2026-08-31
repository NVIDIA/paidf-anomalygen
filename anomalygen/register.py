# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single registration entry point for the plugin.

Importing this module performs every ConfigStore registration the plugin needs (model
group, ckpt_type, callbacks). The experiment recipe is registered separately by the
launcher, not here.
"""

from __future__ import annotations

from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.utils.lazy_config import LazyCall as L
from hydra.core.config_store import ConfigStore

from anomalygen.callbacks.early_stop import EarlyStop
from anomalygen.callbacks.training_report import TrainingReport
from anomalygen.callbacks.validation_kpi import ValidationKPI
from anomalygen.checkpoint.selective_dcp import SelectiveCheckpointer
from anomalygen.checkpoint.trained_keys import TRAINED_KEY_PREFIXES
from anomalygen.configs.texture.model_config import AnomalyGenTextureDiffusionExpertConfig
from anomalygen.models.texture.model import AnomalyGenTextureMoTModel

_registered = False


def register_anomalygen_texture_ft(cs: ConfigStore) -> None:
    """Register the AnomalyGen texture fine-tuning task's ConfigStore groups (model, ckpt_type, callbacks)."""
    # Model.
    anomalygen_texture_ft_mot_fsdp = dict(
        trainer=dict(distributed_parallelism="fsdp"),
        model=L(AnomalyGenTextureMoTModel)(
            config=OmniMoTModelConfig(
                # -1 auto-selects the FSDP shard degree from the world size.
                parallelism=ParallelismConfig(data_parallel_shard_degree=-1),
                diffusion_expert_config=AnomalyGenTextureDiffusionExpertConfig(),
            ),
            _recursive_=False,
        ),
    )
    cs.store(
        group="model", package="_global_", name="anomalygen_texture_ft_mot_fsdp", node=anomalygen_texture_ft_mot_fsdp
    )
    anomalygen_texture_ft_mot_ddp = dict(
        trainer=dict(distributed_parallelism="ddp"),
        model=L(AnomalyGenTextureMoTModel)(
            config=OmniMoTModelConfig(
                diffusion_expert_config=AnomalyGenTextureDiffusionExpertConfig(),
            ),
            _recursive_=False,
        ),
    )
    cs.store(
        group="model", package="_global_", name="anomalygen_texture_ft_mot_ddp", node=anomalygen_texture_ft_mot_ddp
    )

    # Checkpoint type.
    cs.store(
        group="ckpt_type",
        package="checkpoint.type",
        name="selective_dcp_anomalygen_texture_ft",
        # Save exactly the trained subset (same single source as the optimizer's keys_to_select).
        node=L(SelectiveCheckpointer)(save_keys_filter=list(TRAINED_KEY_PREFIXES)),
    )

    # Callbacks.
    cs.store(
        group="callbacks",
        package="trainer.callbacks",
        name="anomalygen_texture_ft",
        node=dict(
            validation_kpi=L(ValidationKPI)(),
            early_stop=L(EarlyStop)(),
            training_report=L(TrainingReport)(),
        ),
    )


def register_all() -> None:
    global _registered
    if _registered:
        return
    cs = ConfigStore.instance()
    register_anomalygen_texture_ft(cs)
    _registered = True


register_all()
