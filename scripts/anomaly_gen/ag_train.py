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

import argparse
import importlib
import os
import yaml
import shutil

from loguru import logger as logging

from imaginaire.config import Config, pretty_print_overrides
from imaginaire.lazy_config import instantiate
from imaginaire.lazy_config.lazy import LazyConfig
from imaginaire.utils import distributed
from imaginaire.utils.config_helper import get_config_module, override
from imaginaire.lazy_config import LazyCall as L
from cosmos_predict2.data.anomaly_gen.anomaly_dataset import Dataset, AnomalyInpaintValidationDataset
from cosmos_predict2.data.anomaly_gen.multiview_anomaly_dataset import MultiViewDataset, MultiViewAnomalyInpaintValidationDataset
from cosmos_predict2.configs.base.experiment.anomaly_gen import get_sampler

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_nested_attributes(obj, attrs):
    """
    Dynamically sets nested attributes on an existing class instance based on a nested dictionary.
    Assumes all necessary nested attribute structures already exist.

    Args:
        obj: The instance of the existing class.
        attrs (dict): A nested dictionary containing the attributes to set.
    """
    for key, value in attrs.items():
        if isinstance(value, dict):
            # Recursively set attributes for the nested dictionary
            # No need to check if attribute exists since it's guaranteed
            set_nested_attributes(getattr(obj, key), value)
        else:
            # Set the attribute directly
            setattr(obj, key, value)

@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    # Need to initialize the distributed environment before calling config.validate() because it tries to synchronize
    # a buffer across ranks. If you don't do this, then you end up allocating a bunch of buffers on rank 0, and also that
    # check doesn't actually do anything.
    distributed.init()

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)

    # Create dataset
    # Check if multi-view dataset should be used (when view_types is specified)
    view_types = getattr(config.dataloader_train.dataset, 'view_types', None)
    num_views = len(view_types) if view_types is not None else 1
    
    if view_types is not None and num_views > 1:
        # Multi-view dataset
        logging.info(f"Using MultiViewDataset with {num_views} views: {view_types}")
        train_anomaly_dataset = L(MultiViewDataset)(
            dataset_dir=config.dataloader_train.dataset.dataset_dir,
            anomaly_types=config.dataloader_train.dataset.anomaly_types,
            view_types=config.dataloader_train.dataset.view_types,
            image_size=config.dataloader_train.dataset.image_size,
            num_frames=num_views,
            seed=config.dataloader_train.dataset.seed,
            data_augprob=config.dataloader_train.dataset.data_augprob,
            aug_type=getattr(config.dataloader_train.dataset, 'aug_type', None),
            ratio_range=getattr(config.dataloader_train.dataset, 'ratio_range', None),
        )
        # Use MultiViewAnomalyInpaintValidationDataset for multi-view training
        logging.info("Using MultiViewAnomalyInpaintValidationDataset for validation")
        val_anomaly_dataset = L(MultiViewAnomalyInpaintValidationDataset)(
            input_data_path = config.dataloader_val.dataset.input_data_path,
        )
    else:
        # Single-view dataset (original)
        logging.info("Using single-view Dataset")
        train_anomaly_dataset = L(Dataset)(
            dataset_dir=config.dataloader_train.dataset.dataset_dir,
            num_frames=1,
            anomaly_types=config.dataloader_train.dataset.anomaly_types,
            image_size=config.dataloader_train.dataset.image_size,
            seed=config.dataloader_train.dataset.seed,
            data_augprob=config.dataloader_train.dataset.data_augprob,
            aug_type=getattr(config.dataloader_train.dataset, 'aug_type', None),
            ratio_range=getattr(config.dataloader_train.dataset, 'ratio_range', None),
            random_rotation90=getattr(config.dataloader_train.dataset, 'random_rotation90', False),
        )
        # Use AnomalyInpaintValidationDataset for single-view training
        logging.info("Using AnomalyInpaintValidationDataset for validation")
        val_anomaly_dataset = L(AnomalyInpaintValidationDataset)(
            input_data_path = config.dataloader_val.dataset.input_data_path,
        )
    config.dataloader_train.dataset = train_anomaly_dataset
    config.dataloader_val.dataset = val_anomaly_dataset

    use_cuda_graphs = getattr(config.model.config, 'use_cuda_graphs_for_dit', False)
    # When batch_size=1 (the default), BatchAlignedDistributedSampler reduces to
    # the base DistributedSampler because ceil(N/1)*1 == N -- no extra padding.
    # We only align to the real batch_size when CUDA graphs need fixed-shape batches.
    sampler_batch_size = config.dataloader_train.batch_size if use_cuda_graphs else 1
    config.dataloader_train.sampler = L(get_sampler)(
        dataset=train_anomaly_dataset, batch_size=sampler_batch_size,
    )
    config.dataloader_val.sampler = L(get_sampler)(
        dataset=val_anomaly_dataset, shuffle=False,
    )
    # Create the dataloaders.

    dataloader_train = instantiate(config.dataloader_train)
    dataloader_train.collate_fn = dataloader_train.dataset._collate_fn

    # Validation data loader is required for training
    dataloader_val = instantiate(config.dataloader_val)
    dataloader_val.collate_fn = dataloader_val.dataset._collate_fn

    # Create the model
    model = instantiate(config.model)
    # Start training
    trainer.train(
        model,
        dataloader_train,
        dataloader_val,
    )


if __name__ == "__main__":
    # Usage: torchrun --nproc_per_node=1 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiments=predict2_video2world_training_2b_cosmos_nemo_assets
    # Get the config file from the input arguments.
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", help="Path to the config file", required=True)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--ag_config",
        help="YAML file config for anomaly diffusion training.",
    )
    args = parser.parse_args()
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)

    # override config with ag_config's information
    # Load yaml
    with open(args.ag_config) as fp:
        try:
            ag_config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"[ERROR] Cannot load {args.ag_config} file! Exception: {exc}")

    # override
    set_nested_attributes(config, ag_config)

    # Copy an ad_config file to result's directory
    os.makedirs(config.job.path_local, exist_ok=True)
    shutil.copyfile(args.ag_config, f"{config.job.path_local}/ag_config.yaml")
    LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")

    # Launch the training job.
    launch(config, args)
