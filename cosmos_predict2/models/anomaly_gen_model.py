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

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import math
from typing import Any, Dict, Mapping, Optional, Tuple
import csv

import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn.modules.module import _IncompatibleKeys
import numpy as np
from PIL import Image

from cosmos_predict2.conditioner import DataType, T2VCondition
from cosmos_predict2.configs.anomaly_gen.config_text2image import PREDICT2_TEXT2IMAGE_PIPELINE_2B, Text2ImagePipelineConfig
from cosmos_predict2.networks.model_weights_stats import WeightTrainingStat
from cosmos_predict2.pipelines.anomaly_gen import AnomalyGenPipeline
from cosmos_predict2.utils.checkpointer import non_strict_load_model
from cosmos_predict2.utils.optim_instantiate import get_base_scheduler
from cosmos_predict2.utils.torch_future import clip_grad_norm_
from imaginaire.lazy_config import LazyDict, instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import distributed, log
from cosmos_predict2.models.video2world_model import Predict2Video2WorldModel, Predict2Video2WorldModelConfig
from cosmos_predict2.inference.anomaly_gen.inpaint_condition import AnomalyInpaintCondition
from cosmos_predict2.inference.anomaly_gen.inference_anomaly_diffusion_utils import save_images, inpaint_image
from cosmos_predict2.metrics.utils import compute_kpi, log_kpi_table
from cosmos_predict2.metrics.correspondence import DEFAULT_BACKBONE, prefetch_model as _prefetch_correspondence_model
from cosmos_predict2.data.anomaly_gen.anomaly_dataset import _load_image_and_mask

@attrs.define(slots=False)
class Predict2ModelManagerConfig:
    # Local path, use it in fast debug run
    dit_path: str = "checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt"
    dit_ema_path: str = "checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt"
    # For inference
    text_encoder_path: str = ""  # not used in training.


@attrs.define(slots=False)
class Predict2AnomalyGenModelConfig:
    train_architecture: str = "base"
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: str = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
    init_lora_weights: bool = True

    precision: str = "bfloat16"
    input_data_key: str = "video"
    input_image_key: str = "images"
    loss_reduce: str = "mean"
    loss_scale: float = 10.0

    adjust_video_noise: bool = True

    # This is used for the original way to load models
    model_manager_config: Predict2ModelManagerConfig = Predict2ModelManagerConfig()
    # This is a new way to load models
    pipe_config: Text2ImagePipelineConfig = PREDICT2_TEXT2IMAGE_PIPELINE_2B
    # debug flag
    debug_without_randomness: bool = False
    fsdp_shard_size: int = 0  # 0 means not using fsdp, -1 means set to world size
    # High sigma strategy
    high_sigma_ratio: float = 0.0
    # Use CUDA graphs for DiT blocks during training
    use_cuda_graphs_for_dit: bool = False
    # torch.compile frozen encoders — per-encoder control
    compile_vae_encoder: bool = False
    compile_text_encoder: bool = False
    compile_mask_encoder: bool = False

    # Config for anomaly gen
    ag_config: LazyDict = None

    # HuggingFace model ID for the correspondence scoring backbone.
    # Default: DINOv2 ViT-L/14. Override in your ag_config to switch backbones.
    correspondence_backbone: str = DEFAULT_BACKBONE
    correspondence_top_k: int = 3


class Predict2AnomalyGenModel(Predict2Video2WorldModel):
    def __init__(self, config: Predict2AnomalyGenModelConfig):
        super(Predict2Video2WorldModel, self).__init__() # Calling ImaginaireModel's init method

        self.config = config
        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        self.device = torch.device("cuda")

        # 1. set data keys and data information
        self.setup_data_key()

        # 4. Set up loss options, including loss masking, loss reduce and loss scaling
        self.loss_reduce = getattr(config, "loss_reduce", "mean")
        assert self.loss_reduce in ["mean", "sum"]
        self.loss_scale = getattr(config, "loss_scale", 1.0)
        log.critical(f"Using {self.loss_reduce} loss reduce with loss scale {self.loss_scale}")
        if self.config.adjust_video_noise:
            self.video_noise_multiplier = math.sqrt(self.config.pipe_config.state_t)
        else:
            self.video_noise_multiplier = 1.0
        self.max_adaptive_mask_weight = 100

        # 7. training states
        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

        # New way to init pipe.
        # Pass the ag_config's text encoder (t5_model_name, e.g. t5-large) so the
        # pipeline loads it directly instead of eagerly loading the ~45 GB t5-11b
        # default at from_config and discarding it in from_anomaly_gen_config. This
        # keeps t5-11b off the required-checkpoint set for t5-large runs (matters on
        # fresh / air-gapped installs where t5-11b isn't downloaded).
        t5_model_name = getattr(config.ag_config, "t5_model_name", None)
        from_config_kwargs = (
            {"text_encoder_path": t5_model_name} if t5_model_name is not None else {}
        )
        self.pipe = AnomalyGenPipeline.from_config(
            config.pipe_config,
            dit_path=config.model_manager_config.dit_path,
            **from_config_kwargs,
        )
        # Load anomaly gen components from config
        self.pipe.from_anomaly_gen_config(config.ag_config)
        self.freeze_parameters()

        if config.use_cuda_graphs_for_dit:
            self.pipe.dit.disable_selective_checkpoint()
            log.info("Disabled SAC on DiT for CUDA graph compatibility")

        if config.use_cuda_graphs_for_dit and config.fsdp_shard_size != 0:
            raise RuntimeError(
                "CUDA graphs for DiT (use_cuda_graphs_for_dit=True) is incompatible with FSDP "
                "(fsdp_shard_size != 0). Use DDP (fsdp_shard_size=0) for multi-GPU training "
                "with CUDA graphs."
            )

        if config.train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.dit,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_target_modules=config.lora_target_modules,
                init_lora_weights=config.init_lora_weights,
            )
            if self.pipe.dit_ema:
                self.add_lora_to_model(
                    self.pipe.dit_ema,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_target_modules=config.lora_target_modules,
                    init_lora_weights=config.init_lora_weights,
                )
        else: # Don't train the denoising model
            pass #self.pipe.denoising_model().requires_grad_(True)
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Print the number in billions, or in the format of 1,000,000,000
        log.info(
            f"Total parameters: {total_params / 1e9:.2f}B, Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}"
        )
        log.info(f"Trainable parameters: {[name for name, p in self.named_parameters() if p.requires_grad]}")

        if config.fsdp_shard_size != 0 and torch.distributed.is_initialized():
            if config.fsdp_shard_size == -1:
                fsdp_shard_size = torch.distributed.get_world_size()
                replica_group_size = 1
            else:
                fsdp_shard_size = min(config.fsdp_shard_size, torch.distributed.get_world_size())
                replica_group_size = torch.distributed.get_world_size() // fsdp_shard_size
            dp_mesh = init_device_mesh(
                "cuda", (replica_group_size, fsdp_shard_size), mesh_dim_names=("replicate", "shard")
            )
            log.info(f"Using FSDP with shard size {fsdp_shard_size} | device mesh: {dp_mesh}")
            self.pipe.apply_fsdp(dp_mesh)
        else:
            log.info("FSDP (Fully Sharded Data Parallel) is disabled.")

        # Validation
        self.generated_images_dict = {}
        self.real_images_dict = {}
        self.validation_sample_indice = {}
        
        # Distributed training
        self.world_size = distributed.get_world_size()
        self.rank = distributed.get_rank()

        # Fail fast: load correspondence backbone now so any missing-weights error
        # surfaces at startup rather than on the first validation call.
        _prefetch_correspondence_model(config.correspondence_backbone)

    # We don't use EMA for anomaly gen
    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        self.train()
        for module in [self.net, self.pipe.tokenizer]:
            if module is not None:
                module.to(memory_format=memory_format, **self.tensor_kwargs)

        compile_opts = dict(mode="default", fullgraph=False)
        if self.config.compile_vae_encoder:
            log.info("torch.compile: VAE encoder")
            self.pipe.tokenizer.model.model.encoder = torch.compile(
                self.pipe.tokenizer.model.model.encoder, **compile_opts
            )
        if self.config.compile_text_encoder:
            log.info("torch.compile: T5 text encoder")
            self.pipe.text_encoder.text_encoder = torch.compile(
                self.pipe.text_encoder.text_encoder, **compile_opts
            )
        if self.config.compile_mask_encoder:
            log.info("torch.compile: mask encoder")
            self.pipe.mask_encoder = torch.compile(self.pipe.mask_encoder, **compile_opts)

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        optimizer = instantiate(optimizer_config, model=self)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        return optimizer, scheduler

    # ------------------------ training hooks ------------------------
    def freeze_parameters(self) -> None:
        """ We freeze only predict2 components, not anomaly gen components"""
        # Freeze dit, text_encoder, tokenizer
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder.eval()
        self.pipe.dit.requires_grad_(False)
        self.pipe.dit.eval()
        """ Cosmos Tokenizer's VAE is already frozen in the pipeline"""
        #self.pipe.tokenizer.requires_grad_(False)
        #self.pipe.tokenizer.eval()

    def _get_loss(self, x_gt, x_pred, weights_per_sigma_B_T):
        # extra loss mask for each sample, for example, human faces, hands
        pred_mse_B_C_T_H_W = (x_pred - x_gt) ** 2
        edm_loss_B_C_T_H_W = pred_mse_B_C_T_H_W * rearrange(weights_per_sigma_B_T, "b t -> b 1 t 1 1")
        kendall_loss = edm_loss_B_C_T_H_W
        return pred_mse_B_C_T_H_W, edm_loss_B_C_T_H_W, kendall_loss

    def training_step(self, data_batch: dict, data_batch_idx: int) -> tuple[dict, torch.Tensor]:
        self.pipe.device = self.device

        # Loss
        self._update_train_stats(data_batch)

        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        _, x0_B_C_T_H_W, condition = self.pipe.get_data_and_condition(data_batch)

        # Sample pertubation noise levels and N(0, 1) noises
        sigma_B_T, epsilon_B_C_T_H_W = self.draw_training_sigma_and_epsilon(x0_B_C_T_H_W.size(), condition)

        # Broadcast and split the input data and condition for model parallelism
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = self.pipe.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )
        output_batch, kendall_loss, _, _ = self.compute_loss_with_epsilon_and_sigma(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T, data_batch
        )

        if self.loss_reduce == "mean":
            kendall_loss = kendall_loss.mean() * self.loss_scale
        elif self.loss_reduce == "sum":
            kendall_loss = kendall_loss.sum(dim=1).mean() * self.loss_scale
        else:
            raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")

        return output_batch, kendall_loss

    def compute_loss_with_epsilon_and_sigma(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: T2VCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
        data_batch: dict,
    ) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss given epsilon and sigma
        For anomaly diffusion, give additional focus on masked region (masked loss)

        This method is responsible for computing loss give epsilon and sigma. It involves:
        1. Adding noise to the input data.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            x0: image/video latent
            condition: text condition
            epsilon: noise
            sigma: noise level

        Returns:
            tuple: A tuple containing four elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor 1: kendall loss,
                - Tensor 2: MSE loss,
                - Tensor 3: EDM loss

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        # Get the mean and stand deviation of the marginal probability distribution.
        mean_B_C_T_H_W, std_B_T = x0_B_C_T_H_W, sigma_B_T
        # Generate noisy observations
        xt_B_C_T_H_W = mean_B_C_T_H_W + epsilon_B_C_T_H_W * rearrange(std_B_T, "b t -> b 1 t 1 1")
        # make prediction
        model_pred = self.pipe.denoise(
            xt_B_C_T_H_W, sigma_B_T, condition,
            use_cuda_graphs=self.config.use_cuda_graphs_for_dit,
        )
        # loss weights for different noise levels
        weights_per_sigma_B_T = self.get_per_sigma_loss_weights(sigma=sigma_B_T)
        # extra loss mask for each sample, for example, human faces, hands

        # Calculate overall loss & masked loss
        ## Overall loss
        overall_pred_mse_B_C_T_H_W, overall_edm_loss_B_C_T_H_W, overall_kendall_loss = self._get_loss(x0_B_C_T_H_W, 
                                                                                                                                                                                        model_pred.x0, 
                                                                                                                                                                                        weights_per_sigma_B_T)

        ## Get Guided mask
        guided_masks, adaptive_weight = self.pipe._get_guided_mask_and_weight(x0_B_C_T_H_W,
                                                                                                                                       data_batch['mask'],
                                                                                                                                       self.max_adaptive_mask_weight)

        ## Masked loss
        masked_gt, masked_pred = x0_B_C_T_H_W * guided_masks,  model_pred.x0 * guided_masks
        masked_pred_mse_B_C_T_H_W, masked_edm_loss_B_C_T_H_W, masked_kendall_loss = self._get_loss(masked_gt, 
                                                                                                                                                                                                 masked_pred, 
                                                                                                                                                                                                 weights_per_sigma_B_T)
        # Weighted by adaptive weight
        # For multi-view, adaptive_weight has shape [B, T] -> broadcast to [B, 1, T, 1, 1]
        if adaptive_weight.dim() == 2:
            # Multi-view: [B, T] -> [B, 1, T, 1, 1]
            weight_broadcast = adaptive_weight.view(adaptive_weight.shape[0], 1, adaptive_weight.shape[1], 1, 1)
        else:
            # Single-view: [B] -> [B, 1, 1, 1, 1]
            weight_broadcast = adaptive_weight.view(-1, 1, 1, 1, 1)

        masked_mse = masked_pred_mse_B_C_T_H_W * weight_broadcast
        masked_edm = masked_edm_loss_B_C_T_H_W * weight_broadcast
        masked_kendall = masked_kendall_loss * weight_broadcast

        # Calculate adaptive weight
        full_mse = overall_pred_mse_B_C_T_H_W + masked_mse
        full_edm = overall_edm_loss_B_C_T_H_W + masked_edm
        full_kendall = overall_kendall_loss + masked_kendall

        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigma_B_T,
            "weights_per_sigma": weights_per_sigma_B_T,
            "condition": condition,
            "model_pred": model_pred,
            "mse_loss": full_mse.mean(),
            "edm_loss": full_edm.mean(),
            "edm_loss_per_frame": torch.mean(full_edm, dim=[1, 3, 4]),
        }
        output_batch["loss"] = full_kendall.mean()  # check if this is what we want

        return output_batch, full_kendall, full_mse, full_edm

    # ------------------ Validation ---------------------
    def on_validation_start(self, dataset_dir, anomaly_types, image_size):
        log.info("Start validation")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.load_real_images(dataset_dir, anomaly_types, image_size)
        self.generated_images_dict.clear()
        self.validation_sample_indice.clear()
        self.eval()

    def validation_step(self, data_batch):
        inpaint_condition = AnomalyInpaintCondition(**data_batch)
        # inpaint_condition.num_generated_images is always 1. This is guaranteed
        # by the validation dataloader.

        inpainting_result, anomaly_names, indice = inpaint_image(inpaint_condition, self)

        # Restore indice for filtering duplication
        for i in range(len(anomaly_names)):
            anomaly_name = anomaly_names[i]
            index = indice[i]
            if anomaly_name not in self.validation_sample_indice:
                self.validation_sample_indice[anomaly_name] = []
            self.validation_sample_indice[anomaly_name].append(index)

        # Restore np array of images for KPI computation
        for key, image_list in inpainting_result.items():
            if key in ["annotated_image", "cropped_image", "cropped_mask"]:
                for anomaly_name, instances in zip(anomaly_names, image_list):
                    instances_np = []
                    for image in instances:
                        if isinstance(image, Image.Image):
                            if "mask" in key:
                                image = image.convert("L")
                            elif image.mode != "RGB":
                                image = image.convert("RGB")
                            image = np.array(image).astype(np.float32) / 255.0  # Normalize to 0~1
                            instances_np.append(image)
                        else:
                            raise ValueError(f"Expected a PIL.Image.Image, but got {type(image)} for key '{key}' and anomaly '{anomaly_name}'")
                    if anomaly_name not in self.generated_images_dict:
                        self.generated_images_dict[anomaly_name] = {}
                    if key not in self.generated_images_dict[anomaly_name]:
                        self.generated_images_dict[anomaly_name][key] = []
                    self.generated_images_dict[anomaly_name][key].append(instances_np)
            else:
                for anomaly_name, image in zip(anomaly_names, image_list):
                    if isinstance(image, Image.Image):
                        if "mask" in key:
                            image = image.convert("L")
                        elif image.mode != "RGB":
                            image = image.convert("RGB")
                        image = np.array(image).astype(np.float32) / 255.0  # Normalize to 0~1
                        if anomaly_name not in self.generated_images_dict:
                            self.generated_images_dict[anomaly_name] = {}
                        if key not in self.generated_images_dict[anomaly_name]:
                            self.generated_images_dict[anomaly_name][key] = []
                        self.generated_images_dict[anomaly_name][key].append(image)
                    else:
                        raise ValueError(f"Expected a PIL.Image.Image, but got {type(image)} for key '{key}' and anomaly '{anomaly_name}'")

    def on_validation_end(self, save_dir):
        log.info(f"Start computing validation KPI")

        generated_images_dict_list = [self.generated_images_dict] * self.world_size
        validation_sample_indice_list = [self.validation_sample_indice] * self.world_size

        if self.world_size > 1:
            torch.distributed.all_gather_object(generated_images_dict_list, self.generated_images_dict)
            torch.distributed.barrier()
            torch.distributed.all_gather_object(validation_sample_indice_list, self.validation_sample_indice)
            torch.distributed.barrier()

        # Aggregate generation results
        flat_generated_images_dict_list = {}  # ([rank][anomaly_name][image_key] --> [anomaly_name][image_key])
        for dict_data in generated_images_dict_list:
            for anomaly_name, images_dict in dict_data.items():
                if anomaly_name not in flat_generated_images_dict_list:
                    flat_generated_images_dict_list[anomaly_name] = {}
                for key, image_list in images_dict.items():
                    if key not in flat_generated_images_dict_list[anomaly_name]:
                        flat_generated_images_dict_list[anomaly_name][key] = []
                    flat_generated_images_dict_list[anomaly_name][key].extend(image_list)

        # Aggregate sample indice
        flat_validation_sample_indices = {}  # ([rank][anomaly_name] --> [anomaly_name])
        for dict_data in validation_sample_indice_list:
            for anomaly_name, indice in dict_data.items():
                if anomaly_name not in flat_validation_sample_indices:
                    flat_validation_sample_indices[anomaly_name] = []
                flat_validation_sample_indices[anomaly_name].extend(indice)

        # Filter duplicates using sample indices
        seen_sample_indices = set()
        filtered_generated_images_dict_list = {}
        for anomaly_name in flat_generated_images_dict_list.keys():
            if anomaly_name not in filtered_generated_images_dict_list:
                filtered_generated_images_dict_list[anomaly_name] = {}
            indice = flat_validation_sample_indices[anomaly_name]
            for i in range(len(indice)):
                sample_index = indice[i]
                if sample_index in seen_sample_indices:
                    continue
                seen_sample_indices.add(sample_index)
                for key, items in flat_generated_images_dict_list[anomaly_name].items():
                    if key not in filtered_generated_images_dict_list[anomaly_name]:
                        filtered_generated_images_dict_list[anomaly_name][key] = []
                    filtered_generated_images_dict_list[anomaly_name][key].append(items[i])

        valid_kpi = None
        if self.rank == 0:
            real_images_dict, generated_images_dict = self.real_images_dict, filtered_generated_images_dict_list
            valid_kpi = compute_kpi(
                real_images_dict, generated_images_dict,
                correspondence_backbone=self.config.correspondence_backbone,
                correspondence_top_k=self.config.correspondence_top_k,
            )

            # Log KPI per anomaly and type
            log_kpi_table(valid_kpi)

            log.info(f"Start saving validation results to {save_dir}")
            # Save all collected images — threaded because PNG encode + disk I/O dominates here.
            with ThreadPoolExecutor(max_workers=16) as pool:
                for anomaly_name, images_dict in generated_images_dict.items():
                    _dump_images(images_dict, save_dir, anomaly_name, pool)

                for anomaly_name, images_dict in real_images_dict.items():
                    _dump_images(images_dict, save_dir, f"real_{anomaly_name}", pool)
            
            # Save kpi
            save_path = os.path.join(save_dir, f"valid_kpi.csv")
            with open(save_path, 'a', newline='') as f:
                writer = csv.writer(f)
                anomaly_names = sorted(list(real_images_dict.keys()))
                writer.writerow(["kpi"] + anomaly_names + ["Average"])  # header
                kpi_types = sorted(valid_kpi["Average"].keys())
                for kpi_type in kpi_types:
                    writer.writerow([kpi_type] + [valid_kpi[name][kpi_type] for name in anomaly_names] + [valid_kpi["Average"][kpi_type]])

        if self.world_size > 1:
            torch.distributed.barrier()
        log.info("Valdation finished successfully")

        return valid_kpi

    def load_real_images(self, dataset_dir, anomaly_types, image_size):
        for (texture, anomaly_type) in anomaly_types:
            anomaly_name = f"{texture}+{anomaly_type}"
            self.real_images_dict[anomaly_name] = {}

            image_dir = os.path.join(dataset_dir, texture, 'anomaly_image', anomaly_type)
            mask_dir = os.path.join(dataset_dir, texture, 'mask', anomaly_type)

            # Remove Thumbs.db
            if os.path.isfile(f"{image_dir}/Thumbs.db"):
                os.remove(f"{image_dir}/Thumbs.db")
            if os.path.isfile(f"{mask_dir}/Thumbs.db"):
                os.remove(f"{mask_dir}/Thumbs.db")

            anomaly_image_files = sorted(os.listdir(image_dir))
            mask_image_files = []
            try:
                for anomaly_image in anomaly_image_files:
                    extension = anomaly_image.split(".")[-1]
                    mask_image_file = anomaly_image.replace(f".{extension}", f"_mask.{extension}")
                    mask_image_files.append(mask_image_file)
            except Exception as e:
                raise RuntimeError(f"Error: {e}")

            anomaly_image_files=[os.path.join(image_dir, file_name) for file_name in anomaly_image_files]
            mask_image_files=[os.path.join(mask_dir, file_name) for file_name in mask_image_files]

            self.real_images_dict[anomaly_name]["original_image"] = []
            self.real_images_dict[anomaly_name]["original_mask"] = []

            for anomaly_image_file, mask_image_file in zip(anomaly_image_files, mask_image_files):
                image = Image.open(anomaly_image_file).convert("RGB")
                mask = Image.open(mask_image_file).convert("L")
                np_mask = np.array(mask)
                if np_mask.max() != 255:
                    raise ValueError(f"Error: Mask {mask_image_file} is not binary!")
                assert image.size == mask.size,  f"Error: Image filename {anomaly_image_file} 's size with mask filename {mask_image_file}"
                image = np.array(image).astype(np.float32)
                mask = np.array(mask).astype(np.float32)
                image= image / 255 # 0~1
                mask = mask / 255.0
                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = 1

                self.real_images_dict[anomaly_name]["original_image"].append(image)
                self.real_images_dict[anomaly_name]["original_mask"].append(mask)

    # ------------------ Checkpointing ------------------
    def state_dict(self) -> Dict[str, Any]:
        # the checkpoint format should be compatible with traditional imaginaire4
        # pipeline contains both net and net_ema
        # checkpoint should be saved/loaded from Model
        # checkpoint should be loadable from pipeline as well - We don't use Model for inference only jobs.

        state_dict = {}

        # Add mask encoder's weight if it's trainable
        if not self.config.ag_config.mask_encoder.freeze:
            prefix = "pipe.mask_encoder."
            mask_encoder_state_dict = self.pipe.mask_encoder.state_dict()
            for key, val in mask_encoder_state_dict.items():
                state_dict[prefix + key] = val

        # Add anomaly embedding if its trainable
        if not self.config.ag_config.anomaly_embedding.freeze:
            prefix = "pipe.anomaly_embedding."
            anomaly_embedding_state_dict = self.pipe.anomaly_embedding.state_dict()
            for key, val in anomaly_embedding_state_dict.items():
                state_dict[prefix + key] = val

        # Add adapter if its trainable
        if not self.config.ag_config.adapter.freeze:
            prefix = "pipe.adapter."
            adapter_state_dict = self.pipe.adapter.state_dict()
            for key, val in adapter_state_dict.items():
                state_dict[prefix + key] = val

        # convert DTensor to Tensor
        for key, val in state_dict.items():
            if isinstance(val, DTensor):
                # Convert to full tensor
                state_dict[key] = val.full_tensor().detach().cpu()
            else:
                state_dict[key] = val.detach().cpu()

        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """
        Loads a state dictionary into the model and optionally its EMA counterpart.
        Different from torch strict=False mode, the method will not raise error for unmatched state shape while raise warning.

        Parameters:e
            state_dict (Mapping[str, Any]): A dictionary containing separate state dictionaries for the model and
                                            potentially for an EMA version of the model under the keys 'model' and 'ema', respectively.
            strict (bool, optional): If True, the method will enforce that the keys in the state dict match exactly
                                    those in the model and EMA model (if applicable). Defaults to True.
            assign (bool, optional): If True and in strict mode, will assign the state dictionary directly rather than
                                    matching keys one-by-one. This is typically used when loading parts of state dicts
                                    or using customized loading procedures. Defaults to False.
        """
        missing_keys, unexpected_keys = [], []

        # Add mask encoder's weight if it's trainable
        if not self.config.ag_config.mask_encoder.freeze:
            mask_encoder_state_dict = {k.replace("pipe.mask_encoder.", ""): v for k, v in state_dict.items() if "pipe.mask_encoder." in k}
            mask_encoder_results: _IncompatibleKeys = self.pipe.mask_encoder.load_state_dict(
                mask_encoder_state_dict, strict=strict, assign=assign
            )
            missing_keys += mask_encoder_results.missing_keys
            unexpected_keys += mask_encoder_results.unexpected_keys

        # Add anomaly embedding if its trainable
        if not self.config.ag_config.anomaly_embedding.freeze:
            anomaly_embedding_state_dict = {k.replace("pipe.anomaly_embedding.", ""): v for k, v in state_dict.items() if "pipe.anomaly_embedding." in k}
            anomaly_embedding_results: _IncompatibleKeys = self.pipe.anomaly_embedding.load_state_dict(
                anomaly_embedding_state_dict, strict=strict, assign=assign
            )
            missing_keys += anomaly_embedding_results.missing_keys
            unexpected_keys += anomaly_embedding_results.unexpected_keys

        # Add adapter if its trainable
        if not self.config.ag_config.adapter.freeze:
            adapter_state_dict = {k.replace("pipe.adapter.", ""): v for k, v in state_dict.items() if "pipe.adapter." in k}
            adapter_results: _IncompatibleKeys = self.pipe.adapter.load_state_dict(
                adapter_state_dict, strict=strict, assign=assign
            )
            missing_keys += adapter_results.missing_keys
            unexpected_keys += adapter_results.unexpected_keys

        if strict:
            return _IncompatibleKeys(
                missing_keys=missing_keys,
                unexpected_keys=unexpected_keys,
            )
            
        else:
            raise NotImplementedError("Non-strict mode is not supported for anomaly gen")
            log.critical("load model in non-strict mode")
            log.critical(non_strict_load_model(self.pipe.dit, _reg_state_dict), rank0_only=False)
            if self.config.pipe_config.ema.enabled:
                log.critical("load ema model in non-strict mode")
                log.critical(non_strict_load_model(self.pipe.dit_ema, _ema_state_dict), rank0_only=False)

def _ensure_dir(root, subdir):
    path = os.path.join(root, subdir)
    os.makedirs(path, exist_ok=True)
    return path

def _save_png(image, path):
    Image.fromarray((image * 255).clip(0, 255).astype(np.uint8)).save(path, compress_level=1)

def _save_sequence(images, out_dir, prefix, pool):
    for idx, image in enumerate(images):
        pool.submit(_save_png, image, os.path.join(out_dir, f"{prefix}_{idx:05d}.png"))

def _save_instance_groups(flat_crops, counts, out_dir, prefix, pool):
    offset = 0
    for img_idx, count in enumerate(counts):
        for inst_idx in range(count):
            crop = flat_crops[offset + inst_idx]
            pool.submit(
                _save_png, crop, os.path.join(out_dir, f"{prefix}_{img_idx:05d}_{inst_idx:05d}.png")
            )
        offset += count

def _dump_images(image_dict, save_root, prefix, pool):
    for key, payload in image_dict.items():
        if key in ("annotated_image", "cropped_image", "cropped_mask"):
            out_dir = _ensure_dir(save_root, key)
            for idx, instances in enumerate(payload):
                _save_sequence(instances, out_dir, f"{prefix}_{idx:05d}", pool)
        elif key == "mask_cropped_image":
            out_dir = _ensure_dir(save_root, key)
            counts = image_dict.get("num_instance", [])
            _save_instance_groups(payload, counts, out_dir, prefix, pool)
        elif key in ("reconstructed_image", "original_image", "original_mask"):  # reconstructed / original_* and any other per-image arrays
            out_dir = _ensure_dir(save_root, key)
            _save_sequence(payload, out_dir, prefix, pool)
