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
Multi-view Anomaly Generation Model based on Video2World architecture.
"""

import math
from typing import Any, Dict, Mapping, Optional, Tuple
import os
import numpy as np
from PIL import Image
import csv
import gc

import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_predict2.conditioner import DataType, T2VCondition
from cosmos_predict2.configs.anomaly_gen.config_video2world import (
    PREDICT2_ANOMALY_GEN_MULTIVIEW_PIPELINE_2B,
    AnomalyGenVideo2WorldPipelineConfig,
)
from cosmos_predict2.pipelines.anomaly_gen_multiview import AnomalyGenMultiViewPipeline
from imaginaire.lazy_config import LazyDict
from imaginaire.utils import distributed, log
from cosmos_predict2.models.video2world_model import Predict2Video2WorldModel
from cosmos_predict2.models.anomaly_gen_model import Predict2AnomalyGenModel
from cosmos_predict2.inference.anomaly_gen.multiview_inpaint_condition import MultiViewAnomalyInpaintCondition
from cosmos_predict2.inference.anomaly_gen.multiview_inference_utils import inpaint_multiview_image
from cosmos_predict2.metrics.utils import compute_kpi_per_view_multiview, log_kpi_table_per_view
from cosmos_predict2.metrics.correspondence import DEFAULT_BACKBONE, prefetch_model as _prefetch_correspondence_model
from cosmos_predict2.models.anomaly_gen_model import _dump_images


@attrs.define(slots=False)
class Predict2MultiViewModelManagerConfig:
    # Local path for checkpoints - use Text2Image checkpoint (frames encoded independently)
    dit_path: str = "checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt"
    dit_ema_path: str = "checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt"
    text_encoder_path: str = ""


@attrs.define(slots=False)
class Predict2AnomalyGenMultiViewModelConfig:
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

    model_manager_config: Predict2MultiViewModelManagerConfig = Predict2MultiViewModelManagerConfig()
    pipe_config: AnomalyGenVideo2WorldPipelineConfig = PREDICT2_ANOMALY_GEN_MULTIVIEW_PIPELINE_2B
    debug_without_randomness: bool = False
    fsdp_shard_size: int = 0
    high_sigma_ratio: float = 0.0

    # Config for anomaly gen
    ag_config: LazyDict = None

    # HuggingFace model ID for the correspondence scoring backbone.
    # Default: DINOv2 ViT-L/14. Override in your ag_config to switch backbones.
    correspondence_backbone: str = DEFAULT_BACKBONE


class Predict2AnomalyGenMultiViewModel(Predict2AnomalyGenModel):
    """
    Multi-view Anomaly Generation Model using Video2World pipeline.
    Inherits from Predict2AnomalyGenModel to leverage all anomaly-specific logic
    (masked loss, validation, checkpointing, etc.).
    Only overrides the parts that are different for multi-view.
    """
    
    def __init__(self, config: Predict2AnomalyGenMultiViewModelConfig):
        # Skip both Predict2AnomalyGenModel and Predict2Video2WorldModel's __init__
        # to avoid initializing their pipelines. Call ImaginaireModel's __init__ directly.
        super(Predict2Video2WorldModel, self).__init__()  # Calling ImaginaireModel's init method
        
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
        
        # New way to init pipe - Use Multi-View Pipeline instead of Single-View Pipeline
        self.pipe = AnomalyGenMultiViewPipeline.from_config(
            config.pipe_config,
            dit_path=config.model_manager_config.dit_path,
        )
        # Load anomaly gen components from config
        self.pipe.from_anomaly_gen_config(config.ag_config)
        self.freeze_parameters()
        
        # Don't train the denoising model (same as parent but without LoRA support for now)
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

    def load_real_images(self, dataset_dir, anomaly_types, image_size):
        """
        Override to handle multi-view validation with shared masks.
        """        
        for sample_name, anomaly_type in anomaly_types:
            # For multi-view, the directory structure is:
            # dataset_dir/sample_name/anomaly_image/anomaly_type/*.png
            # dataset_dir/sample_name/mask/anomaly_type/*_mask.png
            # Use the same key format as single-view: "sample_name+anomaly_type"
            anomaly_name = f"{sample_name}+{anomaly_type}"
            image_dir = os.path.join(dataset_dir, sample_name, "anomaly_image", anomaly_type)
            mask_dir = os.path.join(dataset_dir, sample_name, "mask", anomaly_type)
            
            self.real_images_dict[anomaly_name] = {}
            
            if os.path.isfile(f"{image_dir}/Thumbs.db"):
                os.remove(f"{image_dir}/Thumbs.db")
            if os.path.isfile(f"{mask_dir}/Thumbs.db"):
                os.remove(f"{mask_dir}/Thumbs.db")
            
            # For multi-view, we need to find the base image files (without view suffix)
            # The naming pattern is: {id}_{anomaly}_view{X}.png for images
            # and {id}_{anomaly}_mask.png for masks (shared across views)
            all_image_files = sorted(os.listdir(image_dir))
            
            # Extract unique base names by removing view suffixes
            # e.g., "25_bumps_view0.png" -> "25_bumps"
            base_names = set()
            for img_file in all_image_files:
                # Remove extension
                name_without_ext = os.path.splitext(img_file)[0]
                # Remove view suffix (e.g., "_view0", "_view1", etc.)
                # Pattern: ends with _viewX where X is a number
                import re
                base_name = re.sub(r'_view\d+$', '', name_without_ext)
                base_names.add(base_name)
            
            base_names = sorted(list(base_names))
            
            self.real_images_dict[anomaly_name]["original_image"] = []
            self.real_images_dict[anomaly_name]["original_mask"] = []
            
            for base_name in base_names:
                # For multi-view, we need to load ALL views for each base_name
                # Find all view images for this base_name
                import re
                view_image_files = []
                for filename in sorted(os.listdir(image_dir)):
                    # Match pattern: {base_name}_view{number}.{ext}
                    pattern = re.compile(rf"^{re.escape(base_name)}_view(\d+)\.(png|jpg|jpeg|PNG|JPG)$")
                    match = pattern.match(filename)
                    if match:
                        view_num = int(match.group(1))
                        view_image_files.append((view_num, os.path.join(image_dir, filename)))
                
                if not view_image_files:
                    log.warning(f"Could not find any view images for base name {base_name} in {image_dir}")
                    continue
                
                # Sort by view number
                view_image_files.sort(key=lambda x: x[0])
                
                # Find the mask(s) - could be shared or per-view
                # First try shared mask
                shared_mask_file = None
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                    candidate = os.path.join(mask_dir, f"{base_name}_mask{ext}")
                    if os.path.exists(candidate):
                        shared_mask_file = candidate
                        break
                
                # If no shared mask, try per-view masks
                per_view_mask_files = {}
                if shared_mask_file is None:
                    for filename in sorted(os.listdir(mask_dir)):
                        # Match pattern: {base_name}_view{number}_mask.{ext}
                        pattern = re.compile(rf"^{re.escape(base_name)}_view(\d+)_mask\.(png|jpg|jpeg|PNG|JPG)$")
                        match = pattern.match(filename)
                        if match:
                            view_num = int(match.group(1))
                            per_view_mask_files[view_num] = os.path.join(mask_dir, filename)
                
                # Load images and masks for all views
                for view_num, view_image_file in view_image_files:
                    # Determine mask file for this view
                    if shared_mask_file:
                        mask_file = shared_mask_file
                    elif view_num in per_view_mask_files:
                        mask_file = per_view_mask_files[view_num]
                    else:
                        log.warning(f"Could not find mask for base name {base_name}, view {view_num} in {mask_dir}")
                        continue
                    
                    try:
                        image = Image.open(view_image_file)
                        mask = Image.open(mask_file).convert("L")
                        np_mask = np.array(mask)
                        
                        # Check if mask is binary (allow all-black masks for validation)
                        unique_values = np.unique(np_mask)
                        if not np.all(np.isin(unique_values, [0, 255])):
                            log.warning(f"Mask {mask_file} is not binary! Unique values: {unique_values}")
                            continue
                        
                        assert image.size == mask.size, f"Error: Image {view_image_file} size {image.size} != mask {mask_file} size {mask.size}"
                        
                        if not image.mode == "RGB":
                            image = image.convert("RGB")
                        
                        image = np.array(image).astype(np.float32)
                        mask = np.array(mask).astype(np.float32)
                        image = image / 255  # 0~1
                        mask = mask / 255.0
                        mask[mask < 0.5] = 0
                        mask[mask >= 0.5] = 1
                        
                        # Append image and mask for this view
                        self.real_images_dict[anomaly_name]["original_image"].append(image)
                        self.real_images_dict[anomaly_name]["original_mask"].append(mask)
                        
                    except Exception as e:
                        log.error(f"Error loading image/mask for {base_name}, view {view_num}: {e}")
                        continue
    
    def validation_step(self, data_batch):
        """
        Override to handle multi-view validation using MultiViewAnomalyInpaintCondition.
        """
        # Create MultiViewAnomalyInpaintCondition from data_batch
        inpaint_condition = MultiViewAnomalyInpaintCondition(**data_batch)
        
        # Run multi-view inpainting
        inpainting_result, anomaly_names, indice = inpaint_multiview_image(inpaint_condition, self)
        
        # Restore indice for filtering duplication
        # For multi-view, each sample has num_views views, so we need to save the sample_index for each view
        # In validation, B=1, so anomaly_names has only one element
        anomaly_name = anomaly_names[0] if len(anomaly_names) > 0 else "unknown"
        sample_index = indice[0] if len(indice) > 0 else None
        
        # Get num_views from the pipeline
        num_views = self.pipe.num_views
        
        # For each view, save the sample_index (needed for deduplication)
        if anomaly_name not in self.validation_sample_indice:
            self.validation_sample_indice[anomaly_name] = []
        # Append sample_index for each view (num_views times)
        for _ in range(num_views):
            self.validation_sample_indice[anomaly_name].append(sample_index)
        
        # Debug: log what we're saving
        log.debug(f"[validation_step] {anomaly_name}: sample_index={sample_index}, saving {num_views} views")
        
        # Restore np array of images for KPI computation
        # For multi-view, we save ALL views separately (each view as an independent entry)
        # 
        # Data structure from inpaint_multiview_image:
        # - For "cropped_image", "cropped_mask", "annotated_image": [num_views][B][instances]
        #   In validation, B=1, so it's [num_views][1][instances]
        # - For other keys: [num_views][B]
        #   In validation, B=1, so it's [num_views][1]
        for key, image_list in inpainting_result.items():
            if key in ["annotated_image", "cropped_image", "cropped_mask"]:
                # image_list is [num_views][B][instances]
                # In validation, B=1, so we iterate over views
                # Each view's data should be saved separately
                for view_idx, view_instances_list in enumerate(image_list):
                    # view_instances_list is [B][instances], where B=1 in validation
                    # Get the instances for batch_idx=0 (validation always has B=1)
                    if len(view_instances_list) > 0:
                        instances = view_instances_list[0]  # [instances] - a list of PIL.Image (one per instance)
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
                                raise ValueError(f"Expected a PIL.Image.Image, but got {type(image)} for key '{key}', view {view_idx}")
                        
                        # Save each view's instances separately
                        if anomaly_name not in self.generated_images_dict:
                            self.generated_images_dict[anomaly_name] = {}
                        if key not in self.generated_images_dict[anomaly_name]:
                            self.generated_images_dict[anomaly_name][key] = []
                        self.generated_images_dict[anomaly_name][key].append(instances_np)
            else:
                # For other keys (e.g., "reconstructed_image", "original_image", "original_mask"):
                # image_list is [num_views][B], where B=1 in validation
                # We need to iterate over all views and save each view separately
                for view_idx, view_images in enumerate(image_list):
                    # view_images is [B], where B=1 in validation
                    # Get the image for batch_idx=0
                    if len(view_images) > 0:
                        view_image = view_images[0]  # PIL.Image for this view
                        if isinstance(view_image, Image.Image):
                            if "mask" in key:
                                view_image = view_image.convert("L")
                            elif view_image.mode != "RGB":
                                view_image = view_image.convert("RGB")
                            view_image_np = np.array(view_image).astype(np.float32) / 255.0  # Normalize to 0~1
                            if anomaly_name not in self.generated_images_dict:
                                self.generated_images_dict[anomaly_name] = {}
                            if key not in self.generated_images_dict[anomaly_name]:
                                self.generated_images_dict[anomaly_name][key] = []
                            self.generated_images_dict[anomaly_name][key].append(view_image_np)
                        else:
                            raise ValueError(f"Expected a PIL.Image.Image, but got {type(view_image)} for key '{key}', view {view_idx}")
        
        # Debug: log how many images we've saved after this step
        if "reconstructed_image" in self.generated_images_dict.get(anomaly_name, {}):
            num_saved = len(self.generated_images_dict[anomaly_name]["reconstructed_image"])
            num_indice = len(self.validation_sample_indice.get(anomaly_name, []))
            log.debug(f"[validation_step] After saving: {anomaly_name} has {num_saved} images, {num_indice} indices")
    
    def on_validation_end(self, save_dir):
        """
        Override to use compute_kpi_multiview instead of compute_kpi.
        This handles empty masks gracefully, which can occur in multi-view validation.
        """
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
        
        # Debug: log aggregated data before deduplication
        if self.rank == 0:
            for anomaly_name in flat_generated_images_dict_list.keys():
                if "reconstructed_image" in flat_generated_images_dict_list[anomaly_name]:
                    num_images = len(flat_generated_images_dict_list[anomaly_name]["reconstructed_image"])
                    num_indice = len(flat_validation_sample_indices.get(anomaly_name, []))
                    log.info(f"[{anomaly_name}] Before deduplication: {num_images} images, {num_indice} indices")

        # Filter duplicates using sample indices
        # For multi-view, each sample has num_views views, and each view has its own entry
        # We need to deduplicate by (sample_index, view_idx) pairs
        # Since indice is [sample_0, sample_0, ..., sample_0, sample_1, ..., sample_1] (each sample_index repeated num_views times)
        # We can use (sample_index, i % num_views) as the unique key
        num_views = self.pipe.num_views
        seen_indices = set()  # Track seen (sample_index, view_idx) pairs
        filtered_generated_images_dict_list = {}
        for anomaly_name in flat_generated_images_dict_list.keys():
            if anomaly_name not in filtered_generated_images_dict_list:
                filtered_generated_images_dict_list[anomaly_name] = {}
            indice = flat_validation_sample_indices[anomaly_name]
            
            # Debug: log the structure
            if len(indice) > 0:
                log.info(f"[{anomaly_name}] Deduplication: indice length={len(indice)}, num_views={num_views}, expected samples={len(indice) // num_views}")
            
            for i in range(len(indice)):
                sample_index = indice[i]
                # Calculate view_idx: since indice is [sample_0 (×6), sample_1 (×6), ...]
                # view_idx = i % num_views gives us the view index within the current sample
                view_idx = i % num_views
                unique_key = (sample_index, view_idx)
                
                if unique_key in seen_indices:
                    continue
                seen_indices.add(unique_key)
                for key, items in flat_generated_images_dict_list[anomaly_name].items():
                    if key not in filtered_generated_images_dict_list[anomaly_name]:
                        filtered_generated_images_dict_list[anomaly_name][key] = []
                    filtered_generated_images_dict_list[anomaly_name][key].append(items[i])
        
        # Debug: log filtered data after deduplication
        if self.rank == 0:
            for anomaly_name in filtered_generated_images_dict_list.keys():
                if "reconstructed_image" in filtered_generated_images_dict_list[anomaly_name]:
                    num_images = len(filtered_generated_images_dict_list[anomaly_name]["reconstructed_image"])
                    log.info(f"[{anomaly_name}] After deduplication: {num_images} images")
        
        if self.rank == 0:
            real_images_dict, generated_images_dict = self.real_images_dict, filtered_generated_images_dict_list
            # Get num_views from pipeline
            num_views = self.pipe.num_views
            
            # Use compute_kpi_per_view_multiview for per-view KPI computation
            valid_kpi = compute_kpi_per_view_multiview(real_images_dict, generated_images_dict, num_views)

            # Log KPI per view, per anomaly and type
            log_kpi_table_per_view(valid_kpi, num_views)

            log.info(f"Start saving validation results to {save_dir}")
            # Save all collected images
            for anomaly_name, images_dict in filtered_generated_images_dict_list.items():
                _dump_images(images_dict, save_dir, anomaly_name)

            for anomaly_name, images_dict in real_images_dict.items():
                _dump_images(images_dict, save_dir, f"real_{anomaly_name}")
            
            # Save kpi per view
            save_path = os.path.join(save_dir, f"valid_kpi.csv")
            with open(save_path, 'a', newline='') as f:
                writer = csv.writer(f)
                anomaly_names = sorted([name for name in real_images_dict.keys() if name != "Average"])
                view_names = [f"view{i}" for i in range(num_views)]
                
                # Header: kpi, view, anomaly1, anomaly2, ..., Average
                writer.writerow(["kpi", "view"] + anomaly_names + ["Average"])
                
                # Get KPI types (e.g., "cradio_v3_base_fid")
                kpi_types = set()
                for anomaly_name in anomaly_names:
                    if anomaly_name in valid_kpi:
                        for view_name in view_names:
                            if view_name in valid_kpi[anomaly_name]:
                                kpi_types.update(valid_kpi[anomaly_name][view_name].keys())
                
                # Write each KPI type for each view
                for kpi_type in sorted(kpi_types):
                    for view_name in view_names:
                        row = [kpi_type, view_name]
                        # Add scores for each anomaly type
                        for anomaly_name in anomaly_names:
                            if anomaly_name in valid_kpi and view_name in valid_kpi[anomaly_name]:
                                score = valid_kpi[anomaly_name][view_name].get(kpi_type)
                                row.append(f"{score:.4f}" if score is not None else "None")
                            else:
                                row.append("None")
                        # Add average score
                        if "Average" in valid_kpi and view_name in valid_kpi["Average"]:
                            score = valid_kpi["Average"][view_name].get(kpi_type)
                            row.append(f"{score:.4f}" if score is not None else "None")
                        else:
                            row.append("None")
                        writer.writerow(row)
                    
                    # Add average row for each anomaly type (across all views)
                    row = [kpi_type, "all views"]
                    for anomaly_name in anomaly_names:
                        if anomaly_name in valid_kpi:
                            # Calculate average across all views for this anomaly type
                            scores = []
                            for view_name in view_names:
                                if view_name in valid_kpi[anomaly_name]:
                                    score = valid_kpi[anomaly_name][view_name].get(kpi_type)
                                    if score is not None:
                                        scores.append(score)
                            if scores:
                                avg_score = sum(scores) / len(scores)
                                row.append(f"{avg_score:.4f}")
                            else:
                                row.append("None")
                        else:
                            row.append("None")
                    # Add average of averages (average across all anomaly types, across all views)
                    all_scores = []
                    for anomaly_name in anomaly_names:
                        if anomaly_name in valid_kpi:
                            for view_name in view_names:
                                if view_name in valid_kpi[anomaly_name]:
                                    score = valid_kpi[anomaly_name][view_name].get(kpi_type)
                                    if score is not None:
                                        all_scores.append(score)
                    if all_scores:
                        overall_avg = sum(all_scores) / len(all_scores)
                        row.append(f"{overall_avg:.4f}")
                    else:
                        row.append("None")
                    writer.writerow(row)

        if self.world_size > 1:
            torch.distributed.barrier()
        
        # Clean up validation data and free GPU memory
        # This is critical to prevent OOM when returning to training
        log.info("Cleaning up validation data and freeing GPU memory...")
        
        # Clear validation dictionaries (they contain large numpy arrays)
        # Note: numpy arrays will be automatically freed by Python's reference counting
        # after .clear() removes the dictionary references
        self.generated_images_dict.clear()
        self.validation_sample_indice.clear()
        # Note: real_images_dict is kept for reuse across validation runs
        
        # Force garbage collection
        # This is necessary because:
        # 1. During validation, we create many local variables (flat_generated_images_dict_list,
        #    filtered_generated_images_dict_list, etc.) that may have circular references
        # 2. compute_kpi_per_view_multiview() creates GPU tensors and models that may have
        #    circular references with Python objects
        # 3. gc.collect() is SAFE - it only reclaims objects with zero references.
        #    Objects still referenced (e.g., self.real_images_dict) will NOT be cleared.
        gc.collect()
        
        # Free GPU memory cached by PyTorch
        # This is critical because compute_feats() loads backbone models to GPU.
        # Even after Python objects are freed, PyTorch's CUDA allocator may retain memory blocks.
        torch.cuda.synchronize()  # Ensure all CUDA operations complete
        torch.cuda.empty_cache()  # Release cached memory blocks back to CUDA
        
        log.info("Validation cleanup complete.")
