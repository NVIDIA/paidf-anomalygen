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

import time
import os
from typing import Dict, Optional, Union, List
import torch
from torch import Tensor
from torchvision.io import write_png
from PIL import Image
import numpy as np
from pathlib import Path

from imaginaire.callbacks.every_n import EveryN
from imaginaire.utils import distributed, log
from imaginaire.utils.distributed import rank0_only
from imaginaire.model import ImaginaireModel
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils.callback import Callback


class LogImage(EveryN):
    """Callback to log generated images during training.
    
    Args:
        exp_path (str): Path to save experiment outputs
        num_steps (int, optional): Number of diffusion steps. Defaults to 35.
        guidance (float, optional): Guidance scale. Defaults to 1.5.
        seed (int, optional): Random seed. Defaults to 1.
        is_negative_prompt (bool, optional): Whether to use negative prompts. Defaults to True.
        n_sample (int, optional): Number of samples to generate. Defaults to 2.
    """

    def __init__(
        self, 
        *args,
        exp_path: str,
        num_steps: int = 35,
        guidance: float = 1.5,
        seed: int = 1,
        is_negative_prompt: bool = True,
        n_sample: int = 2, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Input validation
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if guidance <= 0:
            raise ValueError("guidance must be positive")
        if n_sample < 1:
            raise ValueError("n_sample must be positive")
            
        self.num_steps = num_steps
        self.guidance = guidance
        self.seed = seed
        self.is_negative_prompt = is_negative_prompt
        self.n_sample = n_sample
        self.exp_path = Path(exp_path)
        
    def on_train_start(self, model, iteration=0):
        self.world_size = distributed.get_world_size()
        self.rank = distributed.get_rank()

        # Use job path from trainer config if available, fallback to exp_path
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'config') and hasattr(self.trainer.config, 'job'):
            job_path = Path(self.trainer.config.job.path_local) / "trialrun"
            log.info(f"LogImage: Using job path {job_path} instead of {self.exp_path}")
            self.exp_path = job_path
        else:
            log.warning(f"LogImage: Could not find job path in trainer config, using default {self.exp_path}")

    # Since we might use model parallelism
    # (`model_parallel.context_parallel_size > 1`), `rank0_only` cannot be used
    # here. Otherwise, the worker will be stuck.
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: Dict[str, Tensor],
        output_batch: Dict[str, Tensor],
        loss: Tensor,
        iteration: int,
    ) -> None:
        """Process and log images for the current iteration.
        
        Args:
            trainer: The trainer instance
            model: The model instance
            data_batch: Input data batch
            output_batch: Output data batch
            loss: Current loss value
            iteration: Current iteration number
        """
        log.info("Start logging images:")
        
        # Support both single-view (images) and multi-view (video)
        is_multiview = 'video' in data_batch and 'images' not in data_batch
        image_key = 'video' if is_multiview else 'images'
        
        # Ensure required keys exist in data_batch
        required_keys = [image_key, 'mask']
        missing_keys = [key for key in required_keys if key not in data_batch]
        if missing_keys:
            log.error(f"Missing required keys in data_batch: {missing_keys}")
            return
        
        # For multi-view, use the first frame for visualization
        if is_multiview:
            # Convert video to images format for the pipeline
            # video shape: [B, C, T, H, W] -> images shape: [B, C, T, H, W] (same, but we'll use first frame for GT display)
            data_batch['images'] = data_batch['video']
            log.info(f"Multi-view mode: using video with {data_batch['video'].shape[2]} frames")

        # Create output directory
        output_dir = self.exp_path / str(iteration)
        if self.rank == 0:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                log.error(f"Failed to create output directory: {e}")
                return

        try:
            with torch.inference_mode():
                # Process images in batches to manage memory
                denoised_result = model.pipe(
                    data_batch=data_batch,
                    seed=self.seed,
                    guidance=self.guidance,
                    num_steps=self.num_steps,
                    is_negative_prompt=self.is_negative_prompt,
                    n_sample=self.n_sample,
                    use_cuda_graphs=False
                )
                data_batch['denoised_result'] = denoised_result

                # Perform `all_gather_object` to gather data across all
                # processes.
                data_list = [data_batch] * self.world_size
                if self.world_size > 1:
                    torch.distributed.all_gather_object(data_list, data_batch)
                    torch.distributed.barrier()

                # Reconstruction of image and save
                if self.rank == 0:
                    self._save_image_batch(
                        model=model,
                        # Aggregate the data list across all processes.
                        data_batch=self._aggregate_data_list(data_list),
                        output_dir=output_dir,
                        iteration=iteration
                    )


        except Exception as e:
            log.error(f"Error during image generation: {str(e)}")
            raise

    def _aggregate_data_list(
        self,
        data_list: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Aggregate the data list across all processes."""
        data_batch = {}
        for key in data_list[0].keys():
            if not isinstance(data_list[0][key], torch.Tensor):
                # Special handling for caption (list of strings)
                if key == 'caption':
                    # Concatenate caption lists from all processes
                    captions = []
                    for item in data_list:
                        if 'caption' in item:
                            captions.extend(item['caption'])
                    data_batch['caption'] = captions
                continue
            data = torch.concatenate(
                [item[key].to(torch.cuda.current_device()) for item in data_list], dim=0
            )
            data_batch[key] = data
        return data_batch

    def _save_image_batch(
        self,
        model: ImaginaireModel,
        data_batch: Dict[str, Tensor],
        output_dir: Path,
        iteration: int
    ) -> None:
        """Save a batch of processed images with error handling."""
        try:
            # Check if multi-view (has temporal dimension > 1)
            is_multiview = data_batch['denoised_result'].shape[2] > 1
            num_frames = data_batch['denoised_result'].shape[2]
            
            if is_multiview:
                log.info(f"Saving multi-view results with {num_frames} frames")
            
            # For multi-view, save all frames; for single-view, save frame 0
            frames_to_save = range(num_frames) if is_multiview else [0]
            
            for frame_idx in frames_to_save:
                # Image reconstruction
                reconstructed_images = data_batch['denoised_result'][:, :, frame_idx, :, :]
                reconstructed_images = ((reconstructed_images + 1).clamp(0, 2) * 127.5).to(torch.uint8).cpu()

                # Model gt image reconstruction
                gt_image = ((data_batch['images'] + 1).clamp(0, 2) * 127.5).to(torch.uint8)[:, :, frame_idx, :, :].cpu()

                # Guided mask
                guided_mask = ((1 - data_batch['guided_mask']) * 255).to(torch.uint8)[:, :, frame_idx, :, :].cpu()

                # VAE reconstructed real image (encode all frames, decode specific frame for display)
                if frame_idx == 0:  # Only encode once
                    vae_encoded_image = model.pipe.encode(data_batch['images'])
                    vae_decoded_image = model.pipe.decode(vae_encoded_image)
                reconstructed_vae_image = ((vae_decoded_image + 1).clamp(0, 2) * 127.5).to(torch.uint8)[:, :, frame_idx, :, :].cpu()

                # Write images and info
                B, C, H, W = reconstructed_images.shape
                
                # For multi-view, include frame index in filename
                frame_suffix = f"_frame={frame_idx}" if is_multiview else ""
                
                for i in range(B):
                    # Save images with error handling
                    images_to_save = {
                        'recon': reconstructed_images[i],
                        'guided_mask': guided_mask[i],
                        'gt_image': gt_image[i],
                        'gt_vae_reconstructed': reconstructed_vae_image[i]
                    }
                    
                    for name, img in images_to_save.items():
                        try:
                            output_path = output_dir / f"{name}_iter={iteration}_data_idx={i}{frame_suffix}.png"
                            write_png(input=img, filename=str(output_path), compression_level=0)
                        except Exception as e:
                            log.error(f"Failed to save {name} image: {str(e)}")
            
            # Write info file once (outside frame loop)
            info_path = output_dir / "info.txt"
            B = data_batch['denoised_result'].shape[0]
            
            with open(info_path, 'w') as fp:
                for i in range(B):
                    fp.write(f"======== Data {i} ========\n")
                    
                    # Write caption explicitly (it's a list, so handle separately)
                    if 'caption' in data_batch:
                        captions = data_batch['caption']
                        if isinstance(captions, list) and i < len(captions):
                            fp.write(f"caption={captions[i]}\n")
                    
                    # Write other non-tensor/list values
                    for k, v in data_batch.items():
                        if k == 'caption':  # Already handled above
                            continue
                        if type(v) not in [torch.Tensor, list]:
                            continue
                        fp.write(f"{k}={v[i]}\n")
                        if torch.is_tensor(v[i]):
                            fp.write(f"{k}={v[i].shape}\n")
                            fp.write(f"Min={v[i].min()}, Max={v[i].max()}, dtype={v[i].dtype}, has nan={torch.isnan(v[i]).any()}\n")

                fp.write(f"======== Model Parameters ========\n")
                # Write parameters
                fp.write(f"Anomaly embedding:")
                for key, value in model.pipe.anomaly_embedding.named_parameters():
                    if value.requires_grad:
                        fp.write(f"{key}: {value}\n")

                # Write adapter
                fp.write(f"Adapter:")
                for key, value in model.pipe.adapter.named_parameters():
                    if value.requires_grad:
                        fp.write(f"{key}: {value}\n")

                fp.write(f"Mask encoder:")
                for key, value in model.pipe.mask_encoder.named_parameters():
                    if value.requires_grad:
                        fp.write(f"{key}: {value}\n")

        except Exception as e:
            log.error(f"Error in _save_image_batch: {str(e)}")
            raise
