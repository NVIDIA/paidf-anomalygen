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
Multi-view Synthetic Dataset Generation Script

This script performs inference using a trained multi-view anomaly generation model
to generate synthetic anomaly images across multiple views simultaneously.

Usage:
    CUDA_HOME=$CONDA_PREFIX \
    CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node=1 -m scripts.anomaly_gen.multiview_synthetic_dataset_generation \
        --config=cosmos_predict2/configs/base/ag_config.py \
        --ag_checkpoint_dir=results/anomaly_gen/PeppermintCandy/PeppermintCandy_multiview_2B_512 \
        --step=75000 \
        --input_data_path=ag_inference/peppermint_validation.jsonl \
        --output_image_path=inference_output/multiview \
        --seed=0 \
        -- experiment=predict2_anomaly_gen_multiview_ddp_2b
"""

import argparse
import os
import importlib
import csv
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from imaginaire.utils import log, misc
from scripts.anomaly_gen.ag_train import set_nested_attributes
from imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2.inference.anomaly_gen.initialize import initialize_anomaly_diffusion_model
from cosmos_predict2.data.anomaly_gen.multiview_anomaly_dataset import MultiViewAnomalyInpaintDataset
from cosmos_predict2.inference.anomaly_gen.multiview_inpaint_condition import MultiViewAnomalyInpaintCondition
from cosmos_predict2.inference.anomaly_gen.multiview_inference_utils import inpaint_multiview_image
import yaml

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.enable_grad(False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-view SDG with Pretrained PAIDF AnomalyGen model")
    
    parser.add_argument(
        "--config",
        default="cosmos_predict2/configs/base/ag_config.py",
        help="Path to the config file",
    )
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--ag_checkpoint_dir",
        type=str,
        default="results/anomaly_gen/PeppermintCandy/PeppermintCandy_multiview_2B_512",
        help="Multi-view anomaly gen checkpoint directory",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=75000,
        help="Step of the checkpoint to use",
    )
    parser.add_argument(
        "--input_data_path",
        type=str,
        default="ag_inference/peppermint_validation.jsonl",
        help="Path to the input JSONL file",
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="inference_output/multiview",
        help="Path to save output images",
    )

    return parser.parse_args()


def demo(args):
    """Run multi-view anomaly generation inference.
    
    This function follows the same structure as single-view demo():
    - Load dataset using MultiViewAnomalyInpaintDataset
    - Initialize model using initialize_anomaly_diffusion_model
    - Iterate through dataloader
    - Call inpaint_multiview_image for each batch
    - Save results and write CSV
    """
    misc.set_random_seed(args.seed, by_rank=True)

    # Initialize cuDNN
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load input data using MultiViewAnomalyInpaintDataset
    dataset = MultiViewAnomalyInpaintDataset(args.input_data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset._collate_fn
    )

    # Load config
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)

    # Load config from pretrained checkpoint
    with open(f"{args.ag_checkpoint_dir}/ag_config.yaml") as fp:
        try:
            ag_config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"[ERROR] Cannot load ag_config.yaml file! Exception: {exc}")

    # Merge config w/ ag_config
    set_nested_attributes(config, ag_config)

    # Initialize model (same as single-view)
    model = initialize_anomaly_diffusion_model(config, args.ag_checkpoint_dir, args.step)

    # Extract view_names from ag_config for output filenames
    view_names = None
    if 'dataloader_train' in ag_config and 'dataset' in ag_config['dataloader_train']:
        dataset_config = ag_config['dataloader_train']['dataset']
        if 'view_types' in dataset_config:
            view_names = dataset_config['view_types']
            log.info(f"Using view_names from ag_config: {view_names}")
    else:
        raise ValueError("view_names not found in ag_config")

    # Create output directory
    os.makedirs(args.output_image_path, exist_ok=True)
    
    # CSV for results metadata
    with open(args.output_image_path + "/multiview_SDG_result.csv", "w") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "output_filename",
            "image_filenames",
            "mask_filenames",
            "anomaly_type",
            "guidance",
            "num_steps",
            "seed",
            "num_generated_images",
            "crop_and_paste",
            "crop_grid_X",
            "crop_grid_Y",
            "crop_ratio",
            "poisson_blend",
            "shift_values",
            "rotation_angle",
            "morph_operation",
            "PSNR",
            "index",
            "guardrail_pass",
        ])

        anomaly_name_counter = defaultdict(int)

        for i, input_dict in enumerate(dataloader):
            log.info(f"Processing {i}th multi-view sample")
            
            # Form data_batch to align API usage (same as single-view)
            inpaint_condition = MultiViewAnomalyInpaintCondition(**input_dict)
            
            # Run inpainting flow (multi-view version)
            inpainting_result, anomaly_names, _ = inpaint_multiview_image(inpaint_condition, model)

            anomaly_index_map = {}
            for idx, name in enumerate(anomaly_names):
                anomaly_index_map[idx] = anomaly_name_counter[name]
                anomaly_name_counter[name] += 1

            # Save results (similar to single-view)
            # inpainting_result format: {key: [num_views][B] or [num_views][B][num_instances]}
            for key, images_per_view in inpainting_result.items():
                num_views = len(images_per_view)
                for idx, anomaly_name in enumerate(anomaly_names):
                    save_dir = os.path.join(args.output_image_path, key)
                    os.makedirs(save_dir, exist_ok=True)
                    anomaly_idx = anomaly_index_map[idx]
                    
                    if key in ["annotated_image", "cropped_image", "cropped_mask"]:
                        # Save multiple instances per view
                        for view_idx in range(num_views):
                            view_name = view_names[view_idx] if view_idx < len(view_names) else f"view{view_idx}"
                            for j, image in enumerate(images_per_view[view_idx][idx]):  # images_per_view[view_idx][idx] is a list of instances
                                filename = f"{anomaly_name}_{anomaly_idx:05d}_{view_name}_{j}.png"
                                image.save(os.path.join(save_dir, filename))
                    else:
                        # Save single image per view
                        for view_idx in range(num_views):
                            view_name = view_names[view_idx] if view_idx < len(view_names) else f"view{view_idx}"
                            filename = f"{anomaly_name}_{anomaly_idx:05d}_{view_name}.png"
                            item = images_per_view[view_idx][idx]  # Single PIL image
                            item.save(os.path.join(save_dir, filename))

                    # Write corresponding metadata only once (for reconstructed_image)
                    if key == "reconstructed_image":
                        # Build output filename (just the first view as representative)
                        output_filename = os.path.join(save_dir, f"{anomaly_name}_{anomaly_idx:05d}_{view_names[0]}.png")
                        
                        # Extract PSNR for this sample (average across views)
                        psnr_values = [inpaint_condition.PSNR[view_idx][idx] for view_idx in range(num_views)]
                        avg_psnr = sum(p for p in psnr_values if p is not None) / len([p for p in psnr_values if p is not None]) if any(p is not None for p in psnr_values) else None
                        
                        writer.writerow([
                            output_filename,
                            str(inpaint_condition.image_filenames[idx]),
                            str(inpaint_condition.mask_filename[idx]),
                            str(inpaint_condition.anomaly_type[idx]),
                            inpaint_condition.guidance,
                            inpaint_condition.num_steps,
                            inpaint_condition.seed,
                            inpaint_condition.num_generated_images,
                            inpaint_condition.crop_and_paste[idx],
                            inpaint_condition.crop_grid_X[idx],
                            inpaint_condition.crop_grid_Y[idx],
                            inpaint_condition.crop_ratio[idx],
                            inpaint_condition.poisson_blend[idx],
                            ",".join(map(str, inpaint_condition.shift_values[idx])),
                            inpaint_condition.rotation_angle[idx],
                            inpaint_condition.morph_operation[idx],
                            avg_psnr,
                            inpaint_condition.index[idx],
                            # 1 only if this sample passed the guardrail in every view.
                            # `guardrail_safe` is always set by inpaint_multiview_image
                            # (a [num_views][B] nested list), so index it directly.
                            1 if all(
                                inpaint_condition.guardrail_safe[view_idx][idx]
                                for view_idx in range(num_views)
                            ) else 0,
                        ])

    log.success(f"Multi-view SDG complete! Results saved to {args.output_image_path}")


if __name__ == "__main__":
    args = parse_arguments()
    demo(args)
