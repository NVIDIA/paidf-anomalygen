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
import os
import importlib
import csv
import json
import time

import torch
from torch.utils.data import DataLoader, Subset

from imaginaire.utils import log, misc
# Reuse flow from training
from scripts.anomaly_gen import convert_to_daft_format
from scripts.anomaly_gen.ag_train import set_nested_attributes
from imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2.inference.anomaly_gen.distributed_inference_utils import (
    aggregate_rank_timings,
    build_sample_output_plans,
    configure_local_cuda_device,
    destroy_distributed_collectives,
    get_rank_work_items,
    get_runtime_context,
    initialize_distributed_collectives,
    merge_rank_timings,
    merge_rank_rows,
    wait_for_all_rank_timings,
    wait_for_all_rank_rows,
)
from cosmos_predict2.inference.anomaly_gen.inference_anomaly_diffusion_utils import inpaint_image
from cosmos_predict2.inference.anomaly_gen.initialize import initialize_anomaly_diffusion_model
from cosmos_predict2.inference.anomaly_gen.inpaint_condition import AnomalyInpaintCondition
from cosmos_predict2.data.anomaly_gen.anomaly_dataset import AnomalyInpaintDataset
import yaml

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.enable_grad(False)

CSV_HEADER = [
    "output_filename",
    "image_filename",
    "mask_filename",
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
]
def extract_class_names(nested_dict):
    """
    Recursively processes a nested dictionary and extracts class names from values.
    For any value containing "<Class 'xxxx'>", replaces it with just "xxxx".

    Args:
        nested_dict (dict): A nested dictionary to process

    Returns:
        dict: The processed dictionary with class names extracted
    """
    if isinstance(nested_dict, dict):
        for key, value in nested_dict.items():
            if isinstance(value, (dict, list)):
                # Recursively process nested dictionaries and lists
                extract_class_names(value)
            elif isinstance(value, str) and "<class '" in value:
                # Extract the class name between single quotes
                class_name = value.split("'")[1]
                nested_dict[key] = class_name
    elif isinstance(nested_dict, list):
        for i, item in enumerate(nested_dict):
            if isinstance(item, (dict, list)):
                # Recursively process dictionaries and lists within lists
                extract_class_names(item)
            elif isinstance(item, str) and "<class '" in item:
                # Extract the class name between single quotes
                class_name = item.split("'")[1]
                nested_dict[i] = class_name

    return nested_dict


def convert_nested_dict_values(d):
    """
    Recursively convert values in a nested dict to int or float if possible.
    Modifies the dictionary in-place.
    """
    def try_convert(val):
        """
        Try to convert a value to int or float. If not possible, return as is.
        """
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                try:
                    return float(val)
                except ValueError:
                    return val
        return val

    for k, v in d.items():
        if isinstance(v, dict):
            convert_nested_dict_values(v)
        elif isinstance(v, list):
            # Optionally handle lists of values or dicts
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    convert_nested_dict_values(item)
                else:
                    v[i] = try_convert(item)
        else:
            d[k] = try_convert(v)
    return d


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perform SDG w/ Pretrained PAIDF AnomalyGen model")
    # Add common arguments
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

    # Add text2anomaly specific arguments
    parser.add_argument(
        "--ag_checkpoint_dir",
        type=str,
        default="results/anomaly_gen/MeiweiPCB/trial",
        help="Anomaly gen checkpoint directory name relative to checkpoint_dir",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=200000,
        help="Step of the anomaly diffusion checkpoint to use",
    )

    # Add text2anomaly specific arguments
    parser.add_argument(
        "--input_data_path",
        type=str,
        default="ad_inference/example.jsonl",
        help="Path to the input data file",
    )

    # Add text2anomaly specific arguments
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="inference_output",
        help="Path to the output image file",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers per rank",
    )

    return parser.parse_args()


def _to_csv_scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _write_csv(csv_path: str, rows: list[list]) -> None:
    with open(csv_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def _write_timing_summary(output_image_path: str, timing_summary: dict) -> None:
    timing_summary_path = os.path.join(output_image_path, "timing_summary.json")
    with open(timing_summary_path, "w", encoding="utf-8") as fp:
        json.dump(timing_summary, fp, ensure_ascii=True, indent=2)
        fp.write("\n")


def _finalize_rows(args, runtime, rows: list[dict]) -> None:
    csv_path = os.path.join(args.output_image_path, "SDG_result.csv")
    if not runtime.is_multi_gpu:
        ordered_rows = [row["row"] for row in sorted(rows, key=lambda item: tuple(item["sort_key"]))]
        _write_csv(csv_path, ordered_rows)
        return

    wait_for_all_rank_rows(runtime.world_size)
    merged_rows = merge_rank_rows(rows, runtime.world_size)
    if runtime.rank == 0:
        _write_csv(csv_path, merged_rows)


def _finalize_timing_summary(args, runtime, timing_info: dict) -> None:
    if not runtime.is_multi_gpu:
        _write_timing_summary(args.output_image_path, aggregate_rank_timings([timing_info]))
        return

    wait_for_all_rank_timings(runtime.world_size)
    timing_summary = aggregate_rank_timings(merge_rank_timings(timing_info, runtime.world_size))
    if runtime.rank == 0:
        _write_timing_summary(args.output_image_path, timing_summary)


def demo(args):
    """Run text-to-anomaly generation demo.

    This function handles the main text-to-anomaly generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts from input
    - Generating videos from text prompts
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files
        - Text files containing the processed prompts

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """
    total_start = time.perf_counter()
    runtime = get_runtime_context()
    configure_local_cuda_device(runtime)
    misc.set_random_seed(args.seed, by_rank=False)

    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load input data
    dataset = AnomalyInpaintDataset(args.input_data_path)
    sample_output_plans = build_sample_output_plans(dataset.input_data)
    rank_work_items = get_rank_work_items(len(dataset), runtime.rank, runtime.world_size)
    dataset_for_rank = Subset(dataset, rank_work_items) if runtime.is_multi_gpu else dataset
    dataloader = DataLoader(
        dataset_for_rank,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset._collate_fn
    )

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

    if runtime.is_multi_gpu and getattr(config.model.config, "fsdp_shard_size", 0) != 0:
        log.info("Disabling FSDP for rank-sharded multi-GPU inference.")
        config.model.config.fsdp_shard_size = 0

    os.makedirs(args.output_image_path, exist_ok=True)

    if runtime.is_multi_gpu:
        log.info(
            f"Multi-GPU inference enabled via rank sharding: rank {runtime.rank + 1}/{runtime.world_size}, "
            f"local_rank={runtime.local_rank}, assigned_samples={len(rank_work_items)}"
        )

    local_rows = []
    setup_seconds = time.perf_counter() - total_start
    model_init_seconds = 0.0
    generation_seconds = 0.0
    if rank_work_items:
        # Initialize model only on ranks that actually have work to do.
        model_init_start = time.perf_counter()
        model = initialize_anomaly_diffusion_model(config, args.ag_checkpoint_dir, args.step)
        model_init_seconds = time.perf_counter() - model_init_start

        generation_start = time.perf_counter()
        for i, input_dict in enumerate(dataloader):
            inpaint_condition = AnomalyInpaintCondition(**input_dict)

            sample_index = inpaint_condition.index[0]
            sample_plan = sample_output_plans[sample_index]

            inpainting_result, anomaly_names, _ = inpaint_image(inpaint_condition, model)
            if len(anomaly_names) != sample_plan.num_outputs:
                raise RuntimeError(
                    "Unexpected number of generated outputs for input sample "
                    f"{sample_index}: expected {sample_plan.num_outputs}, got {len(anomaly_names)}"
                )

            anomaly_index_map = {
                idx: sample_plan.anomaly_offset + idx
                for idx, _ in enumerate(anomaly_names)
            }

            for key, image_list in inpainting_result.items():
                for idx, (anomaly_name, item) in enumerate(zip(anomaly_names, image_list)):
                    save_dir = os.path.join(args.output_image_path, key)
                    os.makedirs(save_dir, exist_ok=True)
                    anomaly_idx = anomaly_index_map[idx]
                    if key in ["annotated_image", "cropped_image", "cropped_mask"]:
                        for j, image in enumerate(item):
                            filename = f"{anomaly_name}_{anomaly_idx:05d}_{j}.png"
                            image.save(os.path.join(save_dir, filename))
                    else:
                        filename = f"{anomaly_name}_{anomaly_idx:05d}.png"
                        item.save(os.path.join(save_dir, filename))

                        if key == "reconstructed_image":
                            local_rows.append(
                                {
                                    "sort_key": [sample_plan.global_order, idx],
                                    "row": [
                                        os.path.join(save_dir, filename),
                                        inpaint_condition.image_filename[0] if isinstance(inpaint_condition.image_filename, list) else inpaint_condition.image_filename,
                                        inpaint_condition.mask_filename[0] if isinstance(inpaint_condition.mask_filename, list) else inpaint_condition.mask_filename,
                                        inpaint_condition.anomaly_type[0] if isinstance(inpaint_condition.anomaly_type, list) else inpaint_condition.anomaly_type,
                                        _to_csv_scalar(inpaint_condition.guidance[0]),
                                        _to_csv_scalar(inpaint_condition.num_steps),
                                        _to_csv_scalar(inpaint_condition.seed),
                                        _to_csv_scalar(inpaint_condition.num_generated_images),
                                        _to_csv_scalar(inpaint_condition.crop_and_paste[0]),
                                        _to_csv_scalar(inpaint_condition.crop_grid_X[0]),
                                        _to_csv_scalar(inpaint_condition.crop_grid_Y[0]),
                                        _to_csv_scalar(inpaint_condition.crop_ratio[0]),
                                        _to_csv_scalar(inpaint_condition.poisson_blend[0]),
                                        ",".join(map(str, inpaint_condition.shift_values[0])),
                                        _to_csv_scalar(inpaint_condition.rotation_angle[0]),
                                        _to_csv_scalar(inpaint_condition.morph_operation[0]),
                                        _to_csv_scalar(inpaint_condition.PSNR[idx]),
                                        _to_csv_scalar(inpaint_condition.index[0]),
                                        _to_csv_scalar(
                                            1 if getattr(inpaint_condition, "guardrail_safe", [True] * (idx + 1))[idx] else 0
                                        ),
                                    ],
                                }
                            )
        generation_seconds = time.perf_counter() - generation_start

    finalize_start = time.perf_counter()
    initialized_collectives = initialize_distributed_collectives(runtime)
    try:
        _finalize_rows(args, runtime, local_rows)
        finalize_seconds = time.perf_counter() - finalize_start

        timing_info = {
            "rank": runtime.rank,
            "world_size": runtime.world_size,
            "local_rank": runtime.local_rank,
            "assigned_samples": len(rank_work_items),
            "generated_images": len(local_rows),
            "setup_seconds": setup_seconds,
            "model_init_seconds": model_init_seconds,
            "generation_seconds": generation_seconds,
            "finalize_seconds": finalize_seconds,
            "measured_total_seconds": time.perf_counter() - total_start,
        }
        _finalize_timing_summary(args, runtime, timing_info)
        if not runtime.is_multi_gpu or runtime.rank == 0:
            log.success(f"SDG complete! Results saved to {args.output_image_path}")
    finally:
        try:
            destroy_distributed_collectives(initialized_collectives)
        except Exception as exc:
            log.warning(f"Failed to destroy distributed process group cleanly: {exc}")


    if not runtime.is_multi_gpu or runtime.rank == 0:
        # Also emit a TAO DAFT v3.0 scene alongside the raw SDG output.
        log.info(f"Converting SDG output to TAO DAFT v3.0 format: {args.output_image_path}_daft_v3")
        convert_to_daft_format.main(["--input", args.output_image_path])


if __name__ == "__main__":
    args = parse_arguments()
    demo(args)
