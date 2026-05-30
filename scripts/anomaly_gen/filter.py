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
import argparse
import shutil

import numpy as np
import pandas as pd

from cosmos_predict2.metrics.utils import compute_sample_wise_kpi, load_real_images, load_generated_images
from imaginaire.utils import log

SEED = 123
np.random.seed(SEED)

def filter_topk(scores, drop_ratio):
    """Global top-K filtering by score."""
    sorted_idx = np.argsort(scores)[::-1]
    num_keep = int(round((1 - drop_ratio) * len(scores)))
    return sorted_idx[:num_keep], sorted_idx[num_keep:]

def filter_generated_images(
    gen_dict,
    real_dict,
    output_path="output",
    drop_ratio=0.3,
    backbone_name="cradio_v3_base",
    K=1,
    rotation_range=None,
    rotation_step=15,
):
    
    gen_dict = compute_sample_wise_kpi(gen_dict, real_dict, K, rotation_range, rotation_step, backbone_name=backbone_name)

    kept_basenames, dropped_basenames = [], []

    for split in ["keep", "drop"]:
        for sub in ["reconstructed_image", "original_mask", "original_image"]:
            os.makedirs(os.path.join(output_path, f"{split}", sub), exist_ok=True)

    for anomaly_key, item in gen_dict.items():
        scores = item[f"{backbone_name}_giqa"]
        img_paths, mask_paths = item["img_path"], item["mask_path"]
        orig_paths = item["orig_path"]

        if len(scores) != len(img_paths):
            raise ValueError(
                f"Mismatch between scores and images for {anomaly_key}. Score: {len(scores)} !=  image:{len(img_paths)} "
            )

        # Strategy dispatcher
        keep_idx, drop_idx = filter_topk(scores, drop_ratio)

        # Save results
        for idx in keep_idx:
            base_name = os.path.basename(img_paths[idx])

            shutil.copy(img_paths[idx], os.path.join(output_path, "keep", "reconstructed_image", base_name))
            shutil.copy(mask_paths[idx], os.path.join(output_path, "keep", "original_mask", base_name))
            shutil.copy(orig_paths[idx], os.path.join(output_path, "keep", "original_image", base_name))
            kept_basenames.append(base_name)

        for idx in drop_idx:
            base_name = os.path.basename(img_paths[idx])

            shutil.copy(img_paths[idx], os.path.join(output_path, "drop", "reconstructed_image", base_name))
            shutil.copy(mask_paths[idx], os.path.join(output_path, "drop", "original_mask", base_name))
            shutil.copy(orig_paths[idx], os.path.join(output_path, "drop", "original_image", base_name))
            dropped_basenames.append(base_name)

        log.info(f"[{anomaly_key}] kept: {len(keep_idx)} / dropped: {len(drop_idx)}")

    return gen_dict, kept_basenames, dropped_basenames

def save_filter_result_csv(gen_dict, output_path, backbone_name):
    all_rows = []
    for anomaly_key in gen_dict:
        img_paths = gen_dict[anomaly_key]["img_path"]
        giqas = gen_dict[anomaly_key].get(f"{backbone_name}_giqa", [float("nan")] * len(img_paths))

        for img_path, giqa in zip(img_paths, giqas):
            all_rows.append({
                "anomaly_key": anomaly_key,
                "filename": os.path.basename(img_path),
                f"{backbone_name}_giqa": giqa
            })

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(output_path, "filter_result.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"Saved filtering results to {csv_path}")

def save_filtered_sdg_csv(generated_path, output_path, kept_basenames, dropped_basenames):
    sdg_path = os.path.join(generated_path, "SDG_result.csv")
    if not os.path.exists(sdg_path):
        log.info(f"No SDG_result.csv found under {generated_path}, skipping CSV split")
        return

    df = pd.read_csv(sdg_path)

    def _filter_rows(target_names):
        if not target_names:
            return df.iloc[0:0]
        names = set(target_names)
        return df[df["output_filename"].apply(lambda p: os.path.basename(p) in names)]

    keep_df = _filter_rows(kept_basenames)
    drop_df = _filter_rows(dropped_basenames)

    keep_csv = os.path.join(output_path, "keep", "SDG_result.csv")
    drop_csv = os.path.join(output_path, "drop", "SDG_result.csv")
    keep_df.to_csv(keep_csv, index=False)
    drop_df.to_csv(drop_csv, index=False)
    log.info(f"Saved split SDG CSVs to {keep_csv} and {drop_csv}")


def main():
    parser = argparse.ArgumentParser(description="Filter generated anomaly images by KPI.")
    parser.add_argument('--real_path', type=str, required=True, help='Path to real images dataset')
    parser.add_argument('--generated_path', type=str, required=True, help='Path to generated images')
    parser.add_argument('--output_path', type=str, required=True, help='Path to filtered images')
    parser.add_argument('--backbone', type=str, default="cradio_v3_base", help='Feature Backbone')
    parser.add_argument('--drop_ratio', type=float, required=True, help='Ratio to drop')
    parser.add_argument('--anomaly_types', type=str, nargs='+', required=True,
                        help='List of anomalies in the format TEXTURE+TYPE (e.g., SEM_IC+crack)')
    parser.add_argument(
        '--rotation_range', type=int, nargs=2, metavar=('MIN_DEG', 'MAX_DEG'),
        default=[0, 0],
        help='Rotation range in degrees; set 0 0 to disable (e.g., 0 180).'
    )
    parser.add_argument(
        '--rotation_step', type=int, default=15,
        help='Rotation step in degrees between MIN_DEG and MAX_DEG (inclusive).'
    )
    args = parser.parse_args()

    log.init_loguru_file(f"{args.output_path}/filter_stdout.log")

    rotation_range = tuple(args.rotation_range)
    rotation_step = args.rotation_step 
    # --- validate ---
    if len(rotation_range) != 2:
        raise ValueError(f"--rotation_range must have exactly 2 values, got {rotation_range}")

    min_deg, max_deg = rotation_range
    if min_deg > max_deg:
        raise ValueError(f"--rotation_range invalid: min {min_deg} > max {max_deg}")
    if rotation_step <= 0:
        raise ValueError(f"--rotation_step must be positive, got {rotation_step}")

    # detect disable case
    use_aug = not (min_deg == 0 and max_deg == 0)

    anomaly_types =  [entry.split("+") for entry in args.anomaly_types]

    log.info("Loading real images...")
    real_dict = load_real_images(args.real_path, anomaly_types)

    log.info("Loading generated images...")
    gen_dict = load_generated_images(args.generated_path, anomaly_types)

    log.info("Filtering by CRADIOv3 Base KNN-GIQA (K=1)...")
    gen_dict, kept_basenames, dropped_basenames = filter_generated_images(
        gen_dict,
        real_dict,
        args.output_path,
        args.drop_ratio,
        args.backbone,
        rotation_range=rotation_range if use_aug else None,
        rotation_step=rotation_step,
    )

    save_filter_result_csv(gen_dict, args.output_path, args.backbone)
    save_filtered_sdg_csv(args.generated_path, args.output_path, kept_basenames, dropped_basenames)

if __name__ == "__main__":
    main()
