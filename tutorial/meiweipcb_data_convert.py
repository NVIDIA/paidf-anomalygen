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
import shutil
import argparse


def main(args):
    source = args.source
    target = args.target
    
    # Define source directories
    images_dir = os.path.join(source, "images")
    images_nor_dir = os.path.join(source, "images_nor")
    mask_dir = os.path.join(source, "mask")
    
    # Validate source directories
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    if not os.path.exists(images_nor_dir):
        raise ValueError(f"Normal images directory not found: {images_nor_dir}")
    if not os.path.exists(mask_dir):
        raise ValueError(f"Mask directory not found: {mask_dir}")
    
    # Define target directories
    normal_img_dir = os.path.join(target, "Anomaly_test", "normal_img")
    anomaly_img_dir = os.path.join(target, "train_downsample", "PCB", "anomaly_image", "defect")
    mask_out_dir = os.path.join(target, "train_downsample", "PCB", "mask", "defect")
    
    # Create target directories
    os.makedirs(normal_img_dir, exist_ok=True)
    os.makedirs(anomaly_img_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    
    # Process triplets: mask (_Cur.png), anomaly image (_Cur.jpg), normal image (_Ref.jpg)
    count = 0
    for mask_name in os.listdir(mask_dir):
        if mask_name.lower().endswith(".png"):
            # Extract base name (without _Cur suffix)
            base = mask_name.rsplit("_Cur", 1)[0]
            mask_file = os.path.join(mask_dir, mask_name)
            anomaly_file = os.path.join(images_dir, f"{base}_Cur.jpg")
            normal_file = os.path.join(images_nor_dir, f"{base}_Ref.jpg")
            
            if os.path.exists(anomaly_file) and os.path.exists(normal_file):
                # Copy mask with _mask suffix (rename to .jpg)
                shutil.copy2(mask_file, os.path.join(mask_out_dir, f"{base}_Cur_mask.jpg"))
                # Copy anomaly image
                shutil.copy2(anomaly_file, os.path.join(anomaly_img_dir, f"{base}_Cur.jpg"))
                # Copy normal image
                shutil.copy2(normal_file, os.path.join(normal_img_dir, f"{base}_Ref.jpg"))
                count += 1
    
    print(f"\nProcessed {count} image triplets (anomaly + mask + normal)")
    print(f"Dataset conversion complete!")
    print(f"Output directory: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MeiweiPCB dataset to PAIDF AnomalyGen format.")
    parser.add_argument("--source", required=True, help="Path to the source directory containing images/, images_nor/, and mask/ folders.")
    parser.add_argument("--target", required=True, help="Path to the target directory for the converted dataset.")
    
    args = parser.parse_args()
    main(args)
