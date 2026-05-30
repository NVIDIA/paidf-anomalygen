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
import json
import os

import torch

from imaginaire.utils import log
from roi_generate.model import ROIGenerateModels
from roi_generate.pipeline import run_pipeline
from transformers import set_seed 

def parse_args():
    parser = argparse.ArgumentParser(description="Run ROI segmentation with the specified configuration.")
    parser.add_argument(
        "--input_samples",
        type=str,
        help="Path to the input sample JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/roi_generate",
        help="Output directory",
    )

    return parser.parse_args()


def main():
    set_seed(42)

    args = parse_args()

    log.info(f"Output Directory: {args.output}")

    # Read input
    if not os.path.exists(args.input_samples):
        raise FileNotFoundError(f"Samples JSON file not found: {args.input_samples}")

    with open(args.input_samples, "r") as f:
        samples = json.load(f)

    if not isinstance(samples, list) or not samples:
        raise TypeError(f"Invalid or empty samples JSON: {args.input_samples}")

    log.info(f"Loaded {len(samples)} samples from {args.input_samples}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    roi_generate_models = ROIGenerateModels(device)

    run_pipeline(samples, args.output, roi_generate_models)


if __name__ == "__main__":
    main()
