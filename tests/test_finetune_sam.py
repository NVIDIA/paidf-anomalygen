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
Usage:

pytest tests/test_finetune_sam.py
"""

import argparse
import os
import pathlib
import tempfile
import unittest

import numpy as np
import pytest as pytest
from PIL import Image

from pseudo_label.infosam import build_infosam2
from scripts.anomaly_gen import finetune_sam


class TestFinetuneSam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up fake data here. The dir structure:
        # temp_dir/
        # ├── anomaly_image
        # │   ├── image.png
        # |   ...
        # ├── mask
        # │   ├── image.png
        # |   ...
        # └── output_dir
        temp_dir = tempfile.TemporaryDirectory()

        # Make dirs.
        anomaly_image_dir = pathlib.Path(temp_dir.name, "anomaly_image")
        mask_dir = pathlib.Path(temp_dir.name, "mask")
        output_dir = pathlib.Path(temp_dir.name, "output_dir")
        os.makedirs(anomaly_image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        for i in range(5):
            anomaly_image = Image.fromarray(
                np.zeros((384, 640, 3), dtype=np.uint8), mode="RGB"
            )
            anomaly_image.save(anomaly_image_dir / f"image_{i}.png")
            mask = Image.fromarray(np.ones((384, 640), dtype=np.uint8) * 255, mode="L")
            mask.save(mask_dir / f"image_{i}.png")

        cls.temp_dir = temp_dir

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_finetune_infosam2(self):
        """Test the finetuning of InfoSAM2 model."""
        # Prepare.
        args = argparse.Namespace(
            image_dir=str(pathlib.Path(self.temp_dir.name, "anomaly_image")),
            mask_dir=str(pathlib.Path(self.temp_dir.name, "mask")),
            output_dir=str(pathlib.Path(self.temp_dir.name, "output_dir")),
            train_val_ratio=0.8,
            epochs=1,
            patient_epochs=5,
            batch_size=4,
            lr=2e-4,
            weight_decay=1e-4,
            eval_dilate_sizes=[7, 9, 11, 13],
            eval_dbscan_eps=0.2,
            eval_dbscan_min_samples=5,
            eval_crop_ratio=2.0,
            eval_strength=1.0,
            eval_fallback_ratio=0.5,
        )

        # Run.
        finetune_sam.main(args)

        # Verify.
        checkpoint_path = pathlib.Path(self.temp_dir.name, "output_dir", "best.pt")
        self.assertTrue(checkpoint_path.exists())
        build_infosam2("checkpoints/sam2/sam2.1_hiera_large.pt", str(checkpoint_path))
