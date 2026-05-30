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

pytest tests/test_evaluate_mask_refinement.py
"""

import argparse
import os
import pathlib
import tempfile
import unittest

import numpy as np
import pytest as pytest
from PIL import Image

from scripts.anomaly_gen import evaluate_mask_refinement


class TestEvaluateMaskRefinement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up fake data here. The dir structure:
        # temp_dir/
        # ├── image
        # │   ├── image.png
        # |   ...
        # ├── mask
        # │   ├── image.png
        # |   ...
        # └── output_dir
        temp_dir = tempfile.TemporaryDirectory()

        # Make dirs.
        image_dir = pathlib.Path(temp_dir.name, "image")
        mask_dir = pathlib.Path(temp_dir.name, "mask")
        output_dir = pathlib.Path(temp_dir.name, "output_dir")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        for i in range(5):
            image = Image.fromarray(np.zeros((384, 640, 3), dtype=np.uint8), mode="RGB")
            image.save(image_dir / f"image_{i}.png")
            mask = Image.fromarray(np.full((384, 640), 255, dtype=np.uint8), mode="L")
            mask.save(mask_dir / f"image_{i}.png")

        cls.temp_dir = temp_dir

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_evaluate_mask_refinement(self):
        """Test the evaluation of mask refinement."""
        # Prepare.
        args = argparse.Namespace(
            image_dir=str(pathlib.Path(self.temp_dir.name, "image")),
            mask_dir=str(pathlib.Path(self.temp_dir.name, "mask")),
            output_dir=str(pathlib.Path(self.temp_dir.name, "output_dir")),
            checkpoint_path=None,
            crop_ratio=2.0,
            dilate_sizes=[7, 9, 11, 13],
            rectangular_mask=False,
            strength=1.0,
            fallback_ratio=0.5,
        )

        # Run.
        evaluate_mask_refinement.main(args)

        # Verify.
        self.assertTrue(
            pathlib.Path(
                self.temp_dir.name, "output_dir", "evaluation_metrics.json"
            ).exists()
        )
        self.assertTrue(
            pathlib.Path(
                self.temp_dir.name, "output_dir", "fallback_records.csv"
            ).exists()
        )
