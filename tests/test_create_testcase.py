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

pytest tests/test_create_testcase.py
"""

import argparse
import json
import os
import pathlib
import shutil
import tempfile
import unittest

import numpy as np
import pytest as pytest
from PIL import Image

from cosmos_predict2.inference.anomaly_gen.inpaint_condition import (
    AnomalyInpaintCondition,
)
from scripts.anomaly_gen import create_testcase


class TestCreateTestcase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up fake data here. The dir structure:
        # temp_dir/
        # ├── ok_images
        # │   ├── image.png
        # |   ...
        # └── anomaly_data
        #     └── texture
        #         ├── anomaly_image
        #         |   ├── anomaly_type
        #         |       ├── anomaly_image.png
        #         |   ...
        #         └── mask
        #         |   ├── anomaly_type
        #         |       ├── anomaly_image_mask.png
        #         |   ...
        temp_dir = tempfile.TemporaryDirectory()

        # Make dirs.
        ok_image_dir = pathlib.Path(temp_dir.name, "ok_images")
        os.makedirs(ok_image_dir, exist_ok=True)
        anomaly_data_dir = pathlib.Path(temp_dir.name, "anomaly_data")
        os.makedirs(anomaly_data_dir, exist_ok=True)
        texture_dir = pathlib.Path(anomaly_data_dir, "texture")
        os.makedirs(texture_dir, exist_ok=True)
        anomaly_image_dir = pathlib.Path(texture_dir, "anomaly_image")
        os.makedirs(anomaly_image_dir, exist_ok=True)
        anomaly_image_anomaly_type_dir = pathlib.Path(anomaly_image_dir, "anomaly_type")
        os.makedirs(anomaly_image_anomaly_type_dir, exist_ok=True)
        mask_dir = pathlib.Path(texture_dir, "mask")
        os.makedirs(mask_dir, exist_ok=True)
        mask_anomaly_type_dir = pathlib.Path(mask_dir, "anomaly_type")
        os.makedirs(mask_anomaly_type_dir, exist_ok=True)
        # Fake images.
        for i in range(5):
            ok_image = Image.fromarray(
                np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB"
            )
            ok_image.save(ok_image_dir / f"image_{i}.png")
            anomaly_image = Image.fromarray(
                np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB"
            )
            anomaly_image.save(anomaly_image_anomaly_type_dir / f"image_{i}.png")
            anomaly_mask = Image.fromarray(
                np.zeros((100, 100), dtype=np.uint8), mode="L"
            )
            anomaly_mask.save(mask_anomaly_type_dir / f"image_{i}.png")
        # Global variables.
        cls.test_name = "test"
        cls.temp_dir = temp_dir
        cls.output_dir = pathlib.Path(f"ag_inference/{cls.test_name}")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()
        shutil.rmtree(cls.output_dir)

    def test_create_testcase_integration(self):
        """Test the end-to-end create_testcase process."""
        # Prepare.
        args = argparse.Namespace(
            SDG_RATIO=int(16),
            GUIDANCE=str("7.0"),
            CROP_RATIO=str("2.0"),
            POISSON_BLEND=str("False"),
            dataset_dir=str(pathlib.Path(self.temp_dir.name, "anomaly_data")),
            clean_image_dir=str(pathlib.Path(self.temp_dir.name, "ok_images")),
            name=self.test_name,
            tSNE_sample=False,
            disable_augmentation=False,
        )

        # Run.
        create_testcase.main(args)

        # Verify.
        output_file = list(self.output_dir.glob("*.jsonl"))[0]
        with open(output_file, "r") as fp:
            for line in fp:
                data = json.loads(line)
                input_condition = AnomalyInpaintCondition(**data)

                self.assertEqual(input_condition.anomaly_type, ["texture+anomaly_type"])
                self.assertEqual(input_condition.guidance, [7.0])
                self.assertEqual(input_condition.num_steps, 35)
                self.assertEqual(input_condition.num_generated_images, 1)
                self.assertEqual(input_condition.crop_and_paste, [True])
                self.assertEqual(input_condition.crop_grid_X, ["none"])
                self.assertEqual(input_condition.crop_grid_Y, ["none"])
                self.assertEqual(input_condition.crop_ratio, [2.0])
                self.assertEqual(input_condition.poisson_blend, [False])
                self.assertEqual(input_condition.iteration_generation_max_instance, 5)
