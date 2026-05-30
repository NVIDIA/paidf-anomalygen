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

pytest tests/test_pseudo_labeling.py
"""

import argparse
import csv
import json
import os
import pathlib
import tempfile
import unittest

import numpy as np
import pytest as pytest
from PIL import Image

from pseudo_label import bbox as pl_bbox
from pseudo_label import mask as pl_mask
from scripts.anomaly_gen import pseudo_label


class TestPseudoLabeling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up fake data here. The dir structure:
        # temp_dir/
        # ├── original_image
        # │   ├── image.png
        # |   ...
        # ├── original_mask
        # │   ├── image.png
        # |   ...
        # ├── reconstructed_image
        # │   ├── image.png
        # |   ...
        # └── result.csv
        temp_dir = tempfile.TemporaryDirectory()

        # Make dirs.
        original_image_dir = pathlib.Path(temp_dir.name, "original_image")
        original_mask_dir = pathlib.Path(temp_dir.name, "original_mask")
        reconstructed_image_dir = pathlib.Path(temp_dir.name, "reconstructed_image")
        os.makedirs(original_image_dir, exist_ok=True)
        os.makedirs(original_mask_dir, exist_ok=True)
        os.makedirs(reconstructed_image_dir, exist_ok=True)
        # Fake images and CSV file.
        with open(pathlib.Path(temp_dir.name, "result.csv"), "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
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
                ]
            )
            for i, mode in enumerate(["RGB", "L"]):
                shape = (3, 3, 3) if mode == "RGB" else (3, 3)
                original_image = Image.fromarray(
                    np.zeros(shape, dtype=np.uint8), mode=mode
                )
                original_image.save(original_image_dir / f"image_{i}.png")
                original_mask = Image.fromarray(
                    np.array(
                        [[0, 255, 0], [255, 255, 255], [0, 255, 0]], dtype=np.uint8
                    ),
                    mode="L",
                )
                original_mask.save(original_mask_dir / f"image_{i}.png")
                reconstructed_image = Image.fromarray(
                    np.ones(shape, dtype=np.uint8), mode=mode
                )
                reconstructed_image.save(reconstructed_image_dir / f"image_{i}.png")
                writer.writerow(
                    [
                        f"reconstructed_image/image_{i}.png",
                        f"original_image/image_{i}.png",
                        f"original_mask/image_{i}.png",
                        "SEM_IC+Corrosion",
                        "7.0",
                        "35",
                        "1",
                        "1",
                        "True",
                        "384",
                        "256",
                        "2.0",
                        "False",
                        "-38-28",
                        "142",
                        "none",
                        "9.032803190562575",
                    ]
                )
        cls.temp_dir = temp_dir

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_pseudo_label_integration(self):
        """Test the end-to-end pseudo labeling process."""
        # Prepare.
        args = argparse.Namespace(
            ori_image_dir=str(pathlib.Path(self.temp_dir.name, "original_image")),
            gen_image_dir=str(pathlib.Path(self.temp_dir.name, "reconstructed_image")),
            mask_dir=str(pathlib.Path(self.temp_dir.name, "original_mask")),
            csv_path=str(pathlib.Path(self.temp_dir.name, "result.csv")),
            output_dir=str(pathlib.Path(self.temp_dir.name, "output")),
            no_mask_refinement=False,
            no_caption=True,  # Skip captioning for test.
            # Clustering.
            dbscan_eps=0.2,
            dbscan_min_samples=5,
            # Mask refinement.
            mask_refinement_checkpoint_path=None,
            mask_refinement_strength=1.0,
            mask_refinement_fallback_ratio=0.5,
            # Captioning.
            captioner_prompt_path="",
            captioner_num_gpus=1,
            captioner_temperature=0.01,
            captioner_max_tokens=4096,
            captioner_seed=42,
        )

        # Run.
        pseudo_label.main(args)

        # Verify.
        self.assertTrue(pathlib.Path(self.temp_dir.name, "output").exists())
        # Check classification dir.
        self.assertTrue(
            pathlib.Path(self.temp_dir.name, "output", "classification").exists()
        )
        for i in range(2):
            image = Image.open(
                pathlib.Path(
                    self.temp_dir.name,
                    "output",
                    "classification",
                    "original",
                    f"image_{i}.png",
                )
            )
            np.testing.assert_allclose(
                np.array(image),
                np.zeros((3, 3, 3), dtype=np.uint8)
                if image.mode == "RGB"
                else np.zeros((3, 3), dtype=np.uint8),
            )
            image = Image.open(
                pathlib.Path(
                    self.temp_dir.name,
                    "output",
                    "classification",
                    "SEM_IC+Corrosion",
                    f"image_{i}.png",
                )
            )
            np.testing.assert_allclose(
                np.array(image),
                np.ones((3, 3, 3), dtype=np.uint8)
                if image.mode == "RGB"
                else np.ones((3, 3), dtype=np.uint8),
            )
        with open(
            pathlib.Path(self.temp_dir.name, "output", "classification", "classes.txt")
        ) as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0].strip(), "original")
            self.assertEqual(lines[1].strip(), "SEM_IC+Corrosion")
        # Check COCO JSON.
        # We only test the original COCO JSON here because the refined one is
        # unpredictable.
        with open(
            pathlib.Path(self.temp_dir.name, "output", "ori_coco_annotations.json")
        ) as f:
            coco_data = json.load(f)
            # Annotations.
            self.assertEqual(len(coco_data["annotations"]), 2)
            for i in range(2):
                self.assertEqual(coco_data["annotations"][i]["image_id"], i + 1)
                self.assertEqual(coco_data["annotations"][i]["category_id"], 1)
                self.assertEqual(
                    coco_data["annotations"][i]["segmentation"]["size"], [3, 3]
                )
                self.assertEqual(
                    coco_data["annotations"][i]["segmentation"]["counts"], "11120N0"
                )
                self.assertEqual(coco_data["annotations"][i]["bbox"], [0, 0, 3, 3])
                self.assertEqual(coco_data["annotations"][i]["area"], 5)
                self.assertEqual(coco_data["annotations"][i]["iscrowd"], 0)
            # Images.
            self.assertEqual(len(coco_data["images"]), 2)
            for i in range(2):
                self.assertEqual(coco_data["images"][i]["id"], i + 1)
                self.assertEqual(coco_data["images"][i]["width"], 3)
                self.assertEqual(coco_data["images"][i]["height"], 3)
                self.assertEqual(coco_data["images"][i]["file_name"], f"image_{i}.png")
            # Categories.
            self.assertEqual(len(coco_data["categories"]), 1)
            self.assertEqual(coco_data["categories"][0]["id"], 1)
            self.assertEqual(coco_data["categories"][0]["name"], "SEM_IC+Corrosion")

    def test_get_bboxes(self):
        """Test get_bboxes."""
        # Test with single mask
        mask_array = np.zeros((10, 10), dtype=np.uint8)
        mask_array[2:5, 3:7] = 255  # Rectangle from (3,2) to (6,4), width=4, height=3
        mask = Image.fromarray(mask_array, mode="L")

        bboxes = pl_bbox.get_bboxes(mask)

        self.assertIsInstance(bboxes, list)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(bboxes[0], (3, 2, 4, 3))  # (x, y, width, height)

        # Test with list of masks
        mask_array2 = np.zeros((10, 10), dtype=np.uint8)
        mask_array2[7:9, 1:3] = 255  # Rectangle from (1,7) to (2,8), width=2, height=2
        mask2 = Image.fromarray(mask_array2, mode="L")

        bboxes_list = pl_bbox.get_bboxes([mask, mask2])

        self.assertIsInstance(bboxes_list, list)
        self.assertEqual(len(bboxes_list), 2)
        self.assertEqual(bboxes_list[0], (3, 2, 4, 3))
        self.assertEqual(bboxes_list[1], (1, 7, 2, 2))

    def test_binary_mask_to_rle(self):
        """Test binary_mask_to_rle."""
        # Test simple 3x3 mask
        mask = np.array(
            [[True, False, True], [False, True, False], [True, False, True]],
            dtype=np.bool_,
        )

        rle = pl_mask.binary_mask_to_rle(mask)

        self.assertIsInstance(rle, dict)
        self.assertIn("counts", rle)
        self.assertIn("size", rle)
        self.assertEqual(rle["size"], [3, 3])
        self.assertEqual(rle["counts"], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_coco_rle_roundtrip(self):
        """Test coco_encode_rle and coco_decode_rle."""
        uncompressed_rle = {"counts": [0, 2, 1, 2, 1, 2, 0], "size": [3, 3]}

        encoded_rle = pl_mask.coco_encode_rle(uncompressed_rle)

        self.assertIsInstance(encoded_rle, dict)
        self.assertIn("counts", encoded_rle)
        self.assertIn("size", encoded_rle)
        self.assertEqual(encoded_rle["size"], [3, 3])
        self.assertIsInstance(encoded_rle["counts"], str)

        decoded_mask = pl_mask.coco_decode_rle(encoded_rle)

        self.assertIsInstance(decoded_mask, np.ndarray)
        self.assertEqual(decoded_mask.shape, (3, 3))
        self.assertEqual(decoded_mask.dtype, np.uint8)
