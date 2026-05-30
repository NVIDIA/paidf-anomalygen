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

import json

import numpy as np
from PIL import Image

from cosmos_predict2.data.anomaly_gen.anomaly_dataset import AnomalyInpaintDataset
from cosmos_predict2.inference.anomaly_gen.inpaint_condition import AnomalyInpaintCondition


def test_anomaly_inpaint_dataset_getitem_preloads_image_and_mask(tmp_path):
    image_array = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )
    mask_array = np.array(
        [
            [0, 255],
            [255, 0],
        ],
        dtype=np.uint8,
    )
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    jsonl_path = tmp_path / "input.jsonl"

    Image.fromarray(image_array, mode="RGB").save(image_path)
    Image.fromarray(mask_array, mode="L").save(mask_path)
    jsonl_path.write_text(
        json.dumps(
            {
                "image_filename": str(image_path),
                "mask_filename": str(mask_path),
                "anomaly_type": "metal_surface+MT_Blowhole",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = AnomalyInpaintDataset(str(jsonl_path))
    sample = dataset[0]

    assert sample["loaded_image_mode"] == "RGB"
    assert sample["loaded_mask_mode"] == "L"
    np.testing.assert_array_equal(sample["loaded_image_array"], image_array)
    np.testing.assert_array_equal(sample["loaded_mask_array"], mask_array)


def test_anomaly_inpaint_condition_duplicates_preloaded_arrays():
    image_array = np.zeros((2, 2, 3), dtype=np.uint8)
    mask_array = np.zeros((2, 2), dtype=np.uint8)

    condition = AnomalyInpaintCondition(
        image_filename="image.png",
        mask_filename="mask.png",
        anomaly_type="metal_surface+MT_Blowhole",
        guidance=1.5,
        num_steps=35,
        num_generated_images=2,
        crop_and_paste=True,
        crop_ratio=2.0,
        crop_grid_X="none",
        crop_grid_Y="none",
        poisson_blend=False,
        shift_values="0,0",
        rotation_angle=0,
        morph_operation="none",
        loaded_image_array=image_array,
        loaded_image_mode="RGB",
        loaded_mask_array=mask_array,
        loaded_mask_mode="L",
    )

    assert len(condition.loaded_image_array) == 2
    assert len(condition.loaded_mask_array) == 2
    assert condition.loaded_image_mode == ["RGB", "RGB"]
    assert condition.loaded_mask_mode == ["L", "L"]
    np.testing.assert_array_equal(condition.loaded_image_array[0], image_array)
    np.testing.assert_array_equal(condition.loaded_image_array[1], image_array)
    np.testing.assert_array_equal(condition.loaded_mask_array[0], mask_array)
    np.testing.assert_array_equal(condition.loaded_mask_array[1], mask_array)
