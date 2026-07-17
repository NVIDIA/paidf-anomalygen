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

"""Multi-view analogue of test_anomaly_inpaint_prefetch.py.

Covers the DataLoader-worker prefetch path for MultiViewAnomalyInpaintDataset
(per-view image/mask preloading, including the shared-string mask case) and the
B->B*N condition duplication for num_generated_images.
"""

import json

import numpy as np
from PIL import Image

from cosmos_predict2.data.anomaly_gen.multiview_anomaly_dataset import MultiViewAnomalyInpaintDataset
from cosmos_predict2.inference.anomaly_gen.multiview_inpaint_condition import MultiViewAnomalyInpaintCondition


def _rgb(fill, size=32):
    return np.full((size, size, 3), fill, dtype=np.uint8)


def _binary_mask(size=32, blob=slice(8, 24)):
    m = np.zeros((size, size), dtype=np.uint8)
    m[blob, blob] = 255  # single connected component
    return m


def _write(path, array, mode):
    Image.fromarray(array, mode=mode).save(path)
    return str(path)


def test_multiview_dataset_getitem_preloads_per_view_images_and_masks(tmp_path):
    """__getitem__ preloads per-view images and (binarized) masks as lists."""
    img0, img1 = _rgb(50), _rgb(150)
    mask0 = _binary_mask()
    mask1 = _binary_mask(blob=slice(4, 12))
    image_filenames = [
        _write(tmp_path / "v0.png", img0, "RGB"),
        _write(tmp_path / "v1.png", img1, "RGB"),
    ]
    mask_filename = [
        _write(tmp_path / "m0.png", mask0, "L"),
        _write(tmp_path / "m1.png", mask1, "L"),
    ]
    jsonl_path = tmp_path / "input.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "image_filenames": image_filenames,
                "mask_filename": mask_filename,
                "anomaly_type": "PeppermintCandy+bumps",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = MultiViewAnomalyInpaintDataset(str(jsonl_path))
    sample = dataset[0]

    assert sample["loaded_image_mode"] == ["RGB", "RGB"]
    assert sample["loaded_mask_mode"] == ["L", "L"]
    assert len(sample["loaded_image_array"]) == 2
    assert len(sample["loaded_mask_array"]) == 2
    np.testing.assert_array_equal(sample["loaded_image_array"][0], img0)
    np.testing.assert_array_equal(sample["loaded_image_array"][1], img1)
    np.testing.assert_array_equal(sample["loaded_mask_array"][0], mask0)
    np.testing.assert_array_equal(sample["loaded_mask_array"][1], mask1)


def test_multiview_dataset_getitem_shared_string_mask(tmp_path):
    """A shared *string* mask_filename must be normalized to a per-view list in
    __init__ (sort_by_instance_num), so __getitem__ iterates a list of paths --
    NOT the characters of the string. Regression test for the shared-mask flow
    used by the shipped PeppermintCandy generation jsonls."""
    img0, img1 = _rgb(50), _rgb(150)
    mask = _binary_mask()
    image_filenames = [
        _write(tmp_path / "v0.png", img0, "RGB"),
        _write(tmp_path / "v1.png", img1, "RGB"),
    ]
    shared_mask = _write(tmp_path / "shared_mask.png", mask, "L")
    jsonl_path = tmp_path / "input.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "image_filenames": image_filenames,
                "mask_filename": shared_mask,  # shared mask as a bare string
                "anomaly_type": "PeppermintCandy+bumps",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = MultiViewAnomalyInpaintDataset(str(jsonl_path))

    # __init__ replicates the shared string to one entry per view.
    assert dataset.input_data[0]["mask_filename"] == [shared_mask, shared_mask]

    sample = dataset[0]  # must not per-character iterate the string
    assert sample["loaded_mask_mode"] == ["L", "L"]
    assert len(sample["loaded_mask_array"]) == 2
    np.testing.assert_array_equal(sample["loaded_mask_array"][0], mask)
    np.testing.assert_array_equal(sample["loaded_mask_array"][1], mask)


def test_multiview_dataset_binarizes_non_binary_mask(tmp_path):
    """_load_cached_mask binarizes at threshold 127 so downstream gets 0/255."""
    gray = np.full((32, 32), 100, dtype=np.uint8)  # <=127 -> 0
    gray[8:24, 8:24] = 200  # >127 -> 255
    image_filenames = [_write(tmp_path / "v0.png", _rgb(120), "RGB")]
    shared_mask = _write(tmp_path / "gray_mask.png", gray, "L")
    jsonl_path = tmp_path / "input.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "image_filenames": image_filenames,
                "mask_filename": shared_mask,
                "anomaly_type": "PeppermintCandy+bumps",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = MultiViewAnomalyInpaintDataset(str(jsonl_path))
    loaded = dataset[0]["loaded_mask_array"][0]

    assert set(np.unique(loaded).tolist()) <= {0, 255}
    np.testing.assert_array_equal(loaded, np.where(gray > 127, 255, 0).astype(np.uint8))


def test_multiview_condition_duplicates_preloaded_arrays():
    """num_generated_images>1 duplicates the outer B axis (B -> B*N), including
    the preloaded per-view arrays -- mirroring single-view AnomalyInpaintCondition."""
    image_array = np.zeros((32, 32, 3), dtype=np.uint8)
    mask_array = np.zeros((32, 32), dtype=np.uint8)

    condition = MultiViewAnomalyInpaintCondition(
        image_filenames=[["v0.png", "v1.png"]],  # B=1 sample, 2 views
        mask_filename=[["m0.png", "m1.png"]],
        anomaly_type="PeppermintCandy+bumps",
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
    # The per-sample filename lists expand too.
    assert len(condition.image_filenames) == 2
    assert len(condition.mask_filename) == 2
