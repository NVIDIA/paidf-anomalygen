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

import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

filter_module = importlib.import_module("scripts.anomaly_gen.filter")
metrics_utils = importlib.import_module("cosmos_predict2.metrics.utils")


def _write_rgb(path, size, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_mask(path, size, offset=0):
    width, height = size
    arr = np.zeros((height, width), dtype=np.uint8)
    x0 = min(width - 3, 1 + offset)
    y0 = min(height - 3, 1 + offset)
    arr[y0 : y0 + min(4, height - y0), x0 : x0 + min(4, width - x0)] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def _prepare_dataset(base_dir, anomaly_types, real_count, gen_count, image_size):
    real_root = base_dir / "real_dataset"
    gen_root = base_dir / "generated_dataset"
    recon_dir = gen_root / "reconstructed_image"
    mask_dir_gen = gen_root / "original_mask"
    orig_dir_gen = gen_root / "original_image"

    for texture, anomaly in anomaly_types:
        img_dir = real_root / texture / "anomaly_image" / anomaly
        mask_dir = real_root / texture / "mask" / anomaly
        for idx in range(real_count):
            filename = f"{texture}_{anomaly}_{idx:05d}.png"
            _write_rgb(
                img_dir / filename, image_size, ((30 + idx * 17) % 255, (90 + idx * 13) % 255, (140 + idx * 19) % 255,)
            )
            _write_mask(mask_dir / filename.replace(".png", "_mask.png"), image_size, offset=idx)

    for texture, anomaly in anomaly_types:
        for idx in range(gen_count):
            filename = f"{texture}+{anomaly}_sample_{idx:05d}.png"
            _write_rgb(
                recon_dir / filename, image_size, ((50 + idx * 23) % 255, (80 + idx * 11) % 255, (20 + idx * 7) % 255,)
            )
            _write_mask(mask_dir_gen / filename, image_size, offset=idx)
            _write_rgb(
                orig_dir_gen / filename,
                image_size,
                ((60 + idx * 5) % 255, (70 + idx * 9) % 255, (100 + idx * 3) % 255,),
            )

    return real_root, gen_root, recon_dir


def test_filter_topk_basic():
    scores = np.array([0.1, 0.3, 0.2, 0.4])
    keep_idx, drop_idx = filter_module.filter_topk(scores, drop_ratio=0.25)

    assert list(keep_idx) == [3, 1, 2]
    assert list(drop_idx) == [0]


def test_filter_topk_drop_all():
    keep_idx, drop_idx = filter_module.filter_topk(np.array([1.0, 0.5]), drop_ratio=1.0)

    assert list(keep_idx) == []
    assert set(drop_idx) == {0, 1}


@pytest.mark.parametrize("backbone", ["cradio_v3_base"])
@pytest.mark.parametrize(
    "image_size", [(256, 256), (512, 512), (1024, 1024), (256, 512), (512, 256), (128, 1024), (1024, 128),],
)
def test_filter_generated_images_end_to_end(tmp_path, backbone, image_size):
    anomaly_types = [["SEM_IC", "crack"], ["SEM_IC", "scratch"]]
    real_count = 20
    gen_count = 20

    base_dir = tmp_path / "end_to_end"
    real_root, gen_root, recon_dir = _prepare_dataset(
        base_dir, anomaly_types, real_count=real_count, gen_count=gen_count, image_size=image_size,
    )

    real_dict = metrics_utils.load_real_images(str(real_root), anomaly_types)
    gen_dict = metrics_utils.load_generated_images(str(gen_root), anomaly_types)

    output_dir = base_dir / "filter"
    result_dict, kept_names, dropped_names = filter_module.filter_generated_images(
        gen_dict=gen_dict, real_dict=real_dict, output_path=str(output_dir), drop_ratio=0.5,
    )

    sdg_csv = gen_root / "SDG_result.csv"
    pd.DataFrame(
        [
            {"output_filename": str(recon_dir / name)}
            for idx, name in enumerate(sorted(p.name for p in recon_dir.iterdir()))
        ]
    ).to_csv(sdg_csv, index=False)

    filter_module.save_filtered_sdg_csv(
        generated_path=str(gen_root),
        output_path=str(output_dir),
        kept_basenames=kept_names,
        dropped_basenames=dropped_names,
    )

    keep_csv = output_dir / "keep" / "SDG_result.csv"
    drop_csv = output_dir / "drop" / "SDG_result.csv"
    assert keep_csv.exists()
    assert drop_csv.exists()

    keep_df = pd.read_csv(keep_csv)
    drop_df = pd.read_csv(drop_csv)
    assert set(keep_df["output_filename"].apply(lambda p: Path(p).name)) == set(kept_names)
    assert set(drop_df["output_filename"].apply(lambda p: Path(p).name)) == set(dropped_names)

    expected_files = {
        f"{texture}+{anomaly}_sample_{idx:05d}.png" for texture, anomaly in anomaly_types for idx in range(gen_count)
    }

    keep_files = {p.name for p in (output_dir / "keep" / "reconstructed_image").iterdir()}
    drop_files = {p.name for p in (output_dir / "drop" / "reconstructed_image").iterdir()}

    assert keep_files | drop_files == expected_files
    assert keep_files & drop_files == set()
    assert len(kept_names) + len(dropped_names) == len(expected_files)
    assert set(kept_names) == keep_files
    assert set(dropped_names) == drop_files

    for texture, anomaly in anomaly_types:
        key = f"{texture}+{anomaly}"
        scores = result_dict[key][f"{backbone}_giqa"]
        assert len(scores) == gen_count
        assert all(isinstance(score, float) for score in scores)


@pytest.mark.parametrize("drop_ratio", [0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
@pytest.mark.parametrize("backbone", ["cradio_v3_base"])
def test_filter_generated_images_drop_ratio_end_to_end(tmp_path, drop_ratio, backbone):
    anomaly_types = [["SEM_IC", "crack"], ["SEM_IC", "scratch"]]
    image_size = (512, 512)
    real_count = 20
    gen_count = 20

    base_dir = tmp_path / f"drop_ratio_{str(drop_ratio).replace('.', '_')}"
    real_root, gen_root, _ = _prepare_dataset(
        base_dir, anomaly_types, real_count=real_count, gen_count=gen_count, image_size=image_size,
    )

    real_dict = metrics_utils.load_real_images(str(real_root), anomaly_types)
    gen_dict = metrics_utils.load_generated_images(str(gen_root), anomaly_types)

    output_dir = base_dir / "filter_drop_ratio"
    _, kept_names, dropped_names = filter_module.filter_generated_images(
        gen_dict=gen_dict,
        real_dict=real_dict,
        output_path=str(output_dir),
        drop_ratio=drop_ratio,
        backbone_name=backbone,
    )

    per_anomaly_total = gen_count
    expected_keep_per_anomaly = int(round((1 - drop_ratio) * per_anomaly_total))
    expected_total_keep = expected_keep_per_anomaly * len(anomaly_types)
    total_samples = per_anomaly_total * len(anomaly_types)

    assert len(kept_names) == expected_total_keep
    assert len(dropped_names) == total_samples - expected_total_keep

    keep_dir = output_dir / "keep" / "reconstructed_image"
    drop_dir = output_dir / "drop" / "reconstructed_image"
    assert len(list(keep_dir.iterdir())) == expected_total_keep
    assert len(list(drop_dir.iterdir())) == total_samples - expected_total_keep

    for texture, anomaly in anomaly_types:
        key_prefix = f"{texture}+{anomaly}"
        kept_for_key = [name for name in kept_names if name.startswith(key_prefix)]
        dropped_for_key = [name for name in dropped_names if name.startswith(key_prefix)]
        assert len(kept_for_key) == expected_keep_per_anomaly
        assert len(dropped_for_key) == per_anomaly_total - expected_keep_per_anomaly


@pytest.mark.parametrize(
    "rotation_range, rotation_step", [((0, 0), 15), ((-15, 15), 15), ((-30, 45), 30),],
)
@pytest.mark.parametrize("backbone", ["cradio_v3_base"])
def test_filter_generated_images_rotation_aug(tmp_path, rotation_range, rotation_step, backbone):
    base_dir = tmp_path / f"rotation_aug_{rotation_range[0]}_{rotation_range[1]}_{rotation_step}"
    output_dir = base_dir / "filter_rotation_aug"
    log_file = output_dir / "filter_stdout.log"
    filter_module.log.init_loguru_file(log_file)

    anomaly_types = [["SEM_IC", "crack"], ["SEM_IC", "scratch"]]
    image_size = (512, 512)
    real_count = 20
    gen_count = 20

    real_root, gen_root, _ = _prepare_dataset(
        base_dir, anomaly_types, real_count=real_count, gen_count=gen_count, image_size=image_size,
    )

    real_dict = metrics_utils.load_real_images(str(real_root), anomaly_types)
    gen_dict = metrics_utils.load_generated_images(str(gen_root), anomaly_types)

    result_dict, kept_names, dropped_names = filter_module.filter_generated_images(
        gen_dict=gen_dict,
        real_dict=real_dict,
        output_path=str(output_dir),
        drop_ratio=0.4,
        backbone_name=backbone,
        rotation_range=rotation_range,
        rotation_step=rotation_step,
    )

    log_file = output_dir / "filter_stdout.log"
    assert log_file.exists()
    log_content = log_file.read_text()

    if rotation_range is None:
        aug_factor = 1
    else:
        angles = np.arange(
            rotation_range[0], rotation_range[1] + np.sign(rotation_step) * rotation_step, rotation_step, dtype=int,
        )
        if 0 not in angles:
            angles = np.sort(np.append(angles, 0))
        aug_factor = len(angles)

    for texture, anomaly in anomaly_types:
        expected_real = real_count * aug_factor
        expected_generated = gen_count
        expected_log_text = (
            f"[{texture}+{anomaly}] feature counts -> real: {expected_real}, generated: {expected_generated}"
        )
        assert expected_log_text in log_content


@pytest.mark.parametrize("backbone", ["cradio_v3_base"])
def test_compute_sample_wise_kpi_populates_scores(tmp_path, backbone):
    anomaly_types = [["SEM_IC", "crack"], ["SEM_IC", "scratch"]]
    image_size = (512, 512)
    real_count = 20
    gen_count = 1

    base_dir = tmp_path / "compute_kpi"
    real_root, gen_root, _ = _prepare_dataset(
        base_dir, anomaly_types, real_count=real_count, gen_count=gen_count, image_size=image_size,
    )

    real_dict = metrics_utils.load_real_images(str(real_root), anomaly_types)
    gen_dict = metrics_utils.load_generated_images(str(gen_root), anomaly_types)

    updated_dict = metrics_utils.compute_sample_wise_kpi(
        gen_dict, real_dict, K=1, rotation_range=None, rotation_step=15, backbone_name=backbone,
    )

    for texture, anomaly in anomaly_types:
        key = f"{texture}+{anomaly}"
        assert f"{backbone}_giqa" in updated_dict[key]
        giqas = updated_dict[key][f"{backbone}_giqa"]
        assert len(giqas) == gen_count


@pytest.mark.parametrize("backbone", ["cradio_v3_base"])
def test_save_filter_result_csv(tmp_path, backbone):
    output_dir = tmp_path / "results"
    output_dir.mkdir()

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_one = img_dir / "one.png"
    img_two = img_dir / "two.png"
    img_one.write_text("one")
    img_two.write_text("two")

    gen_dict = {
        "anomalyA": {"img_path": [str(img_one)], f"{backbone}_giqa": [0.8],},
        "anomalyB": {"img_path": [str(img_two)],},
    }

    filter_module.save_filter_result_csv(gen_dict, str(output_dir), backbone)

    csv_path = output_dir / "filter_result.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    expected = pd.DataFrame(
        [
            {"anomaly_key": "anomalyA", "filename": img_one.name, f"{backbone}_giqa": 0.8},
            {"anomaly_key": "anomalyB", "filename": img_two.name, f"{backbone}_giqa": np.nan},
        ]
    )

    df_sorted = df.sort_values(["anomaly_key"]).reset_index(drop=True)
    expected_sorted = expected.sort_values(["anomaly_key"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df_sorted, expected_sorted)


def test_save_filtered_sdg_csv(tmp_path):
    gen_dir = tmp_path / "generated"
    recon_dir = gen_dir / "reconstructed_image"
    recon_dir.mkdir(parents=True)
    sdg_path = gen_dir / "SDG_result.csv"

    keep_file = recon_dir / "keep.png"
    drop_file = recon_dir / "drop.png"
    keep_file.write_text("keep")
    drop_file.write_text("drop")

    df = pd.DataFrame(
        [{"output_filename": str(keep_file), "metric": 1.0}, {"output_filename": str(drop_file), "metric": 0.1},]
    )
    df.to_csv(sdg_path, index=False)

    output_dir = tmp_path / "filtered"
    for split in ("keep", "drop"):
        (output_dir / split).mkdir(parents=True)

    filter_module.save_filtered_sdg_csv(
        generated_path=str(gen_dir),
        output_path=str(output_dir),
        kept_basenames=[keep_file.name],
        dropped_basenames=[drop_file.name],
    )

    keep_csv = output_dir / "keep" / "SDG_result.csv"
    drop_csv = output_dir / "drop" / "SDG_result.csv"
    assert keep_csv.exists()
    assert drop_csv.exists()

    keep_df = pd.read_csv(keep_csv)
    drop_df = pd.read_csv(drop_csv)

    assert list(keep_df["output_filename"]) == [str(keep_file)]
    assert list(drop_df["output_filename"]) == [str(drop_file)]


def test_save_filtered_sdg_csv_missing_file(tmp_path):
    gen_dir = tmp_path / "generated"
    gen_dir.mkdir()

    output_dir = tmp_path / "filtered"
    for split in ("keep", "drop"):
        (output_dir / split).mkdir(parents=True)

    filter_module.save_filtered_sdg_csv(
        generated_path=str(gen_dir),
        output_path=str(output_dir),
        kept_basenames=["foo.png"],
        dropped_basenames=["bar.png"],
    )

    keep_csv = output_dir / "keep" / "SDG_result.csv"
    drop_csv = output_dir / "drop" / "SDG_result.csv"

    assert not keep_csv.exists()
    assert not drop_csv.exists()
