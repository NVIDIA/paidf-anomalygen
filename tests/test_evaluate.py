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
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

metrics_utils = importlib.import_module("cosmos_predict2.metrics.utils")


def _write_rgb(path: Path, size, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_mask(path: Path, size, offset=0):
    width, height = size
    arr = np.zeros((height, width), dtype=np.uint8)
    x0 = min(width - 8, 4 + offset)
    y0 = min(height - 8, 4 + offset)
    arr[y0 : y0 + min(12, height - y0), x0 : x0 + min(12, width - x0)] = 255
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

    return real_root, gen_root


@pytest.mark.parametrize("backbone", ["cradio_v3_base"])
@pytest.mark.parametrize(
    "image_size", [(256, 256), (512, 512), (1024, 1024), (256, 512), (512, 256), (128, 1024), (1024, 128),],
)
def test_compute_kpi_end_to_end(tmp_path, backbone, image_size):
    anomaly_types = [["SEM_IC", "crack"], ["SEM_IC", "scratch"]]
    real_root, gen_root = _prepare_dataset(
        tmp_path / "compute_kpi", anomaly_types, real_count=4, gen_count=4, image_size=image_size,
    )

    real_dict = metrics_utils.load_real_images(str(real_root), anomaly_types)
    gen_dict = metrics_utils.load_generated_images(str(gen_root), anomaly_types)

    kpis = metrics_utils.compute_kpi(real_dict, gen_dict)

    expected_keys = {f"{texture}+{anomaly}" for texture, anomaly in anomaly_types}
    expected_keys.add("Average")
    assert set(kpis.keys()) == expected_keys

    for key in expected_keys:
        assert f"{backbone}_fid" in kpis[key]
        assert isinstance(kpis[key][f"{backbone}_fid"], float)


def test_log_kpi_table_outputs_expected_table(tmp_path):
    sample = {
        "SEM_IC+crack": {"cradio_v3_base_fid": 12.3456},
        "Average": {"cradio_v3_base_fid": 12.3456},
    }

    log_path = tmp_path / "eval_stdout.log"
    sink_id = metrics_utils.log.logger.add(log_path, level="INFO")
    try:
        metrics_utils.log_kpi_table(sample)
    finally:
        metrics_utils.log.logger.remove(sink_id)

    out = log_path.read_text()

    assert "KPI Results:" in out
    assert "SEM_IC+crack" in out
    assert "Average" in out
    assert "cradio_v3_base_fid" in out
    assert "12.3456" in out


def test_compute_kpi_skips_fid_with_insufficient_generated_samples(tmp_path, monkeypatch):
    anomaly_types = [["SEM_IC", "crack"]]
    image_size = (512, 512)

    real_root, gen_root = _prepare_dataset(
        tmp_path / "insufficient", anomaly_types, real_count=2, gen_count=1, image_size=image_size,
    )

    real_dict = metrics_utils.load_real_images(str(real_root), anomaly_types, image_size)
    gen_dict = metrics_utils.load_generated_images(str(gen_root), anomaly_types, image_size)

    monkeypatch.setattr(
        metrics_utils,
        "compute_correspondence_kpi",
        lambda *args, **kwargs: {
            "SEM_IC+crack": {"nn_score": 0.5, "mnn_score": 0.25},
            "Average": {"nn_score": 0.5, "mnn_score": 0.25},
        },
    )
    # Mock the feature extractor: gen has only 1 image so feats_gen.size(0) is
    # at most 1 → triggers the ≥2 gate that returns FID = None. Avoids loading
    # the real backbone checkpoint (not required by this test's intent).
    monkeypatch.setattr(
        metrics_utils,
        "compute_feats",
        lambda images, **kwargs: torch.zeros(len(images), 1024),
    )

    kpis = metrics_utils.compute_kpi(real_dict, gen_dict)

    assert kpis["SEM_IC+crack"]["cradio_v3_base_fid"] is None
    assert kpis["Average"]["cradio_v3_base_fid"] is None
    assert kpis["SEM_IC+crack"]["nn_score"] == 0.5
    assert kpis["SEM_IC+crack"]["mnn_score"] == 0.25
