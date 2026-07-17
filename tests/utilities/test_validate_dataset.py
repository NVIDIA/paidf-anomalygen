# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for scripts/utilities/validate_dataset.py (QC 2026-07-06 #3).

Regression: the validator used to print pairing issues as warnings and exit
0, letting a dataset that would crash training mid-iteration pass the gate.
"""
from tests.utilities.util import make_png, run_script


def _make_dataset(root, mask_stem):
    img_dir = root / "Tex" / "anomaly_image" / "scratch"
    mask_dir = root / "Tex" / "mask" / "scratch"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    make_png(img_dir / "img_001.png")
    make_png(mask_dir / f"{mask_stem}.png", mode="L")


def test_paired_dataset_passes(tmp_path):
    _make_dataset(tmp_path, "img_001_mask")
    r = run_script("validate_dataset.py", tmp_path)
    assert r.returncode == 0, r.stderr


def test_unpaired_mask_fails(tmp_path):
    _make_dataset(tmp_path, "img_001_masked")  # wrong suffix
    r = run_script("validate_dataset.py", tmp_path)
    assert r.returncode == 1
    assert "issue(s) found" in r.stderr


def test_empty_dataset_fails(tmp_path):
    r = run_script("validate_dataset.py", tmp_path)
    assert r.returncode == 1
    assert "no anomaly types" in r.stderr
