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
"""Unit tests for scripts/utilities/validate_jsonl.py (QC 2026-07-06 #10).

Regression: a config without dataloader_train.dataset.anomaly_types, or a
JSONL line missing a key, used to crash with a raw KeyError traceback
instead of the fail-fast message the script promises.
"""
import yaml

from tests.utilities.util import run_script, write_jsonl

GOOD_CFG = {"dataloader_train": {"dataset": {"anomaly_types": [["Tex", "scratch"]]}}}


def _ckpt(tmp_path, cfg):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "ag_config.yaml").write_text(yaml.safe_dump(cfg))
    return ckpt


def _entry(tmp_path, drop_key=None, **overrides):
    img = tmp_path / "img.png"
    mask = tmp_path / "mask.png"
    img.touch()
    mask.touch()
    entry = {"anomaly_type": "Tex+scratch",
             "image_filename": str(img),
             "mask_filename": str(mask)}
    entry.update(overrides)
    if drop_key:
        del entry[drop_key]
    return entry


def _write_jsonl(tmp_path, entries):
    path = tmp_path / "in.jsonl"
    write_jsonl(path, entries)
    return path


def test_supported_types_pass(tmp_path):
    jsonl = _write_jsonl(tmp_path, [_entry(tmp_path)])
    r = run_script("validate_jsonl.py", _ckpt(tmp_path, GOOD_CFG), jsonl)
    assert r.returncode == 0, r.stderr


def test_unsupported_type_fails(tmp_path):
    jsonl = _write_jsonl(tmp_path, [_entry(tmp_path, anomaly_type="Tex+crack")])
    r = run_script("validate_jsonl.py", _ckpt(tmp_path, GOOD_CFG), jsonl)
    assert r.returncode == 1
    assert "not in the checkpoint" in r.stderr


def test_malformed_config_gives_clean_error(tmp_path):
    jsonl = _write_jsonl(tmp_path, [_entry(tmp_path)])
    r = run_script("validate_jsonl.py", _ckpt(tmp_path, {"unexpected": {}}), jsonl)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "anomaly_types" in r.stderr


def test_missing_entry_key_gives_clean_error(tmp_path):
    jsonl = _write_jsonl(tmp_path, [_entry(tmp_path, drop_key="anomaly_type")])
    r = run_script("validate_jsonl.py", _ckpt(tmp_path, GOOD_CFG), jsonl)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "missing key(s): anomaly_type" in r.stderr


def test_null_anomaly_types_gives_clean_error(tmp_path):
    cfg = {"dataloader_train": {"dataset": {"anomaly_types": None}}}
    jsonl = _write_jsonl(tmp_path, [_entry(tmp_path)])
    r = run_script("validate_jsonl.py", _ckpt(tmp_path, cfg), jsonl)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "anomaly_types" in r.stderr


def test_non_object_jsonl_line_gives_clean_error(tmp_path):
    ckpt = _ckpt(tmp_path, GOOD_CFG)
    jsonl = tmp_path / "in.jsonl"
    jsonl.write_text("[1, 2]\n")
    r = run_script("validate_jsonl.py", ckpt, jsonl)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "not a JSON object" in r.stderr
