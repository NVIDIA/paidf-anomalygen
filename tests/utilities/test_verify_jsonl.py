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
"""Functional tests for scripts/utilities/verify_jsonl.py (QC 2026-07-06 #11
closed the leaked image file handle; these lock the surrounding resize
behavior).
"""
import json

from PIL import Image

from tests.utilities.util import make_png, run_script, write_jsonl


def _write(tmp_path, img_size, mask_size):
    img = tmp_path / "img.png"
    mask = tmp_path / "mask.png"
    make_png(img, img_size)
    make_png(mask, mask_size, mode="L")
    jsonl = tmp_path / "in.jsonl"
    write_jsonl(jsonl, [{"image_filename": str(img),
                         "mask_filename": str(mask)}])
    return jsonl


def test_mismatched_mask_is_resized_into_cache(tmp_path):
    jsonl = _write(tmp_path, (8, 8), (4, 4))
    cache = tmp_path / "cache"
    r = run_script("verify_jsonl.py", "--jsonl", jsonl, "--cache-dir", cache)
    assert r.returncode == 0, r.stderr
    entry = json.loads(jsonl.read_text())
    assert str(cache) in entry["mask_filename"]
    with Image.open(entry["mask_filename"]) as m:
        assert m.size == (8, 8)


def test_matching_mask_untouched(tmp_path):
    jsonl = _write(tmp_path, (8, 8), (8, 8))
    before = json.loads(jsonl.read_text())
    r = run_script("verify_jsonl.py", "--jsonl", jsonl,
                   "--cache-dir", tmp_path / "cache")
    assert r.returncode == 0, r.stderr
    assert json.loads(jsonl.read_text()) == before


def test_image_handle_is_closed(tmp_path):
    # QC #11 regression guard: the pre-fix code leaked the image fd
    # (Image.open(...).size without a context manager), which CPython
    # reports as "ResourceWarning: unclosed file" when the temporary is
    # collected. Surface warnings and assert none is emitted.
    jsonl = _write(tmp_path, (8, 8), (4, 4))
    r = run_script("verify_jsonl.py", "--jsonl", jsonl,
                   "--cache-dir", tmp_path / "cache",
                   env={"PYTHONWARNINGS": "always::ResourceWarning"})
    assert r.returncode == 0, r.stderr
    assert "ResourceWarning" not in r.stderr


def test_missing_mask_fails(tmp_path):
    jsonl = _write(tmp_path, (8, 8), (4, 4))
    (tmp_path / "mask.png").unlink()
    r = run_script("verify_jsonl.py", "--jsonl", jsonl,
                   "--cache-dir", tmp_path / "cache")
    assert r.returncode == 1
    assert "missing mask" in r.stderr
