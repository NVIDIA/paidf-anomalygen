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
"""Unit tests for scripts/utilities/verify_output.sh (QC 2026-07-06 #15).

Regression: with an empty input JSONL, `grep -c` exited 1 under `set -e`
and the script died with a bare exit code and no diagnostic at all.
"""
from tests.utilities.util import make_png, run_script, write_jsonl


def _sdg_output(tmp_path, n_rows, n_images=None):
    out = tmp_path / "sdg"
    (out / "reconstructed_image").mkdir(parents=True)
    csv_lines = ["anomaly_type,output_filename"]
    csv_lines += [f"stain,stain_{i:05d}.png" for i in range(n_rows)]
    (out / "SDG_result.csv").write_text("\n".join(csv_lines) + "\n")
    for i in range(n_rows if n_images is None else n_images):
        make_png(out / "reconstructed_image" / f"stain_{i:05d}.png", (2, 2))
    return out


def _jsonl(tmp_path, n):
    path = tmp_path / "in.jsonl"
    write_jsonl(path, [{"i": i} for i in range(n)])
    return path


def test_counts_match_passes(tmp_path):
    r = run_script("verify_output.sh",
                   _jsonl(tmp_path, 2), _sdg_output(tmp_path, 2))
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_count_mismatch_fails(tmp_path):
    r = run_script("verify_output.sh",
                   _jsonl(tmp_path, 2), _sdg_output(tmp_path, 2, n_images=1))
    assert r.returncode == 1
    assert "count mismatch" in r.stderr


def test_empty_jsonl_gives_diagnostic(tmp_path):
    empty = tmp_path / "in.jsonl"
    empty.write_text("\n   \n")
    r = run_script("verify_output.sh", empty, _sdg_output(tmp_path, 0))
    assert r.returncode == 1
    assert "no non-empty lines" in r.stderr


def test_missing_jsonl_gives_diagnostic(tmp_path):
    r = run_script("verify_output.sh",
                   tmp_path / "absent.jsonl", _sdg_output(tmp_path, 0))
    assert r.returncode == 1
    assert "not found" in r.stderr
