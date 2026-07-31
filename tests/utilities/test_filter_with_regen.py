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
"""Unit tests for the pure planning helpers in
scripts/utilities/filter_with_regen.py (QC 2026-07-06 #13).

Regression: the regen target used to be derived only from the rows already
present in the source bucket, so a bucket left short (e.g. by an interrupted
SDG) was never topped up to --num-sdg; and samples with no nn_score were
silently dropped from both keep and fallback.
"""
import json
import pathlib

from scripts.utilities.filter_with_regen import (
    compute_target_alloc,
    partition_source,
)
from tests.utilities.util import run_script


def _rows(**counts):
    rows = []
    for defect, n in counts.items():
        for i in range(n):
            rows.append({"anomaly_type": defect,
                         "output_filename": f"/sdg/{defect}_{i:05d}.png"})
    return rows


def test_target_from_source_bucket_matches_num_sdg():
    alloc, warnings = compute_target_alloc(_rows(stain=3, scratch=1), None, 4)
    assert dict(alloc) == {"stain": 3, "scratch": 1}
    assert warnings == []


def test_short_source_bucket_without_allocation_warns():
    alloc, warnings = compute_target_alloc(_rows(stain=3, scratch=1), None, 6)
    assert sum(alloc.values()) == 4
    assert len(warnings) == 1
    assert "cannot top it up" in warnings[0]


def test_oversized_source_bucket_does_not_claim_shortfall():
    _, warnings = compute_target_alloc(_rows(stain=8), None, 5)
    assert len(warnings) == 1
    assert "sums to 8" in warnings[0]
    assert "cannot top it up" not in warnings[0]


def test_allocation_tops_up_short_bucket():
    alloc, warnings = compute_target_alloc(
        _rows(stain=3, scratch=1), {"stain": 4, "scratch": 2}, 6)
    assert dict(alloc) == {"stain": 4, "scratch": 2}
    assert warnings == []


def test_allocation_num_sdg_mismatch_warns():
    _, warnings = compute_target_alloc(_rows(stain=3), {"stain": 4}, 6)
    assert len(warnings) == 1
    assert "sums to 4" in warnings[0]
    assert "cannot top it up" not in warnings[0]


def test_allocation_missing_source_defect_warns_about_drop():
    _, warnings = compute_target_alloc(
        _rows(stain=2, scratch=2), {"stain": 4}, 4)
    assert any("scratch" in w and "dropped" in w for w in warnings)


def test_partition_source_splits_by_threshold_and_counts_nan():
    rows = _rows(stain=3)
    nn = {rows[0]["output_filename"]: 0.9,
          rows[1]["output_filename"]: 0.1}
    # rows[2] absent from per_sample.csv → NaN
    passing, dropped, nan_rows = partition_source(
        rows, nn, 0.4, pathlib.Path("/searched"))
    assert [e["nn"] for e in passing["stain"]] == [0.9]
    assert [e["nn"] for e in dropped["stain"]] == [0.1]
    assert nan_rows == 1
    entry = passing["stain"][0]
    assert entry["source_attempt"] == 0
    assert entry["prev_nn"] == 0.9
    assert entry["src_basename"] == "stain_00000.png"
    assert entry["src_dir"] == pathlib.Path("/searched")


def test_partition_source_counts_explicit_nan_scores():
    rows = _rows(stain=1)
    passing, dropped, nan_rows = partition_source(
        rows, {rows[0]["output_filename"]: float("nan")},
        0.4, pathlib.Path("/searched"))
    assert nan_rows == 1
    assert not passing and not dropped


# --- CLI guard: --allocation file validation happens before any heavy work ---

def _cli_args(tmp_path):
    searched = tmp_path / "searched"
    searched.mkdir()
    (searched / "SDG_result.csv").write_text(
        "anomaly_type,output_filename\nstain,/sdg/stain_00000.png\n")
    per_sample = tmp_path / "per_sample.csv"
    per_sample.write_text("anomaly_type,path,nn_score\n"
                          "stain,/sdg/stain_00000.png,0.9\n")
    return ["--searched-dir", searched, "--per-sample-csv", per_sample,
            "--threshold", 0.4, "--num-sdg", 1,
            "--rounds-dir", tmp_path / "rounds",
            "--regens-dir", tmp_path / "regens",
            "--real-path", tmp_path, "--anomaly-types", "stain",
            "--checkpoint-dir", tmp_path, "--step", 100,
            "--dataset-dir", tmp_path,
            "--defect-spec", tmp_path / "spec.jsonl"]


def test_cli_missing_allocation_file_fails_cleanly(tmp_path):
    r = run_script("filter_with_regen.py", *_cli_args(tmp_path),
                   "--allocation", tmp_path / "absent.json")
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "--allocation" in r.stderr and "not found" in r.stderr


def test_cli_malformed_allocation_file_fails_cleanly(tmp_path):
    bad = tmp_path / "allocation.json"
    bad.write_text(json.dumps({"stain": None}))
    r = run_script("filter_with_regen.py", *_cli_args(tmp_path),
                   "--allocation", bad)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "not a valid allocation.json" in r.stderr


def test_cli_non_dict_allocation_fails_cleanly(tmp_path):
    bad = tmp_path / "allocation.json"
    bad.write_text("[1, 2]")
    r = run_script("filter_with_regen.py", *_cli_args(tmp_path),
                   "--allocation", bad)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "not a valid allocation.json" in r.stderr


def test_cli_negative_allocation_count_fails_cleanly(tmp_path):
    bad = tmp_path / "allocation.json"
    bad.write_text(json.dumps({"stain": -1}))
    r = run_script("filter_with_regen.py", *_cli_args(tmp_path),
                   "--allocation", bad)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "not a valid allocation.json" in r.stderr


def test_cli_boolean_allocation_count_fails_cleanly(tmp_path):
    # isinstance(True, int) is True — booleans must still be rejected.
    bad = tmp_path / "allocation.json"
    bad.write_text(json.dumps({"stain": True}))
    r = run_script("filter_with_regen.py", *_cli_args(tmp_path),
                   "--allocation", bad)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "not a valid allocation.json" in r.stderr


def test_cli_allocation_directory_says_not_a_file(tmp_path):
    r = run_script("filter_with_regen.py", *_cli_args(tmp_path),
                   "--allocation", tmp_path)  # a directory
    assert r.returncode == 1
    assert "is not a file" in r.stderr
