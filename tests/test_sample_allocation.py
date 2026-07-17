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
"""Unit tests for the SDG sample-planning stage.

Locks the pure, deterministic planning logic that has no other guard:

  * scripts/utilities/allocate_samples.py  -- uniform / proportional
    (largest-remainder) allocation and the validation-mode "≥1 per defect"
    KPI floor.
  * scripts/utilities/build_amp_samples.py -- (submask, clean) pairing and the
    global n_seeds derivation.
  * scripts/utilities/build_jsonl.py       -- iteration_generation_max_instance
    connected-component auto-detect and the "0 entries -> exit(1)" guard.

allocate_samples exposes pure functions, so those are exercised directly. The
pairing / n_seeds / exit logic in the other two scripts lives inside main(), so
it is driven end-to-end through main() over an on-disk fixture (plus the
importable helpers). No production code is refactored -- the point is to lock
the current, correct behavior, not change it.

Usage:

    pytest tests/test_sample_allocation.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utilities import allocate_samples as A  # noqa: E402
from scripts.utilities import build_amp_samples as B  # noqa: E402
from scripts.utilities import build_jsonl as J  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write_png(path: Path, arr: np.ndarray = None):
    """Write a small valid PNG. Default is a single filled blob."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr is None:
        arr = np.zeros((32, 32), np.uint8)
        arr[8:24, 8:24] = 255
    assert cv2.imwrite(str(path), arr)


def _single_blob():
    a = np.zeros((40, 40), np.uint8)
    a[5:18, 5:18] = 255
    return a


def _two_blobs():
    a = np.zeros((40, 40), np.uint8)
    a[5:14, 5:14] = 255
    a[26:35, 26:35] = 255
    return a


def _run_main(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__name__] + [str(x) for x in argv])
    module.main()


# ===========================================================================
# allocate_samples.py -- uniform allocation
# ===========================================================================
def test_uniform_sum_preserved_and_balanced():
    defects = ["a", "b", "c"]
    assert A._uniform(10, defects) == {"a": 4, "b": 3, "c": 3}
    assert A._uniform(9, defects) == {"a": 3, "b": 3, "c": 3}
    # invariants across a range of totals: sum preserved, spread <= 1, the
    # first (num_sdg % N) defects get the +1.
    for num_sdg in range(0, 40):
        for defs in (["a"], ["a", "b"], ["a", "b", "c", "d"]):
            alloc = A._uniform(num_sdg, defs)
            counts = list(alloc.values())
            assert sum(counts) == num_sdg
            assert max(counts) - min(counts) <= 1
            rem = num_sdg % len(defs)
            for d in defs[:rem]:
                assert alloc[d] == num_sdg // len(defs) + 1


def test_allocate_inference_delegates_to_uniform():
    defects = ["x+a", "y+b", "z+c"]
    assert A.allocate(7, defects, {}, mode="inference") == A._uniform(7, defects)


# ===========================================================================
# allocate_samples.py -- proportional (largest-remainder) allocation
# ===========================================================================
def test_proportional_largest_remainder_known_case():
    # raw: a=2.5 (floor 2, rem .5), b=7.5 (floor 7, rem .5); remainder 1 goes to
    # the first tied type in stable order -> a. Total is preserved.
    assert A._proportional(10, ["a", "b"], {"a": 1, "b": 3}) == {"a": 3, "b": 7}


def test_proportional_sum_preserved_and_monotone():
    defects = ["a", "b", "c"]
    counts = {"a": 1, "b": 2, "c": 7}
    for num_sdg in (1, 5, 10, 13, 50, 101):
        alloc = A._proportional(num_sdg, defects, counts)
        assert sum(alloc.values()) == num_sdg
        # a larger mask count never gets a smaller allocation
        assert alloc["a"] <= alloc["b"] <= alloc["c"]


def test_proportional_raises_when_no_masks():
    with pytest.raises(ValueError, match="mask counts must be > 0"):
        A._proportional(10, ["a", "b"], {"a": 0, "b": 0})


# ===========================================================================
# allocate_samples.py -- validation-mode KPI floor
# ===========================================================================
def test_validation_kpi_floor_raises_on_zero_coverage():
    # b rounds to 0 with num_sdg=2 (raw b = 2*1/101 ≈ 0.02) -> KPI floor trips.
    with pytest.raises(ValueError) as ei:
        A.allocate(2, ["a", "b"], {"a": 100, "b": 1}, mode="validation")
    msg = str(ei.value)
    assert "coverage broken" in msg
    assert "'b'" in msg  # names the starved defect
    assert "Increase num_sdg" in msg


def test_validation_ok_when_all_covered():
    alloc = A.allocate(100, ["a", "b"], {"a": 100, "b": 1}, mode="validation")
    assert sum(alloc.values()) == 100
    assert all(v >= 1 for v in alloc.values())


# ===========================================================================
# allocate_samples.py -- override + guard rails
# ===========================================================================
def test_override_projects_onto_defect_types():
    # unlisted defect -> 0; unknown key ignored (warning to stderr).
    alloc = A.allocate(
        15, ["a", "b", "c"], {}, mode="inference",
        override={"a": 5, "b": 10, "zzz": 99},
    )
    assert alloc == {"a": 5, "b": 10, "c": 0}


def test_override_rejected_in_validation_mode():
    with pytest.raises(ValueError, match="only valid with --mode inference"):
        A.allocate(10, ["a"], {"a": 5}, mode="validation", override={"a": 10})


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(num_sdg=-1, defect_types=["a"], counts={}), "must be >= 0"),
        (dict(num_sdg=5, defect_types=[], counts={}), "non-empty"),
        (dict(num_sdg=5, defect_types=["a"], counts={}, mode="bogus"), "unknown --mode"),
    ],
)
def test_allocate_guard_rails(kwargs, match):
    with pytest.raises(ValueError, match=match):
        A.allocate(**kwargs)


# ===========================================================================
# build_amp_samples.py -- (submask, clean) pairing and n_seeds
# ===========================================================================
def _amp_fixture(tmp_path, *, n_submasks, n_cleans, allocation,
                 texture="metal", anomaly="scratch"):
    full = f"{texture}+{anomaly}"
    dataset_dir = tmp_path / "dataset"
    clean_dir = tmp_path / "clean"
    for i in range(n_submasks):
        _write_png(dataset_dir / texture / "mask" / anomaly / f"sub{i}.png")
    for i in range(n_cleans):
        _write_png(clean_dir / texture / "clean_image" / f"clean{i}.png")
    spec = tmp_path / "defect_spec.jsonl"
    spec.write_text(json.dumps({"defect_type": full, "spatial_dependency": "free"}) + "\n")
    alloc = tmp_path / "allocation.json"
    alloc.write_text(json.dumps({full: allocation}))
    out = tmp_path / "amp_samples.json"
    return dataset_dir, clean_dir, spec, alloc, out, full


def _run_build_amp(tmp_path, monkeypatch, *, n_submasks, n_cleans, allocation, seed=0):
    dataset_dir, clean_dir, spec, alloc, out, full = _amp_fixture(
        tmp_path, n_submasks=n_submasks, n_cleans=n_cleans, allocation=allocation)
    _run_main(B, [
        "--dataset-dir", dataset_dir,
        "--clean-dir", clean_dir,
        "--defect-spec", spec,
        "--allocation", alloc,
        "--output", out,
        "--seed", seed,
    ], monkeypatch)
    records = json.loads(out.read_text())
    n_seeds = int(out.with_suffix(out.suffix + ".n_seeds").read_text())
    return records, n_seeds, full


def test_amp_pairs_unique_when_within_budget(tmp_path, monkeypatch):
    # budget = 3 * 4 = 12; allocation 6 <= budget -> n_seeds 1, all pairs unique.
    records, n_seeds, _ = _run_build_amp(
        tmp_path, monkeypatch, n_submasks=3, n_cleans=4, allocation=6)
    assert n_seeds == 1
    assert len(records) == 6
    pairs = {(r["submask"], r["clean_image"]) for r in records}
    assert len(pairs) == len(records)  # no repeated (submask, clean) pair


def test_amp_full_coverage_at_budget(tmp_path, monkeypatch):
    # allocation == budget (12) -> every (submask, clean) combination used once.
    records, n_seeds, _ = _run_build_amp(
        tmp_path, monkeypatch, n_submasks=3, n_cleans=4, allocation=12)
    assert n_seeds == 1
    assert len(records) == 12
    submasks = {r["submask"] for r in records}
    cleans = {r["clean_image"] for r in records}
    pairs = {(r["submask"], r["clean_image"]) for r in records}
    assert len(submasks) == 3 and len(cleans) == 4
    assert pairs == {(s, c) for s in submasks for c in cleans}


def test_amp_n_seeds_scales_above_budget(tmp_path, monkeypatch):
    # budget = 2 * 2 = 4; allocation 10 -> n_seeds = ceil(10/4) = 3,
    # records = ceil(10 / 3) = 4.
    records, n_seeds, _ = _run_build_amp(
        tmp_path, monkeypatch, n_submasks=2, n_cleans=2, allocation=10)
    assert n_seeds == 3
    assert len(records) == 4


def test_amp_deterministic_for_fixed_seed(tmp_path, monkeypatch):
    r1, _, _ = _run_build_amp(tmp_path / "a", monkeypatch, n_submasks=3, n_cleans=4, allocation=8)
    r2, _, _ = _run_build_amp(tmp_path / "b", monkeypatch, n_submasks=3, n_cleans=4, allocation=8)
    assert [(Path(r["submask"]).name, Path(r["clean_image"]).name) for r in r1] == \
           [(Path(r["submask"]).name, Path(r["clean_image"]).name) for r in r2]


def test_amp_name_field_format(tmp_path, monkeypatch):
    records, _, _ = _run_build_amp(
        tmp_path, monkeypatch, n_submasks=2, n_cleans=2, allocation=3)
    for r in records:
        assert r["name"] == f"{Path(r['clean_image']).stem}__{Path(r['submask']).stem}"
        assert r["cad_mask"] is None and r["cad_mask_label"] is None  # sd == "free"


# ===========================================================================
# build_jsonl.py -- connected-component auto-detect
# ===========================================================================
def test_all_multi_instance_true_only_when_every_mask_multi(tmp_path):
    texture, anomaly = "metal", "scratch"
    full = f"{texture}+{anomaly}"
    d = tmp_path / "all_multi" / texture / "mask" / anomaly
    _write_png(d / "m0.png", _two_blobs())
    _write_png(d / "m1.png", _two_blobs())
    assert J._all_multi_instance_masks(tmp_path / "all_multi", full) is True

    d2 = tmp_path / "mixed" / texture / "mask" / anomaly
    _write_png(d2 / "m0.png", _two_blobs())
    _write_png(d2 / "m1.png", _single_blob())  # one single-instance mask
    assert J._all_multi_instance_masks(tmp_path / "mixed", full) is False


def test_all_multi_instance_false_when_no_masks(tmp_path):
    assert J._all_multi_instance_masks(tmp_path / "empty", "metal+scratch") is False


# ===========================================================================
# build_jsonl.py -- iter_max wiring and the 0-entries exit(1) guard
# ===========================================================================
def _jsonl_fixture(tmp_path, *, mask_arr, n_amp_masks, allocation,
                   texture="metal", anomaly="scratch", clean_stem="clean0"):
    full = f"{texture}+{anomaly}"
    dataset_dir = tmp_path / "dataset"
    clean_dir = tmp_path / "clean"
    amp_dir = tmp_path / "amp_out"
    # training masks drive the connected-component auto-detect
    for i in range(2):
        _write_png(dataset_dir / texture / "mask" / anomaly / f"train{i}.png", mask_arr)
    # clean image the AMP mask pairs back to
    _write_png(clean_dir / texture / "clean_image" / f"{clean_stem}.png")
    # AMP output layout: <amp_out>/<clean_stem>/<full>/<submask>__seed<i>.png
    for i in range(n_amp_masks):
        _write_png(amp_dir / clean_stem / full / f"sub0__seed{i}.png")
    alloc = tmp_path / "allocation.json"
    alloc.write_text(json.dumps({full: allocation}))
    out = tmp_path / "out.jsonl"
    return dataset_dir, clean_dir, amp_dir, alloc, out, full


def _run_build_jsonl(tmp_path, monkeypatch, *, mask_arr, n_amp_masks, allocation,
                     with_dataset_dir=True):
    dataset_dir, clean_dir, amp_dir, alloc, out, full = _jsonl_fixture(
        tmp_path, mask_arr=mask_arr, n_amp_masks=n_amp_masks, allocation=allocation)
    argv = [
        "--amp-output-dir", amp_dir,
        "--clean-dir", clean_dir,
        "--allocation", alloc,
        "--defect-types", full,
        "--output", out,
    ]
    if with_dataset_dir:
        argv += ["--dataset-dir", dataset_dir]
    _run_main(J, argv, monkeypatch)
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    return lines


def test_jsonl_iter_max_1_when_all_multi_instance(tmp_path, monkeypatch):
    lines = _run_build_jsonl(
        tmp_path, monkeypatch, mask_arr=_two_blobs(), n_amp_masks=2, allocation=2)
    assert len(lines) == 2
    assert all(e["iteration_generation_max_instance"] == 1 for e in lines)


def test_jsonl_iter_max_5_when_single_instance_present(tmp_path, monkeypatch):
    lines = _run_build_jsonl(
        tmp_path, monkeypatch, mask_arr=_single_blob(), n_amp_masks=2, allocation=2)
    assert all(e["iteration_generation_max_instance"] == 5 for e in lines)


def test_jsonl_iter_max_defaults_to_5_without_dataset_dir(tmp_path, monkeypatch):
    lines = _run_build_jsonl(
        tmp_path, monkeypatch, mask_arr=_two_blobs(), n_amp_masks=2, allocation=2,
        with_dataset_dir=False)
    assert all(e["iteration_generation_max_instance"] == 5 for e in lines)


def test_jsonl_exits_1_when_no_entries_written(tmp_path, monkeypatch):
    # allocation asks for masks but none exist on disk -> empty JSONL -> exit(1).
    with pytest.raises(SystemExit) as ei:
        _run_build_jsonl(
            tmp_path, monkeypatch, mask_arr=_two_blobs(), n_amp_masks=0, allocation=2)
    assert ei.value.code == 1
    assert (tmp_path / "out.jsonl").read_text().strip() == ""
