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
"""Unit tests for scripts/utilities/reamp_swap_masks.py (QC 2026-07-06 #9).

Regression: --seed-index used to build `seed{N}.png`, but AMP masks are
named `<submask_stem>__seed{N}.png` (see run_auto_roi_amp.py), so the
documented override never found a file and always exited 1.
"""
import json

from tests.utilities.util import run_script

STEM = "part_0001_mask_largest"


def _setup(tmp_path):
    old_amp = tmp_path / "old_amp"
    new_amp = tmp_path / "new_amp"
    old_mask = old_amp / "clean_A" / "Tex+scratch" / f"{STEM}__seed1.png"
    old_mask.parent.mkdir(parents=True)
    old_mask.touch()
    new_dir = new_amp / "clean_A" / "Tex+scratch"
    new_dir.mkdir(parents=True)
    for s in (0, 1):
        (new_dir / f"{STEM}__seed{s}.png").touch()
    base = tmp_path / "base.jsonl"
    base.write_text(json.dumps({"mask_filename": str(old_mask)}) + "\n")
    return base, new_amp


def test_preserves_seed_index_by_default(tmp_path):
    base, new_amp = _setup(tmp_path)
    out = tmp_path / "out.jsonl"
    r = run_script("reamp_swap_masks.py", "--base-jsonl", base,
                   "--new-amp-dir", new_amp, "--output", out)
    assert r.returncode == 0, r.stderr
    row = json.loads(out.read_text())
    assert row["mask_filename"] == str(
        new_amp / "clean_A" / "Tex+scratch" / f"{STEM}__seed1.png")


def test_seed_index_override_finds_amp_mask(tmp_path):
    base, new_amp = _setup(tmp_path)
    out = tmp_path / "out.jsonl"
    r = run_script("reamp_swap_masks.py", "--base-jsonl", base,
                   "--new-amp-dir", new_amp, "--output", out,
                   "--seed-index", 0)
    assert r.returncode == 0, r.stderr
    row = json.loads(out.read_text())
    assert row["mask_filename"] == str(
        new_amp / "clean_A" / "Tex+scratch" / f"{STEM}__seed0.png")


def test_seed_index_strips_only_trailing_seed_token(tmp_path):
    # A submask stem may itself contain "__seed"; only the trailing
    # __seed<N> may be replaced.
    stem = "region__seed_map_mask_largest"
    old_mask = tmp_path / "old_amp" / "clean_A" / "Tex+scratch" / f"{stem}__seed3.png"
    old_mask.parent.mkdir(parents=True)
    old_mask.touch()
    new_dir = tmp_path / "new_amp" / "clean_A" / "Tex+scratch"
    new_dir.mkdir(parents=True)
    (new_dir / f"{stem}__seed0.png").touch()
    base = tmp_path / "base.jsonl"
    base.write_text(json.dumps({"mask_filename": str(old_mask)}) + "\n")

    out = tmp_path / "out.jsonl"
    r = run_script("reamp_swap_masks.py", "--base-jsonl", base,
                   "--new-amp-dir", tmp_path / "new_amp", "--output", out,
                   "--seed-index", 0)
    assert r.returncode == 0, r.stderr
    row = json.loads(out.read_text())
    assert row["mask_filename"] == str(new_dir / f"{stem}__seed0.png")
