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
"""Tests for scripts/utilities/check.sh checkpoint verification.

Covers the 2B-only default, the --model-sizes selector and its guards, the
either-T5 rule, and the Cosmos-Guardrail1 checks.
"""
from tests.utilities.util import make_checkpoints, run_script


def test_default_2b_all_present_passes(tmp_path):
    # Only the 2B base is present (no 14B) — the default must still pass.
    root = make_checkpoints(tmp_path / "ckpts", sizes=("2B",))
    r = run_script("check.sh", "--checkpoint-dir", root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "all required artifacts present" in r.stdout


def test_both_sizes_requires_14b(tmp_path):
    root = make_checkpoints(tmp_path / "ckpts", sizes=("2B",))  # 14B absent
    r = run_script("check.sh", "--checkpoint-dir", root, "--model-sizes", "2B 14B")
    assert r.returncode == 1
    assert "Cosmos-Predict2-14B-Text2Image/model.pt" in r.stdout


def test_missing_guardrail_fails(tmp_path):
    root = make_checkpoints(tmp_path / "ckpts")
    (root / "nvidia/Cosmos-Guardrail1/video_content_safety_filter/safety_filter.pt").unlink()
    r = run_script("check.sh", "--checkpoint-dir", root)
    assert r.returncode == 1
    assert "Cosmos-Guardrail1" in r.stdout


def test_either_t5_variant_suffices(tmp_path):
    # t5-11b present, t5-large absent — training accepts either.
    root = make_checkpoints(tmp_path / "ckpts", with_t5_large=False, with_t5_11b=True)
    r = run_script("check.sh", "--checkpoint-dir", root)
    assert r.returncode == 0, r.stdout


def test_rejects_unknown_model_size(tmp_path):
    root = make_checkpoints(tmp_path / "ckpts")
    r = run_script("check.sh", "--checkpoint-dir", root, "--model-sizes", "7B")
    assert r.returncode == 2
    assert "must be from {2B, 14B}" in r.stderr


def test_rejects_empty_model_sizes(tmp_path):
    root = make_checkpoints(tmp_path / "ckpts")
    r = run_script("check.sh", "--checkpoint-dir", root, "--model-sizes", "")
    assert r.returncode == 2
    assert "cannot be empty" in r.stderr
