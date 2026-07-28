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
"""Regression tests for scripts/download_checkpoints.py.

Locks in the download-default behavior change: t5-11b (T5-XXL, ~45 GB) is
fetched only when ``--with_t5_11b`` is passed, and ``--model_sizes`` is
restricted to {2B, 14B}.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import download_checkpoints as D  # noqa: E402


def _parse(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["download_checkpoints.py", *argv])
    return D.parse_args()


def test_with_t5_11b_defaults_off(monkeypatch):
    assert _parse(monkeypatch, []).with_t5_11b is False


def test_with_t5_11b_opt_in(monkeypatch):
    assert _parse(monkeypatch, ["--with_t5_11b"]).with_t5_11b is True


def test_model_sizes_rejects_unknown(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--model_sizes", "7B"])


def _mock_network(monkeypatch):
    """Replace every network/download call in main() so it runs offline, and
    record which repos download_model() was asked to fetch."""
    requested = []
    monkeypatch.setattr(D, "download_model",
                        lambda ckpt, repo_id, **kw: requested.append(repo_id))
    monkeypatch.setattr(D, "hf_hub_download", lambda *a, **kw: None)
    monkeypatch.setattr(D, "snapshot_download", lambda *a, **kw: None)
    monkeypatch.setattr(D.subprocess, "run", lambda *a, **kw: None)
    return requested


def test_main_skips_t5_11b_by_default(tmp_path, monkeypatch):
    requested = _mock_network(monkeypatch)
    args = _parse(monkeypatch, ["--model_types", "text2image",
                                "--model_sizes", "2B",
                                "--checkpoint_dir", str(tmp_path)])
    D.main(args)
    assert "google-t5/t5-large" in requested
    assert "google-t5/t5-11b" not in requested


def test_main_downloads_t5_11b_when_opted_in(tmp_path, monkeypatch):
    requested = _mock_network(monkeypatch)
    args = _parse(monkeypatch, ["--model_types", "text2image",
                                "--model_sizes", "2B", "--with_t5_11b",
                                "--checkpoint_dir", str(tmp_path)])
    D.main(args)
    assert "google-t5/t5-11b" in requested
