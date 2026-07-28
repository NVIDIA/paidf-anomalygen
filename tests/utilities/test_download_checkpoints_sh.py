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
"""Tests for scripts/utilities/download_checkpoints.sh.

Covers the 2B-only default, the --model-sizes / --with-t5-11b plumbing into
the module, the presence-gate skip, and the argument guards. The heavy
download never runs — a stub ``python``/``hf`` on PATH captures the module
invocation and lets the wrapper's skip/need branches be exercised offline.
"""
import os

from tests.utilities.util import make_checkpoints, run_script


def _stub_path(tmp_path):
    """A PATH dir with `hf` and `python` stubs. `python` appends its argv to
    ``python_args.txt`` so a test can assert what the wrapper forwarded."""
    binp = tmp_path / "bin"
    binp.mkdir()
    argfile = tmp_path / "python_args.txt"
    (binp / "hf").write_text("#!/usr/bin/env bash\nexit 0\n")
    (binp / "python").write_text(
        f'#!/usr/bin/env bash\nprintf "%s " "$@" >> "{argfile}"\nexit 0\n')
    for name in ("hf", "python"):
        (binp / name).chmod(0o755)
    return binp, argfile


def _env(binp):
    return {"HF_TOKEN": "dummy", "PATH": f"{binp}:{os.environ['PATH']}"}


def test_help_lists_new_flags(tmp_path):
    r = run_script("download_checkpoints.sh", "--help")
    assert r.returncode == 0
    assert "--model-sizes" in r.stdout
    assert "--with-t5-11b" in r.stdout


def test_rejects_unknown_model_size(tmp_path):
    r = run_script("download_checkpoints.sh", "--model-sizes", "7B")
    assert r.returncode == 2
    assert "must be from {2B, 14B}" in r.stderr


def test_rejects_empty_model_sizes(tmp_path):
    r = run_script("download_checkpoints.sh", "--model-sizes", "")
    assert r.returncode == 2
    assert "cannot be empty" in r.stderr


def test_skips_when_all_present(tmp_path):
    root = make_checkpoints(tmp_path / "ckpts")  # full default (2B) set
    binp, argfile = _stub_path(tmp_path)
    r = run_script("download_checkpoints.sh", "--checkpoint-dir", root, env=_env(binp))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[skip]" in r.stdout
    assert not argfile.exists()  # module (python) never invoked on the skip path


def test_missing_artifact_invokes_module_default_flags(tmp_path):
    root = make_checkpoints(tmp_path / "ckpts")
    (root / "NVDINOV2/nv_dinov2_classification_model.ckpt").unlink()  # force a download
    binp, argfile = _stub_path(tmp_path)
    r = run_script("download_checkpoints.sh", "--checkpoint-dir", root, env=_env(binp))
    assert r.returncode == 0, r.stdout + r.stderr
    args = argfile.read_text()
    assert "-m scripts.download_checkpoints" in args
    assert "--model_sizes 2B " in args        # 2B-only default
    assert "--with_t5_11b" not in args        # t5-11b off by default


def test_with_t5_11b_flag_forwarded(tmp_path):
    root = make_checkpoints(tmp_path / "ckpts")
    (root / "NVDINOV2/nv_dinov2_classification_model.ckpt").unlink()
    binp, argfile = _stub_path(tmp_path)
    r = run_script("download_checkpoints.sh", "--checkpoint-dir", root,
                   "--with-t5-11b", env=_env(binp))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--with_t5_11b" in argfile.read_text()


def test_both_sizes_forwarded(tmp_path):
    root = make_checkpoints(tmp_path / "ckpts")
    (root / "NVDINOV2/nv_dinov2_classification_model.ckpt").unlink()
    binp, argfile = _stub_path(tmp_path)
    r = run_script("download_checkpoints.sh", "--checkpoint-dir", root,
                   "--model-sizes", "2B 14B", env=_env(binp))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--model_sizes 2B 14B " in argfile.read_text()
