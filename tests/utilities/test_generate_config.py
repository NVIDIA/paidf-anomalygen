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
"""Unit tests for scripts/utilities/generate_config.py (QC 2026-07-06 #12).

Regression: a placeholder present in the template but absent from the
substitution map used to be written to the output YAML literally, surfacing
much later as a cryptic parse failure or wrong training behavior.
"""
import re
import sys

import pytest
import yaml

from scripts.utilities import generate_config
from tests.utilities.util import run_script


def _spec(tmp_path):
    spec = tmp_path / "defect_spec.jsonl"
    spec.write_text('{"defect_type": "Tex+scratch"}\n')
    return spec


def test_real_template_renders_with_no_leftover_placeholders(tmp_path):
    out = tmp_path / "cfg.yaml"
    r = run_script("generate_config.py",
                   "--name", "unit",
                   "--dataset-dir", tmp_path,
                   "--defect-spec", _spec(tmp_path),
                   "--validation-jsonl", "val.jsonl",
                   "--output", out)
    assert r.returncode == 0, r.stderr
    rendered = out.read_text()
    assert not re.findall(r"<[A-Z][A-Z_]*>", rendered)
    assert yaml.safe_load(rendered)


def test_user_value_containing_caps_token_is_not_flagged(tmp_path):
    # The sync check diffs template tokens against subs keys; a user value
    # that happens to contain <CAPS> must not abort the run.
    out = tmp_path / "cfg.yaml"
    r = run_script("generate_config.py",
                   "--name", "unit<VER>",
                   "--dataset-dir", tmp_path,
                   "--defect-spec", _spec(tmp_path),
                   "--validation-jsonl", "val.jsonl",
                   "--output", out)
    assert r.returncode == 0, r.stderr
    assert "<VER>" in out.read_text()


def test_unknown_placeholder_fails(tmp_path, monkeypatch, capsys):
    template = tmp_path / "template.yaml"
    template.write_text("name: <NAME>\nnew_knob: <NEW_KNOB>\n")
    monkeypatch.setattr(generate_config, "TEMPLATE_PATH", template)
    out = tmp_path / "cfg.yaml"
    monkeypatch.setattr(sys, "argv", [
        "generate_config.py",
        "--name", "unit",
        "--dataset-dir", str(tmp_path),
        "--defect-spec", str(_spec(tmp_path)),
        "--validation-jsonl", "val.jsonl",
        "--output", str(out),
    ])
    with pytest.raises(SystemExit) as exc:
        generate_config.main()
    assert exc.value.code == 1
    assert "<NEW_KNOB>" in capsys.readouterr().err
    assert not out.exists()
