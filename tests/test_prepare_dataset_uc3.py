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

"""Regression tests for prepare_dataset_uc3.

Guards two behaviors:
  * collect_entries scans ALL COCO splits, not just json_entries[0] (curated
    stems in another split were silently dropped).
  * A missing curated stem is reported (and a real extract refuses to run) but
    --dry-run still completes so the user can inspect what is missing.
"""

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (123, 45, 67)).save(buf, format="PNG")
    return buf.getvalue()

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "utilities" / "prepare_dataset_uc3.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_dataset_uc3", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _roboflow_name(stem):
    """Build a Roboflow-style export filename that original_name() recovers."""
    return f"{stem}_jpg.rf.deadbeef01.jpg"


def _write_split(zf, split, stems):
    """Write one split dir: images + a matching _annotations.coco.json."""
    images = []
    for i, stem in enumerate(stems):
        fname = _roboflow_name(stem)
        zf.writestr(f"{split}/{fname}", _png_bytes())
        images.append({"id": i, "file_name": fname})
    zf.writestr(
        f"{split}/_annotations.coco.json",
        json.dumps({"images": images, "annotations": [], "categories": []}),
    )


def _make_zip(tmp_path, splits):
    zip_path = tmp_path / "roboflow.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for split, stems in splits.items():
            _write_split(zf, split, stems)
    return zip_path


def test_stems_are_collected_across_all_splits(tmp_path, monkeypatch):
    """A curated stem living only in a non-first split must still be found."""
    mod = _load_module()
    monkeypatch.setattr(
        mod, "KEEP_ANOMALY", {"oil": {"Oil_0001"}, "scratch": {"Scr_0002"}}
    )
    # Oil_0001 only in 'train', Scr_0002 only in 'valid'. The old code scanned
    # json_entries[0] and would miss one of them.
    zip_path = _make_zip(
        tmp_path,
        {"train": ["Oil_0001"], "valid": ["Scr_0002"]},
    )

    entries, missing = mod.collect_entries(zip_path)

    found = {(dtype, stem) for _, dtype, stem in entries}
    assert ("oil", "Oil_0001") in found
    assert ("scratch", "Scr_0002") in found
    assert missing == []


def test_missing_curated_stem_is_reported(tmp_path, monkeypatch):
    """A curated stem absent from every split is returned in `missing`, while
    the stems that ARE present are still collected (no silent short subset)."""
    mod = _load_module()
    monkeypatch.setattr(mod, "KEEP_ANOMALY", {"oil": {"Oil_0001", "Oil_9999"}})
    zip_path = _make_zip(tmp_path, {"train": ["Oil_0001"]})

    entries, missing = mod.collect_entries(zip_path)

    assert ("oil", "Oil_9999") in missing
    assert ("oil", "Oil_0001") in {(d, s) for _, d, s in entries}


def test_duplicate_stem_across_splits_kept_once(tmp_path, monkeypatch):
    """A stem present in more than one split is collected exactly once."""
    mod = _load_module()
    monkeypatch.setattr(mod, "KEEP_ANOMALY", {"oil": {"Oil_0001"}})
    zip_path = _make_zip(
        tmp_path,
        {"train": ["Oil_0001"], "valid": ["Oil_0001"]},
    )

    entries, missing = mod.collect_entries(zip_path)

    oil = [e for e in entries if e[1] == "oil" and e[2] == "Oil_0001"]
    assert len(oil) == 1
    assert missing == []


def test_dry_run_does_not_abort_on_missing_stem(tmp_path, monkeypatch):
    """--dry-run must complete even when curated stems are missing, so the user
    can inspect the shortfall instead of getting a hard abort."""
    mod = _load_module()
    monkeypatch.setattr(mod, "KEEP_ANOMALY", {"oil": {"Oil_0001", "Oil_9999"}})
    zip_path = _make_zip(tmp_path, {"train": ["Oil_0001"]})
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_dataset_uc3", str(out_dir), "--zip", str(zip_path), "--dry-run"],
    )

    # Returns normally (no SystemExit) despite the missing stem.
    mod.main()


def test_real_extract_continues_on_missing_stem(tmp_path, monkeypatch):
    """A real extract (no --dry-run) warns about the missing stem but still
    extracts the stems that ARE present (usable partial dataset, not nothing)."""
    mod = _load_module()
    monkeypatch.setattr(mod, "KEEP_ANOMALY", {"oil": {"Oil_0001", "Oil_9999"}})
    zip_path = _make_zip(tmp_path, {"train": ["Oil_0001"]})
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_dataset_uc3", str(out_dir), "--zip", str(zip_path)],
    )

    mod.main()  # no SystemExit

    # The present stem was extracted despite Oil_9999 being missing.
    assert (out_dir / "Phone" / "anomaly_image" / "oil" / "Oil_0001.png").exists()


def test_missing_stem_warning_is_emitted(tmp_path, monkeypatch, capsys):
    """The shortfall is still signalled (loud WARNING), so it's not silent."""
    mod = _load_module()
    monkeypatch.setattr(mod, "KEEP_ANOMALY", {"oil": {"Oil_0001", "Oil_9999"}})
    zip_path = _make_zip(tmp_path, {"train": ["Oil_0001"]})

    mod.collect_entries(zip_path)

    out = capsys.readouterr().out
    assert "WARNING" in out and "Oil_9999" in out
