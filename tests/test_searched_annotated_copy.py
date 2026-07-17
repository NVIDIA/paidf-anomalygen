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

"""Regression tests for annotated_image propagation into the searched/filtered
buckets.

SDG writes annotated_image per anomaly instance as "<stem>_<j>.png", while
reconstructed_image is a single "<stem>.png". The assemble/filter copy paths
used to look up annotated images by the reconstructed basename (exact match),
so annotated overlays were silently never copied. These tests guard that the
per-instance files are now globbed and copied.
"""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "utilities"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _make_sdg_sample(src, stem, n_instances):
    """Lay out one SDG sample: single-file kinds + per-instance annotated."""
    _touch(src / "reconstructed_image" / f"{stem}.png")
    _touch(src / "original_mask" / f"{stem}.png")
    _touch(src / "original_image" / f"{stem}.png")
    for j in range(n_instances):
        _touch(src / "annotated_image" / f"{stem}_{j}.png")


def test_assemble_searched_copies_all_annotated_instances(tmp_path):
    mod = _load("assemble_searched")
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_sdg_sample(src, "oil_00042", n_instances=2)

    mod._copy_sample_outputs(src, dst, idx=5, basename="oil_00042.png")

    # Single-file kind copied 1:1 with the idx-prefixed name.
    assert (dst / "reconstructed_image" / "idx000005_oil_00042.png").exists()
    # BOTH per-instance annotated files copied, suffix preserved (regression).
    assert (dst / "annotated_image" / "idx000005_oil_00042_0.png").exists()
    assert (dst / "annotated_image" / "idx000005_oil_00042_1.png").exists()
    assert len(list((dst / "annotated_image").glob("*.png"))) == 2


def test_filter_with_regen_copies_all_annotated_instances(tmp_path):
    mod = _load("filter_with_regen")
    src = tmp_path / "src"
    staging = tmp_path / "staging"
    for k in mod.KINDS:            # main() pre-creates the staging subdirs
        (staging / k).mkdir(parents=True)
    _make_sdg_sample(src, "oil_00042", n_instances=3)

    mod._copy_sample_kinds(src, "oil_00042.png", staging, new_stem="oil_00003")

    # Single-file kind renamed to the new sequential basename.
    assert (staging / "reconstructed_image" / "oil_00003.png").exists()
    # All annotated instances renamed to <new_stem>_<j>.png (regression).
    assert (staging / "annotated_image" / "oil_00003_0.png").exists()
    assert (staging / "annotated_image" / "oil_00003_1.png").exists()
    assert (staging / "annotated_image" / "oil_00003_2.png").exists()
    assert len(list((staging / "annotated_image").glob("*.png"))) == 3
