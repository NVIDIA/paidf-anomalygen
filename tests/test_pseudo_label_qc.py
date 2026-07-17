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
"""Regression tests for the QC-sweep pseudo_label fixes.

Covers four of the five findings; each test fails on the pre-fix code:

  * caption.py            -- missing <answer> tags must fall back to the raw
    response, not a garbage slice.
  * utils.py              -- visualize must not crash / reuse stale coords when
    an instance has bbox=None but a class name.
  * infosam/dataset.py    -- InfoSAM2Dataset must reject length mismatches and
    substring-only stem matches ("1" vs "10_mask").
  * iou_metric.py         -- intersect_and_union must raise a clear error on
    out-of-range labels instead of a cryptic bincount broadcast crash.

The fifth finding (infosam/build_infosam2.py checkpoint normalization) is not
unit-tested here: build_infosam2 constructs a full SAM2 model before the
weight-load, so it has no cheap harness. It is verified by inspection (the fix
is a reorder that normalizes the pretrained dict to its inner "model" once
before merging).

Usage:
    pytest tests/test_pseudo_label_qc.py
"""
import importlib
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo_label import iou_metric  # noqa: E402  (numpy-only, safe)
from pseudo_label import utils as pl_utils  # noqa: E402  (numpy + PIL, safe)
from pseudo_label.infosam import dataset as infosam_dataset  # noqa: E402


# ===========================================================================
# caption.py — Captioner.postprocess_response
# ===========================================================================
@pytest.fixture(scope="module")
def Captioner():
    """Import pseudo_label.caption with its heavy deps (vllm/transformers/…)
    stubbed. postprocess_response uses none of them, and patch.dict restores
    sys.modules on teardown so no other test is affected."""
    def _stub(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    stubs = {
        "vllm": _stub("vllm", LLM=object, RequestOutput=object, SamplingParams=object),
        "qwen_vl_utils": _stub("qwen_vl_utils", process_vision_info=lambda *a, **k: None),
        "transformers": _stub("transformers", AutoProcessor=object),
        "transformers.models": _stub("transformers.models"),
        "transformers.models.qwen2_5_vl": _stub(
            "transformers.models.qwen2_5_vl", Qwen2_5_VLProcessor=object
        ),
    }
    with mock.patch.dict(sys.modules, stubs):
        cap = importlib.import_module("pseudo_label.caption")
        yield cap.Captioner


def _pp(Captioner, response, num_bboxes):
    # postprocess_response doesn't use self; call it unbound.
    return Captioner.postprocess_response(None, response, num_bboxes)


def test_caption_missing_both_tags_returns_raw(Captioner):
    # Must contain the "\n\n**Anomaly " delimiter, else even the buggy code
    # early-returns raw; with the delimiter the pre-fix code returns the
    # corrupted response[7:-1] slice re-wrapped in tags.
    raw = "A clean caption sentence.\n\n**Anomaly 1**: a small scratch"
    assert _pp(Captioner, raw, 1) == raw


def test_caption_missing_end_tag_returns_raw(Captioner):
    raw = "<answer>A clean caption.\n\n**Anomaly 1**: a small scratch"  # no </answer>
    assert _pp(Captioner, raw, 1) == raw


def test_caption_end_before_start_returns_raw(Captioner):
    raw = "</answer> stuff <answer>"
    assert _pp(Captioner, raw, 1) == raw


def test_caption_wellformed_extracts_and_truncates(Captioner):
    raw = "pre <answer>CAP\n\n**Anomaly 1**: foo\n\n**Anomaly 2**: bar</answer> post"
    out = _pp(Captioner, raw, 1)
    assert out != raw
    assert "CAP" in out
    assert "**Anomaly 1**: foo" in out
    assert "bar" not in out  # second anomaly dropped (num_bboxes=1)


# ===========================================================================
# utils.py — visualize
# ===========================================================================
def _img(size=(32, 32)):
    return Image.new("RGB", size, (10, 20, 30))


def _mask(size=(32, 32)):
    a = np.zeros(size[::-1], np.uint8)
    a[4:12, 4:12] = 255
    return Image.fromarray(a, mode="L")


def test_visualize_bbox_none_first_instance_no_crash():
    # bbox None on the FIRST instance with a class name: pre-fix -> NameError.
    out = pl_utils.visualize(_img(), classes=["defect"], bboxes=[None],
                             instance_masks=[_mask()])
    assert isinstance(out, Image.Image)


def test_visualize_mixed_bbox_then_none_completes():
    # Second instance bbox None: pre-fix silently reused the first bbox coords.
    out = pl_utils.visualize(
        _img(), classes=["a", "b"], bboxes=[(2, 2, 10, 10), None],
        instance_masks=[_mask(), _mask()],
    )
    assert isinstance(out, Image.Image)


def test_visualize_bbox_present_still_draws():
    out = pl_utils.visualize(_img(), classes=["a"], bboxes=[(1, 1, 8, 8)],
                             instance_masks=[_mask()])
    assert isinstance(out, Image.Image)


# ===========================================================================
# infosam/dataset.py — InfoSAM2Dataset.__init__ validation
# ===========================================================================
_SENTINEL_TF = object()  # pass as image_transforms to skip SAM2Transforms build


def _ds(image_stems, mask_stems):
    imgs = [Path("images") / f"{s}.png" for s in image_stems]
    masks = [Path("masks") / f"{s}.png" for s in mask_stems]
    return infosam_dataset.InfoSAM2Dataset(imgs, masks, image_transforms=_SENTINEL_TF)


def test_dataset_equal_stems_ok():
    ds = _ds(["image_0", "image_1"], ["image_0", "image_1"])
    assert len(ds) == 2


def test_dataset_underscore_suffix_mask_ok():
    ds = _ds(["image_0", "image_1"], ["image_0_mask", "image_1_mask"])
    assert len(ds) == 2


def test_dataset_length_mismatch_raises():
    with pytest.raises(ValueError, match="differ"):
        _ds(["image_0", "image_1"], ["image_0"])


def test_dataset_substring_false_positive_rejected():
    # "1" is a substring of "10_mask" -> pre-fix wrongly paired; now rejected.
    with pytest.raises(ValueError, match="do not match"):
        _ds(["1"], ["10_mask"])


# ===========================================================================
# iou_metric.py — MeanIoUMeter.intersect_and_union
# ===========================================================================
def test_iou_valid_binary_counts():
    pred = np.array([0, 1, 1, 0])
    label = np.array([0, 1, 0, 0])
    ai, au, apl, al = iou_metric.MeanIoUMeter.intersect_and_union(
        pred, label, num_classes=2, ignore_index=255)
    assert list(ai) == [2, 1]   # intersect over classes 0,1
    assert list(apl) == [2, 2]  # pred histogram
    assert list(al) == [3, 1]   # label histogram
    assert list(au) == [3, 2]   # union = apl + al - ai


def test_iou_ignore_index_excluded():
    pred = np.array([0, 1])
    label = np.array([0, 255])  # second pixel ignored
    ai, au, apl, al = iou_metric.MeanIoUMeter.intersect_and_union(
        pred, label, num_classes=2, ignore_index=255)
    assert list(al) == [1, 0]   # only the un-ignored pixel counted


def test_iou_label_out_of_range_raises():
    pred = np.array([0, 1])
    label = np.array([0, 5])  # 5 >= num_classes=2, not the ignore_index
    with pytest.raises(ValueError, match=r"label contains values outside \[0, 2\)"):
        iou_metric.MeanIoUMeter.intersect_and_union(
            pred, label, num_classes=2, ignore_index=255)


def test_iou_pred_out_of_range_raises():
    pred = np.array([0, 3])  # 3 >= num_classes=2
    label = np.array([0, 1])
    with pytest.raises(ValueError, match=r"pred_label contains values outside"):
        iou_metric.MeanIoUMeter.intersect_and_union(
            pred, label, num_classes=2, ignore_index=255)
