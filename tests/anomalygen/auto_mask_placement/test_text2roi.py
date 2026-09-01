# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for text2roi grounded detection (``Text2BoxDetector``).

Downloads one COCO sample image (a bear, which fills most of the frame) on-the-fly to a pytest temp dir,
runs detection with the prompt "the bear" using the default Text2BoxDetector model (Cosmos3-Nano), and
checks the predicted box (IoU) + center point against a reference. Model-dependent: skips cleanly if the
image can't be downloaded or the VLM can't be loaded (no network / weights / GPU), so it doesn't break
environments without the model.
"""

import math
import urllib.request

import numpy as np
import pytest

pytest.importorskip("torch")

COCO_URL = "http://images.cocodataset.org/val2017/000000000285.jpg"  # COCO val image_id 285: a bear
PROMPT = "the bear"

# Reference box/point in 0-1000 normalized coords (the bear fills most of the frame); converted to
# pixels against the actual image size at test time. Both Cosmos3-Nano and Qwen3-VL-8B land here.
EXPECTED_BBOX_NORM = (0, 110, 1000, 1000)
EXPECTED_POINT_NORM = (500, 550)
IOU_MIN = 0.70
POINT_DIST_MAX = 0.15  # as a fraction of the image diagonal


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@pytest.fixture(scope="module")
def sample_image(tmp_path_factory):
    from PIL import Image

    dest = tmp_path_factory.mktemp("text2roi") / "coco_000000000285.jpg"
    try:
        urllib.request.urlretrieve(COCO_URL, dest)
    except Exception as e:  # noqa: BLE001 - network is optional in CI
        pytest.skip(f"could not download COCO sample image: {e}")
    return Image.open(dest).convert("RGB")


@pytest.mark.gpu
def test_text2roi_detects_bear(sample_image):
    from anomalygen.auto_mask_placement.text2roi import Text2BoxDetector

    detector = Text2BoxDetector(device="auto")  # default model_id = nvidia/Cosmos3-Nano

    try:
        detector.load()
    except Exception as e:  # noqa: BLE001 - weights/GPU are optional in CI
        pytest.skip(f"text2roi VLM unavailable: {e}")

    try:
        detections = detector.detect(sample_image, PROMPT)
    finally:
        detector.unload()

    assert detections, "expected at least one detection for the bear"
    top = max(detections, key=lambda d: d.get("confidence", 0.0))

    w, h = sample_image.size
    exp_box = [
        EXPECTED_BBOX_NORM[0] / 1000 * w,
        EXPECTED_BBOX_NORM[1] / 1000 * h,
        EXPECTED_BBOX_NORM[2] / 1000 * w,
        EXPECTED_BBOX_NORM[3] / 1000 * h,
    ]
    iou = _iou(top["bbox"], exp_box)
    assert iou >= IOU_MIN, f"bbox IoU {iou:.3f} < {IOU_MIN} (pred={top['bbox']}, expected≈{exp_box})"

    point = top.get("point")
    assert point is not None, "expected a center point in the detection"
    exp_pt = (EXPECTED_POINT_NORM[0] / 1000 * w, EXPECTED_POINT_NORM[1] / 1000 * h)
    dist = math.hypot(point[0] - exp_pt[0], point[1] - exp_pt[1]) / math.hypot(w, h)
    assert dist <= POINT_DIST_MAX, f"point distance {dist:.3f} > {POINT_DIST_MAX} (pred={point}, expected≈{exp_pt})"


def _blank_mask(h, w):
    return np.zeros((h, w), np.uint8)


def test_corner_object_with_hole_is_not_all_white():
    from anomalygen.auto_mask_placement.text2roi.sam_utils import _keep_largest_and_fill

    m = _blank_mask(20, 20)
    m[0:15, 0:15] = 255
    m[5:10, 5:10] = 0

    out = _keep_largest_and_fill(m)

    assert out[7, 7] == 255
    # The exterior background far from the object must remain background. This
    # is the regression assertion: the old code set it to 255 (all-white).
    assert out[17, 17] == 0
    assert not np.all(out == 255)


def test_corner_object_without_hole_is_preserved():
    from anomalygen.auto_mask_placement.text2roi.sam_utils import _keep_largest_and_fill

    m = _blank_mask(20, 20)
    m[0:10, 0:10] = 255

    out = _keep_largest_and_fill(m)

    assert out[3, 3] == 255
    assert out[15, 15] == 0
    assert not np.all(out == 255)


def test_center_object_with_hole_still_filled():
    from anomalygen.auto_mask_placement.text2roi.sam_utils import _keep_largest_and_fill

    m = _blank_mask(20, 20)
    m[5:15, 5:15] = 255
    m[8:11, 8:11] = 0

    out = _keep_largest_and_fill(m)

    assert out[9, 9] == 255
    assert out[0, 0] == 0
    assert out[18, 18] == 0


def test_multiple_components_keeps_largest():
    from anomalygen.auto_mask_placement.text2roi.sam_utils import _keep_largest_and_fill

    m = _blank_mask(20, 20)
    m[2:12, 2:12] = 255
    m[15:18, 15:18] = 255

    out = _keep_largest_and_fill(m)

    assert out[6, 6] == 255
    assert out[16, 16] == 0
    assert out[0, 0] == 0
    assert not np.all(out == 255)
