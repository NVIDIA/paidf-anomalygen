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

"""Unit tests for the pure mask helpers in multiview_inference_utils.

Multi-view-specific, model-free (PIL/numpy + cv2) helpers that turn raw per-view
masks into the binary denoise conditions and shared per-instance crop masks the
diffusion pipeline consumes. One test per helper; each pins the behavioral
contract so a real regression (dropped binarization, union<->intersect swap,
transposed [instance][view] nesting, threshold boundary flip) fails loudly.
"""

import numpy as np
import pytest
from PIL import Image

from cosmos_predict2.inference.anomaly_gen.multiview_inference_utils import (
    _intersect_masks,
    _resize_if_needed,
    _resize_mask_to_binary,
    _split_union_mask_for_multiview,
    _union_masks,
)


def _mask(size=16, regions=()):
    """L-mode mask; `regions` is a list of (yslice, xslice) set to 255."""
    arr = np.zeros((size, size), dtype=np.uint8)
    for ys, xs in regions:
        arr[ys, xs] = 255
    return Image.fromarray(arr, mode="L")


def _arr(img):
    return np.array(img.convert("L"))


def test_resize_if_needed():
    img = Image.fromarray(np.zeros((16, 16, 3), np.uint8), mode="RGB")
    assert _resize_if_needed(img, (16, 16)) is img          # no-op / no needless copy
    assert _resize_if_needed(img, (32, 24)).size == (32, 24)  # resizes to target
    with pytest.raises(TypeError):                           # non-PIL rejected
        _resize_if_needed(np.zeros((16, 16, 3), np.uint8), (16, 16))


def test_resize_mask_to_binary():
    # same size, grayscale -> must still come out strictly 0/255 (drops the raw-
    # grayscale leak flagged in review).
    gray = np.full((16, 16), 100, np.uint8)  # < 127 -> 0
    gray[4:12, 4:12] = 200                    # >= 127 -> 255
    out = _arr(_resize_mask_to_binary(Image.fromarray(gray, "L"), (16, 16)))
    assert set(np.unique(out).tolist()) <= {0, 255}
    np.testing.assert_array_equal(out, np.where(gray >= 127, 255, 0).astype(np.uint8))

    # threshold is inclusive: pixel == threshold -> 255 (contract is `>=`, not `>`).
    thr = np.full((16, 16), 126, np.uint8)
    thr[:8, :] = 127
    out = _arr(_resize_mask_to_binary(Image.fromarray(thr, "L"), (16, 16), threshold=127))
    assert out[:8, :].min() == 255 and out[8:, :].max() == 0

    # custom threshold respected.
    cst = np.full((16, 16), 199, np.uint8)
    cst[:8, :] = 200
    out = _arr(_resize_mask_to_binary(Image.fromarray(cst, "L"), (16, 16), threshold=200))
    assert out[:8, :].min() == 255 and out[8:, :].max() == 0

    # resize path reaches the target size AND stays binary.
    small = np.full((8, 8), 100, np.uint8)
    small[2:6, 2:6] = 200
    res = _resize_mask_to_binary(Image.fromarray(small, "L"), (32, 32))
    assert res.size == (32, 32)
    assert set(np.unique(_arr(res)).tolist()) <= {0, 255}

    # already binary + right size -> returned untouched.
    m = _mask(regions=[(slice(4, 12), slice(4, 12))])
    assert _resize_mask_to_binary(m, (16, 16)) is m

    with pytest.raises(TypeError):  # non-PIL rejected
        _resize_mask_to_binary(np.zeros((16, 16), np.uint8), (16, 16))


def test_union_masks():
    # union = logical OR of every view region (guards np.maximum, not minimum).
    a = _mask(regions=[(slice(2, 6), slice(2, 6))])
    b = _mask(regions=[(slice(2, 6), slice(10, 14))])  # disjoint from a
    out = _arr(_union_masks([a, b]))
    assert out[2:6, 2:6].min() == 255      # a present
    assert out[2:6, 10:14].min() == 255    # b present
    assert out[8:, :].max() == 0           # nothing invented elsewhere

    # grayscale inputs binarized before the union.
    g = np.zeros((16, 16), np.uint8)
    g[2:6, 2:6] = 200   # >= 127 -> foreground
    g[8:12, 8:12] = 100  # < 127 -> background
    out = _arr(_union_masks([Image.fromarray(g, "L")]))
    assert out[2:6, 2:6].min() == 255 and out[8:12, 8:12].max() == 0

    with pytest.raises(ValueError):  # empty list
        _union_masks([])
    with pytest.raises(ValueError):  # size mismatch
        _union_masks([_mask(16), _mask(8)])


def test_intersect_masks():
    # intersection = logical AND (guards np.minimum, not maximum).
    a = _mask(regions=[(slice(2, 10), slice(2, 10))])
    b = _mask(regions=[(slice(6, 14), slice(6, 14))])  # overlap = [6:10, 6:10]
    out = _arr(_intersect_masks(a, b))
    assert out[6:10, 6:10].min() == 255    # overlap kept
    assert out[2:6, 2:6].max() == 0        # a-only dropped
    assert out[10:14, 10:14].max() == 0    # b-only dropped

    # disjoint -> empty.
    c = _mask(regions=[(slice(2, 6), slice(2, 6))])
    d = _mask(regions=[(slice(10, 14), slice(10, 14))])
    assert _arr(_intersect_masks(c, d)).max() == 0

    with pytest.raises(ValueError):  # size mismatch
        _intersect_masks(_mask(16), _mask(8))


def test_split_union_mask_for_multiview():
    # Single instance (max_k=1): union kept whole. Since union ⊇ every view mask,
    # intersecting back reproduces each view's own mask; structure = [1][num_views].
    v0 = _mask(regions=[(slice(4, 8), slice(4, 8))])
    v1 = _mask(regions=[(slice(4, 8), slice(6, 12))])
    instances, per_inst_views = _split_union_mask_for_multiview([v0, v1], max_k=1)
    assert len(instances) == 1
    assert len(per_inst_views) == 1 and len(per_inst_views[0]) == 2
    np.testing.assert_array_equal(_arr(per_inst_views[0][0]), _arr(v0))
    np.testing.assert_array_equal(_arr(per_inst_views[0][1]), _arr(v1))

    # Two disjoint blobs across the union -> 2 instances. Result must be nested
    # [num_instances][num_views] (not transposed), and entry [i][v] must equal
    # intersect(instance_i, view_v).
    w0 = _mask(regions=[(slice(2, 6), slice(2, 6))])
    w1 = _mask(regions=[(slice(2, 6), slice(10, 14))])
    instances, per_inst_views = _split_union_mask_for_multiview([w0, w1], max_k=5)
    assert len(instances) == 2
    assert len(per_inst_views) == len(instances)  # outer = instances
    for i, per_view in enumerate(per_inst_views):
        assert len(per_view) == 2  # inner = views
        for v, view_mask in enumerate([w0, w1]):
            expected = _arr(_intersect_masks(instances[i], view_mask))
            np.testing.assert_array_equal(_arr(per_view[v]), expected)


def test_e2e_union_split_intersect_isolates_each_instance_per_view():
    """End-to-end gate for the *purpose* of union -> split -> intersect, chained in
    the real prepare-batch order (resize + binarize -> split_union):

    - split: two physically separate defects must become two separate, shared
      instances (each generated on its own iteration).
    - intersect: each per-view denoise condition must isolate EXACTLY ONE instance
      in that view -- equal to instance∩view, empty where the view can't see that
      instance, disjoint from the other instance, and together covering the whole
      view mask.

    (The shared instance mask drives crop_and_paste consistency -- not tested here;
    mask augmentation between binarize and split is a separate helper.)
    """
    size = 40
    A = (slice(6, 14), slice(6, 14))     # defect A
    B = (slice(6, 14), slice(26, 34))    # defect B, disjoint from A

    # Raw inputs: images at varying resolutions; masks grayscale (non-binary).
    raw_images = [
        Image.fromarray(np.full((h, w, 3), 120, np.uint8), "RGB")
        for (h, w) in [(30, 24), (40, 40), (48, 36)]
    ]

    def _gray(regions):
        a = np.full((size, size), 90, np.uint8)  # 90 -> background after binarize
        for ys, xs in regions:
            a[ys, xs] = 200                       # 200 -> foreground
        return Image.fromarray(a, "L")

    # view 0 sees both defects, view 1 only A, view 2 only B.
    raw_masks = [_gray([A, B]), _gray([A]), _gray([B])]

    # Step 1 -- resize images + binarize/resize masks to the model resolution.
    images = [_resize_if_needed(im, (size, size)) for im in raw_images]
    masks = [_resize_mask_to_binary(m, (size, size)) for m in raw_masks]
    for im, m in zip(images, masks):
        assert im.size == (size, size) == m.size
        assert set(np.unique(_arr(m)).tolist()) <= {0, 255}

    # Step 2 -- union -> split into instances -> per-view intersections.
    instances, per_inst_views = _split_union_mask_for_multiview(masks, max_k=5)
    inst_arrs = [_arr(m) for m in instances]
    only_a, only_b = _arr(_mask(size, [A])), _arr(_mask(size, [B]))

    # PURPOSE of split: two separate defects -> two separate, non-overlapping instances.
    assert len(instances) == 2
    assert {a.tobytes() for a in inst_arrs} == {only_a.tobytes(), only_b.tobytes()}
    assert np.all(np.minimum(inst_arrs[0], inst_arrs[1]) == 0)  # disjoint instances

    # PURPOSE of intersect: each per-view condition == instance∩view (isolates one
    # instance), and is empty in a view that can't see that instance.
    for i, inst in enumerate(inst_arrs):
        is_a = inst.tobytes() == only_a.tobytes()
        for v, view in enumerate(masks):
            np.testing.assert_array_equal(_arr(per_inst_views[i][v]), np.minimum(inst, _arr(view)))
        lacking_view = 2 if is_a else 1  # view 2 lacks A; view 1 lacks B
        assert _arr(per_inst_views[i][lacking_view]).max() == 0

    for v in range(len(masks)):
        c0, c1 = _arr(per_inst_views[0][v]), _arr(per_inst_views[1][v])
        assert np.all(np.minimum(c0, c1) == 0)                       # instances never overlap in a view
        np.testing.assert_array_equal(np.maximum(c0, c1), _arr(masks[v]))  # together cover the view mask
