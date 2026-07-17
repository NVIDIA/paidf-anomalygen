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

"""Regression tests for keep_largest_and_fill hole-filling.

Guards the bug where an object touching the top-left (0, 0) corner made the
flood seed land on foreground, so the background was never flooded and the
whole mask was turned white.
"""

import numpy as np

from automatic_mask_placement.text2roi.sam_utils import keep_largest_and_fill


def _mask255(h, w):
    return np.zeros((h, w), np.uint8)


def test_corner_object_with_hole_is_not_all_white():
    """Object touching (0, 0) with an enclosed hole: hole is filled, but the
    exterior background must stay 0 (previously the whole frame went white)."""
    m = _mask255(20, 20)
    m[0:15, 0:15] = 255          # filled square anchored at the (0, 0) corner
    m[5:10, 5:10] = 0            # enclosed hole inside it

    out = keep_largest_and_fill(m)

    # The enclosed hole must be filled in.
    assert out[7, 7] == 255
    # The exterior background far from the object must remain background. This
    # is the regression assertion: the old code set it to 255 (all-white).
    assert out[17, 17] == 0
    # And the result must not be a degenerate all-white mask.
    assert not np.all(out == 255)


def test_corner_object_without_hole_is_preserved():
    """Object touching (0, 0) with no hole must be returned unchanged, not
    inverted to an all-white mask."""
    m = _mask255(20, 20)
    m[0:10, 0:10] = 255          # square anchored at the corner, no hole

    out = keep_largest_and_fill(m)

    assert out[3, 3] == 255       # object preserved
    assert out[15, 15] == 0       # background preserved
    assert not np.all(out == 255)


def test_center_object_with_hole_still_filled():
    """Sanity: an object that does NOT touch the corner (the case that already
    worked) still has its enclosed hole filled and its background untouched."""
    m = _mask255(20, 20)
    m[5:15, 5:15] = 255           # centered square, no corner contact
    m[8:11, 8:11] = 0            # enclosed hole

    out = keep_largest_and_fill(m)

    assert out[9, 9] == 255       # hole filled
    assert out[0, 0] == 0         # exterior background intact
    assert out[18, 18] == 0
