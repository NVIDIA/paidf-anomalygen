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
"""Unit tests for ``RandomRotation`` bidirectional angle sampling.

``RandomRotation`` samples an angle with ``secure_randint`` (backed by
``secrets``, which is *not* seedable), so these tests are statistical: the
angle is captured by mocking out ``torchvision.transforms.functional.rotate``
(no real image work, no GPU) and the assertions hold with overwhelming
probability over the sample count.

Verified behaviour (per MR !98):
  * every sampled angle stays inside the symmetric closed range
    ``[-max_angle, max_angle]``;
  * both negative and positive angles are reachable (the fix's point -- the
    old ``secrets.randbelow(max_angle)`` could only produce ``[0, max_angle)``);
  * ``max_angle == 0`` yields angle ``0`` instead of raising ``ValueError``.
"""

from unittest import mock

from cosmos_predict2.data.anomaly_gen.anomaly_dataset_utils import RandomRotation

_MODULE = "cosmos_predict2.data.anomaly_gen.anomaly_dataset_utils"


def _sample_angles(max_angle, n):
    """Return the ``n`` angles ``RandomRotation`` feeds to ``tF.rotate``.

    ``tF`` is mocked, so the image argument is never inspected and a sentinel
    stands in for a real tensor. With ``target=None`` each call rotates exactly
    once, so there is one captured angle per invocation.
    """
    rotate = RandomRotation(max_angle)
    image = object()
    with mock.patch(f"{_MODULE}.tF") as mock_tf:
        for _ in range(n):
            rotate(image, target=None)
        return [call.args[1] for call in mock_tf.rotate.call_args_list]


def test_random_rotation_stays_within_symmetric_range():
    max_angle = 30
    angles = _sample_angles(max_angle, 2000)
    assert len(angles) == 2000
    assert all(-max_angle <= angle <= max_angle for angle in angles)


def test_random_rotation_reaches_negative_and_positive():
    max_angle = 30
    angles = _sample_angles(max_angle, 2000)
    assert min(angles) < 0, "negative angles must be reachable (bidirectional)"
    assert max(angles) > 0, "positive angles must be reachable"


def test_random_rotation_zero_max_angle_does_not_raise():
    # Regression guard: the old secrets.randbelow(0) raised ValueError;
    # secure_randint(0, 0) deterministically returns 0.
    assert _sample_angles(0, 5) == [0, 0, 0, 0, 0]
