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

import torch
import secrets


"""
Secure random number generation
"""

def secure_randint(x1, x2):
    # Convert to int if x1 or x2 is a tensor since secrets.randbelow only accepts int
    if torch.is_tensor(x1):
        x1 = x1.item()
    if torch.is_tensor(x2):
        x2 = x2.item()

    # secrets.randbelow returns a number in [0, (x2 - x1) + 1)
    return x1 + secrets.randbelow((x2 - x1) + 1)

def secure_random():
    return secrets.randbelow(10**9) / 10**9