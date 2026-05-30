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

"""Pipeline configuration helpers for Auto ROI → AMP."""

import json


def load_defect_descriptions(jsonl_path):
    """Load defect descriptions from JSONL file.

    Returns:
        (prompts, spatial_deps) dicts keyed by defect_type.
    """
    prompts = {}
    spatial_deps = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            dt = entry["defect_type"]
            spatial_deps[dt] = entry.get("spatial_dependency", "text")
            roi_prompt = entry.get("roi_prompt_defect_location", "")
            if roi_prompt:
                prompts[dt] = roi_prompt
    return prompts, spatial_deps


def get_sample_route(sample, spatial_deps):
    """Determine route: 'free', 'cad', or 'text'."""
    if "cad_mask" in sample and sample["cad_mask"] is not None:
        return "cad"
    dep = spatial_deps.get(sample["defect_type"], "text")
    if dep == "free":
        return "free"
    return "text"
