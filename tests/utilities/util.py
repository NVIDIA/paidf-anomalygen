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
"""Helpers for tests/utilities: run a scripts/utilities CLI as a subprocess.

These tests exercise the scripts through their CLI on purpose — exit codes
and stderr messages are the contract the pipeline shell scripts rely on.
"""
import json
import os
import pathlib
import subprocess
import sys

from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
UTILITIES = REPO_ROOT / "scripts" / "utilities"


def run_script(script, *argv, env=None):
    if str(script).endswith(".sh"):
        cmd = ["bash", str(UTILITIES / script)]
    else:
        cmd = [sys.executable, str(UTILITIES / script)]
    return subprocess.run([*cmd, *map(str, argv)],
                          capture_output=True, text=True, cwd=REPO_ROOT,
                          env={**os.environ, **env} if env else None)


def make_png(path, size=(4, 4), mode="RGB"):
    Image.new(mode, size).save(path)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_checkpoints(root, sizes=("2B",), with_t5_large=True, with_t5_11b=False,
                     with_guardrail=True):
    """Populate a fake ``checkpoints/`` tree with the artifacts ``check.sh`` and
    ``download_checkpoints.sh`` look for (superset of both). Files carry dummy
    content — the scripts only test for presence / non-empty dirs. Returns the
    root as a ``pathlib.Path``.
    """
    root = pathlib.Path(root)

    def touch(rel):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")

    for s in sizes:
        touch(f"nvidia/Cosmos-Predict2-{s}-Text2Image/model.pt")
    if with_t5_large:
        touch("google-t5/t5-large/config.json")
    if with_t5_11b:
        touch("google-t5/t5-11b/config.json")
    if with_guardrail:
        touch("nvidia/Cosmos-Guardrail1/video_content_safety_filter/safety_filter.pt")
        touch("nvidia/Cosmos-Guardrail1/face_blur_filter/Resnet50_Final.pth")
        touch("nvidia/Cosmos-Guardrail1/video_content_safety_filter/"
              "models--google--siglip-so400m-patch14-384/snapshots/abc/model.safetensors")
    touch("NVDINOV2/nv_dinov2_classification_model.ckpt")
    touch("nvidia/C-RADIO-V3/model.safetensors")
    touch("facebook/dinov2-large/config.json")
    touch("sam2/sam2.1_hiera_large.pt")
    touch("Qwen/Qwen3-VL-4B-Instruct/model-00001-of-00002.safetensors")
    touch("Qwen/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors")
    return root
