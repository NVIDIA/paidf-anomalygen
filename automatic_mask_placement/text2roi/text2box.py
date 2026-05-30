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

"""
Text2Box detector (2pointbox variant): joint point + bounding box from a
single Qwen-VL inference call. Produces detections whose JSON entries are
drop-in compatible with the text2box_bboxes.json schema consumed by
``scripts/amgen_amp_pipeline.py``.

Ported (2pointbox path only) from the sibling ``text2box`` repository: the
detection logic asks the VLM for ``{point_2d, bbox_2d, label}`` in normalized
0-1000 coords, then converts to pixels independently -- the point does not
constrain the box.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


QWEN_JOINT_POINT_BOX_SYSTEM_PROMPT = (
    'You are a precise object detector. Detect the requested region and return a JSON array:\n'
    '[{"bbox_2d": [x1, y1, x2, y2], "point_2d": [cx, cy], "label": "..."}]\n'
    'All coordinates are in 0-1000 normalized scale. '
    'bbox_2d is the bounding box. point_2d is the center point of the region. '
    'Return empty array [] if not found.'
)


_INSTRUCTION_WORDS = ("detect", "find", "locate", "identify", "show", "return", "where")


def _wrap_prompt(prompt: str) -> str:
    return prompt


def _extract_json_array(text: str) -> list:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    objects = []
    for m in re.finditer(r"\{[^{}]+\}", text):
        try:
            objects.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    if objects:
        return objects

    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if len(nums) == 4:
        return [[float(n) if "." in n else int(n) for n in nums]]
    return []


def _clamp_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> list[float]:
    x1 = max(0.0, min(float(x1), w))
    y1 = max(0.0, min(float(y1), h))
    x2 = max(0.0, min(float(x2), w))
    y2 = max(0.0, min(float(y2), h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _clamp_point(x: float, y: float, w: int, h: int) -> list[float]:
    return [max(0.0, min(float(x), w)), max(0.0, min(float(y), h))]


class Text2BoxDetector:
    """Qwen2.5-VL / Qwen3-VL based joint point + bounding box detector.

    Output per detection is a plain dict so downstream AMP code can consume it
    without pulling in text2box dataclasses:
        {"bbox":  [x0, y0, x1, y1],   # pixel coords
         "point": [x,  y],            # pixel coords, independent of bbox
         "label": str,
         "confidence": float}
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str = "cuda",
        seed: int = 0,
    ):
        self.model_id = model_id
        self.device = device
        self.seed = seed
        self.model = None
        self.processor = None

    def load(self) -> None:
        if self.model is not None:
            return

        from transformers import AutoConfig, AutoProcessor

        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        if getattr(config, "model_type", "") == "qwen3_vl" or "Qwen3" in self.model_id:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self.model = ModelClass.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=self.device if self.device == "auto" else {"": self.device},
        )

    def unload(self) -> None:
        """Free VRAM between detection and downstream stages."""
        import gc

        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def _generate_raw(self, messages: list[dict], max_new_tokens: int = 1024) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Pin the RNG to make the model's built-in sampling reproducible.
        # Qwen3-VL's generation_config uses do_sample=True, temperature=0.7
        # by design, and overriding to greedy decoding stably picks the
        # degenerate [0,0,1000,1000] full-image bbox on certain cluttered
        # images. Keeping the trained sampling distribution but seeding the
        # RNG gives deterministic-per-image output without that failure mode.
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = output_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def detect(
        self,
        image: Image.Image,
        prompt: str,
        *,
        box_threshold: float = 0.3,
        max_detections: int = 50,
    ) -> list[dict[str, Any]]:
        self.load()

        w, h = image.size
        wrapped = _wrap_prompt(prompt)
        messages = [
            {"role": "system", "content": QWEN_JOINT_POINT_BOX_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": wrapped},
                ],
            },
        ]

        raw = self._generate_raw(messages)
        items = _extract_json_array(raw)

        if (
            len(items) == 4
            and all(isinstance(v, (int, float)) for v in items)
        ):
            items = [items]

        detections: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                bbox_raw = item.get("bbox_2d") or item.get("bbox")
                pt_raw = item.get("point_2d") or item.get("point")
                label = item.get("label", "object")
                conf = float(item.get("confidence", 1.0))
            elif isinstance(item, (list, tuple)) and len(item) == 4:
                bbox_raw = item
                pt_raw = None
                label = "object"
                conf = 1.0
            else:
                continue

            if not bbox_raw or len(bbox_raw) != 4:
                continue
            if conf < box_threshold:
                continue

            x1, y1, x2, y2 = bbox_raw
            bbox_px = _clamp_box(
                x1 / 1000 * w, y1 / 1000 * h,
                x2 / 1000 * w, y2 / 1000 * h,
                w, h,
            )

            point_px: list[float] | None = None
            if pt_raw and len(pt_raw) == 2:
                point_px = _clamp_point(
                    pt_raw[0] / 1000 * w,
                    pt_raw[1] / 1000 * h,
                    w, h,
                )

            detections.append({
                "bbox": bbox_px,
                "point": point_px,
                "label": label,
                "confidence": conf,
            })
            if len(detections) >= max_detections:
                break

        return detections


def draw_overlay(
    image: Image.Image | np.ndarray,
    bbox: list[float] | None,
    point: list[float] | None,
    label: str = "",
) -> np.ndarray:
    """Return a BGR numpy image with red bbox + green filled circle + crosshair."""
    if isinstance(image, Image.Image):
        canvas = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        canvas = image.copy()

    h, w = canvas.shape[:2]
    line_w = max(2, w // 500)

    if bbox is not None and len(bbox) == 4:
        x0, y0, x1, y1 = (int(round(v)) for v in bbox)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 255), thickness=line_w)
        if label:
            font_scale = max(0.5, w / 3000)
            font_thick = max(1, w // 1200)
            tx, ty = x0 + 6, max(y0 - 12, 22)
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), font_thick + 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    if point is not None and len(point) == 2:
        px, py = (int(round(v)) for v in point)
        radius = max(6, w // 200)
        cv2.circle(canvas, (px, py), radius, (0, 200, 0), thickness=-1)
        cv2.line(canvas, (px - radius * 2, py), (px + radius * 2, py), (0, 200, 0), line_w)
        cv2.line(canvas, (px, py - radius * 2), (px, py + radius * 2), (0, 200, 0), line_w)

    return canvas


def run_text2box(
    image_path: str | Path,
    prompt: str,
    *,
    detector: Text2BoxDetector | None = None,
    box_threshold: float = 0.3,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run 2pointbox on a single image and return an AMP-compatible entry.

    Returns a dict matching the schema consumed by ``amgen_amp_pipeline.py``:
        {"image_path":  str (as given by caller),
         "text_prompt": str,
         "confidence":  float,
         "bbox":        [x0, y0, x1, y1],
         "point":       [x, y] | None,
         "image_size":  [W, H]}

    If the model returns no detections, the entry has bbox/point=None and
    confidence=0.0 -- callers should filter or handle that.
    """
    image_path = str(image_path)
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    det = detector or Text2BoxDetector()
    results = det.detect(image, prompt, box_threshold=box_threshold)

    if results:
        top = max(results, key=lambda r: r["confidence"])
        bbox = top["bbox"]
        point = top["point"]
        conf = top["confidence"]
        label = top.get("label", "")
    else:
        bbox, point, conf, label = None, None, 0.0, ""

    entry = {
        "image_path": image_path,
        "text_prompt": prompt,
        "confidence": conf,
        "bbox": bbox,
        "point": point,
        "image_size": [w, h],
    }

    if output_path is not None:
        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        overlay = draw_overlay(image, bbox, point, label=label or prompt[:40])
        cv2.imwrite(output_path, overlay)

    return entry
