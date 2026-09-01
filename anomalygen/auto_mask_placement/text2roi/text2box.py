# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Text2Box detector (2pointbox variant): joint point + bounding box from a single VLM inference call.

The model family is chosen from ``model_id`` — ``nvidia/Cosmos3-Nano`` by default, or a Qwen-VL model
(Qwen3-VL / Qwen2.5-VL) — and the detection system prompt is matched to it
(``COSMOS_JOINT_POINT_BOX_SYSTEM_PROMPT`` vs ``QWEN_JOINT_POINT_BOX_SYSTEM_PROMPT``). The Cosmos prompt was
tuned to reproduce Qwen3-VL-8B outputs (see ``refine_cosmos_prompt.py``).

The VLM is asked for ``{bbox_2d, point_2d, label}`` in normalized 0-1000 coords; results are converted to
pixels independently — the point does not constrain the box. Each detection is returned as a plain dict
(``bbox``/``point`` in pixel coords, ``label``, ``confidence``) for downstream AMP code.
"""

from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoConfig,
    AutoProcessor,
    Cosmos3OmniForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

QWEN_JOINT_POINT_BOX_SYSTEM_PROMPT = (
    "You are a precise object detector. Detect the requested region and return a JSON array:\n"
    '[{"bbox_2d": [x1, y1, x2, y2], "point_2d": [cx, cy], "label": "..."}]\n'
    "All coordinates are in 0-1000 normalized scale. "
    "bbox_2d is the bounding box. point_2d is the center point of the region. "
    "Return empty array [] if not found."
)

# Cosmos3-Nano needs a stricter prompt to match Qwen's output (it otherwise emits corner-tuples,
# omits point_2d/label, and over-covers). Tuned to match Qwen3-VL-8B outputs.
COSMOS_JOINT_POINT_BOX_SYSTEM_PROMPT = (
    "You are a precise object detector. Detect the requested region and return ONLY a JSON array "
    "(no markdown, no extra text).\n"
    'Format each detection as {"bbox_2d": [x1, y1, x2, y2], "point_2d": [cx, cy], "label": "<name>"} '
    "with integer 0-1000 normalized coordinates.\n"
    'Example: [{"bbox_2d": [120, 340, 880, 910], "point_2d": [500, 625], "label": "phone screen"}]\n'
    "bbox_2d must tightly enclose the whole object; point_2d is its center. Return [] if not found."
)


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


def _resolve_model_path(model_id: str) -> str:
    """Prefer a local ``checkpoints/<org>/<name>`` copy (from scripts/download_checkpoints.sh) over the
    HF hub. Falls back to the given ``model_id`` (repo id or explicit path) when no local copy exists."""
    if "/" in model_id and not Path(model_id).exists():
        local = Path(__file__).resolve().parents[3] / "checkpoints" / model_id
        if (local / "config.json").exists():
            return str(local)
    return model_id


class Text2BoxDetector:
    """Cosmos3-Nano / Qwen-VL based joint point + bounding box detector.

    The model family is chosen from ``model_id`` (Cosmos3-Nano by default), and the detection
    system prompt is selected to match: Cosmos3-Nano uses the stricter
    ``COSMOS_JOINT_POINT_BOX_SYSTEM_PROMPT`` (to match Qwen's output format/tightness), Qwen models
    use ``QWEN_JOINT_POINT_BOX_SYSTEM_PROMPT``.

    Output per detection is a plain dict so downstream AMP code can consume it
    without pulling in text2box dataclasses:
        {"bbox":  [x0, y0, x1, y1],   # pixel coords
         "point": [x,  y],            # pixel coords, independent of bbox
         "label": str,
         "confidence": float}
    """

    def __init__(
        self,
        model_id: str = "nvidia/Cosmos3-Nano",
        device: str = "cuda",
        seed: int = 0,
    ):
        self.model_id = model_id
        self.device = device
        self.seed = seed
        self.model = None
        self.processor = None
        self.system_prompt = (
            COSMOS_JOINT_POINT_BOX_SYSTEM_PROMPT
            if "cosmos3" in model_id.lower()
            else QWEN_JOINT_POINT_BOX_SYSTEM_PROMPT
        )

    def load(self) -> None:
        if self.model is not None:
            return

        # Prefer a local checkpoints/<org>/<name> copy if present; else load from the HF hub.
        model_path = _resolve_model_path(self.model_id)
        config = AutoConfig.from_pretrained(model_path)
        model_type = getattr(config, "model_type", "")
        if model_type == "cosmos3_omni" or "cosmos3" in self.model_id.lower():
            ModelClass = Cosmos3OmniForConditionalGeneration
        elif model_type == "qwen3_vl" or "Qwen3" in self.model_id:
            ModelClass = Qwen3VLForConditionalGeneration
        else:
            ModelClass = Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self.model = ModelClass.from_pretrained(
            model_path,
            dtype="auto",
            device_map=self.device if self.device == "auto" else {"": self.device},
        )

    def unload(self) -> None:
        """Free VRAM between detection and downstream stages."""
        self.model = None
        self.processor = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _generate_raw(self, messages: list[dict], max_new_tokens: int = 1024) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        trimmed = output_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

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
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        raw = self._generate_raw(messages)
        items = _extract_json_array(raw)

        if len(items) == 4 and all(isinstance(v, (int, float)) for v in items):
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
                x1 / 1000 * w,
                y1 / 1000 * h,
                x2 / 1000 * w,
                y2 / 1000 * h,
                w,
                h,
            )

            point_px: list[float] | None = None
            if pt_raw and len(pt_raw) == 2:
                point_px = _clamp_point(
                    pt_raw[0] / 1000 * w,
                    pt_raw[1] / 1000 * h,
                    w,
                    h,
                )

            detections.append(
                {
                    "bbox": bbox_px,
                    "point": point_px,
                    "label": label,
                    "confidence": conf,
                }
            )
            if len(detections) >= max_detections:
                break

        return detections


def _draw_overlay(
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
            cv2.putText(
                canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick + 2, cv2.LINE_AA
            )
            cv2.putText(
                canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA
            )

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
        overlay = _draw_overlay(image, bbox, point, label=label or prompt[:40])
        cv2.imwrite(output_path, overlay)

    return entry
