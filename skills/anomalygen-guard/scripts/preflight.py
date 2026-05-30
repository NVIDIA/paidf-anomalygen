#!/usr/bin/env python3
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
"""Product-mode preflight checks for AnomalyGen workflows.

This script is intentionally strict: it should fail before expensive GPU work
when inputs are missing, unsupported, or likely to produce misleading results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import re
import sys
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VALID_SPATIAL_DEPENDENCY = {"free", "text", "cad"}
GUIDANCE_RANGE = (1.5, 10.0)
CROP_RATIO_RANGE = (1.5, 10.0)
PRODUCT_MODE_ENV = "ANOMALYGEN_PRODUCT_MODE"


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.info.append(msg)

    def finish(self) -> int:
        for msg in self.info:
            print(f"OK: {msg}")
        for msg in self.warnings:
            print(f"WARN: {msg}")
        if self.errors:
            print("BLOCKED: fix these before running GPU work")
            for msg in self.errors:
                print(f"  - {msg}")
            return 1
        print("READY: safe to continue")
        return 0


def _images(path: pathlib.Path) -> list[pathlib.Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)


def _strip_mask_suffix(path: pathlib.Path) -> str:
    stem = path.stem
    return stem[:-5] if stem.endswith("_mask") else stem


def load_defect_spec(path: pathlib.Path, report: Report) -> list[dict[str, Any]]:
    if not path.is_file():
        report.error(f"defect_spec not found: {path}")
        return []

    entries: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            report.error(f"{path}:{line_no}: invalid JSONL: {exc}")
            continue
        full = entry.get("defect_type")
        if not isinstance(full, str) or "+" not in full:
            report.error(f"{path}:{line_no}: defect_type must be TEXTURE+ANOMALY")
            continue
        sd = entry.get("spatial_dependency", "free")
        if sd not in VALID_SPATIAL_DEPENDENCY:
            report.error(
                f"{full}: spatial_dependency must be one of "
                f"{sorted(VALID_SPATIAL_DEPENDENCY)} (got {sd!r})"
            )
        if sd == "text" and not (entry.get("roi_prompt_defect_location") or "").strip():
            report.error(f"{full}: spatial_dependency='text' requires non-empty roi_prompt_defect_location")
        entries.append(entry)

    if not entries:
        report.error(f"defect_spec has no valid defect entries: {path}")
    else:
        report.note(f"defect_spec parsed ({len(entries)} defect type(s))")
    return entries


def validate_mode(args: argparse.Namespace, report: Report) -> None:
    if args.mode == "inference_only":
        if not args.checkpoint_dir or args.step is None:
            report.error("inference_only requires both --checkpoint-dir and --step")
    elif args.mode == "full":
        if args.checkpoint_dir or args.step is not None:
            report.error("full mode runs finetune; do not pass checkpoint_dir or step")
    elif args.mode == "finetune_only":
        if args.checkpoint_dir or args.step is not None:
            report.warn("finetune_only ignores checkpoint_dir/step; it does not resume")

    if args.mode != "finetune_only":
        if args.num_sdg is None:
            report.error(f"{args.mode} requires --num-sdg")
        elif args.num_sdg <= 0:
            report.error(f"num_SDG must be > 0 (got {args.num_sdg})")

    if args.num_search_run is not None and args.num_search_run < 0:
        report.error(f"num_search_run must be >= 0 (got {args.num_search_run})")
    if args.num_search_run is not None and args.num_search_run > 5:
        report.warn(
            f"num_search_run={args.num_search_run} can launch many SDG rounds; "
            "confirm the runtime budget"
        )

    if args.model_size == "14b":
        report.warn("model_size=14b is slower and requires more VRAM than 2b")


def validate_dataset(
    dataset_dir: pathlib.Path,
    entries: list[dict[str, Any]],
    report: Report,
) -> None:
    if not dataset_dir.is_dir():
        report.error(f"dataset_dir not found: {dataset_dir}")
        return

    has_cad = any(e.get("spatial_dependency", "free") == "cad" for e in entries)
    labels = dataset_dir / "semantic_segmentation_labels.json"
    if has_cad and not labels.is_file():
        report.error(f"missing {labels} (required for spatial_dependency=cad)")

    for entry in entries:
        full = entry["defect_type"]
        texture, anomaly = full.split("+", 1)
        sd = entry.get("spatial_dependency", "free")

        anomaly_dir = dataset_dir / texture / "anomaly_image" / anomaly
        mask_dir = dataset_dir / texture / "mask" / anomaly
        anomaly_images = _images(anomaly_dir)
        masks = _images(mask_dir)
        if not anomaly_images:
            report.error(f"{full}: no anomaly images under {anomaly_dir}")
        if not masks:
            report.error(f"{full}: no masks under {mask_dir}")

        anomaly_stems = {p.stem for p in anomaly_images}
        paired = sum(1 for m in masks if _strip_mask_suffix(m) in anomaly_stems)
        if anomaly_images and masks and paired == 0:
            report.error(
                f"{full}: no image/mask pairs; masks should use the _mask suffix"
            )
        elif anomaly_images and masks:
            report.note(
                f"{full}: {len(anomaly_images)} image(s), {len(masks)} mask(s), "
                f"{paired} paired"
            )

        if sd == "cad":
            cad_dir = dataset_dir / texture / "cad_mask"
            if not cad_dir.is_dir():
                report.error(f"{full}: missing CAD mask directory {cad_dir}")


def _supported_types_from_checkpoint(
    checkpoint_dir: pathlib.Path, report: Report
) -> set[str]:
    cfg_path = checkpoint_dir / "ag_config.yaml"
    if not cfg_path.is_file():
        report.error(f"checkpoint missing ag_config.yaml: {cfg_path}")
        return set()
    if yaml is None:
        report.error("PyYAML is required to inspect checkpoint ag_config.yaml")
        return set()
    try:
        cfg = yaml.safe_load(cfg_path.read_text())
        raw = cfg["dataloader_train"]["dataset"]["anomaly_types"]
    except Exception as exc:  # noqa: BLE001 - convert config errors to UX errors.
        report.error(f"failed to read anomaly_types from {cfg_path}: {exc}")
        return set()
    supported = {f"{item[0]}+{item[1]}" for item in raw}
    if not supported:
        report.error(f"checkpoint has empty anomaly_types: {cfg_path}")
    else:
        report.note(f"checkpoint supports {len(supported)} defect type(s)")
    return supported


def _available_steps(checkpoint_dir: pathlib.Path) -> list[int]:
    steps: set[int] = set()
    for path in (checkpoint_dir / "checkpoints" / "model").glob("iter_*.pt"):
        match = re.search(r"iter_(\d+)", path.name)
        if match:
            steps.add(int(match.group(1)))
    return sorted(steps)


def _infer_model_size(checkpoint_dir: pathlib.Path) -> str | None:
    text = str(checkpoint_dir)
    if re.search(r"(^|[_/-])2B([_/-]|$)", text):
        return "2b"
    if re.search(r"(^|[_/-])14B([_/-]|$)", text):
        return "14b"
    cfg_path = checkpoint_dir / "ag_config.yaml"
    if cfg_path.is_file() and yaml is not None:
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            name = str(cfg.get("job", {}).get("name", ""))
            if "_2B_" in name:
                return "2b"
            if "_14B_" in name:
                return "14b"
        except Exception:
            return None
    return None


def validate_checkpoint(
    checkpoint_dir: pathlib.Path | None,
    step: int | None,
    model_size: str,
    entries: list[dict[str, Any]],
    report: Report,
) -> set[str]:
    if checkpoint_dir is None:
        return set()
    if not checkpoint_dir.is_dir():
        report.error(f"checkpoint_dir not found: {checkpoint_dir}")
        return set()

    supported = _supported_types_from_checkpoint(checkpoint_dir, report)
    requested = {e["defect_type"] for e in entries}
    unsupported = sorted(requested - supported) if supported else []
    if unsupported:
        report.error(
            "defect_spec requests defect(s) not supported by checkpoint: "
            + ", ".join(unsupported)
        )

    inferred = _infer_model_size(checkpoint_dir)
    if inferred and inferred != model_size:
        report.error(
            f"model_size={model_size} does not match checkpoint model size {inferred}"
        )
    elif inferred:
        report.note(f"checkpoint model size matches requested model_size={model_size}")
    else:
        report.warn("could not infer checkpoint model size from path/config")

    if step is not None:
        steps = _available_steps(checkpoint_dir)
        if not steps:
            report.error(
                f"no saved iter_*.pt checkpoints found under "
                f"{checkpoint_dir / 'checkpoints' / 'model'}"
            )
        elif step not in steps:
            report.error(
                f"step {step} not found; available steps: "
                + ", ".join(str(s) for s in steps[:20])
            )
        else:
            report.note(f"saved step found: {step}")

    return supported


def load_jsonl(path: pathlib.Path, report: Report, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        report.error(f"{label} not found: {path}")
        return []
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            report.error(f"{path}:{line_no}: invalid JSON: {exc}")
    if rows:
        report.note(f"{label} parsed ({len(rows)} row(s))")
    else:
        report.error(f"{label} has no rows: {path}")
    return rows


def validate_input_jsonl(
    input_jsonl: pathlib.Path | None,
    supported: set[str],
    report: Report,
    label: str = "input_jsonl",
    allowed_types: set[str] | None = None,
    require_full_coverage: set[str] | None = None,
) -> list[dict[str, Any]]:
    if input_jsonl is None:
        return []
    rows = load_jsonl(input_jsonl, report, label)
    seen_types: set[str] = set()
    for i, row in enumerate(rows):
        for key in ("image_filename", "mask_filename", "anomaly_type"):
            if key not in row:
                report.error(f"{input_jsonl}:{i + 1}: missing required key {key}")
        for key in ("image_filename", "mask_filename"):
            path_value = row.get(key)
            if isinstance(path_value, str) and not pathlib.Path(path_value).exists():
                report.error(f"{input_jsonl}:{i + 1}: missing {key}: {path_value}")
        anomaly_type = row.get("anomaly_type")
        if isinstance(anomaly_type, str):
            seen_types.add(anomaly_type)
        if allowed_types is not None and anomaly_type not in allowed_types:
            report.error(
                f"{input_jsonl}:{i + 1}: anomaly_type {anomaly_type!r} "
                "is not listed in defect_spec"
            )
        if supported and anomaly_type not in supported:
            report.error(f"{input_jsonl}:{i + 1}: unsupported anomaly_type {anomaly_type}")
    if require_full_coverage is not None:
        missing = require_full_coverage - seen_types
        if missing:
            report.error(
                f"{input_jsonl}: {label} must cover every defect_spec type; "
                f"missing: {sorted(missing)}"
            )
    return rows


def validate_complete_output(
    input_rows: list[dict[str, Any]],
    generated_dir: pathlib.Path | None,
    report: Report,
) -> None:
    if generated_dir is None:
        report.error("--require-complete-output requires --generated-dir")
        return
    if not input_rows:
        report.error("--require-complete-output requires a valid --input-jsonl")
        return
    if not generated_dir.is_dir():
        report.error(f"generated_dir not found: {generated_dir}")
        return

    expected = len(input_rows)
    csv_path = generated_dir / "SDG_result.csv"
    if not csv_path.is_file():
        report.error(f"SDG_result.csv not found: {csv_path}")
    else:
        with csv_path.open() as fp:
            csv_rows = max(sum(1 for _ in fp) - 1, 0)
        if csv_rows != expected:
            report.error(
                f"SDG_result.csv row count mismatch: expected {expected}, got {csv_rows}"
            )
        else:
            report.note(f"SDG_result.csv row count matches input ({expected})")

    recon = generated_dir / "reconstructed_image"
    images = _images(recon)
    if len(images) != expected:
        report.error(
            f"reconstructed_image count mismatch: expected {expected}, got {len(images)}"
        )
    else:
        report.note(f"reconstructed_image count matches input ({expected})")


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def validate_draws(
    base_jsonl: pathlib.Path | None,
    draws_json: pathlib.Path | None,
    report: Report,
) -> None:
    if draws_json is None and base_jsonl is None:
        return
    if draws_json is None or base_jsonl is None:
        report.error("--base-jsonl and --draws-json must be provided together")
        return

    base_rows = load_jsonl(base_jsonl, report, "base_jsonl")
    if not draws_json.is_file():
        report.error(f"draws_json not found: {draws_json}")
        return
    try:
        draws = json.loads(draws_json.read_text())
    except json.JSONDecodeError as exc:
        report.error(f"{draws_json}: invalid JSON: {exc}")
        return
    if not isinstance(draws, dict):
        report.error("draws_json must be an object: {\"<sample_index>\": {...}}")
        return
    if not draws:
        report.warn("draws_json is empty; the refine round will retry no samples")

    for key, value in draws.items():
        try:
            idx = int(key)
        except ValueError:
            report.error(f"draws index must be an integer string (got {key!r})")
            continue
        if idx < 0 or idx >= len(base_rows):
            report.error(f"draws index {idx} out of range for {len(base_rows)} JSONL rows")
            continue
        if not isinstance(value, dict):
            report.error(f"draws[{key}] must be an object")
            continue
        guidance = _as_number(value.get("guidance"))
        crop_ratio = _as_number(value.get("crop_ratio"))
        if guidance is None:
            report.error(f"draws[{key}].guidance must be a finite number")
        elif not (GUIDANCE_RANGE[0] <= guidance <= GUIDANCE_RANGE[1]):
            report.error(
                f"draws[{key}].guidance={guidance} outside safe range "
                f"{GUIDANCE_RANGE}"
            )
        if crop_ratio is None:
            report.error(f"draws[{key}].crop_ratio must be a finite number")
        elif not (CROP_RATIO_RANGE[0] <= crop_ratio <= CROP_RATIO_RANGE[1]):
            report.error(
                f"draws[{key}].crop_ratio={crop_ratio} outside safe range "
                f"{CROP_RATIO_RANGE}"
            )

    if base_rows:
        report.note(f"draws_json covers {len(draws)} of {len(base_rows)} sample(s)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=["full", "inference_only", "finetune_only"])
    p.add_argument("--name", required=True)
    p.add_argument("--dataset-dir", required=True, type=pathlib.Path)
    p.add_argument("--clean-dir", type=pathlib.Path)
    p.add_argument("--defect-spec", required=True, type=pathlib.Path)
    p.add_argument("--num-sdg", type=int)
    p.add_argument("--num-search-run", type=int, default=0)
    p.add_argument("--checkpoint-dir", type=pathlib.Path)
    p.add_argument("--step", type=int)
    p.add_argument("--model-size", default="2b", choices=["2b", "14b"])
    p.add_argument("--input-jsonl", type=pathlib.Path)
    p.add_argument("--validation-jsonl", type=pathlib.Path)
    p.add_argument("--generated-dir", type=pathlib.Path)
    p.add_argument("--require-complete-output", action="store_true")
    p.add_argument("--base-jsonl", type=pathlib.Path)
    p.add_argument("--draws-json", type=pathlib.Path)
    return p.parse_args()


def main() -> int:
    if os.environ.get(PRODUCT_MODE_ENV) != "1":
        print(
            f"SKIP: {PRODUCT_MODE_ENV} is not 1; "
            "AnomalyGen guard preflight is disabled in develop mode."
        )
        return 0

    args = parse_args()
    report = Report()

    validate_mode(args, report)
    entries = load_defect_spec(args.defect_spec, report)
    if entries:
        validate_dataset(args.dataset_dir, entries, report)
    supported = validate_checkpoint(
        args.checkpoint_dir, args.step, args.model_size, entries, report
    )
    requested_types = {entry["defect_type"] for entry in entries}
    validate_input_jsonl(
        args.validation_jsonl,
        set(),
        report,
        label="validation_jsonl",
        allowed_types=requested_types,
        require_full_coverage=requested_types,
    )
    input_rows = validate_input_jsonl(args.input_jsonl, supported, report)
    if args.require_complete_output:
        validate_complete_output(input_rows, args.generated_dir, report)
    validate_draws(args.base_jsonl, args.draws_json, report)

    if args.name in {"test", "tmp", "default"}:
        report.warn(f"name={args.name!r} is generic; use unique experiment names")

    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
