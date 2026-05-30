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

"""Convert an anomaly-generation dataset to TAO DAFT v3.0 format.

Two input layouts are auto-detected:

1. "component_defect" — a labeled dataset:
       <input_dir>/<component>/anomaly_image/<defect>/*.png
       <input_dir>/<component>/mask/<defect>/<basename>_mask.png   (optional)

2. "inference_output" — the SDG inference-result directory. Only
   reconstructed_image/ and original_mask/ are consumed; other sibling
   folders (annotated_image/, cropped_image/, cropped_mask/, original_image/)
   are ignored.
       <input_dir>/reconstructed_image/<type>_<idx5>.png   -> raw/rgb/
       <input_dir>/original_mask/<type>_<idx5>.png         -> raw/mask/
       <input_dir>/SDG_result.csv                          -> task/

Output scene (<output_dir>):
    raw/rgb/image_<N>.png                      (canonical RGB)
    raw/mask/image_<N>.png                     (segmentation mask; same filename as rgb)
    contextual/image_<N>.json                  (image metadata, v3.0 schema)
    task/<validation_jsonl_basename>           (if --validation-jsonl)
    task/<inference_jsonl_basename>            (if --inference-jsonl)
    task/SDG_result.csv                        (inference_output only)

N is zero-padded to max(6, len(str(total-1))) — 6-digit minimum, grows as needed.

Example:
    python scripts/anomaly_gen/convert_to_daft_format.py --input datasets/.../train
    python scripts/anomaly_gen/convert_to_daft_format.py --input .../example_output
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
from pathlib import Path

from PIL import Image

from imaginaire.utils import log

VERSION = "3.0"

INFERENCE_MARKERS = {"reconstructed_image", "original_mask"}


def image_size(img_path: Path) -> tuple[int, int]:
    with Image.open(img_path) as im:
        return im.width, im.height


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        return True
    return src.exists()


def id_width_for(total: int) -> int:
    return max(6, len(str(max(total - 1, 0))))


def scenario_csv(component: str, defect: str, filename: str) -> str:
    """Return '{component},{defect},{filename}' as 3-column CSV.

    Any comma inside the values is replaced with '_' so the field stays
    safely parseable with a plain split(','). None of the current inputs
    contain commas, but the guard is cheap and keeps the contract explicit.
    """
    return ",".join(v.replace(",", "_") for v in (component, defect, filename))


# ---------------------------------------------------------------------------
# Layout detection + dispatch
# ---------------------------------------------------------------------------

def detect_layout(input_dir: Path) -> str:
    children = {p.name for p in input_dir.iterdir() if p.is_dir()}
    if INFERENCE_MARKERS.issubset(children):
        return "inference_output"
    for p in input_dir.iterdir():
        if p.is_dir() and (p / "anomaly_image").is_dir():
            return "component_defect"
    raise ValueError(
        f"Cannot detect dataset layout at {input_dir}. Expected either "
        "<component>/anomaly_image/<defect>/*.png or an inference output "
        "directory with reconstructed_image/, original_image/, etc."
    )


# ---------------------------------------------------------------------------
# Route 1: component/defect labeled dataset
# ---------------------------------------------------------------------------

def enumerate_component_defect(input_dir: Path) -> list[dict]:
    entries: list[dict] = []
    for comp_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        anomaly_root = comp_dir / "anomaly_image"
        mask_root = comp_dir / "mask"
        if not anomaly_root.is_dir():
            continue
        for defect_dir in sorted(p for p in anomaly_root.iterdir() if p.is_dir()):
            for img_path in sorted(defect_dir.glob("*.png")):
                entries.append(
                    {
                        "component": comp_dir.name,
                        "defect": defect_dir.name,
                        "img_path": img_path,
                        "mask_path": mask_root / defect_dir.name / f"{img_path.stem}_mask.png",
                    }
                )
    return entries


def build_scene_component_defect(args: argparse.Namespace) -> None:
    input_dir: Path = args.input
    scene_dir: Path = args.output

    rgb_dir = scene_dir / "raw" / "rgb"
    mask_dir = scene_dir / "raw" / "mask"
    ctx_dir = scene_dir / "contextual"
    for d in (rgb_dir, mask_dir, ctx_dir):
        d.mkdir(parents=True, exist_ok=True)

    entries = enumerate_component_defect(input_dir)
    total = len(entries)
    id_width = id_width_for(total)

    per_defect_count: dict[str, int] = {}
    for idx, entry in enumerate(entries):
        component = entry["component"]
        defect = entry["defect"]
        img_path: Path = entry["img_path"]
        mask_path: Path = entry["mask_path"]
        basename = img_path.stem
        image_id = f"image_{idx:0{id_width}d}"

        per_defect_count[defect] = per_defect_count.get(defect, 0) + 1

        copy_if_exists(img_path, rgb_dir / f"{image_id}.png")
        copy_if_exists(mask_path, mask_dir / f"{image_id}.png")

        width, height = image_size(img_path)
        write_image_json(
            ctx_dir / f"{image_id}.json",
            image_id=image_id,
            height=height,
            width=width,
            scenario_info=scenario_csv(component, defect, f"{basename}.png"),
            description=f"{component} image with {defect} defect",
            date=args.date,
            license_str=args.license_str,
        )

    copy_task_files(scene_dir, args.validation_jsonl, args.inference_jsonl)
    log.info(f"{scene_dir}: {total} samples")
    for defect, count in sorted(per_defect_count.items()):
        log.info(f"{defect}: {count}")


# ---------------------------------------------------------------------------
# Route 2: inference output directory
# ---------------------------------------------------------------------------

def parse_inference_stem(stem: str) -> tuple[str, str, str]:
    """'Capacitor+bridge_00000' -> (component, defect, idx_str)."""
    anomaly_type, _, idx_str = stem.rpartition("_")
    component, _, defect = anomaly_type.partition("+")
    if not component or not defect or not idx_str.isdigit():
        raise ValueError(f"Unparseable inference filename stem: {stem!r}")
    return component, defect, idx_str


def build_scene_inference_output(args: argparse.Namespace) -> None:
    input_dir: Path = args.input
    scene_dir: Path = args.output

    rgb_dir = scene_dir / "raw" / "rgb"
    mask_dir = scene_dir / "raw" / "mask"
    ctx_dir = scene_dir / "contextual"
    for d in (rgb_dir, mask_dir, ctx_dir):
        d.mkdir(parents=True, exist_ok=True)

    recon_src = input_dir / "reconstructed_image"
    orig_msk_src = input_dir / "original_mask"

    entries = sorted(recon_src.glob("*.png"))
    total = len(entries)
    id_width = id_width_for(total)

    per_type_count: dict[str, int] = {}
    for idx, recon_path in enumerate(entries):
        stem = recon_path.stem
        component, defect, sample_idx = parse_inference_stem(stem)
        anomaly_type = f"{component}+{defect}"
        per_type_count[anomaly_type] = per_type_count.get(anomaly_type, 0) + 1

        image_id = f"image_{idx:0{id_width}d}"

        copy_if_exists(recon_path, rgb_dir / f"{image_id}.png")
        copy_if_exists(orig_msk_src / f"{stem}.png", mask_dir / f"{image_id}.png")

        width, height = image_size(recon_path)
        write_image_json(
            ctx_dir / f"{image_id}.json",
            image_id=image_id,
            height=height,
            width=width,
            scenario_info=scenario_csv(component, defect, f"{stem}.png"),
            description=f"Reconstructed {component} image with {defect} defect",
            date=args.date,
            license_str=args.license_str,
        )

    csv_src = input_dir / "SDG_result.csv"
    extra = [csv_src] if csv_src.exists() else []
    copy_task_files(
        scene_dir, args.validation_jsonl, args.inference_jsonl, *extra
    )

    log.info(f"{scene_dir}: {total} samples")
    for t, count in sorted(per_type_count.items()):
        log.info(f"{t}: {count}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def write_image_json(
    path: Path,
    *,
    image_id: str,
    height: int,
    width: int,
    scenario_info: str,
    description: str,
    date: str,
    license_str: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "version": VERSION,
                "image_id": image_id,
                "format": "png",
                "scenario_info": scenario_info,
                "metadata": {
                    "type": "image",
                    "date": date,
                    "description": description,
                    "license": license_str,
                },
                "height": height,
                "width": width,
            },
            indent=2,
        )
    )


def copy_task_files(scene_dir: Path, *sources: Path | None) -> None:
    """Copy provided task files verbatim into <scene_dir>/task/."""
    real_sources = [p for p in sources if p is not None]
    if not real_sources:
        return
    task_dir = scene_dir / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    for src in real_sources:
        shutil.copy2(src, task_dir / src.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Dataset root. Supported layouts: component/defect or inference_output.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output scene directory. Defaults to <input>_daft_v3 alongside the "
            "input directory."
        ),
    )
    ap.add_argument(
        "--validation-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL copied as-is into <output>/task/",
    )
    ap.add_argument(
        "--inference-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL copied as-is into <output>/task/",
    )
    ap.add_argument(
        "--date",
        default=datetime.date.today().isoformat(),
        help="ISO 8601 date for metadata.date (default: today)",
    )
    ap.add_argument(
        "--license",
        default="CC-BY-4.0",
        dest="license_str",
        help="License identifier for metadata.license",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output is None:
        args.output = args.input.parent / f"{args.input.name}_daft_v3"
    layout = detect_layout(args.input)
    log.info(f"Detected layout: {layout}")
    if layout == "component_defect":
        build_scene_component_defect(args)
    elif layout == "inference_output":
        build_scene_inference_output(args)
    else:
        raise RuntimeError(f"Unhandled layout: {layout}")


if __name__ == "__main__":
    main()
