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

"""Convert a TAO DAFT v3.0 scene back to the component/defect dataset layout.

Inverse of convert_to_daft_format.py. Each image_<N>.json contains a
scenario_info string of the form '{component},{defect},{filename}' — that
field drives where each image/mask is placed in the output tree.

Input (DAFT v3.0 scene):
    <input>/raw/rgb/image_<N>.png
    <input>/raw/mask/image_<N>.png             (optional)
    <input>/contextual/image_<N>.json          (scenario_info drives mapping)

Output (train/val-style split):
    <output>/<component>/anomaly_image/<defect>/<original_filename>
    <output>/<component>/mask/<defect>/<original_stem>_mask<ext>   (if mask present)
    <output>/<basename>                                            (task/* files flattened here)

Example:
    python scripts/anomaly_gen/convert_from_daft_format.py \
        --input datasets/uc1_pcbs/val_daft_v3
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from imaginaire.utils import log


def parse_scenario_info(text: str) -> tuple[str, str, str]:
    """Parse '{component},{defect},{filename}' into its three parts."""
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError(
            f"scenario_info must be 'component,defect,filename' (3 fields); got {text!r}"
        )
    return parts[0], parts[1], parts[2]


def default_output(input_dir: Path) -> Path:
    name = input_dir.name
    stem = name[: -len("_daft_v3")] if name.endswith("_daft_v3") else name
    return input_dir.parent / f"{stem}_restored"


def restore_scene(args: argparse.Namespace) -> None:
    scene_dir: Path = args.input
    output_dir: Path = args.output

    ctx_dir = scene_dir / "contextual"
    rgb_dir = scene_dir / "raw" / "rgb"
    mask_dir = scene_dir / "raw" / "mask"

    if not ctx_dir.is_dir():
        raise FileNotFoundError(f"Missing contextual/: {ctx_dir}")
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing raw/rgb/: {rgb_dir}")

    per_class_count: dict[tuple[str, str], int] = {}
    missing_rgb: list[str] = []
    missing_mask: list[str] = []

    for json_path in sorted(ctx_dir.glob("*.json")):
        data = json.loads(json_path.read_text())
        if data.get("metadata", {}).get("type") != "image":
            continue

        image_id = data["image_id"]
        fmt = data.get("format", "png")
        component, defect, filename = parse_scenario_info(data["scenario_info"])

        src_rgb = rgb_dir / f"{image_id}.{fmt}"
        src_mask = mask_dir / f"{image_id}.{fmt}"

        orig = Path(filename)
        dst_rgb_dir = output_dir / component / "anomaly_image" / defect
        dst_mask_dir = output_dir / component / "mask" / defect

        if src_rgb.exists():
            dst_rgb_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_rgb, dst_rgb_dir / orig.name)
        else:
            missing_rgb.append(image_id)

        if src_mask.exists():
            dst_mask_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_mask, dst_mask_dir / f"{orig.stem}_mask{orig.suffix}")
        else:
            missing_mask.append(image_id)

        per_class_count[(component, defect)] = per_class_count.get((component, defect), 0) + 1

    task_src = scene_dir / "task"
    task_copied: list[str] = []
    if task_src.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in sorted(task_src.iterdir()):
            if not item.is_file():
                continue
            shutil.copy2(item, output_dir / item.name)
            task_copied.append(item.name)

    total = sum(per_class_count.values())
    log.info(f"{output_dir}: {total} images")
    for (component, defect), count in sorted(per_class_count.items()):
        log.info(f"{component}/{defect}: {count}")
    if task_copied:
        log.info(f"{len(task_copied)} task file(s) copied: {', '.join(task_copied)}")
    if missing_rgb:
        log.warning(f"Missing RGB for {len(missing_rgb)} image(s); first few: {missing_rgb[:3]}")
    if missing_mask:
        log.warning(f"Missing mask for {len(missing_mask)} image(s); first few: {missing_mask[:3]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="DAFT v3.0 scene directory (must contain raw/ and contextual/).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output split directory. Defaults to <input_without_daft_v3_suffix>_restored "
            "alongside the input."
        ),
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output is None:
        args.output = default_output(args.input)
    restore_scene(args)


if __name__ == "__main__":
    main()
