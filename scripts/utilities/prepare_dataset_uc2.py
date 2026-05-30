#!/usr/bin/env python3
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
Prepare UC2 dataset (Magnetic Tile Defect) for AnomalyGen training.

Downloads from: https://github.com/abin24/Magnetic-tile-defect-datasets.

Source layout per defect folder:
  MT_<TYPE>/Imgs/
    exp*_num_*.jpg   <- defect image
    exp*_num_*.png   <- pixel-level mask

  MT_Free/Imgs/
    exp*_num_*.jpg   <- clean (normal) image
    exp*_num_*.png   <- ignored

Output layout:
  <output_dir>/
    metal_surface/
      anomaly_image/<TYPE>/   exp*_num_*.png   (jpg converted to png)
      mask/<TYPE>/            exp*_num_*_mask.png
      clean_image/            exp*_num_*.jpg
    defect_spec.jsonl

Usage:
  The dataset is downloaded automatically from GitHub — no manual step required.

  Step 1 — Run the script (from repo root, or inside the container):
    python3 -m scripts.utilities.prepare_dataset_uc2 <output_dir>

  Optional — keep the raw downloaded zip for debugging:
    python3 -m scripts.utilities.prepare_dataset_uc2 <output_dir> --keep-download /tmp/magnetic_tile_raw
"""

import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ZIP_URL = (
    "https://github.com/abin24/Magnetic-tile-defect-datasets./archive/refs/heads/master.zip"
)
DEFECT_TYPES = ["MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Uneven"]
CLEAN_TYPE = "MT_Free"

DEFECT_SPEC = [
    {"defect_type": "metal_surface+MT_Blowhole", "spatial_dependency": "free", "roi_prompt_defect_location": ""},
    {"defect_type": "metal_surface+MT_Break",    "spatial_dependency": "free", "roi_prompt_defect_location": ""},
    {"defect_type": "metal_surface+MT_Crack",    "spatial_dependency": "free", "roi_prompt_defect_location": ""},
    {"defect_type": "metal_surface+MT_Fray",     "spatial_dependency": "free", "roi_prompt_defect_location": ""},
    {"defect_type": "metal_surface+MT_Uneven",   "spatial_dependency": "free", "roi_prompt_defect_location": ""},
]

# Curated subset: exactly the files shipped in the reference UC2_data dataset.
KEEP_ANOMALY = {
    "MT_Blowhole": {"exp1_num_265077", "exp1_num_4727", "exp2_num_51697", "exp5_num_297552", "exp5_num_7054"},
    "MT_Break":    {"exp2_num_116961", "exp2_num_271384", "exp3_num_148977", "exp3_num_26146", "exp5_num_98336"},
    "MT_Crack":    {"exp1_num_265613", "exp1_num_276355", "exp1_num_339819", "exp4_num_116553", "exp5_num_116575"},
    "MT_Fray":     {"exp1_num_135544", "exp1_num_331149", "exp3_num_135593", "exp3_num_136351", "exp3_num_20409"},
    "MT_Uneven":   {"exp0_num_461", "exp5_num_109313", "exp5_num_352540", "exp6_num_139916", "exp6_num_172365"},
}
KEEP_CLEAN = {
    "exp0_num_743", "exp1_num_10181", "exp1_num_10334", "exp1_num_10903", "exp1_num_11276",
    "exp1_num_13526", "exp1_num_16503", "exp1_num_1810", "exp1_num_18129", "exp1_num_19695",
    "exp1_num_2038", "exp1_num_2610", "exp1_num_3504", "exp1_num_3786", "exp1_num_5907",
    "exp1_num_6508", "exp1_num_6539", "exp1_num_7871", "exp1_num_824", "exp1_num_855",
}


def download_and_extract(dest: Path) -> Path:
    zip_path = dest / "magnetic_tile.zip"
    print(f"Downloading {REPO_ZIP_URL} ...")
    try:
        urllib.request.urlretrieve(REPO_ZIP_URL, zip_path)
    except Exception as e:
        sys.exit(f"Download failed: {e}")

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()

    extracted = sorted(dest.iterdir())
    if not extracted:
        sys.exit("Extraction produced no files.")
    return extracted[0]


def process_defect_type(src_imgs: Path, dst_anomaly: Path, dst_mask: Path, keep: set):
    try:
        from PIL import Image
        use_pil = True
    except ImportError:
        use_pil = False

    dst_anomaly.mkdir(parents=True, exist_ok=True)
    dst_mask.mkdir(parents=True, exist_ok=True)

    jpg_files = [f for f in sorted(src_imgs.glob("*.jpg")) if f.stem in keep]
    png_files = [f for f in sorted(src_imgs.glob("*.png")) if f.stem in keep]

    for jpg in jpg_files:
        dst = dst_anomaly / f"{jpg.stem}.png"
        if use_pil:
            Image.open(jpg).convert("RGB").save(dst)
        else:
            shutil.copy2(jpg, dst)

    for png in png_files:
        shutil.copy2(png, dst_mask / f"{png.stem}_mask.png")

    print(f"  anomaly_image/{dst_anomaly.name}: {len(jpg_files)} images, {len(png_files)} masks")


def process_clean(src_imgs: Path, dst_clean: Path):
    dst_clean.mkdir(parents=True, exist_ok=True)
    jpg_files = [f for f in sorted(src_imgs.glob("*.jpg")) if f.stem in KEEP_CLEAN]
    for jpg in jpg_files:
        shutil.copy2(jpg, dst_clean / jpg.name)
    print(f"  clean_image: {len(jpg_files)} images")


def write_defect_spec(output_dir: Path):
    spec_path = output_dir / "defect_spec.jsonl"
    with spec_path.open("w") as f:
        for entry in DEFECT_SPEC:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {spec_path}")


def _run(repo_root: Path, metal_dir: Path, output_dir: Path):
    print(f"\nRepo extracted to: {repo_root}")

    for defect_type in DEFECT_TYPES:
        src = repo_root / defect_type / "Imgs"
        if not src.exists():
            print(f"WARNING: {src} not found, skipping {defect_type}.")
            continue
        print(f"\nProcessing {defect_type} ...")
        process_defect_type(
            src,
            metal_dir / "anomaly_image" / defect_type,
            metal_dir / "mask" / defect_type,
            KEEP_ANOMALY[defect_type],
        )

    print(f"\nProcessing {CLEAN_TYPE} (clean images) ...")
    src_clean = repo_root / CLEAN_TYPE / "Imgs"
    if src_clean.exists():
        process_clean(src_clean, metal_dir / "clean_image")
    else:
        print(f"WARNING: {src_clean} not found.")

    write_defect_spec(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare the UC2 Magnetic Tile dataset for AnomalyGen."
    )
    parser.add_argument("output_dir", help="Destination directory for the prepared dataset")
    parser.add_argument(
        "--keep-download",
        metavar="DIR",
        help="Keep the raw downloaded files in this directory instead of a temp dir",
    )
    args = parser.parse_args()

    # Disclaimer.
    print("Notice: The user is responsible for checking if the dataset license is fit for the intended purpose.")

    output_dir = Path(args.output_dir)
    metal_dir = output_dir / "metal_surface"

    if args.keep_download:
        download_dir = Path(args.keep_download)
        download_dir.mkdir(parents=True, exist_ok=True)
        repo_root = download_and_extract(download_dir)
        _run(repo_root, metal_dir, output_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = download_and_extract(Path(tmp))
            _run(repo_root, metal_dir, output_dir)

    print(f"\nDone. Dataset ready at: {output_dir}")


if __name__ == "__main__":
    main()
