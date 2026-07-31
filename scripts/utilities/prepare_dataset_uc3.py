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
Prepare UC3 dataset (Mobile Phone Screen Defect) for AnomalyGen training.

Sources:
  - Anomaly + clean images: Roboflow COCO export (single zip).
      Download from: https://universe.roboflow.com/vu-thi-thu-huyen/mobile-screen
      The zip contains one split directory (e.g. train/) with images and
      _annotations.coco.json.
  - Masks + defect_spec.jsonl: Hugging Face dataset repo
      nvidia/Cosmos-AnomalyGen-Glass-Masks (fetched with --masks-from-hf).

Image filename mapping (Roboflow hash removed):
  Oil_0017_jpg.rf.<hash>.jpg  ->  anomaly_image/oil/Oil_0017.png
  Scr_0016_jpg.rf.<hash>.jpg  ->  anomaly_image/scratch/Scr_0016.png
  Sta_0067_jpg.rf.<hash>.jpg  ->  anomaly_image/stain/Sta_0067.png
  0001_png.rf.<hash>.png      ->  clean_image/0001.png

Output layout (after both --zip and --masks-from-hf):
  <output_dir>/
    Phone/
      anomaly_image/
        oil/        Oil_XXXX.png    (from Roboflow)
        scratch/    Scr_XXXX.png    (from Roboflow)
        stain/      Sta_XXXX.png    (from Roboflow)
      clean_image/  XXXX.png        (from Roboflow)
      mask/
        oil/                        (from HF)
        scratch/                    (from HF)
        stain/                      (from HF)
    defect_spec.jsonl               (from HF)

Usage:
  Step 1 — Download the dataset zip from Roboflow (browser required):
    - Follow the instructions in datasets/UC3_dataset_download_instructions.pdf to download the zip file.

  Step 2 — Authenticate with Hugging Face (only if using --masks-from-hf):
    - export HF_TOKEN=<your-token>, OR run `hf auth login` once.

  Step 3 — Run the script (from repo root, or inside the container):
    python3 -m scripts.utilities.prepare_dataset_uc3 <output_dir> \\
        --zip <path/to/downloaded.zip> --masks-from-hf

  Optional — preview what will be extracted without writing files:
    python3 -m scripts.utilities.prepare_dataset_uc3 <output_dir> \\
        --zip <path/to/downloaded.zip> --dry-run

  To pull just the masks/defect_spec (skipping the Roboflow zip):
    python3 -m scripts.utilities.prepare_dataset_uc3 <output_dir> --masks-from-hf
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
import zipfile

HF_MASKS_REPO_ID = "nvidia/Cosmos-AnomalyGen-Glass-Masks"
HF_MASKS_DEFAULT_REVISION = "main"

try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Curated subset: exactly the files shipped in the reference UC3_data dataset.
KEEP_ANOMALY = {
    "oil":     {"Oil_0001", "Oil_0021", "Oil_0041", "Oil_0061", "Oil_0081"},
    "scratch": {"Scr_0001", "Scr_0021", "Scr_0041", "Scr_0061", "Scr_0081"},
    "stain":   {"Sta_0001", "Sta_0021", "Sta_0041", "Sta_0061", "Sta_0081"},
}


def original_name(roboflow_fname: str) -> str:
    """Strip Roboflow hash suffix and recover original filename.

    Oil_0017_jpg.rf.<hash>.jpg  ->  Oil_0017.jpg
    0001_png.rf.<hash>.png      ->  0001.png
    """
    m = re.match(r"^(.+?)_(jpg|png)\.rf\.[A-Za-z0-9]+\.(?:jpg|png)$", roboflow_fname)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return roboflow_fname


def classify(orig: str):
    """Return (defect_type, output_stem) or (None, output_stem) for clean images."""
    if orig.startswith("Oil_"):
        return "oil", Path(orig).stem
    if orig.startswith("Scr_"):
        return "scratch", Path(orig).stem
    if orig.startswith("Sta_"):
        return "stain", Path(orig).stem
    return None, Path(orig).stem


def collect_entries(zip_path: Path):
    """Return list of (zip_entry, defect_type_or_None, output_stem)."""
    entries = []

    seen = set()  # (dtype, stem) already collected — dedup across splits

    with zipfile.ZipFile(zip_path) as zf:
        all_names = set(zf.namelist())
        json_entries = sorted(
            n for n in all_names if n.endswith("/_annotations.coco.json"))
        if not json_entries:
            sys.exit("ERROR: no _annotations.coco.json found in zip.")

        # Scan every split, not just the first. A Roboflow export can carry the
        # curated stems spread across train/valid/test; picking only json_entries[0]
        # silently dropped stems living in another split.
        for json_entry in json_entries:
            split = json_entry.rsplit("/_annotations.coco.json", 1)[0]
            print(f"Found split: {split}")

            with zf.open(json_entry) as f:
                coco = json.load(f)

            for img in coco["images"]:
                fname = img["file_name"]
                zip_entry = f"{split}/{fname}"
                if zip_entry not in all_names:
                    print(f"WARNING: {zip_entry} not in zip, skipping.")
                    continue

                orig = original_name(fname)
                dtype, stem = classify(orig)

                if dtype is not None and stem not in KEEP_ANOMALY.get(dtype, set()):
                    continue

                # A stem can appear in more than one split; keep the first only.
                if (dtype, stem) in seen:
                    continue

                entries.append((zip_entry, dtype, stem))
                seen.add((dtype, stem))

    # Completeness: every curated anomaly stem should have been found in some
    # split. Report the shortfall to the caller (and warn) rather than aborting
    # here, so --dry-run can still print the summary and show what is missing.
    expected = {(dtype, stem)
                for dtype, stems in KEEP_ANOMALY.items() for stem in stems}
    missing = sorted(expected - seen)
    if missing:
        listing = ", ".join(f"{d}/{s}" for d, s in missing)
        print(f"WARNING: {len(missing)} curated KEEP_ANOMALY stem(s) not found "
              f"in any split: {listing}")

    return entries, missing


def print_summary(entries):
    from collections import Counter
    counts = Counter(dtype if dtype else "clean" for _, dtype, _ in entries)
    print(f"\nTotal images to extract: {sum(counts.values())}")
    for k in ["oil", "scratch", "stain", "clean"]:
        print(f"  {k:10s}: {counts.get(k, 0)}")


def extract(zip_path: Path, entries, phone_dir: Path):
    for dtype in ["oil", "scratch", "stain"]:
        (phone_dir / "anomaly_image" / dtype).mkdir(parents=True, exist_ok=True)
    (phone_dir / "clean_image").mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        for zip_entry, dtype, stem in entries:
            data = zf.read(zip_entry)
            out_fname = f"{stem}.png"

            if dtype is None:
                dst = phone_dir / "clean_image" / out_fname
            else:
                dst = phone_dir / "anomaly_image" / dtype / out_fname

            if HAS_PIL:
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img.save(dst)
            else:
                # Fallback: write raw bytes (stays as jpg-encoded, .png extension)
                dst.write_bytes(data)

    print(f"\nExtracted {len(entries)} images to {phone_dir}")


def hf_masks_preflight() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: `huggingface_hub` is not installed in the active env.\n"
            "  - In the container: it ships with the cosmos-predict2 conda env.\n"
            "  - On the host: `pip install 'huggingface_hub>=1.0'`."
        )
    if not os.environ.get("HF_TOKEN") and not (Path.home() / ".cache/huggingface/token").is_file():
        sys.exit(
            "ERROR: Hugging Face is not authenticated.\n"
            "  Either:\n"
            "    - export HF_TOKEN=<your-token>, OR\n"
            "    - run `hf auth login` once to persist your token.\n"
            f"  The token needs read access to {HF_MASKS_REPO_ID}."
        )


def fetch_masks_from_hf(output_dir: Path, revision: str) -> None:
    """Snapshot the Glass-Masks HF dataset repo directly into output_dir.

    The HF repo is laid out so its files land at the paths the downstream
    pipeline expects (Phone/mask/<type>/..., defect_spec.jsonl).
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[HF] snapshot_download repo_id={HF_MASKS_REPO_ID} "
        f"revision={revision} local_dir={output_dir}"
    )
    try:
        snapshot_download(
            repo_id=HF_MASKS_REPO_ID,
            repo_type="dataset",
            revision=revision,
            local_dir=str(output_dir),
            token=os.environ.get("HF_TOKEN"),
        )
    except RepositoryNotFoundError:
        sys.exit(
            f"ERROR: Hugging Face repo {HF_MASKS_REPO_ID} not found or your "
            "token lacks read access."
        )
    except HfHubHTTPError as exc:
        sys.exit(f"ERROR: Hugging Face download failed: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare UC3 Mobile Phone Screen dataset for AnomalyGen "
                    "(Roboflow images + HF masks).",
    )
    parser.add_argument("output_dir", help="Destination directory for the prepared dataset")
    parser.add_argument("--zip", metavar="UC3_dataset.zip",
                        help="Path to the downloaded Roboflow zip (required for image extraction)")
    parser.add_argument("--masks-from-hf", action="store_true",
                        help=f"Also fetch masks + defect_spec.jsonl from {HF_MASKS_REPO_ID}")
    parser.add_argument("--masks-revision", default=HF_MASKS_DEFAULT_REVISION,
                        help=f"HF git revision for the masks repo (default: {HF_MASKS_DEFAULT_REVISION})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be extracted/downloaded without writing files")
    args = parser.parse_args()

    # Disclaimer.
    print("Notice: The user is responsible for checking if the dataset license is fit for the intended purpose.")

    if not args.zip and not args.masks_from_hf:
        sys.exit(
            "ERROR: nothing to do. Pass --zip <path> to extract Roboflow "
            "images and/or --masks-from-hf to fetch masks + defect_spec."
        )

    output_dir = Path(args.output_dir)

    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            sys.exit(f"ERROR: {zip_path} not found")

        print(f"Scanning {zip_path.name} ...")
        # collect_entries already warns (loudly) about any missing curated
        # stems; we still extract whatever IS present rather than aborting, so
        # a maintainer-edited export that drops one stem yields a usable
        # partial dataset instead of nothing.
        entries, _missing = collect_entries(zip_path)
        print_summary(entries)

        if not args.dry_run:
            phone_dir = output_dir / "Phone"
            extract(zip_path, entries, phone_dir)

    if args.masks_from_hf:
        if args.dry_run:
            print(
                f"\n[DRY-RUN] hf download {HF_MASKS_REPO_ID} --repo-type dataset "
                f"--revision {args.masks_revision} --local-dir {output_dir}"
            )
        else:
            hf_masks_preflight()
            fetch_masks_from_hf(output_dir, args.masks_revision)

    if args.dry_run:
        print("\n--dry-run: stopping here.")
        return

    print(f"\nDone. Dataset ready at: {output_dir}")


if __name__ == "__main__":
    main()
