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
Prepare UC1 dataset (PCB) for AnomalyGen training.

Source: Hugging Face dataset repo ``nvidia/Cosmos-AnomalyGen-PCB-Dataset``.

Unlike UC2 (curated subset from a public GitHub repo) and UC3 (extracted from a
Roboflow zip), the UC1 bundle is shipped on Hugging Face in the exact layout
the downstream pipeline expects, so this script is a thin wrapper around
``huggingface_hub.snapshot_download`` plus a defensive single-wrapper-dir
flatten step in case the snapshot lands inside one outer directory.

Output layout (whatever the HF snapshot ships — typically):
  <output_dir>/
    PCB/
      anomaly_image/<TYPE>/
      mask/<TYPE>/
      clean_image/
    defect_spec.jsonl

Usage:
  Step 1 — Authenticate with Hugging Face. Either:
    - export HF_TOKEN=<your-token>          (one-shot, env-only), OR
    - run `hf auth login` once              (persists to ~/.cache/huggingface)
    Your token must have read access to nvidia/Cosmos-AnomalyGen-PCB-Dataset.

  Step 2 — Run the script (from repo root, or inside the container):
    python3 -m scripts.utilities.prepare_dataset_uc1 <output_dir>

  Optional:
    --revision main          Pin a different git revision / tag / commit
    --keep-download DIR      Stage HF files in DIR instead of a temp dir
    --dry-run                Print the resolved HF target and exit
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

HF_REPO_ID = "nvidia/Cosmos-AnomalyGen-PCB-Dataset"
DEFAULT_REVISION = "main"


def preflight() -> None:
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
            f"  The token needs read access to {HF_REPO_ID}."
        )


def flatten_single_wrapper(out: Path) -> None:
    """If the HF snapshot landed inside a single wrapper dir, lift its contents
    up one level so downstream code sees a predictable shape."""
    children = [p for p in out.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(children) != 1:
        print(f"[FLATTEN] skipped (found {len(children)} top-level dirs)")
        return
    wrapper = children[0]
    stage = out / f".flatten.{os.getpid()}"
    wrapper.rename(stage)
    for item in stage.iterdir():
        shutil.move(str(item), str(out / item.name))
    stage.rmdir()
    print(f"[FLATTEN] removed wrapper: {wrapper.name}")


def hf_download(repo_id: str, revision: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[HF] snapshot_download repo_id={repo_id} revision={revision} local_dir={dest}")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(dest),
            token=os.environ.get("HF_TOKEN"),
        )
    except RepositoryNotFoundError:
        sys.exit(
            f"ERROR: Hugging Face repo {repo_id} not found or your token "
            "lacks read access."
        )
    except HfHubHTTPError as exc:
        sys.exit(f"ERROR: Hugging Face download failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and stage the UC1 PCB dataset from Hugging Face."
    )
    parser.add_argument("output_dir", help="Destination directory for the prepared dataset")
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"Hugging Face git revision / tag / commit (default: {DEFAULT_REVISION})",
    )
    parser.add_argument(
        "--keep-download",
        metavar="DIR",
        help="Stage the HF snapshot in DIR instead of a temp dir (kept after the run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved HF target and exit without downloading",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.dry_run:
        print(
            f"[DRY-RUN] hf download {HF_REPO_ID} --repo-type dataset "
            f"--revision {args.revision} --local-dir {output_dir}"
        )
        return

    preflight()

    def _run(stage: Path) -> None:
        hf_download(HF_REPO_ID, args.revision, stage)
        flatten_single_wrapper(stage)
        if stage != output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            for item in stage.iterdir():
                if item.name.startswith(".cache"):
                    continue
                shutil.move(str(item), str(output_dir / item.name))
        file_count = sum(1 for _ in output_dir.rglob("*") if _.is_file())
        if file_count == 0:
            sys.exit("ERROR: 0 files downloaded.")
        print(f"\nDone. {file_count} files staged at: {output_dir}")

    if args.keep_download:
        stage = Path(args.keep_download)
        stage.mkdir(parents=True, exist_ok=True)
        _run(stage)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            _run(Path(tmp))


if __name__ == "__main__":
    main()
