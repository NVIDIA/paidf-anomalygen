# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSONL-driven dataset for SDG inference, plus the sharded validation-dataloader factory.

Ported from cosmos-anomalygen ``AnomalyInpaintDataset``: read a JSONL of testcases, assign a
preset-aware ``index``, fill per-sample generation defaults, and sort ascending by mask
instance count for more efficient iterative generation. Also hosts ``get_inpaint_val_dataloader``
(the ``dataloader_val`` LazyCall target used by the texture experiment config) and its private
collate / batch-index helpers. Public surface: ``InpaintInferenceDataset`` and
``get_inpaint_val_dataloader``; everything else is module-private (``_``-prefixed).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from cosmos_framework.utils import distributed, log
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from anomalygen.configs.texture.constants import (
    DEFAULT_CROP_RATIO,
    DEFAULT_GUIDANCE,
    DEFAULT_MAX_INSTANCES,
    DEFAULT_NUM_STEPS,
)

# Noise seed = base_seed + index * SEED_RECORD_STRIDE + output_n * SEED_OUTPUT_STRIDE + instance_j.
# The strides stop the three offsets colliding (testcase 0's 2nd instance vs testcase 1's 1st would
# otherwise share a noise tensor). Room for 64 instances per output, 16 outputs per testcase — past
# ``num_generated_images: 16`` the next testcase's seeds get reused.
SEED_RECORD_STRIDE = 1024
SEED_OUTPUT_STRIDE = 64


def _count_mask_instances(pil_mask: Image.Image) -> int:
    """Connected-component count of a mask, background excluded."""
    arr = np.array(pil_mask.convert("L"))
    num_labels = cv2.connectedComponentsWithStats(arr, connectivity=8)[0]
    return num_labels - 1  # exclude background


def _sort_records_by_instance_num(records: List[dict], resolve_mask_path: Callable[[str], str]) -> List[dict]:
    """Return ``records`` reordered ascending by mask instance count (stable; ties keep input order).

    ``resolve_mask_path`` maps a record's ``mask_filename`` to a loadable path.
    """
    num_instances = []
    for rec in records:
        mask = Image.open(resolve_mask_path(rec["mask_filename"])).convert("L")
        num_instances.append(_count_mask_instances(mask))

    counts = Counter(num_instances)
    log.info("Instance count distribution across testcases:")
    for k in sorted(counts):
        log.info(f"  {k} instance(s): {counts[k]} testcase(s)")

    paired = sorted(zip(num_instances, records), key=lambda pair: pair[0])
    return [rec for _, rec in paired]


class InpaintInferenceDataset(Dataset):
    """Reads a JSONL of inpaint testcases, fills defaults, and yields one (record, image, mask) per item."""

    def __init__(
        self,
        input_data_path: str,
        default_guidance: float = DEFAULT_GUIDANCE,
        default_num_steps: int = DEFAULT_NUM_STEPS,
        default_max_instances: int = DEFAULT_MAX_INSTANCES,
        base_dir: Optional[str] = None,
        base_seed: int = 1,
    ) -> None:
        super().__init__()
        # Relative image/mask paths in the JSONL resolve against ``base_dir`` (the repo root) when
        # set, else the current working directory. Training instantiates the val loader under a
        # chdir to the framework checkout, so ``base_dir`` keeps repo-root-relative testcase paths
        # correct; the SDG CLI leaves it None (cwd-relative, unchanged).
        self._base_dir = base_dir
        log.info(f"Loading generation settings from JSONL file: {input_data_path}")
        with open(input_data_path, "r") as f:
            self.input_data = [json.loads(line) for line in f if line.strip()]
        if not self.input_data:
            raise ValueError(f"No testcases found in {input_data_path}.")

        self._raise_on_duplicates(input_data_path)

        # Sample index: honor a preset if present, else default to line position.
        for i, rec in enumerate(self.input_data):
            rec.setdefault("index", i)

        for rec in self.input_data:
            rec.setdefault("guidance", default_guidance)
            rec.setdefault("num_steps", default_num_steps)
            # Own noise per testcase, reproducible run to run since ``index`` is stable. A flat
            # default would give every sample identical noise; an explicit JSONL "seed" wins.
            rec.setdefault("seed", int(base_seed) + int(rec["index"]) * SEED_RECORD_STRIDE)
            rec.setdefault("num_generated_images", 1)
            rec.setdefault("iteration_generation_max_instance", default_max_instances)
            rec.setdefault("crop_and_paste", True)
            rec.setdefault("crop_ratio", DEFAULT_CROP_RATIO)
            rec.setdefault("poisson_blend", False)

        self.input_data = _sort_records_by_instance_num(self.input_data, self._resolve)

    def _raise_on_duplicates(self, input_data_path: str) -> None:
        encoded = [json.dumps(rec, sort_keys=True) for rec in self.input_data]
        seen: set[str] = set()
        dupes: set[str] = set()
        for s in encoded:
            if s in seen:
                dupes.add(s)
            seen.add(s)
        if dupes:
            raise ValueError(
                f"Found duplicated samples in {input_data_path}. Duplicated samples:\n" + "\n".join(sorted(dupes))
            )

    def _resolve(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        base = self._base_dir if self._base_dir is not None else os.getcwd()
        return os.path.abspath(os.path.join(base, path))

    def __len__(self) -> int:
        return len(self.input_data)

    def __getitem__(self, idx: int) -> dict:
        rec = dict(self.input_data[idx])
        rec["image"] = Image.open(self._resolve(rec["image_filename"])).convert("RGB")
        rec["mask"] = Image.open(self._resolve(rec["mask_filename"])).convert("L")
        return rec

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """Single-sample collate (the SDG loop runs ``batch_size=1``)."""
        if len(batch) != 1:
            raise ValueError(f"InpaintInferenceDataset expects batch_size=1, got {len(batch)}.")
        return batch[0]


def _val_collate(batch: List[dict]) -> dict:
    """Collate per-testcase dicts into a dict of per-key lists, with ``index`` as a ``[B]`` tensor.

    Values stay as per-key lists (no tensor stacking) so PIL images / strings / scalars pass
    through. ``index`` is additionally emitted as a tensor because the framework's per-step
    callbacks infer the batch size from a tensor in the batch (``misc.get_data_batch_size``), and
    our batch is otherwise all lists of str/float/PIL. ``_generate_validation_batch`` coerces indices
    via ``int(x)``, so the tensor works identically to the list form.
    """
    out = {k: [rec[k] for rec in batch] for k in batch[0]}
    out["index"] = torch.tensor([int(rec["index"]) for rec in batch], dtype=torch.long)
    return out


def _build_val_batch_indices(
    records: List[dict], world_size: int, rank: int, batch_size: int, default_shift: float
) -> List[List[int]]:
    """Per-rank list of index-lists, each batch homogeneous in ``(guidance, shift)``.

    Records are grouped by ``(guidance, shift)``; each group is padded (by repeating from the
    front) to a multiple of ``world_size`` and split contiguously across ranks, so every rank
    yields the same number of same-sized batches (FSDP lockstep). Padding repeats are removed
    later by the ``index`` dedup in ``on_validation_end``.
    """
    groups: Dict[tuple, List[int]] = {}
    for i, rec in enumerate(records):
        key = (float(rec.get("guidance", 6.0)), float(rec.get("shift", default_shift)))
        groups.setdefault(key, []).append(i)

    batch_indices: List[List[int]] = []
    for key in sorted(groups):  # deterministic, identical on every rank
        idxs = groups[key]
        target = ((len(idxs) + world_size - 1) // world_size) * world_size
        padded = (idxs * ((target // len(idxs)) + 1))[:target]
        per = target // world_size
        mine = padded[rank * per : (rank + 1) * per]
        for c in range(0, len(mine), batch_size):
            batch_indices.append(mine[c : c + batch_size])
    return batch_indices


def get_inpaint_val_dataloader(
    input_data_path: str,
    val_batch_size: int,
    shift: float,
    default_guidance: float = DEFAULT_GUIDANCE,
    default_num_steps: int = DEFAULT_NUM_STEPS,
    default_max_instances: int = DEFAULT_MAX_INSTANCES,
    num_workers: int = 2,
    base_dir: Optional[str] = None,
    base_seed: int = 1,
) -> DataLoader:
    """LazyCall target for ``dataloader_val``: a sharded, param-homogeneous validation loader.

    Reads world_size/rank from the framework's distributed utils at instantiation time (so it
    also works single-process), builds this rank's batch-index lists, and returns a plain
    ``DataLoader`` whose ``batch_sampler`` is those lists and ``collate_fn`` keeps values as per-key lists.
    ``base_dir`` resolves repo-root-relative testcase paths (training instantiates this under a
    chdir to the framework checkout, so cwd-relative resolution would be wrong).
    """
    dataset = InpaintInferenceDataset(
        input_data_path,
        default_guidance=default_guidance,
        default_num_steps=default_num_steps,
        default_max_instances=default_max_instances,
        base_dir=base_dir,
        base_seed=base_seed,
    )
    # Ensure a uniform ``shift`` key so _val_collate never KeyErrors and grouping can key on it.
    for rec in dataset.input_data:
        rec.setdefault("shift", float(shift))

    world_size = distributed.get_world_size()
    rank = distributed.get_rank()
    batch_indices = _build_val_batch_indices(dataset.input_data, world_size, rank, int(val_batch_size), float(shift))
    return DataLoader(dataset, batch_sampler=batch_indices, collate_fn=_val_collate, num_workers=int(num_workers))
