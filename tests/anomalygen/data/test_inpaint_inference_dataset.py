# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SDG inference dataset + the validation dataloader helpers.

Covers the JSONL-driven ``InpaintInferenceDataset`` (public API used by generate.py) and the
validation-loader helpers it hosts: ``_val_collate``, ``_build_val_batch_indices``, and the public
``get_inpaint_val_dataloader`` factory.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from anomalygen.configs.texture.constants import (
    DEFAULT_CROP_RATIO,
    DEFAULT_GUIDANCE,
    DEFAULT_MAX_INSTANCES,
    DEFAULT_NUM_STEPS,
)
from anomalygen.data.inpaint_inference_dataset import (
    SEED_OUTPUT_STRIDE,
    SEED_RECORD_STRIDE,
    InpaintInferenceDataset,
    _build_val_batch_indices,
    _count_mask_instances,
    _sort_records_by_instance_num,
    _val_collate,
    get_inpaint_val_dataloader,
)


def _save_image(path, size=(32, 32)):
    Image.fromarray(np.full((size[1], size[0], 3), 128, np.uint8), mode="RGB").save(path)


def _save_mask(path, blobs, size=32):
    """Save an L-mode mask with a filled 5x5 square at each (row, col) top-left in ``blobs``."""
    arr = np.zeros((size, size), dtype=np.uint8)
    for r, c in blobs:
        arr[r : r + 5, c : c + 5] = 255
    Image.fromarray(arr, mode="L").save(path)


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records))


# --- module-level helpers ------------------------------------------------------------------------


def test_count_mask_instances_counts_components():
    mask = Image.fromarray(np.zeros((40, 40), np.uint8), mode="L")
    arr = np.array(mask)
    arr[2:7, 2:7] = 255
    arr[30:35, 30:35] = 255  # second, well-separated blob
    two = Image.fromarray(arr, mode="L")
    assert _count_mask_instances(two) == 2
    assert _count_mask_instances(mask) == 0  # empty -> no foreground components


def test_count_mask_instances_counts_nonbinary_components():
    arr = np.zeros((40, 40), np.uint8)
    arr[2:7, 2:7] = 255
    arr[20:25, 20:25] = 128  # a mid-grey (non-binary) blob still counts as foreground
    mask = Image.fromarray(arr, mode="L")
    # Any non-zero pixel is foreground for connected-components, so both blobs count.
    assert _count_mask_instances(mask) == 2


def test_sort_records_by_instance_num(tmp_path):
    one = tmp_path / "one.png"
    two = tmp_path / "two.png"
    _save_mask(one, blobs=[(2, 2)])
    _save_mask(two, blobs=[(2, 2), (20, 20)])

    records = [{"mask_filename": str(two)}, {"mask_filename": str(one)}]
    ordered = _sort_records_by_instance_num(records, resolve_mask_path=lambda p: p)
    # Ascending by instance count: the single-blob record comes first.
    assert ordered[0]["mask_filename"] == str(one)
    assert ordered[1]["mask_filename"] == str(two)


# --- InpaintInferenceDataset ---------------------------------------------------------------------


def _make_dataset(tmp_path):
    img = tmp_path / "img.png"
    m_one = tmp_path / "m_one.png"
    m_two = tmp_path / "m_two.png"
    _save_image(img)
    _save_mask(m_one, blobs=[(2, 2)])
    _save_mask(m_two, blobs=[(2, 2), (20, 20)])

    jsonl = tmp_path / "cases.jsonl"
    # Deliberately list the two-instance case first so we can observe the ascending sort.
    _write_jsonl(
        jsonl,
        [
            {"image_filename": str(img), "mask_filename": str(m_two), "anomaly_type": "t+d"},
            {"image_filename": str(img), "mask_filename": str(m_one), "anomaly_type": "t+d"},
        ],
    )
    return jsonl, m_one, m_two


def test_dataset_fills_defaults_and_sorts(tmp_path):
    jsonl, m_one, _ = _make_dataset(tmp_path)
    ds = InpaintInferenceDataset(
        jsonl.as_posix(),
        default_guidance=DEFAULT_GUIDANCE,
        default_num_steps=DEFAULT_NUM_STEPS,
        default_max_instances=DEFAULT_MAX_INSTANCES,
    )

    assert len(ds) == 2
    # Sorted ascending by instance count: single-blob mask first.
    assert ds.input_data[0]["mask_filename"] == str(m_one)

    rec = ds.input_data[0]
    assert rec["guidance"] == DEFAULT_GUIDANCE
    assert rec["num_steps"] == DEFAULT_NUM_STEPS
    # Seed is derived per testcase (base_seed + index * stride), not flat, so every testcase gets
    # its own noise. ``rec`` is the single-blob mask, which was line 1 of the JSONL before the sort.
    assert rec["index"] == 1
    assert rec["seed"] == 1 + 1 * SEED_RECORD_STRIDE
    assert rec["num_generated_images"] == 1
    assert rec["iteration_generation_max_instance"] == DEFAULT_MAX_INSTANCES
    assert rec["crop_and_paste"] is True
    assert rec["crop_ratio"] == DEFAULT_CROP_RATIO
    assert rec["poisson_blend"] is False
    # index defaults to original line position (assigned before the sort).
    assert set(r["index"] for r in ds.input_data) == {0, 1}


def test_dataset_getitem_loads_pil_pair(tmp_path):
    jsonl, _, _ = _make_dataset(tmp_path)
    ds = InpaintInferenceDataset(jsonl.as_posix())
    item = ds[0]
    assert isinstance(item["image"], Image.Image) and item["image"].mode == "RGB"
    assert isinstance(item["mask"], Image.Image) and item["mask"].mode == "L"


def test_dataset_collate_fn_single_sample(tmp_path):
    jsonl, _, _ = _make_dataset(tmp_path)
    ds = InpaintInferenceDataset(jsonl.as_posix())
    sample = ds[0]
    assert InpaintInferenceDataset.collate_fn([sample]) is sample
    with pytest.raises(ValueError):
        InpaintInferenceDataset.collate_fn([sample, sample])


def test_dataset_rejects_duplicates(tmp_path):
    img = tmp_path / "img.png"
    mask = tmp_path / "m.png"
    _save_image(img)
    _save_mask(mask, blobs=[(2, 2)])
    rec = {"image_filename": str(img), "mask_filename": str(mask), "anomaly_type": "t+d"}
    jsonl = tmp_path / "dup.jsonl"
    _write_jsonl(jsonl, [rec, dict(rec)])  # identical records
    with pytest.raises(ValueError):
        InpaintInferenceDataset(jsonl.as_posix())


def test_dataset_empty_jsonl_raises(tmp_path):
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("")
    with pytest.raises(ValueError):
        InpaintInferenceDataset(jsonl.as_posix())


def test_dataset_base_dir_resolves_relative_testcase_paths(tmp_path):
    # Repo-root-relative image/mask paths must resolve against base_dir, not cwd (training
    # instantiates the loader under a chdir to the framework checkout). The files exist ONLY under
    # base_dir, so cwd-relative resolution (cwd = repo root during pytest) would FileNotFoundError.
    (tmp_path / "imgs").mkdir()
    _save_image(tmp_path / "imgs" / "img.png")
    _save_mask(tmp_path / "imgs" / "msk.png", blobs=[(2, 2)])
    jsonl = tmp_path / "tc.jsonl"
    _write_jsonl(
        jsonl,
        [{"image_filename": "imgs/img.png", "mask_filename": "imgs/msk.png", "anomaly_type": "t+d"}],
    )
    ds = InpaintInferenceDataset(jsonl.as_posix(), base_dir=str(tmp_path))
    item = ds[0]
    assert item["image"].size == (32, 32) and item["mask"].mode == "L"


# --- noise-seed envelope -------------------------------------------------------------------------
#
# A generation's seed is assembled from three offsets applied in three different places:
#   dataset:   base_seed + index * SEED_RECORD_STRIDE      (per testcase)
#   generate:  + output_n * SEED_OUTPUT_STRIDE             (per requested output image)
#   iterative: + instance_j                                (per defect instance, batched depth)
# The strides give each testcase room for SEED_RECORD_STRIDE // SEED_OUTPUT_STRIDE outputs and each
# output room for SEED_OUTPUT_STRIDE instances. Those limits are the whole reason the strides exist,
# so they are pinned here.

_MAX_OUTPUTS_PER_TESTCASE = SEED_RECORD_STRIDE // SEED_OUTPUT_STRIDE  # 16
_MAX_INSTANCES_PER_OUTPUT = SEED_OUTPUT_STRIDE  # 64


def _seed_for(base_seed, index, output_n, instance_j):
    """Seed a given (testcase, output, instance) triple generates with, per the layout above."""
    return base_seed + index * SEED_RECORD_STRIDE + output_n * SEED_OUTPUT_STRIDE + instance_j


def test_dataset_base_seed_propagates_into_derived_seeds(tmp_path):
    # --base_seed is the run-level re-roll knob: same value -> bit-identical noise, bump it for a
    # fresh draw. It must reach every testcase's derived seed.
    jsonl, _, _ = _make_dataset(tmp_path)
    ds = InpaintInferenceDataset(jsonl.as_posix(), base_seed=9000)
    by_index = {r["index"]: r["seed"] for r in ds.input_data}
    assert by_index == {0: 9000, 1: 9000 + SEED_RECORD_STRIDE}


def test_dataset_explicit_seed_wins_over_derived(tmp_path):
    # A testcase that pins its own seed is reproducing a specific generation; the derivation must
    # not overwrite it (setdefault), and it must not leak into its neighbours.
    img = tmp_path / "img.png"
    m_one = tmp_path / "m_one.png"
    m_two = tmp_path / "m_two.png"
    _save_image(img)
    _save_mask(m_one, blobs=[(2, 2)])
    _save_mask(m_two, blobs=[(2, 2), (20, 20)])
    jsonl = tmp_path / "cases.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"image_filename": str(img), "mask_filename": str(m_one), "anomaly_type": "t+d", "seed": 4242},
            {"image_filename": str(img), "mask_filename": str(m_two), "anomaly_type": "t+d"},
        ],
    )
    ds = InpaintInferenceDataset(jsonl.as_posix(), base_seed=7)
    by_index = {r["index"]: r["seed"] for r in ds.input_data}
    assert by_index[0] == 4242  # explicit
    assert by_index[1] == 7 + SEED_RECORD_STRIDE  # derived, unaffected


def test_seeds_are_unique_within_the_documented_envelope():
    # Inside the envelope no two (testcase, output, instance) triples may share a noise draw —
    # that is exactly what the two strides buy.
    seeds = [
        _seed_for(1, index, n, j)
        for index in range(32)
        for n in range(_MAX_OUTPUTS_PER_TESTCASE)
        for j in range(_MAX_INSTANCES_PER_OUTPUT)
    ]
    assert len(set(seeds)) == len(seeds)


def test_seed_collides_past_the_outputs_per_testcase_limit():
    # CHARACTERISATION, not an endorsement: request more than _MAX_OUTPUTS_PER_TESTCASE images and
    # a testcase's extra outputs reuse the NEXT testcase's seeds (both 1025 below) — silently
    # duplicated noise across two different testcases. Guarded in generate.py; if this test starts
    # failing because the strides moved, the guard's limits must move with them.
    assert _seed_for(1, 0, _MAX_OUTPUTS_PER_TESTCASE, 0) == _seed_for(1, 1, 0, 0)


def test_seed_collides_past_the_instances_per_output_limit():
    # Same story one level down: past _MAX_INSTANCES_PER_OUTPUT instances, an output's later
    # instances reuse the next output's seeds (both 65).
    assert _seed_for(1, 0, 0, _MAX_INSTANCES_PER_OUTPUT) == _seed_for(1, 0, 1, 0)


def test_generate_rejects_records_outside_the_seed_envelope():
    # The two counts come straight from user JSONL, so the collisions characterised above are
    # reachable by configuration alone; generate.py must refuse them up front rather than emit
    # duplicate noise. Imported lazily: importing generate.py runs the framework's init_script.
    from anomalygen.scripts.texture.generate import _validate_seed_envelope

    ok = [{"index": 0, "num_generated_images": _MAX_OUTPUTS_PER_TESTCASE, "iteration_generation_max_instance": 1}]
    assert _validate_seed_envelope(ok) is None  # at the limit is fine

    with pytest.raises(ValueError, match="num_generated_images"):
        _validate_seed_envelope([{"index": 3, "num_generated_images": _MAX_OUTPUTS_PER_TESTCASE + 1}])
    with pytest.raises(ValueError, match="iteration_generation_max_instance"):
        _validate_seed_envelope([{"index": 3, "iteration_generation_max_instance": _MAX_INSTANCES_PER_OUTPUT + 1}])


# --- _val_collate --------------------------------------------------------------------------------


def test_val_collate_index_is_size_inferable_tensor():
    # The framework infers validation batch size from a tensor in the batch; _val_collate must emit
    # ``index`` as a [B] tensor while other fields stay per-key lists.
    batch = [{"index": 3, "anomaly_type": "t+d"}, {"index": 7, "anomaly_type": "t+d"}]
    out = _val_collate(batch)
    assert isinstance(out["index"], torch.Tensor)
    assert out["index"].tolist() == [3, 7]
    assert out["anomaly_type"] == ["t+d", "t+d"]


# --- _build_val_batch_indices ------------------------------------------------------------------------


def _recs(specs):
    # specs: list of (guidance, shift-or-None); index = position.
    out = []
    for i, (g, s) in enumerate(specs):
        r = {"index": i, "guidance": g}
        if s is not None:
            r["shift"] = s
        out.append(r)
    return out


def test_plan_batches_are_guidance_shift_homogeneous():
    recs = _recs([(6.0, None), (6.0, None), (7.0, None), (7.0, None)])
    plan = _build_val_batch_indices(recs, world_size=1, rank=0, batch_size=4, default_shift=5.0)
    for batch in plan:
        keys = {(recs[i]["guidance"], recs[i].get("shift", 5.0)) for i in batch}
        assert len(keys) == 1  # one (guidance, shift) per batch


def test_plan_equal_batch_count_across_ranks_and_full_coverage():
    recs = _recs([(6.0, None)] * 15)
    world = 4
    plans = [
        _build_val_batch_indices(recs, world_size=world, rank=r, batch_size=4, default_shift=5.0) for r in range(world)
    ]
    counts = [len(p) for p in plans]
    assert len(set(counts)) == 1  # every rank runs the same number of batches (FSDP lockstep)
    seen = set()
    for p in plans:
        for batch in p:
            seen.update(batch)
    assert seen == set(range(15))  # union (post padding-dedup) covers every record


def test_plan_small_group_pads_to_world_multiple():
    recs = _recs([(6.0, None)])  # single record, 8 ranks
    plans = [_build_val_batch_indices(recs, world_size=8, rank=r, batch_size=4, default_shift=5.0) for r in range(8)]
    assert all(len(p) == 1 and p[0] == [0] for p in plans)  # each rank gets the (repeated) sole record


# --- get_inpaint_val_dataloader ------------------------------------------------------------------


def test_get_inpaint_val_dataloader_builds_batches(tmp_path):
    img = tmp_path / "img.png"
    msk = tmp_path / "msk.png"
    _save_image(img)
    _save_mask(msk, blobs=[(8, 8)])
    jsonl = tmp_path / "testcase.jsonl"
    # Distinct index per record so they aren't rejected as duplicates.
    _write_jsonl(
        jsonl,
        [
            {
                "index": i,
                "image_filename": str(img),
                "mask_filename": str(msk),
                "anomaly_type": "t+d",
                "guidance": DEFAULT_GUIDANCE,
                "num_steps": DEFAULT_NUM_STEPS,
                "iteration_generation_max_instance": 1,
            }
            for i in range(5)
        ],
    )
    dl = get_inpaint_val_dataloader(str(jsonl), val_batch_size=2, shift=5.0, num_workers=0)
    batches = list(dl)
    # 5 records, batch_size 2 -> 3 batches (2, 2, 1); collate yields dict-of-lists incl loaded PIL.
    assert sum(len(b["anomaly_type"]) for b in batches) == 5
    assert "image" in batches[0] and "index" in batches[0]
    assert all(len(b["anomaly_type"]) <= 2 for b in batches)
