# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the iterative inpainting orchestrator (public API used by generate.py).

``run_iterative_inpaint`` is driven with a fake ``model_inpaint`` closure, so the whole
crop -> generate -> paste-back loop (real crop_paste geometry) is exercised without a model.
"""

import numpy as np
import pytest
from PIL import Image

from anomalygen.configs.texture.constants import DEFAULT_CROP_RATIO, DEFAULT_MAX_INSTANCES
from anomalygen.inference.iterative import (
    run_iterative_inpaint,
    run_iterative_inpaint_batch,
    split_mask_into_instances,
)


def _mask(size=100, blobs=((40, 40),), side=15):
    arr = np.zeros((size, size), dtype=np.uint8)
    for r, c in blobs:
        arr[r : r + side, c : c + side] = 255
    return Image.fromarray(arr, mode="L")


def _gray(size=100, value=128):
    return Image.fromarray(np.full((size, size, 3), value, np.uint8), mode="RGB")


def _fg(pil_mask):
    return int((np.array(pil_mask) > 127).sum())


# --- split_mask_into_instances -------------------------------------------------------------------


def test_split_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        split_mask_into_instances(_mask(), max_k=0)


def test_split_k1_returns_mask_unchanged():
    m = _mask()
    assert split_mask_into_instances(m, max_k=1) == [m]


def test_split_empty_mask_returns_empty():
    empty = Image.fromarray(np.zeros((100, 100), np.uint8), mode="L")
    assert split_mask_into_instances(empty, max_k=5) == []


def test_split_components_below_k_one_mask_each():
    m = _mask(blobs=((10, 10), (70, 70)), side=10)
    instances = split_mask_into_instances(m, max_k=5)
    assert len(instances) == 2
    # Each returned mask carries exactly one blob; together they cover the original foreground.
    assert sum(_fg(i) for i in instances) == _fg(m)


def test_split_more_components_than_k_clusters_deterministically():
    m = _mask(blobs=((5, 5), (5, 80), (80, 5), (80, 80)), side=8)
    instances = split_mask_into_instances(m, max_k=2)  # 4 blobs -> KMeans into 2 clusters
    assert len(instances) == 2
    # Clustering partitions the blobs, so total foreground is preserved.
    assert sum(_fg(i) for i in instances) == _fg(m)


# --- run_iterative_inpaint -----------------------------------------------------------------------


def _recording_paint(color=(255, 0, 0)):
    """Fake model_inpaint that records calls and returns a solid-colour crop."""
    calls = []

    def paint(cropped_image, cropped_mask, anomaly_name):
        calls.append((cropped_image.size, anomaly_name))
        return Image.new("RGB", cropped_image.size, color)

    return paint, calls


def test_run_iterative_empty_mask_returns_copy():
    image = _gray()
    empty = Image.fromarray(np.zeros((100, 100), np.uint8), mode="L")
    paint, calls = _recording_paint()
    out = run_iterative_inpaint(image, empty, "t+d", paint, max_instances=DEFAULT_MAX_INSTANCES)
    assert calls == []  # nothing to inpaint
    assert isinstance(out, Image.Image)
    assert np.array_equal(np.array(out), np.array(image))


def test_run_iterative_pastes_reconstruction_into_mask_region():
    image = _gray(value=128)
    mask = _mask(blobs=((40, 40),), side=16)
    paint, calls = _recording_paint(color=(255, 0, 0))
    out = run_iterative_inpaint(image, mask, "scratch+dent", paint, crop_grid=(32, 32), max_instances=1)

    assert len(calls) == 1
    assert calls[0][1] == "scratch+dent"  # anomaly_name threaded through
    # A pixel well inside the mask is replaced by the reconstruction colour.
    px = np.array(out)[47, 47]
    assert px[0] == 255 and px[1] == 0 and px[2] == 0


def test_run_iterative_no_crop_uses_full_image():
    image = _gray()
    mask = _mask(blobs=((40, 40),), side=16)
    paint, calls = _recording_paint()
    run_iterative_inpaint(image, mask, "t+d", paint, crop_and_paste=False, max_instances=1)
    # Without crop-and-paste the model sees the whole image as the crop.
    assert calls[0][0] == image.size


def test_run_iterative_crop_ratio_sizes_window_from_bbox():
    image = _gray()
    mask = _mask(blobs=((40, 40),), side=16)
    paint, calls = _recording_paint()
    run_iterative_inpaint(image, mask, "t+d", paint, crop_ratio=DEFAULT_CROP_RATIO, max_instances=1)
    # crop_ratio path sizes a square window via crop_grid_by_ratio (bbox side * ratio).
    assert len(calls) == 1
    assert calls[0][0][0] == calls[0][0][1]  # square crop


def test_run_iterative_returns_artifacts():
    image = _gray()
    mask = _mask(blobs=((20, 20), (70, 70)), side=10)
    paint, _ = _recording_paint()
    out, artifacts = run_iterative_inpaint(image, mask, "t+d", paint, crop_grid=(32, 32), return_artifacts=True)
    assert isinstance(out, Image.Image)
    assert set(artifacts) == {"cropped_image", "cropped_mask", "annotated_image", "mask_cropped_image"}
    # One entry per instance (two blobs -> two instances).
    for key in artifacts:
        assert len(artifacts[key]) == 2


# --- run_iterative_inpaint_batch -----------------------------------------------------------------


def _batch_paint(color=(255, 0, 0)):
    """Fake batched paint that records the seeds of each depth's call; one solid crop per slot."""
    calls = []

    def paint(crops, masks, names, seeds):
        assert len(crops) == len(masks) == len(names) == len(seeds)
        calls.append(list(seeds))
        return [Image.new("RGB", c.size, color) for c in crops]

    return paint, calls


def test_batch_matches_serial_for_images_with_different_instance_counts():
    imgs = [_gray(value=128), _gray(value=64)]
    masks = [_mask(blobs=[(40, 40)], side=16), _mask(blobs=[(20, 20), (70, 70)], side=10)]  # 1 vs 2 instances
    names = ["t+d", "t+d"]

    # Serial per-image reference with the same (solid-colour) paint.
    serial_paint = _recording_paint()[0]
    serial = [
        run_iterative_inpaint(
            imgs[i], masks[i], names[i], serial_paint, crop_grid=(32, 32), max_instances=DEFAULT_MAX_INSTANCES
        )
        for i in range(2)
    ]
    composites, artifacts = run_iterative_inpaint_batch(
        imgs,
        masks,
        names,
        _batch_paint()[0],
        num_depth=2,
        seeds=[1, 1],
        crop_grids=[(32, 32), (32, 32)],
        crop_ratios=[None, None],
        crop_and_pastes=[True, True],
        poisson_blends=[False, False],
        max_instances_list=[DEFAULT_MAX_INSTANCES, DEFAULT_MAX_INSTANCES],
    )
    assert len(composites) == 2
    for i in range(2):
        # Batched instance-depth generation (with dummy padding) matches the serial per-image path.
        assert np.array_equal(np.array(composites[i]), np.array(serial[i]))
    assert len(artifacts[0]["cropped_image"]) == 1  # image 0: 1 instance
    assert len(artifacts[1]["cropped_image"]) == 2  # image 1: 2 instances


def test_batch_seeds_are_offset_by_instance_depth():
    # Every instance of a sample must get its OWN noise draw: depth j is generated with
    # ``seed + j``, mirroring build_inpaint_one's per-instance counter in the serial path. Reusing
    # one seed across depths would give a multi-instance sample identical noise for every defect.
    imgs = [_gray(value=128), _gray(value=64)]
    masks = [_mask(blobs=[(20, 20), (70, 70)], side=10), _mask(blobs=[(20, 20), (70, 70)], side=10)]
    paint, calls = _batch_paint()
    run_iterative_inpaint_batch(
        imgs,
        masks,
        ["t+d", "t+d"],
        paint,
        num_depth=2,
        seeds=[500, 9000],
        crop_grids=[(32, 32), (32, 32)],
        crop_ratios=[None, None],
        crop_and_pastes=[True, True],
        poisson_blends=[False, False],
        max_instances_list=[DEFAULT_MAX_INSTANCES, DEFAULT_MAX_INSTANCES],
    )
    # One call per depth; each sample keeps its own base seed and is offset by the depth only.
    assert calls == [[500, 9000], [501, 9001]]


def test_batch_empty_mask_returns_original():
    imgs = [_gray()]
    empty = Image.fromarray(np.zeros((100, 100), np.uint8), mode="L")
    composites, artifacts = run_iterative_inpaint_batch(
        imgs,
        [empty],
        ["t+d"],
        _batch_paint()[0],
        num_depth=1,
        seeds=[1],
        crop_grids=[(32, 32)],
        crop_ratios=[None],
        crop_and_pastes=[True],
        poisson_blends=[False],
        max_instances_list=[DEFAULT_MAX_INSTANCES],
    )
    assert np.array_equal(np.array(composites[0]), np.array(imgs[0]))
    assert artifacts[0]["cropped_image"] == []
