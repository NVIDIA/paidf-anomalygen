# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the anomaly inpainting TRAINING dataset.

Driven off a tiny on-disk fixture in the layout ``InpaintingDataset._index`` expects, so the whole
read -> augment -> source/target path runs for real. Tokenization is the one thing stubbed out: it
needs the framework's Qwen tokenizer (a download + a GPU-env dependency) and is orthogonal to the
augmentation/source-item behaviour under test.

The augmentations are probabilistic, so every test pins the per-sample probabilities to 0.0/1.0 and
seeds both the dataset's own NumPy generator (``seed=``) and torch's global RNG (the ring jitter is
a torchvision transform).
"""

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from anomalygen.data.inpainting_dataset import InpaintingDataset

_ANOMALY_TYPES = [["metal", "scratch"]]
_IMAGE_SIZE = (64, 64)
_BG_VALUE = 128  # flat mid-grey background: any near-black pixel in a sample must come from an aug


def _write_pair(root, blobs, size=64, side=8, bg=_BG_VALUE, name="a"):
    """Write one (anomaly_image, mask) pair in the on-disk layout the dataset indexes.

    ``blobs`` are (row, col) top-left corners of ``side``x``side`` filled mask squares.
    """
    img_dir = root / "metal" / "anomaly_image" / "scratch"
    mask_dir = root / "metal" / "mask" / "scratch"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(np.full((size, size, 3), bg, np.uint8), mode="RGB").save(img_dir / f"{name}.png")
    arr = np.zeros((size, size), dtype=np.uint8)
    for r, c in blobs:
        arr[r : r + side, c : c + side] = 255
    Image.fromarray(arr, mode="L").save(mask_dir / f"{name}_mask.png")
    return root


@pytest.fixture
def stub_tokenizer(monkeypatch):
    """Skip real tokenization (needs the framework's Qwen tokenizer); captions aren't under test."""
    monkeypatch.setattr(
        InpaintingDataset, "_tokenize", lambda self, caption: torch.zeros(4, dtype=torch.long), raising=True
    )


def _dataset(root, **kwargs):
    """Dataset over ``root`` with every probabilistic aug off unless the test turns one on."""
    params = dict(
        dataset_dir=str(root),
        anomaly_types=_ANOMALY_TYPES,
        image_size=_IMAGE_SIZE,
        # A single fixed zoom ratio, big enough that the crop window covers the whole fixture image,
        # so a test's second mask blob cannot be cropped out from under it.
        ratio_range=(8.0, 8.0),
        background_dropout_prob=0.0,
        inst_aug_prob=0.0,
        ring_jitter_prob=0.0,
        seed=1,
    )
    params.update(kwargs)
    return InpaintingDataset(**params)


def _components(edit_mask: torch.Tensor) -> int:
    """Connected-component count of an ``edit_mask`` [1,H,W], background excluded."""
    arr = (edit_mask[0].numpy() > 0).astype(np.uint8)
    return cv2.connectedComponents(arr, connectivity=8)[0] - 1


# --- construction --------------------------------------------------------------------------------


def test_empty_dataset_dir_raises(tmp_path):
    # An empty dataset would hang the loader with no batch emitted, so it must fail at build time.
    with pytest.raises(FileNotFoundError):
        _dataset(tmp_path)


@pytest.mark.parametrize("prob", [-0.1, 1.5])
@pytest.mark.parametrize("field", ["background_dropout_prob", "inst_aug_prob", "ring_jitter_prob"])
def test_per_sample_probabilities_are_range_validated(tmp_path, field, prob):
    # Out-of-range probabilities must fail at dataset build, not silently change the aug rate
    # (p > 1 makes an "optional" aug unconditional; p < 0 makes it dead).
    _write_pair(tmp_path, blobs=[(28, 28)])
    with pytest.raises(ValueError):
        _dataset(tmp_path, **{field: prob})


# --- source item / background dropout ------------------------------------------------------------


def test_background_dropout_blacks_background_and_keeps_defect_noise(tmp_path, stub_tokenizer):
    _write_pair(tmp_path, blobs=[(28, 28)])
    torch.manual_seed(0)  # the source item's defect noise comes from torch's global RNG
    item = _dataset(tmp_path, background_dropout_prob=1.0)[0]

    source, target = item["images"]
    m3 = item["edit_mask"].expand_as(target) > 0
    # Background dropped to -1 (black) everywhere outside the defect ...
    assert torch.equal(source[~m3], torch.full_like(source[~m3], -1.0))
    # ... while the defect region is noise, not more black.
    assert not torch.all(source[m3] == -1.0)
    assert float(source[m3].max()) > 0.0


def test_no_background_dropout_keeps_target_background(tmp_path, stub_tokenizer):
    _write_pair(tmp_path, blobs=[(28, 28)])
    torch.manual_seed(0)  # the source item's defect noise comes from torch's global RNG
    item = _dataset(tmp_path, background_dropout_prob=0.0)[0]

    source, target = item["images"]
    m3 = item["edit_mask"].expand_as(target) > 0
    # Source background is the clean target verbatim; only the defect region is substituted.
    assert torch.equal(source[~m3], target[~m3])
    assert not torch.equal(source[m3], target[m3])


# --- instance augmentation -----------------------------------------------------------------------


def test_instance_aug_keeps_one_instance_and_blacks_the_dropped_one(tmp_path, stub_tokenizer):
    # Two well-separated blobs; with inst_aug_prob=1.0 exactly one survives in the mask and the
    # other is blacked out in the IMAGE, so the pair stays self-consistent (no unmarked defect).
    _write_pair(tmp_path, blobs=[(24, 12), (24, 44)])
    two_blob_item = _dataset(tmp_path, inst_aug_prob=0.0)[0]
    assert _components(two_blob_item["edit_mask"]) == 2  # baseline: both instances present

    item = _dataset(tmp_path, inst_aug_prob=1.0)[0]
    assert _components(item["edit_mask"]) == 1
    _, target = item["images"]
    assert float(target.min()) < -0.9  # the dropped blob is black (-1) in the target pixels
    assert float(two_blob_item["images"][1].min()) > -0.9  # ... and was not, without the aug


def test_instance_aug_is_a_noop_on_a_single_instance_mask(tmp_path, stub_tokenizer):
    # Nothing to drop: the mask keeps its one instance and no blackout is painted into the image.
    _write_pair(tmp_path, blobs=[(28, 28)])
    item = _dataset(tmp_path, inst_aug_prob=1.0)[0]

    assert _components(item["edit_mask"]) == 1
    assert float(item["images"][1].min()) > -0.9  # flat mid-grey survives untouched


# --- ring colour jitter --------------------------------------------------------------------------


def test_ring_jitter_never_touches_the_defect_pixels(tmp_path, stub_tokenizer):
    # The whole point of the band is to destroy the colour cue AROUND the defect; recolouring the
    # defect itself would corrupt the thing the model is being taught to draw.
    _write_pair(tmp_path, blobs=[(28, 28)], bg=200)
    ds = _dataset(tmp_path, ring_jitter_prob=1.0)
    image = Image.open(ds.samples[0]["image"]).convert("RGB")
    mask = Image.open(ds.samples[0]["mask"]).convert("L")

    torch.manual_seed(0)  # the jitter draw is a torchvision transform (torch's global RNG)
    jittered, _ = ds.augmentations[0](image, mask)  # [0] is the ring jitter

    before, after = np.asarray(image), np.asarray(jittered)
    inside = np.asarray(mask) > 0
    assert np.array_equal(after[inside], before[inside])  # defect pixels byte-identical
    assert not np.array_equal(after[~inside], before[~inside])  # the surrounding band did change


def test_ring_jitter_sample_is_well_formed(tmp_path, stub_tokenizer):
    _write_pair(tmp_path, blobs=[(28, 28)], bg=200)
    torch.manual_seed(0)
    item = _dataset(tmp_path, ring_jitter_prob=1.0)[0]

    source, target = item["images"]
    assert source.shape == target.shape == (3, *_IMAGE_SIZE)
    assert float(item["edit_mask"].sum()) > 0.0


# --- item contract -------------------------------------------------------------------------------


def test_item_shapes_and_class_id(tmp_path, stub_tokenizer):
    _write_pair(tmp_path, blobs=[(28, 28)])
    item = _dataset(tmp_path)[0]

    assert len(item["images"]) == 2  # [source, target] -> two vision items
    assert all(t.shape == (3, *_IMAGE_SIZE) for t in item["images"])
    assert item["edit_mask"].shape == (1, *_IMAGE_SIZE)
    assert set(torch.unique(item["edit_mask"]).tolist()) <= {0.0, 1.0}  # binary
    assert item["anomaly_class_id"] == 0
    assert item["num_frames"] == 2


@pytest.mark.parametrize("inst_aug_prob", [0.0, 1.0])
def test_edit_mask_is_never_all_zero(tmp_path, stub_tokenizer, inst_aug_prob):
    # An all-zero edit_mask is a degenerate "inpaint nothing" sample whose masked loss has no
    # grad_fn — the crop must always keep the defect in frame, across repeated draws.
    _write_pair(tmp_path, blobs=[(24, 12), (24, 44)])
    ds = _dataset(tmp_path, inst_aug_prob=inst_aug_prob, ring_jitter_prob=1.0)
    torch.manual_seed(0)
    for i in range(10):
        assert float(ds[i]["edit_mask"].sum()) > 0.0
