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

"""
CPU-only unit tests for the roi_generate fixes from the 2026-07-06 QC sweep
(items #1, #4, #5, #6, #7). No GPU or model checkpoints required, but the
roi_generate import chain needs torch/cv2/pycocotools/omegaconf — the tests
skip cleanly where those are absent (run them inside the dev container).

Usage:

pytest tests/test_roi_generate_unit.py
"""
import json

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("scipy")
pytest.importorskip("pycocotools")
pytest.importorskip("omegaconf")
pytest.importorskip("skimage")
pytest.importorskip("PIL")

from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

from roi_generate.box_to_mask import BoxToMaskPostProcess  # noqa: E402
from roi_generate.default_config import DefaultConfig, validate_sample_config  # noqa: E402
from roi_generate.template_box_to_masks import (  # noqa: E402
    PipelineStage,
    PostProcessStage,
    ProposalGenerationStage,
    load_cached_context,
    quantize_box_to_grid,
)
from roi_generate.utils import generate_augmented_variants  # noqa: E402


# --- QC #1: image and mask augmentation lists must stay index-aligned -------

def _asymmetric_image(size=32):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[2:6, 2:6] = (255, 0, 0)      # top-left red block
    img[-8:-2, -8:-2] = (0, 255, 0)  # bottom-right green block
    return img


def _symmetric_mask(size=32):
    mask = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    c = (size - 1) / 2
    mask[(yy - c) ** 2 + (xx - c) ** 2 <= (size // 4) ** 2] = 255  # centered disc
    return mask


def test_augmented_variants_stay_aligned_for_symmetric_masks():
    degrees = [0.0, 90.0, 180.0, 270.0]
    img_variants, img_meta = generate_augmented_variants(
        [_asymmetric_image()], degrees, True, True)
    mask_variants, mask_meta = generate_augmented_variants(
        [_symmetric_mask()], degrees, True, True, is_mask=True)

    # A rotation-symmetric mask used to collapse via content-hash dedup
    # while its image variants stayed distinct, desynchronizing the zip in
    # HOGFilteringStage.
    assert len(img_variants) == len(mask_variants)

    def key(m):
        return (m["source"], m["rotation"], m["flip_lr"], m["flip_ud"])

    assert [key(m) for m in img_meta] == [key(m) for m in mask_meta]


def test_augmented_variants_dedup_on_canonical_transform():
    # 0° and 360° canonicalize identically → one variant, not two.
    variants, meta = generate_augmented_variants(
        [_asymmetric_image()], [0.0, 360.0], False, False)
    assert len(variants) == 1


# --- QC #4: degenerate template box must still cover one feature cell -------

def test_quantize_thin_box_gets_at_least_one_cell():
    # A 2px-wide box in a 1024px image collapses to zero cells on a 512 grid.
    tx0, ty0, tx1, ty1 = quantize_box_to_grid((100.0, 100.0, 101.5, 300.0),
                                              1024, 1024, 512, 512)
    assert tx1 > tx0 and ty1 > ty0


def test_quantize_box_at_far_edge_stays_in_bounds():
    tx0, ty0, tx1, ty1 = quantize_box_to_grid((1023.0, 1023.0, 1024.0, 1024.0),
                                              1024, 1024, 512, 512)
    assert 0 <= tx0 < tx1 <= 512
    assert 0 <= ty0 < ty1 <= 512


def test_quantize_normal_box_unchanged():
    assert quantize_box_to_grid((256.0, 256.0, 768.0, 768.0),
                                1024, 1024, 512, 512) == (128, 128, 384, 384)


def test_quantize_out_of_range_box_still_gets_one_cell():
    # Boxes are validated upstream, but the helper must stay safe on its
    # own: negative coordinates truncate toward zero and used to produce an
    # empty (0, 0) slice.
    tx0, ty0, tx1, ty1 = quantize_box_to_grid((-3.0, -3.0, 0.4, 0.4),
                                              1024, 1024, 512, 512)
    assert 0 <= tx0 < tx1 <= 512
    assert 0 <= ty0 < ty1 <= 512


# --- QC #5: proposal subsampling is seeded via config ------------------------

def test_proposal_seed_default_and_validation():
    cfg = OmegaConf.structured(DefaultConfig)
    assert cfg.template_box_to_masks.proposal_seed == 0
    assert validate_sample_config(cfg)

    cfg.template_box_to_masks.proposal_seed = -1
    with pytest.raises(ValueError, match="proposal_seed"):
        validate_sample_config(cfg)


def test_proposal_seed_participates_in_dependency_hash():
    # The load-bearing half of QC #5: the seed must be part of the stage's
    # cache deps, or the dependency-hash cache can report a match while the
    # real (differently-seeded) output would differ.
    def stage_with_seed(seed):
        cfg = OmegaConf.structured(DefaultConfig)
        cfg.template_box_to_masks.proposal_seed = seed
        return ProposalGenerationStage(cfg, None)

    s0 = stage_with_seed(0)
    assert s0.deps["proposal_seed"] == 0

    h0 = s0.compute_dependency_hash({})
    assert stage_with_seed(0).compute_dependency_hash({}) == h0  # stable
    assert stage_with_seed(7).compute_dependency_hash({}) != h0  # seed-sensitive


# --- QC #6: resume returns the verified hash chain ---------------------------

class _FakeStage(PipelineStage):
    def __init__(self, name, dep_value):
        super().__init__(name)
        self.deps = {"value": dep_value}
        self.result = {"payload": name}

    def run(self, ctx):
        ctx[self.name] = self.result
        return ctx


def _ctx(tmp_path):
    return {"input": {"output_dir": str(tmp_path)}}


def test_load_cached_context_returns_continuable_prev_hash(tmp_path):
    stages = [_FakeStage("s0", 1), _FakeStage("s1", 2), _FakeStage("s2", 3)]
    ctx = _ctx(tmp_path)
    cache_dir = str(tmp_path / "template_box_to_masks" / "cache")

    # Simulate a full prior run that cached the first two stages.
    prev = None
    for stage in stages[:2]:
        h = stage.compute_dependency_hash(ctx, prev)
        stage.save_cache(ctx, h)
        prev = h
    full_chain_hash = prev

    restored_ctx, start_idx, prev_hash = load_cached_context(
        stages, _ctx(tmp_path), cache_dir)
    assert start_idx == 2
    assert restored_ctx["s0"] == {"payload": "s0"}
    assert restored_ctx["s1"] == {"payload": "s1"}
    # The chain must continue from the last verified hash — restarting at
    # None permanently invalidated every stage past the resume point.
    assert prev_hash == full_chain_hash


def test_load_cached_context_cold_start(tmp_path):
    stages = [_FakeStage("s0", 1), _FakeStage("s1", 2)]
    _, start_idx, prev_hash = load_cached_context(
        stages, _ctx(tmp_path), str(tmp_path / "nope"))
    assert start_idx == 0
    assert prev_hash is None


# --- QC #7: result.json records both coordinate spaces -----------------------

def test_box_to_mask_result_json_marks_coordinate_spaces(tmp_path):
    cfg = OmegaConf.structured(DefaultConfig)
    post = BoxToMaskPostProcess(cfg)

    # One 8x8 blob at (16,16) in a 64x64 processed mask.
    mask = np.zeros((1, 64, 64), dtype=np.uint8)
    mask[0, 16:24, 16:24] = 1
    post.run(mask)

    ctx = {"input": {
        "image_path": "img.png",
        "image": Image.new("RGB", (64, 64)),
        "boxes": [[16, 16, 24, 24]],
        "ori_image_size": (128, 128),
        "output_dir": str(tmp_path),
    }}
    post.save_result(ctx)

    result = json.loads((tmp_path / "box_to_mask" / "output" / "result.json").read_text())
    assert result["original_image_size"] == [128, 128]
    assert result["processed_image_size"] == [64, 64]
    # input_boxes stay in processed space; boxes come from the mask resized
    # back to original space (2x here).
    assert result["input_boxes"] == [[16, 16, 24, 24]]
    assert result["boxes"] == [[32, 32, 48, 48]]


def test_template_result_json_marks_coordinate_spaces(tmp_path):
    # Same contract as above, on the parallel PostProcessStage.save_result
    # path in template_box_to_masks.py.
    cfg = OmegaConf.structured(DefaultConfig)
    stage = PostProcessStage(cfg)

    binary_mask = np.zeros((64, 64), dtype=np.uint8)
    binary_mask[16:24, 16:24] = 255
    stage.result["binary_mask"] = binary_mask

    small_mask = np.zeros((64, 64), dtype=np.uint8)
    small_mask[16:24, 16:24] = 1
    ctx = {
        "input": {
            "image_path": "img.png",
            "image": Image.new("RGB", (64, 64)),
            "boxes": [[16, 16, 24, 24]],
            "ori_image_size": (128, 128),
            "output_dir": str(tmp_path),
        },
        "template_prepare": {"refined_template_boxes": [[16, 16, 24, 24]]},
        "sam_inference": {"template_masks": [small_mask],
                          "candidate_masks": [small_mask]},
        "proposal_generation": {"proposal_boxes": np.array([[16.0, 16.0, 24.0, 24.0]])},
        "box_filter": {"size_diff": np.array([0.0]), "aspect_diff": np.array([0.0])},
        "mask_filter": {"component_diffs": np.array([0]),
                        "chamfer_score": np.array([0.0])},
        "hog_filter": {"sim_hog": np.array([1.0])},
        "color_filter": {"sim_lightness": np.array([1.0]),
                         "sim_color": np.array([1.0])},
    }
    stage.save_result(ctx)

    result = json.loads(
        (tmp_path / "template_box_to_masks" / "output" / "result.json").read_text())
    assert result["original_image_size"] == [128, 128]
    assert result["processed_image_size"] == [64, 64]
    assert result["template_boxes"] == [[16, 16, 24, 24]]  # processed space
    assert result["boxes"] == [[32, 32, 48, 48]]           # original space
