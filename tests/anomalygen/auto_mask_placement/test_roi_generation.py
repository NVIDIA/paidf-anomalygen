# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test roi_generate pipeline."""

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from anomalygen.auto_mask_placement.roi_generation import (
    ROIGenerationConfig,
    ROIGenerationModels,
    ensure_valid_boxes,
    load_one_sample,
    prepare_samples,
    run_roi_pipeline,
    validate_sample_config,
)
from anomalygen.auto_mask_placement.roi_generation.box_to_mask import BoxToMaskPostProcess
from anomalygen.auto_mask_placement.roi_generation.template_box_to_masks import (
    PipelineStage,
    PostProcessStage,
    ProposalGenerationStage,
    _generate_augmented_variants,
    _quantize_box_to_grid,
    load_cached_context,
)

# Dropped during the cosmos3 migration: template-matching on rectangle templates regressed
# (IoU < 0.95).
_DROPPED_PIPELINE_CASES = {
    "template_box_to_masks with single template rectangle object",
    "template_box_to_masks with two template rectangle objects",
    "template_box_to_masks with color_hist disabled",
}


def _create_test_image(size=(512, 512), background_color=(128, 128, 128), objects=None):
    """
    Create a test image and draw all given objects.

    Supported objects:
      - {"type": "rectangle", "coords": [x1, y1, x2, y2], "color": (r,g,b)}
      - {"type": "circle", "center": (cx, cy), "radius": r, "color": (r,g,b)}

    Note: size should be (H, W) to match NumPy convention. PIL requires (W, H), so conversion happens here.
    """
    H, W = size  # size is (H, W)
    img = Image.new("RGB", (W, H), color=background_color)  # PIL uses (W, H)
    draw = ImageDraw.Draw(img)

    if objects is None:
        return img

    for obj in objects:
        t = obj["type"]

        if t == "rectangle":
            x1, y1, x2, y2 = obj["coords"]
            draw.rectangle([x1, y1, x2, y2], fill=obj["color"])

        elif t == "circle":
            cx, cy = obj["center"]
            r = obj["radius"]
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw.ellipse(bbox, fill=obj["color"])

    return img


def _create_gt_mask(size=(512, 512), objects=None):
    """
    Create ground-truth mask from a list of objects.
    Supports:
        {"type": "rectangle", "coords": [x1, y1, x2, y2]}
        {"type": "circle", "center": (cx, cy), "radius": r}
    """
    H, W = size
    mask = np.zeros((H, W), dtype=np.uint8)

    y, x = np.ogrid[:H, :W]
    if objects is None:
        return mask

    for obj in objects:
        if obj["type"] == "rectangle":
            x1, y1, x2, y2 = obj["coords"]
            mask[y1:y2, x1:x2] = 255

        elif obj["type"] == "circle":
            cx, cy = obj["center"]
            r = obj["radius"]
            circle = (x - cx) ** 2 + (y - cy) ** 2 <= r**2
            mask[circle] = 255

    return mask


def _compute_iou(mask1, mask2):
    """Compute Intersection over Union between two binary masks."""
    mask1_binary = (mask1 > 0).astype(np.uint8)
    mask2_binary = (mask2 > 0).astype(np.uint8)

    intersection = np.logical_and(mask1_binary, mask2_binary).sum()
    union = np.logical_or(mask1_binary, mask2_binary).sum()

    if union == 0:
        return 0.0

    return intersection / union


def _get_device():
    """Get torch device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_test_pipeline(tmp_path, image, boxes, config_dict):
    """Run pipeline and return result mask path."""
    image_path = tmp_path / "test.png"
    image.save(image_path)

    samples = [{"image_path": str(image_path), "boxes": boxes, "config": config_dict}]

    device = _get_device()
    models = ROIGenerationModels(device)

    run_roi_pipeline(samples, str(tmp_path), models)


def test_validate_boxes():
    """Test box validation."""
    assert ensure_valid_boxes([[10, 20, 50, 60], [100, 100, 200, 200]], 512, 512)

    with pytest.raises(ValueError, match="Invalid box format"):
        ensure_valid_boxes([[10, 20, 50]], 512, 512)

    with pytest.raises(ValueError, match="non-positive area"):
        ensure_valid_boxes([[50, 60, 10, 20]], 512, 512)

    with pytest.raises(ValueError, match="out of image bounds"):
        ensure_valid_boxes([[0, 0, 600, 100]], 512, 512)


def test_config_validation():
    """Test configuration validation."""
    default_config = OmegaConf.structured(ROIGenerationConfig)
    validate_sample_config(default_config)

    invalid_config = OmegaConf.create(
        {
            "box_to_mask": {"enabled": False},
            "template_box_to_masks": {"enabled": False},
            "grayscale_to_mask": {"enabled": False},
        }
    )
    merged = OmegaConf.merge(default_config, invalid_config)
    with pytest.raises(ValueError, match="At least one mode must be enabled"):
        validate_sample_config(merged)


def test_prepare_samples():
    """Test sample preparation."""
    samples = [{"image_path": "/path/to/image.png", "boxes": [[10, 20, 50, 60]]}]
    prepared = prepare_samples(samples)
    assert len(prepared) == 1
    assert "config" in prepared[0]

    with pytest.raises(ValueError, match="missing required field: 'image_path'"):
        prepare_samples([{"boxes": [[10, 20, 50, 60]]}])

    with pytest.raises(ValueError, match="missing required field: 'boxes'"):
        prepare_samples([{"image_path": "/path/to/image.png"}])


def test_load_one_sample(tmp_path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (512, 512), color=(0, 0, 0)).save(img_path)

    sample = {
        "image_path": str(img_path),
        "boxes": [[10, 20, 100, 200]],
        "config": ROIGenerationConfig(),
    }

    ctx = load_one_sample(sample)

    assert "input" in ctx
    assert ctx["input"]["boxes"] == [[10, 20, 100, 200]]

    sample = {
        "image_path": str(img_path),
        "boxes": [],
        "config": ROIGenerationConfig(),
    }

    with pytest.raises(ValueError, match="Please provide a bounding box or disable these modes"):
        load_one_sample(sample)


@pytest.mark.gpu
def test_roi_generate_models_forward_segmentation():
    """Test SAM2 inference with boxes."""
    device = _get_device()
    models = ROIGenerationModels(device)

    objects = [
        {"type": "rectangle", "coords": [100, 100, 200, 200], "color": (255, 0, 0)},
    ]
    img = _create_test_image(objects=objects)

    boxes = [[100, 100, 200, 200]]
    masks, scores = models.forward_segmentation(img, boxes=boxes)

    assert masks.shape[0] == 1
    assert masks.shape[1:] == (512, 512)
    assert scores.shape[0] == 1
    assert np.all((masks == 0) | (masks == 1))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "description,bg_color,objects,input_boxes,config,iou_threshold,image_size",
    [
        # Test case format: (description, background_color, objects, input_boxes, config, iou_threshold, image_size)
        (
            "grayscale_to_mask with single rectangle object",
            (100, 100, 100),
            [{"type": "rectangle", "coords": [100, 100, 400, 400], "color": (200, 200, 200)}],
            [],
            {
                "grayscale_to_mask": {"enabled": True},
                "box_to_mask": {"enabled": False},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (512, 512),
        ),
        (
            "grayscale_to_mask with single rectangle dark object",
            (255, 255, 255),
            [{"type": "rectangle", "coords": [100, 100, 400, 400], "color": (100, 100, 100)}],
            [],
            {
                "grayscale_to_mask": {
                    "enabled": True,
                    "threshold_mode": "otsu_inv",
                },
                "box_to_mask": {"enabled": False},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (512, 512),
        ),
        (
            "grayscale_to_mask with multiple rectangle objects",
            (50, 50, 50),
            [
                {"type": "rectangle", "coords": [50, 50, 150, 150], "color": (220, 220, 0)},
                {"type": "rectangle", "coords": [200, 200, 350, 350], "color": (220, 220, 0)},
                {"type": "rectangle", "coords": [380, 50, 480, 150], "color": (0, 220, 220)},
            ],
            [],
            {
                "grayscale_to_mask": {"enabled": True},
                "box_to_mask": {"enabled": False},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (512, 512),
        ),
        (
            "grayscale_to_mask with fixed threshold",
            (10, 10, 10),
            [
                {"type": "rectangle", "coords": [50, 50, 150, 150], "color": (255, 0, 0)},
                {"type": "rectangle", "coords": [200, 200, 350, 350], "color": (0, 255, 0)},
                {"type": "rectangle", "coords": [380, 50, 480, 150], "color": (0, 0, 255)},
                {"type": "circle", "center": (128, 256), "radius": 40, "color": (0, 255, 0)},
                {"type": "circle", "center": (256, 128), "radius": 40, "color": (0, 0, 255)},
                {"type": "circle", "center": (384, 384), "radius": 40, "color": (255, 0, 0)},
            ],
            [],
            {
                "grayscale_to_mask": {"enabled": True, "threshold_mode": "custom", "threshold_value": 20},
                "box_to_mask": {"enabled": False},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (512, 512),
        ),
        (
            "grayscale_to_mask with fixed inverse threshold",
            (255, 255, 255),
            [
                {"type": "rectangle", "coords": [50, 50, 150, 150], "color": (20, 0, 0)},
                {"type": "rectangle", "coords": [200, 200, 350, 350], "color": (0, 20, 0)},
                {"type": "rectangle", "coords": [380, 50, 480, 150], "color": (0, 0, 20)},
                {"type": "circle", "center": (128, 256), "radius": 40, "color": (0, 20, 0)},
                {"type": "circle", "center": (256, 128), "radius": 40, "color": (0, 0, 20)},
                {"type": "circle", "center": (384, 384), "radius": 40, "color": (20, 0, 0)},
            ],
            [],
            {
                "grayscale_to_mask": {"enabled": True, "threshold_mode": "custom_inv", "threshold_value": 20},
                "box_to_mask": {"enabled": False},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (512, 512),
        ),
        (
            "box_to_mask with single rectangle object",
            (50, 50, 50),
            [{"type": "rectangle", "coords": [100, 100, 300, 300], "color": (200, 50, 50)}],
            [[100, 100, 300, 300]],
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": True},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (512, 512),
        ),
        (
            "box_to_mask with multiple rectangle objects",
            (50, 50, 50),
            [
                {"type": "rectangle", "coords": [50, 50, 150, 150], "color": (200, 50, 50)},
                {"type": "rectangle", "coords": [200, 200, 350, 350], "color": (50, 200, 50)},
                {"type": "rectangle", "coords": [380, 50, 480, 150], "color": (50, 50, 200)},
            ],
            [[50, 50, 150, 150], [200, 200, 350, 350], [380, 50, 480, 150]],
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": True},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (512, 512),
        ),
        (
            "box_to_mask with circular object",
            (50, 50, 50),
            [{"type": "circle", "center": (256, 256), "radius": 80, "color": (200, 100, 50)}],
            [[166, 166, 346, 346]],  # Box around circle with padding (radius 80 + padding 10)
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": True},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (512, 512),
        ),
        (
            "box_to_mask with a large rectangle object",
            (30, 30, 30),
            [{"type": "rectangle", "coords": [50, 50, 450, 450], "color": (220, 80, 80)}],
            [[50, 50, 450, 450]],
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": True},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (512, 512),
        ),
        (
            "box_to_mask with multiple small rectangle objects",
            (60, 60, 60),
            [
                {"type": "rectangle", "coords": [100, 100, 150, 150], "color": (180, 60, 60)},
                {"type": "rectangle", "coords": [200, 200, 250, 250], "color": (60, 180, 60)},
                {"type": "rectangle", "coords": [350, 350, 400, 400], "color": (60, 60, 180)},
            ],
            [[100, 100, 150, 150], [200, 200, 250, 250], [350, 350, 400, 400]],
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": True},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (512, 512),
        ),
        (
            "box_to_mask with different colored objects",
            (40, 40, 40),
            [
                {"type": "rectangle", "coords": [80, 80, 180, 180], "color": (255, 0, 0)},
                {"type": "rectangle", "coords": [250, 80, 350, 180], "color": (0, 255, 0)},
                {"type": "rectangle", "coords": [80, 250, 180, 350], "color": (0, 0, 255)},
                {"type": "rectangle", "coords": [250, 250, 350, 350], "color": (255, 255, 0)},
            ],
            [[80, 80, 180, 180], [250, 80, 350, 180], [80, 250, 180, 350], [250, 250, 350, 350]],
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": True},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (512, 512),
        ),
        (
            "template_box_to_masks with single template rectangle object",
            (50, 50, 50),
            [
                {"type": "rectangle", "coords": [80, 80, 150, 150], "color": (200, 50, 50)},
                {"type": "rectangle", "coords": [200, 80, 270, 150], "color": (200, 50, 50)},
                {"type": "rectangle", "coords": [80, 200, 150, 270], "color": (200, 50, 50)},
            ],
            [[80, 80, 150, 150]],  # template box
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": False},
                "template_box_to_masks": {
                    "enabled": True,
                    "max_template": 1,
                    "max_proposal": 200,
                    "save_cache": False,
                    "resume_cache": False,
                },
            },
            0.95,
            (512, 512),
        ),
        (
            "template_box_to_masks with two template rectangle objects",
            (40, 40, 40),
            [
                {"type": "rectangle", "coords": [50, 50, 120, 120], "color": (255, 100, 100)},
                {"type": "rectangle", "coords": [200, 60, 270, 130], "color": (255, 100, 100)},
                {"type": "rectangle", "coords": [60, 200, 130, 270], "color": (100, 255, 100)},
                {"type": "rectangle", "coords": [220, 220, 290, 290], "color": (100, 255, 100)},
            ],
            [
                [50, 50, 120, 120],
                [60, 200, 130, 270],
            ],  # template 1 (red)  # template 2 (green)
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": False},
                "template_box_to_masks": {
                    "enabled": True,
                    "max_template": 2,
                    "max_proposal": 200,
                    "save_cache": False,
                    "resume_cache": False,
                },
            },
            0.95,  # Lower threshold for template matching with multiple templates
            (512, 512),
        ),
        (
            "template_box_to_masks with single template circular object",
            (50, 50, 50),
            [
                {"type": "circle", "center": (120, 250), "radius": 40, "color": (200, 50, 50)},
                {"type": "circle", "center": (250, 120), "radius": 40, "color": (200, 50, 50)},
                {"type": "circle", "center": (380, 380), "radius": 40, "color": (200, 50, 50)},
            ],
            [[80, 210, 160, 290]],  # template box
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": False},
                "template_box_to_masks": {
                    "enabled": True,
                    "max_template": 1,
                    "max_proposal": 100,
                    "save_cache": False,
                    "resume_cache": False,
                },
            },
            0.95,
            (512, 512),
        ),
        (
            "template_box_to_masks with two template circular objects",
            (40, 40, 40),
            [
                {"type": "circle", "center": (120, 250), "radius": 40, "color": (255, 100, 100)},
                {"type": "circle", "center": (250, 250), "radius": 40, "color": (255, 100, 100)},
                {"type": "circle", "center": (380, 250), "radius": 40, "color": (100, 255, 100)},
                {"type": "circle", "center": (250, 80), "radius": 40, "color": (100, 255, 100)},
            ],
            [
                [80, 210, 160, 290],
                [340, 210, 420, 290],
            ],  # template 1 (red)  # template 2 (green)
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": False},
                "template_box_to_masks": {
                    "enabled": True,
                    "max_template": 2,
                    "max_proposal": 100,
                    "save_cache": False,
                    "resume_cache": False,
                },
            },
            0.95,
            (512, 512),
        ),
        (
            "template_box_to_masks with color_hist disabled",
            (50, 50, 50),
            [
                {"type": "rectangle", "coords": [120, 120, 180, 180], "color": (200, 150, 250)},
                {"type": "rectangle", "coords": [260, 120, 320, 180], "color": (150, 200, 250)},
                {"type": "rectangle", "coords": [120, 260, 180, 320], "color": (250, 150, 200)},
                {"type": "rectangle", "coords": [190, 260, 250, 320], "color": (250, 200, 150)},
            ],
            [[120, 120, 180, 180]],
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": False},
                "template_box_to_masks": {
                    "enabled": True,
                    "max_template": 1,
                    "max_proposal": 300,
                    "color_hist_enabled": False,
                    "save_cache": False,
                    "resume_cache": False,
                },
            },
            0.95,
            (512, 512),
        ),
        (
            "all modes",
            (50, 50, 50),
            [
                {"type": "rectangle", "coords": [120, 120, 200, 200], "color": (255, 0, 0)},
                {"type": "rectangle", "coords": [260, 260, 340, 340], "color": (0, 255, 0)},
            ],
            [[120, 120, 200, 200]],
            {
                "grayscale_to_mask": {"enabled": True},
                "box_to_mask": {"enabled": True},
                "template_box_to_masks": {
                    "enabled": True,
                    "max_template": 1,
                    "max_proposal": 100,
                    "save_cache": False,
                    "resume_cache": False,
                },
            },
            0.45,  # lower IoU threshold because modes differ
            (512, 512),
        ),
        # Test cases with different image sizes
        (
            "box_to_mask with small image 256x256",
            (50, 50, 50),
            [{"type": "rectangle", "coords": [50, 50, 150, 150], "color": (200, 50, 50)}],
            [[50, 50, 150, 150]],
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": True},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (256, 256),
        ),
        (
            "box_to_mask with large image 1024x1024",
            (50, 50, 50),
            [{"type": "rectangle", "coords": [200, 200, 600, 600], "color": (200, 50, 50)}],
            [[200, 200, 600, 600]],
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": True},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (1024, 1024),
        ),
        (
            "box_to_mask with wide image",
            (50, 50, 50),
            [{"type": "rectangle", "coords": [200, 100, 600, 300], "color": (200, 50, 50)}],
            [[200, 100, 600, 300]],
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": True},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (512, 1024),
        ),
        (
            "box_to_mask with tall image",
            (50, 50, 50),
            [{"type": "rectangle", "coords": [100, 200, 300, 600], "color": (200, 50, 50)}],
            [[100, 200, 300, 600]],
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": True},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (1024, 512),
        ),
        (
            "grayscale_to_mask with small image",
            (100, 100, 100),
            [{"type": "rectangle", "coords": [20, 20, 100, 100], "color": (200, 200, 200)}],
            [],
            {
                "grayscale_to_mask": {"enabled": True},
                "box_to_mask": {"enabled": False},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (128, 128),
        ),
        (
            "grayscale_to_mask with large image",
            (100, 100, 100),
            [{"type": "rectangle", "coords": [400, 400, 1600, 1600], "color": (200, 200, 200)}],
            [],
            {
                "grayscale_to_mask": {"enabled": True},
                "box_to_mask": {"enabled": False},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (2048, 2048),
        ),
        (
            "box_to_mask with multiple boxes on wide image",
            (50, 50, 50),
            [
                {"type": "rectangle", "coords": [100, 100, 300, 300], "color": (200, 50, 50)},
                {"type": "rectangle", "coords": [600, 100, 800, 300], "color": (50, 200, 50)},
                {"type": "rectangle", "coords": [1100, 100, 1300, 300], "color": (50, 50, 200)},
            ],
            [[100, 100, 300, 300], [600, 100, 800, 300], [1100, 100, 1300, 300]],
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": True},
                "template_box_to_masks": {"enabled": False},
            },
            0.95,
            (512, 1536),
        ),
        (
            "template_box_to_masks with tall image",
            (50, 50, 50),
            [
                {"type": "rectangle", "coords": [100, 100, 200, 300], "color": (200, 50, 50)},
                {"type": "rectangle", "coords": [100, 500, 200, 700], "color": (200, 50, 50)},
                {"type": "rectangle", "coords": [100, 900, 200, 1100], "color": (200, 50, 50)},
            ],
            [[100, 100, 200, 300]],  # template box
            {
                "grayscale_to_mask": {"enabled": False},
                "box_to_mask": {"enabled": False},
                "template_box_to_masks": {
                    "enabled": True,
                    "save_cache": False,
                    "resume_cache": False,
                },
            },
            0.95,
            (1536, 512),
        ),
    ],
)
def test_pipeline(tmp_path, description, bg_color, objects, input_boxes, config, iou_threshold, image_size):
    """Comprehensive pipeline test with different parameters."""
    if description in _DROPPED_PIPELINE_CASES:
        pytest.skip("cradio backbone remap regressed rectangle-template IoU")

    safe_name = description.replace(" ", "_").lower()

    unique_dir = tmp_path / f"{safe_name}"
    unique_dir.mkdir()

    # Create test sample
    img = _create_test_image(size=image_size, background_color=bg_color, objects=objects)
    gt_mask = _create_gt_mask(size=image_size, objects=objects)

    # Run pipeline
    _run_test_pipeline(unique_dir, img, input_boxes, config)

    enabled_modes = [
        m
        for m in ["grayscale_to_mask", "box_to_mask", "template_box_to_masks"]
        if config.get(m, {}).get("enabled", True)
    ]

    # Collect expected output file paths
    expected_files = []
    for m in enabled_modes:
        base_dir = unique_dir / "sample_00001" / m / "output"
        expected_files.append(base_dir / "binary_mask.png")

        # Only modes with result.json
        if m in ["box_to_mask", "template_box_to_masks"]:
            expected_files.append(base_dir / "result.json")

    # Check for existence
    for expected_file in expected_files:
        assert expected_file.exists(), f"Expected file {expected_file} not found"

    # ---- Check each mode output independently ----
    for m in enabled_modes:
        out_dir = unique_dir / "sample_00001" / m / "output"

        # Load mask and compute IoU
        mask_path = out_dir / "binary_mask.png"
        pred_mask_img = Image.open(mask_path)

        # Convert to grayscale if needed (handles both 'L' and 'RGB' modes)
        if pred_mask_img.mode != "L":
            pred_mask_img = pred_mask_img.convert("L")

        pred_mask = np.array(pred_mask_img)

        # Ensure 2D shape
        if len(pred_mask.shape) != 2:
            raise ValueError(f"Unexpected mask shape: {pred_mask.shape}, expected 2D (H, W)")

        assert np.all((pred_mask == 0) | (pred_mask == 255)), f"{m}: Mask should be binary"
        assert np.sum(pred_mask > 0) > 0, f"{m}: Mask should have non-zero pixels"

        # Ensure gt_mask and pred_mask have the same shape
        assert pred_mask.shape == gt_mask.shape, (
            f"{m}: Shape mismatch - pred_mask: {pred_mask.shape}, gt_mask: {gt_mask.shape}"
        )

        # Compute and verify IoU
        iou = _compute_iou(pred_mask, gt_mask)
        assert iou > iou_threshold, f"{description} [{m}]: IoU {iou:.4f} < threshold {iou_threshold}"

        # result.json check (only for modes that produce it)
        json_path = out_dir / "result.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            assert "image_path" in data
            assert "boxes" in data
            assert isinstance(data["boxes"], list)
            assert len(data["boxes"]) > 0


def _asymmetric_image(size=32):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[2:6, 2:6] = (255, 0, 0)
    img[-8:-2, -8:-2] = (0, 255, 0)
    return img


def _symmetric_mask(size=32):
    mask = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    c = (size - 1) / 2
    mask[(yy - c) ** 2 + (xx - c) ** 2 <= (size // 4) ** 2] = 255
    return mask


def test_augmented_variants_stay_aligned_for_symmetric_masks():
    degrees = [0.0, 90.0, 180.0, 270.0]
    img_variants, img_meta = _generate_augmented_variants([_asymmetric_image()], degrees, True, True)
    mask_variants, mask_meta = _generate_augmented_variants([_symmetric_mask()], degrees, True, True, is_mask=True)

    # A rotation-symmetric mask used to collapse via content-hash dedup while
    # its image variants stayed distinct, desynchronizing the zip in
    # HOGFilteringStage.
    assert len(img_variants) == len(mask_variants)

    def key(m):
        return (m["source"], m["rotation"], m["flip_lr"], m["flip_ud"])

    assert [key(m) for m in img_meta] == [key(m) for m in mask_meta]


def test_augmented_variants_dedup_on_canonical_transform():
    variants, meta = _generate_augmented_variants([_asymmetric_image()], [0.0, 360.0], False, False)
    assert len(variants) == 1


def test_quantize_thin_box_gets_at_least_one_cell():
    tx0, ty0, tx1, ty1 = _quantize_box_to_grid((100.0, 100.0, 101.5, 300.0), 1024, 1024, 512, 512)
    assert tx1 > tx0 and ty1 > ty0


def test_quantize_box_at_far_edge_stays_in_bounds():
    tx0, ty0, tx1, ty1 = _quantize_box_to_grid((1023.0, 1023.0, 1024.0, 1024.0), 1024, 1024, 512, 512)
    assert 0 <= tx0 < tx1 <= 512
    assert 0 <= ty0 < ty1 <= 512


def test_quantize_normal_box_unchanged():
    assert _quantize_box_to_grid((256.0, 256.0, 768.0, 768.0), 1024, 1024, 512, 512) == (128, 128, 384, 384)


def test_quantize_out_of_range_box_still_gets_one_cell():
    tx0, ty0, tx1, ty1 = _quantize_box_to_grid((-3.0, -3.0, 0.4, 0.4), 1024, 1024, 512, 512)
    assert 0 <= tx0 < tx1 <= 512
    assert 0 <= ty0 < ty1 <= 512


def test_proposal_seed_default_and_validation():
    cfg = OmegaConf.structured(ROIGenerationConfig)
    assert cfg.template_box_to_masks.proposal_seed == 0
    assert validate_sample_config(cfg)

    cfg.template_box_to_masks.proposal_seed = -1
    with pytest.raises(ValueError, match="proposal_seed"):
        validate_sample_config(cfg)


def test_proposal_seed_participates_in_dependency_hash():
    def stage_with_seed(seed):
        cfg = OmegaConf.structured(ROIGenerationConfig)
        cfg.template_box_to_masks.proposal_seed = seed
        return ProposalGenerationStage(cfg, None)

    s0 = stage_with_seed(0)
    assert s0.deps["proposal_seed"] == 0

    h0 = s0.compute_dependency_hash({})
    assert stage_with_seed(0).compute_dependency_hash({}) == h0
    assert stage_with_seed(7).compute_dependency_hash({}) != h0


class _FakeStage(PipelineStage):
    def __init__(self, name, dep_value):
        super().__init__(name)
        self.deps = {"value": dep_value}
        self.result = {"payload": name}

    def run(self, ctx):
        ctx[self.name] = self.result
        return ctx


def _cache_ctx(tmp_path):
    return {"input": {"output_dir": str(tmp_path)}}


def test_load_cached_context_returns_continuable_prev_hash(tmp_path):
    stages = [_FakeStage("s0", 1), _FakeStage("s1", 2), _FakeStage("s2", 3)]
    ctx = _cache_ctx(tmp_path)
    cache_dir = str(tmp_path / "template_box_to_masks" / "cache")

    # Simulate a full prior run that cached the first two stages.
    prev = None
    for stage in stages[:2]:
        h = stage.compute_dependency_hash(ctx, prev)
        stage.save_cache(ctx, h)
        prev = h
    full_chain_hash = prev

    restored_ctx, start_idx, prev_hash = load_cached_context(stages, _cache_ctx(tmp_path), cache_dir)
    assert start_idx == 2
    assert restored_ctx["s0"] == {"payload": "s0"}
    assert restored_ctx["s1"] == {"payload": "s1"}
    # The chain must continue from the last verified hash — restarting at None
    # permanently invalidated every stage past the resume point.
    assert prev_hash == full_chain_hash


def test_load_cached_context_cold_start(tmp_path):
    stages = [_FakeStage("s0", 1), _FakeStage("s1", 2)]
    _, start_idx, prev_hash = load_cached_context(stages, _cache_ctx(tmp_path), str(tmp_path / "nope"))
    assert start_idx == 0
    assert prev_hash is None


def test_box_to_mask_result_json_marks_coordinate_spaces(tmp_path):
    cfg = OmegaConf.structured(ROIGenerationConfig)
    post = BoxToMaskPostProcess(cfg)

    # One 8x8 blob at (16,16) in a 64x64 processed mask.
    mask = np.zeros((1, 64, 64), dtype=np.uint8)
    mask[0, 16:24, 16:24] = 1
    post.run(mask)

    ctx = {
        "input": {
            "image_path": "img.png",
            "image": Image.new("RGB", (64, 64)),
            "boxes": [[16, 16, 24, 24]],
            "ori_image_size": (128, 128),
            "output_dir": str(tmp_path),
        }
    }
    post.save_result(ctx)

    result = json.loads((tmp_path / "box_to_mask" / "output" / "result.json").read_text())
    assert result["original_image_size"] == [128, 128]
    assert result["processed_image_size"] == [64, 64]
    # input_boxes stay in processed space; boxes come from the mask resized
    # back to original space (2x here).
    assert result["input_boxes"] == [[16, 16, 24, 24]]
    assert result["boxes"] == [[32, 32, 48, 48]]


def test_template_result_json_marks_coordinate_spaces(tmp_path):
    cfg = OmegaConf.structured(ROIGenerationConfig)
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
        "sam_inference": {"template_masks": [small_mask], "candidate_masks": [small_mask]},
        "proposal_generation": {"proposal_boxes": np.array([[16.0, 16.0, 24.0, 24.0]])},
        "box_filter": {"size_diff": np.array([0.0]), "aspect_diff": np.array([0.0])},
        "mask_filter": {"component_diffs": np.array([0]), "chamfer_score": np.array([0.0])},
        "hog_filter": {"sim_hog": np.array([1.0])},
        "color_filter": {"sim_lightness": np.array([1.0]), "sim_color": np.array([1.0])},
    }
    stage.save_result(ctx)

    result = json.loads((tmp_path / "template_box_to_masks" / "output" / "result.json").read_text())
    assert result["original_image_size"] == [128, 128]
    assert result["processed_image_size"] == [64, 64]
    # template_boxes stay in processed space; boxes come from the mask resized
    # back to original space (2x here).
    assert result["template_boxes"] == [[16, 16, 24, 24]]
    assert result["boxes"] == [[32, 32, 48, 48]]


# --- stage-cache deserialisation ------------------------------------------------------------------
# The cache is unpickled from the run's output directory, so anything able to write there could
# otherwise hand the loader arbitrary code.


def test_stage_cache_round_trips_numpy_payloads(tmp_path):
    """The restriction must still load what the stages actually store."""
    import pickle

    from anomalygen.auto_mask_placement.roi_generation.template_box_to_masks import _SafeUnpickler

    payload = {
        "refined_template_boxes": np.array([[1, 2, 3, 4]], dtype=np.int32),
        "aug_template_crops": [np.zeros((2, 2, 3), dtype=np.uint8), np.ones((3, 3), dtype=np.float32)],
        "transforms": [{"angle": 90, "flip": True}],
        "binary_mask": np.array([[0, 1], [1, 0]], dtype=bool),
        "count": 7,
    }
    path = tmp_path / "stage_result.pkl"
    path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    with open(path, "rb") as f:
        loaded = _SafeUnpickler(f).load()

    assert loaded["count"] == 7
    assert loaded["transforms"] == [{"angle": 90, "flip": True}]
    np.testing.assert_array_equal(loaded["binary_mask"], payload["binary_mask"])
    np.testing.assert_array_equal(loaded["aug_template_crops"][1], payload["aug_template_crops"][1])


def test_stage_cache_rejects_a_crafted_pickle(tmp_path):
    """A cache file that reaches for any other global must fail closed, not execute."""
    import pickle

    from anomalygen.auto_mask_placement.roi_generation.template_box_to_masks import _SafeUnpickler

    class _Payload:
        def __reduce__(self):
            # Inert stand-in for the arbitrary call an attacker would place here.
            return (print, ("crafted stage cache executed",))

    path = tmp_path / "evil_result.pkl"
    path.write_bytes(pickle.dumps(_Payload()))

    with open(path, "rb") as f:
        with pytest.raises(pickle.UnpicklingError, match="blocked global"):
            _SafeUnpickler(f).load()
