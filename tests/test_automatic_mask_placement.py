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

"""Test automatic_mask_placement module."""

import json
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from automatic_mask_placement import (
    AlignmentPoint,
    AugmentationParams,
    AutomaticMaskPlacement,
    BoundingBox,
    MaskAugmentor,
    MaskCropper,
    MaskPlacer,
    PlacementResult,
    ROI,
    ROISeparator,
    validate_binary_mask,
)


# =============================================================================
# Helper Functions
# =============================================================================

def create_test_mask(size=(100, 100), shape="rectangle", position="center"):
    """
    Create a test binary mask with a simple shape.
    
    Args:
        size: (height, width) of the mask
        shape: "rectangle", "circle", or "cross"
        position: "center", "top_left", "bottom_right"
    
    Returns:
        np.ndarray: Binary mask with values 0 and 255
    """
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if position == "center":
        cx, cy = w // 2, h // 2
    elif position == "top_left":
        cx, cy = w // 4, h // 4
    elif position == "bottom_right":
        cx, cy = 3 * w // 4, 3 * h // 4
    else:
        cx, cy = w // 2, h // 2
    
    if shape == "rectangle":
        rect_w, rect_h = w // 4, h // 4
        x1 = max(0, cx - rect_w // 2)
        y1 = max(0, cy - rect_h // 2)
        x2 = min(w, cx + rect_w // 2)
        y2 = min(h, cy + rect_h // 2)
        mask[y1:y2, x1:x2] = 255
        
    elif shape == "circle":
        y, x = np.ogrid[:h, :w]
        r = min(w, h) // 6
        circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        mask[circle_mask] = 255
        
    elif shape == "cross":
        thickness = max(2, min(w, h) // 20)
        arm_length = min(w, h) // 4
        # Horizontal arm
        mask[cy - thickness:cy + thickness, cx - arm_length:cx + arm_length] = 255
        # Vertical arm
        mask[cy - arm_length:cy + arm_length, cx - thickness:cx + thickness] = 255
    
    return mask


def create_roi_mask(size=(512, 512), roi_boxes=None):
    """
    Create a ROI mask with specified bounding boxes.
    
    Args:
        size: (height, width) of the mask
        roi_boxes: List of (x, y, w, h) tuples
    
    Returns:
        np.ndarray: ROI mask with values 0 and 255
    """
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if roi_boxes is None:
        # Default: center rectangle
        roi_boxes = [(w // 4, h // 4, w // 2, h // 2)]
    
    for x, y, bw, bh in roi_boxes:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        mask[y1:y2, x1:x2] = 255
    
    return mask


def create_multi_roi_mask(size=(512, 512), n_rois=3, min_area=1000):
    """
    Create a mask with multiple separated ROI regions.
    
    Args:
        size: (height, width) of the mask
        n_rois: Number of ROI regions
        min_area: Minimum area for each ROI
    
    Returns:
        np.ndarray: ROI mask with multiple connected components
    """
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create n_rois separate regions
    roi_size = int(np.sqrt(min_area))
    spacing = (w - n_rois * roi_size) // (n_rois + 1)
    
    for i in range(n_rois):
        x = spacing + i * (roi_size + spacing)
        y = h // 2 - roi_size // 2
        mask[y:y + roi_size, x:x + roi_size] = 255
    
    return mask


def compute_coverage(placed_mask: np.ndarray, roi_mask: np.ndarray) -> float:
    """
    Compute how much of the placed mask is within the ROI.
    
    Args:
        placed_mask: The placed binary mask
        roi_mask: The ROI binary mask
        
    Returns:
        float: Coverage ratio (1.0 = fully within ROI)
    """
    placed_binary = (placed_mask > 127).astype(np.uint8)
    roi_binary = (roi_mask > 127).astype(np.uint8)
    
    placed_pixels = np.sum(placed_binary)
    if placed_pixels == 0:
        return 0.0
    
    within_roi = np.sum(np.logical_and(placed_binary, roi_binary))
    return float(within_roi) / float(placed_pixels)


# =============================================================================
# Test Data Types
# =============================================================================

class TestBoundingBox:
    """Tests for BoundingBox dataclass"""
    
    def test_bounding_box_creation(self):
        """Test basic BoundingBox creation"""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50
    
    def test_bounding_box_center(self):
        """Test BoundingBox center property"""
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        assert bbox.center == (50, 50)
        
        bbox2 = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox2.center == (60, 45)
    
    def test_bounding_box_area(self):
        """Test BoundingBox area property"""
        bbox = BoundingBox(x=0, y=0, width=100, height=50)
        assert bbox.area == 5000
        
        bbox2 = BoundingBox(x=10, y=20, width=10, height=10)
        assert bbox2.area == 100


class TestAlignmentPoint:
    """Tests for AlignmentPoint enum"""
    
    def test_alignment_point_values(self):
        """Test that all alignment point values are accessible"""
        assert AlignmentPoint.CENTER.value == "center"
        assert AlignmentPoint.TOP_LEFT.value == "top_left"
        assert AlignmentPoint.TOP_RIGHT.value == "top_right"
        assert AlignmentPoint.BOTTOM_LEFT.value == "bottom_left"
        assert AlignmentPoint.BOTTOM_RIGHT.value == "bottom_right"
        assert AlignmentPoint.TOP_CENTER.value == "top_center"
        assert AlignmentPoint.BOTTOM_CENTER.value == "bottom_center"
        assert AlignmentPoint.LEFT_CENTER.value == "left_center"
        assert AlignmentPoint.RIGHT_CENTER.value == "right_center"
        assert AlignmentPoint.RANDOM.value == "random"


class TestROI:
    """Tests for ROI dataclass"""
    
    def test_roi_creation(self):
        """Test basic ROI creation"""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        roi = ROI(bbox=bbox, is_legal=True, roi_id="test_roi")
        
        assert roi.bbox == bbox
        assert roi.is_legal == True
        assert roi.roi_id == "test_roi"
    
    def test_roi_illegal(self):
        """Test ROI with is_legal=False"""
        bbox = BoundingBox(x=0, y=0, width=50, height=50)
        roi = ROI(bbox=bbox, is_legal=False, roi_id="illegal_roi")
        
        assert roi.bbox == bbox
        assert roi.is_legal == False
        assert roi.roi_id == "illegal_roi"


# =============================================================================
# Test AugmentationParams (Config)
# =============================================================================

class TestAugmentationParams:
    """Tests for AugmentationParams configuration"""
    
    def test_default_params(self):
        """Test default parameter values"""
        params = AugmentationParams()
        
        assert params.shift_x_probability == 1.0
        assert params.shift_y_probability == 1.0
        assert params.rotation_probability == 1.0
        assert params.scale_x_probability == 1.0
        assert params.scale_y_probability == 1.0
        assert params.shear_x_probability == 1.0
        assert params.shear_y_probability == 1.0
        assert params.flip_x_probability == 0.5
        assert params.flip_y_probability == 0.5
        assert params.shift_x_range == (-30, 30)
        assert params.shift_y_range == (-30, 30)
        assert params.rotation_range == (-30, 30)
        assert params.scale_x_range == (0.25, 4)
        assert params.scale_y_range == (0.25, 4)
    
    def test_custom_params(self):
        """Test custom parameter values"""
        params = AugmentationParams(
            shift_x_probability=0.5,
            rotation_range=(-10, 10),
            scale_x_range=(0.5, 2.0),
        )
        
        assert params.shift_x_probability == 0.5
        assert params.rotation_range == (-10, 10)
        assert params.scale_x_range == (0.5, 2.0)
    
    def test_custom_ranges_disable_dynamic_mode(self):
        """Test that specifying custom ranges in config file disables corresponding dynamic modes"""
        # Test 1: Specifying shift_x_range disables dynamic shift
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"shift_x_range": [-50, 50]}, f)
            temp_path = f.name
        
        try:
            loaded_params = AugmentationParams.from_config_file(temp_path)
            assert loaded_params.shift_x_range == (-50, 50)
            assert loaded_params.use_dynamic_shift_range == False, \
                "Dynamic shift should be disabled when shift_x_range is specified"
        finally:
            os.unlink(temp_path)
        
        # Test 2: Specifying rotation_range disables dynamic rotation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"rotation_range": [-45, 45]}, f)
            temp_path = f.name
        
        try:
            loaded_params = AugmentationParams.from_config_file(temp_path)
            assert loaded_params.rotation_range == (-45, 45)
            assert loaded_params.use_dynamic_rotation_range == False, \
                "Dynamic rotation should be disabled when rotation_range is specified"
        finally:
            os.unlink(temp_path)
        
        # Test 3: Specifying shear_x_range disables dynamic shear
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"shear_x_range": [-20, 20]}, f)
            temp_path = f.name
        
        try:
            loaded_params = AugmentationParams.from_config_file(temp_path)
            assert loaded_params.shear_x_range == (-20, 20)
            assert loaded_params.use_dynamic_shear_range == False, \
                "Dynamic shear should be disabled when shear_x_range is specified"
        finally:
            os.unlink(temp_path)
        
        # Test 4: Not specifying ranges keeps dynamic modes enabled
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"shift_x_probability": 0.5}, f)  # Only set probability, not range
            temp_path = f.name
        
        try:
            loaded_params = AugmentationParams.from_config_file(temp_path)
            assert loaded_params.use_dynamic_shift_range == True, \
                "Dynamic shift should remain enabled when no range is specified"
            assert loaded_params.use_dynamic_rotation_range == True, \
                "Dynamic rotation should remain enabled when no range is specified"
            assert loaded_params.use_dynamic_shear_range == True, \
                "Dynamic shear should remain enabled when no range is specified"
        finally:
            os.unlink(temp_path)
    
    def test_save_and_load_json(self):
        """Test saving and loading params from JSON"""
        params = AugmentationParams(
            shift_x_probability=0.8,
            shift_y_probability=0.8,
            rotation_range=(-15, 15),
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            params.save_to_config_file(temp_path)
            loaded_params = AugmentationParams.from_config_file(temp_path)
            
            assert loaded_params.shift_x_probability == 0.8
            assert loaded_params.shift_y_probability == 0.8
            assert loaded_params.rotation_range == (-15, 15)
        finally:
            os.unlink(temp_path)
    
    def test_save_and_load_yaml(self):
        """Test saving and loading params from YAML"""
        params = AugmentationParams(
            scale_x_probability=0.7,
            flip_x_probability=0.3,
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            params.save_to_config_file(temp_path)
            loaded_params = AugmentationParams.from_config_file(temp_path)
            
            assert loaded_params.scale_x_probability == 0.7
            assert loaded_params.flip_x_probability == 0.3
        finally:
            os.unlink(temp_path)
    
    def test_invalid_config_key(self):
        """Test that invalid config keys raise an error"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"invalid_key": 123}, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid keys"):
                AugmentationParams.from_config_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_dynamic_operations_report(self):
        """Test dynamic operations report"""
        params = AugmentationParams()
        report = params.report_dynamic_operations()
        assert isinstance(report, str)
        assert len(report) > 0


# =============================================================================
# Test Utils
# =============================================================================

class TestValidateBinaryMask:
    """Tests for validate_binary_mask utility function"""
    
    def test_valid_binary_mask(self):
        """Test with a valid binary mask (0 and 255 only)"""
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        result = validate_binary_mask(mask, "test_mask")
        np.testing.assert_array_equal(result, mask)
    
    def test_all_zeros_mask(self):
        """Test with an all-zeros mask"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = validate_binary_mask(mask, "zero_mask")
        np.testing.assert_array_equal(result, mask)
    
    def test_all_white_mask(self):
        """Test with an all-255 mask"""
        mask = np.full((100, 100), 255, dtype=np.uint8)
        result = validate_binary_mask(mask, "white_mask")
        np.testing.assert_array_equal(result, mask)
    
    def test_normalize_grayscale_mask(self):
        """Test that grayscale values get normalized"""
        mask = np.array([[0, 128, 200], [50, 127, 255]], dtype=np.uint8)
        result = validate_binary_mask(mask, "grayscale_mask")
        
        # Values > 127 become 255, others become 0
        expected = np.array([[0, 255, 255], [0, 0, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)


# =============================================================================
# Test MaskCropper
# =============================================================================

class TestMaskCropper:
    """Tests for MaskCropper class"""
    
    def test_crop_centered_mask(self):
        """Test cropping a mask with content in center"""
        # Create 100x100 mask with 20x20 white region in center
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        
        cropped = MaskCropper.crop_mask_by_max_dimensions(mask)
        
        assert cropped.shape == (20, 20)
        assert np.all(cropped == 255)
    
    def test_crop_corner_mask(self):
        """Test cropping a mask with content in corner"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:20] = 255
        
        cropped = MaskCropper.crop_mask_by_max_dimensions(mask)
        
        assert cropped.shape == (10, 20)
    
    def test_crop_empty_mask(self):
        """Test cropping an empty mask (returns original)"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        cropped = MaskCropper.crop_mask_by_max_dimensions(mask)
        
        # Should return original mask for empty input
        assert cropped.shape == (100, 100)
    
    def test_crop_full_mask(self):
        """Test cropping a fully filled mask"""
        mask = np.full((50, 80), 255, dtype=np.uint8)
        
        cropped = MaskCropper.crop_mask_by_max_dimensions(mask)
        
        assert cropped.shape == (50, 80)


# =============================================================================
# Test MaskAugmentor
# =============================================================================

class TestMaskAugmentor:
    """Tests for MaskAugmentor class"""
    
    def test_augmentor_creation(self):
        """Test MaskAugmentor creation"""
        params = AugmentationParams()
        augmentor = MaskAugmentor(params)
        
        assert augmentor.params == params
        assert augmentor.roi_alignment_point == AlignmentPoint.RANDOM
    
    def test_augmentor_with_custom_alignment(self):
        """Test MaskAugmentor with custom alignment"""
        params = AugmentationParams()
        augmentor = MaskAugmentor(
            params, 
            roi_alignment_point=AlignmentPoint.TOP_LEFT,
            submask_alignment_point=AlignmentPoint.BOTTOM_CENTER
        )
        
        assert augmentor.roi_alignment_point == AlignmentPoint.TOP_LEFT
        assert augmentor.submask_alignment_point == AlignmentPoint.BOTTOM_CENTER
    
    def test_get_submask_fixed_point_center(self):
        """Test fixed point calculation for CENTER alignment"""
        params = AugmentationParams()
        augmentor = MaskAugmentor(params, submask_alignment_point=AlignmentPoint.CENTER)
        
        fixed_point = augmentor.get_submask_fixed_point(h=100, w=200)
        assert fixed_point == (100, 50)  # (w//2, h//2)
    
    def test_get_submask_fixed_point_top_left(self):
        """Test fixed point calculation for TOP_LEFT alignment"""
        params = AugmentationParams()
        augmentor = MaskAugmentor(params, submask_alignment_point=AlignmentPoint.TOP_LEFT)
        
        fixed_point = augmentor.get_submask_fixed_point(h=100, w=200)
        assert fixed_point == (0, 0)
    
    def test_get_submask_fixed_point_bottom_right(self):
        """Test fixed point calculation for BOTTOM_RIGHT alignment"""
        params = AugmentationParams()
        augmentor = MaskAugmentor(params, submask_alignment_point=AlignmentPoint.BOTTOM_RIGHT)
        
        fixed_point = augmentor.get_submask_fixed_point(h=100, w=200)
        assert fixed_point == (199, 99)  # (w-1, h-1)
    
    def test_augment_mask_returns_tuple(self):
        """Test that augment_mask returns (mask, fixed_point) tuple"""
        params = AugmentationParams()
        augmentor = MaskAugmentor(params)
        mask = create_test_mask(size=(50, 50))
        
        result = augmentor.augment_mask(mask)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], tuple)
        assert len(result[1]) == 2
    
    def test_augment_mask_preserves_binary(self):
        """Test that augmented mask remains binary"""
        params = AugmentationParams(
            shift_x_probability=1.0,
            shift_y_probability=1.0,
            rotation_probability=1.0,
            scale_x_probability=1.0,
            scale_y_probability=1.0,
        )
        augmentor = MaskAugmentor(params)
        mask = create_test_mask(size=(50, 50))
        
        augmented, _ = augmentor.augment_mask(mask)
        
        # Check that values are either 0 or close to 255
        unique_values = np.unique(augmented)
        for val in unique_values:
            assert val < 10 or val > 245, f"Unexpected value {val} in augmented mask"
    
    def test_augment_mask_no_augmentation(self):
        """Test augment_mask with all probabilities set to 0"""
        params = AugmentationParams(
            shift_x_probability=0.0,
            shift_y_probability=0.0,
            rotation_probability=0.0,
            scale_x_probability=0.0,
            scale_y_probability=0.0,
            flip_x_probability=0.0,
            flip_y_probability=0.0,
            shear_x_probability=0.0,
            shear_y_probability=0.0,
            morph_probability=0.0,
        )
        augmentor = MaskAugmentor(params)
        mask = create_test_mask(size=(50, 50))
        
        augmented, fixed_point = augmentor.augment_mask(mask)
        
        # With no augmentation, mask should be identical
        np.testing.assert_array_equal(augmented, mask)
        assert fixed_point == (25, 25)  # Center of 50x50 mask
    
    def test_augment_mask_deterministic_with_seed(self):
        """Test that augmentation is deterministic with same random seed"""
        params = AugmentationParams()
        
        mask = create_test_mask(size=(50, 50))
        
        # First run with seed
        np.random.seed(42)
        import random
        random.seed(42)
        augmentor1 = MaskAugmentor(params)
        result1, fp1 = augmentor1.augment_mask(mask.copy())
        
        # Second run with same seed
        np.random.seed(42)
        random.seed(42)
        augmentor2 = MaskAugmentor(params)
        result2, fp2 = augmentor2.augment_mask(mask.copy())
        
        np.testing.assert_array_equal(result1, result2)
        assert fp1 == fp2


# =============================================================================
# Test MaskPlacer
# =============================================================================

class TestMaskPlacer:
    """Tests for MaskPlacer class"""
    
    def test_mask_placer_creation(self):
        """Test MaskPlacer creation"""
        placer = MaskPlacer(image_width=512, image_height=512)
        
        assert placer.image_width == 512
        assert placer.image_height == 512
        assert placer.roi_alignment_point == AlignmentPoint.RANDOM
    
    def test_mask_placer_custom_alignment(self):
        """Test MaskPlacer with custom alignment"""
        placer = MaskPlacer(
            image_width=512, 
            image_height=512,
            roi_alignment_point=AlignmentPoint.TOP_LEFT
        )
        
        assert placer.roi_alignment_point == AlignmentPoint.TOP_LEFT
    
    def test_place_mask_success(self):
        """Test successful mask placement"""
        placer = MaskPlacer(image_width=512, image_height=512)
        
        # Create a simple mask and ROI
        augmented_mask = np.zeros((50, 50), dtype=np.uint8)
        augmented_mask[10:40, 10:40] = 255
        
        roi_mask = create_roi_mask(size=(512, 512), roi_boxes=[(200, 200, 100, 100)])
        
        fixed_point = (25, 25)  # Center of the mask
        
        result = placer.place_mask(augmented_mask, fixed_point, roi_mask)
        
        assert isinstance(result, PlacementResult)
        assert result.success == True
        assert result.pixel_count > 0
        assert result.final_mask is not None
        assert result.final_mask.shape == (512, 512)
    
    def test_place_mask_clipped_by_roi(self):
        """Test mask placement where mask is offset away from ROI, resulting in clipping"""
        placer = MaskPlacer(
            image_width=512, 
            image_height=512,
            roi_alignment_point=AlignmentPoint.CENTER
        )
        
        # Create a 50x50 mask with white pixels
        augmented_mask = np.zeros((50, 50), dtype=np.uint8)
        augmented_mask[10:40, 10:40] = 255  # 30x30 white region
        
        # Small ROI at center of image
        roi_mask = create_roi_mask(size=(512, 512), roi_boxes=[(230, 230, 50, 50)])
        # ROI center is at (255, 255)
        
        # Use a very large fixed_point offset to push the mask away from ROI center
        # If fixed_point = (500, 500), mask will be placed at (255-500, 255-500) = (-245, -245)
        # This is completely outside the image, so clipping will result in 0 pixels
        fixed_point = (500, 500)
        
        result = placer.place_mask(augmented_mask, fixed_point, roi_mask)
        
        # The mask should fail because it's placed completely outside the ROI
        assert isinstance(result, PlacementResult)
        assert result.success == False
        assert result.pixel_count == 0
    
    def test_calculate_alignment_position_center(self):
        """Test alignment position calculation for CENTER"""
        placer = MaskPlacer(
            image_width=512, 
            image_height=512,
            roi_alignment_point=AlignmentPoint.CENTER
        )
        
        roi_mask = create_roi_mask(size=(512, 512), roi_boxes=[(100, 100, 200, 200)])
        fixed_point = (25, 25)
        
        pos = placer.calculate_alignment_position(roi_mask, 50, 50, fixed_point)
        
        # ROI pixels are from (100,100) to (299,299), center = (199, 199)
        # Position = center - fixed_point = (199-25, 199-25) = (174, 174)
        assert pos == (174, 174)
    
    def test_calculate_alignment_position_top_left(self):
        """Test alignment position calculation for TOP_LEFT"""
        placer = MaskPlacer(
            image_width=512, 
            image_height=512,
            roi_alignment_point=AlignmentPoint.TOP_LEFT
        )
        
        roi_mask = create_roi_mask(size=(512, 512), roi_boxes=[(100, 100, 200, 200)])
        fixed_point = (0, 0)
        
        pos = placer.calculate_alignment_position(roi_mask, 50, 50, fixed_point)
        
        # ROI top-left is at (100, 100), fixed point is (0,0), so position is (100, 100)
        assert pos == (100, 100)


# =============================================================================
# Test ROISeparator
# =============================================================================

class TestROISeparator:
    """Tests for ROISeparator class"""
    
    def test_roi_separator_creation(self):
        """Test ROISeparator creation"""
        separator = ROISeparator(image_width=512, image_height=512)
        
        assert separator.image_width == 512
        assert separator.image_height == 512
        assert len(separator.rois) == 0
    
    def test_add_roi(self):
        """Test adding a ROI"""
        separator = ROISeparator(image_width=512, image_height=512)
        
        bbox = BoundingBox(x=100, y=100, width=200, height=200)
        roi = separator.add_roi(bbox, is_legal=True, roi_id="roi_1")
        
        assert len(separator.rois) == 1
        assert roi.roi_id == "roi_1"
        assert roi.is_legal == True
    
    def test_add_roi_auto_id(self):
        """Test adding ROI with auto-generated ID"""
        separator = ROISeparator(image_width=512, image_height=512)
        
        bbox = BoundingBox(x=100, y=100, width=100, height=100)
        roi = separator.add_roi(bbox, is_legal=True)
        
        assert roi.roi_id == "roi_0"
    
    def test_add_roi_out_of_bounds(self):
        """Test that out-of-bounds ROI raises an error"""
        separator = ROISeparator(image_width=512, image_height=512)
        
        # ROI that extends beyond image bounds
        bbox = BoundingBox(x=500, y=500, width=100, height=100)
        
        with pytest.raises(ValueError, match="out of image bounds"):
            separator.add_roi(bbox, is_legal=True)
    
    def test_get_legal_rois(self):
        """Test filtering legal ROIs"""
        separator = ROISeparator(image_width=512, image_height=512)
        
        bbox1 = BoundingBox(x=100, y=100, width=100, height=100)
        bbox2 = BoundingBox(x=300, y=100, width=100, height=100)
        bbox3 = BoundingBox(x=100, y=300, width=100, height=100)
        
        separator.add_roi(bbox1, is_legal=True, roi_id="legal_1")
        separator.add_roi(bbox2, is_legal=False, roi_id="illegal_1")
        separator.add_roi(bbox3, is_legal=True, roi_id="legal_2")
        
        legal_rois = separator.get_legal_rois()
        
        assert len(legal_rois) == 2
        assert all(roi.is_legal for roi in legal_rois)
    
    def test_select_random_roi(self):
        """Test random ROI selection"""
        separator = ROISeparator(image_width=512, image_height=512)
        
        bbox1 = BoundingBox(x=100, y=100, width=100, height=100)
        bbox2 = BoundingBox(x=300, y=100, width=100, height=100)
        
        separator.add_roi(bbox1, is_legal=True, roi_id="roi_1")
        separator.add_roi(bbox2, is_legal=True, roi_id="roi_2")
        
        selected = separator.select_random_roi()
        
        assert selected is not None
        assert selected.roi_id in ["roi_1", "roi_2"]
    
    def test_select_random_roi_no_legal(self):
        """Test random selection when no legal ROIs exist"""
        separator = ROISeparator(image_width=512, image_height=512)
        
        bbox = BoundingBox(x=100, y=100, width=100, height=100)
        separator.add_roi(bbox, is_legal=False, roi_id="illegal")
        
        selected = separator.select_random_roi()
        
        assert selected is None
    
    def test_separate_connected_regions(self):
        """Test separation of connected components"""
        separator = ROISeparator(image_width=512, image_height=512)
        
        # Create mask with 3 separate regions
        roi_img = create_multi_roi_mask(size=(512, 512), n_rois=3, min_area=1000)
        
        separated = separator.separate_connected_regions(roi_img, min_area=100)
        
        assert len(separated) == 3
        for mask in separated:
            assert mask.shape == (512, 512)
            assert np.sum(mask > 0) > 0
    
    def test_separate_connected_regions_min_area_filter(self):
        """Test that small regions are filtered out"""
        separator = ROISeparator(image_width=512, image_height=512)
        
        # Create mask with 3 regions, but one is small
        roi_img = np.zeros((512, 512), dtype=np.uint8)
        roi_img[100:150, 100:150] = 255  # 50x50 = 2500 pixels
        roi_img[100:150, 300:350] = 255  # 50x50 = 2500 pixels
        roi_img[300:305, 300:305] = 255  # 5x5 = 25 pixels (small)
        
        separated = separator.separate_connected_regions(roi_img, min_area=100)
        
        # Should only have 2 regions (small one filtered out)
        assert len(separated) == 2
    
    def test_create_separated_roi_masks_from_json(self):
        """Test creating ROI masks from JSON definitions"""
        separator = ROISeparator(image_width=512, image_height=512)
        
        # Add legal and illegal ROIs
        bbox1 = BoundingBox(x=100, y=100, width=100, height=100)
        bbox2 = BoundingBox(x=300, y=100, width=100, height=100)
        bbox3 = BoundingBox(x=200, y=300, width=50, height=50)
        
        separator.add_roi(bbox1, is_legal=True, roi_id="legal_1")
        separator.add_roi(bbox2, is_legal=True, roi_id="legal_2")
        separator.add_roi(bbox3, is_legal=False, roi_id="illegal_1")
        
        legal_masks = separator.create_separated_roi_masks_from_json()
        
        assert len(legal_masks) == 2
        assert len(separator.original_legal_masks) == 2
        assert separator.illegal_regions_mask is not None
        
        # Check illegal mask has pixels in the right place
        illegal_pixel_count = np.sum(separator.illegal_regions_mask[300:350, 200:250] > 0)
        assert illegal_pixel_count > 0


# =============================================================================
# Test AutomaticMaskPlacement (Integration Tests)
# =============================================================================

class TestAutomaticMaskPlacement:
    """Integration tests for AutomaticMaskPlacement main class"""
    
    def test_amp_creation(self):
        """Test AutomaticMaskPlacement creation"""
        amp = AutomaticMaskPlacement(
            image_width=512,
            image_height=512,
            augmentation_params=AugmentationParams(),
            roi_alignment_point=AlignmentPoint.CENTER,
            random_seed=42
        )
        
        assert amp.image_width == 512
        assert amp.image_height == 512
    
    def test_amp_with_custom_params(self):
        """Test AMP with custom augmentation params"""
        params = AugmentationParams(
            shift_x_probability=0.5,
            rotation_range=(-10, 10)
        )
        
        amp = AutomaticMaskPlacement(
            image_width=1024,
            image_height=768,
            augmentation_params=params,
            random_seed=123
        )
        
        assert amp.image_width == 1024
        assert amp.image_height == 768
    
    def test_amp_load_combined_rois_from_image(self):
        """Test loading ROIs from an image file"""
        amp = AutomaticMaskPlacement(
            image_width=512,
            image_height=512,
            augmentation_params=AugmentationParams(),
            random_seed=42
        )
        
        # Create a temporary ROI image
        roi_mask = create_roi_mask(size=(512, 512), roi_boxes=[(100, 100, 200, 200)])
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            Image.fromarray(roi_mask).save(temp_path)
            amp.load_combined_rois(roi_image_paths=[temp_path])
            
            # Verify ROIs were loaded
            assert len(amp.roi_separator.separated_roi_masks) > 0
        finally:
            os.unlink(temp_path)
    
    def test_amp_process_submask(self):
        """Test processing a submask to generate placements"""
        amp = AutomaticMaskPlacement(
            image_width=512,
            image_height=512,
            augmentation_params=AugmentationParams(),
            random_seed=42
        )
        
        # Create temporary files
        roi_mask = create_roi_mask(size=(512, 512), roi_boxes=[(100, 100, 300, 300)])
        submask = create_test_mask(size=(50, 50), shape="rectangle")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            roi_path = os.path.join(tmpdir, 'roi.png')
            submask_path = os.path.join(tmpdir, 'submask.png')
            output_dir = os.path.join(tmpdir, 'output')
            
            Image.fromarray(roi_mask).save(roi_path)
            Image.fromarray(submask).save(submask_path)
            
            amp.load_combined_rois(roi_image_paths=[roi_path])
            
            # Process and generate masks
            results = amp.process_submask(
                submask_path=submask_path,
                n_instances=3,
                output_dir=output_dir
            )
            
            # Check results
            assert len(results) > 0
            
            # Check output files exist
            assert os.path.exists(output_dir)
            output_files = os.listdir(output_dir)
            assert len(output_files) > 0
    
    def test_amp_deterministic_with_seed(self):
        """Test that AMP produces deterministic results with same seed"""
        params = AugmentationParams()
        
        # Create test data
        roi_mask = create_roi_mask(size=(256, 256), roi_boxes=[(50, 50, 150, 150)])
        submask = create_test_mask(size=(30, 30), shape="rectangle")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            roi_path = os.path.join(tmpdir, 'roi.png')
            submask_path = os.path.join(tmpdir, 'submask.png')
            output_dir1 = os.path.join(tmpdir, 'output1')
            output_dir2 = os.path.join(tmpdir, 'output2')
            
            Image.fromarray(roi_mask).save(roi_path)
            Image.fromarray(submask).save(submask_path)
            
            # First run
            amp1 = AutomaticMaskPlacement(
                image_width=256,
                image_height=256,
                augmentation_params=params,
                random_seed=42
            )
            amp1.load_combined_rois(roi_image_paths=[roi_path])
            results1 = amp1.process_submask(
                submask_path=submask_path,
                n_instances=2,
                output_dir=output_dir1
            )
            
            # Second run with same seed
            amp2 = AutomaticMaskPlacement(
                image_width=256,
                image_height=256,
                augmentation_params=params,
                random_seed=42
            )
            amp2.load_combined_rois(roi_image_paths=[roi_path])
            results2 = amp2.process_submask(
                submask_path=submask_path,
                n_instances=2,
                output_dir=output_dir2
            )
            
            # Results should be identical
            assert len(results1) == len(results2)
            
            # Compare generated masks
            for i in range(len(results1)):
                mask1 = np.array(Image.open(results1[i]['output_path']))
                mask2 = np.array(Image.open(results2[i]['output_path']))
                np.testing.assert_array_equal(mask1, mask2)


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""
    
    def test_empty_roi(self):
        """Test handling of empty ROI mask"""
        separator = ROISeparator(image_width=512, image_height=512)
        
        empty_roi = np.zeros((512, 512), dtype=np.uint8)
        separated = separator.separate_connected_regions(empty_roi)
        
        assert len(separated) == 0
    
    def test_full_image_roi(self):
        """Test ROI covering entire image"""
        separator = ROISeparator(image_width=512, image_height=512)
        
        full_roi = np.full((512, 512), 255, dtype=np.uint8)
        separated = separator.separate_connected_regions(full_roi)
        
        assert len(separated) == 1
        assert np.sum(separated[0] > 0) == 512 * 512
    
    def test_single_pixel_mask(self):
        """Test mask with single white pixel"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255
        
        cropped = MaskCropper.crop_mask_by_max_dimensions(mask)
        
        assert cropped.shape == (1, 1)
        assert cropped[0, 0] == 255
    
    def test_mask_at_image_boundary(self):
        """Test placing mask at image boundaries"""
        placer = MaskPlacer(image_width=100, image_height=100)
        
        # Mask larger than it should fit
        large_mask = np.full((50, 50), 255, dtype=np.uint8)
        
        # ROI at corner
        roi_mask = create_roi_mask(size=(100, 100), roi_boxes=[(0, 0, 30, 30)])
        
        fixed_point = (25, 25)
        result = placer.place_mask(large_mask, fixed_point, roi_mask)
        
        # Should still work (with clipping)
        assert isinstance(result, PlacementResult)
        assert result.final_mask.shape == (100, 100)
    
    def test_very_small_submask(self):
        """Test with very small submask"""
        params = AugmentationParams(
            scale_x_probability=0.0,  # Don't scale
            scale_y_probability=0.0,
            shift_x_probability=0.0,
            shift_y_probability=0.0,
            rotation_probability=0.0,
        )
        augmentor = MaskAugmentor(params)
        
        tiny_mask = np.array([[255]], dtype=np.uint8)
        augmented, fixed_point = augmentor.augment_mask(tiny_mask)
        
        assert augmented.shape[0] >= 1
        assert augmented.shape[1] >= 1
    
    def test_non_square_dimensions(self):
        """Test with non-square image dimensions"""
        amp = AutomaticMaskPlacement(
            image_width=1920,
            image_height=1080,
            augmentation_params=AugmentationParams(),
            random_seed=42
        )
        
        assert amp.image_width == 1920
        assert amp.image_height == 1080
        
        # ROI separator should handle non-square
        assert amp.roi_separator.image_width == 1920
        assert amp.roi_separator.image_height == 1080


# =============================================================================
# Parametrized Tests - Alignment Modes
# =============================================================================

class TestAlignmentModes:
    """Parametrized tests for different alignment modes"""
    
    @pytest.mark.parametrize("alignment_point,expected_region", [
        (AlignmentPoint.CENTER, "center"),
        (AlignmentPoint.TOP_LEFT, "top_left"),
        (AlignmentPoint.TOP_RIGHT, "top_right"),
        (AlignmentPoint.BOTTOM_LEFT, "bottom_left"),
        (AlignmentPoint.BOTTOM_RIGHT, "bottom_right"),
        (AlignmentPoint.TOP_CENTER, "top_center"),
        (AlignmentPoint.BOTTOM_CENTER, "bottom_center"),
        (AlignmentPoint.LEFT_CENTER, "left_center"),
        (AlignmentPoint.RIGHT_CENTER, "right_center"),
    ])
    def test_mask_placer_alignment_modes(self, alignment_point, expected_region):
        """Test that different alignment modes place masks in correct regions"""
        placer = MaskPlacer(
            image_width=512,
            image_height=512,
            roi_alignment_point=alignment_point
        )
        
        # Create a small mask
        augmented_mask = np.zeros((30, 30), dtype=np.uint8)
        augmented_mask[5:25, 5:25] = 255
        
        # Create ROI in center of image
        roi_mask = create_roi_mask(size=(512, 512), roi_boxes=[(156, 156, 200, 200)])
        
        # Use center of mask as fixed point
        fixed_point = (15, 15)
        
        result = placer.place_mask(augmented_mask, fixed_point, roi_mask)
        
        assert result.success == True
        assert result.pixel_count > 0
        
        # Verify the mask was placed (has non-zero pixels)
        assert np.sum(result.final_mask > 0) > 0
    
    @pytest.mark.parametrize("submask_alignment,expected_fixed_point", [
        (AlignmentPoint.CENTER, (50, 50)),
        (AlignmentPoint.TOP_LEFT, (0, 0)),
        (AlignmentPoint.TOP_RIGHT, (99, 0)),
        (AlignmentPoint.BOTTOM_LEFT, (0, 99)),
        (AlignmentPoint.BOTTOM_RIGHT, (99, 99)),
        (AlignmentPoint.TOP_CENTER, (50, 0)),
        (AlignmentPoint.BOTTOM_CENTER, (50, 99)),
        (AlignmentPoint.LEFT_CENTER, (0, 50)),
        (AlignmentPoint.RIGHT_CENTER, (99, 50)),
    ])
    def test_submask_fixed_point_calculation(self, submask_alignment, expected_fixed_point):
        """Test fixed point calculation for different submask alignments"""
        params = AugmentationParams()
        augmentor = MaskAugmentor(params, submask_alignment_point=submask_alignment)
        
        fixed_point = augmentor.get_submask_fixed_point(h=100, w=100)
        
        assert fixed_point == expected_fixed_point


# =============================================================================
# Parametrized Tests - Augmentation Operations
# =============================================================================

class TestAugmentationOperations:
    """Parametrized tests for different augmentation configurations"""
    
    @pytest.mark.parametrize("operation,param_name,param_value", [
        ("shift_x", "shift_x_probability", 1.0),
        ("shift_y", "shift_y_probability", 1.0),
        ("rotation", "rotation_probability", 1.0),
        ("scale_x", "scale_x_probability", 1.0),
        ("scale_y", "scale_y_probability", 1.0),
        ("flip_x", "flip_x_probability", 1.0),
        ("flip_y", "flip_y_probability", 1.0),
        ("shear_x", "shear_x_probability", 1.0),
        ("shear_y", "shear_y_probability", 1.0),
        ("morph", "morph_probability", 1.0),
    ])
    def test_individual_augmentation_operations(self, operation, param_name, param_value):
        """Test that individual augmentation operations work correctly"""
        # Disable all operations first
        base_params = {
            "shift_x_probability": 0.0,
            "shift_y_probability": 0.0,
            "rotation_probability": 0.0,
            "scale_x_probability": 0.0,
            "scale_y_probability": 0.0,
            "flip_x_probability": 0.0,
            "flip_y_probability": 0.0,
            "shear_x_probability": 0.0,
            "shear_y_probability": 0.0,
            "morph_probability": 0.0,
        }
        # Enable only the operation we're testing
        base_params[param_name] = param_value
        
        params = AugmentationParams(**base_params)
        augmentor = MaskAugmentor(params)
        
        mask = create_test_mask(size=(50, 50), shape="rectangle")
        augmented, fixed_point = augmentor.augment_mask(mask)
        
        # Check that augmentation produced valid output
        assert isinstance(augmented, np.ndarray)
        assert augmented.dtype == np.uint8
        assert isinstance(fixed_point, tuple)
        assert len(fixed_point) == 2
    
    @pytest.mark.parametrize("scale_range,expected_behavior", [
        ((0.5, 0.5), "shrink"),
        ((2.0, 2.0), "grow"),
        ((1.0, 1.0), "same"),
    ])
    def test_scale_ranges(self, scale_range, expected_behavior):
        """Test different scale ranges"""
        params = AugmentationParams(
            shift_x_probability=0.0,
            shift_y_probability=0.0,
            rotation_probability=0.0,
            scale_x_probability=1.0,
            scale_y_probability=1.0,
            scale_x_range=scale_range,
            scale_y_range=scale_range,
            flip_x_probability=0.0,
            flip_y_probability=0.0,
            shear_x_probability=0.0,
            shear_y_probability=0.0,
            morph_probability=0.0,
        )
        augmentor = MaskAugmentor(params)
        
        mask = create_test_mask(size=(50, 50), shape="rectangle")
        original_pixels = np.sum(mask > 127)
        
        augmented, _ = augmentor.augment_mask(mask)
        augmented_pixels = np.sum(augmented > 127)
        
        if expected_behavior == "shrink":
            assert augmented_pixels < original_pixels
        elif expected_behavior == "grow":
            assert augmented_pixels > original_pixels
        # For "same", pixel count might vary slightly due to interpolation


# =============================================================================
# Parametrized Tests - Image Sizes
# =============================================================================

class TestImageSizes:
    """Parametrized tests for different image sizes"""
    
    @pytest.mark.parametrize("image_size", [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (640, 480),   # Non-square
        (1920, 1080), # HD
    ])
    def test_different_image_sizes(self, image_size):
        """Test AMP with different image sizes"""
        w, h = image_size
        
        amp = AutomaticMaskPlacement(
            image_width=w,
            image_height=h,
            augmentation_params=AugmentationParams(),
            random_seed=42
        )
        
        assert amp.image_width == w
        assert amp.image_height == h
        assert amp.roi_separator.image_width == w
        assert amp.roi_separator.image_height == h
    
    @pytest.mark.parametrize("submask_size,roi_size", [
        ((20, 20), (100, 100)),   # Small submask, medium ROI
        ((50, 50), (200, 200)),   # Medium submask, large ROI
        ((10, 10), (50, 50)),     # Tiny submask, small ROI
        ((30, 60), (100, 200)),   # Non-square submask and ROI
    ])
    def test_different_mask_and_roi_sizes(self, submask_size, roi_size):
        """Test placement with different submask and ROI size combinations"""
        submask_h, submask_w = submask_size
        roi_w, roi_h = roi_size
        
        placer = MaskPlacer(image_width=512, image_height=512)
        
        # Create submask
        submask = np.zeros((submask_h, submask_w), dtype=np.uint8)
        submask[2:-2, 2:-2] = 255
        
        # Create ROI
        roi_mask = create_roi_mask(
            size=(512, 512), 
            roi_boxes=[(256 - roi_w // 2, 256 - roi_h // 2, roi_w, roi_h)]
        )
        
        fixed_point = (submask_w // 2, submask_h // 2)
        result = placer.place_mask(submask, fixed_point, roi_mask)
        
        assert isinstance(result, PlacementResult)
        assert result.final_mask.shape == (512, 512)


# =============================================================================
# Integration Tests with IoU Verification
# =============================================================================

class TestIntegrationWithIoU:
    """Integration tests with IoU and coverage verification"""
    
    @pytest.mark.parametrize("n_instances", [1, 3, 5, 10])
    def test_amp_generates_combined_mask(self, n_instances):
        """Test that AMP generates a single combined mask with n_instances placements
        
        Note: process_submask returns 1 result (a single image containing n_instances masks),
        not n_instances separate results. The number of results is controlled by seed/batch processing.
        """
        params = AugmentationParams(
            shift_x_probability=0.5,
            shift_y_probability=0.5,
            rotation_probability=0.5,
            scale_x_probability=0.5,
            scale_y_probability=0.5,
            scale_x_range=(0.8, 1.2),
            scale_y_range=(0.8, 1.2),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            roi_mask = create_roi_mask(size=(512, 512), roi_boxes=[(100, 100, 300, 300)])
            submask = create_test_mask(size=(40, 40), shape="rectangle")
            
            roi_path = os.path.join(tmpdir, 'roi.png')
            submask_path = os.path.join(tmpdir, 'submask.png')
            output_dir = os.path.join(tmpdir, 'output')
            
            Image.fromarray(roi_mask).save(roi_path)
            Image.fromarray(submask).save(submask_path)
            
            amp = AutomaticMaskPlacement(
                image_width=512,
                image_height=512,
                augmentation_params=params,
                random_seed=42
            )
            amp.load_combined_rois(roi_image_paths=[roi_path])
            
            results = amp.process_submask(
                submask_path=submask_path,
                n_instances=n_instances,
                output_dir=output_dir
            )
            
            # process_submask returns 1 result (single combined image)
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            
            # Verify the result contains metadata
            assert 'output_path' in results[0]
            assert 'n_instances' in results[0]
            assert results[0]['n_instances'] == n_instances
            
            # Verify output file exists and has content
            output_path = results[0]['output_path']
            assert os.path.exists(output_path)
            
            placed_mask = np.array(Image.open(output_path))
            pixel_count = np.sum(placed_mask > 127)
            assert pixel_count > 0, "Combined mask should have non-zero pixels"
    
    def test_placed_masks_within_roi(self):
        """Test that placed masks are fully within the ROI region"""
        params = AugmentationParams(
            shift_x_probability=0.5,
            shift_y_probability=0.5,  # No shift to ensure placement within ROI
            rotation_probability=0.5,
            scale_x_probability=0.5,
            scale_y_probability=0.5,
            scale_x_range=(0.5, 1.0),  # Only shrink
            scale_y_range=(0.5, 1.0),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data - large ROI to ensure mask fits
            roi_mask = create_roi_mask(size=(512, 512), roi_boxes=[(50, 50, 400, 400)])
            submask = create_test_mask(size=(30, 30), shape="rectangle")
            
            roi_path = os.path.join(tmpdir, 'roi.png')
            submask_path = os.path.join(tmpdir, 'submask.png')
            output_dir = os.path.join(tmpdir, 'output')
            
            Image.fromarray(roi_mask).save(roi_path)
            Image.fromarray(submask).save(submask_path)
            
            amp = AutomaticMaskPlacement(
                image_width=512,
                image_height=512,
                augmentation_params=params,
                random_seed=42
            )
            amp.load_combined_rois(roi_image_paths=[roi_path])
            
            results = amp.process_submask(
                submask_path=submask_path,
                n_instances=5,
                output_dir=output_dir
            )
            
            # Verify each placed mask is within ROI
            for result in results:
                placed_mask = np.array(Image.open(result['output_path']))
                coverage = compute_coverage(placed_mask, roi_mask)
                
                # All placed pixels should be within ROI (coverage = 1.0)
                assert coverage == 1.0, f"Mask not fully within ROI, coverage: {coverage}"
    
    def test_masks_have_sufficient_pixels(self):
        """Test that generated masks have sufficient pixel count"""
        params = AugmentationParams(
            scale_x_probability=0.5,
            scale_y_probability=0.5,
            scale_x_range=(0.5, 2.0),
            scale_y_range=(0.5, 2.0),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            roi_mask = create_roi_mask(size=(512, 512), roi_boxes=[(100, 100, 300, 300)])
            submask = create_test_mask(size=(50, 50), shape="rectangle")
            
            roi_path = os.path.join(tmpdir, 'roi.png')
            submask_path = os.path.join(tmpdir, 'submask.png')
            output_dir = os.path.join(tmpdir, 'output')
            
            Image.fromarray(roi_mask).save(roi_path)
            Image.fromarray(submask).save(submask_path)
            
            amp = AutomaticMaskPlacement(
                image_width=512,
                image_height=512,
                augmentation_params=params,
                random_seed=42
            )
            amp.load_combined_rois(roi_image_paths=[roi_path])
            
            results = amp.process_submask(
                submask_path=submask_path,
                n_instances=5,
                output_dir=output_dir
            )
            
            min_pixel_count = 10  # Minimum acceptable pixel count
            
            for result in results:
                placed_mask = np.array(Image.open(result['output_path']))
                pixel_count = np.sum(placed_mask > 127)
                
                assert pixel_count >= min_pixel_count, \
                    f"Mask has too few pixels: {pixel_count} < {min_pixel_count}"


# =============================================================================
# Parametrized Pipeline Tests
# =============================================================================

@pytest.mark.parametrize(
    "description,image_size,roi_config,submask_config,aug_config,n_instances",
    [
        # Basic placement tests
        (
            "basic_center_alignment_small_mask",
            (256, 256),
            {"boxes": [(50, 50, 150, 150)]},
            {"size": (30, 30), "shape": "rectangle"},
            {"shift_x_probability": 0.0, "shift_y_probability": 0.0, "scale_x_probability": 0.0, "scale_y_probability": 0.0,},
            5,
        ),
        (
            "basic_center_alignment_circle_mask",
            (256, 256),
            {"boxes": [(50, 50, 150, 150)]},
            {"size": (40, 40), "shape": "circle"},
            {"shift_x_probability": 0.0, "shift_y_probability": 0.0, "scale_x_probability": 0.0, "scale_y_probability": 0.0,},
            5,
        ),
        # With augmentation
        (
            "with_shift_augmentation",
            (512, 512),
            {"boxes": [(100, 100, 300, 300)]},
            {"size": (40, 40), "shape": "rectangle"},
            {"shift_x_probability": 1.0, "shift_y_probability": 1.0, "shift_x_range": (-20, 20), "shift_y_range": (-20, 20)},
            5,
        ),
        (
            "with_scale_augmentation",
            (512, 512),
            {"boxes": [(100, 100, 300, 300)]},
            {"size": (40, 40), "shape": "rectangle"},
            {"scale_x_probability": 1.0, "scale_y_probability": 1.0, "scale_x_range": (0.5, 1.5), "scale_y_range": (0.5, 1.5)},
            5,
        ),
        (
            "with_rotation_augmentation",
            (512, 512),
            {"boxes": [(100, 100, 300, 300)]},
            {"size": (40, 40), "shape": "rectangle"},
            {"rotation_probability": 1.0, "rotation_range": (-45, 45)},
            5,
        ),
        # Multiple ROIs
        (
            "multiple_rois",
            (512, 512),
            {"boxes": [(50, 50, 100, 100), (300, 50, 100, 100), (150, 300, 100, 100)]},
            {"size": (30, 30), "shape": "rectangle"},
            {"shift_x_probability": 0.0, "shift_y_probability": 0.0},
            10,
        ),
        # Large image
        (
            "large_image_hd",
            (1920, 1080),
            {"boxes": [(200, 200, 600, 400)]},
            {"size": (50, 50), "shape": "rectangle"},
            {"scale_x_probability": 0.5, "scale_y_probability": 0.5, "rotation_probability": 0.5},
            5,
        ),
        # Non-square image
        (
            "non_square_image",
            (640, 480),
            {"boxes": [(100, 100, 300, 200)]},
            {"size": (40, 40), "shape": "rectangle"},
            {},
            5,
        ),
        # Full augmentation pipeline
        (
            "full_augmentation_pipeline",
            (512, 512),
            {"boxes": [(100, 100, 300, 300)]},
            {"size": (40, 40), "shape": "rectangle"},
            {
                "shift_x_probability": 0.8,
                "shift_y_probability": 0.8,
                "rotation_probability": 0.8,
                "scale_x_probability": 0.8,
                "scale_y_probability": 0.8,
                "flip_x_probability": 0.5,
                "flip_y_probability": 0.5,
            },
            10,
        ),
    ],
)
def test_amp_pipeline(
    description, image_size, roi_config, submask_config, aug_config, n_instances
):
    """Comprehensive parametrized pipeline test for AutomaticMaskPlacement.
    
    Note: process_submask returns 1 result (a single combined image containing n_instances masks).
    The number of results is controlled by seed/batch processing in the script, not here.
    """
    width, height = image_size
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create ROI mask
        roi_mask = create_roi_mask(size=(height, width), roi_boxes=roi_config["boxes"])
        
        # Create submask
        submask = create_test_mask(
            size=submask_config["size"],
            shape=submask_config["shape"]
        )
        
        # Save test files
        roi_path = os.path.join(tmpdir, 'roi.png')
        submask_path = os.path.join(tmpdir, 'submask.png')
        output_dir = os.path.join(tmpdir, 'output')
        
        Image.fromarray(roi_mask).save(roi_path)
        Image.fromarray(submask).save(submask_path)
        
        # Create augmentation params
        params = AugmentationParams(**aug_config)
        
        # Run AMP
        amp = AutomaticMaskPlacement(
            image_width=width,
            image_height=height,
            augmentation_params=params,
            random_seed=42
        )
        amp.load_combined_rois(roi_image_paths=[roi_path])
        
        results = amp.process_submask(
            submask_path=submask_path,
            n_instances=n_instances,
            output_dir=output_dir
        )
        
        # Verify that exactly 1 result is returned (single combined image)
        assert len(results) == 1, f"{description}: Expected 1 result, got {len(results)}"
        
        result = results[0]
        
        # Verify result metadata
        assert 'output_path' in result, f"{description}: Missing 'output_path' in result"
        assert 'n_instances' in result, f"{description}: Missing 'n_instances' in result"
        assert result['n_instances'] == n_instances, \
            f"{description}: n_instances mismatch: {result['n_instances']} != {n_instances}"
        
        # Verify output file exists
        output_path = result['output_path']
        assert os.path.exists(output_path), f"{description}: Output file not found: {output_path}"
        
        # Load and verify mask
        placed_mask = np.array(Image.open(output_path))
        assert placed_mask.shape == (height, width), \
            f"{description}: Unexpected mask shape: {placed_mask.shape}, expected ({height}, {width})"
        
        # Verify mask is binary
        unique_values = np.unique(placed_mask)
        assert all(v in [0, 255] for v in unique_values), \
            f"{description}: Mask is not binary: {unique_values}"
        
        # Verify mask has pixels (at least some placements succeeded)
        pixel_count = np.sum(placed_mask > 0)
        assert pixel_count > 0, f"{description}: Combined mask has no white pixels"
        
        # Verify mask is within ROI
        coverage = compute_coverage(placed_mask, roi_mask)
        assert coverage == 1.0, f"{description}: Mask not fully within ROI: coverage={coverage:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

