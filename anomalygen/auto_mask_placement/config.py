# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration classes for Automatic Mask Placement.

This module contains:
- AugmentationParams: Parameters for mask augmentation
"""

import inspect
import json
from dataclasses import asdict, dataclass, field, replace
from dataclasses import fields as dataclass_fields
from typing import List, Tuple

import yaml


@dataclass
class AugmentationParams:
    """Parameters for mask augmentation"""

    # Individual operation probabilities
    shift_x_probability: float = 1.0  # Probability of applying shift in X direction
    shift_y_probability: float = 1.0  # Probability of applying shift in Y direction

    rotation_probability: float = 1.0  # Probability of applying rotation

    scale_x_probability: float = 1.0  # Probability of applying scale in X direction
    scale_y_probability: float = 1.0  # Probability of applying scale in Y direction

    flip_x_probability: float = 0.5  # Probability of applying horizontal flip
    flip_y_probability: float = 0.5  # Probability of applying vertical flip

    shear_x_probability: float = 1.0  # Probability of applying shear in X direction
    shear_y_probability: float = 1.0  # Probability of applying shear in Y direction

    morph_probability: float = 1.0  # Probability of applying morphological operations

    # Operation parameters
    shift_x_range: Tuple[int, int] = (-30, 30)  # X-axis shift range (left, right)
    shift_y_range: Tuple[int, int] = (-30, 30)  # Y-axis shift range (up, down)
    rotation_range: Tuple[float, float] = (-30, 30)  # (min_angle, max_angle) in degrees
    scale_x_range: Tuple[float, float] = (0.25, 4)  # X-axis scaling range
    scale_y_range: Tuple[float, float] = (0.25, 4)  # Y-axis scaling range
    scale_fixed_ratio: bool = False  # If True, preserve x-y aspect ratio during scaling
    shear_x_range: Tuple[float, float] = (-10.0, 10.0)  # X-axis shear angle range in degrees
    shear_y_range: Tuple[float, float] = (-10.0, 10.0)  # Y-axis shear angle range in degrees
    morph_operations: List[str] = field(default_factory=lambda: ["dilate", "erode", "open", "close"])
    kernel_size: int = 3

    # Internal-only fields (not exposed in config files)
    # These are automatically managed based on whether fixed ranges are specified
    use_dynamic_shift_range: bool = field(default=True, metadata={"internal": True})
    use_dynamic_rotation_range: bool = field(default=True, metadata={"internal": True})
    use_dynamic_shear_range: bool = field(default=True, metadata={"internal": True})

    @classmethod
    def _get_internal_fields(cls) -> frozenset:
        """Get set of internal field names that should not be exposed in config files"""
        return frozenset(f.name for f in dataclass_fields(cls) if f.metadata.get("internal", False))

    @classmethod
    def from_config_file(cls, config_path: str) -> "AugmentationParams":
        """Load augmentation parameters from a JSON or YAML config file"""
        # Load config based on file extension
        if config_path.endswith(".json"):
            with open(config_path, "r") as f:
                config = json.load(f)
        elif config_path.endswith((".yml", ".yaml")):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format. Use .json, .yml, or .yaml")

        # Get valid field names from the dataclass, excluding internal fields
        valid_fields = set(inspect.signature(cls).parameters.keys())
        internal_fields = cls._get_internal_fields()

        # Valid config keys = all dataclass fields minus internal fields
        valid_config_keys = valid_fields - internal_fields

        # Check for invalid keys in config
        config_keys = set(config.keys())
        invalid_keys = config_keys - valid_config_keys

        if invalid_keys:
            raise ValueError(
                f"Invalid keys found in config file '{config_path}': {sorted(invalid_keys)}. "
                f"Valid keys are: {sorted(valid_config_keys)}"
            )

        # Convert lists to tuples for range fields (JSON deserializes tuples as lists)
        tuple_fields = [
            "shift_x_range",
            "shift_y_range",
            "rotation_range",
            "scale_x_range",
            "scale_y_range",
            "shear_x_range",
            "shear_y_range",
        ]
        for field_name in tuple_fields:
            if field_name in config and isinstance(config[field_name], list):
                config[field_name] = tuple(config[field_name])

        # Create AugmentationParams instance with config values
        params = cls(**config)

        # Auto-disable dynamic operations if specific values are provided in config
        # Logic: If user specifies a fixed value (e.g., shift_x_range), disable the corresponding dynamic calculation
        # Note: use_dynamic_XXX attributes are internal-only and not exposed in config files
        dynamic_operations = {}

        # For shift: disable dynamic if shift_x_range or shift_y_range is specified
        if "shift_x_range" in config or "shift_y_range" in config:
            dynamic_operations["use_dynamic_shift_range"] = False
        else:
            # Default to True (enable dynamic) if not specified
            dynamic_operations["use_dynamic_shift_range"] = True

        # For rotation: disable dynamic if rotation_range is specified
        if "rotation_range" in config:
            dynamic_operations["use_dynamic_rotation_range"] = False
        else:
            # Default to True (enable dynamic) if not specified
            dynamic_operations["use_dynamic_rotation_range"] = True

        # For shear: disable dynamic if shear_x_range or shear_y_range is specified
        if "shear_x_range" in config or "shear_y_range" in config:
            dynamic_operations["use_dynamic_shear_range"] = False
        else:
            # Default to True (enable dynamic) if not specified
            dynamic_operations["use_dynamic_shear_range"] = True

        # Apply the dynamic settings
        params = replace(params, **dynamic_operations)
        return params

    def report_dynamic_operations(self) -> str:
        """Report which dynamic operations are enabled/disabled"""
        operations = {
            "shift": self.use_dynamic_shift_range,
            "rotation": self.use_dynamic_rotation_range,
            "shear": self.use_dynamic_shear_range,
        }

        enabled = [name for name, enabled in operations.items() if enabled]
        disabled = [name for name, enabled in operations.items() if not enabled]

        report = []
        if enabled:
            report.append(f"Dynamic operations enabled: {', '.join(enabled)}")
        if disabled:
            report.append(f"Dynamic operations disabled: {', '.join(disabled)}")

        return "; ".join(report) if report else "All dynamic operations using default settings"

    @staticmethod
    def _convert_tuples_to_lists(obj):
        """Recursively convert tuples to lists for JSON/YAML serialization"""
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: AugmentationParams._convert_tuples_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [AugmentationParams._convert_tuples_to_lists(item) for item in obj]
        return obj

    def save_to_config_file(self, config_path: str):
        """Save current augmentation parameters to a JSON or YAML config file

        Note: use_dynamic_XXX attributes are internal-only and not saved to config files.
        Dynamic operations are automatically determined based on whether fixed ranges are specified.
        """
        # Convert dataclass to dictionary, excluding internal fields
        config_dict = asdict(self)

        # Remove internal fields
        internal_fields = self._get_internal_fields()
        for field_name in internal_fields:
            config_dict.pop(field_name, None)

        # Convert tuples to lists for JSON/YAML compatibility
        config_dict = self._convert_tuples_to_lists(config_dict)

        # Save based on file extension
        if config_path.endswith(".json"):
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.endswith((".yml", ".yaml")):
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported config file format. Use .json, .yml, or .yaml")
