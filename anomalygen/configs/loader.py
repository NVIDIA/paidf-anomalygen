# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic experiment-recipe loader for ``--recipe`` (Python module or YAML/JSON file).

``register_recipe(recipe)`` accepts either:

- an **importable Python module** (e.g. ``ag_config.exp_texture_ft_phone_screen``) whose import-time
  ``cs.store(...)`` registers the experiment node; or
- a path to a **``.yaml`` / ``.yml`` / ``.json`` file**: a mapping of builder kwargs plus two meta
  keys — ``task_type`` (a friendly name from ``_TASK_TYPE``) and ``experiment`` (the ConfigStore name
  to register under). The selected builder is called with the remaining kwargs (``recipe_path`` is
  injected if the builder accepts it) and its result is stored as the experiment node.

To expose a new experiment type to YAML/JSON recipes, add one entry to ``_TASK_TYPE`` below.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

_RECIPE_FILE_SUFFIXES = (".yaml", ".yml", ".json")
# Importing a module recipe runs it. --recipe comes from argv, so this is not a privilege
# boundary today, but a future caller sourcing it from a file or an API would inherit an
# arbitrary-import primitive; pinning the namespaces keeps that from being silent.
_RECIPE_MODULE_PREFIXES = ("ag_config.", "anomalygen.configs.")
_TASK_TYPE: dict[str, str] = {
    "texture_ft": "anomalygen.configs.texture.exp_config.build_anomalygen_texture_ft_experiment",
}


def register_recipe(recipe: str) -> str | None:
    """Register the experiment node named by ``--recipe`` — a YAML/JSON file or a Python module.

    Returns the registered experiment name for file recipes; ``None`` for Python-module recipes
    (whose ``cs.store`` name is chosen inside the imported module and isn't surfaced here).
    """
    if recipe.endswith(_RECIPE_FILE_SUFFIXES):
        return _register_from_file(recipe)
    if not recipe.startswith(_RECIPE_MODULE_PREFIXES):
        raise ValueError(
            f"Recipe module {recipe!r} is outside the allowed namespaces "
            f"{list(_RECIPE_MODULE_PREFIXES)}; pass a recipe file "
            f"({', '.join(_RECIPE_FILE_SUFFIXES)}) or a module under one of them."
        )
    importlib.import_module(recipe)
    return None


def _register_from_file(path: str) -> str:
    cfg_path = Path(path).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Recipe file not found: {cfg_path}")
    data = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Recipe file {cfg_path} must be a mapping of builder kwargs, got {type(data).__name__}.")
    missing = [k for k in ("task_type", "experiment") if k not in data]
    if missing:
        raise ValueError(
            f"Recipe file {cfg_path} is missing required key(s) {missing}; it must define "
            "'task_type' (one of " + ", ".join(sorted(_TASK_TYPE)) + ") and 'experiment' (ConfigStore node name)."
        )
    task_type = str(data.pop("task_type"))
    target = _TASK_TYPE.get(task_type)
    if target is None:
        raise ValueError(f"Unknown task_type {task_type!r} in {cfg_path}; choose one of {sorted(_TASK_TYPE)}.")
    builder = _import_object(target)
    name = str(data.pop("experiment"))
    # Hand the builder the recipe's own path (for TrainingReport's snapshot) if it accepts one.
    if "recipe_path" in inspect.signature(builder).parameters:
        data.setdefault("recipe_path", str(cfg_path))
    node = builder(**data)
    ConfigStore.instance().store(group="experiment", package="_global_", name=name, node=node)
    return name


def _import_object(dotted: str):
    """Import ``module.attr`` from a dotted path."""
    module_name, _, attr = dotted.rpartition(".")
    if not module_name:
        raise ValueError(f"Builder target must be a dotted 'module.function' path, got {dotted!r}.")
    return getattr(importlib.import_module(module_name), attr)
