# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the experiment-recipe loader (public API used by train.py / generate.py)."""

import pytest
import yaml
from hydra.core.config_store import ConfigStore

from anomalygen.configs.loader import _import_object, register_recipe


def test_import_object_resolves_dotted_path():
    import os

    assert _import_object("os.path.join") is os.path.join


def test_import_object_requires_module_qualifier():
    with pytest.raises(ValueError):
        _import_object("join")  # no 'module.attr' form


def test_register_recipe_python_module_returns_none():
    # A Python-module recipe imports the module for its cs.store side effect and returns None.
    # The module must sit in an allowed namespace; this one is already imported, so the
    # import is a no-op and the assertion is about the return value alone.
    assert register_recipe("anomalygen.configs.loader") is None


def test_register_recipe_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        register_recipe("/no/such/recipe.yaml")


def test_register_recipe_non_mapping_raises(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(["not", "a", "mapping"]))
    with pytest.raises(ValueError):
        register_recipe(str(path))


def test_register_recipe_missing_required_keys_raises(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({"task_type": "texture_ft"}))  # 'experiment' missing
    with pytest.raises(ValueError):
        register_recipe(str(path))


def test_register_recipe_unknown_task_type_raises(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({"task_type": "bogus", "experiment": "e"}))
    with pytest.raises(ValueError):
        register_recipe(str(path))


def test_register_recipe_file_happy_path_registers_real_experiment(tmp_path):
    # The real texture_ft builder imports the cosmos_framework model stack, which pulls in triton
    # (a CUDA dep absent from the CPU-only test env). Skip there; this runs in the full/GPU env.
    pytest.importorskip("triton")

    # Drive the real texture_ft builder through a YAML recipe (config assembly only — no model
    # build or I/O), then read the registered node back from the ConfigStore.
    recipe = {
        "task_type": "texture_ft",
        "experiment": "unittest_texture_ft",
        "dataset_name": "unit_test_ds",
        "anomaly_types": [["metal", "scratch"], ["wood", "hole"]],
        "dataset_path": "data/train",  # absolutized as a string; need not exist
        "testcase_jsonl": "tests/example.jsonl",
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))

    name = register_recipe(str(path))
    assert name == "unittest_texture_ft"

    cs = ConfigStore.instance()
    assert "unittest_texture_ft.yaml" in cs.list("experiment")

    node = cs.load("experiment/unittest_texture_ft.yaml").node
    # The builder accepts recipe_path, so the loader injects the recipe's own resolved path.
    assert node["trainer"]["callbacks"]["training_report"]["recipe_path"] == str(path.resolve())
    # num_classes is derived from the two supplied anomaly_types pairs.
    assert node["model"]["config"]["diffusion_expert_config"]["anomaly_num_classes"] == 2


# --- module-recipe namespace ----------------------------------------------------------------------
# Importing a module recipe runs it, so the namespace is pinned even though --recipe is argv-only.


@pytest.mark.parametrize("recipe", ["os", "antigravity", "evil_pkg.payload", "..sneaky"])
def test_module_recipe_outside_the_allowed_namespaces_is_refused(recipe):
    with pytest.raises(ValueError, match="outside the allowed namespaces"):
        register_recipe(recipe)


def test_module_recipe_inside_an_allowed_namespace_is_imported(monkeypatch):
    imported = []
    monkeypatch.setattr(
        "anomalygen.configs.loader.importlib.import_module",
        lambda name: imported.append(name),
    )
    assert register_recipe("ag_config.exp_texture_ft_phone_screen") is None
    assert imported == ["ag_config.exp_texture_ft_phone_screen"]


def test_file_recipes_are_unaffected_by_the_namespace_check(tmp_path):
    """The allowlist must gate module recipes only; file recipes take the other branch."""
    missing = tmp_path / "no_such_recipe.yaml"
    with pytest.raises(FileNotFoundError):
        register_recipe(str(missing))
