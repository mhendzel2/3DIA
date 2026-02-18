"""Compatibility shim tests for legacy entry modules."""

from __future__ import annotations

import importlib
import sys


def test_main_napari_shim_imports_without_napari() -> None:
    sys.modules.pop("napari", None)
    module = importlib.import_module("main_napari")

    assert callable(module.main)
    assert "napari" not in sys.modules


def test_main_napari_configurable_shim_exports() -> None:
    sys.modules.pop("napari", None)
    module = importlib.import_module("main_napari_configurable")

    assert callable(module.main)
    assert callable(module.load_config)
    assert isinstance(module.DEFAULT_CONFIG, dict)
    assert "enabled_widgets" in module.DEFAULT_CONFIG
    assert "napari" not in sys.modules
