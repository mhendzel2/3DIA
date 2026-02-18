"""Tests for headless core package boundaries."""

from __future__ import annotations

import importlib
import sys


def test_pymaris_import_does_not_import_napari() -> None:
    sys.modules.pop("napari", None)

    import pymaris  # noqa: F401

    assert "napari" not in sys.modules


def test_pymaris_analysis_exports() -> None:
    module = importlib.import_module("pymaris.analysis")

    assert hasattr(module, "load_image")
    assert hasattr(module, "segment_watershed")
    assert hasattr(module, "calculate_object_statistics")


def test_pymaris_top_level_exports() -> None:
    module = importlib.import_module("pymaris")
    assert hasattr(module, "ImageVolume")
    assert hasattr(module, "open_image")
    assert hasattr(module, "save_image")
    assert hasattr(module, "ProjectStore")
    assert hasattr(module, "WorkflowStep")
    assert hasattr(module, "JobRunner")
