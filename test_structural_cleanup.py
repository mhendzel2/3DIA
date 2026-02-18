#!/usr/bin/env python3
"""Structural cleanup regression tests."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.append("src")


def test_main_imports() -> None:
    from main import (
        launch_enhanced_web_interface,
        launch_napari_interface,
        launch_web_interface,
        main,
    )

    assert callable(main)
    assert callable(launch_napari_interface)
    assert callable(launch_web_interface)
    assert callable(launch_enhanced_web_interface)


def test_napari_imports() -> None:
    from main_napari import main

    assert callable(main)


def test_widget_imports() -> None:
    from widgets.analysis_widget import AnalysisWidget
    from widgets.file_io_widget import FileIOWidget
    from widgets.processing_widget import ProcessingWidget
    from widgets.segmentation_widget import SegmentationWidget
    from widgets.visualization_widget import VisualizationWidget

    assert AnalysisWidget is not None
    assert ProcessingWidget is not None
    assert SegmentationWidget is not None
    assert VisualizationWidget is not None
    assert FileIOWidget is not None


def test_web_server_imports() -> None:
    import scientific_analyzer
    from simple_analyzer import run_server

    assert scientific_analyzer is not None
    if scientific_analyzer.HAS_FLASK:
        assert hasattr(scientific_analyzer, "app")
    else:
        pytest.skip("Flask not installed in this environment")
    assert callable(run_server)


def test_removed_files() -> None:
    removed_files = [
        "src/enhanced_web_app.py",
        "src/analysis_widget.py",
        "src/processing_widget.py",
        "src/segmentation_widget.py",
        "src/dist/",
    ]
    for file_path in removed_files:
        assert not os.path.exists(file_path), f"File should be removed: {file_path}"
