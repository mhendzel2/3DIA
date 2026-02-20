#!/usr/bin/env python3
"""Legacy module import/functionality smoke tests."""

from __future__ import annotations

import sys

sys.path.append("src")


def test_basic_imports() -> None:
    from main import launch_enhanced_web_interface, launch_web_interface, main

    assert callable(main)
    assert callable(launch_web_interface)
    assert callable(launch_enhanced_web_interface)


def test_web_servers() -> None:
    from simple_analyzer import SimpleImageAnalyzer, run_server

    assert callable(run_server)
    test_image = SimpleImageAnalyzer.create_test_image(width=10, height=10)
    labels = SimpleImageAnalyzer.simple_threshold(test_image, threshold=128)
    measurements = SimpleImageAnalyzer.measure_objects(labels)
    assert isinstance(measurements, list)


def test_scientific_analyzer() -> None:
    import scientific_analyzer

    assert scientific_analyzer is not None
    assert hasattr(scientific_analyzer, "HAS_FLASK")


def test_bug_fixes_integration() -> None:
    from batch_processor import BatchProcessor
    from fibsem_plugins import ChimeraXIntegration, ThreeDCounter

    assert ChimeraXIntegration is not None
    assert ThreeDCounter is not None
    assert BatchProcessor is not None
