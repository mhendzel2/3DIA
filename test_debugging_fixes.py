#!/usr/bin/env python3
"""Regression tests for prior debugging fixes."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append("src")


def test_aspect_ratio_edge_cases() -> None:
    from utils.analysis_utils import calculate_object_statistics

    labels = np.zeros((16, 16), dtype=np.int32)
    labels[1:3, 1:10] = 1
    labels[5:12, 5:7] = 2
    labels[14:15, 14:15] = 3
    stats = calculate_object_statistics(labels)
    assert "aspect_ratio" in stats
    assert len(stats["aspect_ratio"]) == 3


def test_numpy_fallback_visualization() -> None:
    from widgets.visualization_widget import VisualizationWidget

    assert VisualizationWidget is not None
    assert hasattr(VisualizationWidget, "on_gamma_change")


def test_track_endpoint_logic() -> None:
    import scientific_analyzer as sa

    if not sa.HAS_FLASK:
        pytest.skip("Flask surface is not available in this environment")

    labels = sa._coerce_labels_sequence([np.zeros((8, 8), dtype=np.int32), np.ones((8, 8), dtype=np.int32)])
    assert labels is not None
    assert len(labels) == 2

    generated = sa._labels_from_timeseries(np.random.rand(3, 8, 8))
    assert generated is not None
    assert len(generated) == 3


def test_napari_analyzer_removed() -> None:
    assert not os.path.exists("src/napari_analyzer.py")
