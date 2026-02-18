#!/usr/bin/env python3
"""Batch processing and HCA feature regression tests."""

from __future__ import annotations

import sys

import pytest
import requests

sys.path.append("src")


def test_batch_processing_imports() -> None:
    from scientific_analyzer import HAS_BATCH_PROCESSOR, HAS_FLASK

    assert isinstance(HAS_FLASK, bool)
    assert isinstance(HAS_BATCH_PROCESSOR, bool)

    if HAS_BATCH_PROCESSOR:
        from batch_processor import WORKFLOW_TEMPLATES

        assert isinstance(WORKFLOW_TEMPLATES, dict)
        assert WORKFLOW_TEMPLATES


def test_hca_widget_imports() -> None:
    from utils.analysis_utils import fit_dose_response
    from widgets.hca_widget import HighContentAnalysisWidget, PlateVisualizationWidget

    assert HighContentAnalysisWidget is not None
    assert PlateVisualizationWidget is not None

    concentrations = [0.1, 1.0, 10.0, 100.0, 1000.0]
    responses = [100, 90, 70, 30, 10]
    result = fit_dose_response(concentrations, responses)
    assert "error" not in result


def test_flask_batch_endpoints() -> None:
    try:
        response = requests.get("http://localhost:5000/", timeout=2)
    except requests.exceptions.RequestException:
        pytest.skip("Flask server not accessible")

    if response.status_code != 200:
        pytest.skip("Flask server not running on localhost:5000")

    workflows_response = requests.get("http://localhost:5000/api/workflows", timeout=5)
    assert workflows_response.status_code == 200
    workflows = workflows_response.json()
    assert "workflows" in workflows

    payload = {"files": ["test1.tif", "test2.tif"], "workflow": "cell_counting"}
    batch_response = requests.post(
        "http://localhost:5000/api/batch/process",
        json=payload,
        timeout=5,
    )
    assert batch_response.status_code == 200

    result = batch_response.json()
    batch_id = result.get("batch_id")
    if batch_id:
        status_response = requests.get(
            f"http://localhost:5000/api/batch/status/{batch_id}",
            timeout=5,
        )
        assert status_response.status_code == 200


def test_napari_hca_integration() -> None:
    napari = pytest.importorskip("napari")

    from widgets.hca_widget import HighContentAnalysisWidget

    viewer = napari.Viewer(show=False)
    hca_widget = HighContentAnalysisWidget(viewer)
    assert hasattr(hca_widget, "load_plate_layout")
    assert hasattr(hca_widget, "run_analysis")
    viewer.close()
