"""Headless napari widget smoke tests for migrated workflow-driven widgets."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

pytest.importorskip("napari")
pytest.importorskip("PyQt6")

from napari.components import ViewerModel
from PyQt6.QtWidgets import QApplication

import widgets.deconvolution_widget as deconv_module
import widgets.filament_tracing_widget as filament_module
from widgets.analysis_widget import AnalysisWidget
from widgets.deconvolution_widget import DeconvolutionWidget
from widgets.filament_tracing_widget import FilamentTracingWidget
from widgets.spots_detection_widget import (
    advanced_spatial_statistics_widget,
    spots_detection_widget,
)
from widgets.statistics_widget import StatisticsWidget
from widgets.tracking_widget import AdvancedTrackingWidget
from widgets.widget_manager import WidgetManagerWidget


def _settings_stub(base_dir: Path) -> dict[str, object]:
    return {
        "base_project_dir": str(base_dir),
        "session_naming": "none",
        "session_name": "default",
        "session_prefix": "session",
        "provenance_enabled": False,
    }


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_filament_tracing_widget_workflow_smoke(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(filament_module, "load_project_store_settings", lambda: _settings_stub(tmp_path))
    monkeypatch.setattr(
        filament_module,
        "resolve_project_store_dir",
        lambda settings, session_dir_cache=None: tmp_path / "project",
    )

    viewer = ViewerModel(title="filament-smoke")
    image = np.zeros((64, 64), dtype=np.float32)
    image[32, 10:54] = 100.0
    viewer.add_image(image, name="filament_input")

    widget = FilamentTracingWidget(viewer)
    widget.layer_combo.setCurrentText("filament_input")
    widget.threshold_method_combo.setCurrentText("Manual")
    widget.manual_threshold_spin.setValue(10)
    widget.min_size_spin.setValue(10)
    widget.trace_filaments()

    assert widget.workflow_handle is not None
    result = widget.workflow_handle.result(timeout=60)
    widget.on_workflow_finished(result)

    assert widget.current_results is not None
    assert widget.current_results["num_filaments"] >= 1
    assert "Skeleton_filament_input" in [layer.name for layer in viewer.layers]

    widget.close()
    qapp.processEvents()


def test_deconvolution_widget_workflow_smoke(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(deconv_module, "load_project_store_settings", lambda: _settings_stub(tmp_path))
    monkeypatch.setattr(
        deconv_module,
        "resolve_project_store_dir",
        lambda settings, session_dir_cache=None: tmp_path / "project",
    )

    viewer = ViewerModel(title="deconvolution-smoke")
    image = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    viewer.add_image(image, name="deconv_input")

    widget = DeconvolutionWidget(viewer)
    widget.layer_combo.setCurrentText("deconv_input")
    widget.algo_combo.setCurrentText("richardson_lucy")
    widget.rl_iterations.setValue(2)
    widget.run_deconvolution()

    assert widget.workflow_handle is not None
    result = widget.workflow_handle.result(timeout=60)
    widget.on_workflow_finished(result)

    assert "richardson_lucy_deconvolved" in [layer.name for layer in viewer.layers]

    widget.close()
    qapp.processEvents()


def test_widget_manager_project_store_panel_smoke(qapp: QApplication) -> None:
    viewer = ViewerModel(title="manager-smoke")
    widget = WidgetManagerWidget(viewer)

    widget.project_base_edit.setText(".tmp_project_store")
    widget.session_naming_combo.setCurrentText("fixed")
    widget.session_value_edit.setText("smoke_session")
    widget.provenance_enabled_check.setChecked(False)
    settings = widget._collect_project_store_settings()

    assert settings["base_project_dir"] == ".tmp_project_store"
    assert settings["session_naming"] == "fixed"
    assert settings["session_name"] == "smoke_session"
    assert settings["provenance_enabled"] is False

    widget.close()
    qapp.processEvents()


def test_updated_widgets_headless_smoke(qapp: QApplication) -> None:
    """Smoke test for newly enhanced analysis/statistics/tracking/spots widgets."""
    viewer = ViewerModel(title="updated-widgets-smoke")

    analysis_widget = AnalysisWidget(viewer)
    statistics_widget = StatisticsWidget(viewer)
    tracking_widget = AdvancedTrackingWidget(viewer)

    assert callable(spots_detection_widget)
    assert callable(advanced_spatial_statistics_widget)

    analysis_widget.close()
    statistics_widget.close()
    tracking_widget.close()
    qapp.processEvents()
