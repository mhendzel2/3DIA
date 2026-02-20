"""Primary napari launcher for the PyMaris plugin package."""

from __future__ import annotations

import logging
from typing import Any

from pymaris_napari import _widgets

LOGGER = logging.getLogger(__name__)


DEFAULT_WIDGET_LAYOUT: list[tuple[str, str, str]] = [
    ("file_io", "File I/O", "left"),
    ("processing", "Image Processing", "left"),
    ("segmentation", "Segmentation", "left"),
    ("ai_segmentation", "AI Segmentation", "left"),
    ("analysis", "Analysis & Plotting", "left"),
    ("visualization", "3D Visualization", "left"),
    ("deconvolution", "Deconvolution", "left"),
    ("statistics", "Statistics", "right"),
    ("filament_tracing", "Filament Tracing", "right"),
    ("tracking", "Cell Tracking & Lineage", "right"),
    ("simple_threshold", "Simple Threshold", "left"),
    ("adaptive_threshold", "Adaptive Threshold", "left"),
    ("hca", "High-Content Analysis", "right"),
    ("workflow_runner", "Workflow Runner (Core Backends)", "right"),
]


def _add_widget(viewer: Any, key: str, dock_name: str, area: str) -> None:
    factory = _widgets.CONFIG_WIDGET_FACTORIES[key]
    widget = factory(viewer)
    viewer.window.add_dock_widget(widget, name=dock_name, area=area)


def create_viewer(title: str = "PyMaris Scientific Image Analyzer") -> Any:
    """Create and return a napari viewer instance."""
    import napari

    return napari.Viewer(title=title)


def add_default_widgets(viewer: Any) -> None:
    """Populate the viewer with the default widget layout."""
    for key, dock_name, area in DEFAULT_WIDGET_LAYOUT:
        try:
            _add_widget(viewer=viewer, key=key, dock_name=dock_name, area=area)
        except Exception as exc:  # pragma: no cover - depends on optional extras
            LOGGER.warning("Failed to load widget '%s': %s", key, exc)


def main() -> None:
    """Run the napari desktop application."""
    logging.basicConfig(level=logging.INFO)
    viewer = create_viewer()
    add_default_widgets(viewer)

    import napari

    napari.run()


if __name__ == "__main__":
    main()
