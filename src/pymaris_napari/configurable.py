"""Configurable napari launcher with compatibility helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pymaris_napari import _widgets

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled_widgets": {
        "file_io": True,
        "processing": True,
        "segmentation": True,
        "analysis": True,
        "visualization": True,
        "deconvolution": False,
        "statistics": True,
        "filament_tracing": True,
        "tracking": True,
        "simple_threshold": False,
        "adaptive_threshold": False,
        "hca": True,
        "ai_segmentation": False,
        "biophysics": False,
        "interactive_plotting": False,
        "distance_tools": False,
        "workflow_runner": True,
    },
    "widget_areas": {
        "file_io": "left",
        "processing": "left",
        "segmentation": "left",
        "analysis": "left",
        "visualization": "left",
        "deconvolution": "left",
        "statistics": "right",
        "filament_tracing": "right",
        "tracking": "right",
        "simple_threshold": "left",
        "adaptive_threshold": "left",
        "hca": "right",
        "ai_segmentation": "right",
        "biophysics": "right",
        "interactive_plotting": "right",
        "distance_tools": "right",
        "workflow_runner": "right",
    },
    "load_on_startup": True,
    "show_welcome_message": True,
    "project_store": {
        "base_project_dir": ".pymaris_project",
        "session_naming": "timestamp",
        "session_name": "default",
        "session_prefix": "session",
        "provenance_enabled": True,
    },
}


def _default_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "config" / "widget_config.json"


def load_config(config_file: Path | None = None) -> dict[str, Any]:
    """Load widget configuration from disk."""
    path = config_file or _default_config_path()

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                return DEFAULT_CONFIG
            return _merge_config(DEFAULT_CONFIG, loaded)
        except Exception as exc:
            LOGGER.warning("Could not load widget config '%s': %s", path, exc)
            return DEFAULT_CONFIG

    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG, handle, indent=4)
    return DEFAULT_CONFIG


def _merge_config(defaults: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def load_widget(viewer: Any, widget_name: str, config: dict[str, Any]) -> Any:
    """Load a single configured widget into a viewer."""
    if not config.get("enabled_widgets", {}).get(widget_name, False):
        return None

    area = config.get("widget_areas", {}).get(widget_name, "left")

    if widget_name not in _widgets.CONFIG_WIDGET_FACTORIES:
        LOGGER.warning("Unknown widget key: %s", widget_name)
        return None

    try:
        factory = _widgets.CONFIG_WIDGET_FACTORIES[widget_name]
        if widget_name == "file_io":
            dock_name = "File I/O"
        elif widget_name == "analysis":
            dock_name = "Analysis & Plotting"
        elif widget_name == "tracking":
            dock_name = "Cell Tracking & Lineage"
        elif widget_name == "hca":
            dock_name = "High-Content Analysis"
        elif widget_name == "biophysics":
            dock_name = "Biophysics Analysis"
        elif widget_name == "simple_threshold":
            dock_name = "Simple Threshold"
        elif widget_name == "adaptive_threshold":
            dock_name = "Adaptive Threshold"
        elif widget_name == "interactive_plotting":
            dock_name = "Interactive Plotting"
        elif widget_name == "distance_tools":
            dock_name = "Distance Tools"
        elif widget_name == "workflow_runner":
            dock_name = "Workflow Runner (Core Backends)"
        elif widget_name == "ai_segmentation":
            dock_name = "AI Segmentation"
        elif widget_name == "filament_tracing":
            dock_name = "Filament Tracing"
        elif widget_name == "visualization":
            dock_name = "3D Visualization"
        else:
            dock_name = widget_name.replace("_", " ").title()

        widget = factory(viewer)
        viewer.window.add_dock_widget(widget, name=dock_name, area=area)
        return widget
    except Exception as exc:  # pragma: no cover - optional dependencies
        LOGGER.warning("Could not load widget '%s': %s", widget_name, exc)
        return None


def main() -> None:
    """Run the configurable napari desktop application."""
    logging.basicConfig(level=logging.INFO)

    import napari

    config = load_config()
    viewer = napari.Viewer(title="PyMaris Scientific Image Analyzer")

    try:
        manager = _widgets.widget_manager_widget(viewer)
        viewer.window.add_dock_widget(manager, name="Widget Manager", area="right")
    except Exception as exc:  # pragma: no cover - optional dependencies
        LOGGER.warning("Could not load Widget Manager: %s", exc)

    if config.get("load_on_startup", True):
        for widget_name, enabled in config.get("enabled_widgets", {}).items():
            if enabled:
                load_widget(viewer=viewer, widget_name=widget_name, config=config)

    napari.run()


if __name__ == "__main__":
    main()
