"""Configurable napari launcher with compatibility helpers."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from pymaris_napari import _widgets

LOGGER = logging.getLogger(__name__)

_BASE_ENABLED_WIDGETS: dict[str, bool] = {
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
}

_BASE_WIDGET_AREAS: dict[str, str] = {
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
}

DEFAULT_WORKSPACES: dict[str, dict[str, Any]] = {
    "default": {
        "enabled_widgets": dict(_BASE_ENABLED_WIDGETS),
        "widget_areas": dict(_BASE_WIDGET_AREAS),
    },
    "tracking": {
        "enabled_widgets": {
            **dict(_BASE_ENABLED_WIDGETS),
            "hca": False,
            "filament_tracing": False,
            "deconvolution": False,
            "ai_segmentation": False,
            "biophysics": False,
            "interactive_plotting": False,
            "distance_tools": True,
        },
        "widget_areas": {
            **dict(_BASE_WIDGET_AREAS),
            "tracking": "right",
            "statistics": "right",
            "distance_tools": "right",
            "workflow_runner": "right",
        },
    },
    "high_content_screening": {
        "enabled_widgets": {
            **dict(_BASE_ENABLED_WIDGETS),
            "tracking": False,
            "filament_tracing": False,
            "deconvolution": False,
            "ai_segmentation": True,
            "biophysics": True,
            "interactive_plotting": True,
            "distance_tools": True,
            "hca": True,
        },
        "widget_areas": {
            **dict(_BASE_WIDGET_AREAS),
            "hca": "right",
            "statistics": "right",
            "interactive_plotting": "right",
            "workflow_runner": "right",
        },
    },
    "viz_3d_quant": {
        "enabled_widgets": {
            **dict(_BASE_ENABLED_WIDGETS),
            "tracking": False,
            "hca": False,
            "filament_tracing": True,
            "deconvolution": True,
            "ai_segmentation": True,
            "biophysics": True,
            "interactive_plotting": True,
            "distance_tools": True,
        },
        "widget_areas": {
            **dict(_BASE_WIDGET_AREAS),
            "visualization": "left",
            "analysis": "left",
            "statistics": "right",
            "deconvolution": "right",
            "interactive_plotting": "right",
            "distance_tools": "right",
        },
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "active_workspace": "default",
    "workspaces": DEFAULT_WORKSPACES,
    "enabled_widgets": dict(_BASE_ENABLED_WIDGETS),
    "widget_areas": dict(_BASE_WIDGET_AREAS),
    "workspace_descriptions": {
        "default": "Balanced, general-purpose widget layout",
        "tracking": "Time-lapse and lineage tracking focused workflow",
        "high_content_screening": "Plate-scale high-content screening workflow",
        "viz_3d_quant": "3D visualization and quantitative analysis workflow",
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
            merged = _merge_config(DEFAULT_CONFIG, loaded)
            return _resolve_active_workspace(merged)
        except Exception as exc:
            LOGGER.warning("Could not load widget config '%s': %s", path, exc)
            return DEFAULT_CONFIG

    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG, handle, indent=4)
    return _resolve_active_workspace(dict(DEFAULT_CONFIG))


def _merge_config(defaults: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def _resolve_active_workspace(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    workspaces = merged.get("workspaces", {})
    if not isinstance(workspaces, dict):
        workspaces = {}
    active = str(merged.get("active_workspace", "default")).strip() or "default"
    workspace_payload = workspaces.get(active)
    if not isinstance(workspace_payload, dict):
        workspace_payload = workspaces.get("default") if isinstance(workspaces.get("default"), dict) else {}
        active = "default"

    workspace_enabled = workspace_payload.get("enabled_widgets", {}) if isinstance(workspace_payload, dict) else {}
    workspace_areas = workspace_payload.get("widget_areas", {}) if isinstance(workspace_payload, dict) else {}
    if isinstance(workspace_enabled, dict):
        merged["enabled_widgets"] = _merge_config(dict(_BASE_ENABLED_WIDGETS), workspace_enabled)
    if isinstance(workspace_areas, dict):
        merged["widget_areas"] = _merge_config(dict(_BASE_WIDGET_AREAS), workspace_areas)
    merged["active_workspace"] = active
    merged["workspaces"] = _merge_config(DEFAULT_WORKSPACES, workspaces)
    return merged


def apply_workspace_override(config: dict[str, Any], workspace_name: str) -> dict[str, Any]:
    """Override active workspace if it exists and re-resolve enabled widget payload."""
    updated = dict(config)
    workspaces = updated.get("workspaces", {})
    if not isinstance(workspaces, dict):
        return _resolve_active_workspace(updated)

    candidate = str(workspace_name).strip()
    if candidate and candidate in workspaces:
        updated["active_workspace"] = candidate
    else:
        LOGGER.warning(
            "Requested workspace '%s' not found. Available: %s",
            workspace_name,
            ", ".join(sorted(workspaces.keys())) if workspaces else "(none)",
        )
    return _resolve_active_workspace(updated)


def parse_launch_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for configurable napari launcher."""
    parser = argparse.ArgumentParser(description="Run PyMaris configurable napari launcher")
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace preset name (e.g., tracking, high_content_screening, viz_3d_quant)",
    )
    parser.add_argument(
        "--list-workspaces",
        action="store_true",
        help="Print available workspace presets and exit",
    )
    return parser.parse_args(argv)


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


def main(argv: Sequence[str] | None = None) -> None:
    """Run the configurable napari desktop application."""
    logging.basicConfig(level=logging.INFO)
    args = parse_launch_args(argv)

    config = load_config()
    if args.workspace:
        config = apply_workspace_override(config, args.workspace)

    if args.list_workspaces:
        workspace_names = sorted((config.get("workspaces") or {}).keys())
        if workspace_names:
            print("Available workspaces:")
            for name in workspace_names:
                print(f"- {name}")
        else:
            print("No workspaces configured.")
        return

    import napari

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
