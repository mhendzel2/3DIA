"""Tests for configurable napari workspace preset loading."""

from __future__ import annotations

import json
from pathlib import Path

from pymaris_napari import configurable


def test_load_config_applies_active_workspace(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "widget_config.json"
    payload = {
        "active_workspace": "tracking",
        "workspaces": {
            "tracking": {
                "enabled_widgets": {
                    "file_io": True,
                    "tracking": True,
                    "hca": False,
                    "distance_tools": True,
                },
                "widget_areas": {
                    "tracking": "right",
                    "distance_tools": "right",
                },
            }
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(configurable, "_default_config_path", lambda: config_path)

    loaded = configurable.load_config()

    assert loaded["active_workspace"] == "tracking"
    assert loaded["enabled_widgets"]["tracking"] is True
    assert loaded["enabled_widgets"]["distance_tools"] is True
    assert loaded["enabled_widgets"]["hca"] is False
    assert loaded["widget_areas"]["tracking"] == "right"


def test_load_config_falls_back_when_workspace_missing(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "widget_config.json"
    payload = {
        "active_workspace": "does_not_exist",
        "workspaces": {
            "default": {
                "enabled_widgets": {
                    "file_io": True,
                    "processing": False,
                },
                "widget_areas": {
                    "file_io": "left",
                },
            }
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(configurable, "_default_config_path", lambda: config_path)

    loaded = configurable.load_config()

    assert loaded["active_workspace"] == "default"
    assert loaded["enabled_widgets"]["file_io"] is True
    assert loaded["enabled_widgets"]["processing"] is False


def test_apply_workspace_override_selects_requested_workspace() -> None:
    config = {
        "active_workspace": "default",
        "workspaces": {
            "default": {"enabled_widgets": {"tracking": False}},
            "tracking": {"enabled_widgets": {"tracking": True}},
        },
    }

    overridden = configurable.apply_workspace_override(config, "tracking")

    assert overridden["active_workspace"] == "tracking"
    assert overridden["enabled_widgets"]["tracking"] is True


def test_parse_launch_args_workspace_and_list() -> None:
    args = configurable.parse_launch_args(["--workspace", "tracking", "--list-workspaces"])

    assert args.workspace == "tracking"
    assert args.list_workspaces is True
