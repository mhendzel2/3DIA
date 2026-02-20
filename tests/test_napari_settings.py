"""Tests for shared napari project-store settings helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pymaris_napari import settings as napari_settings


def test_load_project_store_settings_defaults(monkeypatch, tmp_path: Path) -> None:
    fake_config = tmp_path / "widget_config.json"
    monkeypatch.setattr(napari_settings, "widget_config_path", lambda: fake_config)
    settings = napari_settings.load_project_store_settings()
    assert settings["base_project_dir"] == ".pymaris_project"
    assert settings["session_naming"] == "timestamp"
    assert settings["provenance_enabled"] is True


def test_save_project_store_settings_and_resolve_fixed(monkeypatch, tmp_path: Path) -> None:
    fake_config = tmp_path / "widget_config.json"
    monkeypatch.setattr(napari_settings, "widget_config_path", lambda: fake_config)
    napari_settings.save_project_store_settings(
        {
            "base_project_dir": str(tmp_path / "projects"),
            "session_naming": "fixed",
            "session_name": "plate_01",
            "provenance_enabled": False,
        }
    )
    loaded = napari_settings.load_project_store_settings()
    resolved = napari_settings.resolve_project_store_dir(loaded)
    assert loaded["provenance_enabled"] is False
    assert resolved == (tmp_path / "projects" / "plate_01")


def test_resolve_timestamp_uses_cache(monkeypatch, tmp_path: Path) -> None:
    fake_config = tmp_path / "widget_config.json"
    monkeypatch.setattr(napari_settings, "widget_config_path", lambda: fake_config)
    settings = napari_settings.save_project_store_settings(
        {
            "base_project_dir": str(tmp_path / "projects"),
            "session_naming": "timestamp",
            "session_prefix": "napari",
        }
    )
    first = napari_settings.resolve_project_store_dir(
        settings,
        now=datetime(2026, 2, 9, 12, 0, 0),
    )
    second = napari_settings.resolve_project_store_dir(
        settings,
        session_dir_cache=first,
        now=datetime(2026, 2, 9, 13, 0, 0),
    )
    assert first == second
    assert first.name.startswith("napari-20260209-120000")
