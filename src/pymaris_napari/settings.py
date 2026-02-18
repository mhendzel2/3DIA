"""Lightweight napari configuration helpers for project-store settings."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

DEFAULT_PROJECT_STORE_SETTINGS: dict[str, Any] = {
    "base_project_dir": ".pymaris_project",
    "session_naming": "timestamp",  # one of: none, fixed, timestamp
    "session_name": "default",
    "session_prefix": "session",
    "provenance_enabled": True,
}


def widget_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "config" / "widget_config.json"


def load_widget_config() -> dict[str, Any]:
    path = widget_config_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def save_widget_config(config: Mapping[str, Any]) -> None:
    path = widget_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(config), handle, indent=4, sort_keys=True)


def load_project_store_settings(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(config or load_widget_config())
    configured = payload.get("project_store", {})
    settings = dict(DEFAULT_PROJECT_STORE_SETTINGS)
    if isinstance(configured, dict):
        settings.update(configured)
    settings["session_naming"] = _normalize_session_naming(str(settings.get("session_naming", "timestamp")))
    settings["base_project_dir"] = str(settings.get("base_project_dir", ".pymaris_project"))
    settings["session_name"] = str(settings.get("session_name", "default"))
    settings["session_prefix"] = str(settings.get("session_prefix", "session"))
    settings["provenance_enabled"] = bool(settings.get("provenance_enabled", True))
    return settings


def save_project_store_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    payload = load_widget_config()
    merged = load_project_store_settings(payload)
    merged.update(dict(settings))
    merged["session_naming"] = _normalize_session_naming(str(merged.get("session_naming", "timestamp")))
    payload["project_store"] = merged
    save_widget_config(payload)
    return merged


def resolve_project_store_dir(
    settings: Mapping[str, Any],
    *,
    session_dir_cache: Path | None = None,
    now: datetime | None = None,
) -> Path:
    base = Path(str(settings.get("base_project_dir", ".pymaris_project"))).expanduser()
    naming = _normalize_session_naming(str(settings.get("session_naming", "timestamp")))

    if naming == "none":
        return base
    if naming == "fixed":
        session_name = _sanitize_path_component(str(settings.get("session_name", "default")))
        return base / session_name
    if session_dir_cache is not None:
        return session_dir_cache

    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    prefix = _sanitize_path_component(str(settings.get("session_prefix", "session")))
    return base / f"{prefix}-{timestamp}"


def _normalize_session_naming(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"none", "fixed", "timestamp"}:
        return normalized
    return "timestamp"


def _sanitize_path_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned or "session"
