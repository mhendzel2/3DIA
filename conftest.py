"""Global pytest bootstrap for stable headless napari imports."""

from __future__ import annotations

import os
from pathlib import Path


def _configure_headless_environment() -> None:
    root = Path(__file__).resolve().parent
    runtime_root = root / ".pytest_cache" / "runtime"
    xdg_cache_home = runtime_root / "xdg_cache"
    napari_config_dir = runtime_root / "napari"
    napari_config_file = napari_config_dir / "settings.yaml"

    xdg_cache_home.mkdir(parents=True, exist_ok=True)
    napari_config_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_home))
    os.environ.setdefault("NAPARI_CONFIG", str(napari_config_file))


_configure_headless_environment()
