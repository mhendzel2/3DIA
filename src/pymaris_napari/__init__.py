"""Napari plugin package for PyMaris."""

from pathlib import Path

napari_yaml = str(Path(__file__).with_name("napari.yaml"))

__all__ = ["napari_yaml"]
