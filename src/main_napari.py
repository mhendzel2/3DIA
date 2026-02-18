"""Compatibility shim for the legacy napari launcher.

This module is kept so existing scripts (`python src/main_napari.py`) continue
to work while the napari application lives in `pymaris_napari.app`.
"""

from pymaris_napari.app import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
