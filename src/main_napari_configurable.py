"""Compatibility shim for configurable napari launcher.

Existing imports from `main_napari_configurable` are forwarded to the new
package module under `pymaris_napari.configurable`.
"""

from pymaris_napari.configurable import DEFAULT_CONFIG, load_config, load_widget, main

__all__ = ["DEFAULT_CONFIG", "load_config", "load_widget", "main"]


if __name__ == "__main__":
    main()
