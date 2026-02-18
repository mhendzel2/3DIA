"""Napari reader/writer adapters backed by `pymaris` core I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pymaris.io import open_image, save_image
from pymaris.layers import image_volume_from_layer_data, image_volume_to_layer_data

SUPPORTED_EXTENSIONS = {
    ".czi",
    ".lif",
    ".nd2",
    ".oib",
    ".oif",
    ".ims",
    ".lsm",
    ".tif",
    ".tiff",
    ".zarr",
}


def get_reader(path: str | list[str]) -> Any:
    """Return a reader callable for supported microscopy formats."""
    probe = path[0] if isinstance(path, list) else path
    probe_str = str(probe)
    suffix = Path(probe_str).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS and not probe_str.lower().endswith(".ome.zarr"):
        return None
    return _read_paths


def _read_paths(path: str | list[str]) -> list[tuple[Any, dict[str, Any], str]]:
    paths = path if isinstance(path, list) else [path]
    layer_data: list[tuple[Any, dict[str, Any], str]] = []
    for item in paths:
        item_str = str(item)
        image = open_image(item_str)
        layer_data.append(
            image_volume_to_layer_data(
                image=image,
                name=Path(item_str).name,
                layer_type="image",
            )
        )
    return layer_data


def write_tiff(path: str, data: Any, metadata: dict[str, Any]) -> str:
    """Write a single image layer to TIFF format using the core serializer."""
    image = image_volume_from_layer_data(data=data, metadata=metadata)
    output = save_image(image=image, destination=path, format="tiff")
    return str(output)
