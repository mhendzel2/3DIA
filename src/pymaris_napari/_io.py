"""Napari reader/writer adapters backed by `pymaris` core I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

from pymaris.io import open_image, save_image
from pymaris.layers import image_volume_from_layer_data, image_volume_to_layer_data

SUPPORTED_EXTENSIONS = {
    ".czi",
    ".lif",
    ".nd2",
    ".nd",
    ".htd",
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
    probe_path = Path(str(probe))
    probe_lower = str(probe_path).lower()

    if probe_path.is_dir():
        if _is_zarr_directory(probe_path):
            return _read_paths
        if _discover_directory_targets(probe_path):
            return _read_paths
        _notify_no_readable_files(probe_path)
        return None

    if _is_zarr_directory(probe_path):
        return _read_paths

    suffix = probe_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS and not probe_lower.endswith(".ome.zarr"):
        return None
    return _read_paths


def _read_paths(path: str | list[str]) -> list[tuple[Any, dict[str, Any], str]]:
    paths = path if isinstance(path, list) else [path]
    resolved_paths = _resolve_input_paths(paths)

    if not resolved_paths:
        return []

    layer_data: list[tuple[Any, dict[str, Any], str]] = []
    for item in resolved_paths:
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


def _resolve_input_paths(paths: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for item in paths:
        candidate = Path(str(item))
        if candidate.is_dir() and not _is_zarr_directory(candidate):
            resolved.extend(_discover_directory_targets(candidate))
            continue
        resolved.append(candidate)
    return resolved


def _discover_directory_targets(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []

    nd_candidates = sorted(directory.glob("*.nd"))
    if nd_candidates:
        return nd_candidates[:1]

    htd_candidates = sorted(directory.glob("*.htd"))
    if htd_candidates:
        return htd_candidates[:1]

    generic_candidates = sorted(
        child
        for child in directory.iterdir()
        if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return generic_candidates


def _is_zarr_directory(path: Path) -> bool:
    lower = str(path).lower()
    return lower.endswith(".zarr") or lower.endswith(".ome.zarr")


def _notify_no_readable_files(directory: Path) -> None:
    message = f"No readable microscopy files were found in folder: {directory}"
    try:
        from napari.utils.notifications import show_warning

        show_warning(message)
    except Exception:
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def write_tiff(path: str, data: Any, metadata: dict[str, Any]) -> str:
    """Write a single image layer to TIFF format using the core serializer."""
    image = image_volume_from_layer_data(data=data, metadata=metadata)
    output = save_image(image=image, destination=path, format="tiff")
    return str(output)
