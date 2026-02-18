"""Centralized conversion helpers between ImageVolume and napari layer tuples."""

from __future__ import annotations

from typing import Any, Mapping

from pymaris.data_model import ImageVolume, infer_axes_from_shape


def image_volume_to_layer_data(
    image: ImageVolume,
    name: str | None = None,
    layer_type: str = "image",
) -> tuple[Any, dict[str, Any], str]:
    """Convert an ImageVolume to a napari-compatible layer data tuple."""
    layer_data: Any = image.multiscale if image.multiscale else image.array
    metadata = {
        "name": name or image.metadata.get("name") or "Image",
        "scale": image.scale_for_axes(),
        "metadata": image.metadata_dict(),
    }
    if image.multiscale:
        metadata["multiscale"] = True
    return layer_data, metadata, layer_type


def image_volume_from_layer_data(
    data: Any,
    metadata: Mapping[str, Any] | None = None,
) -> ImageVolume:
    """Build an ImageVolume from napari layer tuple components."""
    meta = dict(metadata or {})
    layer_meta = meta.get("metadata", {})
    if not isinstance(layer_meta, dict):
        layer_meta = {}

    multiscale_levels: tuple[Any, ...] | None = None
    primary = data
    if isinstance(data, (list, tuple)) and data:
        multiscale_levels = tuple(data)
        primary = data[0]

    declared_axes = layer_meta.get("axes") or meta.get("axes")
    if declared_axes is None:
        axes = infer_axes_from_shape(primary.shape)
    else:
        axes = tuple(str(axis).upper() for axis in declared_axes)

    pixel_size_raw = layer_meta.get("pixel_size", {})
    pixel_size = {str(key).upper(): float(value) for key, value in pixel_size_raw.items()}
    axis_units_raw = layer_meta.get("axis_units", {})
    axis_units = {str(key).upper(): str(value) for key, value in axis_units_raw.items()}
    channel_names = [str(name) for name in layer_meta.get("channel_names", [])]
    time_spacing = layer_meta.get("time_spacing")
    modality = layer_meta.get("modality")

    merged_metadata = dict(layer_meta.get("source_metadata", {}))
    for key in ("name",):
        if key in meta:
            merged_metadata[key] = meta[key]

    return ImageVolume(
        array=primary,
        axes=axes,
        metadata=merged_metadata,
        pixel_size=pixel_size,
        axis_units=axis_units,
        channel_names=channel_names,
        time_spacing=float(time_spacing) if time_spacing is not None else None,
        modality=str(modality) if modality is not None else None,
        multiscale=multiscale_levels,
    )
