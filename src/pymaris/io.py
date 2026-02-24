"""Scalable image I/O entrypoints for PyMaris core."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v3 as iio
import numpy as np
import tifffile

from pymaris.data_model import CANONICAL_AXES, ImageVolume, infer_axes_from_shape, squeeze_axes
from pymaris.logging import get_logger

LOGGER = get_logger(__name__)

SPECIALIZED_EXTENSIONS = {".czi", ".lif", ".nd2", ".oib", ".oif", ".ims", ".lsm"}
TIFF_EXTENSIONS = {".tif", ".tiff"}

try:  # pragma: no cover - optional dependency
    import dask.array as da

    HAS_DASK = True
except Exception:  # pragma: no cover - optional dependency
    da = None  # type: ignore[assignment]
    HAS_DASK = False

try:  # pragma: no cover - optional dependency
    from aicsimageio import AICSImage

    HAS_AICSIMAGEIO = True
except Exception:  # pragma: no cover - optional dependency
    AICSImage = None  # type: ignore[assignment]
    HAS_AICSIMAGEIO = False

try:  # pragma: no cover - optional dependency
    import zarr

    HAS_ZARR = True
except Exception:  # pragma: no cover - optional dependency
    zarr = None  # type: ignore[assignment]
    HAS_ZARR = False

try:  # pragma: no cover - optional dependency
    import h5py

    HAS_H5PY = True
except Exception:  # pragma: no cover - optional dependency
    h5py = None  # type: ignore[assignment]
    HAS_H5PY = False

_LAZY_FILE_SIZE_THRESHOLD = 256 * 1024 * 1024


def open_image(
    path: str | Path,
    *,
    axes: Sequence[str] | None = None,
    prefer_lazy: bool | None = None,
    chunks: Sequence[int] | None = None,
    scene: str | None = None,
    scene_index: int | None = None,
) -> ImageVolume:
    """Open a microscopy image path into an ImageVolume container."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"image path does not exist: {source}")
    if scene is not None and scene_index is not None:
        raise ValueError("pass either 'scene' or 'scene_index', not both")

    if _is_zarr_path(source):
        if scene is not None or scene_index is not None:
            raise ValueError("scene selection is not supported for Zarr inputs")
        return _open_zarr(source=source, axes=axes, prefer_lazy=prefer_lazy, chunks=chunks)

    suffix = source.suffix.lower()
    if suffix in TIFF_EXTENSIONS:
        if scene is not None or scene_index is not None:
            raise ValueError("scene selection is not supported for TIFF inputs")
        return _open_tiff(source=source, axes=axes, prefer_lazy=prefer_lazy, chunks=chunks)
    aics_error: Exception | None = None
    if suffix in SPECIALIZED_EXTENSIONS:
        if HAS_AICSIMAGEIO:
            try:
                return _open_aics(
                    source=source,
                    axes=axes,
                    prefer_lazy=prefer_lazy,
                    scene=scene,
                    scene_index=scene_index,
                )
            except Exception as exc:
                if suffix == ".ims":
                    aics_error = exc
                    LOGGER.warning(
                        "AICSImageIO failed to read %s (%s); trying h5py Imaris fallback",
                        source,
                        exc,
                    )
                else:
                    raise
        if suffix == ".ims" and HAS_H5PY:
            try:
                return _open_imaris_hdf5(
                    source=source,
                    axes=axes,
                    scene=scene,
                    scene_index=scene_index,
                )
            except Exception as h5_exc:
                if aics_error is not None:
                    raise RuntimeError(
                        f"failed to load {source}: AICSImageIO error ({aics_error}); "
                        f"h5py Imaris fallback error ({h5_exc})"
                    ) from h5_exc
                raise
        if suffix == ".ims":
            if aics_error is not None:
                raise RuntimeError(
                    f"failed to load {source}: AICSImageIO could not read this HDF5/Imaris file "
                    f"({aics_error}). Install `h5py` for direct .ims fallback support."
                ) from aics_error
            raise RuntimeError(
                "Imaris .ims support requires `aicsimageio` or `h5py`; install with optional `.[io]` extras."
            )
        LOGGER.warning(
            "AICSImageIO not available for %s, falling back to imageio; install with `.[io]`",
            source,
        )
    if scene is not None or scene_index is not None:
        raise ValueError(f"scene selection is not supported for {suffix or 'this'} input")

    data = np.asarray(iio.imread(source))
    chosen_axes = tuple(axes) if axes is not None else infer_axes_from_shape(data.shape)
    return ImageVolume(
        array=data,
        axes=chosen_axes,
        metadata={
            "name": source.name,
            "source_path": str(source),
            "reader": "imageio",
        },
    )


def list_scenes(path: str | Path) -> list[str]:
    """Return available scene identifiers for a microscopy file."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"image path does not exist: {source}")

    suffix = source.suffix.lower()
    if _is_zarr_path(source):
        return ["default"]

    if HAS_AICSIMAGEIO and suffix in SPECIALIZED_EXTENSIONS:
        try:
            image = AICSImage(source)  # type: ignore[misc]
            scenes = [str(value) for value in getattr(image, "scenes", [])]
            if scenes:
                return scenes
        except Exception as exc:
            if suffix != ".ims" or not HAS_H5PY:
                raise RuntimeError(f"failed to list scenes for {source}: {exc}") from exc
            LOGGER.warning(
                "AICSImageIO scene listing failed for %s (%s); falling back to h5py Imaris reader",
                source,
                exc,
            )

    if suffix == ".ims":
        if not HAS_H5PY:
            raise RuntimeError(
                "Imaris .ims scene listing requires `h5py` (or `aicsimageio`). Install with `.[io]`."
            )
        return _list_imaris_scenes_hdf5(source)

    return ["default"]


def save_image(
    image: ImageVolume,
    destination: str | Path,
    format: str | None = None,
    *,
    save_multiscale: bool = False,
) -> Path:
    """Save an ImageVolume in TIFF or Zarr format."""
    dest = Path(destination)
    fmt = (format or dest.suffix.lstrip(".") or "tiff").lower()

    if fmt in {"tif", "tiff"}:
        tifffile.imwrite(dest, image.as_numpy())
        return dest

    if fmt == "zarr":
        return _save_zarr(image=image, destination=dest, save_multiscale=save_multiscale)

    raise ValueError(f"unsupported output format: {fmt!r}")


def _open_tiff(
    source: Path,
    axes: Sequence[str] | None,
    prefer_lazy: bool | None,
    chunks: Sequence[int] | None,
) -> ImageVolume:
    lazy = _should_use_lazy(path=source, prefer_lazy=prefer_lazy)
    with tifffile.TiffFile(source) as tif:
        series = tif.series[0]
        raw_axes = getattr(series, "axes", "")
        if lazy:
            try:
                array_data: Any = tifffile.memmap(source)
            except Exception:
                LOGGER.warning("TIFF memmap failed for %s; falling back to eager read", source)
                array_data = series.asarray()
                lazy = False
        else:
            array_data = series.asarray()

    if lazy and HAS_DASK:
        dask_chunks = tuple(chunks) if chunks is not None else "auto"
        array_data = da.from_array(array_data, chunks=dask_chunks)

    inferred_axes = _axes_from_string(raw_axes, ndim=len(array_data.shape))
    chosen_axes = tuple(axes) if axes is not None else inferred_axes
    return ImageVolume(
        array=array_data,
        axes=chosen_axes,
        metadata={
            "name": source.name,
            "source_path": str(source),
            "reader": "tifffile",
            "tiff_axes": raw_axes,
        },
    )


def _open_aics(
    source: Path,
    axes: Sequence[str] | None,
    prefer_lazy: bool | None,
    scene: str | None,
    scene_index: int | None,
) -> ImageVolume:
    image = AICSImage(source)  # type: ignore[misc]
    available_scenes = [str(value) for value in getattr(image, "scenes", [])]
    chosen_scene = _resolve_scene_selection(
        available_scenes,
        source=source,
        scene=scene,
        scene_index=scene_index,
    )
    if chosen_scene is not None and hasattr(image, "set_scene"):
        image.set_scene(chosen_scene)
    lazy = _should_use_lazy(path=source, prefer_lazy=prefer_lazy)

    data = image.get_image_dask_data("TCZYX", S=0)
    axis_order = ("T", "C", "Z", "Y", "X")
    data, squeezed_axes = squeeze_axes(data, axis_order)

    if not lazy:
        data = data.compute() if HAS_DASK and isinstance(data, da.Array) else np.asarray(data)

    pixel_size: dict[str, float] = {}
    pps = image.physical_pixel_sizes
    if pps.Z is not None:
        pixel_size["Z"] = float(pps.Z)
    if pps.Y is not None:
        pixel_size["Y"] = float(pps.Y)
    if pps.X is not None:
        pixel_size["X"] = float(pps.X)

    axis_units = {}
    for axis_name in ("Z", "Y", "X"):
        if axis_name in pixel_size:
            axis_units[axis_name] = "micrometer"

    chosen_axes = tuple(axes) if axes is not None else squeezed_axes
    channel_names = [str(name) for name in (image.channel_names or [])]
    if "C" not in chosen_axes:
        channel_names = []
    return ImageVolume(
        array=data,
        axes=chosen_axes,
        metadata={
            "name": source.name,
            "source_path": str(source),
            "reader": "aicsimageio",
            "dims_order": image.dims.order,
            "scene": chosen_scene,
            "available_scenes": available_scenes,
        },
        pixel_size=pixel_size,
        axis_units=axis_units,
        channel_names=channel_names,
    )


def _open_imaris_hdf5(
    source: Path,
    axes: Sequence[str] | None,
    scene: str | None,
    scene_index: int | None,
) -> ImageVolume:
    if not HAS_H5PY:
        raise RuntimeError("Imaris .ims support requires optional dependency `h5py`.")

    available_scenes = _list_imaris_scenes_hdf5(source)
    selected_scene = _resolve_scene_selection(
        available_scenes,
        source=source,
        scene=scene,
        scene_index=scene_index,
    )
    if selected_scene is None:
        selected_scene = available_scenes[0]

    data, inferred_axes = _read_imaris_scene_data(source, selected_scene)
    chosen_axes = tuple(axes) if axes is not None else inferred_axes
    pixel_size, axis_units = _extract_imaris_spacing(source, axes=chosen_axes)

    return ImageVolume(
        array=data,
        axes=chosen_axes,
        metadata={
            "name": source.name,
            "source_path": str(source),
            "reader": "h5py_imaris",
            "scene": selected_scene,
            "available_scenes": available_scenes,
        },
        pixel_size=pixel_size,
        axis_units=axis_units,
    )


def _list_imaris_scenes_hdf5(source: Path) -> list[str]:
    if not HAS_H5PY:
        raise RuntimeError("h5py is required to inspect Imaris scenes")
    scenes: list[str] = []
    with h5py.File(source, mode="r") as handle:  # type: ignore[union-attr]
        if "DataSet" in handle:
            dataset_group = handle["DataSet"]
            resolution_keys = sorted(dataset_group.keys(), key=_natural_sort_key)
            for resolution_key in resolution_keys:
                resolution_group = dataset_group[resolution_key]
                if not isinstance(resolution_group, h5py.Group):  # type: ignore[attr-defined]
                    continue
                time_keys = [key for key in resolution_group.keys() if key.lower().startswith("timepoint")]
                if time_keys:
                    for time_key in sorted(time_keys, key=_natural_sort_key):
                        scenes.append(f"DataSet/{resolution_key}/{time_key}")
                else:
                    scenes.append(f"DataSet/{resolution_key}")
        if scenes:
            return scenes

        dataset_paths: list[str] = []
        _collect_dataset_paths(handle, prefix="", out=dataset_paths)
        if dataset_paths:
            return sorted(dataset_paths)
    return ["default"]


def _collect_dataset_paths(group: Any, *, prefix: str, out: list[str]) -> None:
    for key, item in group.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if HAS_H5PY and isinstance(item, h5py.Dataset):  # type: ignore[attr-defined]
            out.append(path)
        elif HAS_H5PY and isinstance(item, h5py.Group):  # type: ignore[attr-defined]
            _collect_dataset_paths(item, prefix=path, out=out)


def _read_imaris_scene_data(source: Path, selected_scene: str) -> tuple[np.ndarray, tuple[str, ...]]:
    if not HAS_H5PY:
        raise RuntimeError("h5py is required to read Imaris scenes")
    with h5py.File(source, mode="r") as handle:  # type: ignore[union-attr]
        if selected_scene == "default":
            data, has_channels = _read_first_imaris_data_node(handle)
        else:
            if selected_scene not in handle:
                raise KeyError(f"scene not found in .ims file: {selected_scene}")
            data, has_channels = _read_imaris_node(handle[selected_scene])

    if data is None:
        raise RuntimeError(f"no readable data found for scene '{selected_scene}' in {source}")
    data_array = np.asarray(data)
    return data_array, _infer_imaris_axes(data_array, has_channels=has_channels)


def _read_first_imaris_data_node(root: Any) -> tuple[np.ndarray | None, bool]:
    if HAS_H5PY and isinstance(root, h5py.Dataset):  # type: ignore[attr-defined]
        return np.asarray(root), False
    if HAS_H5PY and isinstance(root, h5py.Group):  # type: ignore[attr-defined]
        for key in sorted(root.keys(), key=_natural_sort_key):
            data, has_channels = _read_first_imaris_data_node(root[key])
            if data is not None:
                return data, has_channels
    return None, False


def _read_imaris_node(node: Any) -> tuple[np.ndarray | None, bool]:
    if HAS_H5PY and isinstance(node, h5py.Dataset):  # type: ignore[attr-defined]
        return np.asarray(node), False
    if not (HAS_H5PY and isinstance(node, h5py.Group)):  # type: ignore[attr-defined]
        return None, False

    if "Data" in node and HAS_H5PY and isinstance(node["Data"], h5py.Dataset):  # type: ignore[attr-defined]
        return np.asarray(node["Data"]), False

    channel_keys = [key for key in node.keys() if key.lower().startswith("channel")]
    if channel_keys:
        channels: list[np.ndarray] = []
        for channel_key in sorted(channel_keys, key=_natural_sort_key):
            channel_node = node[channel_key]
            channel_data, _ = _read_imaris_node(channel_node)
            if channel_data is not None:
                channels.append(np.asarray(channel_data))
        if channels:
            return np.stack(channels, axis=0), True

    for key in sorted(node.keys(), key=_natural_sort_key):
        child_data, has_channels = _read_imaris_node(node[key])
        if child_data is not None:
            return child_data, has_channels
    return None, False


def _infer_imaris_axes(data: np.ndarray, *, has_channels: bool) -> tuple[str, ...]:
    rank = data.ndim
    if not has_channels:
        return infer_axes_from_shape(data.shape)
    if rank == 3:
        return ("C", "Y", "X")
    if rank == 4:
        return ("C", "Z", "Y", "X")
    if rank == 5:
        return ("T", "C", "Z", "Y", "X")
    return infer_axes_from_shape(data.shape)


def _extract_imaris_spacing(source: Path, *, axes: Sequence[str]) -> tuple[dict[str, float], dict[str, str]]:
    if not HAS_H5PY:
        return {}, {}
    pixel_size: dict[str, float] = {}
    axis_units: dict[str, str] = {}
    with h5py.File(source, mode="r") as handle:  # type: ignore[union-attr]
        image_info = None
        if "DataSetInfo" in handle:
            info_group = handle["DataSetInfo"]
            if HAS_H5PY and isinstance(info_group, h5py.Group) and "Image" in info_group:  # type: ignore[attr-defined]
                image_info = info_group["Image"]
        if image_info is None or not hasattr(image_info, "attrs"):
            return pixel_size, axis_units

        attrs = image_info.attrs
        units_value = _decode_h5_attr(attrs.get("Unit", None))
        axis_units_source = units_value if units_value else "micrometer"

        ext_pairs = [("X", "0"), ("Y", "1"), ("Z", "2")]
        for axis_name, axis_index in ext_pairs:
            size_text = _decode_h5_attr(attrs.get(axis_name, None))
            ext_min_text = _decode_h5_attr(attrs.get(f"ExtMin{axis_index}", None))
            ext_max_text = _decode_h5_attr(attrs.get(f"ExtMax{axis_index}", None))
            try:
                size_value = float(size_text) if size_text is not None else 0.0
                ext_min_value = float(ext_min_text) if ext_min_text is not None else 0.0
                ext_max_value = float(ext_max_text) if ext_max_text is not None else 0.0
            except Exception:
                continue
            if size_value > 0:
                pixel_size[axis_name] = float(abs(ext_max_value - ext_min_value) / size_value)
                axis_units[axis_name] = axis_units_source

    filtered_pixel = {axis: value for axis, value in pixel_size.items() if axis in axes}
    filtered_units = {axis: value for axis, value in axis_units.items() if axis in axes}
    return filtered_pixel, filtered_units


def _decode_h5_attr(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore").strip()
        return text or None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        first = value.flat[0]
        return _decode_h5_attr(first)
    text = str(value).strip()
    return text or None


def _resolve_scene_selection(
    available_scenes: Sequence[str],
    *,
    source: Path,
    scene: str | None,
    scene_index: int | None,
) -> str | None:
    if scene is not None and scene_index is not None:
        raise ValueError("pass either 'scene' or 'scene_index', not both")
    scenes = [str(value) for value in available_scenes]
    if not scenes:
        if scene is not None or scene_index is not None:
            raise ValueError(f"{source} does not expose selectable scenes")
        return None
    if scene_index is not None:
        if scene_index < 0 or scene_index >= len(scenes):
            raise IndexError(f"scene_index out of range for {source}: {scene_index} (available: {len(scenes)})")
        return scenes[scene_index]
    if scene is not None:
        if scene not in scenes:
            available = ", ".join(scenes)
            raise KeyError(f"scene '{scene}' not found in {source}. Available scenes: {available}")
        return scene
    return scenes[0]


def _natural_sort_key(value: str) -> tuple[str, int]:
    match = re.search(r"(\d+)\s*$", str(value))
    if not match:
        return str(value), -1
    prefix = str(value)[: match.start(1)]
    return prefix, int(match.group(1))


def _open_zarr(
    source: Path,
    axes: Sequence[str] | None,
    prefer_lazy: bool | None,
    chunks: Sequence[int] | None,
) -> ImageVolume:
    if not HAS_ZARR:
        raise RuntimeError("zarr support is not installed. Install optional dependency set `.[io]`.")

    lazy = _should_use_lazy(path=source, prefer_lazy=prefer_lazy)
    root = zarr.open(source, mode="r")
    multiscale_levels: tuple[Any, ...] | None = None

    if hasattr(root, "attrs") and "multiscales" in root.attrs and hasattr(root, "__getitem__"):
        multiscales = root.attrs["multiscales"]
        dataset_entries = multiscales[0].get("datasets", []) if multiscales else []
        dataset_paths = [entry.get("path", str(index)) for index, entry in enumerate(dataset_entries)]
        levels: list[Any] = []
        for dataset_path in dataset_paths:
            dataset = root[dataset_path]
            levels.append(_zarr_to_array(dataset, lazy=lazy, chunks=chunks))
        if levels:
            multiscale_levels = tuple(levels)
            primary = levels[0]
        else:
            primary = _zarr_to_array(root, lazy=lazy, chunks=chunks)
    else:
        primary = _zarr_to_array(root, lazy=lazy, chunks=chunks)

    attrs = root.attrs if hasattr(root, "attrs") else {}
    ome_meta = _extract_ome_zarr_metadata(attrs)
    declared_axes = ome_meta.get("axes")
    chosen_axes = tuple(axes) if axes is not None else (
        declared_axes if declared_axes is not None else infer_axes_from_shape(primary.shape)
    )

    axis_units = {
        axis: unit
        for axis, unit in dict(ome_meta.get("axis_units", {})).items()
        if axis in chosen_axes
    }

    pixel_size, time_spacing = _scale_to_pixel_size(
        axes=(declared_axes if declared_axes is not None else chosen_axes),
        scale=ome_meta.get("scale"),
    )

    pymaris_attrs = _extract_pymaris_attrs(attrs)
    stored_pixel_size = pymaris_attrs.get("pixel_size")
    if not pixel_size and isinstance(stored_pixel_size, Mapping):
        pixel_size = {
            str(axis).upper(): float(value)
            for axis, value in dict(stored_pixel_size).items()
            if str(axis).upper() in chosen_axes
        }
    if time_spacing is None:
        stored_time = pymaris_attrs.get("time_spacing")
        if stored_time is not None:
            time_spacing = float(stored_time)
    stored_axis_units = pymaris_attrs.get("axis_units")
    if isinstance(stored_axis_units, Mapping):
        for axis, unit in dict(stored_axis_units).items():
            axis_label = str(axis).upper()
            if axis_label in chosen_axes and axis_label not in axis_units:
                axis_units[axis_label] = str(unit)

    channel_names_raw = pymaris_attrs.get("channel_names", [])
    channel_names = [str(name) for name in channel_names_raw] if isinstance(channel_names_raw, list) else []
    if "C" in chosen_axes and channel_names:
        channel_size = int(primary.shape[chosen_axes.index("C")])
        if len(channel_names) != channel_size:
            LOGGER.warning(
                "Ignoring mismatched channel_names metadata in %s (expected %d, got %d)",
                source,
                channel_size,
                len(channel_names),
            )
            channel_names = []
    else:
        channel_names = []

    source_metadata = pymaris_attrs.get("source_metadata")
    metadata = {
        "name": source.name,
        "source_path": str(source),
        "reader": "zarr",
    }
    if isinstance(source_metadata, Mapping):
        metadata.update({str(k): v for k, v in source_metadata.items()})

    return ImageVolume(
        array=primary,
        axes=chosen_axes,
        metadata=metadata,
        pixel_size=pixel_size,
        axis_units=axis_units,
        channel_names=channel_names,
        time_spacing=time_spacing,
        modality=_coerce_optional_string(pymaris_attrs.get("modality")),
        multiscale=multiscale_levels,
    )


def _save_zarr(image: ImageVolume, destination: Path, save_multiscale: bool) -> Path:
    if not HAS_ZARR:
        raise RuntimeError("zarr support is not installed. Install optional dependency set `.[io]`.")

    if destination.suffix.lower() != ".zarr":
        destination = destination.with_suffix(".zarr")
    store = zarr.open_group(destination, mode="w")
    axes_metadata = [_axis_metadata_entry(axis=axis, axis_units=image.axis_units) for axis in image.axes]

    if save_multiscale and image.multiscale:
        dataset_entries: list[dict[str, Any]] = []
        for index, level in enumerate(image.multiscale):
            key = str(index)
            _write_array_to_zarr(store=store, key=key, array=level)
            dataset_entries.append(
                {
                    "path": key,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": _scale_for_level(image=image, level=level)}
                    ],
                }
            )
        store.attrs["multiscales"] = [
            {
                "version": "0.4",
                "datasets": dataset_entries,
                "axes": axes_metadata,
            }
        ]
    else:
        _write_array_to_zarr(store=store, key="0", array=image.array)
        store.attrs["multiscales"] = [
            {
                "version": "0.4",
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": _scale_for_level(image=image, level=image.array)}
                        ],
                    }
                ],
                "axes": axes_metadata,
            }
        ]

    store.attrs["pymaris"] = image.metadata_dict()
    return destination


def _write_array_to_zarr(store: Any, key: str, array: Any) -> None:
    if HAS_DASK and isinstance(array, da.Array):
        array.to_zarr(store=store.store, component=key, overwrite=True)
    else:
        store.create_dataset(key, data=np.asarray(array), overwrite=True)


def _zarr_to_array(dataset: Any, lazy: bool, chunks: Sequence[int] | None) -> Any:
    if lazy and HAS_DASK:
        dask_chunks = tuple(chunks) if chunks is not None else "auto"
        return da.from_zarr(dataset, chunks=dask_chunks)
    return np.asarray(dataset)


def _extract_ome_zarr_metadata(attrs: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "axes": None,
        "axis_units": {},
        "scale": None,
    }
    try:
        multiscales = attrs.get("multiscales", [])
        if not multiscales:
            return result
        first = multiscales[0]
        axes = first.get("axes")
        if axes:
            normalized: list[str] = []
            axis_units: dict[str, str] = {}
            for axis in axes:
                axis_name: str
                axis_unit: str | None = None
                if isinstance(axis, Mapping):
                    axis_name = str(axis.get("name", "")).upper()
                    unit_value = axis.get("unit")
                    if unit_value is not None:
                        axis_unit = str(unit_value)
                else:
                    axis_name = str(axis).upper()
                mapped = _map_axis_label(axis_name)
                normalized.append(mapped)
                if axis_unit is not None:
                    axis_units[mapped] = axis_unit
            if normalized:
                result["axes"] = tuple(normalized)
            result["axis_units"] = axis_units

        datasets = first.get("datasets", [])
        if datasets:
            primary = _select_primary_dataset_entry(datasets)
            scale = _extract_scale_from_dataset_entry(primary)
            if scale is not None and result["axes"] is not None and len(scale) == len(result["axes"]):
                result["scale"] = scale
    except Exception:
        return result
    return result


def _extract_axes_from_ome_zarr_attrs(attrs: Any) -> tuple[str, ...] | None:
    try:
        extracted = _extract_ome_zarr_metadata(attrs)
        axes = extracted.get("axes")
        if isinstance(axes, tuple):
            return axes
        return None
    except Exception:
        return None


def _extract_pymaris_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    payload = attrs.get("pymaris", {})
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _select_primary_dataset_entry(datasets: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for entry in datasets:
        if str(entry.get("path", "")) == "0":
            return entry
    return datasets[0]


def _extract_scale_from_dataset_entry(dataset_entry: Mapping[str, Any]) -> list[float] | None:
    transforms = dataset_entry.get("coordinateTransformations", [])
    if not isinstance(transforms, list):
        return None
    for transform in transforms:
        if not isinstance(transform, Mapping):
            continue
        if str(transform.get("type", "")).lower() != "scale":
            continue
        values = transform.get("scale")
        if not isinstance(values, (list, tuple)):
            continue
        try:
            return [float(value) for value in values]
        except Exception:
            return None
    return None


def _scale_to_pixel_size(
    *,
    axes: Sequence[str],
    scale: Sequence[float] | None,
) -> tuple[dict[str, float], float | None]:
    pixel_size: dict[str, float] = {}
    time_spacing: float | None = None
    if scale is None or len(scale) != len(axes):
        return pixel_size, time_spacing
    for axis, value in zip(axes, scale):
        axis_label = axis.upper()
        if axis_label == "T":
            time_spacing = float(value)
        elif axis_label in {"X", "Y", "Z"}:
            pixel_size[axis_label] = float(value)
    return pixel_size, time_spacing


def _axis_metadata_entry(axis: str, axis_units: Mapping[str, str]) -> dict[str, str]:
    entry = {
        "name": axis.lower(),
        "type": _axis_type(axis),
    }
    unit = axis_units.get(axis)
    if unit:
        entry["unit"] = str(unit)
    return entry


def _axis_type(axis: str) -> str:
    axis_label = axis.upper()
    if axis_label == "T":
        return "time"
    if axis_label == "C":
        return "channel"
    return "space"


def _scale_for_level(image: ImageVolume, level: Any) -> list[float]:
    base_shape = image.shape
    level_shape = tuple(int(value) for value in getattr(level, "shape", base_shape))
    base_scale = image.scale_for_axes()
    scale_values: list[float] = []
    for index, axis in enumerate(image.axes):
        value = float(base_scale[index]) if index < len(base_scale) else 1.0
        if index < len(base_shape) and index < len(level_shape):
            base_dim = int(base_shape[index])
            level_dim = int(level_shape[index])
            if base_dim > 0 and level_dim > 0 and axis != "C":
                value *= float(base_dim) / float(level_dim)
        scale_values.append(value)
    return scale_values


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _map_axis_label(axis: str) -> str:
    mapping = {
        "T": "T",
        "TIME": "T",
        "C": "C",
        "CHANNEL": "C",
        "Z": "Z",
        "Y": "Y",
        "X": "X",
    }
    return mapping.get(axis.upper(), axis.upper())


def _axes_from_string(raw_axes: str, ndim: int) -> tuple[str, ...]:
    if raw_axes:
        mapped = tuple(_map_axis_label(axis) for axis in raw_axes.upper())
        if (
            len(mapped) == ndim
            and len(set(mapped)) == len(mapped)
            and all(axis in CANONICAL_AXES for axis in mapped)
        ):
            return mapped
    return infer_axes_from_shape((0,) * ndim)


def _should_use_lazy(path: Path, prefer_lazy: bool | None) -> bool:
    if prefer_lazy is not None:
        return bool(prefer_lazy)
    if not HAS_DASK:
        return False
    try:
        return path.stat().st_size >= _LAZY_FILE_SIZE_THRESHOLD
    except OSError:
        return False


def _is_zarr_path(path: Path) -> bool:
    return path.suffix.lower() == ".zarr" or path.name.lower().endswith(".ome.zarr")
