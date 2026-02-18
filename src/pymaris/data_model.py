"""Canonical image data model for PyMaris core workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

CANONICAL_AXES = ("T", "C", "Z", "Y", "X")

try:  # pragma: no cover - optional dependency
    import dask.array as da

    HAS_DASK = True
except Exception:  # pragma: no cover - optional dependency
    da = None  # type: ignore[assignment]
    HAS_DASK = False


def normalize_axes(axes: Sequence[str]) -> tuple[str, ...]:
    """Normalize axis labels and validate they are unique."""
    normalized = tuple(axis.upper() for axis in axes)
    if not normalized:
        raise ValueError("axes cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"axes must be unique, got {normalized!r}")
    for axis in normalized:
        if axis not in CANONICAL_AXES:
            raise ValueError(f"unsupported axis {axis!r}; expected subset of {CANONICAL_AXES}")
    return normalized


def infer_axes_from_shape(shape: Sequence[int]) -> tuple[str, ...]:
    """Infer conventional axis order from rank."""
    rank = len(shape)
    if rank == 1:
        return ("X",)
    if rank == 2:
        return ("Y", "X")
    if rank == 3:
        return ("Z", "Y", "X")
    if rank == 4:
        return ("C", "Z", "Y", "X")
    if rank == 5:
        return ("T", "C", "Z", "Y", "X")
    raise ValueError(f"cannot infer canonical axes for rank {rank}")


def squeeze_axes(array: Any, axes: Sequence[str]) -> tuple[Any, tuple[str, ...]]:
    """Drop singleton dimensions and keep axis labels in sync."""
    working = array
    working_axes = list(axes)
    for index in reversed(range(len(working_axes))):
        if getattr(working, "shape", ())[index] == 1:
            if HAS_DASK and isinstance(working, da.Array):
                working = working.squeeze(axis=index)
            else:
                working = np.squeeze(working, axis=index)
            working_axes.pop(index)
    return working, tuple(working_axes)


@dataclass
class ImageVolume:
    """Canonical microscopy image container for core and plugin adapters."""

    array: Any
    axes: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    pixel_size: dict[str, float] = field(default_factory=dict)
    axis_units: dict[str, str] = field(default_factory=dict)
    channel_names: list[str] = field(default_factory=list)
    time_spacing: float | None = None
    modality: str | None = None
    multiscale: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        self.axes = normalize_axes(self.axes)
        ndim = len(getattr(self.array, "shape", ()))
        if ndim != len(self.axes):
            raise ValueError(
                f"array rank ({ndim}) does not match axes length ({len(self.axes)}): {self.axes!r}"
            )
        for key in list(self.pixel_size.keys()):
            axis = key.upper()
            if axis not in self.axes:
                raise ValueError(f"pixel_size contains axis {axis!r} not present in {self.axes!r}")
            self.pixel_size[axis] = float(self.pixel_size[key])
            if axis != key:
                del self.pixel_size[key]
        for key in list(self.axis_units.keys()):
            axis = key.upper()
            if axis not in self.axes:
                raise ValueError(f"axis_units contains axis {axis!r} not present in {self.axes!r}")
            self.axis_units[axis] = str(self.axis_units[key])
            if axis != key:
                del self.axis_units[key]
        if self.channel_names:
            if "C" not in self.axes:
                raise ValueError("channel_names provided but image does not contain a 'C' axis")
            channel_size = int(self.shape[self.axis_index("C")])
            if channel_size > 0 and len(self.channel_names) != channel_size:
                raise ValueError(
                    f"channel_names length ({len(self.channel_names)}) does not match "
                    f"channel axis size ({channel_size})"
                )
        if self.modality is not None:
            normalized_modality = str(self.modality).strip().lower()
            self.modality = normalized_modality if normalized_modality else None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.array.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> Any:
        return getattr(self.array, "dtype", None)

    @property
    def is_lazy(self) -> bool:
        if HAS_DASK and isinstance(self.array, da.Array):
            return True
        if self.multiscale and HAS_DASK:
            return any(isinstance(level, da.Array) for level in self.multiscale)
        return False

    def axis_index(self, axis: str) -> int:
        return self.axes.index(axis.upper())

    def with_array(self, array: Any, axes: Sequence[str] | None = None) -> ImageVolume:
        """Return a shallow-copied image volume with replaced array data."""
        return ImageVolume(
            array=array,
            axes=tuple(axes) if axes is not None else self.axes,
            metadata=dict(self.metadata),
            pixel_size=dict(self.pixel_size),
            axis_units=dict(self.axis_units),
            channel_names=list(self.channel_names),
            time_spacing=self.time_spacing,
            modality=self.modality,
            multiscale=self.multiscale,
        )

    def as_numpy(self, copy: bool = False) -> np.ndarray:
        """Materialize to numpy for algorithms that require eager arrays."""
        if HAS_DASK and isinstance(self.array, da.Array):
            result = self.array.compute()
        else:
            result = np.asarray(self.array)
        if copy:
            return np.array(result, copy=True)
        return result

    def scale_for_axes(self) -> list[float]:
        """Return axis-aligned scale list suitable for napari metadata."""
        scale: list[float] = []
        for axis in self.axes:
            if axis == "T":
                scale.append(float(self.time_spacing) if self.time_spacing is not None else 1.0)
            else:
                scale.append(float(self.pixel_size.get(axis, 1.0)))
        return scale

    def metadata_dict(self) -> dict[str, Any]:
        """Return serializable metadata for persistence and layer conversion."""
        meta: dict[str, Any] = {
            "axes": list(self.axes),
            "pixel_size": dict(self.pixel_size),
            "axis_units": dict(self.axis_units),
            "channel_names": list(self.channel_names),
            "time_spacing": self.time_spacing,
            "modality": self.modality,
        }
        if self.multiscale:
            meta["multiscale_levels"] = len(self.multiscale)
        if self.metadata:
            meta["source_metadata"] = dict(self.metadata)
        return meta

    @classmethod
    def from_array(
        cls,
        array: Any,
        axes: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        axis_units: Mapping[str, str] | None = None,
        modality: str | None = None,
    ) -> ImageVolume:
        """Build a volume while inferring axes when not provided."""
        chosen_axes = tuple(axes) if axes is not None else infer_axes_from_shape(array.shape)
        return cls(
            array=array,
            axes=chosen_axes,
            metadata=dict(metadata or {}),
            axis_units={str(k): str(v) for k, v in dict(axis_units or {}).items()},
            modality=modality,
        )
