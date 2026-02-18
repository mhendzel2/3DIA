"""Serializable workflow steps that execute through backend registries."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Mapping

import numpy as np

from pymaris.backends import DEFAULT_REGISTRY, BackendRegistry
from pymaris.backends.types import ImageResult, RestorationBackend
from pymaris.data_model import ImageVolume
from pymaris.logging import get_logger

LOGGER = get_logger(__name__)

ProgressCallback = Callable[[int, str], None]


class WorkflowCancelledError(RuntimeError):
    """Raised when a workflow step is cancelled."""


class WorkflowResourceLimitError(RuntimeError):
    """Raised when an execution resource limit is exceeded."""


@dataclass
class WorkflowResult:
    """Unified output for an executed workflow step."""

    outputs: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStep:
    """Serializable step that maps UI operations to backend calls."""

    id: str
    name: str
    backend_type: str
    backend_name: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "backend_type": self.backend_type,
            "backend_name": self.backend_name,
            "params": dict(self.params),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkflowStep:
        return cls(
            id=str(payload["id"]),
            name=str(payload.get("name", payload["id"])),
            backend_type=str(payload["backend_type"]),
            backend_name=str(payload["backend_name"]),
            params=dict(payload.get("params", {})),
            inputs=[str(item) for item in payload.get("inputs", [])],
            outputs=[str(item) for item in payload.get("outputs", [])],
        )

    def run(
        self,
        context: Mapping[str, Any],
        *,
        registry: BackendRegistry | None = None,
        on_progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        resource_limits: Mapping[str, Any] | None = None,
    ) -> WorkflowResult:
        target_registry = registry or DEFAULT_REGISTRY
        progress = on_progress or _noop_progress
        progress(5, f"{self.name}: validating inputs")
        self._raise_if_cancelled(cancel_event)

        if self.backend_type == "segmentation":
            result = self._run_segmentation(context, target_registry, progress, resource_limits=resource_limits)
        elif self.backend_type == "tracking":
            result = self._run_tracking(context, target_registry, progress, resource_limits=resource_limits)
        elif self.backend_type == "tracing":
            result = self._run_tracing(context, target_registry, progress, resource_limits=resource_limits)
        elif self.backend_type == "restoration":
            result = self._run_restoration(context, target_registry, progress, resource_limits=resource_limits)
        else:
            raise ValueError(f"unsupported backend_type: {self.backend_type!r}")

        self._raise_if_cancelled(cancel_event)
        progress(100, f"{self.name}: completed")
        return result

    def _run_segmentation(
        self,
        context: Mapping[str, Any],
        registry: BackendRegistry,
        progress: ProgressCallback,
        *,
        resource_limits: Mapping[str, Any] | None,
    ) -> WorkflowResult:
        input_key = self.inputs[0] if self.inputs else "image"
        output_key = self.outputs[0] if self.outputs else "labels"
        image = _expect_image_volume(context[input_key], input_key)
        _enforce_memory_budget(value=image, resource_limits=resource_limits, step_name=self.name, input_key=input_key)
        progress(40, f"{self.name}: running segmentation backend '{self.backend_name}'")
        result = registry.get_segmentation(self.backend_name).segment_instances(image, **self.params)
        return WorkflowResult(
            outputs={output_key: result.labels},
            tables={f"{output_key}_table": result.table},
            metadata=dict(result.metadata),
        )

    def _run_tracking(
        self,
        context: Mapping[str, Any],
        registry: BackendRegistry,
        progress: ProgressCallback,
        *,
        resource_limits: Mapping[str, Any] | None,
    ) -> WorkflowResult:
        input_key = self.inputs[0] if self.inputs else "labels_sequence"
        output_key = self.outputs[0] if self.outputs else "tracks"
        labels_sequence = context[input_key]
        if not isinstance(labels_sequence, (list, tuple)):
            raise TypeError(
                f"tracking input '{input_key}' must be a list/tuple of label arrays; "
                f"got {type(labels_sequence).__name__}"
            )
        _enforce_memory_budget(
            value=labels_sequence,
            resource_limits=resource_limits,
            step_name=self.name,
            input_key=input_key,
        )
        progress(40, f"{self.name}: running tracking backend '{self.backend_name}'")
        result = registry.get_tracking(self.backend_name).track(labels_sequence, **self.params)
        return WorkflowResult(
            outputs={output_key: result.tracks},
            tables={f"{output_key}_table": result.table},
            metadata=dict(result.metadata),
        )

    def _run_tracing(
        self,
        context: Mapping[str, Any],
        registry: BackendRegistry,
        progress: ProgressCallback,
        *,
        resource_limits: Mapping[str, Any] | None,
    ) -> WorkflowResult:
        input_key = self.inputs[0] if self.inputs else "image"
        output_key = self.outputs[0] if self.outputs else "trace"
        image = _expect_image_volume(context[input_key], input_key)
        _enforce_memory_budget(value=image, resource_limits=resource_limits, step_name=self.name, input_key=input_key)
        progress(40, f"{self.name}: running tracing backend '{self.backend_name}'")
        result = registry.get_tracing(self.backend_name).trace(image, **self.params)
        return WorkflowResult(
            outputs={output_key: result.graph},
            tables={f"{output_key}_table": result.table},
            metadata=dict(result.metadata),
        )

    def _run_restoration(
        self,
        context: Mapping[str, Any],
        registry: BackendRegistry,
        progress: ProgressCallback,
        *,
        resource_limits: Mapping[str, Any] | None,
    ) -> WorkflowResult:
        input_key = self.inputs[0] if self.inputs else "image"
        output_key = self.outputs[0] if self.outputs else "restored_image"
        image = _expect_image_volume(context[input_key], input_key)
        params = dict(self.params)
        operation = str(params.pop("operation", "denoise")).strip().lower()
        tiling = _pop_restoration_tiling_config(params=params, image=image)
        estimated_nbytes = (
            _estimate_tiled_input_nbytes(image=image, tile_shape=tiling["tile_shape"]) if tiling is not None else None
        )
        _enforce_memory_budget(
            value=image,
            resource_limits=resource_limits,
            step_name=self.name,
            input_key=input_key,
            estimated_nbytes=estimated_nbytes,
        )
        backend = registry.get_restoration(self.backend_name)
        if tiling is not None:
            progress(20, f"{self.name}: preparing tiled restoration run")
            result = _run_restoration_tiled(
                backend=backend,
                image=image,
                operation=operation,
                params=params,
                tile_shape=tiling["tile_shape"],
                tile_overlap=tiling["tile_overlap"],
            )
        else:
            progress(40, f"{self.name}: running restoration backend '{self.backend_name}' ({operation})")
            result = _invoke_restoration_backend(backend=backend, image=image, operation=operation, params=params)
        return WorkflowResult(
            outputs={output_key: result.image},
            tables={f"{output_key}_table": result.table},
            metadata=dict(result.metadata),
        )

    def _raise_if_cancelled(self, cancel_event: threading.Event | None) -> None:
        if cancel_event and cancel_event.is_set():
            raise WorkflowCancelledError(f"workflow step cancelled: {self.id}")


def _expect_image_volume(value: Any, key: str) -> ImageVolume:
    if not isinstance(value, ImageVolume):
        raise TypeError(
            f"workflow input '{key}' must be an ImageVolume; got {type(value).__name__}"
        )
    return value


def _noop_progress(_: int, __: str) -> None:
    return None


def _enforce_memory_budget(
    *,
    value: Any,
    resource_limits: Mapping[str, Any] | None,
    step_name: str,
    input_key: str,
    estimated_nbytes: int | None = None,
) -> None:
    budget_mb = _memory_budget_mb(resource_limits)
    if budget_mb is None:
        return
    estimated_bytes = estimated_nbytes if estimated_nbytes is not None else _estimate_value_nbytes(value)
    if estimated_bytes is None:
        return
    budget_bytes = max(1, int(budget_mb * 1024 * 1024))
    if estimated_bytes > budget_bytes:
        raise WorkflowResourceLimitError(
            f"{step_name}: memory budget exceeded for input '{input_key}' "
            f"({estimated_bytes / (1024 * 1024):.2f} MiB > {budget_mb:.2f} MiB)"
        )


def _memory_budget_mb(resource_limits: Mapping[str, Any] | None) -> float | None:
    if not resource_limits:
        return None
    raw = resource_limits.get("memory_budget_mb")
    if raw is None:
        return None
    value = float(raw)
    if value <= 0:
        raise ValueError("resource limit 'memory_budget_mb' must be > 0")
    return value


def _estimate_value_nbytes(value: Any) -> int | None:
    if isinstance(value, ImageVolume):
        return _estimate_array_nbytes(value.array)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (list, tuple)):
        total = 0
        seen = False
        for item in value:
            item_bytes = _estimate_value_nbytes(item)
            if item_bytes is None:
                continue
            total += int(item_bytes)
            seen = True
        return total if seen else None
    return _estimate_array_nbytes(value)


def _estimate_array_nbytes(array: Any) -> int | None:
    raw_nbytes = getattr(array, "nbytes", None)
    if raw_nbytes is not None:
        try:
            value = int(raw_nbytes)
            if value >= 0:
                return value
        except Exception:
            pass

    shape = getattr(array, "shape", None)
    dtype = getattr(array, "dtype", None)
    if shape is None or dtype is None:
        return None
    try:
        itemsize = int(np.dtype(dtype).itemsize)
        if itemsize <= 0:
            return None
        elements = math.prod(int(dim) for dim in shape)
    except Exception:
        return None
    if elements < 0:
        return None
    return int(elements * itemsize)


def _invoke_restoration_backend(
    *,
    backend: RestorationBackend,
    image: ImageVolume,
    operation: str,
    params: Mapping[str, Any],
) -> ImageResult:
    backend_params = dict(params)
    if operation == "deconvolve":
        return backend.deconvolve(image, **backend_params)
    if operation in {"distance_map", "euclidean_distance_map", "edt"}:
        backend_params.setdefault("method", "distance_map")
        return backend.denoise(image, **backend_params)
    return backend.denoise(image, **backend_params)


def _pop_restoration_tiling_config(
    *,
    params: dict[str, Any],
    image: ImageVolume,
) -> dict[str, Any] | None:
    raw_tiling = params.pop("tiling", None)
    raw_tile_shape = params.pop("tile_shape", None)
    raw_tile_overlap = params.pop("tile_overlap", None)
    if isinstance(raw_tiling, Mapping):
        if raw_tile_shape is None:
            raw_tile_shape = raw_tiling.get("tile_shape")
        if raw_tile_overlap is None:
            raw_tile_overlap = raw_tiling.get("overlap")

    if raw_tile_shape is None:
        return None

    spatial_indices = _spatial_indices_for_tiling(image.axes)
    tile_shape = _normalize_spatial_shape_argument(raw_tile_shape, image=image, spatial_indices=spatial_indices)
    tile_overlap = _normalize_overlap_argument(
        raw_tile_overlap,
        image=image,
        spatial_indices=spatial_indices,
    )
    if all(tile_dim >= image.shape[axis] for tile_dim, axis in zip(tile_shape, spatial_indices)):
        return None
    return {
        "spatial_indices": spatial_indices,
        "tile_shape": tile_shape,
        "tile_overlap": tile_overlap,
    }


def _spatial_indices_for_tiling(axes: tuple[str, ...]) -> tuple[int, ...]:
    indices = tuple(index for index, axis in enumerate(axes) if axis in {"Z", "Y", "X"})
    if indices:
        return indices
    return tuple(range(len(axes)))


def _normalize_spatial_shape_argument(
    raw: Any,
    *,
    image: ImageVolume,
    spatial_indices: tuple[int, ...],
) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError("tile_shape must be an array of integers")
    values = [int(value) for value in raw]
    if any(value <= 0 for value in values):
        raise ValueError("tile_shape values must be > 0")
    if len(values) == len(spatial_indices):
        return tuple(values)
    if len(values) == image.ndim:
        return tuple(values[index] for index in spatial_indices)
    raise ValueError(
        f"tile_shape rank mismatch: expected {len(spatial_indices)} spatial dims or {image.ndim} full dims, "
        f"got {len(values)}"
    )


def _normalize_overlap_argument(
    raw: Any,
    *,
    image: ImageVolume,
    spatial_indices: tuple[int, ...],
) -> tuple[int, ...]:
    if raw is None:
        return tuple(0 for _ in spatial_indices)
    if isinstance(raw, (int, float)):
        overlap = int(raw)
        if overlap < 0:
            raise ValueError("tile_overlap must be >= 0")
        return tuple(overlap for _ in spatial_indices)
    if isinstance(raw, (list, tuple)):
        values = [int(value) for value in raw]
        if any(value < 0 for value in values):
            raise ValueError("tile_overlap values must be >= 0")
        if len(values) == len(spatial_indices):
            return tuple(values)
        if len(values) == image.ndim:
            return tuple(values[index] for index in spatial_indices)
    raise ValueError("tile_overlap must be an integer or array matching spatial dimensions")


def _estimate_tiled_input_nbytes(*, image: ImageVolume, tile_shape: tuple[int, ...]) -> int | None:
    if image.dtype is None:
        return None
    shape = image.shape
    spatial_indices = _spatial_indices_for_tiling(image.axes)
    effective_shape = list(shape)
    for tile_dim, axis in zip(tile_shape, spatial_indices):
        effective_shape[axis] = min(int(tile_dim), int(shape[axis]))
    array_like = np.empty(tuple(effective_shape), dtype=image.dtype)
    return _estimate_array_nbytes(array_like)


def _run_restoration_tiled(
    *,
    backend: RestorationBackend,
    image: ImageVolume,
    operation: str,
    params: Mapping[str, Any],
    tile_shape: tuple[int, ...],
    tile_overlap: tuple[int, ...],
) -> ImageResult:
    spatial_indices = _spatial_indices_for_tiling(image.axes)
    full_shape = image.shape
    start_values = [
        list(range(0, int(full_shape[axis]), max(1, int(tile_dim))))
        for axis, tile_dim in zip(spatial_indices, tile_shape)
    ]
    output_array: np.ndarray | None = None
    backend_metadata: dict[str, Any] = {}
    backend_table: dict[str, Any] = {}
    tile_count = 0

    for starts in product(*start_values):
        core_ranges: list[tuple[int, int]] = []
        expanded_ranges: list[tuple[int, int]] = []
        for axis, start, core_dim, overlap in zip(spatial_indices, starts, tile_shape, tile_overlap):
            stop = min(int(full_shape[axis]), int(start + core_dim))
            halo_start = max(0, int(start - overlap))
            halo_stop = min(int(full_shape[axis]), int(stop + overlap))
            core_ranges.append((int(start), int(stop)))
            expanded_ranges.append((halo_start, halo_stop))

        input_slices = [slice(None)] * image.ndim
        output_slices = [slice(None)] * image.ndim
        tile_inner_slices = [slice(None)] * image.ndim
        for axis, core_range, expanded_range in zip(spatial_indices, core_ranges, expanded_ranges):
            core_start, core_stop = core_range
            halo_start, halo_stop = expanded_range
            input_slices[axis] = slice(halo_start, halo_stop)
            output_slices[axis] = slice(core_start, core_stop)
            tile_inner_slices[axis] = slice(core_start - halo_start, core_stop - halo_start)

        tile_data = np.asarray(image.array[tuple(input_slices)])
        tile_image = image.with_array(tile_data)
        tile_result = _invoke_restoration_backend(backend=backend, image=tile_image, operation=operation, params=params)
        tile_output = np.asarray(tile_result.image.as_numpy())

        if output_array is None:
            output_array = np.zeros(full_shape, dtype=tile_output.dtype)
        output_array[tuple(output_slices)] = tile_output[tuple(tile_inner_slices)]
        backend_metadata = dict(tile_result.metadata)
        backend_table = dict(tile_result.table)
        tile_count += 1

    if output_array is None:
        raise RuntimeError("tiled restoration produced no output tiles")

    metadata = dict(backend_metadata)
    metadata["tiled_execution"] = {
        "enabled": True,
        "tile_shape": list(tile_shape),
        "tile_overlap": list(tile_overlap),
        "spatial_axes": [image.axes[index] for index in spatial_indices],
        "tile_count": tile_count,
    }
    return ImageResult(image=image.with_array(output_array), table=backend_table, metadata=metadata)
