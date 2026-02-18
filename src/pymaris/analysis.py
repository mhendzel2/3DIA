"""Headless analysis API wrappers.

This module composes existing repository functionality from
``utils.analysis_utils`` behind a napari-free interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import ndimage

from pymaris.measurements import calculate_distance_measurements as _distance_stats
from pymaris.measurements import centroid_distance_measurements as _centroid_distance_stats
from utils import analysis_utils as _legacy_analysis


def load_image(file_path: str) -> Any:
    """Load an image into a numpy-like array via legacy analysis utilities."""
    return _legacy_analysis.load_image(file_path)


def segment_cellpose(image: Any, diameter: int = 30) -> Any:
    """Run Cellpose segmentation when the optional backend is available."""
    return _legacy_analysis.segment_cellpose(image=image, diameter=diameter)


def segment_stardist(image: Any) -> Any:
    """Run StarDist segmentation when the optional backend is available."""
    return _legacy_analysis.segment_stardist(image=image)


def segment_watershed(image: Any) -> Any:
    """Run watershed segmentation."""
    return _legacy_analysis.segment_watershed(image=image)


def calculate_object_statistics(
    labeled_image: Any,
    intensity_image: Any | None = None,
    properties: list[str] | None = None,
    voxel_size: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Return per-object measurements and summary statistics."""
    return _legacy_analysis.calculate_object_statistics(
        labeled_image=labeled_image,
        intensity_image=intensity_image,
        properties=properties,
        voxel_size=voxel_size,
    )


def calculate_colocalization_coefficients(
    image1: Any,
    image2: Any,
    threshold1: float | None = None,
    threshold2: float | None = None,
) -> dict[str, Any]:
    """Return colocalization coefficients for two channels."""
    return _legacy_analysis.calculate_colocalization_coefficients(
        image1=image1,
        image2=image2,
        threshold1=threshold1,
        threshold2=threshold2,
    )


def calculate_distance_measurements(
    coordinates: Any,
    *,
    spacing: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return distance summaries for point coordinates."""
    return _distance_stats(coordinates=coordinates, spacing=spacing)


def calculate_label_distance_measurements(
    labeled_image: Any,
    *,
    spacing: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return centroid-based distance summaries for a label image."""
    return _centroid_distance_stats(labels=np.asarray(labeled_image), spacing=spacing)


def generate_euclidean_distance_map(
    image: Any,
    *,
    axes: Sequence[str] | None = None,
    threshold: float = 0.0,
    spacing: Sequence[float] | None = None,
    distance_to: str = "background",
    absolute: bool = False,
) -> np.ndarray:
    """Generate an Euclidean distance transform map from an intensity image.

    The input image is thresholded into a binary mask first. By default, the
    resulting map represents distance from foreground voxels to background.
    """
    data = np.asarray(image)
    if data.ndim < 1:
        raise ValueError("image must be at least 1D")

    axis_labels = _normalize_axes(axes=axes, ndim=data.ndim)
    spatial_indices = [index for index, axis in enumerate(axis_labels) if axis in {"Z", "Y", "X"}]
    if not spatial_indices:
        spatial_indices = list(range(data.ndim))

    basis = np.abs(data) if absolute else data
    mask = np.asarray(basis > float(threshold), dtype=bool)
    mode = str(distance_to).strip().lower()
    if mode in {"background", "outside", "edge"}:
        target = mask
    elif mode in {"foreground", "inside", "object"}:
        target = ~mask
    else:
        raise ValueError("distance_to must be either 'background' or 'foreground'")

    spatial_spacing = _normalize_spacing(spacing=spacing, spatial_ndim=len(spatial_indices))
    return _distance_transform_over_spatial_axes(
        mask=target,
        spatial_indices=spatial_indices,
        spatial_spacing=spatial_spacing,
    )


def _normalize_axes(axes: Sequence[str] | None, *, ndim: int) -> tuple[str, ...]:
    if axes is None:
        return tuple(f"D{index}" for index in range(ndim))
    normalized = tuple(str(axis).upper() for axis in axes)
    if len(normalized) != ndim:
        raise ValueError(f"axes length ({len(normalized)}) must match image rank ({ndim})")
    return normalized


def _normalize_spacing(
    spacing: Sequence[float] | None,
    *,
    spatial_ndim: int,
) -> tuple[float, ...] | None:
    if spacing is None:
        return None
    values = np.asarray(spacing, dtype=float).reshape(-1)
    if values.shape[0] != spatial_ndim:
        raise ValueError(
            f"spacing length ({values.shape[0]}) must match spatial dimensions ({spatial_ndim})"
        )
    if np.any(values <= 0):
        raise ValueError("spacing values must be positive")
    return tuple(float(value) for value in values)


def _distance_transform_over_spatial_axes(
    *,
    mask: np.ndarray,
    spatial_indices: Sequence[int],
    spatial_spacing: Sequence[float] | None,
) -> np.ndarray:
    ndim = int(mask.ndim)
    spatial = [int(value) for value in spatial_indices]
    if set(spatial) == set(range(ndim)):
        return np.asarray(ndimage.distance_transform_edt(mask, sampling=spatial_spacing), dtype=float)

    non_spatial = [index for index in range(ndim) if index not in spatial]
    permutation = non_spatial + spatial
    transposed = np.transpose(mask, axes=permutation)
    spatial_shape = transposed.shape[len(non_spatial) :]
    flat = transposed.reshape((-1,) + spatial_shape)

    transformed = np.zeros_like(flat, dtype=float)
    for index in range(flat.shape[0]):
        transformed[index] = np.asarray(
            ndimage.distance_transform_edt(flat[index], sampling=spatial_spacing),
            dtype=float,
        )

    restored = transformed.reshape(transposed.shape)
    inverse_permutation = np.argsort(permutation)
    return np.transpose(restored, axes=inverse_permutation)
