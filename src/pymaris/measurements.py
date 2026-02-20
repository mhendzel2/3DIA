"""Distance measurement helpers for object coordinates and label maps."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.spatial.distance import cdist, pdist

try:  # pragma: no cover - optional dependency
    from skimage import measure

    HAS_SKIMAGE = True
except Exception:  # pragma: no cover - skimage is a required dependency in normal installs
    measure = None  # type: ignore[assignment]
    HAS_SKIMAGE = False


def calculate_distance_measurements(
    coordinates: Any,
    *,
    spacing: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute pairwise and nearest-neighbor distance summaries.

    Parameters
    ----------
    coordinates:
        Array-like of shape ``(N, D)`` with points in pixel/grid units.
    spacing:
        Optional per-axis spacing values used to scale coordinates into
        physical units before measuring distances.
    """
    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2:
        raise ValueError("coordinates must be a 2D array shaped (N, D)")
    if points.shape[0] < 2:
        return {
            "point_count": int(points.shape[0]),
            "distance_measurements_available": False,
        }

    scaled = _apply_spacing(points, spacing=spacing)
    pairwise = pdist(scaled)
    matrix = cdist(scaled, scaled)
    np.fill_diagonal(matrix, np.inf)
    nearest = np.min(matrix, axis=1)

    return {
        "point_count": int(points.shape[0]),
        "distance_measurements_available": True,
        "mean_distance": float(np.mean(pairwise)),
        "std_distance": float(np.std(pairwise)),
        "min_distance": float(np.min(pairwise)),
        "max_distance": float(np.max(pairwise)),
        "median_distance": float(np.median(pairwise)),
        "mean_nn_distance": float(np.mean(nearest)),
        "std_nn_distance": float(np.std(nearest)),
        "min_nn_distance": float(np.min(nearest)),
        "max_nn_distance": float(np.max(nearest)),
    }


def centroid_distance_measurements(
    labels: Any,
    *,
    spacing: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Calculate distance summaries from centroids in a label image."""
    labels_array = np.asarray(labels)
    centroids = _label_centroids(labels_array)
    if centroids.size == 0:
        return {"object_count": 0, "distance_measurements_available": False}
    stats = calculate_distance_measurements(centroids, spacing=spacing)
    stats["object_count"] = int(centroids.shape[0])
    return stats


def _label_centroids(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.empty((0, 0), dtype=float)
    if HAS_SKIMAGE:
        centroids = [tuple(float(value) for value in region.centroid) for region in measure.regionprops(labels)]
        if not centroids:
            return np.empty((0, labels.ndim), dtype=float)
        return np.asarray(centroids, dtype=float)

    centroids = []
    for label_id in np.unique(labels):
        if int(label_id) <= 0:
            continue
        indices = np.argwhere(labels == label_id)
        if indices.size == 0:
            continue
        centroids.append(indices.mean(axis=0))
    if not centroids:
        return np.empty((0, labels.ndim), dtype=float)
    return np.asarray(centroids, dtype=float)


def _apply_spacing(points: np.ndarray, *, spacing: Sequence[float] | None) -> np.ndarray:
    if spacing is None:
        return points
    spacing_array = np.asarray(spacing, dtype=float)
    if spacing_array.ndim != 1:
        raise ValueError("spacing must be a 1D sequence")
    if spacing_array.shape[0] != points.shape[1]:
        raise ValueError(
            f"spacing length ({spacing_array.shape[0]}) must match coordinate dims ({points.shape[1]})"
        )
    return points * spacing_array[np.newaxis, :]
