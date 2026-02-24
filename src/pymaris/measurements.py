"""Distance measurement helpers for object coordinates and label maps."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist

try:  # pragma: no cover - optional dependency
    from skimage import measure

    HAS_SKIMAGE = True
except Exception:  # pragma: no cover - skimage is a required dependency in normal installs
    measure = None  # type: ignore[assignment]
    HAS_SKIMAGE = False

try:  # pragma: no cover - optional dependency
    import trimesh

    HAS_TRIMESH = True
except Exception:  # pragma: no cover - optional dependency in advanced mode
    trimesh = None  # type: ignore[assignment]
    HAS_TRIMESH = False


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


def compute_mesh_morphometrics(
    label_image: np.ndarray,
    voxel_spacing: tuple[float, float, float],
    *,
    return_vertex_data: bool = False,
) -> pd.DataFrame:
    """Compute advanced per-object 3D mesh curvature morphometrics.

    Parameters
    ----------
    label_image : np.ndarray
        3D integer label image of shape ``(Z, Y, X)``.
    voxel_spacing : tuple[float, float, float]
        Physical voxel spacing ``(z, y, x)`` used during marching-cubes extraction.
    return_vertex_data : bool, optional
        Whether to include raw per-object vertex/face and curvature arrays in object columns.

    Returns
    -------
    pd.DataFrame
        One row per label containing aggregate statistics for Gaussian curvature,
        mean curvature, and shape index.

    Raises
    ------
    ImportError
        If advanced optional dependency ``trimesh`` is unavailable.
    ValueError
        If input image is not 3D.
    """
    if not HAS_TRIMESH:
        raise ImportError(
            "Advanced mesh morphometrics requires 'trimesh'. "
            "Install extras with: pip install pymaris[advanced]"
        )

    labels = np.asarray(label_image)
    if labels.ndim != 3:
        raise ValueError("label_image must be a 3D array shaped (Z, Y, X)")

    unique_labels = np.unique(labels)
    valid_labels = [int(lbl) for lbl in unique_labels if int(lbl) > 0]
    records: list[dict[str, Any]] = []

    for label_id in valid_labels:
        binary_mask = labels == label_id
        if int(np.sum(binary_mask)) < 8:
            continue

        try:
            vertices, faces, _, _ = measure.marching_cubes(
                binary_mask.astype(np.float32),
                level=0.5,
                spacing=voxel_spacing,
            )
        except Exception:
            continue

        if vertices.shape[0] < 4 or faces.shape[0] < 2:
            continue

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mean_radius = float(np.mean(mesh.edges_unique_length)) if mesh.edges_unique_length.size > 0 else 1.0
        radius = max(mean_radius, 1e-6)

        gaussian_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
            mesh,
            mesh.vertices,
            radius=radius,
        )
        mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(
            mesh,
            mesh.vertices,
            radius=radius,
        )

        discriminant = np.maximum(mean_curvature**2 - gaussian_curvature, 0.0)
        k_sqrt = np.sqrt(discriminant)
        k1 = mean_curvature + k_sqrt
        k2 = mean_curvature - k_sqrt
        shape_index = (2.0 / np.pi) * np.arctan2(k1 + k2, k1 - k2 + 1e-12)

        record: dict[str, Any] = {
            "label": label_id,
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
            "surface_area": float(mesh.area),
            "gaussian_curvature_mean": float(np.mean(gaussian_curvature)),
            "gaussian_curvature_variance": float(np.var(gaussian_curvature)),
            "gaussian_curvature_max": float(np.max(gaussian_curvature)),
            "mean_curvature_mean": float(np.mean(mean_curvature)),
            "mean_curvature_variance": float(np.var(mean_curvature)),
            "mean_curvature_max": float(np.max(mean_curvature)),
            "shape_index_mean": float(np.mean(shape_index)),
            "shape_index_variance": float(np.var(shape_index)),
            "shape_index_max": float(np.max(shape_index)),
        }

        if return_vertex_data:
            record["mesh_vertices"] = vertices
            record["mesh_faces"] = faces
            record["gaussian_curvature_vertices"] = gaussian_curvature
            record["mean_curvature_vertices"] = mean_curvature
            record["shape_index_vertices"] = shape_index

        records.append(record)

    return pd.DataFrame.from_records(records)


def compute_inter_mesh_distances(mesh_A: Any, mesh_B: Any) -> np.ndarray:
    """Compute nearest-surface distances between two meshes.

    Parameters
    ----------
    mesh_A : trimesh.Trimesh
        Source mesh whose vertices are queried.
    mesh_B : trimesh.Trimesh
        Target mesh providing nearest-neighbor candidates.

    Returns
    -------
    np.ndarray
        Distance array of shape ``(len(mesh_A.vertices),)`` containing the nearest
        Euclidean distance from each vertex in ``mesh_A`` to ``mesh_B``.
    """
    if not hasattr(mesh_A, "vertices") or not hasattr(mesh_B, "vertices"):
        raise TypeError("mesh_A and mesh_B must expose a 'vertices' array")

    vertices_a = np.asarray(mesh_A.vertices, dtype=float)
    vertices_b = np.asarray(mesh_B.vertices, dtype=float)
    if vertices_a.ndim != 2 or vertices_b.ndim != 2 or vertices_a.shape[1] != 3 or vertices_b.shape[1] != 3:
        raise ValueError("mesh vertices must be shaped (N, 3)")
    if vertices_a.shape[0] == 0 or vertices_b.shape[0] == 0:
        return np.empty((0,), dtype=float)

    target_tree = cKDTree(vertices_b)
    distances, _ = target_tree.query(vertices_a, k=1)
    return np.asarray(distances, dtype=float)
