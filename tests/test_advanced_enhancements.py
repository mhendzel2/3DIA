"""Tests for advanced morphometric, heterogeneity, and trajectory enhancements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymaris.advanced_analysis import compare_distributions_wasserstein, compute_clark_evans_3d
from pymaris.measurements import compute_mesh_morphometrics
from pymaris.timelapse_processor import cluster_morphokinetic_trajectories


def _has_module(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_module("trimesh"), reason="trimesh not installed")
def test_mesh_curvatures() -> None:
    radius = 8
    shape = (32, 32, 32)
    zz, yy, xx = np.indices(shape)
    center = np.array([16, 16, 16])
    sphere = ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) <= radius**2
    labels = sphere.astype(np.uint16)

    result = compute_mesh_morphometrics(labels, voxel_spacing=(1.0, 1.0, 1.0), return_vertex_data=True)
    assert not result.empty

    expected_area = 4.0 * np.pi * radius**2
    measured_area = float(result.loc[0, "surface_area"])
    assert measured_area > 0
    assert abs(measured_area - expected_area) / expected_area < 0.45

    gaussian_mean = float(result.loc[0, "gaussian_curvature_mean"])
    gaussian_var = float(result.loc[0, "gaussian_curvature_variance"])
    assert gaussian_mean > 0
    assert gaussian_var >= 0


@pytest.mark.skipif(not (_has_module("ot") and _has_module("sklearn")), reason="POT/scikit-learn not installed")
def test_wasserstein_distance() -> None:
    data = pd.DataFrame({"f1": [0.0, 1.0, 2.0], "f2": [1.0, 2.0, 3.0]})
    identical = compare_distributions_wasserstein(data, data.copy(), ["f1", "f2"])
    assert identical == pytest.approx(0.0, abs=1e-8)

    shifted = pd.DataFrame({"f1": [10.0, 11.0, 12.0], "f2": [20.0, 21.0, 22.0]})
    disjoint = compare_distributions_wasserstein(data, shifted, ["f1", "f2"])
    assert disjoint > 0.0


def test_clark_evans() -> None:
    grid_coords = np.stack(np.meshgrid(np.arange(5), np.arange(5), np.arange(5), indexing="ij"), axis=-1).reshape(-1, 3)
    r_grid = compute_clark_evans_3d(grid_coords.astype(float), (6.0, 6.0, 6.0))
    assert r_grid > 1.0

    cluster_a = np.random.default_rng(42).normal(loc=[2.0, 2.0, 2.0], scale=0.1, size=(60, 3))
    cluster_b = np.random.default_rng(43).normal(loc=[4.0, 4.0, 4.0], scale=0.1, size=(60, 3))
    clustered = np.vstack([cluster_a, cluster_b])
    r_clustered = compute_clark_evans_3d(clustered, (6.0, 6.0, 6.0))
    assert r_clustered < 1.0


@pytest.mark.skipif(
    not (_has_module("umap") and _has_module("hdbscan") and _has_module("sklearn")),
    reason="umap/hdbscan/scikit-learn not installed",
)
def test_trajectory_clustering() -> None:
    rows = []
    for track_id in range(6):
        for time in range(6):
            rows.append(
                {
                    "track_id": track_id,
                    "time": time,
                    "area": 50 + 10 * time,
                    "y": float(time * 2.0 + track_id * 0.2),
                    "x": float(time * 2.0 + track_id * 0.2),
                }
            )

    for track_id in range(6, 12):
        for time in range(6):
            rows.append(
                {
                    "track_id": track_id,
                    "time": time,
                    "area": 130 - 8 * time,
                    "y": float(20.0 + np.sin(time * 0.2) + (track_id - 6) * 0.1),
                    "x": float(20.0 + np.cos(time * 0.2) + (track_id - 6) * 0.1),
                }
            )

    tracking_df = pd.DataFrame(rows)
    clustered = cluster_morphokinetic_trajectories(tracking_df, ["area", "y", "x"])

    assert "Behavioral_State_ID" in clustered.columns
    by_track = clustered[["track_id", "Behavioral_State_ID"]].drop_duplicates()
    distinct = {int(state) for state in by_track["Behavioral_State_ID"].unique().tolist() if int(state) >= 0}
    assert len(distinct) >= 2
