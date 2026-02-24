"""Advanced heterogeneity, spatial, and topological analytics for pymaris."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree  # type: ignore[attr-defined]

HAS_POT = importlib.util.find_spec("ot") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_RIPSER = importlib.util.find_spec("ripser") is not None


def compare_distributions_wasserstein(
    features_A: pd.DataFrame,
    features_B: pd.DataFrame,
    feature_cols: list[str],
) -> float:
    """Compare two multi-feature populations using Earth Mover's Distance.

    Parameters
    ----------
    features_A : pd.DataFrame
        DataFrame for population A.
    features_B : pd.DataFrame
        DataFrame for population B.
    feature_cols : list[str]
        Feature columns used to define the morphometric space.

    Returns
    -------
    float
        Squared Earth Mover's Distance ($W_2^2$) between the two distributions.

    Raises
    ------
    ImportError
        If optional dependencies ``POT`` and ``scikit-learn`` are unavailable.
    ValueError
        If required columns are missing or either population is empty.
    """
    if not HAS_POT or not HAS_SKLEARN:
        raise ImportError(
            "Wasserstein comparison requires 'POT' and 'scikit-learn'. "
            "Install extras with: pip install pymaris[advanced]"
        )
    ot = importlib.import_module("ot")
    from sklearn.preprocessing import StandardScaler

    _validate_feature_columns(features_A, feature_cols)
    _validate_feature_columns(features_B, feature_cols)

    array_a = features_A[feature_cols].to_numpy(dtype=float, copy=True)
    array_b = features_B[feature_cols].to_numpy(dtype=float, copy=True)
    if array_a.shape[0] == 0 or array_b.shape[0] == 0:
        raise ValueError("features_A and features_B must each contain at least one row")

    combined = np.vstack([array_a, array_b])
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    split_idx = array_a.shape[0]
    scaled_a = combined_scaled[:split_idx, :]
    scaled_b = combined_scaled[split_idx:, :]

    weights_a = np.full(scaled_a.shape[0], 1.0 / scaled_a.shape[0], dtype=float)
    weights_b = np.full(scaled_b.shape[0], 1.0 / scaled_b.shape[0], dtype=float)

    cost_matrix = ot.dist(scaled_a, scaled_b, metric="euclidean")
    wasserstein_distance = ot.emd2(weights_a, weights_b, cost_matrix)
    return float(wasserstein_distance)


def identify_subpopulations_gmm(
    dataframe: pd.DataFrame,
    feature_cols: list[str],
    max_components: int = 5,
) -> pd.DataFrame:
    """Identify latent subpopulations with Gaussian mixture models and BIC.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input measurements table.
    feature_cols : list[str]
        Numeric feature columns used for clustering.
    max_components : int, optional
        Upper bound for tested Gaussian mixture component counts.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with added integer column ``Subpopulation_ID``.

    Raises
    ------
    ImportError
        If ``scikit-learn`` is unavailable.
    ValueError
        If input data is empty, columns are missing, or component bounds are invalid.
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "GMM subpopulation discovery requires 'scikit-learn'. "
            "Install extras with: pip install pymaris[advanced]"
        )
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    if max_components < 1:
        raise ValueError("max_components must be >= 1")

    _validate_feature_columns(dataframe, feature_cols)
    features = dataframe[feature_cols].to_numpy(dtype=float, copy=True)
    if features.shape[0] < 2:
        raise ValueError("At least two rows are required for GMM clustering")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    component_limit = min(max_components, scaled.shape[0])
    best_model = None
    best_bic = np.inf

    for component_count in range(1, component_limit + 1):
        model = GaussianMixture(
            n_components=component_count,
            covariance_type="full",
            random_state=42,
            n_init=5,
        )
        model.fit(scaled)
        bic = model.bic(scaled)
        if bic < best_bic:
            best_bic = float(bic)
            best_model = model

    if best_model is None:
        raise RuntimeError("Failed to fit Gaussian mixture model")

    labels = best_model.predict(scaled).astype(int)
    result = dataframe.copy()
    result["Subpopulation_ID"] = labels
    return result


def compute_3d_ripleys_k(points: np.ndarray, volume: float, radii: np.ndarray) -> pd.DataFrame:
    """Compute 3D Ripley's K-function for a point pattern.

    Parameters
    ----------
    points : np.ndarray
        Point coordinates shaped ``(N, 3)``.
    volume : float
        Total analyzed 3D volume in physical units.
    radii : np.ndarray
        Radii at which to estimate K(r).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``radius``, ``K``, and ``L`` where
        $L(r) = \\left(\\frac{3K(r)}{4\\pi}\\right)^{1/3} - r$.
    """
    xyz = _validate_points_3d(points)
    test_radii = np.asarray(radii, dtype=float)
    if test_radii.ndim != 1 or np.any(test_radii <= 0):
        raise ValueError("radii must be a 1D array with strictly positive values")
    if volume <= 0:
        raise ValueError("volume must be > 0")

    n_points = xyz.shape[0]
    if n_points < 2:
        return pd.DataFrame({"radius": test_radii, "K": np.zeros_like(test_radii), "L": np.zeros_like(test_radii)})

    density = n_points / float(volume)
    tree = cKDTree(xyz)

    k_values: list[float] = []
    for radius in test_radii:
        neighbors = tree.query_ball_tree(tree, r=float(radius))
        pair_count = sum(len(indices) - 1 for indices in neighbors)
        k_r = pair_count / (density * n_points)
        k_values.append(float(k_r))

    k_array = np.asarray(k_values, dtype=float)
    l_array = np.cbrt((3.0 * k_array) / (4.0 * np.pi)) - test_radii
    return pd.DataFrame({"radius": test_radii, "K": k_array, "L": l_array})


def compute_clark_evans_3d(points: np.ndarray, volume_bbox: tuple[float, float, float]) -> float:
    """Compute the 3D Clark-Evans aggregation index.

    Parameters
    ----------
    points : np.ndarray
        Point coordinates shaped ``(N, 3)``.
    volume_bbox : tuple[float, float, float]
        Bounding box extents ``(dz, dy, dx)`` in physical units.

    Returns
    -------
    float
        Clark-Evans index $R = \bar{r}_{obs} / \bar{r}_{exp}$. Values below 1
        indicate clustering and values above 1 indicate dispersion.
    """
    xyz = _validate_points_3d(points)
    if xyz.shape[0] < 2:
        return float("nan")

    extents = np.asarray(volume_bbox, dtype=float)
    if extents.shape != (3,) or np.any(extents <= 0):
        raise ValueError("volume_bbox must be a tuple of three positive extents")

    volume = float(np.prod(extents))
    density = xyz.shape[0] / volume
    expected_nn = 0.55396 * (density ** (-1.0 / 3.0))

    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=2)
    observed_nn = float(np.mean(distances[:, 1]))
    return observed_nn / expected_nn


def compute_persistent_homology(point_cloud: np.ndarray, max_dimension: int = 2) -> dict[str, Any]:
    """Compute persistent homology summaries from a 3D point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray
        Input points shaped ``(N, 3)``.
    max_dimension : int, optional
        Maximum homology dimension for ripser.

    Returns
    -------
    dict[str, Any]
        Dictionary containing persistence diagrams and maximum persistence
        lifespan per Betti dimension.

    Raises
    ------
    ImportError
        If optional dependency ``ripser`` is unavailable.
    """
    if not HAS_RIPSER:
        raise ImportError(
            "Persistent homology requires 'ripser'. "
            "Install extras with: pip install pymaris[advanced]"
        )
    ripser = importlib.import_module("ripser").ripser

    xyz = _validate_points_3d(point_cloud)
    if max_dimension < 0:
        raise ValueError("max_dimension must be >= 0")

    result = ripser(xyz, maxdim=max_dimension)
    diagrams = result.get("dgms", [])
    max_persistence: dict[str, float] = {}

    for dim_idx, diagram in enumerate(diagrams):
        if diagram.size == 0:
            max_persistence[f"betti_{dim_idx}"] = 0.0
            continue
        finite = diagram[np.isfinite(diagram[:, 1])]
        if finite.size == 0:
            max_persistence[f"betti_{dim_idx}"] = float("inf")
            continue
        lifespans = finite[:, 1] - finite[:, 0]
        max_persistence[f"betti_{dim_idx}"] = float(np.max(lifespans)) if lifespans.size else 0.0

    return {
        "diagrams": diagrams,
        "max_persistence": max_persistence,
    }


def _validate_feature_columns(frame: pd.DataFrame, feature_cols: list[str]) -> None:
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")
    missing = [column for column in feature_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def _validate_points_3d(points: np.ndarray) -> np.ndarray:
    xyz = np.asarray(points, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("points must be a 2D array shaped (N, 3)")
    return xyz
