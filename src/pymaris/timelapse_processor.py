"""Dynamic morpho-kinetic analytics for tracked 3D objects."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_UMAP = importlib.util.find_spec("umap") is not None
HAS_HDBSCAN = importlib.util.find_spec("hdbscan") is not None


def cluster_morphokinetic_trajectories(
    tracking_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Cluster track-level morpho-kinetic trajectories with UMAP + HDBSCAN.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Per-timepoint tracking table containing at least ``track_id``.
    feature_cols : list[str]
        Feature columns used to build track summary vectors.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with added ``Behavioral_State_ID`` column.

    Raises
    ------
    ImportError
        If advanced optional dependencies are unavailable.
    ValueError
        If required columns are missing.
    """
    if not (HAS_SKLEARN and HAS_UMAP and HAS_HDBSCAN):
        raise ImportError(
            "Trajectory clustering requires 'scikit-learn', 'umap-learn', and 'hdbscan'. "
            "Install extras with: pip install pymaris[advanced]"
        )
    hdbscan = importlib.import_module("hdbscan")
    umap = importlib.import_module("umap")
    from sklearn.preprocessing import StandardScaler

    if "track_id" not in tracking_df.columns:
        raise ValueError("tracking_df must include a 'track_id' column")
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")

    missing = [column for column in feature_cols if column not in tracking_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    time_col = _resolve_time_column(tracking_df)
    sorted_df = tracking_df.sort_values(["track_id", time_col]).copy()

    summary_rows: list[dict[str, Any]] = []
    for track_id, group in sorted_df.groupby("track_id", sort=False):
        row: dict[str, Any] = {"track_id": track_id}

        for feature in feature_cols:
            values = group[feature].to_numpy(dtype=float)
            row[f"{feature}_mean"] = float(np.mean(values))
            row[f"{feature}_var"] = float(np.var(values))

        row["max_velocity"] = float(_compute_max_velocity(group, time_col))
        row["volume_change_rate"] = float(_compute_volume_change_rate(group, time_col))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.shape[0] == 0:
        result = tracking_df.copy()
        result["Behavioral_State_ID"] = np.array([], dtype=int)
        return result

    feature_matrix = summary_df.drop(columns=["track_id"]).to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    n_neighbors = min(15, max(2, summary_df.shape[0] - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(scaled)

    min_cluster_size = max(2, min(5, max(2, summary_df.shape[0] // 2)))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embedding).astype(int)

    summary_df["Behavioral_State_ID"] = labels
    merged = tracking_df.copy()
    merged = merged.merge(
        summary_df[["track_id", "Behavioral_State_ID"]],
        on="track_id",
        how="left",
    )
    return merged


def calculate_markov_transitions(tracking_df: pd.DataFrame, state_column: str) -> pd.DataFrame:
    """Estimate empirical first-order transition probabilities between states.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Per-timepoint tracking table containing ``track_id`` and time ordering columns.
    state_column : str
        Column containing state IDs or labels.

    Returns
    -------
    pd.DataFrame
        Transition probability matrix where rows are source states and columns are
        destination states.
    """
    if "track_id" not in tracking_df.columns:
        raise ValueError("tracking_df must include a 'track_id' column")
    if state_column not in tracking_df.columns:
        raise ValueError(f"tracking_df must include state column '{state_column}'")

    time_col = _resolve_time_column(tracking_df)
    sorted_df = tracking_df.sort_values(["track_id", time_col]).copy()

    states = sorted_df[state_column].dropna().unique().tolist()
    transitions = pd.DataFrame(0.0, index=states, columns=states, dtype=float)

    for _, group in sorted_df.groupby("track_id", sort=False):
        sequence = group[state_column].dropna().to_list()
        if len(sequence) < 2:
            continue
        for src, dst in zip(sequence[:-1], sequence[1:]):
            current = np.asarray(transitions.at[src, dst], dtype=float).item()
            transitions.at[src, dst] = current + 1.0

    row_sums = transitions.sum(axis=1)
    nonzero_rows = row_sums > 0
    transitions.loc[nonzero_rows, :] = transitions.loc[nonzero_rows, :].div(row_sums[nonzero_rows], axis=0)
    return transitions


def _resolve_time_column(frame: pd.DataFrame) -> str:
    for candidate in ("time", "frame", "t", "timepoint"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("tracking_df must contain one of: 'time', 'frame', 't', or 'timepoint'")


def _compute_max_velocity(track_group: pd.DataFrame, time_col: str) -> float:
    if "velocity" in track_group.columns:
        return float(np.nanmax(track_group["velocity"].to_numpy(dtype=float)))

    position_options = [
        ("z", "y", "x"),
        ("centroid-0", "centroid-1", "centroid-2"),
        ("pos_z", "pos_y", "pos_x"),
    ]
    selected = None
    for option in position_options:
        if all(name in track_group.columns for name in option):
            selected = option
            break
    if selected is None:
        return 0.0

    positions = track_group.loc[:, list(selected)].to_numpy(dtype=float)
    times = track_group[time_col].to_numpy(dtype=float)
    if positions.shape[0] < 2:
        return 0.0

    dt = np.diff(times)
    dt[dt == 0] = 1.0
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    velocity = displacement / dt
    return float(np.max(velocity)) if velocity.size else 0.0


def _compute_volume_change_rate(track_group: pd.DataFrame, time_col: str) -> float:
    volume_column = "volume" if "volume" in track_group.columns else ("area" if "area" in track_group.columns else None)
    if volume_column is None:
        return 0.0

    volumes = track_group[volume_column].to_numpy(dtype=float)
    times = track_group[time_col].to_numpy(dtype=float)
    if volumes.size < 2:
        return 0.0

    total_dt = float(times[-1] - times[0])
    if total_dt == 0:
        return 0.0
    return float((volumes[-1] - volumes[0]) / total_dt)
