"""Lightweight benchmark harness for core backend throughput and sanity checks."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np

from pymaris.backends import DEFAULT_REGISTRY, BackendRegistry
from pymaris.data_model import ImageVolume
from pymaris.workflow import WorkflowResult, WorkflowStep

CaseFactory = Callable[[], dict[str, Any]]
CaseCheck = Callable[[WorkflowResult], dict[str, Any]]


def run_baseline_benchmark(
    *,
    repeats: int = 1,
    size_2d: int = 128,
    size_3d: tuple[int, int, int] = (16, 64, 64),
    seed: int = 0,
    registry: BackendRegistry | None = None,
) -> dict[str, Any]:
    """Run deterministic baseline benchmarks and return JSON-serializable report."""
    if repeats <= 0:
        raise ValueError("repeats must be >= 1")
    if size_2d <= 8:
        raise ValueError("size_2d must be > 8")
    if any(dim <= 4 for dim in size_3d):
        raise ValueError("size_3d dimensions must be > 4")

    target_registry = registry or DEFAULT_REGISTRY
    cases = _build_cases(size_2d=size_2d, size_3d=size_3d, seed=seed)

    started = perf_counter()
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_started = perf_counter()
        durations: list[float] = []
        last_result: WorkflowResult | None = None
        failure: str | None = None
        for _ in range(repeats):
            context = case["context_factory"]()
            step = case["step"]
            run_started = perf_counter()
            try:
                last_result = step.run(context=context, registry=target_registry)
            except Exception as exc:
                failure = str(exc)
                break
            durations.append(perf_counter() - run_started)

        elapsed_total = perf_counter() - case_started
        status = "ok" if failure is None else "error"
        metrics: dict[str, Any] = {}
        if last_result is not None and failure is None:
            metrics = case["check"](last_result)
        rows.append(
            {
                "id": case["id"],
                "name": case["step"].name,
                "backend_type": case["step"].backend_type,
                "backend_name": case["step"].backend_name,
                "status": status,
                "error": failure,
                "repeats_completed": len(durations),
                "timing": _timing_summary(durations, elapsed_total=elapsed_total),
                "metrics": metrics,
                "output_keys": sorted(last_result.outputs.keys()) if last_result is not None else [],
            }
        )

    elapsed = perf_counter() - started
    successful = sum(1 for row in rows if row["status"] == "ok")
    return {
        "suite": "baseline",
        "seed": seed,
        "repeats": repeats,
        "sizes": {"size_2d": size_2d, "size_3d": list(size_3d)},
        "cases": rows,
        "summary": {
            "case_count": len(rows),
            "successful_cases": successful,
            "failed_cases": len(rows) - successful,
            "elapsed_seconds": float(elapsed),
        },
    }


def _build_cases(*, size_2d: int, size_3d: tuple[int, int, int], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)

    segmentation_image = _synthetic_segmentation_image(size=size_2d, rng=rng)
    restoration_image = _synthetic_restoration_image(size=size_3d, rng=rng)
    distance_image = _synthetic_distance_image(size=size_2d)
    tracking_sequence = _synthetic_tracking_labels(size=size_2d)

    return [
        {
            "id": "segmentation_watershed",
            "step": WorkflowStep(
                id="bench-seg-001",
                name="benchmark-segmentation-watershed",
                backend_type="segmentation",
                backend_name="watershed",
                params={"threshold": float(np.percentile(segmentation_image.array, 70))},
                inputs=["image"],
                outputs=["labels"],
            ),
            "context_factory": lambda: {"image": segmentation_image},
            "check": _check_segmentation,
        },
        {
            "id": "restoration_denoise",
            "step": WorkflowStep(
                id="bench-rest-001",
                name="benchmark-restoration-denoise",
                backend_type="restoration",
                backend_name="classic",
                params={"operation": "denoise", "sigma": 1.0},
                inputs=["image"],
                outputs=["denoised"],
            ),
            "context_factory": lambda: {"image": restoration_image},
            "check": _check_image_output,
        },
        {
            "id": "restoration_distance_map",
            "step": WorkflowStep(
                id="bench-rest-002",
                name="benchmark-restoration-distance-map",
                backend_type="restoration",
                backend_name="classic",
                params={"operation": "distance_map", "threshold": 0.5},
                inputs=["image"],
                outputs=["distance_map"],
            ),
            "context_factory": lambda: {"image": distance_image},
            "check": _check_distance_map,
        },
        {
            "id": "tracking_hungarian",
            "step": WorkflowStep(
                id="bench-track-001",
                name="benchmark-tracking-hungarian",
                backend_type="tracking",
                backend_name="hungarian",
                params={"max_distance": 8.0},
                inputs=["labels_sequence"],
                outputs=["tracks"],
            ),
            "context_factory": lambda: {"labels_sequence": tracking_sequence},
            "check": _check_tracking,
        },
    ]


def _timing_summary(durations: list[float], *, elapsed_total: float) -> dict[str, Any]:
    if not durations:
        return {"mean_seconds": None, "min_seconds": None, "max_seconds": None, "elapsed_seconds": elapsed_total}
    arr = np.asarray(durations, dtype=float)
    return {
        "mean_seconds": float(np.mean(arr)),
        "min_seconds": float(np.min(arr)),
        "max_seconds": float(np.max(arr)),
        "elapsed_seconds": float(elapsed_total),
    }


def _synthetic_segmentation_image(*, size: int, rng: np.random.Generator) -> ImageVolume:
    y, x = np.ogrid[:size, :size]
    center_y = size // 2
    center_x = size // 2
    radius = max(4, size // 6)
    disk = ((y - center_y) ** 2 + (x - center_x) ** 2) <= radius**2
    image = rng.normal(loc=10.0, scale=2.0, size=(size, size)).astype(np.float32)
    image[disk] += 35.0
    image = np.clip(image, 0.0, None)
    return ImageVolume(array=image, axes=("Y", "X"), modality="fluorescence")


def _synthetic_restoration_image(*, size: tuple[int, int, int], rng: np.random.Generator) -> ImageVolume:
    z, y, x = size
    image = rng.normal(loc=20.0, scale=4.0, size=(z, y, x)).astype(np.float32)
    zz, yy, xx = np.ogrid[:z, :y, :x]
    center = (z // 2, y // 2, x // 2)
    radius = max(3, min(size) // 6)
    sphere = ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) <= radius**2
    image[sphere] += 20.0
    return ImageVolume(array=np.clip(image, 0.0, None), axes=("Z", "Y", "X"), modality="fluorescence")


def _synthetic_distance_image(*, size: int) -> ImageVolume:
    image = np.zeros((size, size), dtype=np.float32)
    margin = max(2, size // 6)
    image[margin:-margin, margin:-margin] = 1.0
    return ImageVolume(array=image, axes=("Y", "X"), modality="fluorescence")


def _synthetic_tracking_labels(*, size: int) -> list[np.ndarray]:
    labels: list[np.ndarray] = []
    for timepoint in range(4):
        frame = np.zeros((size, size), dtype=np.int32)
        start_y = max(1, size // 3 + timepoint)
        start_x = max(1, size // 3 + timepoint)
        frame[start_y : start_y + 4, start_x : start_x + 4] = 1
        labels.append(frame)
    return labels


def _check_segmentation(result: WorkflowResult) -> dict[str, Any]:
    labels = np.asarray(result.outputs.get("labels"))
    return {"max_label": int(np.max(labels)) if labels.size else 0, "nonzero_labels": int(np.count_nonzero(labels))}


def _check_image_output(result: WorkflowResult) -> dict[str, Any]:
    image = result.outputs.get("denoised")
    if isinstance(image, ImageVolume):
        return {"shape": list(image.shape), "dtype": str(image.dtype)}
    return {"shape": None, "dtype": None}


def _check_distance_map(result: WorkflowResult) -> dict[str, Any]:
    image = result.outputs.get("distance_map")
    if isinstance(image, ImageVolume):
        array = image.as_numpy()
        return {"max_distance": float(np.max(array)), "mean_distance": float(np.mean(array))}
    return {"max_distance": 0.0, "mean_distance": 0.0}


def _check_tracking(result: WorkflowResult) -> dict[str, Any]:
    tracks_payload = result.outputs.get("tracks", {})
    if not isinstance(tracks_payload, dict):
        return {"total_tracks": 0, "rows": 0}
    tracks = tracks_payload.get("tracks", [])
    napari_tracks = np.asarray(tracks_payload.get("napari_tracks", []))
    return {"total_tracks": int(len(tracks)), "rows": int(napari_tracks.shape[0]) if napari_tracks.ndim >= 1 else 0}
