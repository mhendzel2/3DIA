"""Baseline backend adapters wrapping existing repository functionality."""

from __future__ import annotations

import inspect
from importlib import metadata as importlib_metadata
from typing import Any, Sequence

import numpy as np
from scipy import ndimage
from scipy.signal import wiener as scipy_wiener

from pymaris import analysis
from pymaris.backends.registry import DEFAULT_REGISTRY, BackendRegistry
from pymaris.backends.types import (
    BackendCapability,
    BackendInfo,
    ImageResult,
    LabelsResult,
    RestorationBackend,
    SegmentationBackend,
    TraceResult,
    TracingBackend,
    TrackingBackend,
    TracksResult,
)
from pymaris.data_model import ImageVolume

try:  # pragma: no cover - optional dependency
    import dask.array as da

    HAS_DASK_ARRAY = True
except Exception:  # pragma: no cover - optional dependency
    da = None  # type: ignore[assignment]
    HAS_DASK_ARRAY = False

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    HAS_SCIPY_TRACKING = True
except Exception:  # pragma: no cover - scipy is a required dependency
    HAS_SCIPY_TRACKING = False

try:
    from skimage import feature, filters, measure, morphology, segmentation
    from skimage.restoration import (
        denoise_bilateral,
        denoise_nl_means,
        estimate_sigma,
        richardson_lucy,
        wiener,
    )

    HAS_SKIMAGE = True
except Exception:  # pragma: no cover - skimage is a required dependency
    HAS_SKIMAGE = False

try:  # pragma: no cover - optional helper module
    from advanced_analysis import AIDenoising as LegacyAIDenoising

    HAS_LEGACY_AI_DENOISING = True
except Exception:  # pragma: no cover - optional helper module
    LegacyAIDenoising = None  # type: ignore[assignment]
    HAS_LEGACY_AI_DENOISING = False


def _version_for(package_name: str) -> str:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


class WatershedSegmentationBackend(SegmentationBackend):
    info = BackendInfo(
        name="watershed",
        version=_version_for("scikit-image"),
        capability=BackendCapability(
            task="segmentation",
            dimensions=(2, 3),
            modalities=("fluorescence", "brightfield", "electron_microscopy"),
            supports_multichannel=False,
            notes="Classical threshold + watershed baseline",
        ),
    )

    def segment_instances(self, image: ImageVolume, **params: Any) -> LabelsResult:
        data = image.as_numpy()
        mask = data > float(params.get("threshold", 0))
        labels = _watershed_nd(mask)
        if labels is None:
            raise RuntimeError("watershed segmentation backend returned no labels")
        table = _safe_object_statistics(labels, image=image)
        return LabelsResult(
            labels=labels,
            table=table,
            metadata={
                "backend": self.info.name,
                "backend_version": self.info.version,
                "params": dict(params),
            },
        )


class CellposeSegmentationBackend(SegmentationBackend):
    info = BackendInfo(
        name="cellpose",
        version=_version_for("cellpose"),
        capability=BackendCapability(
            task="segmentation",
            dimensions=(2, 3),
            modalities=("fluorescence", "brightfield"),
            supports_multichannel=True,
            notes="Generalist deep-learning instance segmentation",
        ),
    )

    def segment_instances(self, image: ImageVolume, **params: Any) -> LabelsResult:
        diameter = int(params.get("diameter", 30))
        labels = analysis.segment_cellpose(image=image.as_numpy(), diameter=diameter)
        if labels is None:
            raise RuntimeError("Cellpose backend not available. Install optional dependency set `.[ai]`.")
        labels_array = np.asarray(labels)
        table = _safe_object_statistics(labels_array, image=image)
        return LabelsResult(
            labels=labels,
            table=table,
            metadata={
                "backend": self.info.name,
                "backend_version": self.info.version,
                "params": {"diameter": diameter, **params},
                "model_provenance": {
                    "family": "cellpose",
                    "mode": "ml",
                    "diameter": diameter,
                },
                "qa_report": _label_output_qa(labels_array),
            },
        )


class StarDistSegmentationBackend(SegmentationBackend):
    info = BackendInfo(
        name="stardist",
        version=_version_for("stardist"),
        capability=BackendCapability(
            task="segmentation",
            dimensions=(2, 3),
            modalities=("fluorescence",),
            supports_multichannel=False,
            notes="Star-convex nuclei/cell instance segmentation",
        ),
    )

    def segment_instances(self, image: ImageVolume, **params: Any) -> LabelsResult:
        labels = analysis.segment_stardist(image=image.as_numpy())
        if labels is None:
            raise RuntimeError("StarDist backend not available. Install optional dependency set `.[ai]`.")
        labels_array = np.asarray(labels)
        table = _safe_object_statistics(labels_array, image=image)
        return LabelsResult(
            labels=labels,
            table=table,
            metadata={
                "backend": self.info.name,
                "backend_version": self.info.version,
                "params": dict(params),
                "model_provenance": {
                    "family": "stardist",
                    "mode": "ml",
                },
                "qa_report": _label_output_qa(labels_array),
            },
        )


class HungarianTrackingBackend(TrackingBackend):
    info = BackendInfo(
        name="hungarian_linking",
        version=_version_for("scipy"),
        capability=BackendCapability(
            task="tracking",
            dimensions=(2, 3),
            supports_time=True,
            notes="Centroid-based linear assignment tracking",
        ),
    )

    def track(self, labels_over_time: Sequence[Any], **params: Any) -> TracksResult:
        if not HAS_SCIPY_TRACKING:
            raise RuntimeError("scipy is required for hungarian tracking backend")

        max_distance = float(params.get("max_distance", 50.0))
        detections_by_time = [
            self._extract_centroids(np.asarray(labels), timepoint=index)
            for index, labels in enumerate(labels_over_time)
        ]
        tracks = self._link_detections(detections_by_time=detections_by_time, max_distance=max_distance)
        track_table = self._summarize_tracks(tracks)
        napari_tracks = self._to_napari_tracks_array(tracks)
        return TracksResult(
            tracks={
                "tracks": tracks,
                "napari_tracks": napari_tracks,
                "detections": detections_by_time,
            },
            table=track_table,
            metadata={
                "backend": self.info.name,
                "backend_version": self.info.version,
                "params": {"max_distance": max_distance, **params},
            },
        )

    def _extract_centroids(self, labels: np.ndarray, timepoint: int) -> list[dict[str, Any]]:
        if labels.ndim < 2:
            raise ValueError("tracking labels must be 2D or 3D arrays")
        detections: list[dict[str, Any]] = []

        if HAS_SKIMAGE:
            for region in measure.regionprops(labels):
                detections.append(
                    {
                        "id": int(region.label),
                        "timepoint": timepoint,
                        "position": tuple(float(v) for v in region.centroid),
                        "area": int(region.area),
                    }
                )
            return detections

        for label_id in np.unique(labels):
            if label_id <= 0:
                continue
            indices = np.argwhere(labels == label_id)
            detections.append(
                {
                    "id": int(label_id),
                    "timepoint": timepoint,
                    "position": tuple(float(v) for v in indices.mean(axis=0)),
                    "area": int(indices.shape[0]),
                }
            )
        return detections

    def _link_detections(
        self,
        detections_by_time: Sequence[Sequence[dict[str, Any]]],
        max_distance: float,
    ) -> list[dict[str, Any]]:
        tracks: list[dict[str, Any]] = []
        active_tracks: list[dict[str, Any]] = []
        next_id = 0

        for time_index, detections in enumerate(detections_by_time):
            if time_index == 0:
                for detection in detections:
                    track = {"track_id": next_id, "detections": [detection]}
                    tracks.append(track)
                    active_tracks.append(track)
                    next_id += 1
                continue

            if not active_tracks:
                for detection in detections:
                    track = {"track_id": next_id, "detections": [detection]}
                    tracks.append(track)
                    active_tracks.append(track)
                    next_id += 1
                continue

            if not detections:
                active_tracks = []
                continue

            track_positions = np.asarray([track["detections"][-1]["position"] for track in active_tracks])
            detection_positions = np.asarray([item["position"] for item in detections])
            costs = cdist(track_positions, detection_positions)
            costs[costs > max_distance] = 1e9
            track_indices, detection_indices = linear_sum_assignment(costs)

            matched_detections: set[int] = set()
            next_active: list[dict[str, Any]] = []
            for track_index, detection_index in zip(track_indices, detection_indices):
                if costs[track_index, detection_index] >= 1e9:
                    continue
                active_tracks[track_index]["detections"].append(detections[detection_index])
                next_active.append(active_tracks[track_index])
                matched_detections.add(int(detection_index))

            for detection_index, detection in enumerate(detections):
                if detection_index in matched_detections:
                    continue
                track = {"track_id": next_id, "detections": [detection]}
                tracks.append(track)
                next_active.append(track)
                next_id += 1

            active_tracks = next_active

        return tracks

    def _summarize_tracks(self, tracks: Sequence[dict[str, Any]]) -> dict[str, Any]:
        lengths = [len(track["detections"]) for track in tracks]
        return {
            "total_tracks": len(tracks),
            "track_lengths": lengths,
            "mean_track_length": float(np.mean(lengths)) if lengths else 0.0,
        }

    def _to_napari_tracks_array(self, tracks: Sequence[dict[str, Any]]) -> np.ndarray:
        rows: list[list[float]] = []
        spatial_dims = 0
        for track in tracks:
            track_id = int(track["track_id"])
            for detection in track["detections"]:
                position = tuple(float(v) for v in detection["position"])
                spatial_dims = max(spatial_dims, len(position))
                row = [float(track_id), float(detection["timepoint"]), *position]
                rows.append(row)
        if not rows:
            return np.empty((0, 2 + spatial_dims), dtype=float)
        return np.asarray(rows, dtype=float)


class SkeletonTracingBackend(TracingBackend):
    info = BackendInfo(
        name="skeleton_trace",
        version=_version_for("scikit-image"),
        capability=BackendCapability(
            task="tracing",
            dimensions=(2, 3),
            modalities=("fluorescence", "electron_microscopy"),
            notes="Classical skeletonization-based filament tracing",
        ),
    )

    def trace(self, image: ImageVolume, **params: Any) -> TraceResult:
        if not HAS_SKIMAGE:
            raise RuntimeError("scikit-image is required for skeleton tracing backend")
        data = image.as_numpy()
        gaussian_sigma = float(params.get("gaussian_sigma", 1.0))
        if gaussian_sigma > 0:
            smoothed = np.asarray(filters.gaussian(data, sigma=gaussian_sigma, preserve_range=True))
        else:
            smoothed = np.asarray(data)

        threshold = self._resolve_threshold(smoothed, params)
        binary = smoothed > threshold
        min_size = int(params.get("min_object_size", 50))
        if min_size > 0:
            binary = _remove_small_objects(binary, min_size=min_size)

        skeleton = self._skeletonize(binary)
        graph = self._graph_summary(skeleton)
        branch_points = (
            self._detect_branch_points(skeleton)
            if bool(params.get("detect_branches", True))
            else np.empty((0, skeleton.ndim), dtype=int)
        )
        filaments = self._extract_filaments(skeleton) if bool(params.get("extract_paths", True)) else []
        avg_thickness = (
            self._measure_average_thickness(binary, skeleton)
            if bool(params.get("measure_thickness", True))
            else 0.0
        )

        graph.update(
            {
                "threshold": float(threshold),
                "binary": binary.astype(np.uint8),
                "branch_points": branch_points,
                "filaments": filaments,
                "total_length": float(np.sum(skeleton)),
                "num_filaments": int(len(filaments)),
                "num_branches": int(len(branch_points)),
                "avg_thickness": float(avg_thickness),
            }
        )
        return TraceResult(
            graph=graph,
            table={
                "node_count": graph["node_count"],
                "edge_count": graph["edge_count"],
                "total_length": graph["total_length"],
                "num_filaments": graph["num_filaments"],
                "num_branches": graph["num_branches"],
                "avg_thickness": graph["avg_thickness"],
            },
            metadata={
                "backend": self.info.name,
                "backend_version": self.info.version,
                "params": dict(params),
            },
        )

    def _resolve_threshold(self, smoothed: np.ndarray, params: dict[str, Any]) -> float:
        method = str(params.get("threshold_method", "otsu")).lower()
        if method == "manual":
            return float(params.get("manual_threshold", 0))
        if method == "li":
            return float(filters.threshold_li(smoothed))
        if method == "otsu":
            return float(filters.threshold_otsu(smoothed))
        explicit = params.get("threshold")
        if explicit is not None:
            return float(explicit)
        return float(np.mean(smoothed))

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 2:
            return morphology.skeletonize(mask).astype(np.uint8)
        if mask.ndim == 3:
            if hasattr(morphology, "skeletonize_3d"):
                return np.asarray(morphology.skeletonize_3d(mask), dtype=np.uint8)
            return np.asarray([morphology.skeletonize(frame) for frame in mask], dtype=np.uint8)
        raise ValueError("skeleton tracing supports only 2D/3D images")

    def _detect_branch_points(self, skeleton: np.ndarray) -> np.ndarray:
        kernel = np.ones((3,) * skeleton.ndim, dtype=int)
        kernel[(1,) * skeleton.ndim] = 0
        neighbors = ndimage.convolve(skeleton.astype(int), kernel, mode="constant", cval=0)
        branch_mask = (skeleton > 0) & (neighbors > 2)
        return np.argwhere(branch_mask)

    def _extract_filaments(self, skeleton: np.ndarray) -> list[dict[str, Any]]:
        labeled = measure.label(skeleton)
        filaments: list[dict[str, Any]] = []
        for region in measure.regionprops(labeled):
            coords = np.asarray(region.coords)
            if coords.shape[0] <= 1:
                continue
            ordered = self._order_coordinates(coords)
            filaments.append(
                {
                    "coords": ordered,
                    "length": float(ordered.shape[0]),
                    "label": int(region.label),
                }
            )
        return filaments

    def _order_coordinates(self, coords: np.ndarray) -> np.ndarray:
        if coords.shape[0] <= 2:
            return coords
        ordered = [coords[0]]
        remaining = [point for point in coords[1:]]
        while remaining:
            current = np.asarray(ordered[-1], dtype=float)
            if HAS_SCIPY_TRACKING:
                distances = cdist([current], np.asarray(remaining, dtype=float))
                nearest_idx = int(np.argmin(distances))
            else:
                deltas = np.asarray(remaining, dtype=float) - current
                nearest_idx = int(np.argmin(np.linalg.norm(deltas, axis=1)))
            ordered.append(remaining.pop(nearest_idx))
        return np.asarray(ordered)

    def _measure_average_thickness(self, binary: np.ndarray, skeleton: np.ndarray) -> float:
        if not np.any(skeleton):
            return 0.0
        distance_map = ndimage.distance_transform_edt(binary)
        return float(np.mean(distance_map[skeleton > 0]) * 2.0)

    def _graph_summary(self, skeleton: np.ndarray) -> dict[str, Any]:
        node_count = int(np.count_nonzero(skeleton))
        kernel = np.ones((3,) * skeleton.ndim, dtype=int)
        kernel[(1,) * skeleton.ndim] = 0
        neighbors = ndimage.convolve(skeleton.astype(int), kernel, mode="constant", cval=0)
        edge_count = int(np.sum(neighbors[skeleton > 0]) // 2)
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "skeleton": skeleton.astype(np.uint8),
        }


class ClassicRestorationBackend(RestorationBackend):
    info = BackendInfo(
        name="classic_restoration",
        version=_version_for("scikit-image"),
        capability=BackendCapability(
            task="restoration",
            dimensions=(2, 3),
            modalities=("fluorescence", "brightfield", "electron_microscopy"),
            notes="Gaussian denoise + Richardson-Lucy/Wiener deconvolution",
        ),
    )

    def denoise(self, image: ImageVolume, **params: Any) -> ImageResult:
        method = str(params.get("method", "gaussian")).lower()
        if method in {"gaussian", "gauss"}:
            sigma = float(params.get("sigma", 1.0))
            denoised = _gaussian_filter_from_image(image=image, sigma=sigma)
            operation = "denoise"
            metadata_params: dict[str, Any] = {"method": "gaussian", "sigma": sigma, **params}
        elif method in {"distance_map", "euclidean_distance_map", "edt"}:
            denoised = _distance_map_from_image(image=image, **params)
            operation = "distance_map"
            metadata_params = dict(params)
            metadata_params["method"] = "distance_map"
        else:
            raise ValueError(
                f"unsupported classic restoration method {method!r}; expected one of "
                "'gaussian', 'distance_map'"
            )
        return ImageResult(
            image=image.with_array(denoised),
            metadata={
                "backend": self.info.name,
                "backend_version": self.info.version,
                "operation": operation,
                "lazy_output": _is_dask_array(denoised),
                "params": metadata_params,
            },
        )

    def deconvolve(self, image: ImageVolume, **params: Any) -> ImageResult:
        if not HAS_SKIMAGE:
            raise RuntimeError("scikit-image is required for deconvolution backend")
        method = str(params.get("method", "richardson_lucy"))
        iterations = int(params.get("iterations", 10))
        psf_size = int(params.get("psf_size", 5))
        psf_sigma = float(params.get("psf_sigma", 1.0))
        psf = _gaussian_psf(ndim=image.ndim, size=psf_size, sigma=psf_sigma)
        data = image.as_numpy()

        if method == "wiener":
            result = wiener(data, psf=psf, balance=float(params.get("balance", 0.1)))
        else:
            result = richardson_lucy(data, psf=psf, num_iter=iterations)

        return ImageResult(
            image=image.with_array(np.asarray(result)),
            metadata={
                "backend": self.info.name,
                "backend_version": self.info.version,
                "operation": "deconvolve",
                "params": {
                    "method": method,
                    "iterations": iterations,
                    "psf_size": psf_size,
                    "psf_sigma": psf_sigma,
                    **params,
                },
            },
        )


class AIDenoisingRestorationBackend(RestorationBackend):
    info = BackendInfo(
        name="ai_denoising",
        version=_version_for("scikit-image"),
        capability=BackendCapability(
            task="restoration",
            dimensions=(2, 3),
            modalities=("fluorescence", "brightfield", "electron_microscopy"),
            supports_multichannel=True,
            notes="AI-inspired denoising (NLM/bilateral/wiener) with legacy advanced_analysis integration",
        ),
    )

    def __init__(self) -> None:
        self._classic = ClassicRestorationBackend()

    def denoise(self, image: ImageVolume, **params: Any) -> ImageResult:
        method = str(params.get("method", "non_local_means")).lower()
        data = np.asarray(image.as_numpy(), dtype=float)
        operation = "denoise"
        if method in {"non_local_means", "nlm"}:
            denoised = self._denoise_nlm(data, **params)
        elif method in {"bilateral", "edge_preserving"}:
            denoised = self._denoise_bilateral(data, **params)
        elif method in {"wiener"}:
            denoised = self._denoise_wiener(data, **params)
        elif method in {"distance_map", "euclidean_distance_map", "edt"}:
            denoised = _distance_map_from_image(image=image, **params)
            operation = "distance_map"
        else:
            raise ValueError(
                f"unsupported AI denoising method {method!r}; expected one of "
                "'non_local_means', 'bilateral', 'wiener', 'distance_map'"
            )

        output_array = denoised if _is_dask_array(denoised) else np.asarray(denoised)
        qa_output = np.asarray(image.with_array(output_array).as_numpy())
        metadata_params = dict(params)
        metadata_params["method"] = "distance_map" if operation == "distance_map" else method
        return ImageResult(
            image=image.with_array(output_array),
            metadata={
                "backend": self.info.name,
                "backend_version": self.info.version,
                "operation": operation,
                "ai_denoising": True,
                "lazy_output": _is_dask_array(denoised),
                "model_provenance": {
                    "family": "ai_denoising",
                    "mode": "ml_assisted",
                    "method": metadata_params["method"],
                },
                "qa_report": _image_output_qa(input_data=data, output_data=qa_output),
                "params": metadata_params,
            },
        )

    def deconvolve(self, image: ImageVolume, **params: Any) -> ImageResult:
        result = self._classic.deconvolve(image, **params)
        merged = dict(result.metadata)
        merged["backend"] = self.info.name
        merged["ai_denoising"] = True
        merged["model_provenance"] = {
            "family": "ai_denoising",
            "mode": "ml_assisted",
            "method": "deconvolve",
        }
        merged["qa_report"] = _image_output_qa(
            input_data=np.asarray(image.as_numpy()),
            output_data=np.asarray(result.image.as_numpy()),
        )
        return ImageResult(image=result.image, table=dict(result.table), metadata=merged)

    def _denoise_nlm(self, data: np.ndarray, **params: Any) -> np.ndarray:
        h_value = float(params.get("h", 10.0))
        patch_size = int(params.get("patch_size", 7))
        search_size = int(params.get("search_size", 21))
        if HAS_LEGACY_AI_DENOISING:
            return self._apply_legacy_plane_wise(
                data,
                lambda plane: np.asarray(
                    LegacyAIDenoising.non_local_means_denoising(
                        plane,
                        h=h_value,
                        search_window=search_size,
                        patch_size=patch_size,
                    )
                ),
            )
        if not HAS_SKIMAGE:
            raise RuntimeError("scikit-image is required for AI denoising backend")
        sigma = float(np.mean(estimate_sigma(data, channel_axis=None)))
        return np.asarray(
            denoise_nl_means(
                data,
                h=h_value * max(sigma, 1e-6),
                patch_size=patch_size,
                patch_distance=search_size,
                fast_mode=True,
                preserve_range=True,
                channel_axis=None,
            )
        )

    def _denoise_bilateral(self, data: np.ndarray, **params: Any) -> np.ndarray:
        sigma_spatial = float(params.get("sigma_spatial", 5.0))
        sigma_color = float(params.get("sigma_color", 0.1))
        if HAS_LEGACY_AI_DENOISING:
            return self._apply_legacy_plane_wise(
                data,
                lambda plane: np.asarray(
                    LegacyAIDenoising.bilateral_filter_denoising(
                        plane,
                        sigma_spatial=sigma_spatial,
                        sigma_intensity=sigma_color * 255.0,
                    )
                ),
            )
        if not HAS_SKIMAGE:
            raise RuntimeError("scikit-image is required for AI denoising backend")
        if data.ndim <= 2:
            return np.asarray(
                denoise_bilateral(
                    data,
                    sigma_color=sigma_color,
                    sigma_spatial=sigma_spatial,
                    channel_axis=None,
                )
            )
        output = np.zeros_like(data, dtype=float)
        for index in range(data.shape[0]):
            output[index] = np.asarray(
                denoise_bilateral(
                    data[index],
                    sigma_color=sigma_color,
                    sigma_spatial=sigma_spatial,
                    channel_axis=None,
                )
            )
        return output

    def _denoise_wiener(self, data: np.ndarray, **params: Any) -> np.ndarray:
        if HAS_LEGACY_AI_DENOISING:
            return self._apply_legacy_plane_wise(
                data,
                lambda plane: np.asarray(LegacyAIDenoising.wiener_filter_denoising(plane)),
            )
        noise = float(params.get("noise", 0.01))
        return np.asarray(scipy_wiener(data, noise=noise))

    def _apply_legacy_plane_wise(self, data: np.ndarray, fn: Any) -> np.ndarray:
        if data.ndim <= 2:
            return np.asarray(fn(data))
        output = np.zeros_like(data, dtype=float)
        for index in np.ndindex(data.shape[:-2]):
            output[index] = np.asarray(fn(np.asarray(data[index])))
        return output


def _gaussian_psf(ndim: int, size: int, sigma: float) -> np.ndarray:
    if size < 3:
        size = 3
    if size % 2 == 0:
        size += 1
    coords = np.arange(size, dtype=float) - (size // 2)
    grids = np.meshgrid(*([coords] * ndim), indexing="ij")
    squared_distance = np.zeros_like(grids[0], dtype=float)
    for grid in grids:
        squared_distance += grid * grid
    psf = np.exp(-squared_distance / (2 * sigma * sigma))
    psf /= np.sum(psf)
    return psf


def _watershed_nd(mask: np.ndarray) -> np.ndarray:
    if not HAS_SKIMAGE:
        labels = analysis.segment_watershed(mask)
        if labels is None:
            raise RuntimeError("watershed segmentation requires scikit-image or analysis fallback")
        return labels
    distance = ndimage.distance_transform_edt(mask)
    footprint = np.ones((3,) * mask.ndim, dtype=bool)
    coords = feature.peak_local_max(distance, footprint=footprint, labels=mask)
    marker_mask = np.zeros(distance.shape, dtype=bool)
    if coords.size:
        marker_mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(marker_mask)
    if np.max(markers) == 0:
        markers = ndimage.label(mask)[0]
    return segmentation.watershed(-distance, markers, mask=mask)


def _safe_object_statistics(labels: np.ndarray, image: ImageVolume | None = None) -> dict[str, Any]:
    if labels.size == 0 or int(np.max(labels)) <= 0:
        return {"object_count": 0, "label": [], "area": []}
    try:
        table = analysis.calculate_object_statistics(labeled_image=labels)
    except Exception:
        table = {"object_count": int(np.max(labels)), "label": [], "area": []}

    spacing = _spacing_for_labels(image=image, labels=labels)
    try:
        distance_stats = analysis.calculate_label_distance_measurements(labels, spacing=spacing)
        for key, value in distance_stats.items():
            if key == "object_count" and "object_count" in table:
                continue
            if key not in table:
                table[key] = value
    except Exception:
        pass
    return table


def _label_output_qa(labels: np.ndarray) -> dict[str, Any]:
    count = int(np.max(labels)) if labels.size else 0
    nonzero = int(np.count_nonzero(labels))
    total = int(labels.size) if labels.size else 1
    return {
        "object_count": count,
        "nonzero_fraction": float(nonzero / total),
        "shape": list(labels.shape),
        "dtype": str(labels.dtype),
    }


def _image_output_qa(*, input_data: np.ndarray, output_data: np.ndarray) -> dict[str, Any]:
    finite_ratio = float(np.mean(np.isfinite(output_data))) if output_data.size else 1.0
    input_mean = float(np.mean(input_data)) if input_data.size else 0.0
    output_mean = float(np.mean(output_data)) if output_data.size else 0.0
    max_ratio = 0.0
    if input_data.size and output_data.size:
        input_max = max(float(np.max(np.abs(input_data))), 1e-9)
        output_max = float(np.max(np.abs(output_data)))
        max_ratio = float(output_max / input_max)
    warnings: list[str] = []
    if finite_ratio < 1.0:
        warnings.append("output contains non-finite values")
    if max_ratio > 20.0:
        warnings.append("output dynamic range increased >20x")
    return {
        "status": "ok" if not warnings else "warning",
        "warnings": warnings,
        "finite_ratio": finite_ratio,
        "input_mean": input_mean,
        "output_mean": output_mean,
        "max_abs_ratio": max_ratio,
    }


def _spacing_for_labels(image: ImageVolume | None, labels: np.ndarray) -> tuple[float, ...] | None:
    if image is None:
        return None
    label_ndim = int(labels.ndim)
    if label_ndim <= 0:
        return None
    axes = list(image.axes)
    if len(axes) >= label_ndim:
        relevant_axes = axes[-label_ndim:]
    else:
        relevant_axes = list(axes)
    spacing: list[float] = []
    for axis in relevant_axes:
        axis_label = str(axis).upper()
        if axis_label in {"X", "Y", "Z"}:
            spacing.append(float(image.pixel_size.get(axis_label, 1.0)))
        else:
            spacing.append(1.0)
    if not spacing:
        return None
    return tuple(spacing)


def _spacing_for_distance_map(image: ImageVolume) -> tuple[float, ...] | None:
    spatial_axes = [axis for axis in image.axes if axis in {"Z", "Y", "X"}]
    if not spatial_axes:
        return None
    return tuple(float(image.pixel_size.get(axis, 1.0)) for axis in spatial_axes)


def _distance_map_from_image(image: ImageVolume, **params: Any) -> Any:
    threshold = float(params.get("threshold", 0.0))
    distance_to = str(params.get("distance_to", "background"))
    absolute = bool(params.get("absolute", False))
    spacing = _spacing_for_distance_map(image)
    raw_data = image.array
    if HAS_DASK_ARRAY and isinstance(raw_data, da.Array):
        depth = _distance_map_overlap_depth(image=image, params=params)
        return raw_data.map_overlap(
            _distance_map_block,
            depth=depth,
            boundary=0,
            dtype=float,
            threshold=threshold,
            axes=image.axes,
            spacing=spacing,
            distance_to=distance_to,
            absolute=absolute,
        )
    return np.asarray(
        analysis.generate_euclidean_distance_map(
            image=image.as_numpy(),
            axes=image.axes,
            threshold=threshold,
            spacing=spacing,
            distance_to=distance_to,
            absolute=absolute,
        ),
        dtype=float,
    )


def _distance_map_block(
    block: np.ndarray,
    *,
    threshold: float,
    axes: Sequence[str],
    spacing: Sequence[float] | None,
    distance_to: str,
    absolute: bool,
) -> np.ndarray:
    return np.asarray(
        analysis.generate_euclidean_distance_map(
            image=block,
            axes=axes,
            threshold=threshold,
            spacing=spacing,
            distance_to=distance_to,
            absolute=absolute,
        ),
        dtype=float,
    )


def _distance_map_overlap_depth(image: ImageVolume, params: dict[str, Any]) -> dict[int, int]:
    default_depth = int(params.get("distance_overlap", params.get("tile_overlap", 16)))
    if default_depth < 0:
        default_depth = 0
    return {index: (default_depth if axis in {"Z", "Y", "X"} else 0) for index, axis in enumerate(image.axes)}


def _gaussian_filter_from_image(*, image: ImageVolume, sigma: float) -> Any:
    raw_data = image.array
    if HAS_DASK_ARRAY and isinstance(raw_data, da.Array):
        depth = _gaussian_overlap_depth(image=image, sigma=sigma)
        return raw_data.map_overlap(
            _gaussian_filter_block,
            depth=depth,
            boundary="reflect",
            dtype=float,
            sigma=sigma,
        )
    return ndimage.gaussian_filter(image.as_numpy(), sigma=sigma)


def _gaussian_filter_block(block: np.ndarray, *, sigma: float) -> np.ndarray:
    return np.asarray(ndimage.gaussian_filter(block, sigma=sigma), dtype=float)


def _gaussian_overlap_depth(*, image: ImageVolume, sigma: float) -> dict[int, int]:
    depth = max(1, int(np.ceil(max(0.0, float(sigma)) * 3.0)))
    return {index: (depth if axis in {"Z", "Y", "X"} else 0) for index, axis in enumerate(image.axes)}


def _is_dask_array(value: Any) -> bool:
    return bool(HAS_DASK_ARRAY and isinstance(value, da.Array))


def _remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    remove_small = morphology.remove_small_objects
    params = inspect.signature(remove_small).parameters
    if "max_size" in params:
        # Newer skimage uses `max_size` and removes objects <= threshold.
        return remove_small(mask, max_size=max(0, int(min_size) - 1))
    return remove_small(mask, min_size=min_size)


def register_default_backends(registry: BackendRegistry | None = None) -> BackendRegistry:
    """Register baseline adapters in the provided (or default) registry."""
    target = registry or DEFAULT_REGISTRY
    target.register_segmentation("watershed", WatershedSegmentationBackend())
    target.register_segmentation("cellpose", CellposeSegmentationBackend())
    target.register_segmentation("stardist", StarDistSegmentationBackend())
    target.register_tracking("hungarian", HungarianTrackingBackend())
    target.register_tracing("skeleton", SkeletonTracingBackend())
    target.register_restoration("classic", ClassicRestorationBackend())
    target.register_restoration("ai_denoise", AIDenoisingRestorationBackend())
    return target
