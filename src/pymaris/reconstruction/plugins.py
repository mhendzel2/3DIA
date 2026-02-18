"""Built-in reconstruction plugin implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np
from scipy import ndimage, signal

try:  # pragma: no cover - optional acceleration
    from skimage.feature import peak_local_max

    HAS_SKIMAGE_FEATURE = True
except Exception:  # pragma: no cover
    peak_local_max = None  # type: ignore[assignment]
    HAS_SKIMAGE_FEATURE = False

from pymaris.backends import DEFAULT_REGISTRY
from pymaris.data_model import ImageVolume
from pymaris.reconstruction.calibration import CalibrationArtifact, require_calibrations
from pymaris.reconstruction.plugin import ReconstructionPlugin
from pymaris.reconstruction.registry import DEFAULT_RECONSTRUCTION_REGISTRY, ReconstructionRegistry
from pymaris.reconstruction.types import (
    ReconstructionPlan,
    ReconstructionPluginInfo,
    ReconstructionResult,
)


class DeconvolutionPlugin(ReconstructionPlugin):
    """Baseline deconvolution plugin backed by existing restoration backends."""

    info = ReconstructionPluginInfo(
        name="deconvolution",
        version="1.0",
        license="MIT",
        modality_tags=("deconvolution", "widefield", "confocal"),
        supported_dims=(2, 3, 4, 5),
        required_calibrations=(),
        notes="Uses restoration backends (classic/ai_denoise) with shared provenance contract.",
    )

    def prepare(
        self,
        image: ImageVolume,
        *,
        params: Mapping[str, Any],
        calibrations: Mapping[str, CalibrationArtifact],
    ) -> ReconstructionPlan:
        _ = calibrations
        method = str(params.get("method", "richardson_lucy"))
        backend = str(params.get("backend", "classic"))
        tile_shape = _resolve_tile_shape(params=params, image=image)
        return ReconstructionPlan(
            plugin_name=self.info.name,
            backend=backend,
            tile_shape=tile_shape,
            boundary_mode=str(params.get("boundary_mode", "reflect")),
            details={"method": method, "ndim": image.ndim},
        )

    def run(
        self,
        image: ImageVolume,
        *,
        params: Mapping[str, Any],
        calibrations: Mapping[str, CalibrationArtifact],
        plan: ReconstructionPlan,
    ) -> ReconstructionResult:
        _ = calibrations
        backend_name = str(params.get("backend", plan.backend))
        backend = DEFAULT_REGISTRY.get_restoration(backend_name)
        operation = str(params.get("operation", "deconvolve")).strip().lower()
        backend_params = dict(params)
        backend_params.pop("backend", None)
        backend_params.pop("tile_shape", None)
        backend_params.pop("chunk_shape", None)
        backend_params.pop("boundary_mode", None)
        if operation in {"denoise", "distance_map", "euclidean_distance_map", "edt"}:
            result = backend.denoise(image, **backend_params)
        else:
            result = backend.deconvolve(image, **backend_params)
        return ReconstructionResult(
            image=result.image,
            metadata={
                "plugin": self.info.name,
                "operation": operation,
                "restoration_backend": backend_name,
                "backend_metadata": dict(result.metadata),
            },
            qc={
                "finite_ratio": float(np.mean(np.isfinite(result.image.as_numpy()))),
            },
        )


class SIMPlugin(ReconstructionPlugin):
    """Baseline frequency-domain SIM reconstruction with artifact-aware QC."""

    info = ReconstructionPluginInfo(
        name="sim",
        version="1.0",
        license="MIT",
        modality_tags=("sim", "lattice_sim"),
        supported_dims=(3, 4, 5),
        required_calibrations=("otf", "sim_pattern"),
        notes="Wiener-style SIM reconstruction baseline using explicit OTF/pattern calibration.",
    )

    def prepare(
        self,
        image: ImageVolume,
        *,
        params: Mapping[str, Any],
        calibrations: Mapping[str, CalibrationArtifact],
    ) -> ReconstructionPlan:
        require_calibrations(calibrations, self.info.required_calibrations, plugin_name=self.info.name)
        balance = float(params.get("wiener_balance", 0.01))
        pattern_axis = _resolve_pattern_axis(params=params, image=image)
        combine_patterns = bool(params.get("combine_patterns", True))
        return ReconstructionPlan(
            plugin_name=self.info.name,
            backend=str(params.get("backend", "cpu")),
            boundary_mode=str(params.get("boundary_mode", "reflect")),
            details={
                "algorithm": "wiener_sim_baseline",
                "wiener_balance": balance,
                "pattern_axis": pattern_axis,
                "combine_patterns": combine_patterns,
            },
        )

    def run(
        self,
        image: ImageVolume,
        *,
        params: Mapping[str, Any],
        calibrations: Mapping[str, CalibrationArtifact],
        plan: ReconstructionPlan,
    ) -> ReconstructionResult:
        otf_artifact = calibrations["otf"]
        pattern_artifact = calibrations["sim_pattern"]
        otf = _require_calibration_array(otf_artifact, expected_kind="otf")
        data = np.asarray(image.as_numpy(), dtype=float)
        axes = list(image.axes)
        pattern_axis = _resolve_pattern_axis(params=params, image=image)
        combine_patterns = bool(params.get("combine_patterns", True))
        if pattern_axis is not None and combine_patterns and data.shape[pattern_axis] > 1:
            data = np.mean(data, axis=pattern_axis)
            axes.pop(pattern_axis)

        spatial_indices = _spatial_indices(axes)
        balance = float(params.get("wiener_balance", 0.01))
        reconstructed = _apply_over_spatial_slices(
            data,
            spatial_indices=spatial_indices,
            fn=lambda volume: _wiener_deconvolution(volume, otf=otf, balance=balance),
        )
        reconstructed = np.clip(reconstructed, 0.0, None)

        output_axes = tuple(axes)
        finite_ratio = float(np.mean(np.isfinite(reconstructed)))
        negative_fraction = float(np.mean(reconstructed < 0))
        high_freq_ratio = float(_high_frequency_energy_fraction(reconstructed, spatial_indices=spatial_indices))
        pattern_count = int(pattern_artifact.metadata.get("pattern_count", 0))
        return ReconstructionResult(
            image=image.with_array(reconstructed, axes=output_axes),
            metadata={
                "plugin": self.info.name,
                "algorithm": "wiener_sim_baseline",
                "pattern_count": pattern_count,
                "wiener_balance": balance,
            },
            qc={
                "finite_ratio": finite_ratio,
                "negative_fraction": negative_fraction,
                "high_frequency_energy_fraction": high_freq_ratio,
            },
        )


class STEDPlugin(ReconstructionPlugin):
    """STED deconvolution baseline using measured PSF + optional regularization."""

    info = ReconstructionPluginInfo(
        name="sted",
        version="1.0",
        license="MIT",
        modality_tags=("sted",),
        supported_dims=(3, 4, 5),
        required_calibrations=("psf",),
        notes="Richardson-Lucy baseline with measured PSF and optional Laplacian damping.",
    )

    def prepare(
        self,
        image: ImageVolume,
        *,
        params: Mapping[str, Any],
        calibrations: Mapping[str, CalibrationArtifact],
    ) -> ReconstructionPlan:
        require_calibrations(calibrations, self.info.required_calibrations, plugin_name=self.info.name)
        iterations = int(params.get("iterations", 15))
        regularization = float(params.get("regularization", 0.0))
        return ReconstructionPlan(
            plugin_name=self.info.name,
            backend=str(params.get("backend", "cpu")),
            boundary_mode=str(params.get("boundary_mode", "reflect")),
            details={
                "algorithm": "rl_sted_baseline",
                "iterations": iterations,
                "regularization": regularization,
            },
        )

    def run(
        self,
        image: ImageVolume,
        *,
        params: Mapping[str, Any],
        calibrations: Mapping[str, CalibrationArtifact],
        plan: ReconstructionPlan,
    ) -> ReconstructionResult:
        _ = plan
        psf = _require_calibration_array(calibrations["psf"], expected_kind="psf")
        data = np.asarray(image.as_numpy(), dtype=float)
        data = np.clip(data, 0.0, None)
        iterations = int(params.get("iterations", 15))
        regularization = float(params.get("regularization", 0.0))
        spatial_indices = _spatial_indices(image.axes)

        reconstructed = _apply_over_spatial_slices(
            data,
            spatial_indices=spatial_indices,
            fn=lambda volume: _richardson_lucy_deconvolution(
                volume,
                psf=psf,
                iterations=iterations,
                regularization=regularization,
            ),
        )
        reconstructed = np.clip(reconstructed, 0.0, None)
        psf_fwhm = _estimate_psf_fwhm(psf)
        return ReconstructionResult(
            image=image.with_array(reconstructed),
            metadata={
                "plugin": self.info.name,
                "algorithm": "rl_sted_baseline",
                "iterations": iterations,
                "regularization": regularization,
            },
            qc={
                "finite_ratio": float(np.mean(np.isfinite(reconstructed))),
                "psf_fwhm": psf_fwhm,
                "mean_intensity_gain": float(
                    np.mean(reconstructed) / max(float(np.mean(data)), 1e-6)
                ),
            },
        )


class SMLMPlugin(ReconstructionPlugin):
    """SMLM localization baseline with optional centroid drift correction."""

    info = ReconstructionPluginInfo(
        name="smlm",
        version="1.0",
        license="MIT",
        modality_tags=("smlm", "single_molecule"),
        supported_dims=(2, 3, 4),
        required_calibrations=("smlm_psf_model",),
        notes="Peak-localization baseline with subpixel centroiding and Gaussian rendering.",
    )

    def prepare(
        self,
        image: ImageVolume,
        *,
        params: Mapping[str, Any],
        calibrations: Mapping[str, CalibrationArtifact],
    ) -> ReconstructionPlan:
        require_calibrations(calibrations, self.info.required_calibrations, plugin_name=self.info.name)
        tile_shape = _resolve_tile_shape(params=params, image=image)
        return ReconstructionPlan(
            plugin_name=self.info.name,
            backend=str(params.get("backend", "cpu")),
            tile_shape=tile_shape,
        )

    def run(
        self,
        image: ImageVolume,
        *,
        params: Mapping[str, Any],
        calibrations: Mapping[str, CalibrationArtifact],
        plan: ReconstructionPlan,
    ) -> ReconstructionResult:
        model_params = _smlm_model_params(calibrations["smlm_psf_model"])
        _ = plan
        data = np.asarray(image.as_numpy(), dtype=float)
        spatial_indices = _spatial_indices(image.axes)
        min_distance = int(params.get("min_distance", model_params.get("min_distance", 2)))
        subpixel_radius = int(params.get("subpixel_radius", model_params.get("subpixel_radius", 2)))
        percentile = float(params.get("threshold_percentile", model_params.get("threshold_percentile", 99.7)))
        threshold = float(params.get("threshold", float(np.percentile(data, percentile))))
        localizations = _localize_points(
            data=data,
            axes=image.axes,
            spatial_indices=spatial_indices,
            threshold=threshold,
            min_distance=min_distance,
            subpixel_radius=subpixel_radius,
        )

        drift_mode = str(params.get("drift_correction", "none")).strip().lower()
        drift_summary: dict[str, Any] = {"mode": drift_mode, "applied": False}
        if drift_mode in {"centroid", "mean"} and "T" in image.axes:
            localizations, drift_summary = _apply_centroid_drift_correction(
                rows=localizations,
                time_axis="t",
                spatial_axes=[axis.lower() for axis in image.axes if axis in {"Z", "Y", "X"}],
            )

        render_sigma = float(params.get("render_sigma", model_params.get("sigma", 1.2)))
        rendered = _render_localizations(
            data_shape=data.shape,
            axes=image.axes,
            rows=localizations,
            render_sigma=render_sigma,
            spatial_indices=spatial_indices,
        )
        return ReconstructionResult(
            image=image.with_array(rendered),
            tables={"localizations": localizations},
            metadata={
                "plugin": self.info.name,
                "threshold": threshold,
                "min_distance": min_distance,
                "subpixel_radius": subpixel_radius,
                "drift": drift_summary,
            },
            qc={
                "localization_count": len(localizations),
                "threshold": threshold,
                "drift_applied": drift_summary.get("applied", False),
            },
        )


def register_default_reconstruction_plugins(
    registry: ReconstructionRegistry | None = None,
) -> ReconstructionRegistry:
    """Register built-in reconstruction plugins."""
    target = registry or DEFAULT_RECONSTRUCTION_REGISTRY
    target.register(DeconvolutionPlugin())
    target.register(SIMPlugin())
    target.register(STEDPlugin())
    target.register(SMLMPlugin())
    return target


def _resolve_tile_shape(params: Mapping[str, Any], image: ImageVolume) -> tuple[int, ...] | None:
    tile_shape = params.get("tile_shape")
    if tile_shape is None:
        return None
    values = tuple(int(value) for value in list(tile_shape))
    if len(values) != image.ndim:
        raise ValueError(
            f"tile_shape length ({len(values)}) must match image rank ({image.ndim})"
        )
    if any(value <= 0 for value in values):
        raise ValueError("tile_shape values must be positive integers")
    return values


def _resolve_pattern_axis(params: Mapping[str, Any], image: ImageVolume) -> int | None:
    axis = params.get("pattern_axis")
    if axis is None:
        if "C" in image.axes:
            return image.axes.index("C")
        if image.ndim > 2:
            spatial = set(_spatial_indices(image.axes))
            candidates = [idx for idx in range(image.ndim) if idx not in spatial]
            return candidates[0] if candidates else None
        return None
    if isinstance(axis, str):
        name = axis.upper()
        if name not in image.axes:
            raise ValueError(f"pattern_axis {axis!r} not present in image axes {image.axes!r}")
        return image.axes.index(name)
    index = int(axis)
    if index < 0 or index >= image.ndim:
        raise ValueError(f"pattern_axis index out of range: {index} for ndim={image.ndim}")
    return index


def _require_calibration_array(artifact: CalibrationArtifact, *, expected_kind: str) -> np.ndarray:
    payload = artifact.payload
    if payload is None:
        raise ValueError(
            f"calibration '{artifact.name}' is required as an array payload for kind '{expected_kind}'. "
            f"Source: {artifact.source}"
        )
    array = np.asarray(payload)
    if array.size == 0:
        raise ValueError(f"calibration '{artifact.name}' payload is empty")
    return np.asarray(array, dtype=float)


def _spatial_indices(axes: Sequence[str]) -> list[int]:
    indices = [index for index, axis in enumerate(axes) if str(axis).upper() in {"Z", "Y", "X"}]
    if indices:
        return indices
    return list(range(len(axes)))


def _apply_over_spatial_slices(
    data: np.ndarray,
    *,
    spatial_indices: Sequence[int],
    fn: Any,
) -> np.ndarray:
    ndim = int(data.ndim)
    spatial = [int(value) for value in spatial_indices]
    if set(spatial) == set(range(ndim)):
        return np.asarray(fn(data), dtype=float)

    non_spatial = [index for index in range(ndim) if index not in spatial]
    permutation = non_spatial + spatial
    transposed = np.transpose(data, axes=permutation)
    non_spatial_shape = transposed.shape[: len(non_spatial)]
    spatial_shape = transposed.shape[len(non_spatial) :]
    flat = transposed.reshape((-1,) + spatial_shape)
    transformed = np.zeros_like(flat, dtype=float)
    for index in range(flat.shape[0]):
        transformed[index] = np.asarray(fn(flat[index]), dtype=float)
    restored = transformed.reshape(non_spatial_shape + spatial_shape)
    inverse = np.argsort(permutation)
    return np.transpose(restored, axes=inverse)


def _center_crop_or_pad(array: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    result = np.asarray(array)
    if tuple(result.shape) == tuple(target_shape):
        return result
    slices: list[slice] = []
    pads: list[tuple[int, int]] = []
    for current, target in zip(result.shape, target_shape):
        if current > target:
            start = (current - target) // 2
            slices.append(slice(start, start + target))
            pads.append((0, 0))
        else:
            slices.append(slice(0, current))
            before = (target - current) // 2
            after = target - current - before
            pads.append((before, after))
    cropped = result[tuple(slices)]
    if any(before or after for before, after in pads):
        cropped = np.pad(cropped, pads, mode="constant")
    return np.asarray(cropped)


def _wiener_deconvolution(volume: np.ndarray, *, otf: np.ndarray, balance: float) -> np.ndarray:
    target_shape = tuple(int(value) for value in volume.shape)
    fft_axes = tuple(range(len(target_shape)))
    if otf.ndim != len(target_shape):
        raise ValueError(
            f"OTF dimensionality mismatch: expected {len(target_shape)}D, got {otf.ndim}D"
        )
    if np.iscomplexobj(otf):
        frequency_response = _center_crop_or_pad(np.asarray(otf, dtype=complex), target_shape)
    else:
        kernel = _center_crop_or_pad(np.asarray(otf, dtype=float), target_shape)
        frequency_response = np.fft.fftn(kernel, s=target_shape, axes=fft_axes)
    signal_fft = np.fft.fftn(volume, s=target_shape, axes=fft_axes)
    denominator = (np.abs(frequency_response) ** 2) + max(float(balance), 1e-9)
    reconstructed_fft = signal_fft * np.conj(frequency_response) / denominator
    return np.real(np.fft.ifftn(reconstructed_fft, s=target_shape, axes=fft_axes))


def _high_frequency_energy_fraction(data: np.ndarray, *, spatial_indices: Sequence[int]) -> float:
    if data.size == 0:
        return 0.0
    spatial = [int(value) for value in spatial_indices]
    ndim = int(data.ndim)
    if set(spatial) != set(range(ndim)):
        data = _apply_over_spatial_slices(
            np.asarray(data, dtype=float),
            spatial_indices=spatial,
            fn=lambda x: x,
        )
    transformed = np.fft.fftn(np.asarray(data, dtype=float))
    magnitude = np.abs(np.fft.fftshift(transformed))
    coords = np.indices(magnitude.shape, dtype=float)
    center = np.asarray([(size - 1) / 2 for size in magnitude.shape], dtype=float)
    radius = np.zeros_like(magnitude, dtype=float)
    for axis in range(magnitude.ndim):
        size = max(magnitude.shape[axis], 1)
        radius += ((coords[axis] - center[axis]) / size) ** 2
    high_freq = radius >= np.quantile(radius, 0.75)
    total_energy = float(np.sum(magnitude))
    if total_energy <= 0:
        return 0.0
    return float(np.sum(magnitude[high_freq]) / total_energy)


def _richardson_lucy_deconvolution(
    volume: np.ndarray,
    *,
    psf: np.ndarray,
    iterations: int,
    regularization: float,
) -> np.ndarray:
    psf_kernel = _center_crop_or_pad(psf, volume.shape)
    psf_sum = float(np.sum(psf_kernel))
    if psf_sum <= 0:
        raise ValueError("PSF calibration sum must be > 0")
    psf_kernel = psf_kernel / psf_sum
    psf_mirror = np.flip(psf_kernel)
    estimate = np.clip(np.asarray(volume, dtype=float), 1e-6, None)
    observed = np.clip(np.asarray(volume, dtype=float), 0.0, None)
    epsilon = 1e-6
    for _ in range(max(1, int(iterations))):
        blurred = signal.fftconvolve(estimate, psf_kernel, mode="same")
        relative = observed / np.maximum(blurred, epsilon)
        estimate *= signal.fftconvolve(relative, psf_mirror, mode="same")
        if regularization > 0:
            estimate -= float(regularization) * ndimage.laplace(estimate)
            estimate = np.clip(estimate, 0.0, None)
    return estimate


def _estimate_psf_fwhm(psf: np.ndarray) -> list[float]:
    normalized = np.asarray(psf, dtype=float)
    if normalized.size == 0:
        return []
    max_value = float(np.max(normalized))
    if max_value <= 0:
        return [0.0 for _ in range(normalized.ndim)]
    center = tuple(int(value) for value in np.unravel_index(np.argmax(normalized), normalized.shape))
    half = max_value / 2.0
    widths: list[float] = []
    for axis in range(normalized.ndim):
        slicer = [center[dim] for dim in range(normalized.ndim)]
        slicer[axis] = slice(None)
        profile = np.asarray(normalized[tuple(slicer)], dtype=float)
        above = np.where(profile >= half)[0]
        if above.size == 0:
            widths.append(0.0)
        else:
            widths.append(float(above[-1] - above[0] + 1))
    return widths


def _smlm_model_params(artifact: CalibrationArtifact) -> dict[str, Any]:
    if isinstance(artifact.payload, Mapping):
        return {str(key): value for key, value in dict(artifact.payload).items()}
    return {}


def _localize_points(
    *,
    data: np.ndarray,
    axes: Sequence[str],
    spatial_indices: Sequence[int],
    threshold: float,
    min_distance: int,
    subpixel_radius: int,
) -> list[dict[str, Any]]:
    non_spatial_indices = [index for index in range(data.ndim) if index not in spatial_indices]
    rows: list[dict[str, Any]] = []
    if not non_spatial_indices:
        rows.extend(
            _localize_points_single(
                frame=data,
                frame_prefix={},
                spatial_axis_names=[str(axes[idx]).lower() for idx in spatial_indices],
                threshold=threshold,
                min_distance=min_distance,
                subpixel_radius=subpixel_radius,
            )
        )
    else:
        permutation = non_spatial_indices + list(spatial_indices)
        transposed = np.transpose(data, axes=permutation)
        non_shape = transposed.shape[: len(non_spatial_indices)]
        spatial_shape = transposed.shape[len(non_spatial_indices) :]
        flat = transposed.reshape((-1,) + spatial_shape)
        for flat_index in range(flat.shape[0]):
            non_index = np.unravel_index(flat_index, non_shape)
            frame_prefix: dict[str, Any] = {}
            for axis_offset, axis_idx in enumerate(non_spatial_indices):
                frame_prefix[str(axes[axis_idx]).lower()] = int(non_index[axis_offset])
            rows.extend(
                _localize_points_single(
                    frame=flat[flat_index],
                    frame_prefix=frame_prefix,
                    spatial_axis_names=[str(axes[idx]).lower() for idx in spatial_indices],
                    threshold=threshold,
                    min_distance=min_distance,
                    subpixel_radius=subpixel_radius,
                )
            )
    for row_id, row in enumerate(rows):
        row["id"] = int(row_id)
    return rows


def _localize_points_single(
    *,
    frame: np.ndarray,
    frame_prefix: Mapping[str, Any],
    spatial_axis_names: Sequence[str],
    threshold: float,
    min_distance: int,
    subpixel_radius: int,
) -> list[dict[str, Any]]:
    peaks = _peak_coordinates(frame, threshold=threshold, min_distance=min_distance)
    rows: list[dict[str, Any]] = []
    for peak in peaks:
        refined = _subpixel_centroid(frame, peak=peak, radius=subpixel_radius)
        row: dict[str, Any] = {str(key): value for key, value in frame_prefix.items()}
        for axis_name, value in zip(spatial_axis_names, refined):
            row[axis_name] = float(value)
        row["intensity"] = float(frame[tuple(int(round(v)) for v in refined)])
        rows.append(row)
    return rows


def _peak_coordinates(frame: np.ndarray, *, threshold: float, min_distance: int) -> np.ndarray:
    if HAS_SKIMAGE_FEATURE and peak_local_max is not None:
        coords = peak_local_max(
            np.asarray(frame, dtype=float),
            min_distance=max(1, int(min_distance)),
            threshold_abs=float(threshold),
            exclude_border=False,
        )
        return np.asarray(coords, dtype=int)

    footprint = tuple([max(1, 2 * int(min_distance) + 1)] * frame.ndim)
    max_filtered = ndimage.maximum_filter(frame, size=footprint, mode="nearest")
    peaks_mask = (frame >= max_filtered) & (frame >= float(threshold))
    return np.argwhere(peaks_mask)


def _subpixel_centroid(frame: np.ndarray, *, peak: np.ndarray, radius: int) -> np.ndarray:
    center = np.asarray(peak, dtype=float)
    radius_value = max(0, int(radius))
    if radius_value == 0:
        return center
    slices: list[slice] = []
    for axis, coordinate in enumerate(peak):
        start = max(0, int(coordinate) - radius_value)
        end = min(frame.shape[axis], int(coordinate) + radius_value + 1)
        slices.append(slice(start, end))
    patch = np.asarray(frame[tuple(slices)], dtype=float)
    if patch.size == 0:
        return center
    patch = np.clip(patch - np.min(patch), 0.0, None)
    total = float(np.sum(patch))
    if total <= 0:
        return center
    coords = np.indices(patch.shape, dtype=float)
    refined = []
    for axis in range(patch.ndim):
        weighted = float(np.sum(coords[axis] * patch) / total)
        refined.append(float(weighted + slices[axis].start))
    return np.asarray(refined, dtype=float)


def _apply_centroid_drift_correction(
    *,
    rows: list[dict[str, Any]],
    time_axis: str,
    spatial_axes: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return rows, {"mode": "centroid", "applied": False}
    if time_axis not in rows[0]:
        return rows, {"mode": "centroid", "applied": False}
    frames: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        frames.setdefault(int(row[time_axis]), []).append(row)
    if len(frames) <= 1:
        return rows, {"mode": "centroid", "applied": False}
    reference_frame = min(frames)
    reference_centroid = np.asarray(
        [
            float(np.mean([item[axis] for item in frames[reference_frame]]))
            for axis in spatial_axes
        ],
        dtype=float,
    )
    drift_values: dict[int, list[float]] = {}
    corrected: list[dict[str, Any]] = []
    for frame_idx, frame_rows in frames.items():
        centroid = np.asarray(
            [float(np.mean([item[axis] for item in frame_rows])) for axis in spatial_axes],
            dtype=float,
        )
        delta = centroid - reference_centroid
        drift_values[frame_idx] = [float(value) for value in delta]
        for item in frame_rows:
            updated = dict(item)
            for axis_index, axis in enumerate(spatial_axes):
                updated[axis] = float(updated[axis] - delta[axis_index])
            corrected.append(updated)
    corrected.sort(key=lambda row: int(row["id"]))
    rms = float(
        np.sqrt(
            np.mean(
                [
                    sum(value * value for value in vector)
                    for vector in drift_values.values()
                ]
            )
        )
    )
    return corrected, {"mode": "centroid", "applied": True, "drift_by_frame": drift_values, "rms": rms}


def _render_localizations(
    *,
    data_shape: Sequence[int],
    axes: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    render_sigma: float,
    spatial_indices: Sequence[int],
) -> np.ndarray:
    rendered = np.zeros(tuple(int(value) for value in data_shape), dtype=float)
    if not rows:
        return rendered
    for row in rows:
        index = []
        for axis in axes:
            key = str(axis).lower()
            value = row.get(key, 0.0)
            idx = int(round(float(value)))
            idx = min(max(idx, 0), rendered.shape[len(index)] - 1)
            index.append(idx)
        rendered[tuple(index)] += 1.0
    if render_sigma <= 0:
        return rendered
    return _apply_over_spatial_slices(
        rendered,
        spatial_indices=spatial_indices,
        fn=lambda volume: ndimage.gaussian_filter(volume, sigma=float(render_sigma)),
    )
