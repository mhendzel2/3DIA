"""QA report and model provenance helpers for reconstruction outputs."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from pymaris.data_model import ImageVolume
from pymaris.reconstruction.calibration import CalibrationArtifact
from pymaris.reconstruction.types import ReconstructionPluginInfo, ReconstructionResult


def build_model_provenance(
    *,
    plugin_info: ReconstructionPluginInfo,
    params: Mapping[str, Any],
    metadata: Mapping[str, Any],
    calibrations: Mapping[str, CalibrationArtifact],
) -> dict[str, Any]:
    """Build a normalized model provenance object for reconstruction outputs."""
    backend_name = str(
        metadata.get("restoration_backend")
        or metadata.get("backend")
        or params.get("backend")
        or "cpu"
    )

    mode = "classical"
    if bool(metadata.get("ai_denoising")) or backend_name in {"cellpose", "stardist", "ai_denoise"}:
        mode = "ml_assisted"
    elif plugin_info.name == "smlm":
        mode = "model_based"

    calibrations_summary: dict[str, dict[str, Any]] = {}
    for name, artifact in calibrations.items():
        calibrations_summary[str(name)] = {
            "kind": artifact.kind,
            "source": artifact.source,
            "metadata": dict(artifact.metadata),
        }

    return {
        "plugin": {
            "name": plugin_info.name,
            "version": plugin_info.version,
            "modality_tags": list(plugin_info.modality_tags),
        },
        "backend": backend_name,
        "mode": mode,
        "parameters": dict(params),
        "calibrations": calibrations_summary,
    }


def build_reconstruction_qa_report(
    *,
    input_image: ImageVolume,
    result: ReconstructionResult,
) -> dict[str, Any]:
    """Create a deterministic QA report summarizing reconstruction behavior."""
    input_stats = _image_stats(input_image)
    output_stats = _image_stats(result.image) if result.image is not None else None

    warnings: list[str] = []
    if output_stats is None:
        warnings.append("result image is missing")
    else:
        if not output_stats["all_finite"]:
            warnings.append("output contains non-finite values")
        if output_stats["shape"] != input_stats["shape"]:
            warnings.append("output shape differs from input shape")
        if output_stats["max"] is not None and input_stats["max"] is not None:
            if output_stats["max"] > input_stats["max"] * 20.0 + 1e-9:
                warnings.append("output dynamic range amplified >20x")

    status = "ok" if not warnings else "warning"
    return {
        "status": status,
        "warning_count": len(warnings),
        "warnings": warnings,
        "input": input_stats,
        "output": output_stats,
        "plugin_qc": dict(result.qc),
    }


def _image_stats(image: ImageVolume | None) -> dict[str, Any] | None:
    if image is None:
        return None
    array = np.asarray(image.as_numpy())
    if array.size == 0:
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "all_finite": True,
        }
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "all_finite": bool(np.all(np.isfinite(array))),
    }
