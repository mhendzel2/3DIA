"""Execution engine for reconstruction plugins with provenance logging."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Mapping

from pymaris.data_model import ImageVolume
from pymaris.reconstruction.calibration import CalibrationArtifact
from pymaris.reconstruction.qa import build_model_provenance, build_reconstruction_qa_report
from pymaris.reconstruction.registry import DEFAULT_RECONSTRUCTION_REGISTRY, ReconstructionRegistry
from pymaris.reconstruction.types import ReconstructionResult


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReconstructionEngine:
    """Runs plugin prepare/run/qc lifecycle and captures provenance."""

    def __init__(self, registry: ReconstructionRegistry | None = None) -> None:
        self.registry = registry or DEFAULT_RECONSTRUCTION_REGISTRY

    def run(
        self,
        *,
        plugin_name: str,
        image: ImageVolume,
        params: Mapping[str, Any] | None = None,
        calibrations: Mapping[str, CalibrationArtifact] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> ReconstructionResult:
        plugin = self.registry.get(plugin_name)
        params_dict = dict(params or {})
        calibration_map = dict(calibrations or {})

        started = perf_counter()
        plan = plugin.prepare(image, params=params_dict, calibrations=calibration_map)
        result = plugin.run(
            image,
            params=params_dict,
            calibrations=calibration_map,
            plan=plan,
        )
        qc_payload = plugin.qc(
            result,
            image=image,
            params=params_dict,
            calibrations=calibration_map,
        )
        elapsed_seconds = perf_counter() - started

        combined_qc = dict(result.qc)
        combined_qc.update(dict(qc_payload))
        result.qc = combined_qc

        model_provenance = build_model_provenance(
            plugin_info=plugin.info,
            params=params_dict,
            metadata=result.metadata,
            calibrations=calibration_map,
        )
        qa_report = build_reconstruction_qa_report(input_image=image, result=result)

        artifacts = dict(result.artifacts)
        artifacts["model_provenance"] = model_provenance
        artifacts["qa_report"] = qa_report
        result.artifacts = artifacts

        provenance = {
            "timestamp_utc": _utc_now(),
            "plugin": {
                "name": plugin.info.name,
                "version": plugin.info.version,
                "license": plugin.info.license,
                "modality_tags": list(plugin.info.modality_tags),
                "supported_dims": list(plugin.info.supported_dims),
                "required_calibrations": list(plugin.info.required_calibrations),
                "external_tool": plugin.info.external_tool,
            },
            "params": params_dict,
            "plan": {
                "backend": plan.backend,
                "tile_shape": list(plan.tile_shape) if plan.tile_shape else None,
                "chunk_shape": list(plan.chunk_shape) if plan.chunk_shape else None,
                "boundary_mode": plan.boundary_mode,
                "details": dict(plan.details),
            },
            "calibrations": {key: value.summary() for key, value in calibration_map.items()},
            "model_provenance": model_provenance,
            "qa_report": {
                "status": qa_report.get("status"),
                "warning_count": qa_report.get("warning_count"),
                "warnings": list(qa_report.get("warnings", [])),
            },
            "runtime": {"elapsed_seconds": elapsed_seconds},
            "context": dict(context or {}),
        }

        result.provenance = provenance
        return result
