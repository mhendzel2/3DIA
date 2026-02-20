"""Stable plugin contract for reconstruction modalities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

from pymaris.data_model import ImageVolume
from pymaris.reconstruction.calibration import CalibrationArtifact
from pymaris.reconstruction.types import (
    ReconstructionPlan,
    ReconstructionPluginInfo,
    ReconstructionResult,
)


class ReconstructionPlugin(ABC):
    """Base class for modality plugins (deconv/SIM/STED/SMLM)."""

    info: ReconstructionPluginInfo

    @abstractmethod
    def prepare(
        self,
        image: ImageVolume,
        *,
        params: Mapping[str, Any],
        calibrations: Mapping[str, CalibrationArtifact],
    ) -> ReconstructionPlan:
        """Validate inputs and return an execution plan."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        image: ImageVolume,
        *,
        params: Mapping[str, Any],
        calibrations: Mapping[str, CalibrationArtifact],
        plan: ReconstructionPlan,
    ) -> ReconstructionResult:
        """Execute reconstruction using explicit params/calibrations."""
        raise NotImplementedError

    def qc(
        self,
        result: ReconstructionResult,
        *,
        image: ImageVolume,
        params: Mapping[str, Any],
        calibrations: Mapping[str, CalibrationArtifact],
    ) -> dict[str, Any]:
        """Return method-specific QC metrics."""
        return dict(result.qc)

    def export(
        self,
        result: ReconstructionResult,
        destination: Path,
        *,
        format: str,
    ) -> dict[str, Any]:
        """Optional plugin-defined export hooks; default no-op."""
        _ = destination
        _ = format
        return {}
