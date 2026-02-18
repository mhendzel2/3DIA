"""Explicit calibration objects used by reconstruction plugins."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:  # pragma: no cover - optional dependency
    import imageio.v3 as iio

    HAS_IMAGEIO = True
except Exception:  # pragma: no cover
    iio = None  # type: ignore[assignment]
    HAS_IMAGEIO = False


@dataclass(frozen=True)
class CalibrationArtifact:
    """Serializable calibration descriptor passed into plugin execution."""

    name: str
    kind: str
    source: str | None = None
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "metadata": dict(self.metadata),
            "has_payload": self.payload is not None,
        }


@dataclass(frozen=True)
class PSFCalibration:
    """Measured or synthetic PSF calibration with explicit physical scale."""

    array: np.ndarray
    axes: tuple[str, ...] = ("Z", "Y", "X")
    voxel_size: tuple[float, ...] | None = None
    wavelength_nm: float | None = None
    refractive_index: float | None = None
    source: str | None = None

    def to_artifact(self, name: str = "psf") -> CalibrationArtifact:
        metadata: dict[str, Any] = {
            "axes": list(self.axes),
        }
        if self.voxel_size is not None:
            metadata["voxel_size"] = list(self.voxel_size)
        if self.wavelength_nm is not None:
            metadata["wavelength_nm"] = float(self.wavelength_nm)
        if self.refractive_index is not None:
            metadata["refractive_index"] = float(self.refractive_index)
        return CalibrationArtifact(
            name=name,
            kind="psf",
            source=self.source,
            payload=np.asarray(self.array),
            metadata=metadata,
        )


@dataclass(frozen=True)
class OTFCalibration:
    """Optical transfer function calibration used by SIM-like pipelines."""

    array: np.ndarray
    axes: tuple[str, ...] = ("Z", "Y", "X")
    source: str | None = None

    def to_artifact(self, name: str = "otf") -> CalibrationArtifact:
        return CalibrationArtifact(
            name=name,
            kind="otf",
            source=self.source,
            payload=np.asarray(self.array),
            metadata={"axes": list(self.axes)},
        )


@dataclass(frozen=True)
class SIMPatternCalibration:
    """SIM pattern estimates (angles/phases/frequencies) as explicit input."""

    pattern_count: int
    parameters: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    def to_artifact(self, name: str = "sim_pattern") -> CalibrationArtifact:
        return CalibrationArtifact(
            name=name,
            kind="sim_pattern",
            source=self.source,
            payload=dict(self.parameters),
            metadata={"pattern_count": int(self.pattern_count)},
        )


@dataclass(frozen=True)
class SMLMPSFModelCalibration:
    """SMLM PSF model description for localization fitting."""

    model_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    def to_artifact(self, name: str = "smlm_psf_model") -> CalibrationArtifact:
        return CalibrationArtifact(
            name=name,
            kind="smlm_psf_model",
            source=self.source,
            payload=dict(self.parameters),
            metadata={"model_name": self.model_name},
        )


def calibrations_from_cli_specs(specs: list[str]) -> dict[str, CalibrationArtifact]:
    """Parse `name=path` CLI calibration arguments into artifact descriptors."""
    artifacts: dict[str, CalibrationArtifact] = {}
    for raw in specs:
        if "=" not in raw:
            raise ValueError(f"invalid --calibration value {raw!r}; expected name=path")
        name, value = raw.split("=", 1)
        key = name.strip()
        path_value = value.strip()
        if not key:
            raise ValueError(f"invalid --calibration value {raw!r}; calibration name cannot be empty")
        if not path_value:
            raise ValueError(f"invalid --calibration value {raw!r}; calibration path cannot be empty")
        path = Path(path_value)
        payload, payload_meta = _load_calibration_payload(path)
        metadata = {"exists": path.exists()}
        metadata.update(payload_meta)
        artifacts[key] = CalibrationArtifact(
            name=key,
            kind=key,
            source=str(path),
            payload=payload,
            metadata=metadata,
        )
    return artifacts


def require_calibrations(
    available: Mapping[str, CalibrationArtifact],
    required: tuple[str, ...],
    *,
    plugin_name: str,
) -> None:
    """Raise a clear error when required calibration objects are missing."""
    missing = [name for name in required if name not in available]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"plugin '{plugin_name}' requires calibration objects: {missing_text}. "
            "Provide them explicitly via calibration inputs."
        )


def _load_calibration_payload(path: Path) -> tuple[Any, dict[str, Any]]:
    if not path.exists():
        return None, {"loaded": False, "reason": "path_not_found"}

    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            return np.load(path, allow_pickle=False), {"loaded": True, "format": "npy"}
        if suffix == ".npz":
            archive = np.load(path, allow_pickle=False)
            keys = list(archive.keys())
            if not keys:
                return None, {"loaded": False, "reason": "empty_npz"}
            return archive[keys[0]], {"loaded": True, "format": "npz", "key": keys[0]}
        if suffix in {".json"}:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload, {"loaded": True, "format": "json"}
        if suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"} and HAS_IMAGEIO:
            return np.asarray(iio.imread(path)), {"loaded": True, "format": "image"}
        text = path.read_text(encoding="utf-8").strip()
        if text:
            try:
                payload = json.loads(text)
                return payload, {"loaded": True, "format": "json_text"}
            except Exception:
                pass
        return None, {"loaded": False, "reason": "unsupported_format"}
    except Exception as exc:
        return None, {"loaded": False, "reason": f"load_error:{exc}"}
