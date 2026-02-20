"""Core data contracts for modular reconstruction plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pymaris.data_model import ImageVolume


@dataclass(frozen=True)
class ReconstructionPluginInfo:
    """Describes a reconstruction plugin and its compatibility constraints."""

    name: str
    version: str
    license: str
    modality_tags: tuple[str, ...]
    supported_dims: tuple[int, ...]
    required_calibrations: tuple[str, ...] = ()
    notes: str = ""
    external_tool: bool = False


@dataclass
class ReconstructionPlan:
    """Execution plan returned by plugin prepare() step."""

    plugin_name: str
    backend: str = "cpu"
    tile_shape: tuple[int, ...] | None = None
    chunk_shape: tuple[int, ...] | None = None
    boundary_mode: str = "reflect"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructionResult:
    """Unified plugin output payload."""

    image: ImageVolume | None = None
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
