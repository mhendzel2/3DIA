"""Backend registries for segmentation/tracking/tracing/restoration adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

from pymaris.backends.types import (
    BackendInfo,
    RestorationBackend,
    SegmentationBackend,
    TracingBackend,
    TrackingBackend,
)


@dataclass
class BackendRegistry:
    """Container for typed backend registries."""

    segmentation: Dict[str, SegmentationBackend] = field(default_factory=dict)
    tracking: Dict[str, TrackingBackend] = field(default_factory=dict)
    tracing: Dict[str, TracingBackend] = field(default_factory=dict)
    restoration: Dict[str, RestorationBackend] = field(default_factory=dict)

    def register_segmentation(self, name: str, backend: SegmentationBackend) -> None:
        self.segmentation[name] = backend

    def register_tracking(self, name: str, backend: TrackingBackend) -> None:
        self.tracking[name] = backend

    def register_tracing(self, name: str, backend: TracingBackend) -> None:
        self.tracing[name] = backend

    def register_restoration(self, name: str, backend: RestorationBackend) -> None:
        self.restoration[name] = backend

    def get_segmentation(self, name: str) -> SegmentationBackend:
        return self.segmentation[name]

    def get_tracking(self, name: str) -> TrackingBackend:
        return self.tracking[name]

    def get_tracing(self, name: str) -> TracingBackend:
        return self.tracing[name]

    def get_restoration(self, name: str) -> RestorationBackend:
        return self.restoration[name]

    def backend_names(self, backend_type: str) -> list[str]:
        """Return sorted backend names for a backend type."""
        return sorted(self._registry_for_type(backend_type).keys())

    def list_backend_info(
        self,
        backend_type: str | None = None,
    ) -> dict[str, list[BackendInfo]]:
        """Return backend info payloads grouped by backend type."""
        grouped: dict[str, list[BackendInfo]] = {}
        for registry_type, registry in self._iter_registries(backend_type):
            entries: list[BackendInfo] = []
            for name, backend in sorted(registry.items()):
                info = getattr(backend, "info", None)
                if isinstance(info, BackendInfo):
                    entries.append(
                        BackendInfo(
                            name=name,
                            version=info.version,
                            params=dict(info.params),
                            capability=info.capability,
                        )
                    )
                else:
                    entries.append(BackendInfo(name=name))
            grouped[registry_type] = entries
        return grouped

    def find_backends(
        self,
        backend_type: str,
        *,
        ndim: int | None = None,
        modality: str | None = None,
        requires_time: bool | None = None,
        supports_multichannel: bool | None = None,
    ) -> list[str]:
        """Find backend names whose declared capabilities match filters."""
        matches: list[str] = []
        for name, backend in sorted(self._registry_for_type(backend_type).items()):
            info = getattr(backend, "info", None)
            capability = getattr(info, "capability", None)
            if capability is None:
                # Unknown capabilities are treated as generic and always available.
                matches.append(name)
                continue
            if capability.matches(
                ndim=ndim,
                modality=modality,
                requires_time=requires_time,
                supports_multichannel=supports_multichannel,
            ):
                matches.append(name)
        return matches

    def resolve_backend_info(self, backend_type: str, name: str) -> BackendInfo:
        """Return backend info for a registered backend key."""
        backend = self._registry_for_type(backend_type)[name]
        info = getattr(backend, "info", None)
        if isinstance(info, BackendInfo):
            return BackendInfo(
                name=name,
                version=info.version,
                params=dict(info.params),
                capability=info.capability,
            )
        return BackendInfo(name=name)

    def _registry_for_type(self, backend_type: str) -> Dict[str, Any]:
        mapping: dict[str, Dict[str, Any]] = {
            "segmentation": self.segmentation,
            "tracking": self.tracking,
            "tracing": self.tracing,
            "restoration": self.restoration,
        }
        if backend_type not in mapping:
            allowed = ", ".join(sorted(mapping.keys()))
            raise ValueError(f"unsupported backend_type {backend_type!r}; expected one of {allowed}")
        return mapping[backend_type]

    def _iter_registries(
        self,
        backend_type: str | None,
    ) -> Iterable[tuple[str, Dict[str, Any]]]:
        if backend_type is None:
            yield "segmentation", self.segmentation
            yield "tracking", self.tracking
            yield "tracing", self.tracing
            yield "restoration", self.restoration
            return
        yield backend_type, self._registry_for_type(backend_type)


DEFAULT_REGISTRY = BackendRegistry()
