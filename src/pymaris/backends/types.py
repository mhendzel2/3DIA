"""Backend interfaces and standardized backend result payloads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from pymaris.data_model import ImageVolume


@dataclass(frozen=True)
class BackendCapability:
    """Declared backend capability metadata used for backend discovery."""

    task: str
    dimensions: tuple[int, ...] = ()
    modalities: tuple[str, ...] = ()
    supports_time: bool = False
    supports_multichannel: bool = False
    notes: str = ""

    def matches(
        self,
        *,
        ndim: int | None = None,
        modality: str | None = None,
        requires_time: bool | None = None,
        supports_multichannel: bool | None = None,
    ) -> bool:
        """Return True when this capability satisfies the provided filters."""
        if ndim is not None and self.dimensions and ndim not in self.dimensions:
            return False
        if modality is not None and self.modalities:
            normalized = str(modality).strip().lower()
            if normalized and normalized not in {value.lower() for value in self.modalities}:
                return False
        if requires_time is True and not self.supports_time:
            return False
        if requires_time is False and self.supports_time:
            return False
        if supports_multichannel is True and not self.supports_multichannel:
            return False
        if supports_multichannel is False and self.supports_multichannel:
            return False
        return True


@dataclass(frozen=True)
class BackendInfo:
    """Backend identity and runtime parameter metadata."""

    name: str
    version: str = "unknown"
    params: dict[str, Any] = field(default_factory=dict)
    capability: BackendCapability | None = None


@dataclass
class LabelsResult:
    """Standard output for segmentation backends."""

    labels: Any
    table: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TracksResult:
    """Standard output for tracking backends."""

    tracks: Any
    table: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceResult:
    """Standard output for tracing backends."""

    graph: dict[str, Any]
    table: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageResult:
    """Standard output for restoration backends."""

    image: ImageVolume
    table: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class SegmentationBackend(ABC):
    """Base interface for segmentation adapters."""

    info: BackendInfo

    @abstractmethod
    def segment_instances(self, image: ImageVolume, **params: Any) -> LabelsResult:
        raise NotImplementedError


class TrackingBackend(ABC):
    """Base interface for tracking adapters."""

    info: BackendInfo

    @abstractmethod
    def track(self, labels_over_time: Sequence[Any], **params: Any) -> TracksResult:
        raise NotImplementedError


class TracingBackend(ABC):
    """Base interface for tracing adapters."""

    info: BackendInfo

    @abstractmethod
    def trace(self, image: ImageVolume, **params: Any) -> TraceResult:
        raise NotImplementedError


class RestorationBackend(ABC):
    """Base interface for restoration adapters."""

    info: BackendInfo

    @abstractmethod
    def denoise(self, image: ImageVolume, **params: Any) -> ImageResult:
        raise NotImplementedError

    @abstractmethod
    def deconvolve(self, image: ImageVolume, **params: Any) -> ImageResult:
        raise NotImplementedError
