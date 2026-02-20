"""Backend interfaces, registries, and baseline adapters."""

from pymaris.backends.baseline import register_default_backends
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

register_default_backends()

__all__ = [
    "BackendCapability",
    "BackendInfo",
    "BackendRegistry",
    "DEFAULT_REGISTRY",
    "SegmentationBackend",
    "TrackingBackend",
    "TracingBackend",
    "RestorationBackend",
    "LabelsResult",
    "TracksResult",
    "TraceResult",
    "ImageResult",
    "register_default_backends",
]
