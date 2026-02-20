"""Tests for backend interfaces, registry wiring, and baseline adapters."""

from __future__ import annotations

import numpy as np

from pymaris.backends import DEFAULT_REGISTRY
from pymaris.data_model import ImageVolume


def test_default_registry_contains_baseline_backends() -> None:
    assert "watershed" in DEFAULT_REGISTRY.segmentation
    assert "hungarian" in DEFAULT_REGISTRY.tracking
    assert "skeleton" in DEFAULT_REGISTRY.tracing
    assert "classic" in DEFAULT_REGISTRY.restoration
    assert "ai_denoise" in DEFAULT_REGISTRY.restoration


def test_watershed_backend_returns_labels_and_metadata() -> None:
    image = ImageVolume(
        array=np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        axes=("Y", "X"),
    )
    backend = DEFAULT_REGISTRY.get_segmentation("watershed")
    result = backend.segment_instances(image)
    assert result.labels.shape == (4, 4)
    assert result.metadata["backend"] == "watershed"
    assert result.table["object_count"] >= 1


def test_watershed_backend_includes_distance_measurements_for_multiple_objects() -> None:
    image = ImageVolume(
        array=np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1],
                [0, 1, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        axes=("Y", "X"),
        pixel_size={"Y": 0.5, "X": 0.5},
    )
    backend = DEFAULT_REGISTRY.get_segmentation("watershed")
    result = backend.segment_instances(image)
    assert result.table["object_count"] >= 2
    assert result.table["distance_measurements_available"] is True
    assert result.table["mean_nn_distance"] > 0


def test_tracking_backend_links_single_object_over_time() -> None:
    frame0 = np.zeros((8, 8), dtype=np.uint8)
    frame1 = np.zeros((8, 8), dtype=np.uint8)
    frame0[2:4, 2:4] = 1
    frame1[3:5, 3:5] = 1
    backend = DEFAULT_REGISTRY.get_tracking("hungarian")
    result = backend.track([frame0, frame1], max_distance=5.0)
    assert result.metadata["backend"] == "hungarian_linking"
    assert result.table["total_tracks"] >= 1
    assert result.tracks["napari_tracks"].shape[0] >= 2


def test_restoration_backend_denoise_preserves_shape() -> None:
    image = ImageVolume(array=np.random.default_rng(0).random((10, 10)), axes=("Y", "X"))
    backend = DEFAULT_REGISTRY.get_restoration("classic")
    result = backend.denoise(image, sigma=1.0)
    assert result.image.shape == image.shape
    assert result.metadata["operation"] == "denoise"


def test_ai_restoration_backend_denoise_preserves_shape() -> None:
    image = ImageVolume(array=np.random.default_rng(0).random((10, 10)), axes=("Y", "X"))
    backend = DEFAULT_REGISTRY.get_restoration("ai_denoise")
    result = backend.denoise(image, method="non_local_means", h=3.0)
    assert result.image.shape == image.shape
    assert result.metadata["operation"] == "denoise"
    assert result.metadata["ai_denoising"] is True
    assert "model_provenance" in result.metadata
    assert "qa_report" in result.metadata


def test_restoration_backend_generates_euclidean_distance_map() -> None:
    data = np.zeros((5, 5), dtype=np.uint8)
    data[2, 2] = 1
    image = ImageVolume(
        array=data,
        axes=("Y", "X"),
        pixel_size={"Y": 2.0, "X": 2.0},
    )
    backend = DEFAULT_REGISTRY.get_restoration("classic")
    result = backend.denoise(image, method="distance_map", threshold=0.5)

    output = result.image.as_numpy()
    assert result.metadata["operation"] == "distance_map"
    assert np.isclose(output[2, 2], 2.0)
    assert np.isclose(output[0, 0], 0.0)


def test_tracing_backend_returns_filament_summary_fields() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[16, 5:27] = 1.0
    volume = ImageVolume(array=image, axes=("Y", "X"))
    backend = DEFAULT_REGISTRY.get_tracing("skeleton")
    result = backend.trace(
        volume,
        gaussian_sigma=0.0,
        threshold_method="manual",
        manual_threshold=0.2,
        min_object_size=1,
    )
    assert result.metadata["backend"] == "skeleton_trace"
    assert "skeleton" in result.graph
    assert "filaments" in result.graph
    assert result.graph["num_filaments"] >= 1
    assert result.table["total_length"] >= 1


def test_registry_finds_backends_by_capability_filters() -> None:
    tracking_names = DEFAULT_REGISTRY.find_backends("tracking", ndim=2, requires_time=True)
    assert "hungarian" in tracking_names

    fluorescence_segmentation = DEFAULT_REGISTRY.find_backends(
        "segmentation",
        ndim=3,
        modality="fluorescence",
    )
    assert "watershed" in fluorescence_segmentation


def test_registry_lists_backend_info_with_capabilities() -> None:
    grouped = DEFAULT_REGISTRY.list_backend_info("restoration")
    assert "restoration" in grouped
    assert grouped["restoration"]
    first = grouped["restoration"][0]
    assert first.capability is not None
    assert first.capability.task == "restoration"
