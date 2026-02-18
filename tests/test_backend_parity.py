"""Reference parity tests across backend modalities and dimensionalities."""

from __future__ import annotations

import numpy as np

from pymaris.backends import DEFAULT_REGISTRY
from pymaris.data_model import ImageVolume


def test_all_default_backends_declare_capability_with_matching_task() -> None:
    grouped = DEFAULT_REGISTRY.list_backend_info()
    for backend_type, infos in grouped.items():
        assert infos, f"expected at least one backend for {backend_type}"
        for info in infos:
            assert info.capability is not None, f"missing capability declaration for {backend_type}:{info.name}"
            assert info.capability.task == backend_type


def test_watershed_segmentation_parity_for_2d_and_3d() -> None:
    backend = DEFAULT_REGISTRY.get_segmentation("watershed")

    image_2d = ImageVolume(array=np.zeros((32, 32), dtype=np.float32), axes=("Y", "X"))
    image_2d.array[8:24, 8:24] = 1.0
    result_2d = backend.segment_instances(image_2d, threshold=0.5)

    image_3d = ImageVolume(array=np.zeros((8, 32, 32), dtype=np.float32), axes=("Z", "Y", "X"))
    image_3d.array[2:6, 8:24, 8:24] = 1.0
    result_3d = backend.segment_instances(image_3d, threshold=0.5)

    assert result_2d.labels.shape == image_2d.shape
    assert result_3d.labels.shape == image_3d.shape
    assert result_2d.metadata["backend"] == "watershed"
    assert result_3d.metadata["backend"] == "watershed"


def test_classic_restoration_parity_for_2d_and_3d() -> None:
    backend = DEFAULT_REGISTRY.get_restoration("classic")

    image_2d = ImageVolume(array=np.random.default_rng(0).random((32, 32)), axes=("Y", "X"))
    image_3d = ImageVolume(array=np.random.default_rng(1).random((8, 32, 32)), axes=("Z", "Y", "X"))

    result_2d = backend.denoise(image_2d, method="gaussian", sigma=1.0)
    result_3d = backend.denoise(image_3d, method="gaussian", sigma=1.0)

    assert result_2d.image.shape == image_2d.shape
    assert result_3d.image.shape == image_3d.shape
    assert result_2d.metadata["operation"] == "denoise"
    assert result_3d.metadata["operation"] == "denoise"


def test_hungarian_tracking_parity_for_2d_and_3d_sequences() -> None:
    backend = DEFAULT_REGISTRY.get_tracking("hungarian")

    seq_2d: list[np.ndarray] = []
    for t in range(4):
        frame = np.zeros((24, 24), dtype=np.int32)
        frame[6 + t : 10 + t, 8 + t : 12 + t] = 1
        seq_2d.append(frame)

    seq_3d: list[np.ndarray] = []
    for t in range(4):
        frame = np.zeros((6, 24, 24), dtype=np.int32)
        frame[2:4, 6 + t : 10 + t, 8 + t : 12 + t] = 1
        seq_3d.append(frame)

    result_2d = backend.track(seq_2d, max_distance=6.0)
    result_3d = backend.track(seq_3d, max_distance=6.0)

    assert result_2d.table["total_tracks"] >= 1
    assert result_3d.table["total_tracks"] >= 1
    assert result_2d.tracks["napari_tracks"].shape[1] >= 4
    assert result_3d.tracks["napari_tracks"].shape[1] >= 5


def test_skeleton_tracing_parity_for_2d_and_3d() -> None:
    backend = DEFAULT_REGISTRY.get_tracing("skeleton")

    image_2d = np.zeros((48, 48), dtype=np.float32)
    image_2d[24, 8:40] = 1.0
    result_2d = backend.trace(
        ImageVolume(array=image_2d, axes=("Y", "X")),
        gaussian_sigma=0.0,
        threshold_method="manual",
        manual_threshold=0.2,
        min_object_size=1,
    )

    image_3d = np.zeros((8, 48, 48), dtype=np.float32)
    image_3d[4, 24, 8:40] = 1.0
    result_3d = backend.trace(
        ImageVolume(array=image_3d, axes=("Z", "Y", "X")),
        gaussian_sigma=0.0,
        threshold_method="manual",
        manual_threshold=0.2,
        min_object_size=1,
    )

    assert result_2d.graph["node_count"] > 0
    assert result_3d.graph["node_count"] > 0
    assert result_2d.table["total_length"] > 0
    assert result_3d.table["total_length"] > 0
