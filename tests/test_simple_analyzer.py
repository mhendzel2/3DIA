"""Tests for legacy simple analyzer fallback behavior."""

from __future__ import annotations

from simple_analyzer import SimpleImageAnalyzer


def test_measure_objects_computes_mean_intensity_from_image_data() -> None:
    labels = [
        [0, 1, 1],
        [0, 1, 0],
    ]
    image = [
        [0, 10, 20],
        [0, 30, 0],
    ]
    measurements = SimpleImageAnalyzer.measure_objects(labels, image_data=image)
    assert len(measurements) == 1
    assert measurements[0]["label"] == 1
    assert measurements[0]["area"] == 3
    assert measurements[0]["mean_intensity"] == 20.0
