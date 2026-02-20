"""Tests for project storage and provenance persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pymaris.data_model import ImageVolume
from pymaris.project_store import ProjectStore


def test_project_store_roundtrip_image_and_labels(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "project")
    store.initialize()

    image = ImageVolume(
        array=np.arange(64, dtype=np.uint16).reshape(4, 4, 4),
        axes=("Z", "Y", "X"),
        metadata={"name": "image_a"},
    )
    labels = (image.as_numpy() > 10).astype(np.uint16)

    store.save_image_layer(name="image_a", image=image)
    store.save_label_layer(name="labels_a", labels=labels)
    store.record_workflow_step(
        name="watershed",
        params={"threshold": 10},
        inputs=["image_a"],
        outputs=["labels_a"],
        workflow_version="1.0",
        backend={
            "backend_type": "segmentation",
            "registered_name": "watershed",
            "version": "test-version",
        },
        result_metadata={"backend": "watershed"},
    )

    images = store.load_image_layers()
    loaded_labels = store.load_label_layers()
    provenance = store.load_provenance()

    assert "image_a" in images
    assert "labels_a" in loaded_labels
    np.testing.assert_array_equal(images["image_a"].as_numpy(), image.as_numpy())
    np.testing.assert_array_equal(loaded_labels["labels_a"], labels)

    assert provenance["schema_version"] == "0.2.0"
    assert len(provenance["outputs"]) >= 2
    assert len(provenance["workflow_steps"]) == 1
    assert provenance["workflow_steps"][0]["id"] == "step-0001"
    assert provenance["workflow_steps"][0]["workflow_version"] == "1.0"
    assert provenance["workflow_steps"][0]["backend"]["registered_name"] == "watershed"


def test_project_store_records_input_hash(tmp_path: Path) -> None:
    source_file = tmp_path / "input.tif"
    source_file.write_bytes(b"synthetic-input")
    store = ProjectStore(tmp_path / "project")
    record = store.record_input(source_file)
    assert record["path"].endswith("input.tif")
    assert len(record["sha256_head"]) == 64


def test_project_store_compare_environment_has_expected_shape(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "project")
    store.initialize()

    comparison = store.compare_environment()
    assert "matches" in comparison
    assert "python_version" in comparison
    assert "platform" in comparison
    assert "package_mismatches" in comparison
