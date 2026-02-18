"""Tests for napari-side provenance persistence helper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pymaris.workflow import WorkflowResult, WorkflowStep
from pymaris_napari.provenance import record_ui_workflow_result


def test_record_ui_workflow_result_persists_outputs(tmp_path: Path) -> None:
    project_dir = tmp_path / "ui_project"
    step = WorkflowStep(
        id="ui-step-0001",
        name="segmentation:watershed",
        backend_type="segmentation",
        backend_name="watershed",
        params={"threshold": 10},
        inputs=["image"],
        outputs=["labels"],
    )
    labels = np.zeros((8, 8), dtype=np.uint16)
    labels[2:5, 2:5] = 1
    result = WorkflowResult(
        outputs={"labels": labels},
        tables={"labels_table": {"object_count": 1, "area": [9]}},
        metadata={"backend": "watershed"},
    )

    store = record_ui_workflow_result(project_dir=project_dir, step=step, result=result)
    provenance = store.load_provenance()

    assert (project_dir / "outputs" / "labels" / "labels.tif").is_file()
    assert (project_dir / "outputs" / "tables" / "labels_table.csv").is_file()
    assert len(provenance["workflow_steps"]) == 1
    assert provenance["workflow_steps"][0]["name"] == "segmentation:watershed"
