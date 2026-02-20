"""Tests for the shared workflow execution API used by CLI/UI surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite

from pymaris.cli import main
from pymaris.data_model import ImageVolume
from pymaris.workflow import WorkflowStep
from pymaris.workflow_runner import execute_workflow_steps


def test_execute_workflow_steps_updates_shared_context() -> None:
    image = ImageVolume(array=np.random.default_rng(0).random((32, 32)), axes=("Y", "X"))
    context: dict[str, object] = {"image": image}
    steps = [
        WorkflowStep(
            id="wf-1",
            name="denoise",
            backend_type="restoration",
            backend_name="classic",
            params={"operation": "denoise", "method": "gaussian", "sigma": 1.0},
            inputs=["image"],
            outputs=["denoised"],
        )
    ]

    executed = execute_workflow_steps(steps=steps, context=context)

    assert len(executed) == 1
    assert "denoised" in context
    output = context["denoised"]
    assert isinstance(output, ImageVolume)
    assert output.shape == image.shape


def test_cli_and_api_workflow_execution_are_equivalent(tmp_path: Path) -> None:
    image_array = np.random.default_rng(1).random((24, 24)).astype(np.float32)
    image = ImageVolume(array=image_array, axes=("Y", "X"))
    workflow_doc = {
        "workflow_version": "1.0",
        "steps": [
            {
                "id": "wf-1",
                "name": "denoise",
                "backend_type": "restoration",
                "backend_name": "classic",
                "params": {"operation": "denoise", "method": "gaussian", "sigma": 1.0},
                "inputs": ["image"],
                "outputs": ["denoised"],
            }
        ],
    }

    context: dict[str, object] = {"image": image}
    steps = [WorkflowStep.from_dict(step) for step in workflow_doc["steps"]]
    execute_workflow_steps(steps=steps, context=context)
    api_output = np.asarray(context["denoised"].as_numpy())  # type: ignore[index]

    image_path = tmp_path / "input.tif"
    workflow_path = tmp_path / "workflow.json"
    project_dir = tmp_path / "project"

    imwrite(image_path, image_array)
    workflow_path.write_text(json.dumps(workflow_doc), encoding="utf-8")

    exit_code = main(
        [
            "run-workflow",
            "--input",
            str(image_path),
            "--workflow",
            str(workflow_path),
            "--project",
            str(project_dir),
        ]
    )
    cli_output = np.asarray(imread(project_dir / "outputs" / "images" / "denoised.tif"))

    assert exit_code == 0
    assert api_output.shape == cli_output.shape
    assert np.allclose(api_output, cli_output, atol=1e-6)
