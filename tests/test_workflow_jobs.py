"""Tests for workflow-step execution and background job runner behavior."""

from __future__ import annotations

import time

import numpy as np
import pytest

from pymaris.data_model import ImageVolume
from pymaris.jobs import JobCancelledError, JobRunner
from pymaris.workflow import WorkflowCancelledError, WorkflowResourceLimitError, WorkflowStep


def test_workflow_step_serialization_roundtrip() -> None:
    original = WorkflowStep(
        id="wf-1",
        name="segmentation-step",
        backend_type="segmentation",
        backend_name="watershed",
        params={"threshold": 10},
        inputs=["image"],
        outputs=["labels"],
    )
    restored = WorkflowStep.from_dict(original.to_dict())
    assert restored == original


def test_workflow_step_segmentation_executes() -> None:
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
    step = WorkflowStep(
        id="wf-2",
        name="watershed",
        backend_type="segmentation",
        backend_name="watershed",
        outputs=["labels"],
    )
    result = step.run({"image": image})
    assert "labels" in result.outputs
    assert result.outputs["labels"].shape == (4, 4)
    assert result.metadata["backend"] == "watershed"


def test_job_runner_reports_progress() -> None:
    image = ImageVolume(array=np.ones((4, 4), dtype=np.uint8), axes=("Y", "X"))
    step = WorkflowStep(
        id="wf-3",
        name="watershed",
        backend_type="segmentation",
        backend_name="watershed",
        outputs=["labels"],
    )
    updates: list[tuple[int, str]] = []
    runner = JobRunner(max_workers=1)
    try:
        handle = runner.submit(step, {"image": image}, on_progress=lambda p, m: updates.append((p, m)))
        result = handle.result(timeout=10)
        assert "labels" in result.outputs
    finally:
        runner.shutdown(wait=True)
    assert updates
    assert updates[-1][0] == 100


def test_workflow_step_tracing_executes() -> None:
    image = np.zeros((16, 16), dtype=np.float32)
    image[8, 3:13] = 1.0
    step = WorkflowStep(
        id="wf-trace",
        name="trace-skeleton",
        backend_type="tracing",
        backend_name="skeleton",
        params={
            "gaussian_sigma": 0.0,
            "threshold_method": "manual",
            "manual_threshold": 0.2,
            "min_object_size": 1,
        },
        outputs=["trace"],
    )
    result = step.run({"image": ImageVolume(array=image, axes=("Y", "X"))})
    trace = result.outputs["trace"]
    assert isinstance(trace, dict)
    assert "skeleton" in trace
    assert trace["num_filaments"] >= 1


def test_workflow_step_restoration_deconvolution_executes() -> None:
    image = ImageVolume(array=np.random.default_rng(0).random((12, 12)), axes=("Y", "X"))
    step = WorkflowStep(
        id="wf-deconv",
        name="deconv",
        backend_type="restoration",
        backend_name="classic",
        params={"operation": "deconvolve", "method": "richardson_lucy", "iterations": 2},
        outputs=["restored_image"],
    )
    result = step.run({"image": image})
    restored = result.outputs["restored_image"]
    assert isinstance(restored, ImageVolume)
    assert restored.shape == image.shape


def test_job_runner_cancellation_callable() -> None:
    runner = JobRunner(max_workers=1)

    def slow_task(progress, cancel_event):
        for idx in range(100):
            if cancel_event.is_set():
                raise WorkflowCancelledError("cancelled from test")
            progress(idx, "running")
            time.sleep(0.01)
        return "done"

    try:
        handle = runner.submit_callable(slow_task)
        time.sleep(0.05)
        handle.cancel()
        try:
            handle.result(timeout=5)
            raise AssertionError("expected cancellation")
        except JobCancelledError:
            pass
    finally:
        runner.shutdown(wait=True)


def test_workflow_step_honors_memory_budget_limit() -> None:
    image = ImageVolume(array=np.ones((64, 64), dtype=np.float32), axes=("Y", "X"))
    step = WorkflowStep(
        id="wf-budget",
        name="watershed-budget",
        backend_type="segmentation",
        backend_name="watershed",
        outputs=["labels"],
    )
    with pytest.raises(WorkflowResourceLimitError, match="memory budget exceeded"):
        step.run({"image": image}, resource_limits={"memory_budget_mb": 0.001})


def test_workflow_step_restoration_tiling_allows_low_memory_budget() -> None:
    image = ImageVolume(array=np.random.default_rng(4).random((128, 128), dtype=np.float32), axes=("Y", "X"))
    step = WorkflowStep(
        id="wf-rest-tiled",
        name="denoise-tiled",
        backend_type="restoration",
        backend_name="classic",
        params={
            "operation": "denoise",
            "method": "gaussian",
            "sigma": 1.0,
            "tile_shape": [32, 32],
            "tile_overlap": 6,
        },
        outputs=["restored_image"],
    )
    result = step.run({"image": image}, resource_limits={"memory_budget_mb": 0.01})
    restored = result.outputs["restored_image"]
    assert isinstance(restored, ImageVolume)
    assert restored.shape == image.shape
    tiled = result.metadata.get("tiled_execution", {})
    assert tiled.get("enabled") is True
    assert tiled.get("tile_count", 0) > 1


def test_workflow_step_restoration_supports_lazy_dask_output() -> None:
    da = pytest.importorskip("dask.array")
    data = da.from_array(np.random.default_rng(5).random((64, 64), dtype=np.float32), chunks=(16, 16))
    image = ImageVolume(array=data, axes=("Y", "X"))
    step = WorkflowStep(
        id="wf-rest-lazy",
        name="denoise-lazy",
        backend_type="restoration",
        backend_name="classic",
        params={"operation": "denoise", "method": "gaussian", "sigma": 1.0},
        outputs=["restored_image"],
    )
    result = step.run({"image": image})
    restored = result.outputs["restored_image"]
    assert isinstance(restored, ImageVolume)
    assert isinstance(restored.array, da.Array)
