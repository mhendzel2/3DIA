"""Shared workflow execution helpers for CLI and UI surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from pymaris.backends import DEFAULT_REGISTRY, BackendRegistry
from pymaris.jobs import JobCancelledError, JobRunner
from pymaris.workflow import WorkflowResult, WorkflowStep

WorkflowProgressCallback = Callable[[WorkflowStep, int, str], None]


@dataclass(frozen=True)
class ExecutedWorkflowStep:
    """Step/result pair returned by document execution."""

    step: WorkflowStep
    result: WorkflowResult


def execute_workflow_steps(
    *,
    steps: Sequence[WorkflowStep],
    context: dict[str, Any],
    registry: BackendRegistry | None = None,
    runner: JobRunner | None = None,
    on_progress: WorkflowProgressCallback | None = None,
    resource_limits: Mapping[str, Any] | None = None,
) -> list[ExecutedWorkflowStep]:
    """Execute a sequence of workflow steps with shared context mutation.

    Results from each step are injected into ``context`` so later steps can bind
    previous outputs by key.
    """
    target_registry = registry or DEFAULT_REGISTRY
    owns_runner = runner is None
    target_runner = runner or JobRunner(max_workers=2)
    executed: list[ExecutedWorkflowStep] = []

    try:
        for step in steps:
            handle = target_runner.submit(
                step=step,
                context=context,
                registry=target_registry,
                on_progress=(
                    (lambda percent, message, current=step: on_progress(current, percent, message))
                    if on_progress is not None
                    else None
                ),
                resource_limits=dict(resource_limits or {}),
            )
            try:
                result = handle.result()
            except JobCancelledError:
                raise RuntimeError(f"workflow step cancelled: {step.id}") from None

            for output_name, value in result.outputs.items():
                context[output_name] = value

            executed.append(ExecutedWorkflowStep(step=step, result=result))
    finally:
        if owns_runner:
            target_runner.shutdown(wait=True)

    return executed
