"""Validate checked-in workflow schema and example JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pymaris.workflow_validation import load_workflow_schema, validate_workflow_payload


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_workflow_schema_exists_and_has_step_definition() -> None:
    schema = load_workflow_schema()
    assert "$defs" in schema
    assert "step" in schema["$defs"]
    required = set(schema["$defs"]["step"]["required"])
    assert {"id", "name", "backend_type", "backend_name"}.issubset(required)


def test_workflow_examples_match_expected_shape() -> None:
    examples_dir = Path("examples/workflows")
    assert examples_dir.is_dir()
    files = sorted(examples_dir.glob("*.json"))
    assert files, "No workflow examples found"

    valid_backend_types = {"segmentation", "tracking", "tracing", "restoration"}
    for path in files:
        payload = _load_json(path)
        validated_steps = validate_workflow_payload(payload)
        assert validated_steps
        if isinstance(payload, dict):
            steps = payload.get("steps")
        else:
            steps = payload
        assert isinstance(steps, list), f"{path} does not contain a steps list"
        assert steps, f"{path} contains no steps"
        for step in steps:
            assert isinstance(step, dict), f"{path} step is not an object"
            assert step.get("id"), f"{path} missing step id"
            assert step.get("name"), f"{path} missing step name"
            assert step.get("backend_name"), f"{path} missing backend_name"
            assert step.get("backend_type") in valid_backend_types, (
                f"{path} has unsupported backend_type {step.get('backend_type')!r}"
            )
