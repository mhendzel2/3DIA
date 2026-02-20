"""Workflow payload validation helpers for CLI and tests."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ALLOWED_BACKEND_TYPES = {"segmentation", "tracking", "tracing", "restoration"}
ALLOWED_STEP_KEYS = {"id", "name", "backend_type", "backend_name", "params", "inputs", "outputs"}
REQUIRED_STEP_KEYS = {"id", "name", "backend_type", "backend_name"}
WORKFLOW_DOCUMENT_KEYS = {"workflow_version", "metadata", "steps"}
ALLOWED_EXECUTION_METADATA_KEYS = {"prefer_lazy", "chunks", "memory_budget_mb"}
DEFAULT_WORKFLOW_VERSION = "1.0"
SUPPORTED_WORKFLOW_MAJOR_VERSION = 1
EMBEDDED_WORKFLOW_SCHEMA: dict[str, Any] = {
    "properties": {
        "workflow_version": {"type": "string"},
        "metadata": {"type": "object"},
    },
    "$defs": {
        "step": {
            "required": ["id", "name", "backend_type", "backend_name"],
            "properties": {
                "backend_type": {"enum": sorted(ALLOWED_BACKEND_TYPES)},
            },
        }
    }
}


def workflow_schema_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "schemas" / "workflow.schema.json"


def load_workflow_schema() -> dict[str, Any]:
    path = workflow_schema_path()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    return EMBEDDED_WORKFLOW_SCHEMA


def normalize_workflow_steps_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        raw_steps = payload
    elif isinstance(payload, dict) and isinstance(payload.get("steps"), list):
        raw_steps = payload["steps"]
    else:
        raise ValueError("workflow JSON must be a list or an object containing a 'steps' list")
    return [dict(step) for step in raw_steps]


def normalize_workflow_document_payload(payload: Any) -> dict[str, Any]:
    """Normalize workflow payload to a versioned document envelope."""
    if isinstance(payload, list):
        return {
            "workflow_version": DEFAULT_WORKFLOW_VERSION,
            "metadata": {},
            "steps": [dict(step) for step in payload],
        }
    if isinstance(payload, dict):
        if not isinstance(payload.get("steps"), list):
            raise ValueError("workflow JSON object must contain a 'steps' list")
        unknown = set(payload.keys()).difference(WORKFLOW_DOCUMENT_KEYS)
        if unknown:
            unknown_keys = ", ".join(sorted(unknown))
            raise ValueError(f"workflow document contains unsupported keys: {unknown_keys}")
        workflow_version = str(payload.get("workflow_version", DEFAULT_WORKFLOW_VERSION))
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("workflow 'metadata' must be an object")
        return {
            "workflow_version": workflow_version,
            "metadata": dict(metadata),
            "steps": [dict(step) for step in payload["steps"]],
        }
    raise ValueError("workflow JSON must be a list or an object containing a 'steps' list")


def validate_workflow_document(payload: Any) -> dict[str, Any]:
    """Validate workflow document and return normalized document payload."""
    document = normalize_workflow_document_payload(payload)
    workflow_version = str(document["workflow_version"]).strip()
    if not workflow_version:
        raise ValueError("workflow_version must be a non-empty string")
    _validate_workflow_version(workflow_version)

    metadata = _validate_metadata(document["metadata"])
    steps = list(document["steps"])
    if not steps:
        raise ValueError("workflow 'steps' list must contain at least one step")

    seen_step_ids: set[str] = set()
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"step {index} must be an object")
        missing = REQUIRED_STEP_KEYS.difference(step.keys())
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(f"step {index} missing required keys: {missing_keys}")
        unknown = set(step.keys()).difference(ALLOWED_STEP_KEYS)
        if unknown:
            unknown_keys = ", ".join(sorted(unknown))
            raise ValueError(f"step {index} contains unsupported keys: {unknown_keys}")

        for key in ("id", "name", "backend_name"):
            if not isinstance(step.get(key), str) or not step[key].strip():
                raise ValueError(f"step {index} key '{key}' must be a non-empty string")
        step_id = str(step["id"])
        if step_id in seen_step_ids:
            raise ValueError(f"step {index} duplicates id '{step_id}'")
        seen_step_ids.add(step_id)

        backend_type = step.get("backend_type")
        if backend_type not in ALLOWED_BACKEND_TYPES:
            allowed = ", ".join(sorted(ALLOWED_BACKEND_TYPES))
            raise ValueError(f"step {index} backend_type must be one of: {allowed}")

        params = step.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"step {index} key 'params' must be an object")

        inputs = step.get("inputs", [])
        outputs = step.get("outputs", [])
        if not isinstance(inputs, list) or not all(isinstance(item, str) for item in inputs):
            raise ValueError(f"step {index} key 'inputs' must be an array of strings")
        if outputs and (not isinstance(outputs, list) or not all(isinstance(item, str) for item in outputs)):
            raise ValueError(f"step {index} key 'outputs' must be an array of strings")

    return {
        "workflow_version": workflow_version,
        "metadata": metadata,
        "steps": steps,
    }


def validate_workflow_payload(payload: Any) -> list[dict[str, Any]]:
    """Backward-compatible helper returning only validated workflow steps."""
    return list(validate_workflow_document(payload)["steps"])


def _validate_workflow_version(version: str) -> None:
    match = re.match(r"^\s*(\d+)(?:\.\d+){0,2}\s*$", version)
    if not match:
        raise ValueError(
            "workflow_version must look like 'MAJOR.MINOR[.PATCH]' (example: "
            f"'{DEFAULT_WORKFLOW_VERSION}')"
        )
    major_version = int(match.group(1))
    if major_version != SUPPORTED_WORKFLOW_MAJOR_VERSION:
        raise ValueError(
            f"unsupported workflow_version major '{major_version}'; "
            f"supported major is {SUPPORTED_WORKFLOW_MAJOR_VERSION}"
        )


def _validate_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("workflow 'metadata' must be an object")
    normalized = dict(metadata)
    execution_raw = normalized.get("execution")
    if execution_raw is None:
        return normalized
    if not isinstance(execution_raw, dict):
        raise ValueError("workflow metadata 'execution' must be an object")

    unknown_execution = set(execution_raw.keys()).difference(ALLOWED_EXECUTION_METADATA_KEYS)
    if unknown_execution:
        unknown_keys = ", ".join(sorted(unknown_execution))
        allowed = ", ".join(sorted(ALLOWED_EXECUTION_METADATA_KEYS))
        raise ValueError(
            f"workflow metadata 'execution' contains unsupported keys: {unknown_keys}; "
            f"allowed keys: {allowed}"
        )

    execution: dict[str, Any] = {}
    if "prefer_lazy" in execution_raw:
        if not isinstance(execution_raw["prefer_lazy"], bool):
            raise ValueError("workflow metadata 'execution.prefer_lazy' must be a boolean")
        execution["prefer_lazy"] = bool(execution_raw["prefer_lazy"])

    if "chunks" in execution_raw:
        chunks = execution_raw["chunks"]
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("workflow metadata 'execution.chunks' must be a non-empty array of integers")
        normalized_chunks: list[int] = []
        for value in chunks:
            if not isinstance(value, int) or value <= 0:
                raise ValueError("workflow metadata 'execution.chunks' values must be integers > 0")
            normalized_chunks.append(int(value))
        execution["chunks"] = normalized_chunks

    if "memory_budget_mb" in execution_raw:
        raw_budget = execution_raw["memory_budget_mb"]
        if not isinstance(raw_budget, (int, float)):
            raise ValueError("workflow metadata 'execution.memory_budget_mb' must be a number")
        budget = float(raw_budget)
        if budget <= 0:
            raise ValueError("workflow metadata 'execution.memory_budget_mb' must be > 0")
        execution["memory_budget_mb"] = budget

    normalized["execution"] = execution
    return normalized
