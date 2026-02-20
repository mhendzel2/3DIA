"""Tests for workflow payload validation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pymaris.workflow_validation import (
    load_workflow_schema,
    validate_workflow_document,
    validate_workflow_payload,
    workflow_schema_path,
)


def test_workflow_schema_loader_reads_schema() -> None:
    assert workflow_schema_path().is_file()
    schema = load_workflow_schema()
    assert "$defs" in schema


def test_validate_workflow_payload_accepts_example() -> None:
    payload = json.loads(
        Path("examples/workflows/segmentation_watershed.json").read_text(encoding="utf-8")
    )
    steps = validate_workflow_payload(payload)
    assert len(steps) == 1
    assert steps[0]["backend_type"] == "segmentation"


def test_validate_workflow_payload_rejects_invalid_backend() -> None:
    payload = {
        "steps": [
            {
                "id": "bad-1",
                "name": "invalid",
                "backend_type": "unknown",
                "backend_name": "x",
            }
        ]
    }
    with pytest.raises(ValueError, match="backend_type must be one of"):
        validate_workflow_payload(payload)


def test_validate_workflow_document_accepts_versioned_object() -> None:
    payload = {
        "workflow_version": "1.2",
        "metadata": {"author": "test"},
        "steps": [
            {
                "id": "s1",
                "name": "segment",
                "backend_type": "segmentation",
                "backend_name": "watershed",
            }
        ],
    }
    document = validate_workflow_document(payload)
    assert document["workflow_version"] == "1.2"
    assert document["metadata"]["author"] == "test"
    assert len(document["steps"]) == 1


def test_validate_workflow_document_rejects_unsupported_major_version() -> None:
    payload = {
        "workflow_version": "2.0",
        "steps": [
            {
                "id": "s1",
                "name": "segment",
                "backend_type": "segmentation",
                "backend_name": "watershed",
            }
        ],
    }
    with pytest.raises(ValueError, match="unsupported workflow_version major"):
        validate_workflow_document(payload)


def test_validate_workflow_document_rejects_duplicate_step_ids() -> None:
    payload = {
        "steps": [
            {
                "id": "dup",
                "name": "first",
                "backend_type": "segmentation",
                "backend_name": "watershed",
            },
            {
                "id": "dup",
                "name": "second",
                "backend_type": "restoration",
                "backend_name": "classic",
            },
        ]
    }
    with pytest.raises(ValueError, match="duplicates id"):
        validate_workflow_document(payload)


def test_validate_workflow_document_accepts_execution_metadata() -> None:
    payload = {
        "workflow_version": "1.0",
        "metadata": {"execution": {"prefer_lazy": True, "chunks": [1, 64, 64], "memory_budget_mb": 32.0}},
        "steps": [
            {
                "id": "s1",
                "name": "segment",
                "backend_type": "segmentation",
                "backend_name": "watershed",
            }
        ],
    }
    document = validate_workflow_document(payload)
    execution = document["metadata"]["execution"]
    assert execution["prefer_lazy"] is True
    assert execution["chunks"] == [1, 64, 64]
    assert execution["memory_budget_mb"] == 32.0


def test_validate_workflow_document_rejects_invalid_execution_memory_budget() -> None:
    payload = {
        "metadata": {"execution": {"memory_budget_mb": 0}},
        "steps": [
            {
                "id": "s1",
                "name": "segment",
                "backend_type": "segmentation",
                "backend_name": "watershed",
            }
        ],
    }
    with pytest.raises(ValueError, match="memory_budget_mb"):
        validate_workflow_document(payload)
