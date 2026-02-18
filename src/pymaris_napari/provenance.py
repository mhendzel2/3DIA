"""Helpers to persist napari UI workflow results into ProjectStore."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pymaris.data_model import ImageVolume
from pymaris.project_store import ProjectStore
from pymaris.workflow import WorkflowResult, WorkflowStep


def record_ui_workflow_result(
    *,
    project_dir: str | Path,
    step: WorkflowStep,
    result: WorkflowResult,
    source_paths: Sequence[str] | None = None,
    output_format: str = "tiff",
) -> ProjectStore:
    """Persist a workflow result from napari UI interactions."""
    store = ProjectStore(project_dir)
    store.initialize()

    for source in source_paths or []:
        try:
            store.record_input(source)
        except Exception:
            continue

    for output_name, value in result.outputs.items():
        if isinstance(value, ImageVolume):
            store.save_image_layer(output_name, image=value, format=output_format)
        elif isinstance(value, np.ndarray):
            store.save_label_layer(output_name, labels=value)
        elif isinstance(value, dict) and "napari_tracks" in value:
            store.save_tracks(output_name, tracks_payload=value)
        elif isinstance(value, dict):
            store.save_graph(output_name, graph=value)

    for table_name, table_payload in result.tables.items():
        if isinstance(table_payload, dict):
            rows = table_dict_to_rows(table_payload)
            if rows:
                store.save_table(table_name, rows)

    store.record_workflow_step(
        name=step.name,
        params=step.params,
        inputs=step.inputs,
        outputs=list(result.outputs.keys()),
    )
    return store


def table_dict_to_rows(table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert summary/table dictionaries into row records."""
    list_columns = {key: value for key, value in table.items() if isinstance(value, list)}
    if not list_columns:
        return [dict(table)]
    row_count = max(len(column) for column in list_columns.values())
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        row: dict[str, Any] = {}
        for key, value in table.items():
            if isinstance(value, list):
                row[key] = value[index] if index < len(value) else None
            else:
                row[key] = value
        rows.append(row)
    return rows
