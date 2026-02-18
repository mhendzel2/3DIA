"""Tests for headless CLI workflow/project commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tifffile import imread, imwrite

import pymaris.io as pymaris_io
from pymaris.cli import main


def test_cli_open_json(tmp_path: Path, capsys) -> None:
    image_path = tmp_path / "input.tif"
    imwrite(image_path, np.arange(16, dtype=np.uint16).reshape(4, 4))

    exit_code = main(["open", str(image_path), "--json"])
    captured = capsys.readouterr().out
    payload = json.loads(captured)

    assert exit_code == 0
    assert payload["path"].endswith("input.tif")
    assert payload["shape"] == [4, 4]


def test_cli_run_workflow_and_export(tmp_path: Path) -> None:
    image_path = tmp_path / "input.tif"
    workflow_path = tmp_path / "workflow.json"
    project_dir = tmp_path / "project"
    export_dir = tmp_path / "export"

    imwrite(image_path, np.arange(64, dtype=np.uint16).reshape(4, 4, 4))
    workflow = {
        "workflow_version": "1.0",
        "metadata": {"purpose": "unit-test"},
        "steps": [
            {
                "id": "wf-1",
                "name": "segment",
                "backend_type": "segmentation",
                "backend_name": "watershed",
                "inputs": ["image"],
                "outputs": ["labels"],
                "params": {},
            }
        ]
    }
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")

    run_exit = main(
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
    export_exit = main(
        [
            "export",
            "--project",
            str(project_dir),
            "--destination",
            str(export_dir),
        ]
    )

    assert run_exit == 0
    assert export_exit == 0
    assert (project_dir / "outputs" / "labels" / "labels.tif").is_file()
    assert (project_dir / "metadata" / "provenance.json").is_file()
    assert (export_dir / "outputs" / "labels" / "labels.tif").is_file()

    provenance = json.loads((project_dir / "metadata" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["workflow_steps"][0]["workflow_version"] == "1.0"
    assert provenance["workflow_steps"][0]["backend"]["registered_name"] == "watershed"


def test_cli_run_project(tmp_path: Path) -> None:
    image_path = tmp_path / "input.tif"
    workflow_path = tmp_path / "workflow.json"
    project_dir = tmp_path / "project"

    imwrite(image_path, np.arange(16, dtype=np.uint16).reshape(4, 4))
    workflow = {
        "steps": [
            {
                "id": "wf-restore",
                "name": "denoise",
                "backend_type": "restoration",
                "backend_name": "classic",
                "inputs": ["image"],
                "outputs": ["denoised"],
                "params": {"operation": "denoise", "sigma": 0.5},
            }
        ]
    }
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")

    first_run = main(
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
    second_run = main(
        [
            "run-project",
            "--project",
            str(project_dir),
            "--workflow",
            str(workflow_path),
            "--image-layer",
            "input",
        ]
    )
    assert first_run == 0
    assert second_run == 0
    assert (project_dir / "outputs" / "images" / "denoised.tif").is_file()


def test_cli_run_workflow_distance_map(tmp_path: Path) -> None:
    image_path = tmp_path / "input.tif"
    workflow_path = tmp_path / "workflow_distance.json"
    project_dir = tmp_path / "project"

    image = np.zeros((7, 7), dtype=np.uint16)
    image[3, 3] = 10
    imwrite(image_path, image)
    workflow = {
        "workflow_version": "1.0",
        "steps": [
            {
                "id": "wf-distance",
                "name": "distance-map",
                "backend_type": "restoration",
                "backend_name": "classic",
                "inputs": ["image"],
                "outputs": ["distance_map"],
                "params": {
                    "operation": "distance_map",
                    "threshold": 0.5,
                },
            }
        ],
    }
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")

    run_exit = main(
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

    output_path = project_dir / "outputs" / "images" / "distance_map.tif"
    assert run_exit == 0
    assert output_path.is_file()
    assert float(np.max(imread(output_path))) > 0.0


def test_cli_list_reconstruction_plugins(capsys) -> None:
    exit_code = main(["list-reconstruction-plugins", "--json"])
    payload = json.loads(capsys.readouterr().out)

    names = {row["name"] for row in payload["plugins"]}
    assert exit_code == 0
    assert "deconvolution" in names
    assert "sim" in names


def test_cli_run_reconstruction_deconvolution(tmp_path: Path) -> None:
    image_path = tmp_path / "input.tif"
    project_dir = tmp_path / "project"
    params_path = tmp_path / "params.json"

    image = np.zeros((9, 9), dtype=np.uint16)
    image[4, 4] = 100
    imwrite(image_path, image)
    params_path.write_text(
        json.dumps(
            {
                "operation": "distance_map",
                "method": "distance_map",
                "threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run-reconstruction",
            "--input",
            str(image_path),
            "--plugin",
            "deconvolution",
            "--project",
            str(project_dir),
            "--output-name",
            "recon",
            "--params-file",
            str(params_path),
        ]
    )

    output_path = project_dir / "outputs" / "images" / "recon.tif"
    qc_table_path = project_dir / "outputs" / "tables" / "recon_qc.csv"
    qa_report_path = project_dir / "outputs" / "meshes" / "qa_report.json"
    model_provenance_path = project_dir / "outputs" / "meshes" / "model_provenance.json"
    assert exit_code == 0
    assert output_path.is_file()
    assert qc_table_path.is_file()
    assert qa_report_path.is_file()
    assert model_provenance_path.is_file()
    assert float(np.max(imread(output_path))) > 0.0


def test_cli_run_reconstruction_sim_requires_calibration(tmp_path: Path, capsys) -> None:
    image_path = tmp_path / "input.tif"
    project_dir = tmp_path / "project"
    imwrite(image_path, np.ones((4, 4), dtype=np.uint16))

    exit_code = main(
        [
            "run-reconstruction",
            "--input",
            str(image_path),
            "--plugin",
            "sim",
            "--project",
            str(project_dir),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "requires calibration objects" in captured.err


def test_cli_rejects_invalid_workflow_schema(tmp_path: Path, capsys) -> None:
    image_path = tmp_path / "input.tif"
    workflow_path = tmp_path / "invalid_workflow.json"
    project_dir = tmp_path / "project"

    imwrite(image_path, np.arange(16, dtype=np.uint16).reshape(4, 4))
    invalid_workflow = {"steps": [{"id": "bad-step", "backend_name": "watershed"}]}
    workflow_path.write_text(json.dumps(invalid_workflow), encoding="utf-8")

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
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "missing required keys" in captured.err


def test_cli_debug_flag_raises_invalid_workflow_error(tmp_path: Path) -> None:
    image_path = tmp_path / "input.tif"
    workflow_path = tmp_path / "invalid_workflow.json"
    project_dir = tmp_path / "project"

    imwrite(image_path, np.arange(16, dtype=np.uint16).reshape(4, 4))
    workflow_path.write_text(json.dumps({"steps": [{"id": "bad"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        main(
            [
                "--debug",
                "run-workflow",
                "--input",
                str(image_path),
                "--workflow",
                str(workflow_path),
                "--project",
                str(project_dir),
            ]
        )


def test_cli_list_backends_filters_tracking_time(capsys) -> None:
    exit_code = main(
        [
            "list-backends",
            "--backend-type",
            "tracking",
            "--requires-time",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["count"] >= 1
    assert "tracking" in payload["backends"]
    assert payload["backends"]["tracking"][0]["name"] == "hungarian"


def test_cli_run_workflow_rejects_unsupported_workflow_version(tmp_path: Path, capsys) -> None:
    image_path = tmp_path / "input.tif"
    workflow_path = tmp_path / "workflow_bad_version.json"
    project_dir = tmp_path / "project"

    imwrite(image_path, np.arange(16, dtype=np.uint16).reshape(4, 4))
    workflow_path.write_text(
        json.dumps(
            {
                "workflow_version": "2.0",
                "steps": [
                    {
                        "id": "wf-1",
                        "name": "segment",
                        "backend_type": "segmentation",
                        "backend_name": "watershed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

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
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "unsupported workflow_version major" in captured.err


def test_cli_open_with_scene_index_and_list_scenes(tmp_path: Path, capsys, monkeypatch) -> None:
    h5py = pytest.importorskip("h5py")
    monkeypatch.setattr(pymaris_io, "HAS_AICSIMAGEIO", False)
    ims_path = tmp_path / "synthetic_cli.ims"
    with h5py.File(ims_path, "w") as handle:
        dataset = handle.create_group("DataSet")
        res0 = dataset.create_group("ResolutionLevel 0")
        tp0 = res0.create_group("TimePoint 0")
        tp1 = res0.create_group("TimePoint 1")
        tp0.create_group("Channel 0").create_dataset("Data", data=np.zeros((2, 2, 2), dtype=np.uint16))
        tp1.create_group("Channel 0").create_dataset("Data", data=np.ones((2, 2, 2), dtype=np.uint16))

    list_exit = main(["list-scenes", str(ims_path), "--json"])
    list_payload = json.loads(capsys.readouterr().out)
    assert list_exit == 0
    assert list_payload["scene_count"] >= 2

    open_exit = main(["open", str(ims_path), "--scene-index", "1", "--json"])
    open_payload = json.loads(capsys.readouterr().out)
    assert open_exit == 0
    assert open_payload["scene"].endswith("TimePoint 1")


def test_cli_run_workflow_rejects_memory_budget_exceeded(tmp_path: Path, capsys) -> None:
    image_path = tmp_path / "input.tif"
    workflow_path = tmp_path / "workflow_budget.json"
    project_dir = tmp_path / "project"

    imwrite(image_path, np.arange(128 * 128, dtype=np.uint16).reshape(128, 128))
    workflow = {
        "workflow_version": "1.0",
        "metadata": {"execution": {"memory_budget_mb": 0.001}},
        "steps": [
            {
                "id": "wf-1",
                "name": "segment",
                "backend_type": "segmentation",
                "backend_name": "watershed",
                "outputs": ["labels"],
            }
        ],
    }
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")

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
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "memory budget exceeded" in captured.err


def test_cli_benchmark_outputs_json(capsys) -> None:
    exit_code = main(
        [
            "benchmark",
            "--suite",
            "baseline",
            "--repeats",
            "1",
            "--size-2d",
            "32",
            "--size-3d",
            "8,24,24",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["suite"] == "baseline"
    assert payload["summary"]["case_count"] >= 3
    assert "cases" in payload


def test_cli_run_workflow_tiled_restoration_under_memory_budget(tmp_path: Path, capsys) -> None:
    image_path = tmp_path / "input.tif"
    workflow_path = tmp_path / "workflow_tiled.json"
    project_dir = tmp_path / "project"

    imwrite(image_path, np.arange(256 * 256, dtype=np.uint16).reshape(256, 256))
    workflow = {
        "workflow_version": "1.0",
        "metadata": {"execution": {"memory_budget_mb": 0.01}},
        "steps": [
            {
                "id": "wf-1",
                "name": "denoise-tiled",
                "backend_type": "restoration",
                "backend_name": "classic",
                "params": {
                    "operation": "denoise",
                    "method": "gaussian",
                    "sigma": 1.0,
                    "tile_shape": [32, 32],
                    "tile_overlap": 4,
                },
                "outputs": ["denoised"],
            }
        ],
    }
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")

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
    _ = capsys.readouterr()
    assert exit_code == 0
    assert (project_dir / "outputs" / "images" / "denoised.tif").is_file()


def test_cli_run_batch_executes_all_inputs(tmp_path: Path, capsys) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    workflow_path = tmp_path / "workflow_batch.json"
    projects_root = tmp_path / "projects"

    imwrite(input_dir / "a.tif", np.arange(64, dtype=np.uint16).reshape(8, 8))
    imwrite(input_dir / "b.tif", np.arange(64, dtype=np.uint16).reshape(8, 8) + 1)

    workflow_path.write_text(
        json.dumps(
            {
                "workflow_version": "1.0",
                "steps": [
                    {
                        "id": "wf-1",
                        "name": "denoise",
                        "backend_type": "restoration",
                        "backend_name": "classic",
                        "params": {"operation": "denoise", "method": "gaussian", "sigma": 1.0},
                        "outputs": ["denoised"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run-batch",
            "--inputs",
            str(input_dir / "*.tif"),
            "--workflow",
            str(workflow_path),
            "--projects-root",
            str(projects_root),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["total_inputs"] == 2
    assert payload["failed_inputs"] == 0
    assert len(payload["results"]) == 2
    for row in payload["results"]:
        project = Path(row["project"])
        assert row["status"] == "ok"
        assert (project / "outputs" / "images" / "denoised.tif").is_file()
