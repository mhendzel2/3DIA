"""Headless command-line interface for PyMaris workflows and projects."""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pymaris.backends import DEFAULT_REGISTRY
from pymaris.benchmark import run_baseline_benchmark
from pymaris.data_model import ImageVolume
from pymaris.io import list_scenes, open_image
from pymaris.logging import get_logger
from pymaris.project_store import ProjectStore
from pymaris.reconstruction import (
    DEFAULT_RECONSTRUCTION_REGISTRY,
    ReconstructionEngine,
    calibrations_from_cli_specs,
)
from pymaris.workflow import WorkflowResult, WorkflowStep
from pymaris.workflow_runner import execute_workflow_steps
from pymaris.workflow_validation import validate_workflow_document

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pymaris-cli", description="Headless PyMaris workflow runner")
    parser.add_argument("--debug", action="store_true", help="Show traceback details on command failures")
    subparsers = parser.add_subparsers(dest="command", required=True)

    open_parser = subparsers.add_parser("open", help="Open an image and print summary")
    open_parser.add_argument("path", help="Input image path")
    open_parser.add_argument("--scene", default=None, help="Scene identifier for scene-based formats")
    open_parser.add_argument("--scene-index", type=int, default=None, help="Scene index for scene-based formats")
    open_parser.add_argument("--json", action="store_true", dest="as_json", help="Print JSON output")

    scenes_parser = subparsers.add_parser("list-scenes", help="List scenes for a scene-based microscopy file")
    scenes_parser.add_argument("path", help="Input image path")
    scenes_parser.add_argument("--json", action="store_true", dest="as_json", help="Print JSON output")

    run_workflow = subparsers.add_parser(
        "run-workflow",
        help="Open an input image, execute a workflow JSON, and persist outputs to a project store",
    )
    run_workflow.add_argument("--input", required=True, help="Input image path")
    run_workflow.add_argument("--workflow", required=True, help="Workflow JSON path")
    run_workflow.add_argument("--project", required=True, help="Project directory")
    run_workflow.add_argument("--output-format", default="tiff", choices=["tiff", "zarr"])

    run_project = subparsers.add_parser(
        "run-project",
        help="Run a workflow JSON against an existing project's stored image layer",
    )
    run_project.add_argument("--project", required=True, help="Existing project directory")
    run_project.add_argument("--workflow", required=True, help="Workflow JSON path")
    run_project.add_argument("--image-layer", default=None, help="Stored image layer name")
    run_project.add_argument("--output-format", default="tiff", choices=["tiff", "zarr"])

    run_batch = subparsers.add_parser(
        "run-batch",
        help="Run a workflow JSON across multiple input files (desktop/HPC batch mode)",
    )
    run_batch.add_argument(
        "--inputs",
        required=True,
        help="Input glob (e.g. data/*.tif), a directory, or a single file path",
    )
    run_batch.add_argument("--workflow", required=True, help="Workflow JSON path")
    run_batch.add_argument("--projects-root", required=True, help="Root directory for per-input project outputs")
    run_batch.add_argument("--output-format", default="tiff", choices=["tiff", "zarr"])

    export_parser = subparsers.add_parser("export", help="Copy project outputs to a destination directory")
    export_parser.add_argument("--project", required=True, help="Project directory")
    export_parser.add_argument("--destination", required=True, help="Destination directory")

    backends_parser = subparsers.add_parser(
        "list-backends",
        help="List registered backends and optionally filter by capabilities",
    )
    backends_parser.add_argument(
        "--backend-type",
        choices=["segmentation", "tracking", "tracing", "restoration"],
        default=None,
        help="Limit results to a single backend type",
    )
    backends_parser.add_argument("--ndim", type=int, default=None, help="Require support for this dimensionality")
    backends_parser.add_argument("--modality", default=None, help="Require support for modality (e.g. fluorescence)")
    backends_parser.add_argument(
        "--requires-time",
        action="store_true",
        help="Require backends that explicitly support time-resolved data",
    )
    backends_parser.add_argument(
        "--supports-multichannel",
        action="store_true",
        help="Require backends that explicitly support multichannel input",
    )
    backends_parser.add_argument("--json", action="store_true", dest="as_json", help="Print JSON output")

    reconstruction_parser = subparsers.add_parser(
        "list-reconstruction-plugins",
        help="List registered reconstruction plugins and capability constraints",
    )
    reconstruction_parser.add_argument(
        "--modality",
        default=None,
        help="Filter by modality tag (deconvolution, sim, sted, smlm)",
    )
    reconstruction_parser.add_argument(
        "--ndim",
        type=int,
        default=None,
        help="Require support for this dimensionality",
    )
    reconstruction_parser.add_argument("--json", action="store_true", dest="as_json", help="Print JSON output")

    run_reconstruction = subparsers.add_parser(
        "run-reconstruction",
        help="Run a reconstruction plugin with explicit calibration objects and provenance logging",
    )
    run_reconstruction.add_argument("--input", required=True, help="Input image path")
    run_reconstruction.add_argument("--plugin", required=True, help="Reconstruction plugin name")
    run_reconstruction.add_argument("--project", required=True, help="Project directory")
    run_reconstruction.add_argument("--output-name", default="reconstruction", help="Saved output layer name")
    run_reconstruction.add_argument("--output-format", default="tiff", choices=["tiff", "zarr"])
    run_reconstruction.add_argument(
        "--params-file",
        default=None,
        help="Path to JSON file containing plugin parameter object",
    )
    run_reconstruction.add_argument(
        "--calibration",
        action="append",
        default=[],
        help="Calibration descriptor in name=path format (repeatable)",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run deterministic baseline throughput/correctness benchmarks",
    )
    benchmark_parser.add_argument("--suite", choices=["baseline"], default="baseline")
    benchmark_parser.add_argument("--repeats", type=int, default=1, help="Number of repetitions per case")
    benchmark_parser.add_argument("--size-2d", type=int, default=128, help="2D synthetic benchmark size")
    benchmark_parser.add_argument(
        "--size-3d",
        default="16,64,64",
        help="3D synthetic size as Z,Y,X (example: 16,64,64)",
    )
    benchmark_parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic data")
    benchmark_parser.add_argument("--output", default=None, help="Optional output path for benchmark JSON report")
    benchmark_parser.add_argument("--json", action="store_true", dest="as_json", help="Print JSON output")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "open":
            return _cmd_open(
                path=Path(args.path),
                as_json=args.as_json,
                scene=args.scene,
                scene_index=args.scene_index,
            )
        if args.command == "list-scenes":
            return _cmd_list_scenes(path=Path(args.path), as_json=args.as_json)
        if args.command == "run-workflow":
            return _cmd_run_workflow(
                input_path=Path(args.input),
                workflow_path=Path(args.workflow),
                project_dir=Path(args.project),
                output_format=args.output_format,
            )
        if args.command == "run-project":
            return _cmd_run_project(
                project_dir=Path(args.project),
                workflow_path=Path(args.workflow),
                image_layer=args.image_layer,
                output_format=args.output_format,
            )
        if args.command == "run-batch":
            return _cmd_run_batch(
                inputs=args.inputs,
                workflow_path=Path(args.workflow),
                projects_root=Path(args.projects_root),
                output_format=args.output_format,
            )
        if args.command == "export":
            return _cmd_export(project_dir=Path(args.project), destination=Path(args.destination))
        if args.command == "list-backends":
            return _cmd_list_backends(
                backend_type=args.backend_type,
                ndim=args.ndim,
                modality=args.modality,
                requires_time=True if args.requires_time else None,
                supports_multichannel=True if args.supports_multichannel else None,
                as_json=args.as_json,
            )
        if args.command == "list-reconstruction-plugins":
            return _cmd_list_reconstruction_plugins(
                modality=args.modality,
                ndim=args.ndim,
                as_json=args.as_json,
            )
        if args.command == "run-reconstruction":
            return _cmd_run_reconstruction(
                input_path=Path(args.input),
                plugin_name=args.plugin,
                project_dir=Path(args.project),
                output_name=args.output_name,
                output_format=args.output_format,
                params_file=Path(args.params_file) if args.params_file else None,
                calibration_specs=[str(value) for value in args.calibration],
            )
        if args.command == "benchmark":
            return _cmd_benchmark(
                suite=args.suite,
                repeats=args.repeats,
                size_2d=args.size_2d,
                size_3d=args.size_3d,
                seed=args.seed,
                output=Path(args.output) if args.output else None,
                as_json=args.as_json,
            )
        raise ValueError(f"unsupported command: {args.command!r}")
    except KeyboardInterrupt:
        print("Error: interrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _cmd_open(path: Path, as_json: bool, scene: str | None, scene_index: int | None) -> int:
    image = open_image(path, scene=scene, scene_index=scene_index)
    payload = {
        "path": str(path),
        "shape": list(image.shape),
        "axes": list(image.axes),
        "dtype": str(image.dtype),
        "is_lazy": image.is_lazy,
        "scene": image.metadata.get("scene"),
        "available_scenes": image.metadata.get("available_scenes", []),
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"path={payload['path']}")
        print(f"shape={payload['shape']}")
        print(f"axes={payload['axes']}")
        print(f"dtype={payload['dtype']}")
        print(f"is_lazy={payload['is_lazy']}")
        print(f"scene={payload['scene']}")
        print(f"available_scenes={payload['available_scenes']}")
    return 0


def _cmd_list_scenes(path: Path, as_json: bool) -> int:
    scenes = list_scenes(path)
    payload = {"path": str(path), "scene_count": len(scenes), "scenes": scenes}
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"path={payload['path']}")
        print(f"scene_count={payload['scene_count']}")
        for item in scenes:
            print(item)
    return 0


def _cmd_run_workflow(
    input_path: Path,
    workflow_path: Path,
    project_dir: Path,
    output_format: str,
) -> int:
    workflow_document = _load_workflow_document(workflow_path)
    execution_options = _execution_options_from_metadata(workflow_document.get("metadata", {}))
    image = open_image(
        input_path,
        prefer_lazy=execution_options.get("prefer_lazy"),
        chunks=execution_options.get("chunks"),
    )
    store = ProjectStore(project_dir)
    store.initialize()
    store.record_input(input_path)
    store.save_image_layer(name="input", image=image, format=output_format)
    environment_delta = store.compare_environment()
    context: dict[str, Any] = {"image": image}
    _execute_workflow(
        store=store,
        workflow_document=workflow_document,
        context=context,
        output_format=output_format,
        environment_delta=environment_delta,
        execution_options=execution_options,
    )
    return 0


def _cmd_run_project(
    project_dir: Path,
    workflow_path: Path,
    image_layer: str | None,
    output_format: str,
) -> int:
    workflow_document = _load_workflow_document(workflow_path)
    execution_options = _execution_options_from_metadata(workflow_document.get("metadata", {}))
    store = ProjectStore(project_dir)
    store.initialize()
    environment_delta = store.compare_environment()
    images = store.load_image_layers()
    if not images:
        raise RuntimeError(
            f"project contains no saved image layers: {project_dir / 'outputs' / 'images'}"
        )
    chosen_name = image_layer or sorted(images.keys())[0]
    if chosen_name not in images:
        raise KeyError(f"image layer not found: {chosen_name!r}")

    context: dict[str, Any] = {"image": images[chosen_name]}
    _execute_workflow(
        store=store,
        workflow_document=workflow_document,
        context=context,
        output_format=output_format,
        environment_delta=environment_delta,
        execution_options=execution_options,
    )
    return 0


def _cmd_export(project_dir: Path, destination: Path) -> int:
    store = ProjectStore(project_dir)
    export_root = store.export_outputs(destination)
    print(json.dumps({"project": str(project_dir), "exported_to": str(export_root)}, indent=2))
    return 0


def _cmd_run_batch(
    *,
    inputs: str,
    workflow_path: Path,
    projects_root: Path,
    output_format: str,
) -> int:
    workflow_document = _load_workflow_document(workflow_path)
    execution_options = _execution_options_from_metadata(workflow_document.get("metadata", {}))
    input_paths = _resolve_batch_inputs(inputs)
    projects_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failure_count = 0
    for index, input_path in enumerate(input_paths):
        project_dir = projects_root / f"{index + 1:03d}_{_slugify(input_path.stem)}"
        try:
            image = open_image(
                input_path,
                prefer_lazy=execution_options.get("prefer_lazy"),
                chunks=execution_options.get("chunks"),
            )
            store = ProjectStore(project_dir)
            store.initialize()
            store.record_input(input_path)
            store.save_image_layer(name="input", image=image, format=output_format)
            environment_delta = store.compare_environment()
            context: dict[str, Any] = {"image": image}
            summary = _execute_workflow(
                store=store,
                workflow_document=workflow_document,
                context=context,
                output_format=output_format,
                environment_delta=environment_delta,
                execution_options=execution_options,
                emit_summary=False,
            )
            rows.append(
                {
                    "input": str(input_path),
                    "project": str(project_dir),
                    "status": "ok",
                    "steps": summary.get("steps", []),
                }
            )
        except Exception as exc:
            failure_count += 1
            rows.append(
                {
                    "input": str(input_path),
                    "project": str(project_dir),
                    "status": "error",
                    "error": str(exc),
                }
            )

    payload = {
        "mode": "batch",
        "workflow": str(workflow_path),
        "projects_root": str(projects_root),
        "total_inputs": len(input_paths),
        "failed_inputs": failure_count,
        "results": rows,
    }
    print(json.dumps(payload, indent=2))
    return 0 if failure_count == 0 else 1


def _cmd_list_backends(
    *,
    backend_type: str | None,
    ndim: int | None,
    modality: str | None,
    requires_time: bool | None,
    supports_multichannel: bool | None,
    as_json: bool,
) -> int:
    infos = DEFAULT_REGISTRY.list_backend_info(backend_type=backend_type)
    output: dict[str, Any] = {
        "filters": {
            "backend_type": backend_type,
            "ndim": ndim,
            "modality": modality,
            "requires_time": requires_time,
            "supports_multichannel": supports_multichannel,
        },
        "backends": {},
    }

    total_count = 0
    for group_name, group_infos in infos.items():
        filtered_names = set(
            DEFAULT_REGISTRY.find_backends(
                group_name,
                ndim=ndim,
                modality=modality,
                requires_time=requires_time,
                supports_multichannel=supports_multichannel,
            )
        )
        rows: list[dict[str, Any]] = []
        for info in group_infos:
            if info.name not in filtered_names:
                continue
            row: dict[str, Any] = {"name": info.name, "version": info.version}
            if info.capability is not None:
                row["capability"] = {
                    "task": info.capability.task,
                    "dimensions": list(info.capability.dimensions),
                    "modalities": list(info.capability.modalities),
                    "supports_time": info.capability.supports_time,
                    "supports_multichannel": info.capability.supports_multichannel,
                    "notes": info.capability.notes,
                }
            rows.append(row)
        output["backends"][group_name] = rows
        total_count += len(rows)

    output["count"] = total_count

    if as_json:
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0

    print(
        "filters:",
        json.dumps(output["filters"], sort_keys=True),
    )
    for group_name, rows in output["backends"].items():
        print(f"[{group_name}]")
        if not rows:
            print("  (none)")
            continue
        for row in rows:
            capability = row.get("capability", {})
            dims = capability.get("dimensions", [])
            modalities = capability.get("modalities", [])
            supports_time_value = capability.get("supports_time", False)
            print(
                f"  - {row['name']} (v{row['version']}) "
                f"dims={dims or 'any'} modality={modalities or 'any'} "
                f"time={supports_time_value}"
            )
    return 0


def _cmd_list_reconstruction_plugins(
    *,
    modality: str | None,
    ndim: int | None,
    as_json: bool,
) -> int:
    infos = DEFAULT_RECONSTRUCTION_REGISTRY.list_info(modality=modality, ndim=ndim)
    payload: dict[str, Any] = {
        "filters": {"modality": modality, "ndim": ndim},
        "count": len(infos),
        "plugins": [],
    }
    for info in infos:
        payload["plugins"].append(
            {
                "name": info.name,
                "version": info.version,
                "license": info.license,
                "modality_tags": list(info.modality_tags),
                "supported_dims": list(info.supported_dims),
                "required_calibrations": list(info.required_calibrations),
                "external_tool": info.external_tool,
                "notes": info.notes,
            }
        )

    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print("filters:", json.dumps(payload["filters"], sort_keys=True))
    for plugin in payload["plugins"]:
        print(
            f"  - {plugin['name']} (v{plugin['version']}, license={plugin['license']}) "
            f"modalities={plugin['modality_tags']} dims={plugin['supported_dims']} "
            f"required_calibrations={plugin['required_calibrations'] or 'none'}"
        )
    return 0


def _cmd_run_reconstruction(
    *,
    input_path: Path,
    plugin_name: str,
    project_dir: Path,
    output_name: str,
    output_format: str,
    params_file: Path | None,
    calibration_specs: list[str],
) -> int:
    params: dict[str, Any] = {}
    if params_file is not None:
        params = _load_json_object(params_file)

    image = open_image(input_path)
    calibrations = calibrations_from_cli_specs(calibration_specs)
    engine = ReconstructionEngine(DEFAULT_RECONSTRUCTION_REGISTRY)
    result = engine.run(
        plugin_name=plugin_name,
        image=image,
        params=params,
        calibrations=calibrations,
        context={"command": "run-reconstruction"},
    )

    store = ProjectStore(project_dir)
    store.initialize()
    store.record_input(input_path)
    store.save_image_layer(name="input", image=image, format=output_format)

    outputs: list[str] = []
    if result.image is not None:
        store.save_image_layer(name=output_name, image=result.image, format=output_format)
        outputs.append(output_name)

    for table_name, rows in result.tables.items():
        store.save_table(table_name, rows)
        outputs.append(table_name)

    if result.qc:
        qc_table_name = f"{output_name}_qc"
        store.save_table(qc_table_name, [_normalize_table_row(result.qc)])
        outputs.append(qc_table_name)

    for artifact_name, artifact_value in result.artifacts.items():
        if isinstance(artifact_value, Mapping):
            store.save_graph(artifact_name, graph=artifact_value)
            outputs.append(artifact_name)

    plugin_info = DEFAULT_RECONSTRUCTION_REGISTRY.get(plugin_name).info
    store.record_workflow_step(
        name=f"reconstruction:{plugin_name}",
        params=params,
        inputs=["image"],
        outputs=outputs,
        workflow_version="1.0",
        backend={
            "backend_type": "reconstruction",
            "registered_name": plugin_info.name,
            "version": plugin_info.version,
            "license": plugin_info.license,
            "required_calibrations": list(plugin_info.required_calibrations),
            "modality_tags": list(plugin_info.modality_tags),
        },
        result_metadata={
            "plugin_metadata": dict(result.metadata),
            "qc": dict(result.qc),
            "provenance": dict(result.provenance),
        },
    )

    summary = {
        "plugin": plugin_name,
        "project": str(project_dir),
        "output_name": output_name if result.image is not None else None,
        "tables": sorted(result.tables.keys()),
        "artifacts": sorted(result.artifacts.keys()),
        "calibrations": sorted(calibrations.keys()),
        "qc": result.qc,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _execute_workflow(
    *,
    store: ProjectStore,
    workflow_document: Mapping[str, Any],
    context: dict[str, Any],
    output_format: str,
    environment_delta: Mapping[str, Any] | None = None,
    execution_options: Mapping[str, Any] | None = None,
    emit_summary: bool = True,
) -> dict[str, Any]:
    raw_steps = workflow_document.get("steps", [])
    workflow_version = str(workflow_document.get("workflow_version", "1.0"))
    workflow_metadata = workflow_document.get("metadata", {})
    steps = _build_workflow_steps(raw_steps)
    summary: dict[str, Any] = {
        "workflow_version": workflow_version,
        "workflow_metadata": workflow_metadata if isinstance(workflow_metadata, dict) else {},
        "steps": [],
    }
    if isinstance(execution_options, Mapping):
        summary["execution"] = dict(execution_options)
    if isinstance(environment_delta, Mapping):
        summary["environment_check"] = dict(environment_delta)
    resource_limits = (
        {"memory_budget_mb": execution_options["memory_budget_mb"]}
        if isinstance(execution_options, Mapping) and "memory_budget_mb" in execution_options
        else None
    )
    executed = execute_workflow_steps(
        steps=steps,
        context=context,
        on_progress=lambda step, p, m: LOGGER.info("[%s] %3d%% %s", step.id, p, m),
        resource_limits=resource_limits,
    )
    for item in executed:
        _persist_result(
            store=store,
            step=item.step,
            result=item.result,
            context=context,
            output_format=output_format,
            workflow_version=workflow_version,
        )
        summary["steps"].append(
            {
                "id": item.step.id,
                "name": item.step.name,
                "outputs": list(item.result.outputs.keys()),
            }
        )

    if emit_summary:
        print(json.dumps(summary, indent=2))
    return summary


def _load_workflow_document(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    document = validate_workflow_document(payload)
    return {
        "workflow_version": str(document["workflow_version"]),
        "metadata": dict(document["metadata"]),
        "steps": [dict(step) for step in document["steps"]],
    }


def _build_workflow_steps(raw_steps: Any) -> list[WorkflowStep]:
    steps: list[WorkflowStep] = []
    for raw in list(raw_steps):
        steps.append(WorkflowStep.from_dict(raw))
    return steps


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _resolve_batch_inputs(spec: str) -> list[Path]:
    raw = str(spec).strip()
    if not raw:
        raise ValueError("--inputs must not be empty")

    candidate = Path(raw)
    if candidate.is_file():
        return [candidate]
    if candidate.is_dir():
        files = sorted(path for path in candidate.rglob("*") if path.is_file())
        if not files:
            raise ValueError(f"no files found in input directory: {candidate}")
        return files

    matches = sorted(Path(path) for path in glob.glob(raw, recursive=True))
    files = [path for path in matches if path.is_file()]
    if not files:
        raise ValueError(f"no files matched input pattern: {raw!r}")
    return files


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "project"


def _execution_options_from_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    execution = metadata.get("execution")
    if not isinstance(execution, Mapping):
        return {}
    options: dict[str, Any] = {}
    if "prefer_lazy" in execution:
        options["prefer_lazy"] = bool(execution["prefer_lazy"])
    if "chunks" in execution and isinstance(execution["chunks"], list):
        options["chunks"] = [int(value) for value in execution["chunks"]]
    if "memory_budget_mb" in execution:
        options["memory_budget_mb"] = float(execution["memory_budget_mb"])
    return options


def _cmd_benchmark(
    *,
    suite: str,
    repeats: int,
    size_2d: int,
    size_3d: str,
    seed: int,
    output: Path | None,
    as_json: bool,
) -> int:
    if suite != "baseline":
        raise ValueError(f"unsupported benchmark suite: {suite}")
    shape_3d = _parse_3d_size(size_3d)
    report = run_baseline_benchmark(repeats=repeats, size_2d=size_2d, size_3d=shape_3d, seed=seed)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        summary = report.get("summary", {})
        print(
            f"suite={report.get('suite')} cases={summary.get('case_count')} "
            f"ok={summary.get('successful_cases')} failed={summary.get('failed_cases')} "
            f"elapsed={summary.get('elapsed_seconds'):.3f}s"
        )
        for case in report.get("cases", []):
            timing = case.get("timing", {})
            mean_seconds = timing.get("mean_seconds")
            mean_text = f"{mean_seconds:.4f}s" if isinstance(mean_seconds, float) else "n/a"
            print(
                f"  - {case.get('id')}: status={case.get('status')} "
                f"repeat={case.get('repeats_completed')} mean={mean_text}"
            )
        if output is not None:
            print(f"saved={output}")
    return 0


def _parse_3d_size(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise ValueError(f"--size-3d must be formatted as Z,Y,X; got: {value!r}")
    try:
        z, y, x = (int(part) for part in parts)
    except Exception as exc:
        raise ValueError(f"--size-3d must contain integer values; got: {value!r}") from exc
    if z <= 0 or y <= 0 or x <= 0:
        raise ValueError("--size-3d values must be > 0")
    return z, y, x


def _persist_result(
    *,
    store: ProjectStore,
    step: WorkflowStep,
    result: WorkflowResult,
    context: dict[str, Any],
    output_format: str,
    workflow_version: str,
) -> None:
    for output_name, value in result.outputs.items():
        context[output_name] = value
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
            rows = _table_dict_to_rows(table_payload)
            if rows:
                store.save_table(table_name, rows)

    store.record_workflow_step(
        name=step.name,
        params=step.params,
        inputs=step.inputs,
        outputs=list(result.outputs.keys()),
        workflow_version=workflow_version,
        backend=_backend_signature_for_step(step),
        result_metadata=result.metadata,
    )


def _backend_signature_for_step(step: WorkflowStep) -> dict[str, Any]:
    try:
        info = DEFAULT_REGISTRY.resolve_backend_info(step.backend_type, step.backend_name)
    except Exception:
        return {
            "backend_type": step.backend_type,
            "registered_name": step.backend_name,
        }

    signature: dict[str, Any] = {
        "backend_type": step.backend_type,
        "registered_name": step.backend_name,
        "info_name": info.name,
        "version": info.version,
    }
    if info.capability is not None:
        signature["capability"] = {
            "task": info.capability.task,
            "dimensions": list(info.capability.dimensions),
            "modalities": list(info.capability.modalities),
            "supports_time": info.capability.supports_time,
            "supports_multichannel": info.capability.supports_multichannel,
            "notes": info.capability.notes,
        }
    return signature


def _table_dict_to_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    list_columns = {key: value for key, value in table.items() if isinstance(value, list)}
    if not list_columns:
        return [table]
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


def _normalize_table_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (dict, list, tuple)):
            row[str(key)] = json.dumps(value, sort_keys=True)
        elif isinstance(value, (np.generic,)):
            row[str(key)] = value.item()
        else:
            row[str(key)] = value
    return row


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
