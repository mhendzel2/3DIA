# Changelog

## Unreleased

### 2026-02-09 - Completed follow-up steps 1-2 (manual napari smoke + CI napari job)
- Added `tests/test_napari_widget_smoke.py` with headless smoke coverage for:
  - `FilamentTracingWidget` workflow execution,
  - `DeconvolutionWidget` workflow execution,
  - `WidgetManagerWidget` project-store settings panel interaction.
- Added a dedicated CI job `napari-widget-smoke` in `.github/workflows/ci.yml` that installs `.[dev,napari]` and runs the smoke test with `QT_QPA_PLATFORM=offscreen`.
- Fixed filament widget compatibility with newer scikit-image where `skeletonize_3d` may be unavailable.

### 2026-02-09 - Follow-up steps 1-3 (filament/deconvolution workflow migration + widget-manager settings + CLI errors)
- Migrated `FilamentTracingWidget` to run through core `WorkflowStep` + `JobRunner` (`tracing:skeleton`) with progress updates and cancellation.
- Migrated `DeconvolutionWidget` to run through core `WorkflowStep` + `JobRunner` (`restoration:classic`) with progress updates and cancellation.
- Added `Project Store Settings` controls to `WidgetManagerWidget` and persisted them to `config/widget_config.json` under `project_store`.
- Hardened `pymaris-cli` failure UX: concise non-traceback errors by default, with optional `--debug` traceback mode.
- Extended baseline tracing backend output/params for filament-style summaries (branch points, filaments, total length, average thickness).
- Added/updated tests for tracing backend outputs, tracing/restoration workflow execution, and CLI error/debug behavior.

### 2026-02-09 - Follow-up steps 1-3 (tracking migration + CLI schema validation + global project-store settings)
- Migrated `AdvancedTrackingWidget` to run tracking through core `WorkflowStep` + `JobRunner` with progress callbacks and cancellation.
- Added tracking result normalization so existing statistics/lineage UI remains compatible with backend-driven outputs.
- Added runtime workflow validation module `src/pymaris/workflow_validation.py` and wired it into `pymaris-cli` load path.
- Added global napari project-store settings helper `src/pymaris_napari/settings.py` with session naming modes (`none`, `fixed`, `timestamp`).
- Wired project-store settings into workflow and segmentation/tracking provenance recording paths.
- Added tests for CLI validation and napari settings behavior.

### 2026-02-09 - Follow-up steps 1-3 (widget migration + workflow docs + UI provenance)
- Migrated `SegmentationWidget` watershed execution to the core `WorkflowStep` + `JobRunner` pathway while preserving legacy operations for other segmentation modes.
- Added cancel/progress wiring for the migrated watershed flow in `src/widgets/segmentation_widget.py`.
- Added napari-side provenance helper `src/pymaris_napari/provenance.py` to persist outputs and workflow steps via `ProjectStore`.
- Added automatic provenance recording in the `Workflow Runner` widget, including configurable project directory.
- Added workflow schema and examples:
  - `schemas/workflow.schema.json`
  - `examples/workflows/segmentation_watershed.json`
  - `examples/workflows/restoration_denoise.json`
  - `examples/workflows/tracing_skeleton.json`
- Added CLI workflow documentation at `docs/CLI_WORKFLOWS.md`.
- Added tests for schema/examples and UI provenance persistence.

### 2026-02-09 - Steps 1-3 completion (thin widgets + JobRunner + CLI)
- Added core `WorkflowStep` + `WorkflowResult` serialization/execution model in `src/pymaris/workflow.py`.
- Added core `JobRunner`/`JobHandle` with cooperative cancellation and progress callbacks in `src/pymaris/jobs.py`.
- Added a thin napari `WorkflowRunnerWidget` that calls core workflow steps via `JobRunner` (no direct backend-library calls in widget code) in `src/pymaris_napari/workflow_widget.py`.
- Registered `workflow_runner` in widget factories and NPE2 manifest (`src/pymaris_napari/_widgets.py`, `src/pymaris_napari/napari.yaml`).
- Added headless CLI entrypoint `pymaris-cli` with `open`, `run-workflow`, `run-project`, and `export` commands (`src/pymaris/cli.py`).
- Extended `ProjectStore` with tracking/graph persistence and output export helpers.
- Added tests for workflow/jobs/CLI/widget registry wiring.

### 2026-02-09 - Phase 2 foundation + Phase 3 backend scaffolding (steps 1-3)
- Added canonical `ImageVolume` data model and centralized napari layer tuple conversions in core (`src/pymaris/data_model.py`, `src/pymaris/layers.py`).
- Added `pymaris.io` APIs (`open_image`, `save_image`) with optional lazy/dask behavior and optional Zarr support.
- Added `ProjectStore` with provenance recording for inputs, outputs, workflow steps, and environment snapshots (`src/pymaris/project_store.py`).
- Added backend interfaces/registries and baseline adapters for segmentation, tracking, tracing, and restoration (`src/pymaris/backends/*`).
- Switched napari NPE2 reader/writer adapters to use core I/O + conversion APIs (`src/pymaris_napari/_io.py`).
- Added deterministic tests for model/IO, project store round-trip, and backend registries (`tests/test_image_volume_and_io.py`, `tests/test_project_store.py`, `tests/test_backends_registry.py`).
- Updated CI lint scope for new modules/tests and added README usage documentation for the new headless APIs.

### 2026-02-09 - Phase 1 (items 1-3)
- Introduced a split package layout with a napari-free core package (`pymaris`) and a napari plugin package (`pymaris_napari`).
- Added compatibility shims for legacy launcher scripts (`src/main_napari.py`, `src/main_napari_configurable.py`).
- Added a valid NPE2 manifest with explicit `commands`, `readers`, `writers`, and `widgets` contributions.
- Updated packaging and dependency strategy in `pyproject.toml` to use optional extras for heavy/optional capabilities.
- Added `ruff` config, `.pre-commit-config.yaml`, and CI workflow (`.github/workflows/ci.yml`) for lint + tests on Python 3.10 and 3.11.
- Added focused tests in `tests/` for package boundaries, manifest validity, and shim imports.
- Added migration notes in `MIGRATION.md`.

### 2026-02-09 - Phase 0 baseline audit
- Added `PROGRESS.md` with reproducible baseline environment/install/launch/test commands and observed failures.
- Added `ARCHITECTURE.md` with factual entrypoint/control-flow mapping, IO/feature location mapping, and coupling/bottleneck notes with file-line references.
- Added `tests/test_phase0_baseline_docs.py` to assert required Phase 0 baseline docs are present.
- Documented baseline limitations before Phase 1 packaging and separation work.
