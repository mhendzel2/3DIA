# PROGRESS

## 2026-02-14 - `bug_fixes.py` template modernization pass

### Completed
- Refactored large embedded legacy string blocks in `bug_fixes.py` into typed helper builders:
  - `_build_track_endpoint_code()`
  - `_build_analysis_utils_module()`
  - `_build_fixed_requirements_text()`
  - shared normalization helper `_normalize_generated_block(...)`
- Updated legacy integration methods to consume helper builders without changing external behavior:
  - `TimelapseLogicFix.fix_app_track_endpoint()`
  - `DependencyFallbackFix.create_analysis_utils_with_fallbacks()`
  - `apply_all_fixes()` requirements generation path.
- Kept generated content contract intact (same functional templates, deterministic trailing newline normalization).
- Verification:
  - `.venv/bin/python -m ruff check bug_fixes.py` passed.
  - `.venv/bin/python -m pytest -q` full suite passed.

## 2026-02-14 - `bug_fixes.py` lint/format normalization pass

### Completed
- Normalized `bug_fixes.py` formatting and lint debt to project Ruff baseline:
  - cleaned import ordering/unused imports,
  - removed trailing whitespace throughout the file,
  - removed unused `scipy.ndimage` import in phase-correlation path.
- Verification:
  - `.venv/bin/python -m ruff check bug_fixes.py` passed.
  - `.venv/bin/python -m pytest -q` full suite passed.

## 2026-02-14 - Warning cleanup pass (headless plotting + external deprecation filter)

### Completed
- Updated plotting behavior in `StatisticsWidget.generate_plot` to avoid `plt.show()` on non-interactive backends (`Agg`) and close figures cleanly during headless runs.
- Added targeted pytest warning filter in `pyproject.toml` for third-party Python 3.14 deprecation noise:
  - `Pickle, copy, and deepcopy support will be removed from itertools in Python 3.14.`
- Kept behavior unchanged for interactive environments (plots still display when interactive backend is active).
- Validation:
  - `.venv/bin/python -m ruff check src/widgets/statistics_widget.py pyproject.toml` passed.
  - `.venv/bin/python -m pytest -q` full suite passed without warning summary output.

## 2026-02-14 - Legacy test hardening and bug-fix stabilization pass

### Completed
- Refactored legacy root test modules to assertion-based pytest style (no `return True/False` patterns):
  - `test_advanced_enhancements.py`
  - `test_batch_hca_features.py`
  - `test_core_functionality.py`
  - `test_debugging_fixes.py`
  - `test_dose_response.py`
  - `test_fixes.py`
  - `test_flask_imports.py`
  - `test_imports_only.py`
  - `test_structural_cleanup.py`
- Updated environment-sensitive checks to use robust skip behavior where services/deps are optional (Flask server availability, napari runtime paths), while keeping meaningful assertions for available paths.
- Fixed a runtime bug in ChimeraX auto-detection:
  - `bug_fixes.py` no longer depends on `os.getlogin()` only.
  - Added fallback to `getpass.getuser()` to avoid `OSError: [Errno 25] Inappropriate ioctl for device` in non-interactive environments.
- Validation:
  - `.venv/bin/python -m ruff check` on all modified test files passed.
  - `.venv/bin/python -m pytest -q` full suite passed.

## 2026-02-14 - Completion pass (headless napari stability + legacy widget test alignment)

### Completed
- Added repository-level `conftest.py` to enforce stable headless test runtime defaults:
  - `QT_QPA_PLATFORM=offscreen`
  - `MPLBACKEND=Agg`
  - writable local `XDG_CACHE_HOME` under `.pytest_cache/runtime/xdg_cache`
  - writable file-based `NAPARI_CONFIG` under `.pytest_cache/runtime/napari/settings.yaml`
- Updated legacy `test_new_widgets.py` to align with migrated deconvolution execution model:
  - switched from deprecated `thread.wait()` to workflow/job handle assertions and result handling,
  - forced layer combo refresh before execution,
  - added environment-aware module skip when napari/PyQt6 are unavailable.
- Verified end-to-end validation:
  - `.venv/bin/python -m pytest -q` passes.
  - `.venv/bin/python -m ruff check conftest.py test_new_widgets.py` passes.

## 2026-02-09 - Completed follow-up steps 1-2 (manual napari smoke + CI napari widget smoke job)

### Completed
- Ran an offscreen/manual napari smoke check for the requested UI paths using `napari.components.ViewerModel`:
  - `FilamentTracingWidget` workflow execution (`trace_filaments`)
  - `DeconvolutionWidget` workflow execution (`run_deconvolution`)
  - `WidgetManagerWidget` project-store panel interactions (`_collect_project_store_settings`)
- Added dedicated headless napari smoke tests:
  - `tests/test_napari_widget_smoke.py`
  - includes smoke coverage for filament tracing, deconvolution, and widget-manager project-store settings controls.
- Added a separate GitHub Actions job for napari smoke coverage:
  - `.github/workflows/ci.yml` job `napari-widget-smoke`
  - installs `.[dev,napari]`
  - runs `pytest tests/test_napari_widget_smoke.py -q` with `QT_QPA_PLATFORM=offscreen`.
- Fixed a scikit-image compatibility issue in legacy filament tracing thread import path:
  - `skeletonize_3d` is now optional; tracing no longer disables itself if that symbol is unavailable.

## 2026-02-09 - Follow-up steps 1-3 (filament/deconvolution workflow migration + widget-manager settings + CLI UX)

### Completed
- Migrated `FilamentTracingWidget` to core workflow/jobs:
  - Uses `WorkflowStep(backend_type="tracing", backend_name="skeleton")`
  - Runs via `JobRunner` with progress callbacks and cancellation.
  - Preserves existing skeleton/branch-point/statistics rendering and export paths.
- Migrated `DeconvolutionWidget` to core workflow/jobs:
  - Uses `WorkflowStep(backend_type="restoration", backend_name="classic")` with `operation="deconvolve"`.
  - Runs via `JobRunner` with progress callbacks and cancellation.
  - Preserves existing output layer names (`richardson_lucy_deconvolved`, `wiener_deconvolved`).
- Extended baseline tracing backend for filament-style outputs:
  - Accepts threshold/smoothing/min-size parameters.
  - Emits graph/table fields for total length, branch points, filament count, and average thickness.
- Added project-store settings panel in `WidgetManagerWidget`:
  - `base_project_dir`
  - `session_naming` (`none`/`fixed`/`timestamp`)
  - session name/prefix
  - `provenance_enabled`
  - persists under `config/widget_config.json -> project_store`.
- Hardened CLI runtime UX:
  - non-debug errors now return concise `Error: ...` messages (no traceback by default).
  - `--debug` flag re-enables full traceback for diagnosis.
- Added/updated tests:
  - `tests/test_cli.py` (`--debug` behavior + non-traceback error path assertions)
  - `tests/test_backends_registry.py` (tracing backend filament summary fields)
  - `tests/test_workflow_jobs.py` (tracing/restoration workflow execution)

## 2026-02-09 - Follow-up steps 1-3 (tracking migration + CLI schema validation + global project-store settings)

### Completed
- Migrated `AdvancedTrackingWidget` execution path to core workflow/jobs:
  - Uses `WorkflowStep(backend_type="tracking", backend_name="hungarian")`
  - Runs via `JobRunner` (background thread), progress callbacks, cancellation button
  - Preserves existing statistics table/tree rendering via normalized result payloads.
- Added runtime workflow validation in CLI:
  - New module `src/pymaris/workflow_validation.py`
  - `pymaris-cli` now validates workflow payloads before execution and fails fast on malformed steps.
- Added global project-store settings with session naming:
  - New `src/pymaris_napari/settings.py`
  - Supports `base_project_dir`, `session_naming` (`none`/`fixed`/`timestamp`), `session_name`, `session_prefix`, `provenance_enabled`
  - Settings persisted in `config/widget_config.json` under `project_store`.
- Updated widgets to use global settings for provenance writes:
  - `src/pymaris_napari/workflow_widget.py`
  - `src/widgets/segmentation_widget.py`
  - `src/widgets/tracking_widget.py`
- Added tests for validation/settings and updated CI lint/test scope.

## 2026-02-09 - Follow-up steps 1-3 (widget migration + workflow docs + UI provenance)

### Completed
- Incrementally migrated the legacy `SegmentationWidget` watershed operation to run through core:
  - `WorkflowStep` + `JobRunner` path with progress and cancellation.
  - Existing spot/surface/labels behavior preserved for compatibility.
- Added a reusable UI provenance persistence helper:
  - `src/pymaris_napari/provenance.py`
  - Persists UI workflow outputs/tables + workflow step metadata via `ProjectStore`.
- Wired automatic provenance recording into napari workflow UI:
  - `src/pymaris_napari/workflow_widget.py`
  - Configurable project directory and on/off toggle.
- Added workflow schema + examples:
  - `schemas/workflow.schema.json`
  - `examples/workflows/*.json`
  - `docs/CLI_WORKFLOWS.md` with CLI usage and workflow format documentation.
- Added tests for:
  - workflow examples/schema shape,
  - napari provenance helper persistence,
  - existing workflow/CLI paths still passing.

## 2026-02-09 - Steps 1-3 completion (thin widgets + JobRunner + CLI)

### Completed
- Added serializable workflow execution primitives in core:
  - `src/pymaris/workflow.py`
  - `WorkflowStep` / `WorkflowResult` with backend-type execution and progress hooks.
- Added background job infrastructure in core:
  - `src/pymaris/jobs.py`
  - `JobRunner` + `JobHandle` with cooperative cancellation and standardized error mapping.
- Added napari thin-controller workflow widget:
  - `src/pymaris_napari/workflow_widget.py`
  - Runs core `WorkflowStep` through `JobRunner`
  - Includes progress reporting and cancellation UI
  - Registered through `pymaris_napari._widgets` + NPE2 manifest.
- Added headless CLI:
  - `src/pymaris/cli.py` + script entrypoint `pymaris-cli`
  - `open`, `run-workflow`, `run-project`, `export`.
- Extended project store exports/persistence for workflow outputs:
  - track payload JSON, graph JSON, output export helper.
- Added tests for workflow, jobs, CLI, and widget registry wiring:
  - `tests/test_workflow_jobs.py`
  - `tests/test_cli.py`
  - `tests/test_napari_widgets_registry.py`

## 2026-02-09 - Phase 2 foundation + Phase 3 backend scaffolding (steps 1-3)

### Completed
- Added canonical image container and conversions:
  - `src/pymaris/data_model.py` (`ImageVolume`, canonical axes, metadata/scale handling)
  - `src/pymaris/layers.py` (centralized napari layer tuple conversion)
- Added scalable core I/O API:
  - `src/pymaris/io.py` with `open_image(...)` and `save_image(...)`
  - TIFF + imageio paths by default
  - optional lazy loading via dask when available
  - optional Zarr/OME-Zarr support with multiscale metadata handling
- Added reproducibility/project persistence:
  - `src/pymaris/project_store.py` (`ProjectStore`)
  - on-disk layout for inputs/outputs
  - provenance JSON with inputs, outputs, workflow steps, environment snapshot
  - image/label save+reload round-trip support
- Added backend interfaces + baseline adapters:
  - `src/pymaris/backends/types.py`
  - `src/pymaris/backends/registry.py`
  - `src/pymaris/backends/baseline.py`
  - `src/pymaris/backends/__init__.py` (default registry bootstrap)
  - baseline backends: watershed/cellpose/stardist segmentation, hungarian tracking, skeleton tracing, classic denoise/deconvolution
- Updated napari plugin I/O adapters to call core I/O/conversion APIs:
  - `src/pymaris_napari/_io.py`
- Expanded tests for new core functionality:
  - `tests/test_image_volume_and_io.py`
  - `tests/test_project_store.py`
  - `tests/test_backends_registry.py`
- Updated CI lint scope to include new modules/tests.
- Updated user docs (`README.md`) and changelog entries.

## 2026-02-09 - Phase 1 (items 1-3) packaging + plugin + tooling

### Completed
- Added headless core package boundary: `src/pymaris/` (no napari imports).
- Added napari package boundary: `src/pymaris_napari/` with:
  - default launcher (`app.py`)
  - configurable launcher (`configurable.py`)
  - widget adapters (`_widgets.py`)
  - reader/writer adapters (`_io.py`)
  - NPE2 manifest (`napari.yaml`)
- Added compatibility shims so existing launch scripts still work:
  - `src/main_napari.py`
  - `src/main_napari_configurable.py`
- Updated widget manager to use new configurable loader and correct config path.
- Updated packaging in `pyproject.toml`:
  - optional extras split (`napari`, `web`, `io`, `ai`, `formats`, `tracking`, `dev`)
  - napari manifest entry point now points to `pymaris_napari:napari.yaml`
  - script entry points for napari launchers
- Added tooling:
  - `ruff` configuration
  - `.pre-commit-config.yaml`
  - GitHub Actions CI workflow for lint + tests on Python 3.10 and 3.11
- Added focused tests for package boundaries and manifest wiring under `tests/`.
- Added migration notes in `MIGRATION.md`.

## 2026-02-09 - Phase 0 Baseline Audit

### Repository baseline state
- Working tree is already dirty (many tracked files modified before this audit).
- Python available in this environment is `Python 3.12.3`.
- Local baseline run was performed from `/mnt/c/Users/mjhen/Github/3DIA`.

### Environment creation (commands)
```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install pytest
```

### Installation baseline (commands + outcomes)
```bash
.venv/bin/python -m pip install -e .
```
- Result: failed during dependency resolution/build.
- Key failure: resolver selected `numpy==1.24.0` (from transitive constraints) which has no compatible wheel for Python 3.12 in this path, then build backend failed with:
  - `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`

```bash
.venv/bin/python -m pip install -e . --no-deps
```
- Result: succeeds (editable install of project metadata/code path only).

### Launch baseline (verified entrypoints)

#### Multi-interface launcher
```bash
timeout 10s .venv/bin/python -u src/main.py web
```
- Result: starts fallback HTTP server (`Simple Scientific Image Analyzer`) on `http://localhost:5000`.

```bash
timeout 10s .venv/bin/python -u src/main.py enhanced
```
- Result: enhanced path fails due missing `numpy` import in `src/scientific_analyzer.py`, then falls back to `simple_analyzer` and serves on port 5000.

```bash
timeout 10s .venv/bin/python -u src/main.py napari
```
- Result: reports missing `napari` dependency (`No module named 'napari'`).

#### Direct entrypoints
```bash
timeout 10s .venv/bin/python -u src/main_napari.py
```
- Result: fails (`No module named 'napari'`).

```bash
timeout 10s .venv/bin/python -u src/main_napari_configurable.py
```
- Result: fails (`No module named 'napari'`).

```bash
timeout 10s .venv/bin/python -u src/scientific_analyzer.py
```
- Result: fails (`No module named 'numpy'`).

```bash
timeout 10s .venv/bin/python -u src/simple_analyzer.py
```
- Result: starts fallback HTTP server on `http://localhost:5000`.

### Tests baseline

Initial command (without pytest installed):
```bash
python3 -m pytest -q
```
- Result: `/usr/bin/python3: No module named pytest`

After installing `pytest`:
```bash
.venv/bin/python -m pytest -q
```
- Result: pytest crashed during capture teardown (`FileNotFoundError`) in this WSL path/temporary-file setup.

Working invocation:
```bash
TMPDIR=/tmp .venv/bin/python -m pytest -q
```
- Result: collection errors (3), no tests executed:
  - `test_batch_hca_features.py` -> `ModuleNotFoundError: No module named 'requests'`
  - `test_new_widgets.py` -> `ModuleNotFoundError: No module named 'numpy'`
  - `test_spots_detection_widget.py` -> `ModuleNotFoundError: No module named 'numpy'`

### Known performance bottlenecks observed in baseline
- Eager full-memory reads for large datasets in file I/O:
  - `src/widgets/file_io_widget.py:234`
  - `src/widgets/file_io_widget.py:271`
  - `src/widgets/file_io_widget.py:273`
  - `src/widgets/file_io_widget.py:653`
- Many processing operations run synchronously in widget methods (UI thread), especially slice loops:
  - `src/widgets/processing_widget.py:403`
  - `src/widgets/processing_widget.py:434`
  - `src/widgets/processing_widget.py:461`
  - `src/widgets/processing_widget.py:474`
- Expensive pure-Python nested loops in denoising/segmentation fallbacks:
  - `src/advanced_analysis.py:332`
  - `src/advanced_analysis.py:342`
  - `src/advanced_analysis.py:401`
  - `src/advanced_analysis.py:408`
- Batch pipeline loads full images and keeps results in memory per file/session:
  - `src/batch_processor.py:175`
  - `src/batch_processor.py:203`
  - `src/batch_processor.py:555`
- Web cache stores full arrays and computes memory by repeatedly rescanning cache contents:
  - `src/scientific_analyzer.py:63`
  - `src/scientific_analyzer.py:80`
  - `src/scientific_analyzer.py:98`
  - `src/scientific_analyzer.py:172`

### Next planned milestone
- Start Phase 1: packaging hardening + explicit core/plugin boundaries while preserving current behavior.
