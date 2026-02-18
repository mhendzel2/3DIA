# Migration Notes

## Phase 1 packaging split (2026-02-09)

### What changed
- New headless core package: `pymaris`
- New napari package: `pymaris_napari`
- Legacy launch scripts remain available as compatibility shims:
  - `src/main_napari.py`
  - `src/main_napari_configurable.py`

### Dependency installation
`pip install -e .` now installs a lean core dependency set.

Install extras based on use case:
- Napari desktop/plugin: `pip install -e ".[napari]"`
- Web interface: `pip install -e ".[web]"`
- Advanced I/O: `pip install -e ".[io,formats]"`
- AI backends: `pip install -e ".[ai]"`
- Tracking extras: `pip install -e ".[tracking]"`
- Development tooling: `pip install -e ".[dev]"`

### Backward compatibility
- Existing direct commands continue to work through shims:
  - `python src/main_napari.py`
  - `python src/main_napari_configurable.py`
- Existing widget modules under `src/widgets/` are still used; they are now loaded through adapters in `pymaris_napari`.

## Phase 2 core APIs and provenance (2026-02-09)

### What changed
- Added canonical image model `pymaris.ImageVolume` with explicit axes and metadata.
- Added centralized I/O:
  - `pymaris.open_image(path) -> ImageVolume`
  - `pymaris.save_image(image, destination, format=...)`
- Added persistence/provenance helper:
  - `pymaris.ProjectStore(project_dir)`

### Optional dependency behavior
- Lazy dask-backed arrays are used only when dask is installed (`.[io]`) and lazy loading is requested/appropriate.
- Zarr paths require optional `zarr` dependency (`.[io]`).
- App behavior remains functional without optional extras.

## Workflow and CLI additions (2026-02-09)

### What changed
- New headless CLI command: `pymaris-cli`
- New napari widget: `Workflow Runner (Core Backends)` for backend-driven, cancellable step execution.
- New core APIs:
  - `pymaris.WorkflowStep`
  - `pymaris.JobRunner`

### Compatibility
- Existing widgets and launchers remain available.
- New workflow/CLI paths are additive and do not remove legacy commands.

## Follow-up widget migration and provenance (2026-02-09)

### What changed
- `SegmentationWidget` watershed now executes via core workflow/jobs (`WorkflowStep` + `JobRunner`) with cancellation support.
- `Workflow Runner` napari widget now records `ProjectStore` provenance automatically by default.
- Added workflow JSON schema and examples for CLI/headless runs.

### Compatibility
- Spot detection, surface creation, and legacy labeling paths in `SegmentationWidget` remain intact.
- Watershed result types and layer behavior remain labels-compatible in napari.

## Tracking migration + config-backed project-store settings (2026-02-09)

### What changed
- `AdvancedTrackingWidget` now runs tracking via core workflow/jobs instead of direct widget-owned algorithm execution.
- Workflow JSON is now validated before CLI execution (`pymaris-cli`) with fail-fast errors.
- Global project-store settings are now persisted in `config/widget_config.json` under `project_store`:
  - `base_project_dir`
  - `session_naming` (`none` / `fixed` / `timestamp`)
  - `session_name`
  - `session_prefix`
  - `provenance_enabled`

### Compatibility
- Existing tracking UI outputs (stats table, lineage tree, napari tracks layer) remain available.
- Legacy `TrackingThread` class remains present for compatibility fallback.

## Filament/deconvolution migration + CLI/runtime UX hardening (2026-02-09)

### What changed
- `FilamentTracingWidget` now executes tracing through core `WorkflowStep` + `JobRunner` (`tracing:skeleton`) with cancel/progress handling.
- `DeconvolutionWidget` now executes restoration through core `WorkflowStep` + `JobRunner` (`restoration:classic`) with cancel/progress handling.
- `WidgetManagerWidget` now exposes project-store settings (`base_project_dir`, session naming/value, provenance toggle) and persists them into `config/widget_config.json`.
- `pymaris-cli` now prints concise error messages by default (no traceback) and supports `--debug` to re-raise exceptions with full traceback.
- Baseline `skeleton` tracing backend now accepts filament-style params (`threshold_method`, `manual_threshold`, `min_object_size`, etc.) and returns enriched trace summary fields used by the widget.

### Compatibility
- Existing output layer names are preserved for deconvolution:
  - `richardson_lucy_deconvolved`
  - `wiener_deconvolved`
- Existing filament/tracking/deconvolution widget modules remain importable.
- Legacy `FilamentTracingThread` and `DeconvolutionThread` classes remain present for compatibility fallback.
