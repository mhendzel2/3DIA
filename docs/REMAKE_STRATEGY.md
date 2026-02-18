# PyMaris Remake Strategy (Spec-Aligned, 2026)

## Goal
Rebuild PyMaris from a widget-centric prototype into a reproducible, scalable microscopy platform with:
- canonical axis-aware data model,
- OME-Zarr-first storage with physical metadata integrity,
- capability-driven backend/plugin selection,
- deterministic headless workflows with provenance.

## Architecture Direction
1. Core first: `pymaris` (data model, I/O, workflows, provenance, backends registry).
2. UI second: `pymaris_napari` and legacy widgets consume core APIs.
3. Storage contract: OME-Zarr multiscales + labels/tables + provenance sidecar.
4. Execution contract: all algorithms registered as typed backends with declared capabilities.

## Phased Roadmap

### Phase 1: Substrate Hardening (current)
Deliverables:
- axis/units-safe image model,
- OME-Zarr scale/unit round-trip,
- backend capability descriptors and filtering APIs,
- CLI backend discovery.

Acceptance criteria:
- physical scale and units persist through `save_image(..., format="zarr")` -> `open_image(...)`,
- registry supports dimension/modality/time capability filtering,
- CLI exposes capability-filtered backend listing,
- regression tests cover all above behavior.

### Phase 2: Workflow Contract and Provenance
Deliverables:
- strict workflow schema evolution (typed params, versioning),
- backend execution provenance includes backend version/capability signature,
- workflow replay with environment fingerprint checks.

Acceptance criteria:
- workflow replay reproduces artifact graph and parameter history,
- schema validation prevents ambiguous backend invocation.

### Phase 3: Scale-Native Processing
Deliverables:
- lazy/chunk-aware algorithm paths (Dask where supported),
- large-volume safeguards (memory budget, tiling),
- benchmark harness for throughput and correctness.

Acceptance criteria:
- representative large datasets execute without full eager materialization,
- benchmark gates integrated into CI.

### Phase 4: Advanced Algorithm Stack
Deliverables:
- standardized registration, restoration, segmentation, tracking adapters,
- model provenance and QA reports for ML outputs,
- backend parity tests across 2D/3D/time modalities.

Acceptance criteria:
- every production backend has capability declaration + reference tests,
- output artifacts include tables + provenance.

### Phase 5: Product Surfaces
Deliverables:
- napari workflows fully mapped to core API,
- headless CLI pipelines for desktop/HPC,
- documentation for reproducible publication workflows.

Acceptance criteria:
- same workflow definition runs in GUI and CLI with equivalent outputs,
- docs include end-to-end reproducibility recipe.

## Implementation Completed in This Iteration
- Extended `ImageVolume` with axis units and modality validation.
- Added OME-Zarr metadata parsing/writing for axis units and coordinate scales.
- Preserved channel/time/physical metadata through Zarr round-trips and layer conversion.
- Added backend capability metadata (`task`, `dimensions`, `modalities`, time/multichannel support).
- Added registry filtering APIs and `pymaris-cli list-backends`.
- Added tests for metadata integrity and capability discovery.
- Added versioned workflow document validation (`workflow_version` + metadata envelope).
- Persisted backend provenance signatures per workflow step in project provenance.
- Added environment fingerprint comparison for workflow replay safety.
- Added workflow execution controls in `metadata.execution` (`prefer_lazy`, `chunks`, `memory_budget_mb`).
- Added per-step memory budget enforcement in workflow/job execution.
- Added deterministic baseline benchmark harness (`pymaris-cli benchmark`) with JSON reporting.
- Added reconstruction QA/model provenance artifacts (`qa_report`, `model_provenance`) and persisted QC tables.
- Added backend/reconstruction parity tests across 2D/3D/time modalities and capability declarations.
- Added shared workflow execution API (`execute_workflow_steps`) used for consistent document execution behavior.
- Added `pymaris-cli run-batch` for desktop/HPC-style batch processing.
- Added reproducibility runbook in `docs/REPRODUCIBLE_PIPELINE.md`.

## Execution Policy
- No new UI feature is accepted unless it uses core contracts.
- No algorithm backend is accepted without capability metadata and tests.
- No format support is accepted without axis/scale/units round-trip verification.
