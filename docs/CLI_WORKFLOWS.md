# CLI Workflows

This document describes headless workflow execution with `pymaris-cli`.

## Commands

### Inspect data

```bash
pymaris-cli open path/to/image.tif --json
```

Open a specific scene from a scene-based format (for example `.ims`):

```bash
pymaris-cli open path/to/sample.ims --scene-index 1 --json
```

List available scenes:

```bash
pymaris-cli list-scenes path/to/sample.ims --json
```

List available reconstruction plugins:

```bash
pymaris-cli list-reconstruction-plugins --json
```

Run baseline benchmarks (throughput + sanity checks):

```bash
pymaris-cli benchmark --suite baseline --repeats 3 --json
```

Run a reconstruction plugin with explicit calibration inputs:

```bash
pymaris-cli run-reconstruction \
  --input path/to/image.tif \
  --plugin deconvolution \
  --project ./my_project \
  --output-name recon \
  --params-file examples/reconstruction/deconvolution_distance_map.params.json \
  --calibration psf=path/to/psf.tif
```

`run-reconstruction` persists image/table outputs plus a QC table and structured
QA/model provenance artifacts (`*_qc.csv`, `qa_report.json`, `model_provenance.json`).

### Run a workflow from input data

```bash
pymaris-cli run-workflow \
  --input path/to/image.tif \
  --workflow examples/workflows/segmentation_watershed.json \
  --project ./my_project
```

### Run a workflow from existing project data

```bash
pymaris-cli run-project \
  --project ./my_project \
  --workflow examples/workflows/restoration_denoise.json \
  --image-layer input
```

### Run a workflow in batch mode (desktop/HPC)

```bash
pymaris-cli run-batch \
  --inputs "data/*.tif" \
  --workflow examples/workflows/restoration_denoise.json \
  --projects-root ./batch_projects
```

### Export project outputs

```bash
pymaris-cli export --project ./my_project --destination ./export_bundle
```

### Error handling and debugging

By default, CLI failures return a concise `Error: ...` message without a traceback.

Use `--debug` to re-raise exceptions and show full traceback details:

```bash
pymaris-cli --debug run-workflow \
  --input path/to/image.tif \
  --workflow bad_workflow.json \
  --project ./my_project
```

## Workflow JSON shape

Workflow files are validated by convention against:

- `schemas/workflow.schema.json`

Accepted top-level formats:

1. Versioned object: `{ "workflow_version": "1.0", "metadata": {...}, "steps": [ ... ] }`
2. Legacy object: `{ "steps": [ ... ] }`
3. Legacy array: `[ ... ]`

`workflow_version` currently supports major version `1` (`1`, `1.0`, `1.2.3`, etc).
Unsupported major versions are rejected before execution.

Each step must provide:

- `id`
- `name`
- `backend_type` one of: `segmentation`, `tracking`, `tracing`, `restoration`
- `backend_name`

Optional fields:

- `params` (object)
- `inputs` (array of strings)
- `outputs` (array of strings)

`metadata.execution` supports deterministic execution controls:

- `prefer_lazy` (boolean): request lazy loading for input data
- `chunks` (array of positive integers): chunk shape hint when lazy loading is enabled
- `memory_budget_mb` (number > 0): per-step input-size safety limit (approximate)

For restoration workflows, an AI denoising backend is available:

- `backend_name: "ai_denoise"`
- `params.method` supports: `non_local_means`, `bilateral`, `wiener`, `distance_map`

Restoration steps also support Euclidean distance maps:

- `params.operation: "distance_map"` (or `"euclidean_distance_map"`)
- Optional: `threshold` (default `0.0`)
- Optional: `distance_to` (`"background"` or `"foreground"`)

Validation is enforced at runtime in `pymaris-cli`:

- malformed workflows fail before execution starts
- errors include step index and missing/invalid keys
- non-debug mode prints a concise error message and exits with status `1`

## Provenance and reproducibility

CLI runs persist outputs and metadata under the project directory:

- `outputs/images/`
- `outputs/labels/`
- `outputs/tracks/`
- `outputs/meshes/`
- `outputs/tables/`
- `metadata/provenance.json`

Each workflow step record now includes:
- workflow document version
- backend signature (`backend_type`, registry key, backend version, declared capability)
- backend result metadata

When running on an existing project (`run-project`), CLI also reports an environment
comparison (`environment_check`) in stdout JSON so replay risks are visible.
