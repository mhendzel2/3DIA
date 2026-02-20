# Reproducible Pipeline Recipe

This recipe ensures identical workflow definitions can be executed through API and CLI with provenance capture.

## 1) Author a versioned workflow document

Use a single JSON file with `workflow_version`, `metadata`, and `steps`.

Example:

```json
{
  "workflow_version": "1.0",
  "metadata": {
    "execution": {
      "prefer_lazy": true,
      "chunks": [8, 128, 128],
      "memory_budget_mb": 512
    }
  },
  "steps": [
    {
      "id": "wf-1",
      "name": "distance-map",
      "backend_type": "restoration",
      "backend_name": "classic",
      "params": {
        "operation": "distance_map",
        "threshold": 0.5
      },
      "inputs": ["image"],
      "outputs": ["distance_map"]
    }
  ]
}
```

## 2) Run headless (single input)

```bash
pymaris-cli run-workflow \
  --input data/sample.tif \
  --workflow workflows/restoration_distance_map.json \
  --project ./project_single
```

## 3) Run headless (batch/HPC style)

```bash
pymaris-cli run-batch \
  --inputs "data/*.tif" \
  --workflow workflows/restoration_distance_map.json \
  --projects-root ./project_batch
```

Each input gets its own project directory with outputs + provenance.

## 4) Validate provenance + environment fingerprint

Inspect:

- `metadata/provenance.json`
- per-step `backend` signature and `result_metadata`
- `environment_check` emitted by CLI for replay compatibility

## 5) Optional reconstruction-module run

```bash
pymaris-cli run-reconstruction \
  --input data/sample.tif \
  --plugin deconvolution \
  --project ./project_recon \
  --params-file examples/reconstruction/deconvolution_distance_map.params.json
```

This stores:

- reconstructed output image,
- plugin tables,
- `*_qc.csv`,
- `qa_report.json` and `model_provenance.json`,
- workflow step provenance metadata.
