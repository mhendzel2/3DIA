"""Tests for reconstruction QA and model provenance artifacts."""

from __future__ import annotations

import numpy as np

from pymaris.data_model import ImageVolume
from pymaris.reconstruction import DEFAULT_RECONSTRUCTION_REGISTRY, ReconstructionEngine


def test_reconstruction_engine_attaches_qa_and_model_provenance_artifacts() -> None:
    image = ImageVolume(array=np.random.default_rng(0).random((24, 24)), axes=("Y", "X"))
    engine = ReconstructionEngine(DEFAULT_RECONSTRUCTION_REGISTRY)

    result = engine.run(
        plugin_name="deconvolution",
        image=image,
        params={"operation": "distance_map", "method": "distance_map", "threshold": 0.8},
    )

    assert "qa_report" in result.artifacts
    assert "model_provenance" in result.artifacts

    qa_report = result.artifacts["qa_report"]
    model_provenance = result.artifacts["model_provenance"]

    assert qa_report["status"] in {"ok", "warning"}
    assert "input" in qa_report
    assert "output" in qa_report
    assert "warnings" in qa_report

    assert model_provenance["plugin"]["name"] == "deconvolution"
    assert "backend" in model_provenance
    assert model_provenance["mode"] in {"classical", "ml_assisted", "model_based"}

    assert "qa_report" in result.provenance
    assert "model_provenance" in result.provenance
    assert result.provenance["qa_report"]["status"] == qa_report["status"]
