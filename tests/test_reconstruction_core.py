"""Tests for modular reconstruction plugin contracts and engine behavior."""

from __future__ import annotations

import numpy as np
import pytest

from pymaris.data_model import ImageVolume
from pymaris.reconstruction import (
    DEFAULT_RECONSTRUCTION_REGISTRY,
    CalibrationArtifact,
    ReconstructionEngine,
)


def test_default_reconstruction_registry_contains_core_plugins() -> None:
    names = sorted(DEFAULT_RECONSTRUCTION_REGISTRY.plugins.keys())
    assert "deconvolution" in names
    assert "sim" in names
    assert "sted" in names
    assert "smlm" in names


def test_registry_filter_by_modality_tag() -> None:
    infos = DEFAULT_RECONSTRUCTION_REGISTRY.list_info(modality="sim")
    assert infos
    assert infos[0].name == "sim"


def test_engine_runs_deconvolution_plugin_without_calibration() -> None:
    image = ImageVolume(array=np.random.default_rng(0).random((16, 16)), axes=("Y", "X"))
    engine = ReconstructionEngine(DEFAULT_RECONSTRUCTION_REGISTRY)
    result = engine.run(
        plugin_name="deconvolution",
        image=image,
        params={"operation": "distance_map", "method": "distance_map", "threshold": 0.8},
    )

    assert result.image is not None
    assert result.image.shape == image.shape
    assert result.provenance["plugin"]["name"] == "deconvolution"
    assert "elapsed_seconds" in result.provenance["runtime"]


def test_sim_plugin_requires_explicit_calibration_objects() -> None:
    image = ImageVolume(array=np.ones((2, 8, 8), dtype=np.float32), axes=("Z", "Y", "X"))
    engine = ReconstructionEngine(DEFAULT_RECONSTRUCTION_REGISTRY)
    with pytest.raises(ValueError, match="requires calibration objects"):
        engine.run(plugin_name="sim", image=image, params={})


def test_sim_plugin_runs_with_explicit_otf_and_pattern_calibration() -> None:
    rng = np.random.default_rng(0)
    image = ImageVolume(array=rng.random((3, 16, 16)), axes=("C", "Y", "X"))
    calibrations = {
        "otf": CalibrationArtifact(
            name="otf",
            kind="otf",
            source="synthetic",
            payload=np.ones((16, 16), dtype=float),
            metadata={},
        ),
        "sim_pattern": CalibrationArtifact(
            name="sim_pattern",
            kind="sim_pattern",
            source="synthetic",
            payload={"angles": [0, 60, 120]},
            metadata={"pattern_count": 3},
        ),
    }
    engine = ReconstructionEngine(DEFAULT_RECONSTRUCTION_REGISTRY)
    result = engine.run(
        plugin_name="sim",
        image=image,
        params={"combine_patterns": True, "wiener_balance": 0.02},
        calibrations=calibrations,
    )
    assert result.image is not None
    assert result.image.ndim == 2
    assert "high_frequency_energy_fraction" in result.qc


def test_sted_plugin_runs_with_psf_calibration() -> None:
    image = ImageVolume(array=np.random.default_rng(1).random((8, 16, 16)), axes=("Z", "Y", "X"))
    z, y, x = np.indices((7, 7, 7), dtype=float)
    center = 3.0
    psf = np.exp(-((z - center) ** 2 + (y - center) ** 2 + (x - center) ** 2) / (2.0 * 1.0**2))
    calibrations = {
        "psf": CalibrationArtifact(
            name="psf",
            kind="psf",
            source="synthetic",
            payload=psf,
            metadata={},
        )
    }
    engine = ReconstructionEngine(DEFAULT_RECONSTRUCTION_REGISTRY)
    result = engine.run(
        plugin_name="sted",
        image=image,
        params={"iterations": 3, "regularization": 0.0},
        calibrations=calibrations,
    )
    assert result.image is not None
    assert result.image.shape == image.shape
    assert "psf_fwhm" in result.qc


def test_smlm_plugin_outputs_localization_table() -> None:
    data = np.zeros((16, 16), dtype=np.float32)
    data[4, 4] = 10.0
    data[8, 7] = 12.0
    image = ImageVolume(array=data, axes=("Y", "X"))
    engine = ReconstructionEngine(DEFAULT_RECONSTRUCTION_REGISTRY)
    calibrations = {
        "smlm_psf_model": CalibrationArtifact(
            name="smlm_psf_model",
            kind="smlm_psf_model",
            source="synthetic",
            payload={"model": "gaussian"},
        )
    }
    result = engine.run(
        plugin_name="smlm",
        image=image,
        params={"threshold": 9.0},
        calibrations=calibrations,
    )

    assert result.image is not None
    assert "localizations" in result.tables
    assert len(result.tables["localizations"]) >= 2
    assert result.qc["localization_count"] >= 2
