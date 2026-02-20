"""Parity tests across reconstruction plugin modality contracts."""

from __future__ import annotations

import numpy as np

from pymaris.data_model import ImageVolume
from pymaris.reconstruction import (
    DEFAULT_RECONSTRUCTION_REGISTRY,
    CalibrationArtifact,
    ReconstructionEngine,
)


def _sim_calibrations(size: int = 16) -> dict[str, CalibrationArtifact]:
    return {
        "otf": CalibrationArtifact(
            name="otf",
            kind="otf",
            source="synthetic",
            payload=np.ones((size, size), dtype=float),
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


def _sted_calibrations(size: int = 7) -> dict[str, CalibrationArtifact]:
    z, y, x = np.indices((size, size, size), dtype=float)
    center = (size - 1) / 2.0
    psf = np.exp(-((z - center) ** 2 + (y - center) ** 2 + (x - center) ** 2) / (2.0 * 1.0**2))
    return {
        "psf": CalibrationArtifact(
            name="psf",
            kind="psf",
            source="synthetic",
            payload=psf,
            metadata={},
        )
    }


def _smlm_calibrations() -> dict[str, CalibrationArtifact]:
    return {
        "smlm_psf_model": CalibrationArtifact(
            name="smlm_psf_model",
            kind="smlm_psf_model",
            source="synthetic",
            payload={"model": "gaussian", "sigma": 1.0},
            metadata={},
        )
    }


def test_deconvolution_plugin_parity_for_2d_and_3d() -> None:
    engine = ReconstructionEngine(DEFAULT_RECONSTRUCTION_REGISTRY)

    image_2d = ImageVolume(array=np.random.default_rng(0).random((24, 24)), axes=("Y", "X"))
    result_2d = engine.run(
        plugin_name="deconvolution",
        image=image_2d,
        params={"operation": "distance_map", "method": "distance_map", "threshold": 0.8},
    )

    image_3d = ImageVolume(array=np.random.default_rng(1).random((8, 24, 24)), axes=("Z", "Y", "X"))
    result_3d = engine.run(
        plugin_name="deconvolution",
        image=image_3d,
        params={"operation": "distance_map", "method": "distance_map", "threshold": 0.8},
    )

    assert result_2d.image is not None and result_2d.image.shape == image_2d.shape
    assert result_3d.image is not None and result_3d.image.shape == image_3d.shape


def test_sim_plugin_parity_for_channel_and_time_dims() -> None:
    engine = ReconstructionEngine(DEFAULT_RECONSTRUCTION_REGISTRY)
    calibrations = _sim_calibrations(size=16)

    image_czyx = ImageVolume(array=np.random.default_rng(2).random((3, 16, 16)), axes=("C", "Y", "X"))
    result_czyx = engine.run(
        plugin_name="sim",
        image=image_czyx,
        params={"combine_patterns": True, "wiener_balance": 0.02},
        calibrations=calibrations,
    )

    image_tczyx = ImageVolume(
        array=np.random.default_rng(3).random((2, 3, 16, 16)),
        axes=("T", "C", "Y", "X"),
    )
    result_tczyx = engine.run(
        plugin_name="sim",
        image=image_tczyx,
        params={"combine_patterns": True, "wiener_balance": 0.02},
        calibrations=calibrations,
    )

    assert result_czyx.image is not None
    assert result_tczyx.image is not None
    assert "high_frequency_energy_fraction" in result_czyx.qc
    assert "high_frequency_energy_fraction" in result_tczyx.qc


def test_sted_plugin_parity_for_3d_and_time_series() -> None:
    engine = ReconstructionEngine(DEFAULT_RECONSTRUCTION_REGISTRY)
    calibrations = _sted_calibrations(size=7)

    image_3d = ImageVolume(array=np.random.default_rng(4).random((8, 16, 16)), axes=("Z", "Y", "X"))
    result_3d = engine.run(
        plugin_name="sted",
        image=image_3d,
        params={"iterations": 2, "regularization": 0.0},
        calibrations=calibrations,
    )

    image_tzyx = ImageVolume(
        array=np.random.default_rng(5).random((2, 8, 16, 16)),
        axes=("T", "Z", "Y", "X"),
    )
    result_tzyx = engine.run(
        plugin_name="sted",
        image=image_tzyx,
        params={"iterations": 2, "regularization": 0.0},
        calibrations=calibrations,
    )

    assert result_3d.image is not None and result_3d.image.shape == image_3d.shape
    assert result_tzyx.image is not None and result_tzyx.image.shape == image_tzyx.shape
    assert "psf_fwhm" in result_3d.qc
    assert "psf_fwhm" in result_tzyx.qc


def test_smlm_plugin_parity_for_2d_and_time_series_with_drift() -> None:
    engine = ReconstructionEngine(DEFAULT_RECONSTRUCTION_REGISTRY)
    calibrations = _smlm_calibrations()

    image_2d = np.zeros((20, 20), dtype=np.float32)
    image_2d[5, 6] = 10.0
    image_2d[11, 12] = 12.0
    result_2d = engine.run(
        plugin_name="smlm",
        image=ImageVolume(array=image_2d, axes=("Y", "X")),
        params={"threshold": 8.0},
        calibrations=calibrations,
    )

    image_tyx = np.zeros((3, 20, 20), dtype=np.float32)
    image_tyx[0, 5, 6] = 10.0
    image_tyx[1, 6, 7] = 12.0
    image_tyx[2, 7, 8] = 14.0
    result_tyx = engine.run(
        plugin_name="smlm",
        image=ImageVolume(array=image_tyx, axes=("T", "Y", "X")),
        params={"threshold": 8.0, "drift_correction": "centroid"},
        calibrations=calibrations,
    )

    assert result_2d.image is not None and result_2d.image.shape == image_2d.shape
    assert result_tyx.image is not None and result_tyx.image.shape == image_tyx.shape
    assert "localizations" in result_2d.tables
    assert "localizations" in result_tyx.tables
    assert result_tyx.metadata["drift"]["mode"] == "centroid"
