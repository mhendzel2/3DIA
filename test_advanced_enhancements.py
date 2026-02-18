#!/usr/bin/env python3
"""Advanced feature regression tests."""

from __future__ import annotations

import sys

import numpy as np

sys.path.append("src")


def test_advanced_processing_methods() -> None:
    from widgets.processing_widget import ADVANCED_DENOISING_AVAILABLE, ProcessingWidget

    assert ProcessingWidget is not None
    assert isinstance(ADVANCED_DENOISING_AVAILABLE, bool)

    if ADVANCED_DENOISING_AVAILABLE:
        from advanced_analysis import AIDenoising

        test_image = np.random.rand(100, 100)
        bilateral = AIDenoising.bilateral_filter_denoising(test_image, sigma_spatial=5.0)
        non_local_means = AIDenoising.non_local_means_denoising(test_image, h=10.0)
        assert bilateral.shape == test_image.shape
        assert non_local_means.shape == test_image.shape


def test_enhanced_morphological_statistics() -> None:
    from utils.analysis_utils import calculate_object_statistics

    labels = np.zeros((100, 100), dtype=int)
    labels[20:40, 20:40] = 1
    labels[60:80, 60:70] = 2

    stats = calculate_object_statistics(labels)
    assert stats["object_count"] == 2
    assert "aspect_ratio" in stats
    assert "roundness" in stats
    assert "moments_hu_0" in stats


def test_colocalization_enhancements() -> None:
    from matplotlib.figure import Figure
    from scipy import stats

    pixels1 = np.random.rand(1000)
    pixels2 = np.random.rand(1000)

    fig = Figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist2d(pixels1, pixels2, bins=50)

    slope, intercept, r_value, _, _ = stats.linregress(pixels1, pixels2)
    line = slope * pixels1 + intercept
    ax1.plot(pixels1, line, "r-", linewidth=1)

    assert np.isfinite(slope)
    assert np.isfinite(intercept)
    assert np.isfinite(r_value)


def test_spatial_distribution_analysis() -> None:
    from utils.analysis_utils import analyze_spatial_distribution

    coordinates = np.random.rand(50, 2) * 100
    stats = analyze_spatial_distribution(coordinates)
    assert "mean_distance" in stats
    assert "mean_nn_distance" in stats
    assert "density" in stats


def test_adaptive_thresholding() -> None:
    from skimage.filters import threshold_local

    test_image = np.random.rand(100, 100)
    threshold = threshold_local(test_image, block_size=35, offset=10)
    binary = test_image > threshold
    assert binary.shape == test_image.shape
    assert binary.dtype == bool


def test_consensus_3d_segmentation() -> None:
    from advanced_analysis import advanced_analyzer

    stack = np.zeros((5, 20, 20), dtype=int)
    stack[1, 5:15, 5:15] = 1
    stack[2, 5:15, 5:15] = 1
    stack[2, 8:12, 8:12] = 0
    stack[3, 5:15, 5:15] = 1
    stack[4, 18:19, 18:19] = 2

    refined = advanced_analyzer.segmentation.consensus_3d_segmentation(
        stack,
        filter_size=3,
        min_object_size=5,
    )

    assert refined.shape == stack.shape
    # small one-voxel noise object should be removed
    assert refined[4, 18, 18] == 0
