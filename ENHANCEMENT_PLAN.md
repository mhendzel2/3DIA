# Enhancement Plan and Tool Evaluation

This document outlines the evaluation of recently proposed tools for 3D segmentation, denoising, and super-resolution, and the plan for enhancing the current software package to incorporate these capabilities.

## 1. Tool Evaluation

The following tools were evaluated for potential integration:

### u-Segment3D (Nature Methods, 2025)
*   **Concept**: Reconstructs 3D consensus segmentation from 2D segmented slices (orthogonal views) without 3D ground truth.
*   **Relevance**: High. Many datasets are acquired as Z-stacks where 2D segmentation is easier or pre-existing.
*   **Integration**:
    *   *Current Action*: Implemented a `consensus_3d_segmentation` method in `AdvancedSegmentation` class. This mimics the core concept by taking a stack of 2D labels and using 3D morphological operations to create a continuous 3D volume.
    *   *Future*: Integrate the official `DanuserLab/u-segment3D` Python package once it is available in the environment for more robust "voting" mechanisms across multiple orthogonal views.

### FAST: Real-time self-supervised denoiser (Nature Communications, 2025)
*   **Concept**: High-speed (>1000 fps) self-supervised denoising for volumetric live imaging.
*   **Relevance**: High for time-lapse/live imaging users.
*   **Integration**:
    *   *Current Action*: The `AIDenoising` class architecture allows for plug-and-play denoising methods. Currently supports Non-Local Means and Bilateral filtering.
    *   *Future*: Add a wrapper for FAST (assuming PyTorch/TensorFlow availability) to `AIDenoising`. The current architecture is ready to accept a `method='fast'` parameter once the dependency is added.

### CellSeg3D (Napari Plugin, 2025)
*   **Concept**: End-to-end 3D segmentation (WNet3D) with inference and review in Napari.
*   **Relevance**: High for user experience.
*   **Integration**:
    *   *Strategy*: Rather than reinventing the wheel, we should ensure our `SegmentationWidget` can export/import compatible formats.
    *   *Enhancement*: We are enhancing our own `AdvancedSegmentation` to handle 3D volumes natively (via `consensus_3d_segmentation` and existing `morphological_snakes`), providing a "lite" built-in alternative to installing the full CellSeg3D suite.

### Datasets and Benchmarks (DL-SMLM, Holography)
*   **Relevance**: These are primarily resources for training models or benchmarking.
*   **Action**: We can add a "Resources" section to our documentation pointing users to these datasets for training their own models if they use the future trainable components of this software.

## 2. Implementation Roadmap

### Phase 1: Morphological & Structural Enhancements (Completed)
*   **Morphological Descriptors**: Added `solidity`, `extent`, `feret_diameter_max`, `moments_hu`, `aspect_ratio`, and `roundness` to `analysis_utils.py`. This improves object discrimination capabilities.
*   **3D Consensus**: Implemented `consensus_3d_segmentation` in `src/advanced_analysis.py` to bridge 2D slice segmentation to 3D volumes.

### Phase 2: Advanced Integration (Future)
*   **Deep Learning Integration**: Add optional dependencies for `torch` or `tensorflow` to enable loading pre-trained models (like FAST or Cellpose 3D) directly within the `AdvancedAnalyzer`.
*   **Interactive 3D Review**: Enhance `SegmentationWidget` to allow users to manually correct the `consensus_3d_segmentation` output in Napari 3D view.

## 3. How to Use New Features

### Morphological Statistics
When running `calculate_object_statistics` (via the Analysis Widget or API), the results dataframe now automatically includes:
*   `solidity`: Measure of convexity.
*   `roundness` & `circularity`: Shape descriptors.
*   `aspect_ratio`: Elongation measure.
*   `moments_hu_0`...`moments_hu_6`: Invariant shape moments.

### 3D Consensus Segmentation
Use `advanced_analyzer.segmentation.consensus_3d_segmentation(label_stack)`:
```python
from advanced_analysis import advanced_analyzer
import numpy as np

# Assume labels_2d is a (Z, Y, X) numpy array where each slice Z is independently segmented
labels_3d = advanced_analyzer.segmentation.consensus_3d_segmentation(labels_2d)
```
