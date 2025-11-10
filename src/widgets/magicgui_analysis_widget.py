"""
MagicGUI-based Analysis Widget
Integrates functions from simple_analyzer and advanced_analysis into the napari UI.
"""
from magicgui import magicgui
import napari
from napari.layers import Image
import numpy as np

# Import functions from other modules
from src.simple_analyzer import SimpleImageAnalyzer
from src.advanced_analysis import AdvancedSegmentation

@magicgui(call_button="Run Threshold")
def simple_threshold_widget(image: Image, threshold: int = 128) -> napari.types.LabelsData:
    """Threshold an image and return a labels layer."""
    if image is None:
        return

    return SimpleImageAnalyzer.simple_threshold(image.data, threshold)

@magicgui(call_button="Run Adaptive Threshold")
def adaptive_threshold_widget(image: Image, block_size: int = 11, offset: float = 2.0) -> napari.types.LabelsData:
    """Apply adaptive thresholding to an image."""
    if image is None:
        return

    return AdvancedSegmentation.adaptive_threshold_segmentation(image.data, block_size, offset)
