# processing_widget.py
# This module defines the image processing widget using magicgui.
# magicgui automatically creates a GUI from the function signature.

from magicgui import magic_factory
from napari.layers import Image
from napari.viewer import Viewer
import numpy as np

# Import image processing functions with fallbacks
try:
    from scipy.ndimage import gaussian_filter, median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage import filters, restoration, morphology, exposure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Basic implementations for when libraries aren't available
def basic_gaussian_filter(image, sigma=1.0):
    """Basic Gaussian filter implementation"""
    if SCIPY_AVAILABLE:
        return gaussian_filter(image, sigma=sigma)
    else:
        # Simple approximation using repeated averaging
        result = image.copy().astype(np.float64)
        for _ in range(int(sigma * 2)):
            kernel = np.ones((3, 3)) / 9
            if result.ndim == 2:
                from scipy.ndimage import convolve
                result = convolve(result, kernel, mode='reflect')
        return result.astype(image.dtype)

def basic_median_filter(image, size=3):
    """Basic median filter implementation"""
    if SCIPY_AVAILABLE:
        return median_filter(image, size=size)
    else:
        # Simple median implementation for 2D images
        result = image.copy()
        pad = size // 2
        for i in range(pad, image.shape[0] - pad):
            for j in range(pad, image.shape[1] - pad):
                neighborhood = image[i-pad:i+pad+1, j-pad:j+pad+1]
                result[i, j] = np.median(neighborhood)
        return result

# The @magic_factory decorator turns this function into a Napari-compatible widget.
# The type hints (e.g., `layer: Image`) are crucial, as they tell magicgui
# what kind of input to expect and create the appropriate GUI element (e.g., a dropdown menu for layers).
@magic_factory(
    sigma={'label': 'Filter Sigma', 'min': 0.1, 'max': 20.0, 'step': 0.1},
    call_button="Apply Gaussian Filter"
)
def gaussian_filter_widget(layer: Image, sigma: float = 2.0) -> Image:
    """Applies a Gaussian filter to the selected image layer."""
    if layer is None:
        return
    print(f"Applying Gaussian filter with sigma={sigma} to '{layer.name}'...")
    
    try:
        filtered_data = basic_gaussian_filter(layer.data, sigma=sigma)
        
        # Create a new layer with the filtered data
        new_layer = Image(filtered_data, name=f"{layer.name}_gauss_{sigma}", scale=layer.scale)
        return new_layer
    except Exception as e:
        print(f"Error applying Gaussian filter: {e}")
        return None

@magic_factory(
    size={'label': 'Filter Size', 'min': 3, 'max': 21, 'step': 2},
    call_button="Apply Median Filter"
)
def median_filter_widget(layer: Image, size: int = 3) -> Image:
    """Applies a Median filter to the selected image layer."""
    if layer is None:
        return
    print(f"Applying Median filter with size={size} to '{layer.name}'...")
    
    try:
        filtered_data = basic_median_filter(layer.data, size=size)
        new_layer = Image(filtered_data, name=f"{layer.name}_median_{size}", scale=layer.scale)
        return new_layer
    except Exception as e:
        print(f"Error applying median filter: {e}")
        return None

@magic_factory(
    method={'choices': ['otsu', 'li', 'yen', 'triangle'], 'label': 'Threshold Method'},
    call_button="Apply Threshold"
)
def threshold_widget(layer: Image, method: str = 'otsu') -> Image:
    """Applies automatic thresholding to create a binary image."""
    if layer is None:
        return
    print(f"Applying {method} threshold to '{layer.name}'...")
    
    try:
        if SKIMAGE_AVAILABLE:
            if method == 'otsu':
                threshold = filters.threshold_otsu(layer.data)
            elif method == 'li':
                threshold = filters.threshold_li(layer.data)
            elif method == 'yen':
                threshold = filters.threshold_yen(layer.data)
            elif method == 'triangle':
                threshold = filters.threshold_triangle(layer.data)
        else:
            # Simple Otsu approximation
            hist, bins = np.histogram(layer.data.flatten(), bins=256)
            threshold = np.mean(layer.data) + np.std(layer.data) * 0.5
        
        binary_data = (layer.data > threshold).astype(np.uint8) * 255
        new_layer = Image(binary_data, name=f"{layer.name}_thresh_{method}", scale=layer.scale)
        return new_layer
    except Exception as e:
        print(f"Error applying threshold: {e}")
        return None

@magic_factory(
    gamma={'label': 'Gamma', 'min': 0.1, 'max': 3.0, 'step': 0.1},
    call_button="Adjust Gamma"
)
def gamma_correction_widget(layer: Image, gamma: float = 1.0) -> Image:
    """Applies gamma correction to adjust image brightness."""
    if layer is None:
        return
    print(f"Applying gamma correction (γ={gamma}) to '{layer.name}'...")
    
    try:
        if SKIMAGE_AVAILABLE:
            corrected_data = exposure.adjust_gamma(layer.data, gamma)
        else:
            # Basic gamma correction
            normalized = layer.data.astype(np.float64) / np.max(layer.data)
            corrected = np.power(normalized, gamma)
            corrected_data = (corrected * np.max(layer.data)).astype(layer.data.dtype)
        
        new_layer = Image(corrected_data, name=f"{layer.name}_gamma_{gamma}", scale=layer.scale)
        return new_layer
    except Exception as e:
        print(f"Error applying gamma correction: {e}")
        return None

@magic_factory(
    method={'choices': ['unsharp_mask', 'wiener'], 'label': 'Deconvolution Method'},
    radius={'label': 'Radius', 'min': 0.5, 'max': 10.0, 'step': 0.5},
    amount={'label': 'Amount', 'min': 0.1, 'max': 3.0, 'step': 0.1},
    call_button="Apply Deconvolution"
)
def deconvolution_widget(layer: Image, method: str = 'unsharp_mask', radius: float = 1.0, amount: float = 1.0) -> Image:
    """Applies deconvolution/sharpening to enhance image details."""
    if layer is None:
        return
    print(f"Applying {method} deconvolution to '{layer.name}'...")
    
    try:
        if SKIMAGE_AVAILABLE and method == 'unsharp_mask':
            enhanced_data = filters.unsharp_mask(layer.data, radius=radius, amount=amount)
        elif SKIMAGE_AVAILABLE and method == 'wiener':
            # Simple Wiener filter approximation
            psf = np.ones((int(radius*2)+1, int(radius*2)+1)) / ((int(radius*2)+1)**2)
            enhanced_data = restoration.wiener(layer.data, psf, balance=0.1)
        else:
            # Basic unsharp mask implementation
            blurred = basic_gaussian_filter(layer.data, sigma=radius)
            mask = layer.data.astype(np.float64) - blurred.astype(np.float64)
            enhanced_data = layer.data.astype(np.float64) + amount * mask
            enhanced_data = np.clip(enhanced_data, 0, np.max(layer.data)).astype(layer.data.dtype)
        
        new_layer = Image(enhanced_data, name=f"{layer.name}_deconv_{method}", scale=layer.scale)
        return new_layer
    except Exception as e:
        print(f"Error applying deconvolution: {e}")
        return None

@magic_factory(
    operation={'choices': ['erosion', 'dilation', 'opening', 'closing'], 'label': 'Operation'},
    footprint_size={'label': 'Footprint Size', 'min': 3, 'max': 15, 'step': 2},
    call_button="Apply Morphology"
)
def morphology_widget(layer: Image, operation: str = 'opening', footprint_size: int = 3) -> Image:
    """Applies morphological operations for noise removal and shape analysis."""
    if layer is None:
        return
    print(f"Applying {operation} morphology to '{layer.name}'...")
    
    try:
        if SKIMAGE_AVAILABLE:
            footprint = morphology.disk(footprint_size // 2)
            if operation == 'erosion':
                result_data = morphology.erosion(layer.data, footprint)
            elif operation == 'dilation':
                result_data = morphology.dilation(layer.data, footprint)
            elif operation == 'opening':
                result_data = morphology.opening(layer.data, footprint)
            elif operation == 'closing':
                result_data = morphology.closing(layer.data, footprint)
        else:
            # Basic morphological operations
            from scipy.ndimage import binary_erosion, binary_dilation
            binary_image = layer.data > np.mean(layer.data)
            footprint = np.ones((footprint_size, footprint_size))
            
            if operation == 'erosion':
                result_binary = binary_erosion(binary_image, structure=footprint)
            elif operation == 'dilation':
                result_binary = binary_dilation(binary_image, structure=footprint)
            elif operation == 'opening':
                eroded = binary_erosion(binary_image, structure=footprint)
                result_binary = binary_dilation(eroded, structure=footprint)
            elif operation == 'closing':
                dilated = binary_dilation(binary_image, structure=footprint)
                result_binary = binary_erosion(dilated, structure=footprint)
            
            result_data = result_binary.astype(layer.data.dtype) * np.max(layer.data)
        
        new_layer = Image(result_data, name=f"{layer.name}_{operation}", scale=layer.scale)
        return new_layer
    except Exception as e:
        print(f"Error applying morphology: {e}")
        return None

# We combine our widgets into one for easier docking
from magicgui.widgets import Container

def processing_widget() -> Container:
    """Create a container with all processing widgets"""
    try:
        return Container(widgets=[
            gaussian_filter_widget(),
            median_filter_widget(),
            threshold_widget(),
            gamma_correction_widget(),
            deconvolution_widget(),
            morphology_widget()
        ])
    except Exception as e:
        print(f"Error creating processing widget container: {e}")
        # Return a minimal container if there are import issues
        try:
            return Container(widgets=[
                gaussian_filter_widget(),
                median_filter_widget()
            ])
        except:
            # Last resort - create empty container
            return Container(widgets=[])