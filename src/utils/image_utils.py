"""
Image Utility Functions for Scientific Image Analyzer
Provides common image processing and validation utilities
"""

import numpy as np
import os
from pathlib import Path
from typing import Union, List, Tuple, Optional
import napari
from napari.layers import Image

# File format support information
SUPPORTED_FORMATS = [
    'czi', 'lif', 'nd2', 'oib', 'oif', 'tif', 'tiff', 
    'ims', 'lsm', 'stk', 'mrc', 'dm3', 'dm4'
]

# Try to import optional dependencies
try:
    from aicsimageio import AICSImage
    HAS_AICSIMAGEIO = True
except (ImportError, AttributeError):
    # AttributeError can occur due to tifffile/aicsimageio version incompatibility
    # (e.g., TIFF.RESUNIT removed in newer tifffile versions)
    HAS_AICSIMAGEIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

def get_supported_formats() -> List[str]:
    """
    Get list of supported image formats
    
    Returns:
        List of supported file extensions
    """
    formats = []
    
    if HAS_AICSIMAGEIO:
        formats.extend(['CZI', 'LIF', 'ND2', 'OIB', 'OIF', 'IMS', 'LSM'])
    
    if HAS_TIFFFILE:
        formats.extend(['TIFF', 'TIF'])
    
    # Always support basic formats
    formats.extend(['PNG', 'JPEG', 'BMP'])
    
    return sorted(list(set(formats)))

def validate_image_layer(layer) -> bool:
    """
    Validate that a layer is a valid image layer with data
    
    Args:
        layer: Napari layer to validate
        
    Returns:
        True if layer is valid, False otherwise
    """
    if layer is None:
        return False
        
    if not isinstance(layer, Image):
        return False
        
    if layer.data is None:
        return False
        
    if layer.data.size == 0:
        return False
        
    return True

def estimate_memory_usage(file_path: Union[str, Path]) -> float:
    """
    Estimate memory usage for loading an image file
    
    Args:
        file_path: Path to image file
        
    Returns:
        Estimated memory usage in MB
    """
    try:
        file_path = Path(file_path)
        
        # Get file size as baseline
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # For compressed formats, estimate decompressed size
        if file_path.suffix.lower() in ['.czi', '.lif', '.nd2']:
            # These formats are typically compressed 2-10x
            estimated_mb = file_size_mb * 5
        elif file_path.suffix.lower() in ['.tif', '.tiff']:
            # TIFF can be compressed or uncompressed
            estimated_mb = file_size_mb * 2
        else:
            # Conservative estimate
            estimated_mb = file_size_mb * 3
            
        return max(estimated_mb, file_size_mb)
        
    except Exception:
        # If estimation fails, return file size
        try:
            return Path(file_path).stat().st_size / (1024 * 1024)
        except:
            return 0

def get_image_info(file_path: Union[str, Path]) -> dict:
    """
    Get basic information about an image file
    
    Args:
        file_path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    info = {
        'file_path': str(file_path),
        'file_size_mb': 0,
        'dimensions': None,
        'shape': None,
        'dtype': None,
        'channels': 0,
        'time_points': 0,
        'z_slices': 0,
        'physical_sizes': None,
        'error': None
    }
    
    try:
        file_path = Path(file_path)
        info['file_size_mb'] = file_path.stat().st_size / (1024 * 1024)
        
        if HAS_AICSIMAGEIO and file_path.suffix.lower() in ['.czi', '.lif', '.nd2', '.oib', '.oif']:
            img = AICSImage(file_path)
            info.update({
                'dimensions': img.dims.order,
                'shape': img.shape,
                'dtype': str(img.dtype),
                'channels': img.dims.C,
                'time_points': img.dims.T,
                'z_slices': img.dims.Z,
            })
            
            if hasattr(img, 'physical_pixel_sizes'):
                info['physical_sizes'] = {
                    'X': img.physical_pixel_sizes.X,
                    'Y': img.physical_pixel_sizes.Y,
                    'Z': img.physical_pixel_sizes.Z
                }
                
        elif HAS_TIFFFILE and file_path.suffix.lower() in ['.tif', '.tiff']:
            with tifffile.TiffFile(file_path) as tif:
                info.update({
                    'shape': tif.series[0].shape,
                    'dtype': str(tif.series[0].dtype),
                })
                
    except Exception as e:
        info['error'] = str(e)
        
    return info

def normalize_image(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize image data to 0-1 range
    
    Args:
        image: Input image array
        method: Normalization method ('minmax', 'percentile', 'zscore')
        
    Returns:
        Normalized image array
    """
    if image.size == 0:
        return image
        
    image = image.astype(np.float64)
    
    if method == 'minmax':
        img_min = np.min(image)
        img_max = np.max(image)
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        else:
            return np.zeros_like(image)
            
    elif method == 'percentile':
        p1, p99 = np.percentile(image, [1, 99])
        if p99 > p1:
            image = np.clip(image, p1, p99)
            return (image - p1) / (p99 - p1)
        else:
            return np.zeros_like(image)
            
    elif method == 'zscore':
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return (image - mean) / std
        else:
            return image - mean
            
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def resize_image(image: np.ndarray, target_shape: Tuple[int, ...], 
                method: str = 'linear') -> np.ndarray:
    """
    Resize image to target shape
    
    Args:
        image: Input image array
        target_shape: Target shape for resizing
        method: Interpolation method ('linear', 'nearest', 'cubic')
        
    Returns:
        Resized image array
    """
    try:
        from scipy import ndimage
        
        if len(target_shape) != image.ndim:
            raise ValueError("Target shape must match image dimensions")
            
        # Calculate zoom factors
        zoom_factors = [target_shape[i] / image.shape[i] for i in range(image.ndim)]
        
        # Choose interpolation order
        if method == 'nearest':
            order = 0
        elif method == 'linear':
            order = 1
        elif method == 'cubic':
            order = 3
        else:
            order = 1
            
        return ndimage.zoom(image, zoom_factors, order=order)
        
    except ImportError:
        # Fallback: simple nearest neighbor using numpy
        print("SciPy not available, using simple resizing")
        return _simple_resize(image, target_shape)

def _simple_resize(image: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Simple nearest neighbor resizing using numpy indexing
    
    Args:
        image: Input image array
        target_shape: Target shape for resizing
        
    Returns:
        Resized image array
    """
    indices = []
    for i in range(image.ndim):
        old_size = image.shape[i]
        new_size = target_shape[i]
        indices.append(np.linspace(0, old_size - 1, new_size).astype(int))
    
    # Create mesh grid for indexing
    mesh = np.meshgrid(*indices, indexing='ij')
    
    return image[tuple(mesh)]

def crop_image(image: np.ndarray, crop_bounds: List[Tuple[int, int]]) -> np.ndarray:
    """
    Crop image to specified bounds
    
    Args:
        image: Input image array
        crop_bounds: List of (start, end) tuples for each dimension
        
    Returns:
        Cropped image array
    """
    if len(crop_bounds) != image.ndim:
        raise ValueError("Crop bounds must match image dimensions")
    
    slices = []
    for i, (start, end) in enumerate(crop_bounds):
        start = max(0, min(start, image.shape[i]))
        end = max(start, min(end, image.shape[i]))
        slices.append(slice(start, end))
    
    return image[tuple(slices)]

def apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Apply lookup table to image
    
    Args:
        image: Input image array
        lut: Lookup table array
        
    Returns:
        Image with LUT applied
    """
    # Ensure image is in appropriate range for LUT
    if image.dtype != np.uint8:
        image_norm = normalize_image(image) * 255
        image_uint8 = image_norm.astype(np.uint8)
    else:
        image_uint8 = image
    
    return lut[image_uint8]

def calculate_histogram(image: np.ndarray, bins: int = 256, 
                       range_min: Optional[float] = None, 
                       range_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate image histogram
    
    Args:
        image: Input image array
        bins: Number of histogram bins
        range_min: Minimum value for histogram range
        range_max: Maximum value for histogram range
        
    Returns:
        Tuple of (histogram values, bin edges)
    """
    if range_min is None:
        range_min = np.min(image)
    if range_max is None:
        range_max = np.max(image)
    
    hist, edges = np.histogram(
        image.flatten(), 
        bins=bins, 
        range=(range_min, range_max)
    )
    
    return hist, edges

def estimate_processing_time(image_shape: Tuple[int, ...], operation: str) -> float:
    """
    Estimate processing time for common operations
    
    Args:
        image_shape: Shape of the image
        operation: Type of operation ('filter', 'threshold', 'morphology', etc.)
        
    Returns:
        Estimated processing time in seconds
    """
    # Calculate number of voxels
    n_voxels = np.prod(image_shape)
    
    # Base processing rates (voxels per second) for different operations
    # These are rough estimates and will vary by hardware
    rates = {
        'filter': 1e7,      # Gaussian filter
        'threshold': 1e8,   # Simple thresholding
        'morphology': 5e6,  # Morphological operations
        'segmentation': 1e6, # Complex segmentation
        'analysis': 1e7,    # Statistical analysis
        'io': 1e8          # File I/O operations
    }
    
    rate = rates.get(operation, 1e7)
    
    # Estimate time with some overhead
    estimated_time = (n_voxels / rate) + 0.1
    
    return max(0.1, estimated_time)  # Minimum 0.1 seconds

def create_test_image(shape: Tuple[int, ...], pattern: str = 'noise') -> np.ndarray:
    """
    Create test image for development and testing
    
    Args:
        shape: Shape of the test image
        pattern: Type of pattern ('noise', 'gradient', 'checkerboard', 'spheres')
        
    Returns:
        Test image array
    """
    if pattern == 'noise':
        return np.random.randint(0, 256, shape, dtype=np.uint8)
    
    elif pattern == 'gradient':
        if len(shape) == 2:
            x = np.linspace(0, 255, shape[1])
            y = np.linspace(0, 255, shape[0])
            X, Y = np.meshgrid(x, y)
            return ((X + Y) / 2).astype(np.uint8)
        else:
            # 3D gradient
            gradient = np.zeros(shape)
            for i in range(shape[0]):
                gradient[i] = (i / shape[0]) * 255
            return gradient.astype(np.uint8)
    
    elif pattern == 'checkerboard':
        checker = np.zeros(shape, dtype=np.uint8)
        if len(shape) == 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if (i // 20 + j // 20) % 2:
                        checker[i, j] = 255
        return checker
    
    elif pattern == 'spheres':
        image = np.zeros(shape, dtype=np.uint8)
        if len(shape) >= 2:
            # Add some circular/spherical objects
            centers = np.random.randint(20, np.array(shape) - 20, (5, len(shape)))
            for center in centers:
                if len(shape) == 2:
                    y, x = np.ogrid[:shape[0], :shape[1]]
                    mask = (x - center[1])**2 + (y - center[0])**2 <= 15**2
                    image[mask] = 255
                elif len(shape) == 3:
                    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
                    mask = ((x - center[2])**2 + (y - center[1])**2 + 
                           (z - center[0])**2) <= 10**2
                    image[mask] = 255
        return image
    
    else:
        return np.zeros(shape, dtype=np.uint8)

def convert_dtype(image: np.ndarray, target_dtype: str) -> np.ndarray:
    """
    Convert image to target data type with proper scaling
    
    Args:
        image: Input image array
        target_dtype: Target data type ('uint8', 'uint16', 'float32', 'float64')
        
    Returns:
        Converted image array
    """
    if str(image.dtype) == target_dtype:
        return image
    
    # Normalize to 0-1 range first
    if image.dtype in [np.uint8, np.uint16, np.uint32]:
        normalized = image.astype(np.float64) / np.iinfo(image.dtype).max
    else:
        # Already float, normalize to 0-1
        img_min, img_max = np.min(image), np.max(image)
        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
        else:
            normalized = np.zeros_like(image, dtype=np.float64)
    
    # Convert to target type
    if target_dtype == 'uint8':
        return (normalized * 255).astype(np.uint8)
    elif target_dtype == 'uint16':
        return (normalized * 65535).astype(np.uint16)
    elif target_dtype == 'float32':
        return normalized.astype(np.float32)
    elif target_dtype == 'float64':
        return normalized.astype(np.float64)
    else:
        raise ValueError(f"Unsupported target dtype: {target_dtype}")

def split_channels(image: np.ndarray, channel_axis: int = -1) -> List[np.ndarray]:
    """
    Split multi-channel image into separate channels
    
    Args:
        image: Multi-channel image array
        channel_axis: Axis containing channels
        
    Returns:
        List of single-channel images
    """
    if image.ndim < 2:
        return [image]
    
    channels = []
    for i in range(image.shape[channel_axis]):
        channel = np.take(image, i, axis=channel_axis)
        channels.append(channel)
    
    return channels

def merge_channels(channels: List[np.ndarray], axis: int = -1) -> np.ndarray:
    """
    Merge separate channels into multi-channel image
    
    Args:
        channels: List of single-channel images
        axis: Axis along which to merge channels
        
    Returns:
        Multi-channel image array
    """
    if len(channels) == 1:
        return channels[0]
    
    return np.stack(channels, axis=axis)
