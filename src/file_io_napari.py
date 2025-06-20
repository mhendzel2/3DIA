# file_io_napari.py
# This module provides a "reader plugin" for Napari. The @napari_get_reader
# decorator allows Napari's file opening mechanism to use this function
# when it encounters a file with a matching extension.

import napari
from napari.types import LayerDataTuple
from pathlib import Path
import numpy as np

# Try to import specialized image readers
try:
    from aicsimageio import AICSImage
    AICSIMAGEIO_AVAILABLE = True
except ImportError:
    AICSIMAGEIO_AVAILABLE = False

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

try:
    from skimage import io as skio
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

@napari.hook_implementation
def napari_get_reader(path):
    """
    A Napari reader hook that supports various microscopy formats.
    
    This function is called by Napari when a file is opened. It checks if
    the file format is supported and, if so, returns the actual reader function.
    """
    if isinstance(path, str) and any(path.endswith(ext) for ext in ['.czi', '.lif', '.nd2', '.oib', '.tif', '.tiff']):
        return read_microscopy_file
    return None

def read_microscopy_file(path):
    """
    Reads a multi-dimensional image file and returns it as a Napari layer.
    
    Args:
        path (str): The path to the image file.
        
    Returns:
        LayerDataTuple: A tuple containing (data, metadata, layer_type).
    """
    try:
        # Try different readers in order of preference
        if AICSIMAGEIO_AVAILABLE and any(path.endswith(ext) for ext in ['.czi', '.lif', '.nd2', '.oib']):
            return read_with_aicsimageio(path)
        elif TIFFFILE_AVAILABLE and path.endswith(('.tif', '.tiff')):
            return read_with_tifffile(path)
        elif SKIMAGE_AVAILABLE:
            return read_with_skimage(path)
        else:
            return read_basic(path)
            
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def read_with_aicsimageio(path):
    """Read using AICSImageIO for specialized microscopy formats"""
    img = AICSImage(path)
    data = img.get_image_data("TCZYX", S=0).squeeze()

    scale = [
        img.physical_pixel_sizes.Z if img.physical_pixel_sizes.Z is not None else 1.0,
        img.physical_pixel_sizes.Y if img.physical_pixel_sizes.Y is not None else 1.0,
        img.physical_pixel_sizes.X if img.physical_pixel_sizes.X is not None else 1.0,
    ]
    
    if 'T' in img.dims.order:
        scale.insert(0, 1.0)  # Placeholder time scale
    
    # Prepare metadata for the Napari layer
    metadata = {
        'name': Path(path).name,
        'scale': scale[-data.ndim:],
        'metadata': {
            'channel_names': img.channel_names,
            'dims': img.dims.order
        }
    }
    
    return [(data, metadata, 'image')]

def read_with_tifffile(path):
    """Read using tifffile for TIFF images"""
    data = tifffile.imread(path)
    
    # Handle different TIFF structures
    if data.ndim > 2:
        # Multi-dimensional TIFF
        metadata = {
            'name': Path(path).name,
            'scale': [1.0] * data.ndim
        }
    else:
        # Simple 2D TIFF
        metadata = {
            'name': Path(path).name,
            'scale': [1.0, 1.0]
        }
    
    return [(data, metadata, 'image')]

def read_with_skimage(path):
    """Read using scikit-image"""
    data = skio.imread(path)
    
    metadata = {
        'name': Path(path).name,
        'scale': [1.0] * data.ndim if data.ndim > 1 else [1.0, 1.0]
    }
    
    return [(data, metadata, 'image')]

def read_basic(path):
    """Basic fallback reader using PIL/Pillow"""
    try:
        from PIL import Image
        img = Image.open(path)
        data = np.array(img)
        
        metadata = {
            'name': Path(path).name,
            'scale': [1.0, 1.0] if data.ndim == 2 else [1.0] * data.ndim
        }
        
        return [(data, metadata, 'image')]
    except Exception as e:
        print(f"Basic reader failed: {e}")
        return None