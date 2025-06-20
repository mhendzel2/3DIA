# segmentation_widget.py
# This module defines the segmentation widget, replicating "Surpass" creation tools.

from magicgui import magic_factory
from napari.layers import Image, Points, Surface, Labels
from napari.types import LayerDataTuple
import numpy as np

# Import segmentation libraries with fallbacks
try:
    from skimage import measure, segmentation, morphology, filters
    from skimage.feature import blob_log, peak_local_maxima
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from scipy.ndimage import label, distance_transform_edt
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

try:
    from stardist.models import StarDist2D
    STARDIST_AVAILABLE = True
except ImportError:
    STARDIST_AVAILABLE = False

def basic_threshold_segmentation(image, threshold_factor=0.5):
    """Basic threshold-based segmentation"""
    if SKIMAGE_AVAILABLE:
        threshold = filters.threshold_otsu(image)
    else:
        threshold = np.mean(image) + threshold_factor * np.std(image)
    
    binary = image > threshold
    
    if SCIPY_AVAILABLE:
        labels, num_features = label(binary)
    else:
        # Simple connected components
        labels = np.zeros_like(binary, dtype=int)
        current_label = 1
        
        def flood_fill(y, x, label_val):
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                if (cy < 0 or cy >= binary.shape[0] or cx < 0 or cx >= binary.shape[1] or
                    labels[cy, cx] != 0 or not binary[cy, cx]):
                    continue
                labels[cy, cx] = label_val
                stack.extend([(cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)])
        
        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary[i, j] and labels[i, j] == 0:
                    flood_fill(i, j, current_label)
                    current_label += 1
    
    return labels

@magic_factory(
    image_layer={'label': 'Source Image'},
    threshold={'label': 'Intensity Threshold', 'min': 0, 'max': 65535},
    call_button="Create IsoSurface"
)
def create_isosurface(image_layer: Image, threshold: int = 100) -> LayerDataTuple:
    """Generates a mesh (surface) based on an intensity threshold."""
    if image_layer is None:
        return
        
    print(f"Creating IsoSurface at threshold {threshold}...")
    try:
        if SKIMAGE_AVAILABLE and image_layer.data.ndim == 3:
            verts, faces, _, _ = measure.marching_cubes(
                image_layer.data, level=threshold, spacing=image_layer.scale
            )
            surface_data = (verts, faces)
            name = f"{image_layer.name}_surf_{threshold}"
            metadata = {'name': name, 'colormap': 'green', 'gamma': 0.8}
            
            return (surface_data, metadata, 'surface')
        else:
            print("IsoSurface requires 3D image data and scikit-image")
            return None
    except Exception as e:
        print(f"Surface creation failed: {e}")
        return None

@magic_factory(
    image_layer={'label': 'Source Image'},
    min_spot_size={'label': 'Min Spot Size (px)', 'min': 1, 'max': 100},
    intensity_threshold={'label': 'Intensity Threshold', 'min': 0, 'max': 65535},
    call_button="Create Spots"
)
def create_spots(image_layer: Image, min_spot_size: int = 5, intensity_threshold: int = 150) -> LayerDataTuple:
    """Detects bright spots in an image."""
    if image_layer is None:
        return

    print("Detecting spots...")
    try:
        if SKIMAGE_AVAILABLE:
            min_sigma = min_spot_size / 2
            threshold = intensity_threshold / 65535.0
            
            if image_layer.data.ndim == 2:
                blobs = blob_log(
                    image_layer.data,
                    min_sigma=min_sigma,
                    max_sigma=min_sigma * 1.5,
                    num_sigma=5,
                    threshold=threshold
                )
            else:
                # 3D spot detection
                blobs = blob_log(
                    image_layer.data,
                    min_sigma=min_sigma,
                    max_sigma=min_sigma * 1.5,
                    num_sigma=5,
                    threshold=threshold
                )
        else:
            # Simple peak detection fallback
            threshold_mask = image_layer.data > intensity_threshold
            if SCIPY_AVAILABLE:
                labels_array, num_labels = label(threshold_mask)
                # Find centroids
                blobs = []
                for i in range(1, num_labels + 1):
                    coords = np.where(labels_array == i)
                    if len(coords[0]) >= min_spot_size:
                        centroid = [np.mean(coord) for coord in coords]
                        centroid.append(min_spot_size)  # Add radius
                        blobs.append(centroid)
                blobs = np.array(blobs) if blobs else np.empty((0, 3))
            else:
                blobs = np.empty((0, 3))
        
        if blobs.shape[0] == 0:
            print("No spots found.")
            return None

        points_data = blobs[:, :-1] * np.array(image_layer.scale[-blobs.shape[1]+1:])
        name = f"{image_layer.name}_spots"
        metadata = {'name': name, 'face_color': 'red', 'size': min_spot_size}
        
        return (points_data, metadata, 'points')
    except Exception as e:
        print(f"Spot detection failed: {e}")
        return None

@magic_factory(
    image_layer={'label': 'Source Image'},
    method={'choices': ['cellpose', 'stardist', 'watershed', 'threshold'], 'label': 'Segmentation Method'},
    diameter={'label': 'Cell Diameter (px)', 'min': 5, 'max': 200},
    call_button="Run Segmentation"
)
def advanced_segmentation(image_layer: Image, method: str = 'threshold', diameter: int = 30) -> LayerDataTuple:
    """Advanced segmentation using various methods."""
    if image_layer is None:
        return
    
    print(f"Running {method} segmentation...")
    
    try:
        if method == 'cellpose' and CELLPOSE_AVAILABLE:
            model = models.Cellpose(gpu=False, model_type='cyto')
            masks, flows, styles, diams = model.eval(
                image_layer.data, 
                diameter=diameter, 
                channels=[0,0]
            )
            labels_data = masks
            
        elif method == 'stardist' and STARDIST_AVAILABLE and image_layer.data.ndim == 2:
            model = StarDist2D.from_pretrained('2D_versatile_fluo')
            labels_data, details = model.predict_instances(image_layer.data)
            
        elif method == 'watershed' and SKIMAGE_AVAILABLE:
            # Watershed segmentation
            if image_layer.data.ndim == 2:
                # 2D watershed
                blurred = filters.gaussian(image_layer.data, sigma=2)
                local_maxima = peak_local_maxima(
                    blurred, 
                    min_distance=diameter//2, 
                    threshold_abs=0.3*blurred.max()
                )
                markers = np.zeros_like(blurred, dtype=int)
                for i, peak in enumerate(local_maxima):
                    markers[peak[0], peak[1]] = i + 1
                
                labels_data = segmentation.watershed(
                    -blurred, 
                    markers, 
                    mask=blurred > filters.threshold_otsu(blurred)
                )
            else:
                # 3D watershed
                from scipy.ndimage import distance_transform_edt
                binary = image_layer.data > filters.threshold_otsu(image_layer.data)
                distance = distance_transform_edt(binary)
                local_maxima = peak_local_maxima(distance, min_distance=diameter//2)
                markers = np.zeros_like(distance, dtype=int)
                for i, peak in enumerate(local_maxima):
                    markers[tuple(peak)] = i + 1
                
                labels_data = segmentation.watershed(-distance, markers, mask=binary)
        else:
            # Fallback to basic threshold segmentation
            labels_data = basic_threshold_segmentation(image_layer.data)
        
        # Remove small objects
        if SKIMAGE_AVAILABLE:
            labels_data = morphology.remove_small_objects(
                labels_data.astype(bool), 
                min_size=diameter
            ).astype(int)
            labels_data = measure.label(labels_data)
        
        name = f"{image_layer.name}_{method}_seg"
        metadata = {'name': name}
        
        return (labels_data, metadata, 'labels')
        
    except Exception as e:
        print(f"{method} segmentation failed: {e}")
        # Fallback to basic segmentation
        try:
            labels_data = basic_threshold_segmentation(image_layer.data)
            name = f"{image_layer.name}_threshold_seg"
            metadata = {'name': name}
            return (labels_data, metadata, 'labels')
        except Exception as e2:
            print(f"Fallback segmentation also failed: {e2}")
            return None

@magic_factory(
    labels_layer={'label': 'Segmented Objects'},
    min_size={'label': 'Min Object Size', 'min': 1, 'max': 10000},
    max_size={'label': 'Max Object Size', 'min': 1, 'max': 100000},
    call_button="Filter Objects"
)
def filter_objects(labels_layer: Labels, min_size: int = 50, max_size: int = 10000) -> LayerDataTuple:
    """Filter objects by size criteria."""
    if labels_layer is None:
        return
    
    print(f"Filtering objects by size: {min_size} - {max_size} pixels...")
    
    try:
        if SKIMAGE_AVAILABLE:
            # Get region properties
            props = measure.regionprops(labels_layer.data)
            
            # Create new labels with filtered objects
            filtered_labels = np.zeros_like(labels_layer.data)
            new_label = 1
            
            for prop in props:
                if min_size <= prop.area <= max_size:
                    mask = labels_layer.data == prop.label
                    filtered_labels[mask] = new_label
                    new_label += 1
        else:
            # Basic size filtering
            unique_labels = np.unique(labels_layer.data)[1:]  # Skip background
            filtered_labels = np.zeros_like(labels_layer.data)
            new_label = 1
            
            for label_val in unique_labels:
                mask = labels_layer.data == label_val
                size = np.sum(mask)
                if min_size <= size <= max_size:
                    filtered_labels[mask] = new_label
                    new_label += 1
        
        name = f"{labels_layer.name}_filtered"
        metadata = {'name': name}
        
        print(f"Kept {np.max(filtered_labels)} objects after filtering")
        return (filtered_labels, metadata, 'labels')
        
    except Exception as e:
        print(f"Object filtering failed: {e}")
        return None

@magic_factory(
    image_layer={'label': 'Source Image'},
    seed_layer={'label': 'Seed Points (optional)'},
    compactness={'label': 'Compactness', 'min': 0.001, 'max': 1.0, 'step': 0.001},
    n_segments={'label': 'Number of Segments', 'min': 10, 'max': 10000},
    call_button="SLIC Superpixels"
)
def slic_segmentation(image_layer: Image, seed_layer: Points = None, 
                     compactness: float = 0.1, n_segments: int = 100) -> LayerDataTuple:
    """SLIC superpixel segmentation."""
    if image_layer is None:
        return
    
    print(f"Running SLIC superpixel segmentation with {n_segments} segments...")
    
    try:
        if SKIMAGE_AVAILABLE:
            labels_data = segmentation.slic(
                image_layer.data,
                n_segments=n_segments,
                compactness=compactness,
                start_label=1
            )
        else:
            # Simple grid-based segmentation as fallback
            h, w = image_layer.data.shape[-2:]
            grid_h = int(np.sqrt(n_segments * h / w))
            grid_w = int(n_segments / grid_h)
            
            labels_data = np.zeros_like(image_layer.data[-2:] if image_layer.data.ndim > 2 else image_layer.data)
            
            step_h = h // grid_h
            step_w = w // grid_w
            label_val = 1
            
            for i in range(0, h, step_h):
                for j in range(0, w, step_w):
                    labels_data[i:i+step_h, j:j+step_w] = label_val
                    label_val += 1
        
        name = f"{image_layer.name}_slic"
        metadata = {'name': name}
        
        return (labels_data, metadata, 'labels')
        
    except Exception as e:
        print(f"SLIC segmentation failed: {e}")
        return None

# We combine our widgets into one for easier docking
from magicgui.widgets import Container

def segmentation_widget() -> Container:
    """Create a container with all segmentation widgets"""
    try:
        widgets = [
            advanced_segmentation(),
            create_spots(),
            filter_objects(),
            slic_segmentation()
        ]
        
        # Add 3D-specific widgets if available
        if SKIMAGE_AVAILABLE:
            widgets.append(create_isosurface())
        
        return Container(widgets=widgets)
        
    except Exception as e:
        print(f"Error creating segmentation widget container: {e}")
        # Return minimal container
        try:
            return Container(widgets=[
                advanced_segmentation(),
                create_spots()
            ])
        except:
            return Container(widgets=[])