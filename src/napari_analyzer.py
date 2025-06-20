"""
Scientific Image Analyzer - Napari-based Implementation
Custom microscopy analysis suite built on top of Napari ecosystem
"""

import os
import sys
from pathlib import Path

# Try to import Napari and related packages
try:
    import napari
    from napari.types import ImageData, LabelsData
    from napari.layers import Image, Labels
    NAPARI_AVAILABLE = True
    print("✓ Napari successfully imported")
except ImportError as e:
    NAPARI_AVAILABLE = False
    print(f"✗ Napari not available: {e}")
    print("Run: pip install 'napari[all]' to install Napari with all dependencies")

# Scientific computing imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("✗ NumPy not available")

try:
    from skimage import measure, segmentation, filters, morphology
    from skimage.feature import peak_local_maxima
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("✗ scikit-image not available")

try:
    from scipy import ndimage
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("✗ SciPy not available")

# Napari plugin imports (these extend Napari's functionality)
CELLPOSE_AVAILABLE = False
STARDIST_AVAILABLE = False
BTRACK_AVAILABLE = False

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
    print("✓ Cellpose available")
except ImportError:
    print("○ Cellpose not available - install with: pip install cellpose")

try:
    from stardist.models import StarDist2D
    STARDIST_AVAILABLE = True
    print("✓ StarDist available")
except ImportError:
    print("○ StarDist not available - install with: pip install stardist")

try:
    from btrack import BayesianTracker
    BTRACK_AVAILABLE = True
    print("✓ BTrack available")
except ImportError:
    print("○ BTrack not available - install with: pip install btrack")

class CustomNapariAnalyzer:
    """
    Main analyzer class that extends Napari with custom functionality
    """
    
    def __init__(self):
        self.viewer = None
        self.current_image = None
        self.current_labels = None
        self.analysis_results = {}
        
    def initialize_viewer(self):
        """Initialize Napari viewer with custom configuration"""
        if not NAPARI_AVAILABLE:
            raise RuntimeError("Napari is not available. Please install with: pip install 'napari[all]'")
        
        # Create viewer with custom settings
        self.viewer = napari.Viewer(
            title="Scientific Image Analyzer - Custom Napari Implementation",
            show=True
        )
        
        # Add custom keybindings
        self._setup_keybindings()
        
        # Add custom widgets to the viewer
        self._setup_custom_widgets()
        
        print("Napari viewer initialized with custom configuration")
        return self.viewer
    
    def _setup_keybindings(self):
        """Setup custom keyboard shortcuts"""
        if not self.viewer:
            return
            
        # Custom shortcuts for analysis functions
        @self.viewer.bind_key('c')
        def cellpose_segmentation(viewer):
            self.run_cellpose_segmentation()
        
        @self.viewer.bind_key('s') 
        def stardist_segmentation(viewer):
            self.run_stardist_segmentation()
        
        @self.viewer.bind_key('m')
        def measure_objects(viewer):
            self.measure_objects()
        
        @self.viewer.bind_key('t')
        def track_objects(viewer):
            self.track_objects()
        
        print("Custom keybindings added:")
        print("  C - Cellpose segmentation")
        print("  S - StarDist segmentation") 
        print("  M - Measure objects")
        print("  T - Track objects")
    
    def _setup_custom_widgets(self):
        """Add custom analysis widgets to Napari"""
        if not self.viewer:
            return
            
        # Custom analysis widget
        from magicgui import magic_factory
        
        @magic_factory(call_button="Run Custom Analysis")
        def custom_analysis_widget(
            method: str = "cellpose",
            diameter: float = 30.0,
            threshold: float = 0.5
        ):
            """Custom analysis widget for the Napari viewer"""
            if method == "cellpose":
                self.run_cellpose_segmentation(diameter=diameter)
            elif method == "stardist":
                self.run_stardist_segmentation(threshold=threshold)
            elif method == "watershed":
                self.run_watershed_segmentation()
        
        # Add widget to viewer
        self.viewer.window.add_dock_widget(
            custom_analysis_widget, 
            name="Custom Analysis",
            area="right"
        )
    
    def load_image(self, image_path):
        """Load image into Napari viewer"""
        if not NAPARI_AVAILABLE or not NUMPY_AVAILABLE:
            print("Required packages not available for image loading")
            return None
            
        try:
            # Load image using skimage or other method
            if SKIMAGE_AVAILABLE:
                from skimage import io
                image = io.imread(image_path)
            else:
                # Fallback to basic loading
                import imageio
                image = imageio.imread(image_path)
            
            # Add to viewer
            self.current_image = image
            layer = self.viewer.add_image(image, name=f"Image: {Path(image_path).name}")
            
            print(f"Loaded image: {image_path}")
            print(f"Shape: {image.shape}, Type: {image.dtype}")
            
            return layer
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def create_test_image(self):
        """Create synthetic test image for demonstration"""
        if not NUMPY_AVAILABLE:
            print("NumPy required for test image creation")
            return None
        
        # Create synthetic microscopy image
        image = np.zeros((512, 512), dtype=np.uint16)
        
        # Add some cell-like objects
        centers = [(100, 100), (200, 150), (350, 200), (400, 400), (150, 350)]
        
        for center in centers:
            y, x = np.ogrid[:512, :512]
            mask = (x - center[1])**2 + (y - center[0])**2 <= (30 + np.random.randint(-10, 10))**2
            image[mask] = np.random.randint(8000, 16000)
        
        # Add noise
        noise = np.random.normal(1000, 200, image.shape).astype(np.uint16)
        image = np.clip(image + noise, 0, 65535)
        
        # Add to viewer
        self.current_image = image
        layer = self.viewer.add_image(image, name="Test Image")
        
        print("Created synthetic test image")
        return layer
    
    def run_cellpose_segmentation(self, diameter=30.0):
        """Run Cellpose segmentation on current image"""
        if not self.current_image is not None:
            print("No image loaded")
            return None
        
        if CELLPOSE_AVAILABLE:
            try:
                # Use actual Cellpose
                model = models.Cellpose(gpu=False, model_type='cyto')
                masks, flows, styles, diams = model.eval(
                    self.current_image, 
                    diameter=diameter, 
                    channels=[0,0]
                )
                
                # Add labels to viewer
                labels_layer = self.viewer.add_labels(masks, name="Cellpose Segmentation")
                self.current_labels = masks
                
                print(f"Cellpose segmentation completed. Found {len(np.unique(masks))-1} objects")
                return labels_layer
                
            except Exception as e:
                print(f"Cellpose error: {e}")
                return None
        else:
            print("Cellpose not available. Install with: pip install cellpose")
            # Fallback to simple segmentation
            return self._fallback_segmentation("cellpose")
    
    def run_stardist_segmentation(self, threshold=0.5):
        """Run StarDist segmentation on current image"""
        if self.current_image is None:
            print("No image loaded")
            return None
        
        if STARDIST_AVAILABLE:
            try:
                # Use actual StarDist
                model = StarDist2D.from_pretrained('2D_versatile_fluo')
                labels, details = model.predict_instances(self.current_image)
                
                # Add labels to viewer
                labels_layer = self.viewer.add_labels(labels, name="StarDist Segmentation")
                self.current_labels = labels
                
                print(f"StarDist segmentation completed. Found {len(np.unique(labels))-1} objects")
                return labels_layer
                
            except Exception as e:
                print(f"StarDist error: {e}")
                return None
        else:
            print("StarDist not available. Install with: pip install stardist")
            # Fallback to simple segmentation
            return self._fallback_segmentation("stardist")
    
    def run_watershed_segmentation(self):
        """Run watershed segmentation using scikit-image"""
        if self.current_image is None:
            print("No image loaded")
            return None
        
        if not SKIMAGE_AVAILABLE:
            print("scikit-image required for watershed segmentation")
            return None
        
        try:
            # Watershed segmentation
            from skimage.segmentation import watershed
            from skimage.feature import peak_local_maxima
            
            # Pre-process image
            blurred = filters.gaussian(self.current_image, sigma=2)
            
            # Find local maxima as seeds
            local_maxima = peak_local_maxima(blurred, min_distance=20, threshold_abs=0.3*blurred.max())
            markers = np.zeros_like(blurred, dtype=int)
            markers[tuple(local_maxima.T)] = np.arange(1, len(local_maxima) + 1)
            
            # Run watershed
            labels = watershed(-blurred, markers, mask=blurred > filters.threshold_otsu(blurred))
            
            # Add to viewer
            labels_layer = self.viewer.add_labels(labels, name="Watershed Segmentation")
            self.current_labels = labels
            
            print(f"Watershed segmentation completed. Found {len(np.unique(labels))-1} objects")
            return labels_layer
            
        except Exception as e:
            print(f"Watershed error: {e}")
            return None
    
    def _fallback_segmentation(self, method="simple"):
        """Fallback segmentation when specialized tools aren't available"""
        if not SKIMAGE_AVAILABLE:
            print("scikit-image required for fallback segmentation")
            return None
        
        try:
            # Simple threshold-based segmentation
            from skimage import measure
            
            threshold = filters.threshold_otsu(self.current_image)
            binary = self.current_image > threshold
            
            # Label connected components
            labels = measure.label(binary)
            
            # Add to viewer
            labels_layer = self.viewer.add_labels(labels, name=f"Fallback {method.title()}")
            self.current_labels = labels
            
            print(f"Fallback segmentation completed. Found {len(np.unique(labels))-1} objects")
            return labels_layer
            
        except Exception as e:
            print(f"Fallback segmentation error: {e}")
            return None
    
    def measure_objects(self):
        """Measure properties of segmented objects"""
        if self.current_labels is None:
            print("No segmentation available. Run segmentation first.")
            return None
        
        if not SKIMAGE_AVAILABLE:
            print("scikit-image required for measurements")
            return None
        
        try:
            # Calculate region properties
            props = measure.regionprops(self.current_labels, intensity_image=self.current_image)
            
            # Extract measurements
            measurements = []
            for prop in props:
                measurements.append({
                    'label': prop.label,
                    'area': prop.area,
                    'centroid': prop.centroid,
                    'perimeter': prop.perimeter,
                    'eccentricity': prop.eccentricity,
                    'mean_intensity': prop.mean_intensity,
                    'max_intensity': prop.max_intensity,
                    'bbox': prop.bbox
                })
            
            self.analysis_results['measurements'] = measurements
            
            print(f"Measured {len(measurements)} objects:")
            for i, m in enumerate(measurements[:5]):  # Show first 5
                print(f"  Object {m['label']}: Area={m['area']:.1f}, Mean Intensity={m['mean_intensity']:.1f}")
            
            if len(measurements) > 5:
                print(f"  ... and {len(measurements)-5} more objects")
            
            return measurements
            
        except Exception as e:
            print(f"Measurement error: {e}")
            return None
    
    def track_objects(self):
        """Track objects across time series (if available)"""
        if BTRACK_AVAILABLE:
            print("BTrack tracking functionality would be implemented here")
            # Actual BTrack implementation would go here
        else:
            print("BTrack not available. Install with: pip install btrack")
            print("Simulating tracking analysis...")
        
        return None
    
    def export_results(self, output_path):
        """Export analysis results to file"""
        if not self.analysis_results:
            print("No analysis results to export")
            return
        
        try:
            import json
            with open(output_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                exportable_results = {}
                for key, value in self.analysis_results.items():
                    if isinstance(value, list):
                        exportable_results[key] = []
                        for item in value:
                            if isinstance(item, dict):
                                clean_item = {}
                                for k, v in item.items():
                                    if isinstance(v, np.ndarray):
                                        clean_item[k] = v.tolist()
                                    elif isinstance(v, (np.integer, np.floating)):
                                        clean_item[k] = float(v)
                                    else:
                                        clean_item[k] = v
                                exportable_results[key].append(clean_item)
                    else:
                        exportable_results[key] = value
                
                json.dump(exportable_results, f, indent=2)
            
            print(f"Results exported to: {output_path}")
            
        except Exception as e:
            print(f"Export error: {e}")

def check_napari_environment():
    """Check what Napari ecosystem components are available"""
    print("\n" + "="*50)
    print("Napari Ecosystem Environment Check")
    print("="*50)
    
    status = {
        'napari': NAPARI_AVAILABLE,
        'numpy': NUMPY_AVAILABLE,
        'scikit-image': SKIMAGE_AVAILABLE,
        'scipy': SCIPY_AVAILABLE,
        'cellpose': CELLPOSE_AVAILABLE,
        'stardist': STARDIST_AVAILABLE,
        'btrack': BTRACK_AVAILABLE
    }
    
    for component, available in status.items():
        symbol = "✓" if available else "✗"
        print(f"{symbol} {component}")
    
    if not NAPARI_AVAILABLE:
        print("\n📋 Installation Instructions:")
        print("pip install 'napari[all]'  # Core Napari with all dependencies")
        print("pip install cellpose       # For Cellpose segmentation")
        print("pip install stardist       # For StarDist segmentation")
        print("pip install btrack         # For object tracking")
    
    print("="*50)
    return status

def main():
    """Main function to run the Napari-based analyzer"""
    # Check environment
    env_status = check_napari_environment()
    
    if not NAPARI_AVAILABLE:
        print("\n❌ Napari is not available in this environment.")
        print("This implementation requires Napari to be installed.")
        print("\nTo use this analyzer:")
        print("1. Install Napari: pip install 'napari[all]'")
        print("2. Install optional plugins:")
        print("   pip install cellpose stardist btrack")
        print("3. Re-run this script")
        return
    
    try:
        # Create analyzer instance
        analyzer = CustomNapariAnalyzer()
        
        # Initialize viewer
        viewer = analyzer.initialize_viewer()
        
        # Create test image for demonstration
        analyzer.create_test_image()
        
        print("\n🔬 Scientific Image Analyzer - Napari Implementation Ready!")
        print("\nAvailable functions:")
        print("  - Load images: analyzer.load_image('path/to/image.tif')")
        print("  - Cellpose segmentation: Press 'C' or analyzer.run_cellpose_segmentation()")
        print("  - StarDist segmentation: Press 'S' or analyzer.run_stardist_segmentation()")
        print("  - Watershed segmentation: analyzer.run_watershed_segmentation()")
        print("  - Measure objects: Press 'M' or analyzer.measure_objects()")
        print("  - Track objects: Press 'T' or analyzer.track_objects()")
        print("  - Export results: analyzer.export_results('results.json')")
        
        # Run Napari event loop
        napari.run()
        
    except Exception as e:
        print(f"Error running analyzer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()