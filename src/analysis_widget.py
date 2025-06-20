# analysis_widget.py
# This module defines the analysis widget for colocalization and measurements.

from magicgui import magic_factory
from napari.layers import Image, Labels
from napari.viewer import Viewer
import numpy as np

# Import analysis libraries with fallbacks
try:
    from scipy.stats import pearsonr
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False

try:
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem
    from PyQt5.QtCore import Qt
    QT5_AVAILABLE = True
    QT6_AVAILABLE = False
except ImportError:
    try:
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem
        from PyQt6.QtCore import Qt
        QT5_AVAILABLE = False
        QT6_AVAILABLE = True
    except ImportError:
        QT5_AVAILABLE = False
        QT6_AVAILABLE = False

# Basic implementations for when libraries aren't available
def basic_pearson_correlation(x, y):
    """Basic Pearson correlation implementation"""
    if SCIPY_AVAILABLE:
        corr, _ = pearsonr(x, y)
        return corr
    else:
        # Manual calculation
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
        return numerator / denominator if denominator > 0 else 0

@magic_factory(
    labels_layer={'label': 'Segmented Objects'},
    intensity_layer={'label': 'Intensity Image (optional)'},
    call_button="Measure Objects"
)
def measure_objects_widget(labels_layer: Labels, intensity_layer: Image = None):
    """Measure properties of segmented objects."""
    if labels_layer is None:
        print("No labels layer selected")
        return
    
    print("Measuring object properties...")
    
    try:
        if SKIMAGE_AVAILABLE:
            # Use scikit-image regionprops
            if intensity_layer is not None:
                props = measure.regionprops(labels_layer.data, intensity_image=intensity_layer.data)
            else:
                props = measure.regionprops(labels_layer.data)
            
            measurements = []
            for prop in props:
                measurement = {
                    'Label': prop.label,
                    'Area': prop.area,
                    'Perimeter': prop.perimeter,
                    'Centroid_Y': prop.centroid[0],
                    'Centroid_X': prop.centroid[1],
                    'Eccentricity': prop.eccentricity,
                    'Solidity': prop.solidity,
                    'Extent': prop.extent
                }
                
                if intensity_layer is not None:
                    measurement.update({
                        'Mean_Intensity': prop.mean_intensity,
                        'Max_Intensity': prop.max_intensity,
                        'Min_Intensity': prop.min_intensity
                    })
                
                measurements.append(measurement)
        else:
            # Basic measurements without scikit-image
            unique_labels = np.unique(labels_layer.data)[1:]  # Skip background
            measurements = []
            
            for label in unique_labels:
                mask = labels_layer.data == label
                area = np.sum(mask)
                coords = np.where(mask)
                
                if len(coords[0]) > 0:
                    centroid_y = np.mean(coords[0])
                    centroid_x = np.mean(coords[1])
                    
                    measurement = {
                        'Label': int(label),
                        'Area': area,
                        'Centroid_Y': centroid_y,
                        'Centroid_X': centroid_x
                    }
                    
                    if intensity_layer is not None:
                        intensities = intensity_layer.data[mask]
                        measurement.update({
                            'Mean_Intensity': np.mean(intensities),
                            'Max_Intensity': np.max(intensities),
                            'Min_Intensity': np.min(intensities)
                        })
                    
                    measurements.append(measurement)
        
        print(f"Measured {len(measurements)} objects")
        
        # Display results in a table (if Qt is available)
        if (QT5_AVAILABLE or QT6_AVAILABLE) and measurements:
            show_measurements_table(measurements)
        
        return measurements
        
    except Exception as e:
        print(f"Error measuring objects: {e}")
        return None

def show_measurements_table(measurements):
    """Display measurements in a table widget"""
    try:
        import napari
        viewer = napari.current_viewer()
        
        # Create table widget
        table_widget = QWidget()
        layout = QVBoxLayout()
        
        # Create table
        table = QTableWidget()
        
        if measurements:
            # Set up table dimensions
            table.setRowCount(len(measurements))
            headers = list(measurements[0].keys())
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            
            # Fill table with data
            for row, measurement in enumerate(measurements):
                for col, key in enumerate(headers):
                    value = measurement[key]
                    if isinstance(value, float):
                        item = QTableWidgetItem(f"{value:.3f}")
                    else:
                        item = QTableWidgetItem(str(value))
                    table.setItem(row, col, item)
            
            layout.addWidget(QLabel(f"Measurements for {len(measurements)} objects:"))
            layout.addWidget(table)
            table_widget.setLayout(layout)
            
            # Add to napari viewer
            viewer.window.add_dock_widget(table_widget, name="Object Measurements")
            
    except Exception as e:
        print(f"Error displaying table: {e}")

# For this widget, we'll create a custom QWidget to hold the plot and stats.
class ColocalizationWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())
        
        # Add a button to trigger the calculation
        if QT5_AVAILABLE or QT6_AVAILABLE:
            self.calc_button = QPushButton("Calculate Colocalization")
            self.calc_button.clicked.connect(self.run_colocalization)
            self.layout().addWidget(self.calc_button)
        
        # Create Matplotlib figure for the 2D histogram
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(4, 4), tight_layout=True)
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            self.layout().addWidget(self.canvas)
        
        # Create a label to display statistics
        if QT5_AVAILABLE or QT6_AVAILABLE:
            self.stats_label = QLabel("Select two image layers to begin.")
            self.layout().addWidget(self.stats_label)
        
    def run_colocalization(self):
        try:
            selected_layers = list(self.viewer.layers.selection)
            image_layers = [l for l in selected_layers if isinstance(l, Image)]
            
            if len(image_layers) != 2:
                if hasattr(self, 'stats_label'):
                    self.stats_label.setText("Error: Please select exactly two\nimage layers.")
                print("Error: Please select exactly two image layers.")
                return

            ch1_layer, ch2_layer = image_layers
            ch1_data = ch1_layer.data.flatten()
            ch2_data = ch2_layer.data.flatten()
            
            # Ensure data has the same shape
            min_size = min(len(ch1_data), len(ch2_data))
            ch1_data = ch1_data[:min_size]
            ch2_data = ch2_data[:min_size]
            
            # --- Calculate Statistics ---
            corr = basic_pearson_correlation(ch1_data, ch2_data)
            
            # Calculate Manders coefficients
            ch1_thresh = np.mean(ch1_data[ch1_data > 0]) if np.any(ch1_data > 0) else 0
            ch2_thresh = np.mean(ch2_data[ch2_data > 0]) if np.any(ch2_data > 0) else 0
            
            if ch1_thresh > 0 and ch2_thresh > 0:
                coloc_mask = (ch1_layer.data > ch1_thresh) & (ch2_layer.data > ch2_thresh)
                ch1_total = np.sum(ch1_layer.data[ch1_layer.data > ch1_thresh])
                if ch1_total > 0:
                    m1 = np.sum(ch1_layer.data[coloc_mask]) / ch1_total
                else:
                    m1 = 0
                
                ch2_total = np.sum(ch2_layer.data[ch2_layer.data > ch2_thresh])
                if ch2_total > 0:
                    m2 = np.sum(ch2_layer.data[coloc_mask]) / ch2_total
                else:
                    m2 = 0
            else:
                m1, m2 = 0, 0
            
            stats_text = f"Pearson's R: {corr:.3f}\nMander's M1: {m1:.3f}\nMander's M2: {m2:.3f}"
            
            if hasattr(self, 'stats_label'):
                self.stats_label.setText(stats_text)
            
            print(f"Colocalization Analysis Results:")
            print(f"Pearson's R: {corr:.3f}")
            print(f"Mander's M1: {m1:.3f}")
            print(f"Mander's M2: {m2:.3f}")
            
            # --- Plot 2D Histogram ---
            if MATPLOTLIB_AVAILABLE and hasattr(self, 'ax'):
                self.ax.clear()
                
                # Calculate reasonable plot ranges
                ch1_max = np.percentile(ch1_data, 99.9)
                ch2_max = np.percentile(ch2_data, 99.9)
                
                self.ax.hist2d(ch1_data, ch2_data, bins=100, cmap='viridis', 
                              range=[[ch1_data.min(), ch1_max], [ch2_data.min(), ch2_max]])
                self.ax.set_xlabel(ch1_layer.name, fontsize=8)
                self.ax.set_ylabel(ch2_layer.name, fontsize=8)
                self.ax.set_title(f"R = {corr:.3f}", fontsize=10)
                self.canvas.draw()
                
        except Exception as e:
            print(f"Error in colocalization analysis: {e}")
            if hasattr(self, 'stats_label'):
                self.stats_label.setText(f"Error: {str(e)}")

@magic_factory(
    image1={'label': 'Channel 1'},
    image2={'label': 'Channel 2'},
    call_button="Quick Colocalization"
)
def quick_colocalization_widget(image1: Image, image2: Image):
    """Quick colocalization analysis between two images."""
    if image1 is None or image2 is None:
        print("Please select two image layers")
        return
    
    try:
        # Flatten the data
        data1 = image1.data.flatten()
        data2 = image2.data.flatten()
        
        # Ensure same size
        min_size = min(len(data1), len(data2))
        data1 = data1[:min_size]
        data2 = data2[:min_size]
        
        # Calculate correlation
        correlation = basic_pearson_correlation(data1, data2)
        
        # Calculate Manders coefficients
        thresh1 = np.mean(data1[data1 > 0]) if np.any(data1 > 0) else 0
        thresh2 = np.mean(data2[data2 > 0]) if np.any(data2 > 0) else 0
        
        if thresh1 > 0 and thresh2 > 0:
            coloc_pixels = (data1 > thresh1) & (data2 > thresh2)
            m1 = np.sum(data1[coloc_pixels]) / np.sum(data1[data1 > thresh1]) if np.sum(data1[data1 > thresh1]) > 0 else 0
            m2 = np.sum(data2[coloc_pixels]) / np.sum(data2[data2 > thresh2]) if np.sum(data2[data2 > thresh2]) > 0 else 0
        else:
            m1, m2 = 0, 0
        
        print(f"Colocalization Results:")
        print(f"Pearson's correlation: {correlation:.3f}")
        print(f"Mander's M1: {m1:.3f}")
        print(f"Mander's M2: {m2:.3f}")
        
        return {
            'pearson_correlation': correlation,
            'manders_m1': m1,
            'manders_m2': m2
        }
        
    except Exception as e:
        print(f"Error in colocalization analysis: {e}")
        return None

# A function that returns our custom widget, making it easy to add to napari
def analysis_widget():
    """Create analysis widget container"""
    try:
        import napari
        viewer = napari.current_viewer()
        
        if QT5_AVAILABLE or QT6_AVAILABLE:
            # Return custom colocalization widget
            return ColocalizationWidget(viewer)
        else:
            # Fallback to simple magicgui widgets
            from magicgui.widgets import Container
            return Container(widgets=[
                measure_objects_widget(),
                quick_colocalization_widget()
            ])
            
    except Exception as e:
        print(f"Error creating analysis widget: {e}")
        # Return minimal widget
        try:
            from magicgui.widgets import Container
            return Container(widgets=[quick_colocalization_widget()])
        except:
            from magicgui.widgets import Container
            return Container(widgets=[])