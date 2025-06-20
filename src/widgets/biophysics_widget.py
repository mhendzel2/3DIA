"""
Biophysics Widget for Scientific Image Analyzer
Provides tools for analyzing FRAP and other biophysical data.
"""
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                             QComboBox, QGroupBox, QTextEdit)
import napari
from napari.layers import Image, Points

try:
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
    SCIPY_MPL_AVAILABLE = True
except ImportError:
    SCIPY_MPL_AVAILABLE = False

def frap_recovery_func(t, mobile_fraction, halftime):
    return mobile_fraction * (1 - np.exp(-np.log(2) * t / halftime))

class BiophysicsWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        frap_group = QGroupBox("FRAP Analysis")
        frap_layout = QVBoxLayout(frap_group)

        frap_layout.addWidget(QLabel("Select Timelapse Image Layer:"))
        self.image_layer_combo = QComboBox()
        frap_layout.addWidget(self.image_layer_combo)

        frap_layout.addWidget(QLabel("Select ROI Points Layer:"))
        self.points_layer_combo = QComboBox()
        frap_layout.addWidget(self.points_layer_combo)

        self.analyze_frap_btn = QPushButton("Analyze FRAP Curve")
        self.analyze_frap_btn.clicked.connect(self.analyze_frap)
        frap_layout.addWidget(self.analyze_frap_btn)

        if SCIPY_MPL_AVAILABLE:
            self.figure = plt.figure()
            self.canvas = FigureCanvas(self.figure)
            frap_layout.addWidget(self.canvas)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(100)
        frap_layout.addWidget(self.results_text)
        
        layout.addWidget(frap_group)
        self.setLayout(layout)

        if not SCIPY_MPL_AVAILABLE:
            self.setDisabled(True)
            self.results_text.setText("Please install scipy and matplotlib for this feature.")

        self.viewer.layers.events.inserted.connect(self.update_layer_choices)
        self.viewer.layers.events.removed.connect(self.update_layer_choices)
        self.update_layer_choices()

    def update_layer_choices(self):
        self.image_layer_combo.clear()
        self.points_layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and layer.ndim >= 3:
                self.image_layer_combo.addItem(layer.name)
            elif isinstance(layer, Points):
                self.points_layer_combo.addItem(layer.name)
    
    def analyze_frap(self):
        if self.image_layer_combo.count() == 0 or self.points_layer_combo.count() == 0:
            self.results_text.setText("Error: Select a timelapse image and a points layer.")
            return

        image_layer = self.viewer.layers[self.image_layer_combo.currentText()]
        points_layer = self.viewer.layers[self.points_layer_combo.currentText()]
        
        roi_center = points_layer.data[0]
        t_dim, *spatial_dims = image_layer.data.shape
        roi_coord_tzyx = [int(c) for c in roi_center] 
        roi_coord_zyx = roi_coord_tzyx[-len(spatial_dims):]

        radius = 5 
        
        intensities = []
        for t in range(t_dim):
            indices = np.indices(spatial_dims)
            dist_sq = sum((indices[i] - roi_coord_zyx[i])**2 for i in range(len(spatial_dims)))
            mask = dist_sq <= radius**2
            
            frame = image_layer.data[t]
            roi_pixels = frame[mask]
            intensities.append(np.mean(roi_pixels) if roi_pixels.size > 0 else 0)

        intensities = np.array(intensities)

        pre_bleach_intensity = np.mean(intensities[:2])
        post_bleach_intensity = intensities[2]
        
        normalized_intensities = (intensities - post_bleach_intensity) / (pre_bleach_intensity - post_bleach_intensity)
        time_points = np.arange(len(normalized_intensities))

        try:
            popt, _ = curve_fit(frap_recovery_func, time_points[2:], normalized_intensities[2:], p0=[0.5, 10])
            mobile_fraction, halftime = popt

            if SCIPY_MPL_AVAILABLE:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.plot(time_points, normalized_intensities, 'bo', label='Experimental Data')
                ax.plot(time_points[2:], frap_recovery_func(time_points[2:], *popt), 'r-', label='Fit')
                ax.set_xlabel("Time (frames)")
                ax.set_ylabel("Normalized Intensity")
                ax.set_title("FRAP Recovery Curve")
                ax.legend()
                self.canvas.draw()

            self.results_text.setText(f"Mobile Fraction: {mobile_fraction:.3f}\nRecovery Halftime: {halftime:.2f} frames")
        except Exception as e:
            self.results_text.setText(f"FRAP analysis failed: {str(e)}")
