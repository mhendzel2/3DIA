"""
Image Processing Widget for Scientific Image Analyzer
Provides various image processing and filtering operations
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSlider, QSpinBox, QDoubleSpinBox, 
                             QComboBox, QGroupBox, QCheckBox, QProgressBar)
from PyQt6.QtCore import Qt, QTimer
import napari
from napari.layers import Image

from scipy.ndimage import gaussian_filter, median_filter, binary_opening, binary_closing
from scipy.signal import convolve
from skimage import morphology
from skimage.restoration import denoise_bilateral
from skimage.filters import threshold_local

from utils.image_utils import validate_image_layer, estimate_processing_time

try:
    from advanced_analysis import AIDenoising
    ADVANCED_DENOISING_AVAILABLE = True
except ImportError:
    ADVANCED_DENOISING_AVAILABLE = False

class DeconvolutionThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, image, psf, iterations):
        super().__init__()
        self.image = image.astype(np.float64)
        self.psf = psf
        self.iterations = iterations

    def run(self):
        try:
            estimate = np.copy(self.image)
            psf_flipped = np.flip(self.psf)
            
            for i in range(self.iterations):
                relative_blur = self.image / (convolve(estimate, self.psf, mode='same') + 1e-10)
                correction = convolve(relative_blur, psf_flipped, mode='same')
                estimate *= correction
                self.progress.emit(int(100 * (i + 1) / self.iterations))
            
            self.finished.emit(estimate)
        except Exception as e:
            self.error.emit(str(e))

class ProcessingWidget(QWidget):
    """Widget for image processing operations"""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.setSingleShot(True)
        self.current_preview_layer = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Layer selection
        layer_group = QGroupBox("Select Image Layer")
        layer_layout = QVBoxLayout()
        
        self.layer_combo = QComboBox()
        self.update_layer_choices()
        layer_layout.addWidget(self.layer_combo)
        
        # Preview option
        self.preview_check = QCheckBox("Real-time preview")
        self.preview_check.setChecked(True)
        self.preview_check.toggled.connect(self.toggle_preview)
        layer_layout.addWidget(self.preview_check)
        
        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)
        
        # Smoothing/Filtering
        filter_group = QGroupBox("Smoothing & Filtering")
        filter_layout = QVBoxLayout()
        
        # Gaussian Filter
        gauss_layout = QHBoxLayout()
        gauss_layout.addWidget(QLabel("Gaussian σ:"))
        self.gauss_sigma = QDoubleSpinBox()
        self.gauss_sigma.setRange(0.1, 10.0)
        self.gauss_sigma.setValue(1.0)
        self.gauss_sigma.setSingleStep(0.1)
        self.gauss_sigma.valueChanged.connect(self.schedule_preview)
        gauss_layout.addWidget(self.gauss_sigma)
        
        self.apply_gauss_btn = QPushButton("Apply Gaussian")
        self.apply_gauss_btn.clicked.connect(self.apply_gaussian_filter)
        gauss_layout.addWidget(self.apply_gauss_btn)
        
        filter_layout.addLayout(gauss_layout)
        
        # Median Filter
        median_layout = QHBoxLayout()
        median_layout.addWidget(QLabel("Median size:"))
        self.median_size = QSpinBox()
        self.median_size.setRange(3, 21)
        self.median_size.setValue(3)
        self.median_size.setSingleStep(2)
        self.median_size.valueChanged.connect(self.schedule_preview)
        median_layout.addWidget(self.median_size)
        
        self.apply_median_btn = QPushButton("Apply Median")
        self.apply_median_btn.clicked.connect(self.apply_median_filter)
        median_layout.addWidget(self.apply_median_btn)
        
        filter_layout.addLayout(median_layout)
        
        # Bilateral Filter
        bilateral_layout = QHBoxLayout()
        bilateral_layout.addWidget(QLabel("Bilateral σ:"))
        self.bilateral_sigma = QDoubleSpinBox()
        self.bilateral_sigma.setRange(0.1, 5.0)
        self.bilateral_sigma.setValue(1.0)
        self.bilateral_sigma.setSingleStep(0.1)
        bilateral_layout.addWidget(self.bilateral_sigma)
        
        self.apply_bilateral_btn = QPushButton("Apply Bilateral")
        self.apply_bilateral_btn.clicked.connect(self.apply_bilateral_filter)
        bilateral_layout.addWidget(self.apply_bilateral_btn)
        
        filter_layout.addLayout(bilateral_layout)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Thresholding
        threshold_group = QGroupBox("Thresholding")
        threshold_layout = QVBoxLayout()
        
        # Manual threshold
        manual_layout = QHBoxLayout()
        manual_layout.addWidget(QLabel("Manual:"))
        self.manual_threshold = QSpinBox()
        self.manual_threshold.setRange(0, 65535)
        self.manual_threshold.setValue(100)
        self.manual_threshold.valueChanged.connect(self.schedule_preview)
        manual_layout.addWidget(self.manual_threshold)
        
        self.apply_manual_threshold_btn = QPushButton("Apply")
        self.apply_manual_threshold_btn.clicked.connect(self.apply_manual_threshold)
        manual_layout.addWidget(self.apply_manual_threshold_btn)
        
        threshold_layout.addLayout(manual_layout)
        
        # Auto threshold methods
        auto_layout = QHBoxLayout()
        auto_layout.addWidget(QLabel("Auto method:"))
        self.auto_method_combo = QComboBox()
        self.auto_method_combo.addItems([
            "Otsu", "Li", "Yen", "Triangle", "Mean", "Minimum"
        ])
        auto_layout.addWidget(self.auto_method_combo)
        
        self.apply_auto_threshold_btn = QPushButton("Apply Auto")
        self.apply_auto_threshold_btn.clicked.connect(self.apply_auto_threshold)
        auto_layout.addWidget(self.apply_auto_threshold_btn)
        
        threshold_layout.addLayout(auto_layout)
        
        # Adaptive threshold
        adaptive_layout = QHBoxLayout()
        adaptive_layout.addWidget(QLabel("Block size:"))
        self.adaptive_block_size = QSpinBox()
        self.adaptive_block_size.setRange(3, 101)
        self.adaptive_block_size.setValue(35)
        self.adaptive_block_size.setSingleStep(2)
        adaptive_layout.addWidget(self.adaptive_block_size)
        
        adaptive_layout.addWidget(QLabel("Offset:"))
        self.adaptive_offset = QSpinBox()
        self.adaptive_offset.setRange(-50, 50)
        self.adaptive_offset.setValue(10)
        adaptive_layout.addWidget(self.adaptive_offset)
        
        self.apply_adaptive_btn = QPushButton("Apply Adaptive (Local)")
        self.apply_adaptive_btn.clicked.connect(self.apply_adaptive_threshold)
        adaptive_layout.addWidget(self.apply_adaptive_btn)
        
        threshold_layout.addLayout(adaptive_layout)
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # Contrast Enhancement
        contrast_group = QGroupBox("Contrast Enhancement")
        contrast_layout = QVBoxLayout()
        
        # Gamma correction
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))
        self.gamma_value = QDoubleSpinBox()
        self.gamma_value.setRange(0.1, 3.0)
        self.gamma_value.setValue(1.0)
        self.gamma_value.setSingleStep(0.1)
        self.gamma_value.valueChanged.connect(self.schedule_preview)
        gamma_layout.addWidget(self.gamma_value)
        
        self.apply_gamma_btn = QPushButton("Apply Gamma")
        self.apply_gamma_btn.clicked.connect(self.apply_gamma_correction)
        gamma_layout.addWidget(self.apply_gamma_btn)
        
        contrast_layout.addLayout(gamma_layout)
        
        # Histogram equalization
        hist_layout = QHBoxLayout()
        self.apply_hist_eq_btn = QPushButton("Histogram Equalization")
        self.apply_hist_eq_btn.clicked.connect(self.apply_histogram_equalization)
        hist_layout.addWidget(self.apply_hist_eq_btn)
        
        self.apply_clahe_btn = QPushButton("CLAHE")
        self.apply_clahe_btn.clicked.connect(self.apply_clahe)
        hist_layout.addWidget(self.apply_clahe_btn)
        
        contrast_layout.addLayout(hist_layout)
        
        contrast_group.setLayout(contrast_layout)
        layout.addWidget(contrast_group)
        
        # Morphological Operations
        morph_group = QGroupBox("Morphological Operations")
        morph_layout = QVBoxLayout()
        
        # Structuring element
        struct_layout = QHBoxLayout()
        struct_layout.addWidget(QLabel("Element:"))
        self.struct_combo = QComboBox()
        self.struct_combo.addItems(["disk", "square", "cross"])
        struct_layout.addWidget(self.struct_combo)
        
        struct_layout.addWidget(QLabel("Size:"))
        self.struct_size = QSpinBox()
        self.struct_size.setRange(1, 15)
        self.struct_size.setValue(3)
        struct_layout.addWidget(self.struct_size)
        
        morph_layout.addLayout(struct_layout)
        
        # Operations
        morph_ops_layout = QHBoxLayout()
        self.erosion_btn = QPushButton("Erosion")
        self.erosion_btn.clicked.connect(self.apply_erosion)
        morph_ops_layout.addWidget(self.erosion_btn)
        
        self.dilation_btn = QPushButton("Dilation")
        self.dilation_btn.clicked.connect(self.apply_dilation)
        morph_ops_layout.addWidget(self.dilation_btn)
        
        morph_layout.addLayout(morph_ops_layout)
        
        morph_ops2_layout = QHBoxLayout()
        self.opening_btn = QPushButton("Opening")
        self.opening_btn.clicked.connect(self.apply_opening)
        morph_ops2_layout.addWidget(self.opening_btn)
        
        self.closing_btn = QPushButton("Closing")
        self.closing_btn.clicked.connect(self.apply_closing)
        morph_ops2_layout.addWidget(self.closing_btn)
        
        morph_layout.addLayout(morph_ops2_layout)
        
        morph_group.setLayout(morph_layout)
        layout.addWidget(morph_group)
        
        deconv_group = QGroupBox("Deconvolution")
        deconv_layout = QVBoxLayout(deconv_group)
        
        deconv_layout.addWidget(QLabel("PSF Sigma (Gaussian approximation):"))
        self.psf_sigma = QDoubleSpinBox()
        self.psf_sigma.setRange(0.5, 5.0)
        self.psf_sigma.setValue(1.5)
        deconv_layout.addWidget(self.psf_sigma)
        
        deconv_layout.addWidget(QLabel("Iterations:"))
        self.deconv_iterations = QSpinBox()
        self.deconv_iterations.setRange(1, 100)
        self.deconv_iterations.setValue(10)
        deconv_layout.addWidget(self.deconv_iterations)
        
        self.apply_deconv_btn = QPushButton("Apply Richardson-Lucy")
        self.apply_deconv_btn.clicked.connect(self.apply_deconvolution)
        deconv_layout.addWidget(self.apply_deconv_btn)
        
        deconv_group.setLayout(deconv_layout)
        layout.addWidget(deconv_group)
        
        if ADVANCED_DENOISING_AVAILABLE:
            denoise_group = QGroupBox("Advanced Denoising")
            denoise_layout = QVBoxLayout()

            # Bilateral Filter (advanced)
            bilateral_adv_layout = QHBoxLayout()
            bilateral_adv_layout.addWidget(QLabel("Bilateral σ (spatial):"))
            self.bilateral_spatial_sigma = QDoubleSpinBox()
            self.bilateral_spatial_sigma.setRange(0.1, 20.0)
            self.bilateral_spatial_sigma.setValue(5.0)
            bilateral_adv_layout.addWidget(self.bilateral_spatial_sigma)
            
            self.apply_bilateral_adv_btn = QPushButton("Apply Bilateral Filter")
            self.apply_bilateral_adv_btn.clicked.connect(self.apply_advanced_bilateral_filter)
            bilateral_adv_layout.addWidget(self.apply_bilateral_adv_btn)
            denoise_layout.addLayout(bilateral_adv_layout)

            nlm_layout = QHBoxLayout()
            nlm_layout.addWidget(QLabel("NLM Denoise Strength (h):"))
            self.nlm_h = QDoubleSpinBox()
            self.nlm_h.setRange(1.0, 50.0)
            self.nlm_h.setValue(10.0)
            nlm_layout.addWidget(self.nlm_h)

            self.apply_nlm_btn = QPushButton("Apply Non-local Means")
            self.apply_nlm_btn.clicked.connect(self.apply_nlm_filter)
            nlm_layout.addWidget(self.apply_nlm_btn)
            denoise_layout.addLayout(nlm_layout)

            denoise_group.setLayout(denoise_layout)
            layout.addWidget(denoise_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def get_current_layer(self):
        """Get the currently selected image layer"""
        layer_name = self.layer_combo.currentText()
        if layer_name == "No layers available":
            return None
            
        for layer in self.viewer.layers:
            if layer.name == layer_name and isinstance(layer, Image):
                return layer
        return None
        
    def schedule_preview(self):
        """Schedule a preview update"""
        if self.preview_check.isChecked():
            self.preview_timer.stop()
            self.preview_timer.start(500)  # 500ms delay
            
    def toggle_preview(self, enabled):
        """Toggle preview mode"""
        if not enabled and self.current_preview_layer:
            self.viewer.layers.remove(self.current_preview_layer)
            self.current_preview_layer = None
        elif enabled:
            self.schedule_preview()
            
    def update_preview(self):
        """Update the preview layer"""
        if not self.preview_check.isChecked():
            return
            
        layer = self.get_current_layer()
        if layer is None:
            return
            
        try:
            # Apply current filter settings for preview
            data = layer.data.copy()
            
            # Apply transformations based on current settings
            if self.gauss_sigma.value() > 0.1:
                data = gaussian_filter(data, sigma=self.gauss_sigma.value())
                
            if self.gamma_value.value() != 1.0:
                try:
                    from skimage import exposure
                    data = exposure.adjust_gamma(data, gamma=self.gamma_value.value())
                except ImportError:
                    pass
                
            # Update or create preview layer
            preview_name = f"{layer.name}_preview"
            
            if self.current_preview_layer:
                self.current_preview_layer.data = data
            else:
                self.current_preview_layer = self.viewer.add_image(
                    data,
                    name=preview_name,
                    scale=layer.scale,
                    opacity=0.7,
                    colormap='viridis'
                )
                
        except Exception as e:
            print(f"Preview update failed: {str(e)}")
            
    def apply_gaussian_filter(self):
        """Apply Gaussian filter to selected layer"""
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer")
            return
            
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            sigma = self.gauss_sigma.value()
            filtered_data = gaussian_filter(layer.data, sigma=sigma)
            
            self.progress_bar.setValue(50)
            
            new_layer = self.viewer.add_image(
                filtered_data,
                name=f"{layer.name}_gauss_{sigma:.1f}",
                scale=layer.scale,
                colormap=layer.colormap.name if hasattr(layer.colormap, 'name') else 'gray'
            )
            
            self.progress_bar.setValue(100)
            print(f"Applied Gaussian filter (σ={sigma}) to {layer.name}")
            
        except Exception as e:
            self.show_error(f"Gaussian filter failed: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            
    def apply_median_filter(self):
        """Apply median filter to selected layer"""
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer")
            return
            
        try:
            self.progress_bar.setVisible(True)
            
            size = self.median_size.value()
            filtered_data = median_filter(layer.data, size=size)
            
            new_layer = self.viewer.add_image(
                filtered_data,
                name=f"{layer.name}_median_{size}",
                scale=layer.scale,
                colormap=layer.colormap.name if hasattr(layer.colormap, 'name') else 'gray'
            )
            
            print(f"Applied median filter (size={size}) to {layer.name}")
            
        except Exception as e:
            self.show_error(f"Median filter failed: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            
    def apply_bilateral_filter(self):
        """Apply bilateral filter to selected layer"""
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer")
            return
            
        try:
            self.progress_bar.setVisible(True)
            
            # For 3D data, apply to each slice
            if layer.data.ndim == 3:
                filtered_data = np.zeros_like(layer.data)
                for i, slice_data in enumerate(layer.data):
                    filtered_data[i] = denoise_bilateral(
                        slice_data, 
                        sigma_spatial=self.bilateral_sigma.value()
                    )
                    self.progress_bar.setValue(int(100 * i / layer.data.shape[0]))
            else:
                filtered_data = denoise_bilateral(
                    layer.data, 
                    sigma_spatial=self.bilateral_sigma.value()
                )
                
            new_layer = self.viewer.add_image(
                filtered_data,
                name=f"{layer.name}_bilateral",
                scale=layer.scale,
                colormap=layer.colormap.name if hasattr(layer.colormap, 'name') else 'gray'
            )
            
            print(f"Applied bilateral filter to {layer.name}")
            
        except Exception as e:
            self.show_error(f"Bilateral filter failed: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            
    def apply_manual_threshold(self):
        """Apply manual threshold to selected layer"""
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer")
            return
            
        try:
            threshold = self.manual_threshold.value()
            binary_data = layer.data > threshold
            
            new_layer = self.viewer.add_image(
                binary_data.astype(np.uint8) * 255,
                name=f"{layer.name}_thresh_{threshold}",
                scale=layer.scale,
                colormap='gray'
            )
            
            print(f"Applied manual threshold ({threshold}) to {layer.name}")
            
        except Exception as e:
            self.show_error(f"Manual threshold failed: {str(e)}")
            
    def apply_auto_threshold(self):
        """Apply automatic threshold to selected layer"""
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer")
            return
            
        try:
            method = self.auto_method_combo.currentText().lower()
            
            # Get threshold value using skimage filters
            from skimage import filters
            if method == "otsu":
                threshold = filters.threshold_otsu(layer.data)
            elif method == "li":
                threshold = filters.threshold_li(layer.data)
            elif method == "yen":
                threshold = filters.threshold_yen(layer.data)
            elif method == "triangle":
                threshold = filters.threshold_triangle(layer.data)
            elif method == "mean":
                threshold = filters.threshold_mean(layer.data)
            elif method == "minimum":
                threshold = filters.threshold_minimum(layer.data)
            else:
                threshold = filters.threshold_otsu(layer.data)
                
            binary_data = layer.data > threshold
            
            new_layer = self.viewer.add_image(
                binary_data.astype(np.uint8) * 255,
                name=f"{layer.name}_{method}_thresh",
                scale=layer.scale,
                colormap='gray'
            )
            
            print(f"Applied {method} threshold ({threshold:.1f}) to {layer.name}")
            
        except Exception as e:
            self.show_error(f"Auto threshold failed: {str(e)}")
            
    def apply_adaptive_threshold(self):
        """Apply adaptive threshold to the selected layer."""
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer to apply adaptive threshold.")
            return

        try:
            # Parameters for adaptive thresholding
            block_size = self.adaptive_block_size.value()
            offset = self.adaptive_offset.value()

            if block_size % 2 == 0:
                block_size += 1

            binary_adaptive = layer.data > threshold_local(layer.data, block_size, offset=offset)
            
            adaptive_data = binary_adaptive.astype(np.uint8) * 255
            
            new_layer = self.viewer.add_image(
                adaptive_data,
                name=f"{layer.name}_adaptive_thresh",
                scale=layer.scale,
                colormap='gray'
            )
            
            print(f"Applied adaptive threshold to {layer.name}")

        except Exception as e:
            self.show_error(f"Adaptive threshold failed: {str(e)}")
            
    def apply_gamma_correction(self):
        """Apply gamma correction to selected layer"""
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer")
            return
            
        try:
            gamma = self.gamma_value.value()
            corrected_data = exposure.adjust_gamma(layer.data, gamma=gamma)
            
            new_layer = self.viewer.add_image(
                corrected_data,
                name=f"{layer.name}_gamma_{gamma:.1f}",
                scale=layer.scale,
                colormap=layer.colormap.name if hasattr(layer.colormap, 'name') else 'gray'
            )
            
            print(f"Applied gamma correction (γ={gamma}) to {layer.name}")
            
        except Exception as e:
            self.show_error(f"Gamma correction failed: {str(e)}")
            
    def apply_histogram_equalization(self):
        """Apply histogram equalization to selected layer"""
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer")
            return
            
        try:
            equalized_data = exposure.equalize_hist(layer.data)
            
            new_layer = self.viewer.add_image(
                equalized_data,
                name=f"{layer.name}_histeq",
                scale=layer.scale,
                colormap=layer.colormap.name if hasattr(layer.colormap, 'name') else 'gray'
            )
            
            print(f"Applied histogram equalization to {layer.name}")
            
        except Exception as e:
            self.show_error(f"Histogram equalization failed: {str(e)}")
            
    def apply_clahe(self):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer")
            return
            
        try:
            clahe_data = exposure.equalize_adapthist(layer.data, clip_limit=0.03)
            
            new_layer = self.viewer.add_image(
                clahe_data,
                name=f"{layer.name}_clahe",
                scale=layer.scale,
                colormap=layer.colormap.name if hasattr(layer.colormap, 'name') else 'gray'
            )
            
            print(f"Applied CLAHE to {layer.name}")
            
        except Exception as e:
            self.show_error(f"CLAHE failed: {str(e)}")
            
    def get_structuring_element(self):
        """Get structuring element for morphological operations"""
        shape = self.struct_combo.currentText()
        size = self.struct_size.value()
        
        if shape == "disk":
            return morphology.disk(size)
        elif shape == "square":
            return morphology.square(size)
        elif shape == "cross":
            return morphology.star(size)
        else:
            return morphology.disk(size)
            
    def apply_erosion(self):
        """Apply morphological erosion"""
        self.apply_morphological_operation("erosion")
        
    def apply_dilation(self):
        """Apply morphological dilation"""
        self.apply_morphological_operation("dilation")
        
    def apply_opening(self):
        """Apply morphological opening"""
        self.apply_morphological_operation("opening")
        
    def apply_closing(self):
        """Apply morphological closing"""
        self.apply_morphological_operation("closing")
        
    def apply_morphological_operation(self, operation):
        """Apply morphological operation to selected layer"""
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer")
            return
            
        try:
            selem = self.get_structuring_element()
            
            if operation == "erosion":
                result_data = morphology.erosion(layer.data, selem)
            elif operation == "dilation":
                result_data = morphology.dilation(layer.data, selem)
            elif operation == "opening":
                result_data = morphology.opening(layer.data, selem)
            elif operation == "closing":
                result_data = morphology.closing(layer.data, selem)
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
            new_layer = self.viewer.add_image(
                result_data,
                name=f"{layer.name}_{operation}",
                scale=layer.scale,
                colormap=layer.colormap.name if hasattr(layer.colormap, 'name') else 'gray'
            )
            
            print(f"Applied {operation} to {layer.name}")
            
        except Exception as e:
            self.show_error(f"Morphological {operation} failed: {str(e)}")
            
    def update_layer_choices(self):
        """Update the layer selection combo box"""
        self.layer_combo.clear()
        
        image_layers = [layer for layer in self.viewer.layers if isinstance(layer, Image)]
        
        if image_layers:
            for layer in image_layers:
                self.layer_combo.addItem(layer.name)
        else:
            self.layer_combo.addItem("No layers available")
            
    def show_error(self, message):
        """Display error message"""
        print(f"ERROR: {message}")
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Processing Error")
        msg.setText(message)
        msg.exec()
        
    def apply_advanced_bilateral_filter(self):
        """Apply advanced bilateral filter using AIDenoising"""
        if not ADVANCED_DENOISING_AVAILABLE:
            self.show_error("Advanced denoising not available. Please check advanced_analysis.py")
            return
            
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer.")
            return
            
        try:
            self.progress_bar.setVisible(True)
            sigma_spatial = self.bilateral_spatial_sigma.value()
            denoised_data = AIDenoising.bilateral_filter_denoising(layer.data, sigma_spatial=sigma_spatial)
            
            new_layer = self.viewer.add_image(
                denoised_data, 
                name=f"{layer.name}_bilateral_adv", 
                scale=layer.scale,
                colormap=layer.colormap.name if hasattr(layer.colormap, 'name') else 'gray'
            )
            
            print(f"Applied advanced bilateral filter to {layer.name}")
            
        except Exception as e:
            self.show_error(f"Advanced bilateral filter failed: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    def apply_nlm_filter(self):
        """Apply non-local means filter using AIDenoising"""
        if not ADVANCED_DENOISING_AVAILABLE:
            self.show_error("Advanced denoising not available. Please check advanced_analysis.py")
            return
            
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer.")
            return
            
        try:
            self.progress_bar.setVisible(True)
            h = self.nlm_h.value()
            denoised_data = AIDenoising.non_local_means_denoising(layer.data, h=h)
            
            new_layer = self.viewer.add_image(
                denoised_data, 
                name=f"{layer.name}_nlm", 
                scale=layer.scale,
                colormap=layer.colormap.name if hasattr(layer.colormap, 'name') else 'gray'
            )
            
            print(f"Applied non-local means denoising to {layer.name}")
            
        except Exception as e:
            self.show_error(f"Non-local means filter failed: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    def apply_deconvolution(self):
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Select a valid image layer for deconvolution.")
            return

        sigma = self.psf_sigma.value()
        psf_size = int(sigma * 4) + 1
        
        psf = np.zeros([psf_size] * layer.ndim)
        center = tuple([s // 2 for s in psf.shape])
        psf[center] = 1
        psf = gaussian_filter(psf.astype(float), sigma=sigma)
        psf /= psf.sum()

        iterations = self.deconv_iterations.value()
        
        self.deconv_thread = DeconvolutionThread(layer.data, psf, iterations)
        self.deconv_thread.finished.connect(self.on_deconv_finished)
        self.deconv_thread.error.connect(self.show_error)
        self.deconv_thread.progress.connect(self.progress_bar.setValue)
        
        self.progress_bar.setVisible(True)
        self.deconv_thread.start()

    def on_deconv_finished(self, deconvolved_image):
        self.progress_bar.setVisible(False)
        layer = self.get_current_layer()
        self.viewer.add_image(deconvolved_image, name=f"{layer.name}_deconvolved", scale=layer.scale)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.current_preview_layer and self.current_preview_layer in self.viewer.layers:
            self.viewer.layers.remove(self.current_preview_layer)
        self.preview_timer.stop()
