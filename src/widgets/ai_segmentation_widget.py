"""
AI Segmentation Widget for Scientific Image Analyzer
Integrates true Cellpose and StarDist models for advanced segmentation.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                             QSpinBox, QDoubleSpinBox, QComboBox, 
                             QGroupBox, QCheckBox, QProgressBar)
from PyQt6.QtCore import QThread, pyqtSignal
import napari
import os
import requests

from utils.image_utils import validate_image_layer

try:
    from cellpose import models
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    AI_MODELS_AVAILABLE = True
except ImportError:
    AI_MODELS_AVAILABLE = False
    print("Warning: cellpose or stardist not installed. AI Segmentation widget will be disabled.")

try:
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_MODELS_AVAILABLE = True
except ImportError:
    SAM_MODELS_AVAILABLE = False
    print("Warning: segment-anything is not installed. SAM widget will be disabled.")

def get_sam_checkpoint(model_type='vit_b'):
    if model_type == 'vit_b':
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        filename = "sam_vit_b_01ec64.pth"
    else:
        raise ValueError("Unknown SAM model type")

    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    return filename

# Valid Cellpose model types
VALID_CELLPOSE_MODELS = {'cyto', 'nuclei', 'cyto2', 'cyto3', 'livecell', 'tissuenet', 'CPx'}

class AISegmentationThread(QThread):
    progress = pyqtSignal(str, int)
    finished_segmentation = pyqtSignal(np.ndarray, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_name, image_data, params):
        super().__init__()
        self.model_name = model_name
        self.image_data = image_data
        self.params = params
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the running operation"""
        self._cancelled = True

    def is_cancelled(self):
        """Check if cancellation was requested"""
        return self._cancelled

    def _validate_params(self) -> tuple:
        """
        Validate parameters before running segmentation.
        Returns (is_valid, error_message).
        """
        # Validate image data
        if self.image_data is None:
            return False, "No image data provided"
        
        if not isinstance(self.image_data, np.ndarray):
            return False, "Image data must be a numpy array"
        
        if self.image_data.size == 0:
            return False, "Image data is empty"
        
        # Model-specific validation
        if self.model_name == 'cellpose':
            model_type = self.params.get('model_type', 'cyto')
            if model_type not in VALID_CELLPOSE_MODELS:
                return False, f"Invalid Cellpose model type '{model_type}'. Valid types: {', '.join(VALID_CELLPOSE_MODELS)}"
            
            diameter = self.params.get('diameter', 30)
            if diameter < 0:
                return False, f"Diameter must be non-negative, got {diameter}"
            
            # diameter=0 means auto-detection, which is valid
            if diameter > 0 and diameter < 5:
                return False, f"Diameter {diameter} is too small. Use at least 5 pixels or 0 for auto-detection"
        
        elif self.model_name == 'stardist':
            # StarDist works best on 2D images
            if self.image_data.ndim > 3:
                return False, "StarDist 2D model requires 2D or single-channel 3D images"
        
        elif self.model_name == 'sam':
            # SAM requires reasonable image size
            if self.image_data.ndim < 2:
                return False, "SAM requires at least 2D images"
            # Warn about large images (memory usage)
            total_pixels = self.image_data.size
            if total_pixels > 100_000_000:  # 100 megapixels
                # Not an error, but could be slow/memory-intensive
                pass
        
        return True, ""

    def run(self):
        if not AI_MODELS_AVAILABLE and not SAM_MODELS_AVAILABLE:
            self.error_occurred.emit("AI libraries not found.")
            return
        
        # Validate parameters first
        is_valid, error_msg = self._validate_params()
        if not is_valid:
            self.error_occurred.emit(f"Parameter validation failed: {error_msg}")
            return

        try:
            if self.model_name == 'cellpose':
                self.run_cellpose()
            elif self.model_name == 'stardist':
                self.run_stardist()
            elif self.model_name == 'sam':
                self.run_sam()
            else:
                self.error_occurred.emit(f"Unknown model: {self.model_name}")
        except Exception as e:
            self.error_occurred.emit(f"AI model error: {str(e)}")

    def run_cellpose(self):
        if self.is_cancelled():
            return
        self.progress.emit("Initializing Cellpose model...", 10)
        model = models.Cellpose(model_type=self.params['model_type'])
        
        if self.is_cancelled():
            return
        self.progress.emit("Running Cellpose inference...", 40)
        masks, flows, styles, diams = model.eval(
            self.image_data, 
            diameter=self.params['diameter'] if self.params['diameter'] > 0 else None,
            channels=[0,0] 
        )
        
        if self.is_cancelled():
            return
        self.progress.emit("Finalizing segmentation...", 90)
        self.finished_segmentation.emit(masks, 'cellpose_labels')

    def run_stardist(self):
        if self.is_cancelled():
            return
        self.progress.emit("Initializing StarDist model...", 10)
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        
        if self.is_cancelled():
            return
        self.progress.emit("Normalizing data for StarDist...", 30)
        img_norm = normalize(self.image_data, 1, 99.8, axis=(0,1))
        
        if self.is_cancelled():
            return
        self.progress.emit("Running StarDist inference...", 50)
        labels, _ = model.predict_instances(img_norm)
        
        if self.is_cancelled():
            return
        self.progress.emit("Finalizing segmentation...", 90)
        self.finished_segmentation.emit(labels, 'stardist_labels')

    def run_sam(self):
        if self.is_cancelled():
            return
        self.progress.emit("Initializing SAM model...", 10)
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_type = "vit_b"
            checkpoint_path = get_sam_checkpoint(model_type)

            if self.is_cancelled():
                return
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=device)

            mask_generator = SamAutomaticMaskGenerator(sam)

            if self.is_cancelled():
                return
            self.progress.emit("Preparing image for SAM...", 30)
            
            # Handle different image formats more robustly
            image_rgb = self._prepare_image_for_sam()
            if image_rgb is None:
                self.error_occurred.emit("Failed to prepare image for SAM")
                return

            if self.is_cancelled():
                return
            self.progress.emit("Running SAM inference...", 50)
            masks = mask_generator.generate(image_rgb)

            if self.is_cancelled():
                return
            self.progress.emit("Finalizing SAM segmentation...", 90)
            if len(masks) > 0:
                sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
                label_image = np.zeros(sorted_masks[0]['segmentation'].shape, dtype=np.uint32)
                
                # Assign labels, handling overlapping masks by giving priority to smaller masks
                for i, mask in enumerate(reversed(sorted_masks)):
                    label_image[mask['segmentation']] = len(sorted_masks) - i
                    
                self.finished_segmentation.emit(label_image, 'sam_labels')
            else:
                self.error_occurred.emit("SAM did not find any objects.")

        except MemoryError:
            self.error_occurred.emit("Out of memory. Try using a smaller image or downsampling.")
        except Exception as e:
            self.error_occurred.emit(f"SAM model error: {str(e)}")
    
    def _prepare_image_for_sam(self):
        """
        Prepare image data for SAM, handling various input formats.
        Returns RGB uint8 image or None on failure.
        """
        try:
            data = self.image_data
            
            # Handle 2D grayscale
            if data.ndim == 2:
                # Normalize to 0-255 range
                if data.dtype != np.uint8:
                    data = self._normalize_to_uint8(data)
                return np.stack((data,) * 3, axis=-1)
            
            # Handle 3D with last dimension as channels
            elif data.ndim == 3:
                if data.shape[-1] == 3:
                    # Already RGB
                    if data.dtype != np.uint8:
                        data = self._normalize_to_uint8(data)
                    return data
                elif data.shape[-1] == 1:
                    # Single channel, expand to RGB
                    data = data[..., 0]
                    if data.dtype != np.uint8:
                        data = self._normalize_to_uint8(data)
                    return np.stack((data,) * 3, axis=-1)
                elif data.shape[0] in [1, 3]:
                    # Channel-first format
                    data = np.moveaxis(data, 0, -1)
                    if data.shape[-1] == 1:
                        data = np.repeat(data, 3, axis=-1)
                    if data.dtype != np.uint8:
                        data = self._normalize_to_uint8(data)
                    return data
                else:
                    # Take first slice of 3D stack
                    data = data[data.shape[0] // 2]  # Middle slice
                    if data.dtype != np.uint8:
                        data = self._normalize_to_uint8(data)
                    return np.stack((data,) * 3, axis=-1)
            
            return None
        except Exception:
            return None
    
    def _normalize_to_uint8(self, data):
        """Normalize image data to uint8 range (0-255)"""
        data = data.astype(np.float64)
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min) * 255
        return data.astype(np.uint8)


class AISegmentationWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        layer_group = QGroupBox("Select Image Layer")
        layer_layout = QVBoxLayout()
        self.layer_combo = QComboBox()
        layer_layout.addWidget(self.layer_combo)
        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)

        cellpose_group = QGroupBox("Cellpose Segmentation")
        cp_layout = QVBoxLayout(cellpose_group)
        
        cp_layout.addWidget(QLabel("Model Type:"))
        self.cp_model_type = QComboBox()
        self.cp_model_type.addItems(['cyto', 'nuclei', 'cyto2', 'cyto3'])
        cp_layout.addWidget(self.cp_model_type)

        cp_layout.addWidget(QLabel("Cell Diameter (pixels, 0=auto):"))
        self.cp_diameter = QSpinBox()
        self.cp_diameter.setRange(0, 500)
        self.cp_diameter.setValue(30)
        cp_layout.addWidget(self.cp_diameter)

        self.run_cellpose_btn = QPushButton("Run Cellpose")
        self.run_cellpose_btn.clicked.connect(self.run_cellpose)
        cp_layout.addWidget(self.run_cellpose_btn)
        layout.addWidget(cellpose_group)

        stardist_group = QGroupBox("StarDist Segmentation")
        sd_layout = QVBoxLayout(stardist_group)
        self.run_stardist_btn = QPushButton("Run StarDist (2D Fluorescent)")
        self.run_stardist_btn.clicked.connect(self.run_stardist)
        sd_layout.addWidget(self.run_stardist_btn)
        layout.addWidget(stardist_group)

        sam_group = QGroupBox("Segment Anything (SAM)")
        sam_layout = QVBoxLayout(sam_group)
        self.run_sam_btn = QPushButton("Run SAM (Automatic Mask Generation)")
        self.run_sam_btn.clicked.connect(self.run_sam)
        sam_layout.addWidget(self.run_sam_btn)
        layout.addWidget(sam_group)

        # Progress and status area
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_segmentation)
        self.cancel_btn.setVisible(False)
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.cancel_btn)

        layout.addStretch()
        self.setLayout(layout)

        if not AI_MODELS_AVAILABLE:
            cellpose_group.setDisabled(True)
            stardist_group.setDisabled(True)
        
        if not SAM_MODELS_AVAILABLE:
            sam_group.setDisabled(True)

        if not AI_MODELS_AVAILABLE and not SAM_MODELS_AVAILABLE:
            self.setDisabled(True)
            self.status_label.setText("Install cellpose, stardist, and segment-anything to enable.")
            self.status_label.setVisible(True)

    def run_cellpose(self):
        params = {
            'model_type': self.cp_model_type.currentText(),
            'diameter': self.cp_diameter.value()
        }
        self.start_segmentation('cellpose', params)

    def run_stardist(self):
        self.start_segmentation('stardist', {})

    def run_sam(self):
        self.start_segmentation('sam', {})

    def start_segmentation(self, model_name, params):
        if self.layer_combo.count() == 0:
            self.status_label.setText("No image layer to select.")
            self.status_label.setVisible(True)
            return

        layer = self.viewer.layers[self.layer_combo.currentText()]
        if not validate_image_layer(layer):
            self.status_label.setText("Please select a valid image layer.")
            self.status_label.setVisible(True)
            return

        self.set_buttons_enabled(False)
        self.cancel_btn.setVisible(True)
        self.thread = AISegmentationThread(model_name, layer.data, params)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished_segmentation.connect(self.on_finished)
        self.thread.error_occurred.connect(self.on_error)
        self.thread.start()

    def cancel_segmentation(self):
        """Cancel the currently running segmentation"""
        if self.thread is not None and self.thread.isRunning():
            self.thread.cancel()
            self.status_label.setText("Cancelling...")
            self.status_label.setVisible(True)
            # Wait for thread to finish (with timeout)
            self.thread.wait(5000)  # 5 second timeout
            self.on_cancelled()
    
    def on_cancelled(self):
        """Handle segmentation cancellation"""
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.status_label.setText("Segmentation cancelled.")
        self.set_buttons_enabled(True)

    def update_progress(self, message, value):
        self.status_label.setText(message)
        self.status_label.setVisible(True)
        self.progress_bar.setValue(value)
        self.progress_bar.setVisible(True)

    def on_finished(self, result_labels, name):
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.status_label.setVisible(False)
        self.set_buttons_enabled(True)
        self.viewer.add_labels(result_labels, name=name)

    def on_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.status_label.setText(f"Error: {error_message}")
        self.set_buttons_enabled(True)

    def set_buttons_enabled(self, enabled):
        self.run_cellpose_btn.setEnabled(enabled and AI_MODELS_AVAILABLE)
        self.run_stardist_btn.setEnabled(enabled and AI_MODELS_AVAILABLE)
        self.run_sam_btn.setEnabled(enabled and SAM_MODELS_AVAILABLE)

    def refresh_layers(self):
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.layer_combo.addItem(layer.name)

