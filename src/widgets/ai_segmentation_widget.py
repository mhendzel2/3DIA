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

from utils.image_utils import validate_image_layer

try:
    from cellpose import models
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    AI_MODELS_AVAILABLE = True
except ImportError:
    AI_MODELS_AVAILABLE = False
    print("Warning: cellpose or stardist not installed. AI Segmentation widget will be disabled.")

class AISegmentationThread(QThread):
    progress = pyqtSignal(str, int)
    finished_segmentation = pyqtSignal(np.ndarray, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_name, image_data, params):
        super().__init__()
        self.model_name = model_name
        self.image_data = image_data
        self.params = params

    def run(self):
        if not AI_MODELS_AVAILABLE:
            self.error_occurred.emit("AI libraries (cellpose, stardist) not found.")
            return

        try:
            if self.model_name == 'cellpose':
                self.run_cellpose()
            elif self.model_name == 'stardist':
                self.run_stardist()
        except Exception as e:
            self.error_occurred.emit(f"AI model error: {str(e)}")

    def run_cellpose(self):
        self.progress.emit("Initializing Cellpose model...", 10)
        model = models.Cellpose(model_type=self.params['model_type'])
        
        self.progress.emit("Running Cellpose inference...", 40)
        masks, flows, styles, diams = model.eval(
            self.image_data, 
            diameter=self.params['diameter'],
            channels=[0,0] 
        )
        
        self.progress.emit("Finalizing segmentation...", 90)
        self.finished_segmentation.emit(masks, 'cellpose_labels')

    def run_stardist(self):
        self.progress.emit("Initializing StarDist model...", 10)
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        
        self.progress.emit("Normalizing data for StarDist...", 30)
        img_norm = normalize(self.image_data, 1, 99.8, axis=(0,1))
        
        self.progress.emit("Running StarDist inference...", 50)
        labels, _ = model.predict_instances(img_norm)
        
        self.progress.emit("Finalizing segmentation...", 90)
        self.finished_segmentation.emit(labels, 'stardist_labels')


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
        self.cp_model_type.addItems(['cyto', 'nuclei', 'cyto2'])
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

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

        if not AI_MODELS_AVAILABLE:
            self.setDisabled(True)
            self.status_label.setText("Install cellpose and stardist to enable.")
            self.status_label.setVisible(True)

    def run_cellpose(self):
        params = {
            'model_type': self.cp_model_type.currentText(),
            'diameter': self.cp_diameter.value()
        }
        self.start_segmentation('cellpose', params)

    def run_stardist(self):
        self.start_segmentation('stardist', {})

    def start_segmentation(self, model_name, params):
        layer = self.viewer.layers[self.layer_combo.currentText()] if self.layer_combo.count() > 0 else None
        if not validate_image_layer(layer):
            self.status_label.setText("Please select a valid image layer.")
            self.status_label.setVisible(True)
            return

        self.set_buttons_enabled(False)
        self.thread = AISegmentationThread(model_name, layer.data, params)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished_segmentation.connect(self.on_finished)
        self.thread.error_occurred.connect(self.on_error)
        self.thread.start()

    def update_progress(self, message, value):
        self.status_label.setText(message)
        self.status_label.setVisible(True)
        self.progress_bar.setValue(value)
        self.progress_bar.setVisible(True)

    def on_finished(self, result_labels, name):
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.set_buttons_enabled(True)
        self.viewer.add_labels(result_labels, name=name)

    def on_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_message}")
        self.set_buttons_enabled(True)

    def set_buttons_enabled(self, enabled):
        self.run_cellpose_btn.setEnabled(enabled)
        self.run_stardist_btn.setEnabled(enabled)
