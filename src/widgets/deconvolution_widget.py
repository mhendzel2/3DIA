"""
Deconvolution Widget for Scientific Image Analyzer
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QSpinBox, QComboBox, QGroupBox, QProgressBar)
from PyQt6.QtCore import QThread, pyqtSignal
import napari
from skimage.restoration import richardson_lucy, wiener

from utils.image_utils import validate_image_layer

class DeconvolutionThread(QThread):
    progress = pyqtSignal(str, int)
    finished_deconvolution = pyqtSignal(np.ndarray, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, algorithm, image_data, params):
        super().__init__()
        self.algorithm = algorithm
        self.image_data = image_data
        self.params = params

    def run(self):
        try:
            if self.algorithm == 'richardson_lucy':
                self.run_richardson_lucy()
            elif self.algorithm == 'wiener':
                self.run_wiener()
        except Exception as e:
            self.error_occurred.emit(f"Deconvolution error: {str(e)}")

    def run_richardson_lucy(self):
        self.progress.emit("Running Richardson-Lucy deconvolution...", 10)

        # Create a PSF (Point Spread Function)
        psf = np.ones((5, 5)) / 25

        deconvolved = richardson_lucy(
            self.image_data, psf, num_iter=self.params['iterations']
        )

        self.progress.emit("Finalizing deconvolution...", 90)
        self.finished_deconvolution.emit(deconvolved, 'richardson_lucy_deconvolved')

    def run_wiener(self):
        self.progress.emit("Running Wiener deconvolution...", 10)

        # Create a PSF (Point Spread Function)
        psf = np.ones((5, 5)) / 25

        deconvolved = wiener(self.image_data, psf, 1)

        self.progress.emit("Finalizing deconvolution...", 90)
        self.finished_deconvolution.emit(deconvolved, 'wiener_deconvolved')


class DeconvolutionWidget(QWidget):
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

        deconv_group = QGroupBox("Deconvolution Algorithm")
        deconv_layout = QVBoxLayout(deconv_group)

        deconv_layout.addWidget(QLabel("Algorithm:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(['richardson_lucy', 'wiener'])
        deconv_layout.addWidget(self.algo_combo)

        self.rl_iterations_label = QLabel("Iterations (Richardson-Lucy):")
        deconv_layout.addWidget(self.rl_iterations_label)
        self.rl_iterations = QSpinBox()
        self.rl_iterations.setRange(1, 100)
        self.rl_iterations.setValue(10)
        deconv_layout.addWidget(self.rl_iterations)

        self.run_btn = QPushButton("Run Deconvolution")
        self.run_btn.clicked.connect(self.run_deconvolution)
        deconv_layout.addWidget(self.run_btn)
        layout.addWidget(deconv_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

        self.algo_combo.currentTextChanged.connect(self.update_ui)
        self.viewer.layers.events.inserted.connect(self.update_layer_combo)
        self.viewer.layers.events.removed.connect(self.update_layer_combo)
        self.update_layer_combo()

    def update_layer_combo(self):
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.layer_combo.addItem(layer.name)

    def update_ui(self, text):
        if text == 'richardson_lucy':
            self.rl_iterations_label.setVisible(True)
            self.rl_iterations.setVisible(True)
        else:
            self.rl_iterations_label.setVisible(False)
            self.rl_iterations.setVisible(False)

    def run_deconvolution(self):
        layer_name = self.layer_combo.currentText()
        if not layer_name:
            self.status_label.setText("Please select an image layer.")
            self.status_label.setVisible(True)
            return

        layer = self.viewer.layers[layer_name]
        if not validate_image_layer(layer):
            self.status_label.setText("Please select a valid image layer.")
            self.status_label.setVisible(True)
            return

        algorithm = self.algo_combo.currentText()
        params = {}
        if algorithm == 'richardson_lucy':
            params['iterations'] = self.rl_iterations.value()

        self.set_buttons_enabled(False)
        self.thread = DeconvolutionThread(algorithm, layer.data, params)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished_deconvolution.connect(self.on_finished)
        self.thread.error_occurred.connect(self.on_error)
        self.thread.start()

    def update_progress(self, message, value):
        self.status_label.setText(message)
        self.status_label.setVisible(True)
        self.progress_bar.setValue(value)
        self.progress_bar.setVisible(True)

    def on_finished(self, result_image, name):
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.set_buttons_enabled(True)
        self.viewer.add_image(result_image, name=name)

    def on_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_message}")
        self.set_buttons_enabled(True)

    def set_buttons_enabled(self, enabled):
        self.run_btn.setEnabled(enabled)
