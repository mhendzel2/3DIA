"""
Enhanced Visualization Widget for Scientific Image Analyzer
Provides unified 2D/3D rendering controls using native napari layer properties.
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSlider, QDoubleSpinBox,
                             QComboBox, QGroupBox, QCheckBox)
from PyQt6.QtCore import Qt
import napari
from napari.layers import Image
import numpy as np

class VisualizationWidget(QWidget):
    """A widget to control the visualization of napari layers."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())
        
        # Layer selection dropdown
        self.layer_combo = QComboBox()
        self.layer_combo.addItems([layer.name for layer in self.viewer.layers if isinstance(layer, Image)])
        self.layer_combo.currentTextChanged.connect(self.on_layer_change)
        self.layout().addWidget(self.layer_combo)

        # Connect to viewer events to keep the layer list updated
        self.viewer.layers.events.inserted.connect(self._update_layer_combo)
        self.viewer.layers.events.removed.connect(self._update_layer_combo)

        # Rendering mode controls
        rendering_group = QGroupBox("Rendering")
        rendering_layout = QVBoxLayout()
        self.rendering_combo = QComboBox()
        self.rendering_combo.addItems(["mip", "translucent", "iso", "additive"])
        self.rendering_combo.currentTextChanged.connect(self.on_rendering_change)
        rendering_layout.addWidget(self.rendering_combo)
        rendering_group.setLayout(rendering_layout)
        self.layout().addWidget(rendering_group)

        # Blending mode controls
        blending_group = QGroupBox("Blending")
        blending_layout = QVBoxLayout()
        self.blending_combo = QComboBox()
        self.blending_combo.addItems(["translucent", "additive", "opaque"])
        self.blending_combo.currentTextChanged.connect(self.on_blending_change)
        blending_layout.addWidget(self.blending_combo)
        blending_group.setLayout(blending_layout)
        self.layout().addWidget(blending_group)

        # Colormap controls
        colormap_group = QGroupBox("Colormap")
        colormap_layout = QVBoxLayout()
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["gray", "viridis", "magma", "plasma"])
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_change)
        colormap_layout.addWidget(self.colormap_combo)
        self.layout().addWidget(colormap_group)

        # Gamma control
        gamma_group = QGroupBox("Gamma")
        gamma_layout = QHBoxLayout()
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(1, 200) # 0.01 to 2.0
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self.on_gamma_change)
        gamma_layout.addWidget(self.gamma_slider)
        gamma_group.setLayout(gamma_layout)
        self.layout().addWidget(gamma_group)

        # Contrast limits
        contrast_group = QGroupBox("Contrast")
        contrast_layout = QHBoxLayout()
        self.contrast_min = QDoubleSpinBox()
        self.contrast_max = QDoubleSpinBox()
        self.contrast_min.valueChanged.connect(self.on_contrast_change)
        self.contrast_max.valueChanged.connect(self.on_contrast_change)
        contrast_layout.addWidget(self.contrast_min)
        contrast_layout.addWidget(self.contrast_max)
        contrast_group.setLayout(contrast_layout)
        self.layout().addWidget(contrast_group)
        
        self.layout().addStretch()

        self.on_layer_change()

    def _update_layer_combo(self):
        """Update the layer combobox with the current image layers."""
        current_selection = self.layer_combo.currentText()
        self.layer_combo.clear()
        
        layers = [layer.name for layer in self.viewer.layers if isinstance(layer, Image)]
        self.layer_combo.addItems(layers)

        if current_selection in layers:
            self.layer_combo.setCurrentText(current_selection)

    def get_selected_layer(self):
        """Get the currently selected napari layer."""
        layer_name = self.layer_combo.currentText()
        if not layer_name:
            return None
        try:
            return self.viewer.layers[layer_name]
        except (KeyError, IndexError):
            return None

    def on_layer_change(self):
        """Update the widget controls when the selected layer changes."""
        layer = self.get_selected_layer()
        if not layer:
            return

        # Update rendering
        self.rendering_combo.setCurrentText(layer.rendering)
        # Update blending
        self.blending_combo.setCurrentText(layer.blending)
        # Update colormap
        self.colormap_combo.setCurrentText(layer.colormap.name)
        # Update gamma
        self.gamma_slider.setValue(int(layer.gamma * 100))

        # Update contrast limits
        data_min = float(np.min(layer.data))
        data_max = float(np.max(layer.data))
        if data_max < data_min:
            data_min, data_max = data_max, data_min
        low, high = layer.contrast_limits
        low = float(low)
        high = float(high)
        if high < low:
            low, high = high, low

        self.contrast_min.blockSignals(True)
        self.contrast_max.blockSignals(True)
        try:
            self.contrast_min.setRange(data_min, data_max)
            self.contrast_max.setRange(data_min, data_max)
            self.contrast_min.setValue(low)
            self.contrast_max.setValue(high)
        finally:
            self.contrast_min.blockSignals(False)
            self.contrast_max.blockSignals(False)

    def on_rendering_change(self, value):
        layer = self.get_selected_layer()
        if layer:
            layer.rendering = value

    def on_blending_change(self, value):
        layer = self.get_selected_layer()
        if layer:
            layer.blending = value

    def on_colormap_change(self, value):
        layer = self.get_selected_layer()
        if layer:
            layer.colormap = value

    def on_gamma_change(self, value):
        layer = self.get_selected_layer()
        if layer:
            layer.gamma = value / 100.0

    def on_contrast_change(self):
        layer = self.get_selected_layer()
        if layer:
            low = float(self.contrast_min.value())
            high = float(self.contrast_max.value())
            if high < low:
                low, high = high, low
                self.contrast_min.blockSignals(True)
                self.contrast_max.blockSignals(True)
                try:
                    self.contrast_min.setValue(low)
                    self.contrast_max.setValue(high)
                finally:
                    self.contrast_min.blockSignals(False)
                    self.contrast_max.blockSignals(False)
            layer.contrast_limits = (low, high)
