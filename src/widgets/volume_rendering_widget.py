"""
Volume Rendering Widget for Scientific Image Analyzer
Implements Imaris-like volume rendering capabilities including MIP, alpha blending, and orthogonal views
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSlider, QSpinBox, QDoubleSpinBox, 
                             QComboBox, QGroupBox, QCheckBox, QTabWidget)
from PyQt6.QtCore import Qt
import napari
from napari.layers import Image

try:
    from skimage import exposure
    from scipy import ndimage
    HAS_SCIKIT = True
except ImportError:
    HAS_SCIKIT = False

class VolumeRenderingWidget(QWidget):
    """Widget for advanced volume rendering similar to Imaris"""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.current_volume = None
        self.mip_cache = {}
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Layer selection
        layer_group = QGroupBox("Select Volume Layer")
        layer_layout = QVBoxLayout()
        
        self.layer_combo = QComboBox()
        self.update_layer_choices()
        layer_layout.addWidget(self.layer_combo)
        
        refresh_btn = QPushButton("Refresh Layers")
        refresh_btn.clicked.connect(self.update_layer_choices)
        layer_layout.addWidget(refresh_btn)
        
        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)
        
        # Rendering modes tabs
        tab_widget = QTabWidget()
        
        # MIP (Maximum Intensity Projection) Tab
        mip_tab = self.create_mip_tab()
        tab_widget.addTab(mip_tab, "MIP Rendering")
        
        # Alpha Blending Tab
        blend_tab = self.create_alpha_blending_tab()
        tab_widget.addTab(blend_tab, "Alpha Blending")
        
        # Orthogonal Views Tab
        ortho_tab = self.create_orthogonal_tab()
        tab_widget.addTab(ortho_tab, "Orthogonal Views")
        
        # Volume Clipping Tab
        clip_tab = self.create_clipping_tab()
        tab_widget.addTab(clip_tab, "Volume Clipping")
        
        layout.addWidget(tab_widget)
        
        # Status label
        self.status_label = QLabel("Select a volume layer to begin")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
        # Connect viewer events
        self.viewer.layers.events.inserted.connect(self.update_layer_choices)
        self.viewer.layers.events.removed.connect(self.update_layer_choices)
        
    def create_mip_tab(self):
        """Create MIP (Maximum Intensity Projection) controls"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Projection axis selection
        axis_group = QGroupBox("Projection Axis")
        axis_layout = QVBoxLayout()
        
        self.mip_axis_combo = QComboBox()
        self.mip_axis_combo.addItems(["Z-axis (XY view)", "Y-axis (XZ view)", "X-axis (YZ view)"])
        axis_layout.addWidget(self.mip_axis_combo)
        
        axis_group.setLayout(axis_layout)
        layout.addWidget(axis_group)
        
        # Projection range
        range_group = QGroupBox("Projection Range")
        range_layout = QVBoxLayout()
        
        # Start slice
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start Slice:"))
        self.mip_start_spin = QSpinBox()
        self.mip_start_spin.setRange(0, 999)
        self.mip_start_spin.setValue(0)
        start_layout.addWidget(self.mip_start_spin)
        range_layout.addLayout(start_layout)
        
        # End slice
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End Slice:"))
        self.mip_end_spin = QSpinBox()
        self.mip_end_spin.setRange(0, 999)
        self.mip_end_spin.setValue(999)
        end_layout.addWidget(self.mip_end_spin)
        range_layout.addLayout(end_layout)
        
        # Use all slices checkbox
        self.mip_all_check = QCheckBox("Use all slices")
        self.mip_all_check.setChecked(True)
        self.mip_all_check.toggled.connect(self.toggle_mip_range)
        range_layout.addWidget(self.mip_all_check)
        
        range_group.setLayout(range_layout)
        layout.addWidget(range_group)
        
        # Intensity adjustment
        intensity_group = QGroupBox("Intensity Adjustment")
        intensity_layout = QVBoxLayout()
        
        # Contrast
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        self.mip_contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.mip_contrast_slider.setRange(1, 200)
        self.mip_contrast_slider.setValue(100)
        contrast_layout.addWidget(self.mip_contrast_slider)
        self.mip_contrast_label = QLabel("1.00")
        contrast_layout.addWidget(self.mip_contrast_label)
        intensity_layout.addLayout(contrast_layout)
        self.mip_contrast_slider.valueChanged.connect(
            lambda v: self.mip_contrast_label.setText(f"{v/100:.2f}")
        )
        
        # Brightness
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("Brightness:"))
        self.mip_brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.mip_brightness_slider.setRange(-100, 100)
        self.mip_brightness_slider.setValue(0)
        brightness_layout.addWidget(self.mip_brightness_slider)
        self.mip_brightness_label = QLabel("0")
        brightness_layout.addWidget(self.mip_brightness_label)
        intensity_layout.addLayout(brightness_layout)
        self.mip_brightness_slider.valueChanged.connect(
            lambda v: self.mip_brightness_label.setText(str(v))
        )
        
        intensity_group.setLayout(intensity_layout)
        layout.addWidget(intensity_group)
        
        # Generate button
        generate_btn = QPushButton("Generate MIP")
        generate_btn.clicked.connect(self.generate_mip)
        generate_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(generate_btn)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_alpha_blending_tab(self):
        """Create alpha blending volume rendering controls"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Rendering method
        method_group = QGroupBox("Rendering Method")
        method_layout = QVBoxLayout()
        
        self.blend_method_combo = QComboBox()
        self.blend_method_combo.addItems([
            "Composite (Standard)",
            "Average Intensity",
            "Maximum Intensity",
            "Minimum Intensity",
            "Attenuated MIP"
        ])
        method_layout.addWidget(self.blend_method_combo)
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # Alpha/Opacity settings
        alpha_group = QGroupBox("Opacity Settings")
        alpha_layout = QVBoxLayout()
        
        # Global alpha
        global_alpha_layout = QHBoxLayout()
        global_alpha_layout.addWidget(QLabel("Global Opacity:"))
        self.global_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.global_alpha_slider.setRange(0, 100)
        self.global_alpha_slider.setValue(50)
        global_alpha_layout.addWidget(self.global_alpha_slider)
        self.global_alpha_label = QLabel("0.50")
        global_alpha_layout.addWidget(self.global_alpha_label)
        alpha_layout.addLayout(global_alpha_layout)
        self.global_alpha_slider.valueChanged.connect(
            lambda v: self.global_alpha_label.setText(f"{v/100:.2f}")
        )
        
        # Opacity threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Opacity Threshold:"))
        self.opacity_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_threshold_slider.setRange(0, 100)
        self.opacity_threshold_slider.setValue(10)
        threshold_layout.addWidget(self.opacity_threshold_slider)
        self.opacity_threshold_label = QLabel("10%")
        threshold_layout.addWidget(self.opacity_threshold_label)
        alpha_layout.addLayout(threshold_layout)
        self.opacity_threshold_slider.valueChanged.connect(
            lambda v: self.opacity_threshold_label.setText(f"{v}%")
        )
        
        alpha_group.setLayout(alpha_layout)
        layout.addWidget(alpha_group)
        
        # Sampling rate
        sample_group = QGroupBox("Sampling")
        sample_layout = QVBoxLayout()
        
        sample_layout.addWidget(QLabel("Sampling Rate (higher = better quality, slower):"))
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(1, 20)
        self.sample_rate_spin.setValue(5)
        self.sample_rate_spin.setSuffix(" samples/pixel")
        sample_layout.addWidget(self.sample_rate_spin)
        
        sample_group.setLayout(sample_layout)
        layout.addWidget(sample_group)
        
        # Render button
        render_btn = QPushButton("Render Volume")
        render_btn.clicked.connect(self.render_volume_blend)
        render_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(render_btn)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_orthogonal_tab(self):
        """Create orthogonal slice viewer controls"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        info_label = QLabel(
            "Orthogonal views show XY, XZ, and YZ cross-sections simultaneously.\n"
            "Click to generate orthogonal views at the current position."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # Position controls
        pos_group = QGroupBox("Cross-Section Position")
        pos_layout = QVBoxLayout()
        
        # X position
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X Position:"))
        self.ortho_x_spin = QSpinBox()
        self.ortho_x_spin.setRange(0, 999)
        x_layout.addWidget(self.ortho_x_spin)
        pos_layout.addLayout(x_layout)
        
        # Y position
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y Position:"))
        self.ortho_y_spin = QSpinBox()
        self.ortho_y_spin.setRange(0, 999)
        y_layout.addWidget(self.ortho_y_spin)
        pos_layout.addLayout(y_layout)
        
        # Z position
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z Position:"))
        self.ortho_z_spin = QSpinBox()
        self.ortho_z_spin.setRange(0, 999)
        z_layout.addWidget(self.ortho_z_spin)
        pos_layout.addLayout(z_layout)
        
        # Use center checkbox
        self.ortho_center_check = QCheckBox("Use volume center")
        self.ortho_center_check.setChecked(True)
        self.ortho_center_check.toggled.connect(self.toggle_ortho_position)
        pos_layout.addWidget(self.ortho_center_check)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # View options
        options_group = QGroupBox("Display Options")
        options_layout = QVBoxLayout()
        
        self.ortho_show_xy = QCheckBox("Show XY view (top)")
        self.ortho_show_xy.setChecked(True)
        options_layout.addWidget(self.ortho_show_xy)
        
        self.ortho_show_xz = QCheckBox("Show XZ view (front)")
        self.ortho_show_xz.setChecked(True)
        options_layout.addWidget(self.ortho_show_xz)
        
        self.ortho_show_yz = QCheckBox("Show YZ view (side)")
        self.ortho_show_yz.setChecked(True)
        options_layout.addWidget(self.ortho_show_yz)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Generate button
        ortho_btn = QPushButton("Generate Orthogonal Views")
        ortho_btn.clicked.connect(self.generate_orthogonal_views)
        ortho_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(ortho_btn)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_clipping_tab(self):
        """Create volume clipping controls"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        info_label = QLabel(
            "Volume clipping allows you to remove portions of the volume\n"
            "to reveal internal structures."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # Clipping planes
        planes_group = QGroupBox("Clipping Planes")
        planes_layout = QVBoxLayout()
        
        # X clipping
        x_clip_group = QGroupBox("X-axis Clipping")
        x_clip_layout = QVBoxLayout()
        self.clip_x_enable = QCheckBox("Enable X clipping")
        x_clip_layout.addWidget(self.clip_x_enable)
        
        x_min_layout = QHBoxLayout()
        x_min_layout.addWidget(QLabel("X Min:"))
        self.clip_x_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_x_min_slider.setRange(0, 100)
        self.clip_x_min_slider.setValue(0)
        x_min_layout.addWidget(self.clip_x_min_slider)
        self.clip_x_min_label = QLabel("0%")
        x_min_layout.addWidget(self.clip_x_min_label)
        x_clip_layout.addLayout(x_min_layout)
        self.clip_x_min_slider.valueChanged.connect(
            lambda v: self.clip_x_min_label.setText(f"{v}%")
        )
        
        x_max_layout = QHBoxLayout()
        x_max_layout.addWidget(QLabel("X Max:"))
        self.clip_x_max_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_x_max_slider.setRange(0, 100)
        self.clip_x_max_slider.setValue(100)
        x_max_layout.addWidget(self.clip_x_max_slider)
        self.clip_x_max_label = QLabel("100%")
        x_max_layout.addWidget(self.clip_x_max_label)
        x_clip_layout.addLayout(x_max_layout)
        self.clip_x_max_slider.valueChanged.connect(
            lambda v: self.clip_x_max_label.setText(f"{v}%")
        )
        
        x_clip_group.setLayout(x_clip_layout)
        planes_layout.addWidget(x_clip_group)
        
        # Y clipping
        y_clip_group = QGroupBox("Y-axis Clipping")
        y_clip_layout = QVBoxLayout()
        self.clip_y_enable = QCheckBox("Enable Y clipping")
        y_clip_layout.addWidget(self.clip_y_enable)
        
        y_min_layout = QHBoxLayout()
        y_min_layout.addWidget(QLabel("Y Min:"))
        self.clip_y_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_y_min_slider.setRange(0, 100)
        self.clip_y_min_slider.setValue(0)
        y_min_layout.addWidget(self.clip_y_min_slider)
        self.clip_y_min_label = QLabel("0%")
        y_min_layout.addWidget(self.clip_y_min_label)
        y_clip_layout.addLayout(y_min_layout)
        self.clip_y_min_slider.valueChanged.connect(
            lambda v: self.clip_y_min_label.setText(f"{v}%")
        )
        
        y_max_layout = QHBoxLayout()
        y_max_layout.addWidget(QLabel("Y Max:"))
        self.clip_y_max_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_y_max_slider.setRange(0, 100)
        self.clip_y_max_slider.setValue(100)
        y_max_layout.addWidget(self.clip_y_max_slider)
        self.clip_y_max_label = QLabel("100%")
        y_max_layout.addWidget(self.clip_y_max_label)
        y_clip_layout.addLayout(y_max_layout)
        self.clip_y_max_slider.valueChanged.connect(
            lambda v: self.clip_y_max_label.setText(f"{v}%")
        )
        
        y_clip_group.setLayout(y_clip_layout)
        planes_layout.addWidget(y_clip_group)
        
        # Z clipping
        z_clip_group = QGroupBox("Z-axis Clipping")
        z_clip_layout = QVBoxLayout()
        self.clip_z_enable = QCheckBox("Enable Z clipping")
        z_clip_layout.addWidget(self.clip_z_enable)
        
        z_min_layout = QHBoxLayout()
        z_min_layout.addWidget(QLabel("Z Min:"))
        self.clip_z_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_z_min_slider.setRange(0, 100)
        self.clip_z_min_slider.setValue(0)
        z_min_layout.addWidget(self.clip_z_min_slider)
        self.clip_z_min_label = QLabel("0%")
        z_min_layout.addWidget(self.clip_z_min_label)
        z_clip_layout.addLayout(z_min_layout)
        self.clip_z_min_slider.valueChanged.connect(
            lambda v: self.clip_z_min_label.setText(f"{v}%")
        )
        
        z_max_layout = QHBoxLayout()
        z_max_layout.addWidget(QLabel("Z Max:"))
        self.clip_z_max_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_z_max_slider.setRange(0, 100)
        self.clip_z_max_slider.setValue(100)
        z_max_layout.addWidget(self.clip_z_max_slider)
        self.clip_z_max_label = QLabel("100%")
        z_max_layout.addWidget(self.clip_z_max_label)
        z_clip_layout.addLayout(z_max_layout)
        self.clip_z_max_slider.valueChanged.connect(
            lambda v: self.clip_z_max_label.setText(f"{v}%")
        )
        
        z_clip_group.setLayout(z_clip_layout)
        planes_layout.addWidget(z_clip_group)
        
        planes_group.setLayout(planes_layout)
        layout.addWidget(planes_group)
        
        # Apply button
        apply_btn = QPushButton("Apply Clipping")
        apply_btn.clicked.connect(self.apply_clipping)
        apply_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(apply_btn)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def update_layer_choices(self):
        """Update the layer selection combo box"""
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and layer.data.ndim >= 3:
                self.layer_combo.addItem(layer.name)
    
    def get_selected_volume(self):
        """Get the currently selected volume layer"""
        layer_name = self.layer_combo.currentText()
        if not layer_name:
            self.status_label.setText("No volume layer selected")
            return None
        
        for layer in self.viewer.layers:
            if layer.name == layer_name and isinstance(layer, Image):
                if layer.data.ndim < 3:
                    self.status_label.setText("Selected layer is not 3D")
                    return None
                return layer.data
        
        self.status_label.setText("Selected layer not found")
        return None
    
    def toggle_mip_range(self, checked):
        """Toggle MIP range spinboxes"""
        self.mip_start_spin.setEnabled(not checked)
        self.mip_end_spin.setEnabled(not checked)
    
    def toggle_ortho_position(self, checked):
        """Toggle orthogonal position spinboxes"""
        self.ortho_x_spin.setEnabled(not checked)
        self.ortho_y_spin.setEnabled(not checked)
        self.ortho_z_spin.setEnabled(not checked)
    
    def generate_mip(self):
        """Generate Maximum Intensity Projection"""
        volume = self.get_selected_volume()
        if volume is None:
            return
        
        try:
            axis_idx = self.mip_axis_combo.currentIndex()
            
            # Determine slice range
            if self.mip_all_check.isChecked():
                start_slice = 0
                end_slice = volume.shape[axis_idx]
            else:
                start_slice = min(self.mip_start_spin.value(), volume.shape[axis_idx] - 1)
                end_slice = min(self.mip_end_spin.value() + 1, volume.shape[axis_idx])
            
            # Extract subvolume
            if axis_idx == 0:  # Z-axis (XY view)
                subvolume = volume[start_slice:end_slice, :, :]
                projection = np.max(subvolume, axis=0)
                view_name = "XY"
            elif axis_idx == 1:  # Y-axis (XZ view)
                subvolume = volume[:, start_slice:end_slice, :]
                projection = np.max(subvolume, axis=1)
                view_name = "XZ"
            else:  # X-axis (YZ view)
                subvolume = volume[:, :, start_slice:end_slice]
                projection = np.max(subvolume, axis=2)
                view_name = "YZ"
            
            # Apply intensity adjustments
            contrast = self.mip_contrast_slider.value() / 100.0
            brightness = self.mip_brightness_slider.value()
            
            # Adjust contrast and brightness
            projection = projection.astype(np.float32)
            mean_val = np.mean(projection)
            projection = (projection - mean_val) * contrast + mean_val + brightness
            projection = np.clip(projection, 0, projection.max())
            
            # Add to viewer
            layer_name = f"MIP_{view_name}_{self.layer_combo.currentText()}"
            if layer_name in [l.name for l in self.viewer.layers]:
                # Update existing layer
                for layer in self.viewer.layers:
                    if layer.name == layer_name:
                        layer.data = projection
                        break
            else:
                # Add new layer
                self.viewer.add_image(projection, name=layer_name, colormap='viridis')
            
            self.status_label.setText(f"MIP generated: {view_name} projection")
            self.status_label.setStyleSheet("color: green; font-style: italic;")
            
        except Exception as e:
            self.status_label.setText(f"Error generating MIP: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-style: italic;")
    
    def render_volume_blend(self):
        """Render volume using alpha blending"""
        volume = self.get_selected_volume()
        if volume is None:
            return
        
        try:
            method = self.blend_method_combo.currentText()
            global_alpha = self.global_alpha_slider.value() / 100.0
            threshold = self.opacity_threshold_slider.value() / 100.0
            
            # Normalize volume
            volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-10)
            
            if method == "Composite (Standard)":
                # Composite rendering with alpha blending
                rendered = self._composite_rendering(volume_norm, global_alpha, threshold)
            elif method == "Average Intensity":
                rendered = np.mean(volume_norm, axis=0)
            elif method == "Maximum Intensity":
                rendered = np.max(volume_norm, axis=0)
            elif method == "Minimum Intensity":
                rendered = np.min(volume_norm, axis=0)
            elif method == "Attenuated MIP":
                # MIP with depth attenuation
                rendered = self._attenuated_mip(volume_norm, global_alpha)
            else:
                rendered = np.max(volume_norm, axis=0)
            
            # Add to viewer
            layer_name = f"VolumeRender_{self.layer_combo.currentText()}"
            if layer_name in [l.name for l in self.viewer.layers]:
                for layer in self.viewer.layers:
                    if layer.name == layer_name:
                        layer.data = rendered
                        break
            else:
                self.viewer.add_image(rendered, name=layer_name, colormap='viridis')
            
            self.status_label.setText(f"Volume rendered using {method}")
            self.status_label.setStyleSheet("color: green; font-style: italic;")
            
        except Exception as e:
            self.status_label.setText(f"Error rendering volume: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-style: italic;")
    
    def _composite_rendering(self, volume, alpha, threshold):
        """Composite volume rendering with alpha blending"""
        depth, height, width = volume.shape
        rendered = np.zeros((height, width), dtype=np.float32)
        accumulated_alpha = np.zeros((height, width), dtype=np.float32)
        
        # Front-to-back compositing
        for z in range(depth):
            slice_data = volume[z, :, :]
            # Opacity based on intensity
            slice_alpha = np.where(slice_data > threshold, slice_data * alpha, 0)
            
            # Composite
            rendered += slice_data * slice_alpha * (1 - accumulated_alpha)
            accumulated_alpha += slice_alpha * (1 - accumulated_alpha)
            
            # Early ray termination
            if np.all(accumulated_alpha > 0.95):
                break
        
        return rendered
    
    def _attenuated_mip(self, volume, attenuation):
        """Maximum intensity projection with depth attenuation"""
        depth, height, width = volume.shape
        mip = np.zeros((height, width), dtype=np.float32)
        
        for z in range(depth):
            # Depth-dependent attenuation factor
            depth_factor = 1.0 - (z / depth) * attenuation
            attenuated_slice = volume[z, :, :] * depth_factor
            mip = np.maximum(mip, attenuated_slice)
        
        return mip
    
    def generate_orthogonal_views(self):
        """Generate orthogonal cross-section views"""
        volume = self.get_selected_volume()
        if volume is None:
            return
        
        try:
            # Determine positions
            if self.ortho_center_check.isChecked():
                z_pos = volume.shape[0] // 2
                y_pos = volume.shape[1] // 2
                x_pos = volume.shape[2] // 2
            else:
                z_pos = min(self.ortho_z_spin.value(), volume.shape[0] - 1)
                y_pos = min(self.ortho_y_spin.value(), volume.shape[1] - 1)
                x_pos = min(self.ortho_x_spin.value(), volume.shape[2] - 1)
            
            base_name = self.layer_combo.currentText()
            
            # XY view (top)
            if self.ortho_show_xy.isChecked():
                xy_slice = volume[z_pos, :, :]
                layer_name = f"Ortho_XY_{base_name}"
                self._add_or_update_layer(xy_slice, layer_name)
            
            # XZ view (front)
            if self.ortho_show_xz.isChecked():
                xz_slice = volume[:, y_pos, :]
                layer_name = f"Ortho_XZ_{base_name}"
                self._add_or_update_layer(xz_slice, layer_name)
            
            # YZ view (side)
            if self.ortho_show_yz.isChecked():
                yz_slice = volume[:, :, x_pos]
                layer_name = f"Ortho_YZ_{base_name}"
                self._add_or_update_layer(yz_slice, layer_name)
            
            self.status_label.setText(f"Orthogonal views generated at position ({x_pos}, {y_pos}, {z_pos})")
            self.status_label.setStyleSheet("color: green; font-style: italic;")
            
        except Exception as e:
            self.status_label.setText(f"Error generating orthogonal views: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-style: italic;")
    
    def apply_clipping(self):
        """Apply volume clipping"""
        volume = self.get_selected_volume()
        if volume is None:
            return
        
        try:
            clipped = volume.copy()
            
            # Apply X clipping
            if self.clip_x_enable.isChecked():
                x_min = int(volume.shape[2] * self.clip_x_min_slider.value() / 100)
                x_max = int(volume.shape[2] * self.clip_x_max_slider.value() / 100)
                clipped = clipped[:, :, x_min:x_max]
            
            # Apply Y clipping
            if self.clip_y_enable.isChecked():
                y_min = int(volume.shape[1] * self.clip_y_min_slider.value() / 100)
                y_max = int(volume.shape[1] * self.clip_y_max_slider.value() / 100)
                clipped = clipped[:, y_min:y_max, :]
            
            # Apply Z clipping
            if self.clip_z_enable.isChecked():
                z_min = int(volume.shape[0] * self.clip_z_min_slider.value() / 100)
                z_max = int(volume.shape[0] * self.clip_z_max_slider.value() / 100)
                clipped = clipped[z_min:z_max, :, :]
            
            # Add clipped volume to viewer
            layer_name = f"Clipped_{self.layer_combo.currentText()}"
            self._add_or_update_layer(clipped, layer_name)
            
            self.status_label.setText("Volume clipping applied")
            self.status_label.setStyleSheet("color: green; font-style: italic;")
            
        except Exception as e:
            self.status_label.setText(f"Error applying clipping: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-style: italic;")
    
    def _add_or_update_layer(self, data, name):
        """Add new layer or update existing one"""
        if name in [l.name for l in self.viewer.layers]:
            for layer in self.viewer.layers:
                if layer.name == name:
                    layer.data = data
                    break
        else:
            self.viewer.add_image(data, name=name, colormap='gray')
