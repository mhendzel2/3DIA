"""
Visualization Widget for Scientific Image Analyzer
Provides 3D rendering controls and visualization options
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSlider, QSpinBox, QDoubleSpinBox, 
                             QComboBox, QGroupBox, QCheckBox, QColorDialog,
                             QTabWidget, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
import napari
from napari.layers import Image, Surface, Points, Labels

from utils.image_utils import validate_image_layer

class VisualizationWidget(QWidget):
    """Widget for 3D visualization and rendering controls"""
    
    # Signals
    rendering_changed = pyqtSignal()
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.current_layer = None
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.apply_rendering_settings)
        self.update_timer.setSingleShot(True)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Layer selection
        layer_group = QGroupBox("Select Layer")
        layer_layout = QVBoxLayout()
        
        self.layer_combo = QComboBox()
        self.update_layer_choices()
        self.layer_combo.currentTextChanged.connect(self.on_layer_changed)
        layer_layout.addWidget(self.layer_combo)
        
        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)
        
        # Create tabs for different visualization aspects
        self.tab_widget = QTabWidget()
        
        # Rendering tab
        self.rendering_tab = self.create_rendering_tab()
        self.tab_widget.addTab(self.rendering_tab, "Rendering")
        
        # Colormap tab
        self.colormap_tab = self.create_colormap_tab()
        self.tab_widget.addTab(self.colormap_tab, "Colormaps")
        
        # 3D View tab
        self.view_tab = self.create_view_tab()
        self.tab_widget.addTab(self.view_tab, "3D View")
        
        # Animation tab
        self.animation_tab = self.create_animation_tab()
        self.tab_widget.addTab(self.animation_tab, "Animation")
        
        layout.addWidget(self.tab_widget)
        
        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QVBoxLayout()
        
        actions_row1 = QHBoxLayout()
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        actions_row1.addWidget(self.reset_view_btn)
        
        self.center_view_btn = QPushButton("Center on Data")
        self.center_view_btn.clicked.connect(self.center_on_data)
        actions_row1.addWidget(self.center_view_btn)
        
        actions_layout.addLayout(actions_row1)
        
        actions_row2 = QHBoxLayout()
        self.screenshot_btn = QPushButton("Take Screenshot")
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        actions_row2.addWidget(self.screenshot_btn)
        
        self.toggle_3d_btn = QPushButton("Toggle 3D")
        self.toggle_3d_btn.clicked.connect(self.toggle_3d_view)
        actions_row2.addWidget(self.toggle_3d_btn)
        
        actions_layout.addLayout(actions_row2)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def create_rendering_tab(self):
        """Create rendering controls tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Rendering method
        method_group = QGroupBox("Rendering Method")
        method_layout = QVBoxLayout()
        
        self.rendering_combo = QComboBox()
        self.rendering_combo.addItems([
            "translucent", "additive", "maximum", "minimum", 
            "average", "iso"
        ])
        self.rendering_combo.currentTextChanged.connect(self.schedule_update)
        method_layout.addWidget(self.rendering_combo)
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # Opacity controls
        opacity_group = QGroupBox("Opacity")
        opacity_layout = QVBoxLayout()
        
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self.schedule_update)
        
        self.opacity_label = QLabel("100%")
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_label.setText(f"{v}%")
        )
        
        opacity_controls = QHBoxLayout()
        opacity_controls.addWidget(QLabel("Opacity:"))
        opacity_controls.addWidget(self.opacity_slider)
        opacity_controls.addWidget(self.opacity_label)
        
        opacity_layout.addLayout(opacity_controls)
        opacity_group.setLayout(opacity_layout)
        layout.addWidget(opacity_group)
        
        # Contrast and brightness
        contrast_group = QGroupBox("Contrast & Brightness")
        contrast_layout = QVBoxLayout()
        
        # Contrast limits
        contrast_limits_layout = QHBoxLayout()
        contrast_limits_layout.addWidget(QLabel("Min:"))
        self.contrast_min = QSpinBox()
        self.contrast_min.setRange(0, 65535)
        self.contrast_min.setValue(0)
        self.contrast_min.valueChanged.connect(self.schedule_update)
        contrast_limits_layout.addWidget(self.contrast_min)
        
        contrast_limits_layout.addWidget(QLabel("Max:"))
        self.contrast_max = QSpinBox()
        self.contrast_max.setRange(0, 65535)
        self.contrast_max.setValue(255)
        self.contrast_max.valueChanged.connect(self.schedule_update)
        contrast_limits_layout.addWidget(self.contrast_max)
        
        contrast_layout.addLayout(contrast_limits_layout)
        
        # Auto contrast button
        self.auto_contrast_btn = QPushButton("Auto Contrast")
        self.auto_contrast_btn.clicked.connect(self.auto_contrast)
        contrast_layout.addWidget(self.auto_contrast_btn)
        
        # Gamma correction
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 3.0)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.valueChanged.connect(self.schedule_update)
        gamma_layout.addWidget(self.gamma_spin)
        
        contrast_layout.addLayout(gamma_layout)
        
        contrast_group.setLayout(contrast_layout)
        layout.addWidget(contrast_group)
        
        # ISO surface controls (for 3D data)
        iso_group = QGroupBox("ISO Surface")
        iso_layout = QVBoxLayout()
        
        iso_threshold_layout = QHBoxLayout()
        iso_threshold_layout.addWidget(QLabel("Threshold:"))
        self.iso_threshold = QSpinBox()
        self.iso_threshold.setRange(0, 65535)
        self.iso_threshold.setValue(128)
        self.iso_threshold.valueChanged.connect(self.schedule_update)
        iso_threshold_layout.addWidget(self.iso_threshold)
        
        iso_layout.addLayout(iso_threshold_layout)
        
        self.iso_enabled_check = QCheckBox("Enable ISO surface")
        self.iso_enabled_check.toggled.connect(self.schedule_update)
        iso_layout.addWidget(self.iso_enabled_check)
        
        iso_group.setLayout(iso_layout)
        layout.addWidget(iso_group)
        
        tab.setLayout(layout)
        return tab
        
    def create_colormap_tab(self):
        """Create colormap controls tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Colormap selection
        colormap_group = QGroupBox("Colormap Selection")
        colormap_layout = QVBoxLayout()
        
        self.colormap_combo = QComboBox()
        # Common scientific colormaps
        colormaps = [
            "gray", "viridis", "plasma", "inferno", "magma",
            "hot", "cool", "spring", "summer", "autumn", "winter",
            "jet", "hsv", "rainbow", "turbo",
            "red", "green", "blue", "cyan", "magenta", "yellow"
        ]
        self.colormap_combo.addItems(colormaps)
        self.colormap_combo.currentTextChanged.connect(self.schedule_update)
        colormap_layout.addWidget(self.colormap_combo)
        
        colormap_group.setLayout(colormap_layout)
        layout.addWidget(colormap_group)
        
        # Color range controls
        range_group = QGroupBox("Color Range")
        range_layout = QVBoxLayout()
        
        # Percentile clipping
        percentile_layout = QHBoxLayout()
        percentile_layout.addWidget(QLabel("Clip percentiles:"))
        
        self.low_percentile = QDoubleSpinBox()
        self.low_percentile.setRange(0, 50)
        self.low_percentile.setValue(0)
        self.low_percentile.setSingleStep(0.1)
        self.low_percentile.valueChanged.connect(self.schedule_update)
        percentile_layout.addWidget(self.low_percentile)
        
        percentile_layout.addWidget(QLabel("-"))
        
        self.high_percentile = QDoubleSpinBox()
        self.high_percentile.setRange(50, 100)
        self.high_percentile.setValue(100)
        self.high_percentile.setSingleStep(0.1)
        self.high_percentile.valueChanged.connect(self.schedule_update)
        percentile_layout.addWidget(self.high_percentile)
        
        range_layout.addLayout(percentile_layout)
        
        # Apply percentile button
        self.apply_percentile_btn = QPushButton("Apply Percentile Clipping")
        self.apply_percentile_btn.clicked.connect(self.apply_percentile_clipping)
        range_layout.addWidget(self.apply_percentile_btn)
        
        range_group.setLayout(range_layout)
        layout.addWidget(range_group)
        
        # Invert colormap
        self.invert_colormap_check = QCheckBox("Invert colormap")
        self.invert_colormap_check.toggled.connect(self.schedule_update)
        layout.addWidget(self.invert_colormap_check)
        
        tab.setLayout(layout)
        return tab
        
    def create_view_tab(self):
        """Create 3D view controls tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Camera controls
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QVBoxLayout()
        
        # View angles
        angles_layout = QHBoxLayout()
        
        self.azimuth_label = QLabel("Azimuth: 0°")
        angles_layout.addWidget(self.azimuth_label)
        
        self.elevation_label = QLabel("Elevation: 0°")
        angles_layout.addWidget(self.elevation_label)
        
        camera_layout.addLayout(angles_layout)
        
        # Predefined views
        views_layout = QHBoxLayout()
        
        self.front_view_btn = QPushButton("Front")
        self.front_view_btn.clicked.connect(lambda: self.set_view(0, 0))
        views_layout.addWidget(self.front_view_btn)
        
        self.side_view_btn = QPushButton("Side")
        self.side_view_btn.clicked.connect(lambda: self.set_view(90, 0))
        views_layout.addWidget(self.side_view_btn)
        
        self.top_view_btn = QPushButton("Top")
        self.top_view_btn.clicked.connect(lambda: self.set_view(0, 90))
        views_layout.addWidget(self.top_view_btn)
        
        camera_layout.addLayout(views_layout)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Zoom and pan
        zoom_group = QGroupBox("Zoom & Pan")
        zoom_layout = QVBoxLayout()
        
        zoom_controls = QHBoxLayout()
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_controls.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_controls.addWidget(self.zoom_out_btn)
        
        self.fit_view_btn = QPushButton("Fit to View")
        self.fit_view_btn.clicked.connect(self.fit_to_view)
        zoom_controls.addWidget(self.fit_view_btn)
        
        zoom_layout.addLayout(zoom_controls)
        zoom_group.setLayout(zoom_layout)
        layout.addWidget(zoom_group)
        
        # Clipping planes
        clipping_group = QGroupBox("Clipping Planes")
        clipping_layout = QVBoxLayout()
        
        self.enable_clipping_check = QCheckBox("Enable clipping")
        self.enable_clipping_check.toggled.connect(self.toggle_clipping)
        clipping_layout.addWidget(self.enable_clipping_check)
        
        # Clipping controls (initially disabled)
        self.clipping_controls = QWidget()
        clip_controls_layout = QVBoxLayout(self.clipping_controls)
        
        # Z clipping
        z_clip_layout = QHBoxLayout()
        z_clip_layout.addWidget(QLabel("Z range:"))
        
        self.z_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_min_slider.setRange(0, 100)
        self.z_min_slider.setValue(0)
        z_clip_layout.addWidget(self.z_min_slider)
        
        self.z_max_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_max_slider.setRange(0, 100)
        self.z_max_slider.setValue(100)
        z_clip_layout.addWidget(self.z_max_slider)
        
        clip_controls_layout.addLayout(z_clip_layout)
        
        self.clipping_controls.setEnabled(False)
        clipping_layout.addWidget(self.clipping_controls)
        
        clipping_group.setLayout(clipping_layout)
        layout.addWidget(clipping_group)
        
        tab.setLayout(layout)
        return tab
        
    def create_animation_tab(self):
        """Create animation controls tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Rotation animation
        rotation_group = QGroupBox("Rotation Animation")
        rotation_layout = QVBoxLayout()
        
        rotation_controls = QHBoxLayout()
        self.start_rotation_btn = QPushButton("Start Rotation")
        self.start_rotation_btn.clicked.connect(self.start_rotation_animation)
        rotation_controls.addWidget(self.start_rotation_btn)
        
        self.stop_rotation_btn = QPushButton("Stop Rotation")
        self.stop_rotation_btn.clicked.connect(self.stop_rotation_animation)
        self.stop_rotation_btn.setEnabled(False)
        rotation_controls.addWidget(self.stop_rotation_btn)
        
        rotation_layout.addLayout(rotation_controls)
        
        # Rotation speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.rotation_speed = QSlider(Qt.Orientation.Horizontal)
        self.rotation_speed.setRange(1, 10)
        self.rotation_speed.setValue(5)
        speed_layout.addWidget(self.rotation_speed)
        
        rotation_layout.addLayout(speed_layout)
        
        rotation_group.setLayout(rotation_layout)
        layout.addWidget(rotation_group)
        
        # Time series animation (if applicable)
        time_group = QGroupBox("Time Series Animation")
        time_layout = QVBoxLayout()
        
        time_controls = QHBoxLayout()
        self.play_time_btn = QPushButton("Play Time Series")
        self.play_time_btn.clicked.connect(self.play_time_series)
        time_controls.addWidget(self.play_time_btn)
        
        self.pause_time_btn = QPushButton("Pause")
        self.pause_time_btn.clicked.connect(self.pause_time_series)
        self.pause_time_btn.setEnabled(False)
        time_controls.addWidget(self.pause_time_btn)
        
        time_layout.addLayout(time_controls)
        
        # Frame rate
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(10)
        fps_layout.addWidget(self.fps_spin)
        
        time_layout.addLayout(fps_layout)
        
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        # Export animation
        export_group = QGroupBox("Export Animation")
        export_layout = QVBoxLayout()
        
        self.export_gif_btn = QPushButton("Export as GIF")
        self.export_gif_btn.clicked.connect(self.export_gif)
        export_layout.addWidget(self.export_gif_btn)
        
        self.export_mp4_btn = QPushButton("Export as MP4")
        self.export_mp4_btn.clicked.connect(self.export_mp4)
        export_layout.addWidget(self.export_mp4_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        tab.setLayout(layout)
        return tab
        
    def schedule_update(self):
        """Schedule rendering update with delay to avoid excessive updates"""
        self.update_timer.stop()
        self.update_timer.start(200)  # 200ms delay
        
    def apply_rendering_settings(self):
        """Apply current rendering settings to the selected layer"""
        if not self.current_layer:
            return
            
        try:
            # Apply rendering method
            if hasattr(self.current_layer, 'rendering'):
                self.current_layer.rendering = self.rendering_combo.currentText()
                
            # Apply opacity
            self.current_layer.opacity = self.opacity_slider.value() / 100.0
            
            # Apply colormap
            colormap_name = self.colormap_combo.currentText()
            if self.invert_colormap_check.isChecked():
                colormap_name += "_r"
            self.current_layer.colormap = colormap_name
            
            # Apply contrast limits
            contrast_limits = [self.contrast_min.value(), self.contrast_max.value()]
            self.current_layer.contrast_limits = contrast_limits
            
            # Apply gamma
            if hasattr(self.current_layer, 'gamma'):
                self.current_layer.gamma = self.gamma_spin.value()
                
            self.rendering_changed.emit()
            
        except Exception as e:
            print(f"Error applying rendering settings: {str(e)}")
            
    def on_layer_changed(self):
        """Handle layer selection change"""
        layer_name = self.layer_combo.currentText()
        if layer_name == "No layers available":
            self.current_layer = None
            return
            
        # Find the selected layer
        for layer in self.viewer.layers:
            if layer.name == layer_name:
                self.current_layer = layer
                self.update_controls_from_layer()
                break
                
    def update_controls_from_layer(self):
        """Update controls to match current layer settings"""
        if not self.current_layer:
            return
            
        try:
            # Block signals to prevent recursive updates
            self.opacity_slider.blockSignals(True)
            self.colormap_combo.blockSignals(True)
            self.contrast_min.blockSignals(True)
            self.contrast_max.blockSignals(True)
            self.gamma_spin.blockSignals(True)
            
            # Update opacity
            self.opacity_slider.setValue(int(self.current_layer.opacity * 100))
            
            # Update colormap
            colormap_name = str(self.current_layer.colormap)
            if colormap_name.endswith('_r'):
                colormap_name = colormap_name[:-2]
                self.invert_colormap_check.setChecked(True)
            else:
                self.invert_colormap_check.setChecked(False)
                
            index = self.colormap_combo.findText(colormap_name)
            if index >= 0:
                self.colormap_combo.setCurrentIndex(index)
                
            # Update contrast limits
            if hasattr(self.current_layer, 'contrast_limits'):
                limits = self.current_layer.contrast_limits
                if limits and len(limits) == 2:
                    self.contrast_min.setValue(int(limits[0]))
                    self.contrast_max.setValue(int(limits[1]))
                    
            # Update data range for controls
            if isinstance(self.current_layer, Image) and self.current_layer.data is not None:
                data_min = int(np.min(self.current_layer.data))
                data_max = int(np.max(self.current_layer.data))
                
                self.contrast_min.setRange(data_min, data_max)
                self.contrast_max.setRange(data_min, data_max)
                self.iso_threshold.setRange(data_min, data_max)
                
            # Re-enable signals
            self.opacity_slider.blockSignals(False)
            self.colormap_combo.blockSignals(False)
            self.contrast_min.blockSignals(False)
            self.contrast_max.blockSignals(False)
            self.gamma_spin.blockSignals(False)
            
        except Exception as e:
            print(f"Error updating controls: {str(e)}")
            
    def auto_contrast(self):
        """Automatically set contrast limits based on data percentiles"""
        if not self.current_layer or not isinstance(self.current_layer, Image):
            return
            
        try:
            data = self.current_layer.data
            low_percentile = np.percentile(data, 1)
            high_percentile = np.percentile(data, 99)
            
            self.contrast_min.setValue(int(low_percentile))
            self.contrast_max.setValue(int(high_percentile))
            
            self.apply_rendering_settings()
            
        except Exception as e:
            print(f"Auto contrast failed: {str(e)}")
            
    def apply_percentile_clipping(self):
        """Apply percentile-based clipping"""
        if not self.current_layer or not isinstance(self.current_layer, Image):
            return
            
        try:
            data = self.current_layer.data
            low_val = np.percentile(data, self.low_percentile.value())
            high_val = np.percentile(data, self.high_percentile.value())
            
            self.contrast_min.setValue(int(low_val))
            self.contrast_max.setValue(int(high_val))
            
            self.apply_rendering_settings()
            
        except Exception as e:
            print(f"Percentile clipping failed: {str(e)}")
            
    def reset_view(self):
        """Reset viewer to default view"""
        try:
            self.viewer.reset_view()
        except Exception as e:
            print(f"Reset view failed: {str(e)}")
            
    def center_on_data(self):
        """Center view on current data"""
        try:
            if self.current_layer:
                # Get layer bounds and center view
                bounds = self.current_layer.extent.world
                center = [(bounds[i][0] + bounds[i][1]) / 2 for i in range(len(bounds))]
                self.viewer.camera.center = center
        except Exception as e:
            print(f"Center on data failed: {str(e)}")
            
    def toggle_3d_view(self):
        """Toggle between 2D and 3D view"""
        try:
            if self.viewer.dims.ndisplay == 2:
                self.viewer.dims.ndisplay = 3
                self.toggle_3d_btn.setText("Toggle 2D")
            else:
                self.viewer.dims.ndisplay = 2
                self.toggle_3d_btn.setText("Toggle 3D")
        except Exception as e:
            print(f"Toggle 3D view failed: {str(e)}")
            
    def set_view(self, azimuth, elevation):
        """Set specific view angles"""
        try:
            # This would need to be implemented based on napari's camera API
            # For now, just update the labels
            self.azimuth_label.setText(f"Azimuth: {azimuth}°")
            self.elevation_label.setText(f"Elevation: {elevation}°")
        except Exception as e:
            print(f"Set view failed: {str(e)}")
            
    def zoom_in(self):
        """Zoom in the view"""
        try:
            self.viewer.camera.zoom *= 1.2
        except Exception as e:
            print(f"Zoom in failed: {str(e)}")
            
    def zoom_out(self):
        """Zoom out the view"""
        try:
            self.viewer.camera.zoom /= 1.2
        except Exception as e:
            print(f"Zoom out failed: {str(e)}")
            
    def fit_to_view(self):
        """Fit data to view"""
        try:
            # Reset camera to fit all layers
            self.viewer.reset_view()
        except Exception as e:
            print(f"Fit to view failed: {str(e)}")
            
    def toggle_clipping(self, enabled):
        """Enable/disable clipping plane controls"""
        self.clipping_controls.setEnabled(enabled)
        
    def take_screenshot(self):
        """Take a screenshot of the current view"""
        try:
            from PyQt6.QtWidgets import QFileDialog
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Screenshot",
                "screenshot.png",
                "PNG Files (*.png);;JPEG Files (*.jpg)"
            )
            
            if file_path:
                screenshot = self.viewer.screenshot()
                screenshot.save(file_path)
                print(f"Screenshot saved to: {file_path}")
                
        except Exception as e:
            print(f"Screenshot failed: {str(e)}")
            
    def start_rotation_animation(self):
        """Start rotation animation"""
        try:
            # This would need to be implemented with a timer
            # rotating the camera view
            self.start_rotation_btn.setEnabled(False)
            self.stop_rotation_btn.setEnabled(True)
            print("Rotation animation started")
        except Exception as e:
            print(f"Start rotation failed: {str(e)}")
            
    def stop_rotation_animation(self):
        """Stop rotation animation"""
        try:
            self.start_rotation_btn.setEnabled(True)
            self.stop_rotation_btn.setEnabled(False)
            print("Rotation animation stopped")
        except Exception as e:
            print(f"Stop rotation failed: {str(e)}")
            
    def play_time_series(self):
        """Play time series animation"""
        try:
            self.play_time_btn.setEnabled(False)
            self.pause_time_btn.setEnabled(True)
            print("Time series animation started")
        except Exception as e:
            print(f"Play time series failed: {str(e)}")
            
    def pause_time_series(self):
        """Pause time series animation"""
        try:
            self.play_time_btn.setEnabled(True)
            self.pause_time_btn.setEnabled(False)
            print("Time series animation paused")
        except Exception as e:
            print(f"Pause time series failed: {str(e)}")
            
    def export_gif(self):
        """Export animation as GIF"""
        try:
            from PyQt6.QtWidgets import QFileDialog
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export GIF",
                "animation.gif",
                "GIF Files (*.gif)"
            )
            
            if file_path:
                print(f"GIF export would save to: {file_path}")
                # Implementation would depend on napari's animation capabilities
                
        except Exception as e:
            print(f"GIF export failed: {str(e)}")
            
    def export_mp4(self):
        """Export animation as MP4"""
        try:
            from PyQt6.QtWidgets import QFileDialog
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export MP4",
                "animation.mp4",
                "MP4 Files (*.mp4)"
            )
            
            if file_path:
                print(f"MP4 export would save to: {file_path}")
                # Implementation would depend on napari's animation capabilities
                
        except Exception as e:
            print(f"MP4 export failed: {str(e)}")
            
    def update_layer_choices(self):
        """Update the layer selection combo box"""
        self.layer_combo.clear()
        
        # Get all image layers
        image_layers = [layer for layer in self.viewer.layers 
                       if isinstance(layer, (Image, Surface, Points, Labels))]
        
        if image_layers:
            for layer in image_layers:
                self.layer_combo.addItem(layer.name)
        else:
            self.layer_combo.addItem("No layers available")
            
    def cleanup(self):
        """Cleanup resources"""
        self.update_timer.stop()
