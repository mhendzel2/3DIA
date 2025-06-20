"""
File I/O Widget for Scientific Image Analyzer
Handles importing and exporting of microscopy image files
"""

import os
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QComboBox, QSpinBox, 
                             QCheckBox, QGroupBox, QProgressBar, QTextEdit)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import napari
from napari.layers import Image

try:
    from aicsimageio import AICSImage
    HAS_AICSIMAGEIO = True
except ImportError:
    HAS_AICSIMAGEIO = False
    print("Warning: aicsimageio not available. Limited file format support.")

from utils.image_utils import get_supported_formats, estimate_memory_usage

class FileLoadThread(QThread):
    """Thread for loading large image files without blocking UI"""
    progress = pyqtSignal(int)
    finished_load = pyqtSignal(object, dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path, load_options=None):
        super().__init__()
        self.file_path = file_path
        self.load_options = load_options or {}
        
    def run(self):
        """Load image file in background thread"""
        try:
            self.progress.emit(10)
            
            if not HAS_AICSIMAGEIO:
                raise ImportError("aicsimageio is required for loading microscopy files")
                
            # Load image
            img = AICSImage(self.file_path)
            self.progress.emit(30)
            
            # Get image data with specified options
            scene = self.load_options.get('scene', 0)
            time_point = self.load_options.get('time_point', None)
            
            if time_point is not None:
                data = img.get_image_data("CZYX", S=scene, T=time_point)
            else:
                data = img.get_image_data("TCZYX", S=scene)
                
            self.progress.emit(60)
            
            # Prepare metadata
            metadata = {
                'name': Path(self.file_path).stem,
                'file_path': self.file_path,
                'dims': img.dims.order,
                'shape': data.shape,
                'dtype': data.dtype,
                'channel_names': img.channel_names if hasattr(img, 'channel_names') else None,
                'physical_pixel_sizes': {
                    'Z': img.physical_pixel_sizes.Z,
                    'Y': img.physical_pixel_sizes.Y,
                    'X': img.physical_pixel_sizes.X
                } if hasattr(img, 'physical_pixel_sizes') else None
            }
            
            self.progress.emit(90)
            
            # Calculate scale for napari
            scale = []
            if 'T' in img.dims.order:
                scale.append(1.0)  # Time scale
            if 'C' in img.dims.order and data.shape[img.dims.order.find('C')] > 1:
                scale.append(1.0)  # Channel scale
            if hasattr(img, 'physical_pixel_sizes'):
                if img.physical_pixel_sizes.Z:
                    scale.append(img.physical_pixel_sizes.Z)
                if img.physical_pixel_sizes.Y:
                    scale.append(img.physical_pixel_sizes.Y)
                if img.physical_pixel_sizes.X:
                    scale.append(img.physical_pixel_sizes.X)
            
            # Ensure scale matches data dimensions
            while len(scale) < data.ndim:
                scale.append(1.0)
            scale = scale[-data.ndim:]
            
            metadata['scale'] = scale
            
            self.progress.emit(100)
            self.finished_load.emit(data, metadata)
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to load {self.file_path}: {str(e)}")

class FileIOWidget(QWidget):
    """Widget for file input/output operations"""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.load_thread = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # File Import Section
        import_group = QGroupBox("Import Images")
        import_layout = QVBoxLayout()
        
        # Supported formats info
        formats_label = QLabel("Supported formats: " + ", ".join(get_supported_formats()))
        formats_label.setWordWrap(True)
        formats_label.setStyleSheet("color: #666; font-size: 10px;")
        import_layout.addWidget(formats_label)
        
        # Import buttons
        button_layout = QHBoxLayout()
        self.open_file_btn = QPushButton("Open File...")
        self.open_file_btn.clicked.connect(self.open_file_dialog)
        button_layout.addWidget(self.open_file_btn)
        
        self.open_series_btn = QPushButton("Open Series...")
        self.open_series_btn.clicked.connect(self.open_series_dialog)
        button_layout.addWidget(self.open_series_btn)
        
        import_layout.addLayout(button_layout)
        
        # Load options
        options_layout = QHBoxLayout()
        
        self.scene_combo = QComboBox()
        self.scene_combo.addItem("Scene 0", 0)
        options_layout.addWidget(QLabel("Scene:"))
        options_layout.addWidget(self.scene_combo)
        
        self.time_point_check = QCheckBox("Single timepoint:")
        self.time_point_spin = QSpinBox()
        self.time_point_spin.setMinimum(0)
        self.time_point_spin.setMaximum(9999)
        self.time_point_spin.setEnabled(False)
        self.time_point_check.toggled.connect(self.time_point_spin.setEnabled)
        
        options_layout.addWidget(self.time_point_check)
        options_layout.addWidget(self.time_point_spin)
        
        import_layout.addLayout(options_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        import_layout.addWidget(self.progress_bar)
        
        import_group.setLayout(import_layout)
        layout.addWidget(import_group)
        
        # Export Section
        export_group = QGroupBox("Export Images")
        export_layout = QVBoxLayout()
        
        # Layer selection for export
        self.export_layer_combo = QComboBox()
        self.update_layer_choices()
        export_layout.addWidget(QLabel("Layer to export:"))
        export_layout.addWidget(self.export_layer_combo)
        
        # Export buttons
        export_button_layout = QHBoxLayout()
        self.export_tiff_btn = QPushButton("Export as TIFF")
        self.export_tiff_btn.clicked.connect(self.export_as_tiff)
        export_button_layout.addWidget(self.export_tiff_btn)
        
        self.export_series_btn = QPushButton("Export Series")
        self.export_series_btn.clicked.connect(self.export_as_series)
        export_button_layout.addWidget(self.export_series_btn)
        
        export_layout.addLayout(export_button_layout)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Information display
        info_group = QGroupBox("File Information")
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(100)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def open_file_dialog(self):
        """Open file dialog for single file selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Microscopy Image",
            "",
            "All Supported (*.czi *.lif *.nd2 *.oib *.oif *.tif *.tiff *.ims *.lsm);;"
            "Zeiss CZI (*.czi);;"
            "Leica LIF (*.lif);;"
            "Nikon ND2 (*.nd2);;"
            "Olympus OIB/OIF (*.oib *.oif);;"
            "TIFF (*.tif *.tiff);;"
            "Imaris (*.ims);;"
            "LSM (*.lsm);;"
            "All Files (*)"
        )
        
        if file_path:
            self.load_image_file(file_path)
            
    def open_series_dialog(self):
        """Open directory dialog for image series"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory with Image Series"
        )
        
        if directory:
            # Find TIFF files in directory
            tiff_files = sorted([f for f in Path(directory).glob("*.tif*")])
            if tiff_files:
                self.load_image_series(tiff_files)
            else:
                self.show_error("No TIFF files found in selected directory")
                
    def load_image_file(self, file_path):
        """Load a single image file"""
        if not Path(file_path).exists():
            self.show_error(f"File not found: {file_path}")
            return
            
        # Check memory usage estimate
        try:
            memory_mb = estimate_memory_usage(file_path)
            if memory_mb > 2000:  # 2GB limit
                reply = self.show_warning(
                    f"Large file detected (~{memory_mb:.0f}MB). "
                    "Loading may take time and use significant memory. Continue?"
                )
                if not reply:
                    return
        except:
            pass  # Continue if estimation fails
            
        # Prepare load options
        load_options = {
            'scene': self.scene_combo.currentData(),
        }
        
        if self.time_point_check.isChecked():
            load_options['time_point'] = self.time_point_spin.value()
            
        # Start loading in background thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.open_file_btn.setEnabled(False)
        
        self.load_thread = FileLoadThread(file_path, load_options)
        self.load_thread.progress.connect(self.progress_bar.setValue)
        self.load_thread.finished_load.connect(self.on_image_loaded)
        self.load_thread.error_occurred.connect(self.on_load_error)
        self.load_thread.start()
        
    def load_image_series(self, file_paths):
        """Load multiple TIFF files as a series"""
        try:
            if not HAS_AICSIMAGEIO:
                raise ImportError("aicsimageio is required for loading image series")
                
            # Load first file to get dimensions
            first_img = AICSImage(str(file_paths[0]))
            first_data = first_img.get_image_data("YX")
            
            # Create array for series
            series_data = np.zeros((len(file_paths),) + first_data.shape, dtype=first_data.dtype)
            series_data[0] = first_data
            
            # Load remaining files
            for i, file_path in enumerate(file_paths[1:], 1):
                img = AICSImage(str(file_path))
                series_data[i] = img.get_image_data("YX")
                
            # Add to viewer
            self.viewer.add_image(
                series_data,
                name=f"Series_{Path(file_paths[0]).parent.name}",
                scale=[1.0] * series_data.ndim
            )
            
            self.update_info_display(f"Loaded series: {len(file_paths)} images")
            print(f"Loaded image series: {len(file_paths)} files")
            
        except Exception as e:
            self.show_error(f"Failed to load image series: {str(e)}")
            
    def on_image_loaded(self, data, metadata):
        """Handle successful image loading"""
        try:
            # Add image to viewer
            layer = self.viewer.add_image(
                data,
                name=metadata['name'],
                scale=metadata.get('scale', [1.0] * data.ndim),
                metadata=metadata
            )
            
            # Update info display
            info_text = f"Loaded: {metadata['name']}\n"
            info_text += f"Shape: {metadata['shape']}\n"
            info_text += f"Data type: {metadata['dtype']}\n"
            info_text += f"Dimensions: {metadata['dims']}"
            
            if metadata.get('channel_names'):
                info_text += f"\nChannels: {', '.join(metadata['channel_names'])}"
                
            self.update_info_display(info_text)
            
            print(f"Successfully loaded: {metadata['name']}")
            
        except Exception as e:
            self.show_error(f"Failed to add image to viewer: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.open_file_btn.setEnabled(True)
            
    def on_load_error(self, error_message):
        """Handle loading errors"""
        self.show_error(error_message)
        self.progress_bar.setVisible(False)
        self.open_file_btn.setEnabled(True)
        
    def export_as_tiff(self):
        """Export selected layer as TIFF"""
        current_layer = self.get_selected_export_layer()
        if current_layer is None:
            self.show_error("No layer selected for export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export as TIFF",
            f"{current_layer.name}.tif",
            "TIFF Files (*.tif *.tiff)"
        )
        
        if file_path:
            try:
                from tifffile import imwrite
                imwrite(file_path, current_layer.data)
                print(f"Exported to: {file_path}")
                self.update_info_display(f"Exported: {Path(file_path).name}")
            except Exception as e:
                self.show_error(f"Export failed: {str(e)}")
                
    def export_as_series(self):
        """Export layer as image series"""
        current_layer = self.get_selected_export_layer()
        if current_layer is None or current_layer.data.ndim < 3:
            self.show_error("Select a 3D+ layer for series export")
            return
            
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory"
        )
        
        if directory:
            try:
                from tifffile import imwrite
                data = current_layer.data
                
                # Export each slice
                for i in range(data.shape[0]):
                    filename = f"{current_layer.name}_slice_{i:04d}.tif"
                    file_path = Path(directory) / filename
                    imwrite(str(file_path), data[i])
                    
                print(f"Exported {data.shape[0]} slices to: {directory}")
                self.update_info_display(f"Exported series: {data.shape[0]} slices")
                
            except Exception as e:
                self.show_error(f"Series export failed: {str(e)}")
                
    def get_selected_export_layer(self):
        """Get the currently selected layer for export"""
        layer_name = self.export_layer_combo.currentText()
        if layer_name == "No layers available":
            return None
            
        for layer in self.viewer.layers:
            if layer.name == layer_name and isinstance(layer, Image):
                return layer
        return None
        
    def update_layer_choices(self):
        """Update the layer selection combo box"""
        self.export_layer_combo.clear()
        
        image_layers = [layer for layer in self.viewer.layers if isinstance(layer, Image)]
        
        if image_layers:
            for layer in image_layers:
                self.export_layer_combo.addItem(layer.name)
        else:
            self.export_layer_combo.addItem("No layers available")
            
    def update_info_display(self, text):
        """Update the information display"""
        self.info_text.setPlainText(text)
        
    def show_error(self, message):
        """Display error message"""
        print(f"ERROR: {message}")
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Error")
        msg.setText(message)
        msg.exec()
        
    def show_warning(self, message):
        """Display warning message and return user choice"""
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Warning")
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return msg.exec() == QMessageBox.StandardButton.Yes
        
    def cleanup(self):
        """Cleanup resources"""
        if self.load_thread and self.load_thread.isRunning():
            self.load_thread.quit()
            self.load_thread.wait()
