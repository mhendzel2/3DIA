"""
Segmentation Widget for Scientific Image Analyzer
Provides spot detection, surface creation, and object analysis tools
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSpinBox, QDoubleSpinBox, QComboBox, 
                             QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
                             QProgressBar, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import napari
from napari.layers import Image, Points, Surface, Labels

from skimage import measure, morphology, segmentation
from skimage.feature import blob_log, blob_dog, blob_doh
from scipy import ndimage
from scipy.spatial.distance import cdist

from utils.image_utils import validate_image_layer
from utils.analysis_utils import calculate_object_statistics

class SegmentationThread(QThread):
    """Thread for time-consuming segmentation operations"""
    progress = pyqtSignal(int)
    finished_segmentation = pyqtSignal(object, dict, str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, operation, data, parameters):
        super().__init__()
        self.operation = operation
        self.data = data
        self.parameters = parameters
        
    def run(self):
        """Execute segmentation operation"""
        try:
            if self.operation == "spots":
                self.detect_spots()
            elif self.operation == "surface":
                self.create_surface()
            elif self.operation == "watershed":
                self.watershed_segmentation()
            elif self.operation == "labels":
                self.label_objects()
        except Exception as e:
            self.error_occurred.emit(str(e))
            
    def detect_spots(self):
        """Detect spots using blob detection"""
        params = self.parameters
        method = params.get('method', 'log')
        
        self.progress.emit(20)
        
        if method == 'log':
            blobs = blob_log(
                self.data,
                min_sigma=params['min_sigma'],
                max_sigma=params['max_sigma'],
                num_sigma=params['num_sigma'],
                threshold=params['threshold']
            )
        elif method == 'dog':
            blobs = blob_dog(
                self.data,
                min_sigma=params['min_sigma'],
                max_sigma=params['max_sigma'],
                threshold=params['threshold']
            )
        elif method == 'doh':
            blobs = blob_doh(
                self.data,
                min_sigma=params['min_sigma'],
                max_sigma=params['max_sigma'],
                threshold=params['threshold']
            )
        else:
            raise ValueError(f"Unknown blob detection method: {method}")
            
        self.progress.emit(80)
        
        if blobs.shape[0] > 0:
            # Extract coordinates and radii
            coords = blobs[:, :-1]
            radii = blobs[:, -1]
            
            metadata = {
                'method': method,
                'count': len(blobs),
                'radii': radii,
                'parameters': params
            }
            
            self.progress.emit(100)
            self.finished_segmentation.emit(coords, metadata, 'points')
        else:
            self.error_occurred.emit("No spots detected with current parameters")
            
    def create_surface(self):
        """Create 3D surface using marching cubes"""
        params = self.parameters
        level = params.get('level', 100)
        
        self.progress.emit(30)
        
        if self.data.ndim < 3:
            self.error_occurred.emit("Surface creation requires 3D data")
            return
            
        try:
            verts, faces, normals, values = measure.marching_cubes(
                self.data, 
                level=level,
                spacing=params.get('spacing', (1, 1, 1))
            )
            
            self.progress.emit(80)
            
            surface_data = (verts, faces)
            metadata = {
                'level': level,
                'vertex_count': len(verts),
                'face_count': len(faces),
                'surface_area': measure.mesh_surface_area(verts, faces),
                'parameters': params
            }
            
            self.progress.emit(100)
            self.finished_segmentation.emit(surface_data, metadata, 'surface')
            
        except Exception as e:
            self.error_occurred.emit(f"Surface creation failed: {str(e)}")
            
    def watershed_segmentation(self):
        """Perform watershed segmentation"""
        params = self.parameters
        
        self.progress.emit(20)
        
        # Create markers
        if params.get('auto_markers', True):
            # Automatic marker detection
            distance = ndimage.distance_transform_edt(self.data > params['threshold'])
            coords = blob_log(distance, min_sigma=1, max_sigma=10, num_sigma=10, threshold=0.1)
            markers = np.zeros_like(self.data, dtype=int)
            for i, coord in enumerate(coords):
                if self.data.ndim == 2:
                    markers[int(coord[0]), int(coord[1])] = i + 1
                else:
                    markers[int(coord[0]), int(coord[1]), int(coord[2])] = i + 1
        else:
            # Use intensity peaks as markers
            markers = morphology.local_maxima(self.data)
            markers = measure.label(markers)
            
        self.progress.emit(50)
        
        # Apply watershed
        labels = segmentation.watershed(
            -self.data, 
            markers, 
            mask=self.data > params['threshold']
        )
        
        self.progress.emit(90)
        
        metadata = {
            'num_objects': len(np.unique(labels)) - 1,  # Exclude background
            'threshold': params['threshold'],
            'auto_markers': params.get('auto_markers', True)
        }
        
        self.progress.emit(100)
        self.finished_segmentation.emit(labels, metadata, 'labels')
        
    def label_objects(self):
        """Label connected components"""
        params = self.parameters
        
        self.progress.emit(30)
        
        # Threshold image
        binary = self.data > params['threshold']
        
        # Remove small objects if specified
        if params.get('min_size', 0) > 0:
            binary = morphology.remove_small_objects(binary, min_size=params['min_size'])
            
        self.progress.emit(60)
        
        # Label connected components
        labels = measure.label(binary)
        
        self.progress.emit(90)
        
        metadata = {
            'num_objects': len(np.unique(labels)) - 1,
            'threshold': params['threshold'],
            'min_size': params.get('min_size', 0)
        }
        
        self.progress.emit(100)
        self.finished_segmentation.emit(labels, metadata, 'labels')

class SegmentationWidget(QWidget):
    """Widget for segmentation and object detection"""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.segmentation_thread = None
        self.current_results = {}
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
        
        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)
        
        # Create tabs for different segmentation methods
        self.tab_widget = QTabWidget()
        
        # Spot Detection Tab
        self.spot_tab = self.create_spot_detection_tab()
        self.tab_widget.addTab(self.spot_tab, "Spot Detection")
        
        # Surface Creation Tab
        self.surface_tab = self.create_surface_tab()
        self.tab_widget.addTab(self.surface_tab, "3D Surfaces")
        
        # Watershed Tab
        self.watershed_tab = self.create_watershed_tab()
        self.tab_widget.addTab(self.watershed_tab, "Watershed")
        
        # Object Labeling Tab
        self.labels_tab = self.create_labels_tab()
        self.tab_widget.addTab(self.labels_tab, "Object Labels")
        
        layout.addWidget(self.tab_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results table
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setMaximumHeight(150)
        results_layout.addWidget(self.results_table)
        
        # Export results button
        self.export_results_btn = QPushButton("Export Results to CSV")
        self.export_results_btn.clicked.connect(self.export_results)
        self.export_results_btn.setEnabled(False)
        results_layout.addWidget(self.export_results_btn)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def create_spot_detection_tab(self):
        """Create spot detection tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Detection method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.spot_method_combo = QComboBox()
        self.spot_method_combo.addItems(["LOG (Laplacian of Gaussian)", 
                                       "DOG (Difference of Gaussian)", 
                                       "DOH (Determinant of Hessian)"])
        method_layout.addWidget(self.spot_method_combo)
        layout.addLayout(method_layout)
        
        # Size parameters
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Min size:"))
        self.min_sigma = QDoubleSpinBox()
        self.min_sigma.setRange(0.5, 50.0)
        self.min_sigma.setValue(1.0)
        self.min_sigma.setSingleStep(0.5)
        size_layout.addWidget(self.min_sigma)
        
        size_layout.addWidget(QLabel("Max size:"))
        self.max_sigma = QDoubleSpinBox()
        self.max_sigma.setRange(1.0, 100.0)
        self.max_sigma.setValue(10.0)
        self.max_sigma.setSingleStep(0.5)
        size_layout.addWidget(self.max_sigma)
        
        layout.addLayout(size_layout)
        
        # Threshold and sigma steps
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Threshold:"))
        self.spot_threshold = QDoubleSpinBox()
        self.spot_threshold.setRange(0.001, 1.0)
        self.spot_threshold.setValue(0.1)
        self.spot_threshold.setSingleStep(0.01)
        self.spot_threshold.setDecimals(3)
        params_layout.addWidget(self.spot_threshold)
        
        params_layout.addWidget(QLabel("Sigma steps:"))
        self.num_sigma = QSpinBox()
        self.num_sigma.setRange(1, 20)
        self.num_sigma.setValue(10)
        params_layout.addWidget(self.num_sigma)
        
        layout.addLayout(params_layout)
        
        # Detect button
        self.detect_spots_btn = QPushButton("Detect Spots")
        self.detect_spots_btn.clicked.connect(self.detect_spots)
        layout.addWidget(self.detect_spots_btn)
        
        tab.setLayout(layout)
        return tab
        
    def create_surface_tab(self):
        """Create surface creation tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Intensity level
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Intensity level:"))
        self.surface_level = QSpinBox()
        self.surface_level.setRange(0, 65535)
        self.surface_level.setValue(100)
        level_layout.addWidget(self.surface_level)
        
        # Auto-calculate level button
        self.auto_level_btn = QPushButton("Auto Level")
        self.auto_level_btn.clicked.connect(self.calculate_auto_level)
        level_layout.addWidget(self.auto_level_btn)
        
        layout.addLayout(level_layout)
        
        # Smoothing option
        self.smooth_surface_check = QCheckBox("Smooth surface")
        self.smooth_surface_check.setChecked(True)
        layout.addWidget(self.smooth_surface_check)
        
        # Create surface button
        self.create_surface_btn = QPushButton("Create 3D Surface")
        self.create_surface_btn.clicked.connect(self.create_surface)
        layout.addWidget(self.create_surface_btn)
        
        tab.setLayout(layout)
        return tab
        
    def create_watershed_tab(self):
        """Create watershed segmentation tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.watershed_threshold = QSpinBox()
        self.watershed_threshold.setRange(0, 65535)
        self.watershed_threshold.setValue(100)
        threshold_layout.addWidget(self.watershed_threshold)
        layout.addLayout(threshold_layout)
        
        # Marker options
        self.auto_markers_check = QCheckBox("Automatic marker detection")
        self.auto_markers_check.setChecked(True)
        layout.addWidget(self.auto_markers_check)
        
        # Watershed button
        self.watershed_btn = QPushButton("Apply Watershed")
        self.watershed_btn.clicked.connect(self.apply_watershed)
        layout.addWidget(self.watershed_btn)
        
        tab.setLayout(layout)
        return tab
        
    def create_labels_tab(self):
        """Create object labeling tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.labels_threshold = QSpinBox()
        self.labels_threshold.setRange(0, 65535)
        self.labels_threshold.setValue(100)
        threshold_layout.addWidget(self.labels_threshold)
        layout.addLayout(threshold_layout)
        
        # Minimum object size
        min_size_layout = QHBoxLayout()
        min_size_layout.addWidget(QLabel("Min object size:"))
        self.min_object_size = QSpinBox()
        self.min_object_size.setRange(0, 10000)
        self.min_object_size.setValue(100)
        min_size_layout.addWidget(self.min_object_size)
        layout.addLayout(min_size_layout)
        
        # Label objects button
        self.label_objects_btn = QPushButton("Label Objects")
        self.label_objects_btn.clicked.connect(self.label_objects)
        layout.addWidget(self.label_objects_btn)
        
        tab.setLayout(layout)
        return tab
        
    def get_current_layer(self):
        """Get the currently selected image layer"""
        layer_name = self.layer_combo.currentText()
        if layer_name == "No layers available":
            return None
            
        for layer in self.viewer.layers:
            if layer.name == layer_name and isinstance(layer, Image):
                return layer
        return None
        
    def detect_spots(self):
        """Detect spots in the selected image"""
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer")
            return
            
        method_text = self.spot_method_combo.currentText()
        method = method_text.split()[0].lower()  # Extract method name
        
        parameters = {
            'method': method,
            'min_sigma': self.min_sigma.value(),
            'max_sigma': self.max_sigma.value(),
            'threshold': self.spot_threshold.value(),
            'num_sigma': self.num_sigma.value()
        }
        
        self.start_segmentation("spots", layer.data, parameters)
        
    def create_surface(self):
        """Create 3D surface from the selected image"""
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer")
            return
            
        if layer.data.ndim < 3:
            self.show_error("Surface creation requires 3D data")
            return
            
        parameters = {
            'level': self.surface_level.value(),
            'spacing': layer.scale[-3:] if len(layer.scale) >= 3 else (1, 1, 1),
            'smooth': self.smooth_surface_check.isChecked()
        }
        
        self.start_segmentation("surface", layer.data, parameters)
        
    def apply_watershed(self):
        """Apply watershed segmentation"""
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer")
            return
            
        parameters = {
            'threshold': self.watershed_threshold.value(),
            'auto_markers': self.auto_markers_check.isChecked()
        }
        
        self.start_segmentation("watershed", layer.data, parameters)
        
    def label_objects(self):
        """Label connected objects"""
        layer = self.get_current_layer()
        if not validate_image_layer(layer):
            self.show_error("Please select a valid image layer")
            return
            
        parameters = {
            'threshold': self.labels_threshold.value(),
            'min_size': self.min_object_size.value()
        }
        
        self.start_segmentation("labels", layer.data, parameters)
        
    def calculate_auto_level(self):
        """Calculate automatic intensity level for surface creation"""
        layer = self.get_current_layer()
        if layer is None:
            return
            
        try:
            # Use Otsu's method for automatic threshold
            from skimage import filters
            auto_level = filters.threshold_otsu(layer.data)
            self.surface_level.setValue(int(auto_level))
            print(f"Auto-calculated surface level: {auto_level:.1f}")
        except Exception as e:
            self.show_error(f"Auto level calculation failed: {str(e)}")
            
    def start_segmentation(self, operation, data, parameters):
        """Start segmentation operation in background thread"""
        if self.segmentation_thread and self.segmentation_thread.isRunning():
            self.show_error("Segmentation already in progress")
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.set_buttons_enabled(False)
        
        self.segmentation_thread = SegmentationThread(operation, data, parameters)
        self.segmentation_thread.progress.connect(self.progress_bar.setValue)
        self.segmentation_thread.finished_segmentation.connect(self.on_segmentation_finished)
        self.segmentation_thread.error_occurred.connect(self.on_segmentation_error)
        self.segmentation_thread.start()
        
    def on_segmentation_finished(self, data, metadata, layer_type):
        """Handle successful segmentation completion"""
        try:
            layer_name = f"{self.get_current_layer().name}_{layer_type}"
            
            if layer_type == 'points':
                # Add points layer
                sizes = metadata.get('radii', 5) * 2  # Convert radius to size
                layer = self.viewer.add_points(
                    data,
                    name=layer_name,
                    size=sizes,
                    face_color='red',
                    edge_color='white',
                    edge_width=0.1
                )
                
                # Calculate statistics for spots
                stats = self.calculate_spot_statistics(data, metadata)
                
            elif layer_type == 'surface':
                # Add surface layer
                layer = self.viewer.add_surface(
                    data,
                    name=layer_name,
                    colormap='viridis',
                    opacity=0.7
                )
                
                # Use metadata as statistics
                stats = metadata
                
            elif layer_type == 'labels':
                # Add labels layer
                layer = self.viewer.add_labels(
                    data,
                    name=layer_name
                )
                
                # Calculate object statistics
                stats = calculate_object_statistics(data)
                
            # Store results and update table
            self.current_results[layer_name] = {
                'layer': layer,
                'statistics': stats,
                'metadata': metadata
            }
            
            self.update_results_table()
            self.export_results_btn.setEnabled(True)
            
            print(f"Segmentation completed: {layer_name}")
            
        except Exception as e:
            self.show_error(f"Failed to create layer: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.set_buttons_enabled(True)
            
    def on_segmentation_error(self, error_message):
        """Handle segmentation errors"""
        self.show_error(error_message)
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        
    def calculate_spot_statistics(self, coords, metadata):
        """Calculate statistics for detected spots"""
        stats = {
            'count': len(coords),
            'method': metadata['method']
        }
        
        if len(coords) > 1:
            # Calculate nearest neighbor distances
            distances = cdist(coords, coords)
            np.fill_diagonal(distances, np.inf)
            nn_distances = np.min(distances, axis=1)
            
            stats.update({
                'mean_nn_distance': np.mean(nn_distances),
                'std_nn_distance': np.std(nn_distances),
                'min_nn_distance': np.min(nn_distances),
                'max_nn_distance': np.max(nn_distances)
            })
            
        if 'radii' in metadata:
            radii = metadata['radii']
            stats.update({
                'mean_radius': np.mean(radii),
                'std_radius': np.std(radii),
                'min_radius': np.min(radii),
                'max_radius': np.max(radii)
            })
            
        return stats
        
    def update_results_table(self):
        """Update the results table with current statistics"""
        if not self.current_results:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            return
            
        # Collect all unique statistic keys
        all_keys = set()
        for result in self.current_results.values():
            all_keys.update(result['statistics'].keys())
        all_keys = sorted(list(all_keys))
        
        # Setup table
        self.results_table.setRowCount(len(self.current_results))
        self.results_table.setColumnCount(len(all_keys) + 1)  # +1 for layer name
        
        headers = ['Layer'] + all_keys
        self.results_table.setHorizontalHeaderLabels(headers)
        
        # Fill table
        for i, (layer_name, result) in enumerate(self.current_results.items()):
            self.results_table.setItem(i, 0, QTableWidgetItem(layer_name))
            
            stats = result['statistics']
            for j, key in enumerate(all_keys):
                value = stats.get(key, '')
                if isinstance(value, float):
                    value = f"{value:.3f}"
                self.results_table.setItem(i, j + 1, QTableWidgetItem(str(value)))
                
        self.results_table.resizeColumnsToContents()
        
    def export_results(self):
        """Export results to CSV file"""
        if not self.current_results:
            self.show_error("No results to export")
            return
            
        from PyQt6.QtWidgets import QFileDialog
        import csv
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "segmentation_results.csv",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                # Collect all unique keys
                all_keys = set()
                for result in self.current_results.values():
                    all_keys.update(result['statistics'].keys())
                all_keys = sorted(list(all_keys))
                
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write header
                    headers = ['Layer'] + all_keys
                    writer.writerow(headers)
                    
                    # Write data
                    for layer_name, result in self.current_results.items():
                        row = [layer_name]
                        stats = result['statistics']
                        for key in all_keys:
                            value = stats.get(key, '')
                            row.append(value)
                        writer.writerow(row)
                        
                print(f"Results exported to: {file_path}")
                
            except Exception as e:
                self.show_error(f"Export failed: {str(e)}")
                
    def set_buttons_enabled(self, enabled):
        """Enable/disable all operation buttons"""
        self.detect_spots_btn.setEnabled(enabled)
        self.create_surface_btn.setEnabled(enabled)
        self.watershed_btn.setEnabled(enabled)
        self.label_objects_btn.setEnabled(enabled)
        
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
        msg.setWindowTitle("Segmentation Error")
        msg.setText(message)
        msg.exec()
        
    def cleanup(self):
        """Cleanup resources"""
        if self.segmentation_thread and self.segmentation_thread.isRunning():
            self.segmentation_thread.quit()
            self.segmentation_thread.wait()
