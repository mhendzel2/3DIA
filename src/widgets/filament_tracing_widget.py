"""
Filament Tracing Widget for Scientific Image Analyzer
Implements neuron/cytoskeleton tracing similar to Imaris FilamentTracer
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSpinBox, QDoubleSpinBox, QComboBox, 
                             QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
                             QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import napari
from napari.layers import Image, Shapes, Points

try:
    from skimage import morphology, filters, measure
    from skimage.morphology import skeletonize, skeletonize_3d
    from scipy import ndimage
    from scipy.spatial.distance import cdist
    HAS_SCIKIT = True
except ImportError:
    HAS_SCIKIT = False

class FilamentTracingThread(QThread):
    """Thread for filament tracing operations"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, image, parameters):
        super().__init__()
        self.image = image
        self.parameters = parameters
        
    def run(self):
        """Execute filament tracing"""
        try:
            if not HAS_SCIKIT:
                self.error.emit("scikit-image is required for filament tracing")
                return
            
            self.progress.emit(10)
            
            # Step 1: Preprocessing - enhance filaments
            sigma = self.parameters.get('gaussian_sigma', 1.0)
            smoothed = filters.gaussian(self.image, sigma=sigma)
            self.progress.emit(20)
            
            # Step 2: Thresholding
            threshold_method = self.parameters.get('threshold_method', 'otsu')
            if threshold_method == 'otsu':
                threshold = filters.threshold_otsu(smoothed)
            elif threshold_method == 'li':
                threshold = filters.threshold_li(smoothed)
            else:
                threshold = self.parameters.get('manual_threshold', 100)
            
            binary = smoothed > threshold
            self.progress.emit(40)
            
            # Step 3: Clean up small objects
            min_size = self.parameters.get('min_object_size', 50)
            binary = morphology.remove_small_objects(binary, min_size=min_size)
            self.progress.emit(50)
            
            # Step 4: Skeletonization
            if binary.ndim == 2:
                skeleton = skeletonize(binary)
            else:
                skeleton = skeletonize_3d(binary)
            self.progress.emit(70)
            
            # Step 5: Analyze skeleton
            analysis_results = self._analyze_skeleton(skeleton, binary)
            self.progress.emit(90)
            
            # Step 6: Extract filaments
            filaments = self._extract_filaments(skeleton)
            analysis_results['filaments'] = filaments
            analysis_results['skeleton'] = skeleton
            analysis_results['binary'] = binary
            
            self.progress.emit(100)
            self.finished.emit(analysis_results)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def _analyze_skeleton(self, skeleton, binary):
        """Analyze skeleton properties"""
        # Label connected components
        labeled_skeleton = measure.label(skeleton)
        regions = measure.regionprops(labeled_skeleton)
        
        # Calculate properties
        total_length = np.sum(skeleton)
        num_filaments = len(regions)
        
        # Detect branch points (pixels with more than 2 neighbors)
        branch_points = self._detect_branch_points(skeleton)
        num_branches = len(branch_points)
        
        # Calculate average thickness from binary image
        distance_map = ndimage.distance_transform_edt(binary)
        avg_thickness = np.mean(distance_map[skeleton > 0]) * 2  # diameter
        
        return {
            'total_length': total_length,
            'num_filaments': num_filaments,
            'num_branches': num_branches,
            'branch_points': branch_points,
            'avg_thickness': avg_thickness,
            'regions': regions
        }
    
    def _detect_branch_points(self, skeleton):
        """Detect branch points in skeleton"""
        if skeleton.ndim == 2:
            # 2D case
            kernel = np.array([[1, 1, 1],
                              [1, 0, 1],
                              [1, 1, 1]])
            neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
            branch_mask = (skeleton > 0) & (neighbor_count > 2)
        else:
            # 3D case
            kernel = np.ones((3, 3, 3), dtype=int)
            kernel[1, 1, 1] = 0
            neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
            branch_mask = (skeleton > 0) & (neighbor_count > 2)
        
        branch_points = np.argwhere(branch_mask)
        return branch_points
    
    def _extract_filaments(self, skeleton):
        """Extract individual filament paths"""
        labeled = measure.label(skeleton)
        filaments = []
        
        for region in measure.regionprops(labeled):
            coords = region.coords
            # Sort coordinates to create a path
            if len(coords) > 1:
                # Use nearest neighbor to order points
                ordered_coords = self._order_coordinates(coords)
                filaments.append({
                    'coords': ordered_coords,
                    'length': len(ordered_coords),
                    'label': region.label
                })
        
        return filaments
    
    def _order_coordinates(self, coords):
        """Order coordinates to form a continuous path"""
        if len(coords) <= 2:
            return coords
        
        # Start with first point
        ordered = [coords[0]]
        remaining = list(coords[1:])
        
        while remaining:
            # Find nearest point to last ordered point
            distances = cdist([ordered[-1]], remaining)
            nearest_idx = np.argmin(distances)
            ordered.append(remaining[nearest_idx])
            remaining.pop(nearest_idx)
        
        return np.array(ordered)

class FilamentTracingWidget(QWidget):
    """Widget for filament tracing and analysis"""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.current_results = None
        self.tracing_thread = None
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
        
        refresh_btn = QPushButton("Refresh Layers")
        refresh_btn.clicked.connect(self.update_layer_choices)
        layer_layout.addWidget(refresh_btn)
        
        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)
        
        # Preprocessing parameters
        preprocess_group = QGroupBox("Preprocessing")
        preprocess_layout = QVBoxLayout()
        
        # Gaussian smoothing
        gauss_layout = QHBoxLayout()
        gauss_layout.addWidget(QLabel("Gaussian σ:"))
        self.gauss_sigma_spin = QDoubleSpinBox()
        self.gauss_sigma_spin.setRange(0.1, 10.0)
        self.gauss_sigma_spin.setValue(1.0)
        self.gauss_sigma_spin.setSingleStep(0.1)
        gauss_layout.addWidget(self.gauss_sigma_spin)
        preprocess_layout.addLayout(gauss_layout)
        
        preprocess_group.setLayout(preprocess_layout)
        layout.addWidget(preprocess_group)
        
        # Segmentation parameters
        segment_group = QGroupBox("Segmentation")
        segment_layout = QVBoxLayout()
        
        # Threshold method
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.threshold_method_combo = QComboBox()
        self.threshold_method_combo.addItems(["Otsu", "Li", "Manual"])
        self.threshold_method_combo.currentTextChanged.connect(self.toggle_manual_threshold)
        threshold_layout.addWidget(self.threshold_method_combo)
        segment_layout.addLayout(threshold_layout)
        
        # Manual threshold value
        manual_layout = QHBoxLayout()
        manual_layout.addWidget(QLabel("Manual value:"))
        self.manual_threshold_spin = QSpinBox()
        self.manual_threshold_spin.setRange(0, 65535)
        self.manual_threshold_spin.setValue(100)
        self.manual_threshold_spin.setEnabled(False)
        manual_layout.addWidget(self.manual_threshold_spin)
        segment_layout.addLayout(manual_layout)
        
        # Min object size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Min object size:"))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(10, 10000)
        self.min_size_spin.setValue(50)
        self.min_size_spin.setSuffix(" pixels")
        size_layout.addWidget(self.min_size_spin)
        segment_layout.addLayout(size_layout)
        
        segment_group.setLayout(segment_layout)
        layout.addWidget(segment_group)
        
        # Tracing options
        trace_group = QGroupBox("Tracing Options")
        trace_layout = QVBoxLayout()
        
        self.detect_branches_check = QCheckBox("Detect branch points")
        self.detect_branches_check.setChecked(True)
        trace_layout.addWidget(self.detect_branches_check)
        
        self.measure_thickness_check = QCheckBox("Measure filament thickness")
        self.measure_thickness_check.setChecked(True)
        trace_layout.addWidget(self.measure_thickness_check)
        
        self.extract_paths_check = QCheckBox("Extract filament paths")
        self.extract_paths_check.setChecked(True)
        trace_layout.addWidget(self.extract_paths_check)
        
        trace_group.setLayout(trace_layout)
        layout.addWidget(trace_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Trace button
        self.trace_btn = QPushButton("Trace Filaments")
        self.trace_btn.clicked.connect(self.trace_filaments)
        self.trace_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(self.trace_btn)
        
        # Results group
        results_group = QGroupBox("Filament Statistics")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Export button
        export_layout = QHBoxLayout()
        
        self.export_skeleton_btn = QPushButton("Export Skeleton")
        self.export_skeleton_btn.clicked.connect(self.export_skeleton)
        self.export_skeleton_btn.setEnabled(False)
        export_layout.addWidget(self.export_skeleton_btn)
        
        self.export_stats_btn = QPushButton("Export Statistics")
        self.export_stats_btn.clicked.connect(self.export_statistics)
        self.export_stats_btn.setEnabled(False)
        export_layout.addWidget(self.export_stats_btn)
        
        layout.addLayout(export_layout)
        
        # Status label
        self.status_label = QLabel("Select an image layer to begin")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Connect viewer events
        self.viewer.layers.events.inserted.connect(self.update_layer_choices)
        self.viewer.layers.events.removed.connect(self.update_layer_choices)
    
    def update_layer_choices(self):
        """Update the layer selection combo box"""
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.layer_combo.addItem(layer.name)
    
    def toggle_manual_threshold(self, method):
        """Enable/disable manual threshold input"""
        self.manual_threshold_spin.setEnabled(method == "Manual")
    
    def trace_filaments(self):
        """Start filament tracing"""
        layer_name = self.layer_combo.currentText()
        if not layer_name:
            self.status_label.setText("No layer selected")
            return
        
        # Get layer data
        layer = None
        for l in self.viewer.layers:
            if l.name == layer_name:
                layer = l
                break
        
        if layer is None or not isinstance(layer, Image):
            self.status_label.setText("Invalid layer selection")
            return
        
        image_data = layer.data
        
        # For 3D data, user might want to trace in 2D slices or full 3D
        if image_data.ndim > 2:
            # For now, use maximum intensity projection
            self.status_label.setText("Processing 3D data (using MIP)...")
            image_data = np.max(image_data, axis=0)
        
        # Prepare parameters
        parameters = {
            'gaussian_sigma': self.gauss_sigma_spin.value(),
            'threshold_method': self.threshold_method_combo.currentText().lower(),
            'manual_threshold': self.manual_threshold_spin.value(),
            'min_object_size': self.min_size_spin.value(),
            'detect_branches': self.detect_branches_check.isChecked(),
            'measure_thickness': self.measure_thickness_check.isChecked(),
            'extract_paths': self.extract_paths_check.isChecked()
        }
        
        # Start tracing thread
        self.tracing_thread = FilamentTracingThread(image_data, parameters)
        self.tracing_thread.progress.connect(self.update_progress)
        self.tracing_thread.finished.connect(self.on_tracing_finished)
        self.tracing_thread.error.connect(self.on_tracing_error)
        
        self.trace_btn.setEnabled(False)
        self.status_label.setText("Tracing filaments...")
        self.tracing_thread.start()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def on_tracing_finished(self, results):
        """Handle successful tracing completion"""
        self.current_results = results
        self.trace_btn.setEnabled(True)
        self.export_skeleton_btn.setEnabled(True)
        self.export_stats_btn.setEnabled(True)
        
        # Display results
        self.results_table.setRowCount(0)
        
        stats = [
            ("Total Length (pixels)", f"{results['total_length']:.1f}"),
            ("Number of Filaments", str(results['num_filaments'])),
            ("Number of Branch Points", str(results['num_branches'])),
            ("Average Thickness (pixels)", f"{results['avg_thickness']:.2f}")
        ]
        
        for row, (prop, value) in enumerate(stats):
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(prop))
            self.results_table.setItem(row, 1, QTableWidgetItem(value))
        
        # Add skeleton to viewer
        skeleton_name = f"Skeleton_{self.layer_combo.currentText()}"
        if skeleton_name in [l.name for l in self.viewer.layers]:
            for layer in self.viewer.layers:
                if layer.name == skeleton_name:
                    layer.data = results['skeleton'].astype(np.uint8) * 255
                    break
        else:
            self.viewer.add_image(
                results['skeleton'].astype(np.uint8) * 255,
                name=skeleton_name,
                colormap='red',
                blending='additive'
            )
        
        # Add branch points if detected
        if len(results['branch_points']) > 0:
            branch_name = f"BranchPoints_{self.layer_combo.currentText()}"
            # Convert to napari coordinate format
            if branch_name in [l.name for l in self.viewer.layers]:
                for layer in self.viewer.layers:
                    if layer.name == branch_name:
                        layer.data = results['branch_points']
                        break
            else:
                self.viewer.add_points(
                    results['branch_points'],
                    name=branch_name,
                    size=5,
                    face_color='yellow',
                    edge_color='black'
                )
        
        self.status_label.setText(f"Tracing complete: {results['num_filaments']} filaments found")
        self.status_label.setStyleSheet("color: green; font-style: italic;")
        self.progress_bar.setValue(100)
    
    def on_tracing_error(self, error_message):
        """Handle tracing errors"""
        self.trace_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("color: red; font-style: italic;")
        self.progress_bar.setValue(0)
    
    def export_skeleton(self):
        """Export skeleton to file"""
        if self.current_results is None:
            return
        
        from PyQt6.QtWidgets import QFileDialog
        import tifffile
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Skeleton",
            "",
            "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        
        if filename:
            try:
                skeleton = self.current_results['skeleton'].astype(np.uint8) * 255
                tifffile.imwrite(filename, skeleton)
                self.status_label.setText(f"Skeleton exported to {filename}")
                self.status_label.setStyleSheet("color: green; font-style: italic;")
            except Exception as e:
                self.status_label.setText(f"Export failed: {str(e)}")
                self.status_label.setStyleSheet("color: red; font-style: italic;")
    
    def export_statistics(self):
        """Export filament statistics to CSV"""
        if self.current_results is None:
            return
        
        from PyQt6.QtWidgets import QFileDialog
        import csv
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Statistics",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Property", "Value"])
                    writer.writerow(["Total Length (pixels)", self.current_results['total_length']])
                    writer.writerow(["Number of Filaments", self.current_results['num_filaments']])
                    writer.writerow(["Number of Branch Points", self.current_results['num_branches']])
                    writer.writerow(["Average Thickness (pixels)", f"{self.current_results['avg_thickness']:.2f}"])
                    
                    # Add individual filament data
                    writer.writerow([])
                    writer.writerow(["Filament ID", "Length (pixels)"])
                    for i, filament in enumerate(self.current_results.get('filaments', [])):
                        writer.writerow([i+1, filament['length']])
                
                self.status_label.setText(f"Statistics exported to {filename}")
                self.status_label.setStyleSheet("color: green; font-style: italic;")
            except Exception as e:
                self.status_label.setText(f"Export failed: {str(e)}")
                self.status_label.setStyleSheet("color: red; font-style: italic;")
