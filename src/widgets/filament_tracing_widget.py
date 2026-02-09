"""
Filament Tracing Widget for Scientific Image Analyzer
Implements neuron/cytoskeleton tracing similar to Imaris FilamentTracer
"""

import inspect

import numpy as np
from napari.layers import Image
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pymaris.data_model import ImageVolume, infer_axes_from_shape
from pymaris.jobs import JobCancelledError, JobRunner
from pymaris.workflow import WorkflowResult, WorkflowStep
from pymaris_napari.provenance import record_ui_workflow_result
from pymaris_napari.settings import load_project_store_settings, resolve_project_store_dir

try:
    from scipy import ndimage
    from scipy.spatial.distance import cdist
    from skimage import filters, measure, morphology
    from skimage.morphology import skeletonize

    try:
        from skimage.morphology import skeletonize_3d
    except Exception:  # pragma: no cover - optional in newer skimage
        skeletonize_3d = None

    HAS_SCIKIT = True
except ImportError:
    skeletonize_3d = None
    HAS_SCIKIT = False


def _remove_small_objects(mask, min_size):
    if not HAS_SCIKIT:
        return mask
    params = inspect.signature(morphology.remove_small_objects).parameters
    if "max_size" in params:
        return morphology.remove_small_objects(mask, max_size=max(0, int(min_size) - 1))
    return morphology.remove_small_objects(mask, min_size=min_size)

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
            binary = _remove_small_objects(binary, min_size)
            self.progress.emit(50)

            # Step 4: Skeletonization
            if binary.ndim == 2:
                skeleton = skeletonize(binary)
            else:
                if skeletonize_3d is not None:
                    skeleton = skeletonize_3d(binary)
                else:
                    skeleton = np.asarray([skeletonize(frame) for frame in binary], dtype=np.uint8)
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
    workflow_progress = pyqtSignal(int, str)
    workflow_finished = pyqtSignal(object)
    workflow_failed = pyqtSignal(str)
    workflow_cancelled = pyqtSignal(str)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.current_results = None
        self.tracing_thread = None
        self.job_runner = JobRunner(max_workers=2)
        self.workflow_handle = None
        self.workflow_step = None
        self.last_workflow_result = None
        self.project_store_settings = load_project_store_settings()
        self.project_session_dir_cache = None
        self.init_ui()
        self.workflow_progress.connect(self.on_workflow_progress)
        self.workflow_finished.connect(self.on_workflow_finished)
        self.workflow_failed.connect(self.on_workflow_error)
        self.workflow_cancelled.connect(self.on_workflow_cancelled)

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
        self.cancel_btn = QPushButton("Cancel Tracing")
        self.cancel_btn.clicked.connect(self.cancel_tracing)
        self.cancel_btn.setEnabled(False)
        layout.addWidget(self.cancel_btn)

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
        if not HAS_SCIKIT:
            self.status_label.setText("scikit-image is required for filament tracing")
            self.status_label.setStyleSheet("color: red; font-style: italic;")
            return
        if self.workflow_handle and not self.workflow_handle.done():
            self.status_label.setText("Tracing already in progress")
            return

        layer_name = self.layer_combo.currentText()
        if not layer_name:
            self.status_label.setText("No layer selected")
            return

        # Get layer data
        layer = None
        for layer_item in self.viewer.layers:
            if layer_item.name == layer_name:
                layer = layer_item
                break

        if layer is None or not isinstance(layer, Image):
            self.status_label.setText("Invalid layer selection")
            return

        image_data = np.asarray(layer.data)

        # For 3D data, user might want to trace in 2D slices or full 3D
        if image_data.ndim > 2:
            # For now, use maximum intensity projection
            self.status_label.setText("Processing 3D data (using MIP)...")
            image_data = np.max(image_data, axis=0)

        # Prepare parameters for core tracing backend
        parameters = {
            'gaussian_sigma': self.gauss_sigma_spin.value(),
            'threshold_method': self.threshold_method_combo.currentText().lower(),
            'manual_threshold': self.manual_threshold_spin.value(),
            'min_object_size': self.min_size_spin.value(),
            'detect_branches': self.detect_branches_check.isChecked(),
            'measure_thickness': self.measure_thickness_check.isChecked(),
            'extract_paths': self.extract_paths_check.isChecked()
        }

        image = ImageVolume(
            array=image_data,
            axes=infer_axes_from_shape(image_data.shape),
            metadata={"name": layer.name},
        )

        self.workflow_step = WorkflowStep(
            id=f"tracing-{layer_name}-{self.min_size_spin.value()}",
            name="tracing:skeleton",
            backend_type="tracing",
            backend_name="skeleton",
            params=parameters,
            inputs=["image"],
            outputs=["trace"],
        )
        self.last_workflow_result = None
        self.trace_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("Tracing filaments...")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        self.progress_bar.setValue(0)
        self.workflow_handle = self.job_runner.submit(
            step=self.workflow_step,
            context={"image": image},
            on_progress=lambda p, m: self.workflow_progress.emit(p, m),
        )
        self.workflow_handle.future.add_done_callback(self._on_tracing_job_done)

    def cancel_tracing(self):
        """Cancel active tracing operation."""
        if self.workflow_handle and not self.workflow_handle.done():
            self.workflow_handle.cancel()
        if self.tracing_thread and self.tracing_thread.isRunning():
            self.tracing_thread.terminate()
        self.cancel_btn.setEnabled(False)

    def _on_tracing_job_done(self, future):
        """Bridge workflow completion back to Qt thread signals."""
        try:
            result = future.result()
            self.workflow_finished.emit(result)
        except JobCancelledError as exc:
            self.workflow_cancelled.emit(str(exc))
        except Exception as exc:
            self.workflow_failed.emit(str(exc))

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def on_workflow_progress(self, value, _message):
        """Handle workflow progress updates."""
        self.update_progress(int(value))

    def on_workflow_finished(self, result):
        """Normalize workflow payload and forward to legacy UI rendering."""
        self.last_workflow_result = result if isinstance(result, WorkflowResult) else None
        normalized = self._normalize_tracing_results(result)
        self.on_tracing_finished(normalized)

    def on_workflow_error(self, message):
        """Handle workflow tracing errors."""
        self.workflow_handle = None
        self.workflow_step = None
        self.cancel_btn.setEnabled(False)
        self.on_tracing_error(message)

    def on_workflow_cancelled(self, message):
        """Handle workflow tracing cancellation."""
        self.workflow_handle = None
        self.workflow_step = None
        self.cancel_btn.setEnabled(False)
        self.trace_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(message or "Tracing cancelled")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")

    def _normalize_tracing_results(self, payload):
        """Normalize workflow and legacy tracing payloads."""
        if isinstance(payload, WorkflowResult):
            graph = payload.outputs.get("trace", {})
            if not isinstance(graph, dict):
                graph = {}
            table = payload.tables.get("trace_table", {})
            if not isinstance(table, dict):
                table = {}
            skeleton = np.asarray(graph.get("skeleton", np.zeros((1, 1), dtype=np.uint8)))
            branch_points = np.asarray(
                graph.get("branch_points", np.empty((0, skeleton.ndim), dtype=int))
            )
            filaments = graph.get("filaments", [])
            total_length = float(graph.get("total_length", table.get("total_length", np.sum(skeleton > 0))))
            num_filaments = int(graph.get("num_filaments", table.get("num_filaments", len(filaments))))
            num_branches = int(graph.get("num_branches", table.get("num_branches", len(branch_points))))
            avg_thickness = float(graph.get("avg_thickness", table.get("avg_thickness", 0.0)))
            return {
                "skeleton": skeleton,
                "binary": np.asarray(graph.get("binary", skeleton > 0)),
                "filaments": filaments if isinstance(filaments, list) else [],
                "branch_points": branch_points,
                "total_length": total_length,
                "num_filaments": num_filaments,
                "num_branches": num_branches,
                "avg_thickness": avg_thickness,
            }
        if isinstance(payload, dict):
            normalized = dict(payload)
            normalized["skeleton"] = np.asarray(normalized.get("skeleton", np.zeros((1, 1), dtype=np.uint8)))
            normalized["branch_points"] = np.asarray(
                normalized.get("branch_points", np.empty((0, normalized["skeleton"].ndim), dtype=int))
            )
            normalized.setdefault("filaments", [])
            normalized.setdefault("total_length", float(np.sum(normalized["skeleton"] > 0)))
            normalized.setdefault("num_filaments", len(normalized["filaments"]))
            normalized.setdefault("num_branches", len(normalized["branch_points"]))
            normalized.setdefault("avg_thickness", 0.0)
            normalized.setdefault("binary", normalized["skeleton"] > 0)
            return normalized
        return {
            "skeleton": np.zeros((1, 1), dtype=np.uint8),
            "binary": np.zeros((1, 1), dtype=np.uint8),
            "filaments": [],
            "branch_points": np.empty((0, 2), dtype=int),
            "total_length": 0.0,
            "num_filaments": 0,
            "num_branches": 0,
            "avg_thickness": 0.0,
        }

    def on_tracing_finished(self, results):
        """Handle successful tracing completion"""
        self.current_results = results
        self.trace_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.export_skeleton_btn.setEnabled(True)
        self.export_stats_btn.setEnabled(True)
        self.workflow_handle = None

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
        if skeleton_name in [layer_item.name for layer_item in self.viewer.layers]:
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
            if branch_name in [layer_item.name for layer_item in self.viewer.layers]:
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

        self._record_tracing_provenance()
        self.workflow_step = None
        self.last_workflow_result = None

        self.status_label.setText(f"Tracing complete: {results['num_filaments']} filaments found")
        self.status_label.setStyleSheet("color: green; font-style: italic;")
        self.progress_bar.setValue(100)

    def on_tracing_error(self, error_message):
        """Handle tracing errors"""
        self.trace_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.workflow_handle = None
        self.workflow_step = None
        self.last_workflow_result = None
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("color: red; font-style: italic;")
        self.progress_bar.setValue(0)

    def export_skeleton(self):
        """Export skeleton to file"""
        if self.current_results is None:
            return

        import tifffile
        from PyQt6.QtWidgets import QFileDialog

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

        import csv

        from PyQt6.QtWidgets import QFileDialog

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

    def _resolve_project_store_dir(self):
        """Resolve project-store path using shared settings."""
        self.project_store_settings = load_project_store_settings()
        naming = str(self.project_store_settings.get("session_naming", "timestamp"))
        if naming == "timestamp":
            resolved = resolve_project_store_dir(
                self.project_store_settings,
                session_dir_cache=self.project_session_dir_cache,
            )
            if self.project_session_dir_cache is None:
                self.project_session_dir_cache = resolved
            return resolved
        return resolve_project_store_dir(self.project_store_settings)

    def _record_tracing_provenance(self):
        """Persist tracing outputs and workflow metadata to ProjectStore."""
        if not self.workflow_step or not self.last_workflow_result:
            return
        self.project_store_settings = load_project_store_settings()
        if not bool(self.project_store_settings.get("provenance_enabled", True)):
            return
        try:
            current_layer_name = self.layer_combo.currentText()
            source_paths = []
            for layer in self.viewer.layers:
                if layer.name == current_layer_name:
                    source_path = getattr(layer, "metadata", {}).get("source_path")
                    if source_path:
                        source_paths.append(str(source_path))
                    break
            project_dir = self._resolve_project_store_dir()
            record_ui_workflow_result(
                project_dir=project_dir,
                step=self.workflow_step,
                result=self.last_workflow_result,
                source_paths=source_paths,
            )
        except Exception as exc:
            self.status_label.setText(f"Provenance warning: {exc}")
            self.status_label.setStyleSheet("color: #a67c00; font-style: italic;")

    def closeEvent(self, event):
        """Shutdown worker pool when widget closes."""
        if self.workflow_handle and not self.workflow_handle.done():
            self.workflow_handle.cancel()
        self.job_runner.shutdown(wait=False)
        super().closeEvent(event)
