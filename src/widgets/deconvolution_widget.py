"""
Deconvolution Widget for Scientific Image Analyzer
"""

import napari
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from skimage.restoration import richardson_lucy, wiener

from pymaris.data_model import ImageVolume, infer_axes_from_shape
from pymaris.jobs import JobCancelledError, JobRunner
from pymaris.workflow import WorkflowResult, WorkflowStep
from pymaris_napari.provenance import record_ui_workflow_result
from pymaris_napari.settings import load_project_store_settings, resolve_project_store_dir
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
    workflow_progress = pyqtSignal(int, str)
    workflow_finished = pyqtSignal(object)
    workflow_failed = pyqtSignal(str)
    workflow_cancelled = pyqtSignal(str)

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.thread = None
        self.job_runner = JobRunner(max_workers=2)
        self.workflow_handle = None
        self.workflow_step = None
        self.last_workflow_result = None
        self.current_output_name = "richardson_lucy_deconvolved"
        self.current_input_layer_name = None
        self.project_store_settings = load_project_store_settings()
        self.project_session_dir_cache = None
        self.init_ui()
        self.workflow_progress.connect(self.on_workflow_progress)
        self.workflow_finished.connect(self.on_workflow_finished)
        self.workflow_failed.connect(self.on_workflow_error)
        self.workflow_cancelled.connect(self.on_workflow_cancelled)

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

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_deconvolution)
        self.cancel_btn.setEnabled(False)
        deconv_layout.addWidget(self.cancel_btn)
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
        if self.workflow_handle and not self.workflow_handle.done():
            self.status_label.setText("Deconvolution already running")
            self.status_label.setVisible(True)
            return

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
        output_name = "richardson_lucy_deconvolved" if algorithm == "richardson_lucy" else "wiener_deconvolved"
        params = {
            "operation": "deconvolve",
            "method": algorithm,
            "iterations": int(self.rl_iterations.value()) if algorithm == "richardson_lucy" else 10,
        }
        image_data = np.asarray(layer.data)
        image = ImageVolume(
            array=image_data,
            axes=infer_axes_from_shape(image_data.shape),
            metadata={"name": layer.name},
        )

        self.workflow_step = WorkflowStep(
            id=f"deconvolution-{algorithm}-{layer.name}",
            name=f"restoration:{algorithm}",
            backend_type="restoration",
            backend_name="classic",
            params=params,
            inputs=["image"],
            outputs=[output_name],
        )
        self.last_workflow_result = None
        self.current_output_name = output_name
        self.current_input_layer_name = layer_name
        self.set_buttons_enabled(False)
        self.cancel_btn.setEnabled(True)
        self.update_progress("Running deconvolution...", 0)

        self.workflow_handle = self.job_runner.submit(
            step=self.workflow_step,
            context={"image": image},
            on_progress=lambda p, m: self.workflow_progress.emit(p, m),
        )
        self.workflow_handle.future.add_done_callback(self._on_deconvolution_job_done)

    def cancel_deconvolution(self):
        """Cancel active deconvolution operation."""
        if self.workflow_handle and not self.workflow_handle.done():
            self.workflow_handle.cancel()
        if self.thread and self.thread.isRunning():
            self.thread.terminate()
        self.cancel_btn.setEnabled(False)

    def _on_deconvolution_job_done(self, future):
        """Bridge workflow completion back to Qt thread signals."""
        try:
            result = future.result()
            self.workflow_finished.emit(result)
        except JobCancelledError as exc:
            self.workflow_cancelled.emit(str(exc))
        except Exception as exc:
            self.workflow_failed.emit(str(exc))

    def update_progress(self, message, value):
        self.status_label.setText(message)
        self.status_label.setVisible(True)
        self.progress_bar.setValue(value)
        self.progress_bar.setVisible(True)

    def on_workflow_progress(self, value, message):
        """Update UI from workflow progress callbacks."""
        self.update_progress(message, int(value))

    def on_workflow_finished(self, result):
        """Handle successful workflow completion."""
        self.last_workflow_result = result if isinstance(result, WorkflowResult) else None
        output_name = self.current_output_name
        if isinstance(result, WorkflowResult):
            output_name = self.workflow_step.outputs[0] if self.workflow_step else output_name
            payload = result.outputs.get(output_name)
            if isinstance(payload, ImageVolume):
                self.on_finished(payload.as_numpy(), output_name)
            elif isinstance(payload, np.ndarray):
                self.on_finished(payload, output_name)
            else:
                self.on_error("Workflow finished without image output")
                return
            self._record_deconvolution_provenance()
            self.workflow_step = None
            self.last_workflow_result = None
            return
        self.on_error("Unexpected workflow result type")

    def on_workflow_error(self, message):
        """Handle workflow execution errors."""
        self.workflow_handle = None
        self.workflow_step = None
        self.last_workflow_result = None
        self.cancel_btn.setEnabled(False)
        self.on_error(message)

    def on_workflow_cancelled(self, message):
        """Handle workflow cancellation."""
        self.workflow_handle = None
        self.workflow_step = None
        self.last_workflow_result = None
        self.progress_bar.setVisible(False)
        self.cancel_btn.setEnabled(False)
        self.set_buttons_enabled(True)
        self.status_label.setVisible(True)
        self.status_label.setText(message or "Deconvolution cancelled")

    def on_finished(self, result_image, name):
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.set_buttons_enabled(True)
        self.cancel_btn.setEnabled(False)
        self.workflow_handle = None
        self.viewer.add_image(result_image, name=name)

    def on_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(True)
        self.status_label.setText(f"Error: {error_message}")
        self.cancel_btn.setEnabled(False)
        self.set_buttons_enabled(True)

    def set_buttons_enabled(self, enabled):
        self.run_btn.setEnabled(enabled)

    def _resolve_project_store_dir(self):
        """Resolve project store location using shared settings."""
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

    def _record_deconvolution_provenance(self):
        """Persist restoration outputs and workflow metadata to ProjectStore."""
        if not self.workflow_step or not self.last_workflow_result:
            return
        self.project_store_settings = load_project_store_settings()
        if not bool(self.project_store_settings.get("provenance_enabled", True)):
            return
        try:
            source_paths = []
            if self.current_input_layer_name:
                for layer in self.viewer.layers:
                    if layer.name == self.current_input_layer_name:
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
            self.status_label.setVisible(True)
            self.status_label.setText(f"Provenance warning: {exc}")

    def closeEvent(self, event):
        """Shutdown worker pool when widget closes."""
        if self.workflow_handle and not self.workflow_handle.done():
            self.workflow_handle.cancel()
        self.job_runner.shutdown(wait=False)
        super().closeEvent(event)
