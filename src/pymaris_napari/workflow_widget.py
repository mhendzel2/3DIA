"""Thin napari widget that executes backend workflow steps through core APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pymaris.backends import DEFAULT_REGISTRY
from pymaris.data_model import ImageVolume
from pymaris.jobs import JobCancelledError, JobHandle, JobRunner
from pymaris.layers import image_volume_from_layer_data
from pymaris.workflow import WorkflowResult, WorkflowStep
from pymaris_napari.provenance import record_ui_workflow_result
from pymaris_napari.settings import (
    load_project_store_settings,
    resolve_project_store_dir,
    save_project_store_settings,
)


class _JobSignals(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal(str)


class WorkflowRunnerWidget(QWidget):
    """Thin controller widget for backend-driven workflow execution."""

    def __init__(self, viewer: Any) -> None:
        super().__init__()
        self.viewer = viewer
        self.job_runner = JobRunner(max_workers=2)
        self.current_job: JobHandle | None = None
        self.current_step: WorkflowStep | None = None
        self.current_source_paths: list[str] = []
        self.project_store_settings = load_project_store_settings()
        self._session_project_dir_cache: Path | None = None
        self._step_counter = 0
        self.signals = _JobSignals()
        self._init_ui()
        self._connect_signals()
        self.refresh_layers()

    def _init_ui(self) -> None:
        root = QVBoxLayout()

        title = QLabel("<b>Workflow Runner (Core Backends)</b>")
        root.addWidget(title)

        form_group = QGroupBox("Step Configuration")
        form = QFormLayout()

        self.layer_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh Layers")
        refresh_row = QWidget()
        refresh_layout = QHBoxLayout(refresh_row)
        refresh_layout.setContentsMargins(0, 0, 0, 0)
        refresh_layout.addWidget(self.layer_combo)
        refresh_layout.addWidget(self.refresh_btn)
        form.addRow("Input Layer", refresh_row)

        self.step_type_combo = QComboBox()
        self.step_type_combo.addItems(["segmentation", "restoration", "tracing", "tracking"])
        form.addRow("Step Type", self.step_type_combo)

        self.backend_combo = QComboBox()
        form.addRow("Backend", self.backend_combo)

        self.output_name_edit = QLineEdit("result")
        form.addRow("Output Name", self.output_name_edit)

        self.params_text = QTextEdit("{}")
        self.params_text.setPlaceholderText('{"operation": "denoise", "sigma": 1.0}')
        self.params_text.setFixedHeight(90)
        form.addRow("Params (JSON)", self.params_text)

        form_group.setLayout(form)
        root.addWidget(form_group)

        provenance_group = QGroupBox("Project Store / Provenance")
        provenance_form = QFormLayout()
        self.enable_provenance_check = QCheckBox("Enable provenance recording")
        provenance_form.addRow("Record Runs", self.enable_provenance_check)
        self.project_base_edit = QLineEdit()
        provenance_form.addRow("Base Project Dir", self.project_base_edit)
        self.session_naming_combo = QComboBox()
        self.session_naming_combo.addItems(["none", "fixed", "timestamp"])
        provenance_form.addRow("Session Naming", self.session_naming_combo)
        self.session_value_label = QLabel("Session Name")
        self.session_value_edit = QLineEdit()
        provenance_form.addRow(self.session_value_label, self.session_value_edit)
        provenance_group.setLayout(provenance_form)
        root.addWidget(provenance_group)

        controls = QHBoxLayout()
        self.run_btn = QPushButton("Run Step")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        controls.addWidget(self.run_btn)
        controls.addWidget(self.cancel_btn)
        root.addLayout(controls)

        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        root.addWidget(self.status_label)
        root.addWidget(self.progress_bar)

        self.setLayout(root)

    def _connect_signals(self) -> None:
        self.refresh_btn.clicked.connect(self.refresh_layers)
        self.step_type_combo.currentTextChanged.connect(self._refresh_backend_choices)
        self.run_btn.clicked.connect(self.run_step)
        self.cancel_btn.clicked.connect(self.cancel_current_job)
        self.signals.progress.connect(self._on_progress)
        self.signals.finished.connect(self._on_finished)
        self.signals.failed.connect(self._on_failed)
        self.signals.cancelled.connect(self._on_cancelled)
        self._load_project_store_settings_into_ui()
        self.enable_provenance_check.toggled.connect(self._persist_project_store_settings)
        self.project_base_edit.editingFinished.connect(self._persist_project_store_settings)
        self.session_value_edit.editingFinished.connect(self._persist_project_store_settings)
        self.session_naming_combo.currentTextChanged.connect(self._on_session_naming_changed)

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        if self.current_job and not self.current_job.done():
            self.current_job.cancel()
        self.job_runner.shutdown(wait=False)
        super().closeEvent(event)

    def refresh_layers(self) -> None:
        current = self.layer_combo.currentText()
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            self.layer_combo.addItem(layer.name)
        if current:
            index = self.layer_combo.findText(current)
            if index >= 0:
                self.layer_combo.setCurrentIndex(index)
        self._refresh_backend_choices()

    def _refresh_backend_choices(self) -> None:
        step_type = self.step_type_combo.currentText()
        self.backend_combo.clear()
        if step_type == "segmentation":
            names = sorted(DEFAULT_REGISTRY.segmentation.keys())
        elif step_type == "restoration":
            names = sorted(DEFAULT_REGISTRY.restoration.keys())
        elif step_type == "tracing":
            names = sorted(DEFAULT_REGISTRY.tracing.keys())
        else:
            names = sorted(DEFAULT_REGISTRY.tracking.keys())
        self.backend_combo.addItems(names)

    def run_step(self) -> None:
        if self.current_job and not self.current_job.done():
            QMessageBox.warning(self, "Job Running", "Wait for the current job or cancel it first.")
            return

        layer_name = self.layer_combo.currentText()
        if not layer_name:
            QMessageBox.warning(self, "Missing Input", "Select an input layer.")
            return

        try:
            params = self._parse_params()
            context, inputs = self._build_context(layer_name)
            step = self._build_step(params=params, inputs=inputs)
            self.current_source_paths = self._get_source_paths(layer_name)
        except Exception as exc:
            QMessageBox.critical(self, "Configuration Error", str(exc))
            return

        self.current_step = step
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Running {step.name}...")

        self.current_job = self.job_runner.submit(
            step=step,
            context=context,
            on_progress=lambda percent, message: self.signals.progress.emit(percent, message),
        )
        self.current_job.future.add_done_callback(self._handle_done)

    def _parse_params(self) -> dict[str, Any]:
        text = self.params_text.toPlainText().strip()
        if not text:
            return {}
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Params JSON must decode to an object")
        return parsed

    def _build_context(self, layer_name: str) -> tuple[dict[str, Any], list[str]]:
        layer = self.viewer.layers[layer_name]
        step_type = self.step_type_combo.currentText()
        if step_type == "tracking":
            if not hasattr(layer, "data"):
                raise ValueError("Tracking requires a labels layer with data")
            data = np.asarray(layer.data)
            if data.ndim < 3:
                raise ValueError("Tracking input must include time dimension (T, ...)")
            labels_over_time = [np.asarray(data[t]) for t in range(data.shape[0])]
            return {"labels_sequence": labels_over_time}, ["labels_sequence"]

        layer_meta = {
            "name": layer.name,
            "scale": list(getattr(layer, "scale", [])),
            "metadata": getattr(layer, "metadata", {}),
        }
        image = image_volume_from_layer_data(data=layer.data, metadata=layer_meta)
        return {"image": image}, ["image"]

    def _get_source_paths(self, layer_name: str) -> list[str]:
        layer = self.viewer.layers[layer_name]
        source_path = getattr(layer, "metadata", {}).get("source_path")
        if source_path:
            return [str(source_path)]
        return []

    def _build_step(self, params: dict[str, Any], inputs: list[str]) -> WorkflowStep:
        self._step_counter += 1
        step_type = self.step_type_combo.currentText()
        backend_name = self.backend_combo.currentText()
        output_name = self.output_name_edit.text().strip() or f"{step_type}_result"
        return WorkflowStep(
            id=f"ui-step-{self._step_counter:04d}",
            name=f"{step_type}:{backend_name}",
            backend_type=step_type,
            backend_name=backend_name,
            params=params,
            inputs=inputs,
            outputs=[output_name],
        )

    def _load_project_store_settings_into_ui(self) -> None:
        settings = dict(self.project_store_settings)
        self.enable_provenance_check.setChecked(bool(settings.get("provenance_enabled", True)))
        self.project_base_edit.setText(str(settings.get("base_project_dir", ".pymaris_project")))
        naming = str(settings.get("session_naming", "timestamp"))
        index = self.session_naming_combo.findText(naming)
        if index >= 0:
            self.session_naming_combo.setCurrentIndex(index)
        value = settings.get("session_prefix", "session")
        if naming == "fixed":
            value = settings.get("session_name", "default")
        self.session_value_edit.setText(str(value))
        self._update_session_value_label(naming)

    def _on_session_naming_changed(self, naming: str) -> None:
        self._update_session_value_label(naming)
        self._session_project_dir_cache = None
        self._persist_project_store_settings()

    def _update_session_value_label(self, naming: str) -> None:
        if naming == "timestamp":
            self.session_value_label.setText("Session Prefix")
            self.session_value_edit.setPlaceholderText("session")
        elif naming == "fixed":
            self.session_value_label.setText("Session Name")
            self.session_value_edit.setPlaceholderText("default")
        else:
            self.session_value_label.setText("Session Value")
            self.session_value_edit.setPlaceholderText("(unused)")

    def _persist_project_store_settings(self) -> None:
        previous = dict(self.project_store_settings)
        naming = self.session_naming_combo.currentText().strip() or "timestamp"
        value = self.session_value_edit.text().strip()
        updated = {
            "base_project_dir": self.project_base_edit.text().strip() or ".pymaris_project",
            "session_naming": naming,
            "provenance_enabled": self.enable_provenance_check.isChecked(),
        }
        if naming == "timestamp":
            updated["session_prefix"] = value or "session"
        elif naming == "fixed":
            updated["session_name"] = value or "default"
        self.project_store_settings = save_project_store_settings(updated)
        if (
            previous.get("base_project_dir") != self.project_store_settings.get("base_project_dir")
            or previous.get("session_naming") != self.project_store_settings.get("session_naming")
            or previous.get("session_name") != self.project_store_settings.get("session_name")
            or previous.get("session_prefix") != self.project_store_settings.get("session_prefix")
        ):
            self._session_project_dir_cache = None

    def _resolve_project_store_dir(self) -> Path:
        settings = dict(self.project_store_settings)
        naming = str(settings.get("session_naming", "timestamp"))
        if naming == "timestamp":
            resolved = resolve_project_store_dir(
                settings,
                session_dir_cache=self._session_project_dir_cache,
            )
            if self._session_project_dir_cache is None:
                self._session_project_dir_cache = resolved
            return resolved
        return resolve_project_store_dir(settings)

    def cancel_current_job(self) -> None:
        if not self.current_job:
            return
        self.current_job.cancel()
        self.status_label.setText("Cancelling...")
        self.cancel_btn.setEnabled(False)

    def _handle_done(self, future: Any) -> None:
        try:
            result: WorkflowResult = future.result()
            self.signals.finished.emit(result)
        except JobCancelledError as exc:
            self.signals.cancelled.emit(str(exc))
        except Exception as exc:  # pragma: no cover - UI pathway
            self.signals.failed.emit(str(exc))

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(max(0, min(100, int(percent))))
        self.status_label.setText(message)

    def _on_finished(self, result: WorkflowResult) -> None:
        step = self.current_step
        if step is None:
            return

        for output_name, value in result.outputs.items():
            if isinstance(value, ImageVolume):
                self.viewer.add_image(
                    value.as_numpy(),
                    name=output_name,
                    scale=value.scale_for_axes(),
                    metadata=value.metadata_dict(),
                )
            elif isinstance(value, np.ndarray):
                if step.backend_type == "segmentation":
                    self.viewer.add_labels(value, name=output_name)
                else:
                    self.viewer.add_image(value, name=output_name)
            elif isinstance(value, dict) and "napari_tracks" in value:
                self.viewer.add_tracks(value["napari_tracks"], name=output_name)
            elif isinstance(value, dict) and "skeleton" in value:
                self.viewer.add_labels(np.asarray(value["skeleton"]), name=f"{output_name}_skeleton")

        self.progress_bar.setValue(100)
        self.status_label.setText("Completed")

        if self.enable_provenance_check.isChecked():
            try:
                self._persist_project_store_settings()
                project_dir = self._resolve_project_store_dir()
                record_ui_workflow_result(
                    project_dir=project_dir,
                    step=step,
                    result=result,
                    source_paths=self.current_source_paths,
                )
            except Exception as exc:
                QMessageBox.warning(self, "Provenance Warning", f"Could not record provenance: {exc}")

        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.current_job = None
        self.current_step = None
        self.current_source_paths = []

    def _on_failed(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.current_job = None
        self.current_step = None
        self.current_source_paths = []
        QMessageBox.critical(self, "Workflow Error", message)

    def _on_cancelled(self, message: str) -> None:
        self.status_label.setText("Cancelled")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.current_job = None
        self.current_step = None
        self.current_source_paths = []
        QMessageBox.information(self, "Workflow Cancelled", message)
