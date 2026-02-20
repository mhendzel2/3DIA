"""Distance tools widget: Euclidean distance map + pairwise distance summaries."""

from __future__ import annotations

import csv
import json
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial.distance import cdist

from pymaris.analysis import calculate_distance_measurements, calculate_label_distance_measurements
from pymaris.jobs import JobHandle, JobRunner
from pymaris.layers import image_volume_from_layer_data
from pymaris.workflow import WorkflowResult, WorkflowStep


class _Signals(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)


class DistanceTransformWidget(QWidget):
    """UI surface for distance map generation and custom pairwise statistics."""

    def __init__(self, viewer: Any) -> None:
        super().__init__()
        self.viewer = viewer
        self.job_runner = JobRunner(max_workers=1)
        self.current_job: JobHandle | None = None
        self.current_stats: dict[str, Any] = {}
        self.signals = _Signals()
        self._step_counter = 0
        self._init_ui()
        self._connect_signals()
        self.refresh_layers()

    def _init_ui(self) -> None:
        root = QVBoxLayout()

        title = QLabel("<b>Distance Tools</b>")
        root.addWidget(title)

        map_group = QGroupBox("Euclidean Distance Map")
        map_form = QFormLayout()

        self.image_combo = QComboBox()
        map_form.addRow("Image Layer", self.image_combo)

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(-1000000, 1000000)
        self.threshold_spin.setValue(0)
        map_form.addRow("Threshold", self.threshold_spin)

        self.distance_to_combo = QComboBox()
        self.distance_to_combo.addItems(["background", "foreground"])
        map_form.addRow("Distance To", self.distance_to_combo)

        self.absolute_check = QCheckBox("Use absolute intensity")
        map_form.addRow("Absolute", self.absolute_check)

        self.output_name_edit = QLineEdit("distance_map")
        map_form.addRow("Output Name", self.output_name_edit)

        map_buttons = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Layers")
        self.run_map_btn = QPushButton("Generate Map")
        map_buttons.addWidget(self.refresh_btn)
        map_buttons.addWidget(self.run_map_btn)
        map_form.addRow(map_buttons)

        map_group.setLayout(map_form)
        root.addWidget(map_group)

        stats_group = QGroupBox("Pairwise Distance Query")
        stats_form = QFormLayout()

        self.source_combo = QComboBox()
        self.target_combo = QComboBox()
        self.target_combo.addItem("(same as source)")

        stats_form.addRow("Source Layer", self.source_combo)
        stats_form.addRow("Target Layer", self.target_combo)

        stats_buttons = QHBoxLayout()
        self.compute_stats_btn = QPushButton("Compute Stats")
        self.export_json_btn = QPushButton("Export JSON")
        self.export_csv_btn = QPushButton("Export CSV")
        stats_buttons.addWidget(self.compute_stats_btn)
        stats_buttons.addWidget(self.export_json_btn)
        stats_buttons.addWidget(self.export_csv_btn)
        stats_form.addRow(stats_buttons)

        stats_group.setLayout(stats_form)
        root.addWidget(stats_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Distance statistics will appear here")
        root.addWidget(self.results_text)

        self.setLayout(root)

    def _connect_signals(self) -> None:
        self.refresh_btn.clicked.connect(self.refresh_layers)
        self.run_map_btn.clicked.connect(self.generate_distance_map)
        self.compute_stats_btn.clicked.connect(self.compute_pairwise_stats)
        self.export_json_btn.clicked.connect(self.export_json)
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.signals.finished.connect(self._on_map_finished)
        self.signals.failed.connect(self._on_map_failed)
        self.viewer.layers.events.connect(self.refresh_layers)

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        if self.current_job and not self.current_job.done():
            self.current_job.cancel()
        self.job_runner.shutdown(wait=False)
        super().closeEvent(event)

    def refresh_layers(self) -> None:
        image_current = self.image_combo.currentText()
        source_current = self.source_combo.currentText()
        target_current = self.target_combo.currentText()

        self.image_combo.clear()
        self.source_combo.clear()
        self.target_combo.clear()
        self.target_combo.addItem("(same as source)")

        for layer in self.viewer.layers:
            if hasattr(layer, "data"):
                self.source_combo.addItem(layer.name)
                self.target_combo.addItem(layer.name)
            if layer.__class__.__name__.lower().endswith("image"):
                self.image_combo.addItem(layer.name)

        self._restore_combo(self.image_combo, image_current)
        self._restore_combo(self.source_combo, source_current)
        self._restore_combo(self.target_combo, target_current)

    def _restore_combo(self, combo: QComboBox, value: str) -> None:
        if not value:
            return
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def generate_distance_map(self) -> None:
        if self.current_job and not self.current_job.done():
            QMessageBox.warning(self, "Busy", "A distance-map job is already running.")
            return

        layer_name = self.image_combo.currentText().strip()
        if not layer_name:
            QMessageBox.warning(self, "Missing Input", "Select an image layer.")
            return

        layer = self.viewer.layers[layer_name]
        metadata = {
            "name": layer.name,
            "scale": list(getattr(layer, "scale", [])),
            "metadata": getattr(layer, "metadata", {}),
        }
        image = image_volume_from_layer_data(data=layer.data, metadata=metadata)

        self._step_counter += 1
        step = WorkflowStep(
            id=f"distance-map-{self._step_counter:04d}",
            name="distance-map",
            backend_type="restoration",
            backend_name="classic",
            params={
                "operation": "distance_map",
                "threshold": float(self.threshold_spin.value()),
                "distance_to": self.distance_to_combo.currentText(),
                "absolute": self.absolute_check.isChecked(),
            },
            inputs=["image"],
            outputs=[self.output_name_edit.text().strip() or "distance_map"],
        )

        self.run_map_btn.setEnabled(False)
        self.current_job = self.job_runner.submit(step=step, context={"image": image})
        self.current_job.future.add_done_callback(self._handle_done)

    def _handle_done(self, future: Any) -> None:
        try:
            result: WorkflowResult = future.result()
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.failed.emit(str(exc))

    def _on_map_finished(self, result: WorkflowResult) -> None:
        self.run_map_btn.setEnabled(True)
        self.current_job = None
        for name, value in result.outputs.items():
            if hasattr(value, "as_numpy"):
                self.viewer.add_image(
                    np.asarray(value.as_numpy()),
                    name=name,
                    scale=list(getattr(value, "scale_for_axes", lambda: [])()),
                    metadata=getattr(value, "metadata_dict", lambda: {})(),
                )
                self.results_text.setPlainText(json.dumps({"status": "ok", "output": name}, indent=2))
                return
        self.results_text.setPlainText(json.dumps({"status": "warning", "message": "No image output"}, indent=2))

    def _on_map_failed(self, message: str) -> None:
        self.run_map_btn.setEnabled(True)
        self.current_job = None
        QMessageBox.critical(self, "Distance Map Error", message)

    def compute_pairwise_stats(self) -> None:
        source_name = self.source_combo.currentText().strip()
        target_name = self.target_combo.currentText().strip()

        if not source_name:
            QMessageBox.warning(self, "Missing Source", "Select a source layer.")
            return

        source_layer = self.viewer.layers[source_name]
        source_points = self._layer_coordinates(source_layer)

        if target_name == "(same as source)" or not target_name:
            stats = calculate_distance_measurements(source_points)
            stats["mode"] = "intra-layer"
        else:
            target_layer = self.viewer.layers[target_name]
            target_points = self._layer_coordinates(target_layer)
            stats = self._cross_layer_stats(source_points, target_points)
            stats["mode"] = "cross-layer"
            stats["source"] = source_name
            stats["target"] = target_name

        self.current_stats = stats
        self.results_text.setPlainText(json.dumps(stats, indent=2, sort_keys=True))

    def _layer_coordinates(self, layer: Any) -> np.ndarray:
        data = np.asarray(layer.data)
        layer_type = layer.__class__.__name__.lower()
        if "points" in layer_type:
            if data.ndim != 2:
                raise ValueError(f"points layer '{layer.name}' must be a 2D coordinate array")
            return np.asarray(data, dtype=float)
        if "labels" in layer_type:
            stats = calculate_label_distance_measurements(data)
            if not bool(stats.get("distance_measurements_available", False)):
                raise ValueError(f"labels layer '{layer.name}' has fewer than two objects")
            return self._label_centroids(data)

        raise ValueError(
            f"layer '{layer.name}' is not supported for pairwise stats (expected points or labels layer)"
        )

    def _label_centroids(self, labels: np.ndarray) -> np.ndarray:
        coords: list[np.ndarray] = []
        for label_id in np.unique(labels):
            if int(label_id) <= 0:
                continue
            indices = np.argwhere(labels == label_id)
            if indices.size == 0:
                continue
            coords.append(np.asarray(indices.mean(axis=0), dtype=float))
        if not coords:
            return np.empty((0, labels.ndim), dtype=float)
        return np.asarray(coords, dtype=float)

    def _cross_layer_stats(self, source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source/target coordinates must be shaped (N, D)")
        if source.shape[1] != target.shape[1]:
            raise ValueError(
                f"source dims ({source.shape[1]}) do not match target dims ({target.shape[1]})"
            )
        if source.shape[0] == 0 or target.shape[0] == 0:
            return {
                "distance_measurements_available": False,
                "source_count": int(source.shape[0]),
                "target_count": int(target.shape[0]),
            }

        matrix = cdist(source, target)
        nearest_source = np.min(matrix, axis=1)
        nearest_target = np.min(matrix, axis=0)
        return {
            "distance_measurements_available": True,
            "source_count": int(source.shape[0]),
            "target_count": int(target.shape[0]),
            "mean_nn_source_to_target": float(np.mean(nearest_source)),
            "max_nn_source_to_target": float(np.max(nearest_source)),
            "mean_nn_target_to_source": float(np.mean(nearest_target)),
            "max_nn_target_to_source": float(np.max(nearest_target)),
            "global_min_distance": float(np.min(matrix)),
            "global_max_distance": float(np.max(matrix)),
        }

    def export_json(self) -> None:
        if not self.current_stats:
            QMessageBox.information(self, "No Data", "Compute stats before exporting.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export Distance Stats", "", "JSON Files (*.json)")
        if not filename:
            return
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(self.current_stats, handle, indent=2, sort_keys=True)

    def export_csv(self) -> None:
        if not self.current_stats:
            QMessageBox.information(self, "No Data", "Compute stats before exporting.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export Distance Stats", "", "CSV Files (*.csv)")
        if not filename:
            return
        with open(filename, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            for key, value in sorted(self.current_stats.items()):
                writer.writerow([key, json.dumps(value) if isinstance(value, (dict, list)) else value])
