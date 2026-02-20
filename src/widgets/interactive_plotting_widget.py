"""Interactive plotting widget with linked napari selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
except ImportError:  # pragma: no cover
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except ImportError:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import QComboBox, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

import napari
from napari.layers import Image, Labels

from utils.analysis_utils import calculate_object_statistics


class InteractivePlottingWidget(QWidget):
    """Interactive object-level plotting with simple multi-plot support."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.dataframe: pd.DataFrame | None = None
        self.scatter_artist = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout()

        controls_group = QGroupBox("Plot Controls")
        controls_layout = QVBoxLayout(controls_group)

        controls_layout.addWidget(QLabel("Select Labels Layer:"))
        self.labels_layer_combo = QComboBox()
        self.labels_layer_combo.currentTextChanged.connect(self.load_statistics)
        controls_layout.addWidget(self.labels_layer_combo)

        controls_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["scatter", "hexbin", "histogram", "box"])
        controls_layout.addWidget(self.plot_type_combo)

        controls_layout.addWidget(QLabel("X-Axis:"))
        self.x_axis_combo = QComboBox()
        controls_layout.addWidget(self.x_axis_combo)

        controls_layout.addWidget(QLabel("Y-Axis:"))
        self.y_axis_combo = QComboBox()
        controls_layout.addWidget(self.y_axis_combo)

        self.plot_btn = QPushButton("Generate Plot")
        self.plot_btn.clicked.connect(self.update_plot)
        controls_layout.addWidget(self.plot_btn)

        layout.addWidget(controls_group)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("pick_event", self.on_pick)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.viewer.layers.events.connect(self.update_layer_choices)
        self.update_layer_choices()

    def update_layer_choices(self) -> None:
        current = self.labels_layer_combo.currentText()
        self.labels_layer_combo.blockSignals(True)
        self.labels_layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                self.labels_layer_combo.addItem(layer.name)
        self.labels_layer_combo.blockSignals(False)
        if current:
            idx = self.labels_layer_combo.findText(current)
            if idx >= 0:
                self.labels_layer_combo.setCurrentIndex(idx)

    def load_statistics(self, layer_name: str) -> None:
        if not layer_name:
            self.dataframe = None
            self.x_axis_combo.clear()
            self.y_axis_combo.clear()
            return

        labels_layer = self.viewer.layers[layer_name]
        image_layer = self._matching_image_layer(layer_name)
        stats = calculate_object_statistics(
            labels_layer.data,
            intensity_image=image_layer.data if image_layer is not None else None,
        )
        self.dataframe = pd.DataFrame(stats)

        numeric_cols = self.dataframe.select_dtypes(include=np.number).columns.tolist()
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        self.x_axis_combo.addItems(numeric_cols)
        self.y_axis_combo.addItems(numeric_cols)

    def _matching_image_layer(self, layer_name: str):
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and layer.name in layer_name:
                return layer
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                return layer
        return None

    def update_plot(self) -> None:
        if self.dataframe is None or self.x_axis_combo.count() == 0:
            return

        x_prop = self.x_axis_combo.currentText()
        y_prop = self.y_axis_combo.currentText() or x_prop
        plot_type = self.plot_type_combo.currentText()

        self.figure.clear()
        self.scatter_artist = None
        ax = self.figure.add_subplot(111)

        x_values = np.asarray(self.dataframe[x_prop], dtype=float)
        y_values = np.asarray(self.dataframe[y_prop], dtype=float)

        if plot_type == "scatter":
            self.scatter_artist = ax.scatter(x_values, y_values, picker=True, pickradius=5, alpha=0.8)
            ax.set_ylabel(y_prop)
            ax.set_title(f"{y_prop} vs {x_prop}")
        elif plot_type == "hexbin":
            hb = ax.hexbin(x_values, y_values, gridsize=30, cmap="viridis")
            self.figure.colorbar(hb, ax=ax, label="count")
            ax.set_ylabel(y_prop)
            ax.set_title(f"Hexbin: {y_prop} vs {x_prop}")
        elif plot_type == "histogram":
            ax.hist(x_values, bins=30, alpha=0.9, color="#3f7f93")
            ax.set_ylabel("count")
            ax.set_title(f"Histogram: {x_prop}")
        else:  # box
            ax.boxplot([x_values, y_values], labels=[x_prop, y_prop])
            ax.set_ylabel("value")
            ax.set_title("Box Plot")

        ax.set_xlabel(x_prop)
        self.canvas.draw_idle()

    def on_pick(self, event) -> None:
        if self.dataframe is None or self.scatter_artist is None:
            return
        if event.artist is not self.scatter_artist:
            return
        if not event.ind:
            return

        index = int(event.ind[0])
        centroid_cols = [col for col in self.dataframe.columns if "centroid" in col]
        if not centroid_cols:
            return

        point_coords = np.asarray(self.dataframe.loc[index, centroid_cols].values, dtype=float)[None, :]
        points_layer_name = "gated_point"
        if points_layer_name in self.viewer.layers:
            self.viewer.layers[points_layer_name].data = point_coords
        else:
            self.viewer.add_points(
                point_coords,
                name=points_layer_name,
                face_color="cyan",
                size=20,
                edge_width=2,
                edge_color="blue",
            )
