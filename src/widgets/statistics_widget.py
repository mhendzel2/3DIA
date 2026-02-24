"""Statistics and heterogeneity widget for Scientific Image Analyzer."""

from __future__ import annotations

import matplotlib.pyplot as plt
import napari
import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from napari.qt.threading import thread_worker
from napari.layers import Image, Labels
from skimage.measure import regionprops_table

sns = None
try:
    from magicgui import magicgui
except Exception:  # pragma: no cover
    magicgui = None

try:
    import seaborn as sns  # type: ignore

    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

from pymaris.advanced_analysis import (
    compare_distributions_wasserstein,
    identify_subpopulations_gmm,
)


class PandasModel(QAbstractTableModel):
    """A model to interface a pandas DataFrame with QTableView."""

    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role: int = int(Qt.ItemDataRole.DisplayRole)):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._data.columns[section]
        return None


class StatisticsWidget(QWidget):
    """Widget for label statistics and heterogeneity analytics."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.dataframe: pd.DataFrame | None = None
        self.init_ui()

    def init_ui(self):
        root_layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_properties_tab(), "Properties")
        self.tabs.addTab(self._build_heterogeneity_tab(), "Heterogeneity & Subpopulations")
        root_layout.addWidget(self.tabs)

        self.setLayout(root_layout)

        self.viewer.layers.events.inserted.connect(self.update_layer_combos)
        self.viewer.layers.events.removed.connect(self.update_layer_combos)
        self.update_layer_combos()

    def _build_properties_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()

        layer_group = QGroupBox("Select Layers")
        layer_layout = QVBoxLayout()
        layer_layout.addWidget(QLabel("Labels Layer:"))
        self.labels_combo = QComboBox()
        layer_layout.addWidget(self.labels_combo)
        layer_layout.addWidget(QLabel("Intensity Image (optional):"))
        self.intensity_combo = QComboBox()
        layer_layout.addWidget(self.intensity_combo)
        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)

        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout()
        self.calc_btn = QPushButton("Calculate Properties")
        self.calc_btn.clicked.connect(self.calculate_properties)
        analysis_layout.addWidget(self.calc_btn)
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        table_group = QGroupBox("Results")
        table_layout = QVBoxLayout()
        self.table_view = QTableView()
        table_layout.addWidget(self.table_view)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        plotting_group = QGroupBox("Plotting")
        plotting_layout = QVBoxLayout()
        plotting_layout.addWidget(QLabel("Property to Plot:"))
        self.plot_column_combo = QComboBox()
        plotting_layout.addWidget(self.plot_column_combo)
        plotting_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Histogram", "Box Plot"])
        plotting_layout.addWidget(self.plot_type_combo)
        self.plot_btn = QPushButton("Generate Plot")
        self.plot_btn.clicked.connect(self.generate_plot)
        plotting_layout.addWidget(self.plot_btn)
        plotting_group.setLayout(plotting_layout)
        layout.addWidget(plotting_group)

        container.setLayout(layout)
        return container

    def _build_heterogeneity_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()

        hetero_group = QGroupBox("Heterogeneity & Subpopulations")
        hetero_layout = QVBoxLayout()

        if magicgui is None:
            hetero_layout.addWidget(QLabel("magicgui is not available. Install with: pip install pymaris[napari]"))
        else:
            self.wasserstein_gui = magicgui(
                self._run_wasserstein,
                call_button="Run Wasserstein Population Comparison",
                labels_layer_a={"choices": self._label_layer_choices},
                labels_layer_b={"choices": self._label_layer_choices},
                feature_cols={"widget_type": "LineEdit", "value": "area,perimeter"},
            )
            self.gmm_gui = magicgui(
                self._run_gmm,
                call_button="Run GMM Subpopulation Discovery",
                labels_layer={"choices": self._label_layer_choices},
                feature_cols={"widget_type": "LineEdit", "value": "area,perimeter"},
                max_components={"min": 1, "max": 10, "value": 5},
            )
            hetero_layout.addWidget(self.wasserstein_gui.native)
            hetero_layout.addWidget(self.gmm_gui.native)

        self.heterogeneity_status = QLabel("Ready")
        hetero_layout.addWidget(self.heterogeneity_status)
        hetero_group.setLayout(hetero_layout)
        layout.addWidget(hetero_group)

        container.setLayout(layout)
        return container

    def _label_layer_choices(self):
        return [layer.name for layer in self.viewer.layers if isinstance(layer, Labels)]

    def update_layer_combos(self):
        self.labels_combo.clear()
        self.intensity_combo.clear()
        self.intensity_combo.addItem("None")
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                self.labels_combo.addItem(layer.name)
            elif isinstance(layer, Image):
                self.intensity_combo.addItem(layer.name)

    def calculate_properties(self):
        labels_layer_name = self.labels_combo.currentText()
        if not labels_layer_name:
            print("No labels layer selected.")
            return

        labels_layer = self.viewer.layers[labels_layer_name]
        properties = ["label", "area", "perimeter", "mean_intensity"]

        intensity_layer = None
        if self.intensity_combo.currentText() != "None":
            intensity_layer = self.viewer.layers[self.intensity_combo.currentText()]

        props = regionprops_table(
            labels_layer.data,
            intensity_image=intensity_layer.data if intensity_layer else None,
            properties=properties,
        )
        self.dataframe = pd.DataFrame(props)

        model = PandasModel(self.dataframe)
        self.table_view.setModel(model)

        self.plot_column_combo.clear()
        self.plot_column_combo.addItems(self.dataframe.columns)

    def generate_plot(self):
        if self.dataframe is None:
            print("Please calculate properties first.")
            return

        column = self.plot_column_combo.currentText()
        plot_type = self.plot_type_combo.currentText()

        plt.figure()
        if plot_type == "Histogram":
            if HAS_SEABORN and sns is not None:
                sns.histplot(data=self.dataframe, x=column, kde=True)
            else:
                plt.hist(self.dataframe[column].astype(float), bins=30, edgecolor="black", alpha=0.7)
            plt.title(f"Histogram of {column}")
        elif plot_type == "Box Plot":
            if HAS_SEABORN and sns is not None:
                sns.boxplot(y=self.dataframe[column])
            else:
                plt.boxplot(self.dataframe[column].astype(float).dropna(), vert=True)
            plt.title(f"Box Plot of {column}")

        backend = str(plt.get_backend()).lower()
        if "agg" in backend:
            plt.close()
            return
        plt.show()

    def _resolve_features_from_labels(self, labels_layer_name: str) -> pd.DataFrame:
        labels_layer = self.viewer.layers[labels_layer_name]
        features = getattr(labels_layer, "features", None)
        if isinstance(features, pd.DataFrame) and not features.empty:
            return features.copy()

        props = regionprops_table(labels_layer.data, properties=["label", "area", "perimeter"])
        return pd.DataFrame(props)

    def _parse_feature_cols(self, feature_cols: str) -> list[str]:
        return [item.strip() for item in feature_cols.split(",") if item.strip()]

    def _run_wasserstein(self, labels_layer_a: str, labels_layer_b: str, feature_cols: str) -> None:
        selected_cols = self._parse_feature_cols(feature_cols)
        features_a = self._resolve_features_from_labels(labels_layer_a)
        features_b = self._resolve_features_from_labels(labels_layer_b)

        def _compute() -> float:
            return compare_distributions_wasserstein(features_a, features_b, selected_cols)

        worker = thread_worker(_compute)()  # type: ignore[call-arg]
        worker.returned.connect(
            lambda value: self.heterogeneity_status.setText(
                f"Wasserstein (EMD^2) between {labels_layer_a} and {labels_layer_b}: {value:.5f}"
            )
        )
        worker.errored.connect(
            lambda exc: self.heterogeneity_status.setText(
                f"Wasserstein failed: {exc}. Install with: pip install pymaris[advanced]"
            )
        )
        worker.start()

    def _run_gmm(self, labels_layer: str, feature_cols: str, max_components: int = 5) -> None:
        selected_cols = self._parse_feature_cols(feature_cols)
        labels_obj = self.viewer.layers[labels_layer]
        base_features = self._resolve_features_from_labels(labels_layer)

        def _compute() -> pd.DataFrame:
            return identify_subpopulations_gmm(base_features, selected_cols, int(max_components))

        def _on_returned(result_df: pd.DataFrame) -> None:
            setattr(labels_obj, "features", result_df)
            self.heterogeneity_status.setText(
                f"Assigned Subpopulation_ID to {len(result_df)} objects on layer '{labels_layer}'"
            )

        worker = thread_worker(_compute)()  # type: ignore[call-arg]
        worker.returned.connect(_on_returned)
        worker.errored.connect(
            lambda exc: self.heterogeneity_status.setText(
                f"GMM failed: {exc}. Install with: pip install pymaris[advanced]"
            )
        )
        worker.start()
