"""
Statistics and Plotting Widget for Scientific Image Analyzer
"""

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QComboBox, QGroupBox, QTableView)
from PyQt6.QtCore import QAbstractTableModel, Qt
import napari
from skimage.measure import regionprops_table
import seaborn as sns
import matplotlib.pyplot as plt

class PandasModel(QAbstractTableModel):
    """A model to interface a pandas DataFrame with QTableView"""
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._data.columns[col]
        return None

class StatisticsWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.init_ui()
        self.dataframe = None

    def init_ui(self):
        layout = QVBoxLayout()

        # Layer Selection
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

        # Analysis
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout()
        self.calc_btn = QPushButton("Calculate Properties")
        self.calc_btn.clicked.connect(self.calculate_properties)
        analysis_layout.addWidget(self.calc_btn)
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # Table View
        table_group = QGroupBox("Results")
        table_layout = QVBoxLayout()
        self.table_view = QTableView()
        table_layout.addWidget(self.table_view)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        # Plotting
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

        self.setLayout(layout)

        self.viewer.layers.events.inserted.connect(self.update_layer_combos)
        self.viewer.layers.events.removed.connect(self.update_layer_combos)
        self.update_layer_combos()

    def update_layer_combos(self):
        self.labels_combo.clear()
        self.intensity_combo.clear()
        self.intensity_combo.addItem("None")
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                self.labels_combo.addItem(layer.name)
            elif isinstance(layer, napari.layers.Image):
                self.intensity_combo.addItem(layer.name)

    def calculate_properties(self):
        labels_layer_name = self.labels_combo.currentText()
        if not labels_layer_name:
            print("No labels layer selected.")
            return

        labels_layer = self.viewer.layers[labels_layer_name]
        properties = ['label', 'area', 'perimeter', 'mean_intensity']

        intensity_layer = None
        if self.intensity_combo.currentText() != "None":
            intensity_layer = self.viewer.layers[self.intensity_combo.currentText()]

        props = regionprops_table(
            labels_layer.data,
            intensity_image=intensity_layer.data if intensity_layer else None,
            properties=properties
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
            sns.histplot(self.dataframe[column], kde=True)
            plt.title(f"Histogram of {column}")
        elif plot_type == "Box Plot":
            sns.boxplot(y=self.dataframe[column])
            plt.title(f"Box Plot of {column}")

        plt.show()
