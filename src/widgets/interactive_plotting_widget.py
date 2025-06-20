"""
Interactive Plotting Widget for Scientific Image Analyzer
Supports interactive gating between plots and the napari viewer.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                             QComboBox, QGroupBox)
import napari
from napari.layers import Image, Labels

from utils.analysis_utils import calculate_object_statistics

class InteractivePlottingWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.dataframe = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        controls_group = QGroupBox("Plot Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        controls_layout.addWidget(QLabel("Select Labels Layer:"))
        self.labels_layer_combo = QComboBox()
        self.labels_layer_combo.currentTextChanged.connect(self.load_statistics)
        controls_layout.addWidget(self.labels_layer_combo)

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
        self.canvas.mpl_connect('pick_event', self.on_pick)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        
        self.viewer.layers.events.connect(self.update_layer_choices)
        self.update_layer_choices()

    def update_layer_choices(self):
        self.labels_layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                self.labels_layer_combo.addItem(layer.name)

    def load_statistics(self, layer_name):
        if not layer_name:
            self.dataframe = None
            return
        
        labels_layer = self.viewer.layers[layer_name]
        image_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and layer.name in layer_name:
                image_layer = layer
                break
        
        stats = calculate_object_statistics(
            labels_layer.data,
            intensity_image=image_layer.data if image_layer else None
        )
        
        self.dataframe = pd.DataFrame(stats)
        
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        
        numeric_cols = self.dataframe.select_dtypes(include=np.number).columns.tolist()
        self.x_axis_combo.addItems(numeric_cols)
        self.y_axis_combo.addItems(numeric_cols)

    def update_plot(self):
        if self.dataframe is None or self.x_axis_combo.count() == 0:
            return

        x_prop = self.x_axis_combo.currentText()
        y_prop = self.y_axis_combo.currentText()

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.scatter(self.dataframe[x_prop], self.dataframe[y_prop], picker=True, pickradius=5)
        
        ax.set_xlabel(x_prop)
        ax.set_ylabel(y_prop)
        ax.set_title(f"{y_prop} vs. {x_prop}")
        self.canvas.draw()

    def on_pick(self, event):
        if self.dataframe is None:
            return
            
        ind = event.ind[0]
        
        centroid_cols = [col for col in self.dataframe.columns if 'centroid' in col]
        if not centroid_cols:
            return
        
        point_coords = self.dataframe.loc[ind, centroid_cols].values
        
        points_layer_name = "gated_point"
        if points_layer_name in self.viewer.layers:
            self.viewer.layers[points_layer_name].data = point_coords
        else:
            self.viewer.add_points(
                point_coords,
                name=points_layer_name,
                face_color='cyan',
                size=20,
                edge_width=2,
                edge_color='blue'
            )
