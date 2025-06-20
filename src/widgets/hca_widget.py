"""
High-Content Analysis (HCA) Widget for Scientific Image Analyzer
Provides multi-well plate analysis workflows for drug screening and automated microscopy
"""

import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSpinBox, QDoubleSpinBox, QComboBox, 
                             QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
                             QTabWidget, QTextEdit, QSplitter, QFileDialog,
                             QProgressBar, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import napari
from napari.layers import Image

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from batch_processor import BatchProcessor, WORKFLOW_TEMPLATES
    BATCH_PROCESSOR_AVAILABLE = True
except ImportError:
    BATCH_PROCESSOR_AVAILABLE = False

from utils.analysis_utils import fit_dose_response

class HCAAnalysisThread(QThread):
    """Thread for HCA analysis processing"""
    progress = pyqtSignal(int)
    finished_analysis = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, plate_data, image_files, workflow_config):
        super().__init__()
        self.plate_data = plate_data
        self.image_files = image_files
        self.workflow_config = workflow_config
        
    def run(self):
        """Perform HCA analysis"""
        try:
            self.progress.emit(10)
            
            if not BATCH_PROCESSOR_AVAILABLE:
                self.error_occurred.emit("Batch processor not available")
                return
                
            batch_processor = BatchProcessor()
            
            self.progress.emit(30)
            
            batch_id = batch_processor.create_batch_session(
                self.image_files, 
                self.workflow_config
            )
            
            self.progress.emit(50)
            
            batch_processor.process_batch(batch_id)
            
            import time
            time.sleep(2)  # Simulate processing time
            
            self.progress.emit(80)
            
            results = batch_processor.get_batch_results(batch_id)
            
            self.progress.emit(100)
            
            combined_results = {
                'batch_id': batch_id,
                'plate_data': self.plate_data.to_dict('records') if hasattr(self.plate_data, 'to_dict') else {},
                'processing_results': results,
                'analysis_complete': True
            }
            
            self.finished_analysis.emit(combined_results)
            
        except Exception as e:
            self.error_occurred.emit(f"HCA analysis failed: {str(e)}")

class PlateVisualizationWidget(QWidget):
    """Widget for visualizing plate-wide results as heatmaps"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(10, 6))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
        else:
            layout.addWidget(QLabel("Matplotlib not available - plate visualization disabled"))
            
    def plot_plate_heatmap(self, plate_data, value_column='mean_intensity'):
        """Plot plate results as heatmap"""
        if not MATPLOTLIB_AVAILABLE or plate_data is None:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        try:
            plate_array = np.zeros((8, 12))
            
            for _, row in plate_data.iterrows():
                if 'row' in row and 'column' in row and value_column in row:
                    r = ord(row['row'].upper()) - ord('A')  # Convert A-H to 0-7
                    c = int(row['column']) - 1  # Convert 1-12 to 0-11
                    if 0 <= r < 8 and 0 <= c < 12:
                        plate_array[r, c] = row[value_column]
            
            im = ax.imshow(plate_array, cmap='viridis', aspect='auto')
            ax.set_title(f'Plate Heatmap - {value_column}')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            
            ax.set_yticks(range(8))
            ax.set_yticklabels([chr(ord('A') + i) for i in range(8)])
            
            ax.set_xticks(range(12))
            ax.set_xticklabels(range(1, 13))
            
            self.figure.colorbar(im, ax=ax)
            self.canvas.draw()
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting heatmap: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
            self.canvas.draw()

class HighContentAnalysisWidget(QWidget):
    """Main HCA widget for multi-well plate analysis"""
    
    def __init__(self, viewer: 'napari.Viewer'):
        super().__init__()
        self.viewer = viewer
        self.plate_data = None
        self.image_files = []
        self.current_results = None
        self.analysis_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        self.create_plate_setup_tab()
        
        self.create_analysis_tab()
        
        self.create_results_tab()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    def create_plate_setup_tab(self):
        """Create plate layout and metadata setup tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        plate_group = QGroupBox("Plate Layout")
        plate_layout = QVBoxLayout(plate_group)
        
        load_layout = QHBoxLayout()
        self.load_plate_btn = QPushButton("Load Plate Layout CSV")
        self.load_plate_btn.clicked.connect(self.load_plate_layout)
        load_layout.addWidget(self.load_plate_btn)
        
        self.plate_info_label = QLabel("No plate layout loaded")
        load_layout.addWidget(self.plate_info_label)
        plate_layout.addLayout(load_layout)
        
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Plate Format:"))
        self.plate_format = QComboBox()
        self.plate_format.addItems(["96-well", "384-well", "24-well"])
        format_layout.addWidget(self.plate_format)
        plate_layout.addLayout(format_layout)
        
        layout.addWidget(plate_group)
        
        files_group = QGroupBox("Image Files")
        files_layout = QVBoxLayout(files_group)
        
        load_files_layout = QHBoxLayout()
        self.load_files_btn = QPushButton("Load Image Files")
        self.load_files_btn.clicked.connect(self.load_image_files)
        load_files_layout.addWidget(self.load_files_btn)
        
        self.files_info_label = QLabel("No image files loaded")
        load_files_layout.addWidget(self.files_info_label)
        files_layout.addLayout(load_files_layout)
        
        layout.addWidget(files_group)
        
        self.tabs.addTab(tab, "Plate Setup")

    def create_analysis_tab(self):
        """Create analysis configuration and execution tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        workflow_group = QGroupBox("Analysis Workflow")
        workflow_layout = QVBoxLayout(workflow_group)
        
        workflow_select_layout = QHBoxLayout()
        workflow_select_layout.addWidget(QLabel("Workflow:"))
        self.workflow_combo = QComboBox()
        if BATCH_PROCESSOR_AVAILABLE:
            for name, config in WORKFLOW_TEMPLATES.items():
                self.workflow_combo.addItem(config.get('name', name), name)
        workflow_select_layout.addWidget(self.workflow_combo)
        workflow_layout.addLayout(workflow_select_layout)
        
        layout.addWidget(workflow_group)
        
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QGridLayout(params_group)
        
        params_layout.addWidget(QLabel("Cell Diameter:"), 0, 0)
        self.cell_diameter = QSpinBox()
        self.cell_diameter.setRange(10, 100)
        self.cell_diameter.setValue(30)
        params_layout.addWidget(self.cell_diameter, 0, 1)
        
        params_layout.addWidget(QLabel("Threshold:"), 1, 0)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.1)
        self.threshold.setValue(0.5)
        params_layout.addWidget(self.threshold, 1, 1)
        
        layout.addWidget(params_group)
        
        self.run_hca_btn = QPushButton("Run High-Content Analysis")
        self.run_hca_btn.clicked.connect(self.run_analysis)
        self.run_hca_btn.setEnabled(False)
        layout.addWidget(self.run_hca_btn)
        
        self.tabs.addTab(tab, "Analysis")

    def create_results_tab(self):
        """Create results visualization and export tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.plate_viz = PlateVisualizationWidget()
        layout.addWidget(self.plate_viz)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_results_btn = QPushButton("Export Results CSV")
        self.export_results_btn.clicked.connect(self.export_results)
        self.export_results_btn.setEnabled(False)
        export_layout.addWidget(self.export_results_btn)
        
        self.export_report_btn = QPushButton("Export HCA Report")
        self.export_report_btn.clicked.connect(self.export_report)
        self.export_report_btn.setEnabled(False)
        export_layout.addWidget(self.export_report_btn)
        
        layout.addLayout(export_layout)
        
        self.tabs.addTab(tab, "Results")

    def load_plate_layout(self):
        """Load a CSV file describing the multi-well plate layout."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Plate Layout", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                self.plate_data = pd.read_csv(file_path)
                self.plate_info_label.setText(f"Loaded: {len(self.plate_data)} wells")
                self.update_analysis_button_state()
                print(f"Loaded plate layout with {len(self.plate_data)} wells.")
            except Exception as e:
                self.plate_info_label.setText(f"Error loading plate: {str(e)}")

    def load_image_files(self):
        """Load image files for HCA analysis."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Image Files", "", 
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg)"
        )
        
        if file_paths:
            self.image_files = file_paths
            self.files_info_label.setText(f"Loaded: {len(self.image_files)} files")
            self.update_analysis_button_state()

    def update_analysis_button_state(self):
        """Enable analysis button when both plate data and images are loaded."""
        can_analyze = (self.plate_data is not None and 
                      len(self.image_files) > 0 and 
                      BATCH_PROCESSOR_AVAILABLE)
        self.run_hca_btn.setEnabled(can_analyze)

    def run_analysis(self):
        """Run the HCA workflow."""
        if self.plate_data is None or not self.image_files:
            return
            
        workflow_name = self.workflow_combo.currentData()
        if not workflow_name or workflow_name not in WORKFLOW_TEMPLATES:
            workflow_name = 'cell_counting'
            
        workflow_config = WORKFLOW_TEMPLATES[workflow_name].copy()
        
        if 'segmentation' in workflow_config:
            workflow_config['segmentation']['diameter'] = self.cell_diameter.value()
        
        self.analysis_thread = HCAAnalysisThread(
            self.plate_data, self.image_files, workflow_config
        )
        self.analysis_thread.progress.connect(self.progress_bar.setValue)
        self.analysis_thread.finished_analysis.connect(self.on_analysis_finished)
        self.analysis_thread.error_occurred.connect(self.on_analysis_error)
        
        self.progress_bar.setVisible(True)
        self.run_hca_btn.setEnabled(False)
        self.analysis_thread.start()

    def on_analysis_finished(self, results):
        """Handle completed HCA analysis."""
        self.current_results = results
        self.progress_bar.setVisible(False)
        self.run_hca_btn.setEnabled(True)
        
        # Enable export buttons
        self.export_results_btn.setEnabled(True)
        self.export_report_btn.setEnabled(True)
        
        if 'plate_data' in results and results['plate_data']:
            plate_df = pd.DataFrame(results['plate_data'])
            self.plate_viz.plot_plate_heatmap(plate_df)
        
        self.tabs.setCurrentIndex(2)
        
        print("HCA analysis completed successfully!")

    def on_analysis_error(self, error_message):
        """Handle analysis errors."""
        self.progress_bar.setVisible(False)
        self.run_hca_btn.setEnabled(True)
        print(f"HCA analysis error: {error_message}")

    def export_results(self):
        """Export HCA results to CSV."""
        if not self.current_results:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export HCA Results", "hca_results.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                if 'plate_data' in self.current_results:
                    plate_df = pd.DataFrame(self.current_results['plate_data'])
                    plate_df.to_csv(file_path, index=False)
                    print(f"HCA results exported to {file_path}")
            except Exception as e:
                print(f"Export failed: {str(e)}")

    def export_report(self):
        """Export comprehensive HCA report."""
        if not self.current_results:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export HCA Report", "hca_report.txt", "Text Files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("High-Content Analysis Report\n")
                    f.write("=" * 40 + "\n\n")
                    
                    if 'batch_id' in self.current_results:
                        f.write(f"Batch ID: {self.current_results['batch_id']}\n")
                    
                    if 'plate_data' in self.current_results:
                        plate_df = pd.DataFrame(self.current_results['plate_data'])
                        f.write(f"Wells analyzed: {len(plate_df)}\n")
                    
                    f.write(f"Image files processed: {len(self.image_files)}\n")
                    f.write(f"Analysis completed: {self.current_results.get('analysis_complete', False)}\n")
                    
                print(f"HCA report exported to {file_path}")
            except Exception as e:
                print(f"Report export failed: {str(e)}")

    def cleanup(self):
        """Clean up resources."""
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.quit()
            self.analysis_thread.wait()
