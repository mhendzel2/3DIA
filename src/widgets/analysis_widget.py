"""
Analysis Widget for Scientific Image Analyzer
Provides colocalization analysis and statistical measurements
"""

import numpy as np
import matplotlib.pyplot as plt
# Robust backend import: try Qt6, then unified QtAgg, then Qt5
try:
    from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    try:
        # Matplotlib >=3.6 unified Qt backend
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # type: ignore
    except Exception:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # type: ignore
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSpinBox, QDoubleSpinBox, QComboBox, 
                             QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
                             QTabWidget, QTextEdit, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import napari
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker

from scipy.stats import pearsonr, spearmanr
from skimage import measure
from pymaris.measurements import compute_mesh_morphometrics
from utils.analysis_utils import (calculate_colocalization_coefficients, 
                                 costes_threshold, manders_coefficients)

class ColocalizationThread(QThread):
    """Thread for colocalization analysis"""
    progress = pyqtSignal(int)
    finished_analysis = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, image1, image2, parameters):
        super().__init__()
        self.image1 = image1
        self.image2 = image2
        self.parameters = parameters
        
    def run(self):
        """Perform colocalization analysis"""
        try:
            self.progress.emit(10)
            
            # Apply thresholds if specified
            if self.parameters.get('use_threshold', False):
                mask1 = self.image1 > self.parameters['threshold1']
                mask2 = self.image2 > self.parameters['threshold2']
                combined_mask = mask1 & mask2
            else:
                combined_mask = np.ones_like(self.image1, dtype=bool)
                
            self.progress.emit(30)
            
            # Calculate basic statistics
            results = {}
            
            # Pixel intensities for analysis
            pixels1 = self.image1[combined_mask].flatten()
            pixels2 = self.image2[combined_mask].flatten()
            
            self.progress.emit(50)
            
            # Correlation coefficients
            if len(pixels1) > 1:
                pearson_r, pearson_p = pearsonr(pixels1, pixels2)
                spearman_r, spearman_p = spearmanr(pixels1, pixels2)
                
                results.update({
                    'pearson_correlation': pearson_r,
                    'pearson_p_value': pearson_p,
                    'spearman_correlation': spearman_r,
                    'spearman_p_value': spearman_p
                })
            else:
                results.update({
                    'pearson_correlation': 0,
                    'pearson_p_value': 1,
                    'spearman_correlation': 0,
                    'spearman_p_value': 1
                })
                
            self.progress.emit(70)
            
            # Manders coefficients
            if self.parameters.get('calculate_manders', True):
                m1, m2 = manders_coefficients(
                    self.image1, self.image2,
                    self.parameters.get('threshold1', 0),
                    self.parameters.get('threshold2', 0)
                )
                results.update({
                    'manders_m1': m1,
                    'manders_m2': m2
                })
                
            # Overlap coefficients
            if self.parameters.get('calculate_overlap', True):
                overlap_coeff = calculate_colocalization_coefficients(
                    self.image1, self.image2
                )
                results.update(overlap_coeff)
                
            self.progress.emit(90)
            
            # Auto-threshold using Costes method if requested
            if self.parameters.get('use_costes', False):
                try:
                    costes_thresh1, costes_thresh2 = costes_threshold(
                        self.image1, self.image2
                    )
                    results.update({
                        'costes_threshold1': costes_thresh1,
                        'costes_threshold2': costes_thresh2
                    })
                except:
                    results.update({
                        'costes_threshold1': 0,
                        'costes_threshold2': 0
                    })
                    
            # Calculate colocalized volume/area
            if self.parameters.get('use_threshold', False):
                colocalized_voxels = np.sum(combined_mask)
                total_voxels = np.sum(np.logical_or(mask1, mask2))
                if total_voxels > 0:
                    colocalization_percentage = (colocalized_voxels / total_voxels) * 100
                else:
                    colocalization_percentage = 0
                    
                results.update({
                    'colocalized_voxels': colocalized_voxels,
                    'total_voxels': total_voxels,
                    'colocalization_percentage': colocalization_percentage
                })
                
            # Store intensity data for plotting
            results['intensity_data'] = {
                'channel1': pixels1,
                'channel2': pixels2,
                'image1_full': self.image1,
                'image2_full': self.image2,
                'mask': combined_mask
            }
            
            self.progress.emit(100)
            self.finished_analysis.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class PlotWidget(QWidget):
    """Widget for displaying analysis plots"""
    
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def clear_plots(self):
        """Clear all plots"""
        self.figure.clear()
        self.canvas.draw()
        
    def plot_colocalization_analysis(self, results):
        """Plot colocalization analysis results with 2D heatmap and regression line."""
        self.figure.clear()

        intensity_data = results.get('intensity_data', {})
        if not intensity_data:
            self.canvas.draw()
            return

        pixels1 = intensity_data['channel1']
        pixels2 = intensity_data['channel2']

        if len(pixels1) < 2 or len(pixels2) < 2:
            self.canvas.draw()
            return

        # Create subplots
        gs = self.figure.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        ax1 = self.figure.add_subplot(gs[0, :])  # Main plot spans top row
        ax2 = self.figure.add_subplot(gs[1, 0])  # Histogram 1
        ax3 = self.figure.add_subplot(gs[1, 1])  # Histogram 2

        try:
            from matplotlib.colors import LogNorm
            heatmap = ax1.hist2d(pixels1, pixels2, bins=100, norm=LogNorm(), cmap='viridis')
            self.figure.colorbar(heatmap[3], ax=ax1, label='Point Density')
        except ImportError:
            ax1.scatter(pixels1, pixels2, alpha=0.1, s=1)

        if len(pixels1) > 1:
            from scipy import stats
            slope, intercept, r_value, _, _ = stats.linregress(pixels1, pixels2)
            line = slope * np.array(pixels1) + intercept
            ax1.plot(pixels1, line, 'r-', linewidth=1, label=f'y={slope:.2f}x+{intercept:.2f}')

        ax1.set_xlabel('Channel 1 Intensity')
        ax1.set_ylabel('Channel 2 Intensity')
        ax1.set_title('Intensity Correlation Heatmap')
        ax1.legend()
        
        pearson_r = results.get('pearson_correlation', 0)
        ax1.text(0.05, 0.95, f'Pearson r = {pearson_r:.3f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Channel 1 histogram
        ax2.hist(intensity_data['image1_full'].flatten(), bins=100, alpha=0.7, color='red', label='Channel 1')
        ax2.set_title('Ch1 Full Histogram')
        ax2.set_xlabel('Intensity')
        ax2.set_ylabel('Frequency')
        ax2.set_yscale('log')

        # Channel 2 histogram
        ax3.hist(intensity_data['image2_full'].flatten(), bins=100, alpha=0.7, color='green', label='Channel 2')
        ax3.set_title('Ch2 Full Histogram')
        ax3.set_xlabel('Intensity')
        ax3.set_yscale('log')
        
        self.canvas.draw()

class AnalysisWidget(QWidget):
    """Widget for colocalization and statistical analysis"""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.analysis_thread = None
        self.surface_worker = None
        self.current_results = {}
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Create tabs
        self.tab_widget = QTabWidget()
        
        # Colocalization tab
        self.coloc_tab = self.create_colocalization_tab()
        self.tab_widget.addTab(self.coloc_tab, "Colocalization")
        
        # Statistics tab
        self.stats_tab = self.create_statistics_tab()
        self.tab_widget.addTab(self.stats_tab, "Statistics")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
        
    def create_colocalization_tab(self):
        """Create colocalization analysis tab"""
        tab = QWidget()

        # Create splitter for layout
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top part: Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Channel selection
        channels_group = QGroupBox("Select Channels")
        channels_layout = QVBoxLayout()

        # Channel 1
        ch1_layout = QHBoxLayout()
        ch1_layout.addWidget(QLabel("Channel 1:"))
        self.channel1_combo = QComboBox()
        ch1_layout.addWidget(self.channel1_combo)
        channels_layout.addLayout(ch1_layout)

        # Channel 2
        ch2_layout = QHBoxLayout()
        ch2_layout.addWidget(QLabel("Channel 2:"))
        self.channel2_combo = QComboBox()
        ch2_layout.addWidget(self.channel2_combo)
        channels_layout.addLayout(ch2_layout)

        # Populate combos now that they exist and keep them updated with layer changes
        self.update_layer_choices()
        self.viewer.layers.events.inserted.connect(self.update_layer_choices)
        self.viewer.layers.events.removed.connect(self.update_layer_choices)

        channels_group.setLayout(channels_layout)
        controls_layout.addWidget(channels_group)

        # Analysis options
        options_group = QGroupBox("Analysis Options")
        options_layout = QVBoxLayout()

        # Thresholding
        threshold_layout = QHBoxLayout()
        self.use_threshold_check = QCheckBox("Use intensity thresholds")
        self.use_threshold_check.toggled.connect(self.toggle_threshold_controls)
        threshold_layout.addWidget(self.use_threshold_check)
        options_layout.addLayout(threshold_layout)

        # Threshold values
        self.threshold_controls = QWidget()
        thresh_layout = QHBoxLayout(self.threshold_controls)

        thresh_layout.addWidget(QLabel("Ch1 threshold:"))
        self.threshold1_spin = QSpinBox()
        self.threshold1_spin.setRange(0, 65535)
        self.threshold1_spin.setValue(100)
        thresh_layout.addWidget(self.threshold1_spin)

        thresh_layout.addWidget(QLabel("Ch2 threshold:"))
        self.threshold2_spin = QSpinBox()
        self.threshold2_spin.setRange(0, 65535)
        self.threshold2_spin.setValue(100)
        thresh_layout.addWidget(self.threshold2_spin)

        self.threshold_controls.setEnabled(False)
        options_layout.addWidget(self.threshold_controls)

        # Auto-threshold options
        auto_thresh_layout = QHBoxLayout()
        self.auto_threshold_btn = QPushButton("Auto Threshold (Otsu)")
        self.auto_threshold_btn.clicked.connect(self.calculate_auto_thresholds)
        self.auto_threshold_btn.setEnabled(False)
        auto_thresh_layout.addWidget(self.auto_threshold_btn)

        self.use_costes_check = QCheckBox("Use Costes method")
        auto_thresh_layout.addWidget(self.use_costes_check)

        options_layout.addLayout(auto_thresh_layout)

        # Coefficient options
        coeff_layout = QHBoxLayout()
        self.manders_check = QCheckBox("Calculate Manders coefficients")
        self.manders_check.setChecked(True)
        coeff_layout.addWidget(self.manders_check)

        self.overlap_check = QCheckBox("Calculate overlap coefficients")
        self.overlap_check.setChecked(True)
        coeff_layout.addWidget(self.overlap_check)

        options_layout.addLayout(coeff_layout)

        # Advanced mesh curvature mapping
        advanced_layout = QHBoxLayout()
        self.advanced_surface_check = QCheckBox("Calculate Advanced Surface Curvatures")
        advanced_layout.addWidget(self.advanced_surface_check)
        advanced_layout.addWidget(QLabel("Labels:"))
        self.curvature_labels_combo = QComboBox()
        advanced_layout.addWidget(self.curvature_labels_combo)
        options_layout.addLayout(advanced_layout)

        options_group.setLayout(options_layout)
        controls_layout.addWidget(options_group)

        # Analysis button
        self.analyze_btn = QPushButton("Analyze Colocalization")
        self.analyze_btn.clicked.connect(self.analyze_colocalization)
        controls_layout.addWidget(self.analyze_btn)

        # Export buttons
        export_layout = QHBoxLayout()
        self.export_analysis_btn = QPushButton("Export Analysis Results")
        self.export_analysis_btn.clicked.connect(self.export_analysis_results)
        self.export_analysis_btn.setEnabled(False)
        export_layout.addWidget(self.export_analysis_btn)

        self.export_report_btn = QPushButton("Export PDF Report")
        self.export_report_btn.clicked.connect(self.export_analysis_report)
        self.export_report_btn.setEnabled(False)
        export_layout.addWidget(self.export_report_btn)

        controls_layout.addLayout(export_layout)

        splitter.addWidget(controls_widget)

        # Bottom part: Results display
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        # Plot widget
        self.plot_widget = PlotWidget()
        results_layout.addWidget(self.plot_widget)

        splitter.addWidget(results_widget)
        splitter.setSizes([300, 500])  # Allocate more space to plots

        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(splitter)

        return tab
        
    def create_statistics_tab(self):
        """Create statistics display tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Results table
        self.results_table = QTableWidget()
        layout.addWidget(self.results_table)
        
        # Detailed results text
        results_group = QGroupBox("Detailed Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        tab.setLayout(layout)
        return tab
        
    def toggle_threshold_controls(self, enabled):
        """Enable/disable threshold controls"""
        self.threshold_controls.setEnabled(enabled)
        self.auto_threshold_btn.setEnabled(enabled)
        
    def calculate_auto_thresholds(self):
        """Calculate automatic thresholds using Otsu's method"""
        layer1 = self.get_selected_layer(self.channel1_combo)
        layer2 = self.get_selected_layer(self.channel2_combo)
        
        if layer1 is None or layer2 is None:
            self.show_error("Please select valid layers for both channels")
            return
            
        try:
            from skimage import filters
            thresh1 = filters.threshold_otsu(layer1.data)
            thresh2 = filters.threshold_otsu(layer2.data)
            
            self.threshold1_spin.setValue(int(thresh1))
            self.threshold2_spin.setValue(int(thresh2))
            
            print(f"Auto thresholds - Ch1: {thresh1:.1f}, Ch2: {thresh2:.1f}")
            
        except Exception as e:
            self.show_error(f"Auto threshold calculation failed: {str(e)}")
            
    def analyze_colocalization(self):
        """Perform colocalization analysis"""
        layer1 = self.get_selected_layer(self.channel1_combo)
        layer2 = self.get_selected_layer(self.channel2_combo)
        
        if layer1 is None or layer2 is None:
            self.show_error("Please select valid layers for both channels")
            return
            
        if layer1.data.shape != layer2.data.shape:
            self.show_error("Selected images must have the same dimensions")
            return
            
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.show_error("Analysis already in progress")
            return
            
        # Prepare parameters
        parameters = {
            'use_threshold': self.use_threshold_check.isChecked(),
            'threshold1': self.threshold1_spin.value(),
            'threshold2': self.threshold2_spin.value(),
            'use_costes': self.use_costes_check.isChecked(),
            'calculate_manders': self.manders_check.isChecked(),
            'calculate_overlap': self.overlap_check.isChecked()
        }
        
        # Start analysis
        self.analyze_btn.setEnabled(False)
        
        self.analysis_thread = ColocalizationThread(
            layer1.data, layer2.data, parameters
        )
        self.analysis_thread.finished_analysis.connect(self.on_analysis_finished)
        self.analysis_thread.error_occurred.connect(self.on_analysis_error)
        self.analysis_thread.start()
        
    def on_analysis_finished(self, results):
        """Handle analysis completion"""
        try:
            self.current_results = results
            
            # Update plots
            self.plot_widget.plot_colocalization_analysis(results)
            
            # Update results table and text
            self.update_results_display(results)
            
            # Enable export
            self.export_analysis_btn.setEnabled(True)

            if self.advanced_surface_check.isChecked():
                self.calculate_advanced_surface_curvatures()
            
            print("Colocalization analysis completed")
            
        except Exception as e:
            self.show_error(f"Failed to display results: {str(e)}")
        finally:
            self.analyze_btn.setEnabled(True)
            # Enable export buttons
            self.export_analysis_btn.setEnabled(True)
            self.export_report_btn.setEnabled(True)
            
    def on_analysis_error(self, error_message):
        """Handle analysis errors"""
        self.show_error(error_message)
        self.analyze_btn.setEnabled(True)
        
    def update_results_display(self, results):
        """Update results table and text display"""
        # Update results table
        table_data = []
        
        for key, value in results.items():
            if key != 'intensity_data' and isinstance(value, (int, float)):
                table_data.append((key.replace('_', ' ').title(), f"{value:.4f}"))
                
        self.results_table.setRowCount(len(table_data))
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(['Measure', 'Value'])
        
        for i, (measure, value) in enumerate(table_data):
            self.results_table.setItem(i, 0, QTableWidgetItem(measure))
            self.results_table.setItem(i, 1, QTableWidgetItem(value))
            
        self.results_table.resizeColumnsToContents()
        
        # Update detailed text
        text_output = []
        text_output.append("COLOCALIZATION ANALYSIS RESULTS")
        text_output.append("=" * 40)
        text_output.append("")
        
        # Correlation coefficients
        text_output.append("Correlation Coefficients:")
        text_output.append(f"  Pearson correlation: {results.get('pearson_correlation', 0):.4f}")
        text_output.append(f"  Spearman correlation: {results.get('spearman_correlation', 0):.4f}")
        text_output.append("")
        
        # Manders coefficients
        if 'manders_m1' in results:
            text_output.append("Manders Coefficients:")
            text_output.append(f"  M1 (Ch1 overlap with Ch2): {results['manders_m1']:.4f}")
            text_output.append(f"  M2 (Ch2 overlap with Ch1): {results['manders_m2']:.4f}")
            text_output.append("")
            
        # Overlap coefficients
        overlap_keys = ['overlap_k1', 'overlap_k2']
        overlap_values = [results.get(key) for key in overlap_keys if key in results]
        if overlap_values:
            text_output.append("Overlap Coefficients:")
            for key in overlap_keys:
                if key in results:
                    text_output.append(f"  {key}: {results[key]:.4f}")
            text_output.append("")
            
        # Thresholds
        if 'costes_threshold1' in results:
            text_output.append("Costes Auto-thresholds:")
            text_output.append(f"  Channel 1: {results['costes_threshold1']:.1f}")
            text_output.append(f"  Channel 2: {results['costes_threshold2']:.1f}")
            text_output.append("")
            
        # Colocalization percentage
        if 'colocalization_percentage' in results:
            text_output.append(f"Colocalization percentage: {results['colocalization_percentage']:.2f}%")
            text_output.append(f"Colocalized voxels: {results.get('colocalized_voxels', 0)}")
            text_output.append(f"Total voxels: {results.get('total_voxels', 0)}")
            
        self.results_text.setPlainText('\n'.join(text_output))
        
    def export_analysis_results(self):
        """Export analysis results to file"""
        if not self.current_results:
            self.show_error("No results to export")
            return
            
        from PyQt6.QtWidgets import QFileDialog
        import json
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis Results",
            "colocalization_analysis.json",
            "JSON Files (*.json);;CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                # Prepare data for export (exclude intensity data)
                export_data = {k: v for k, v in self.current_results.items() 
                              if k != 'intensity_data'}
                              
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(export_data, f, indent=2)
                elif file_path.endswith('.csv'):
                    import csv
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Measure', 'Value'])
                        for key, value in export_data.items():
                            if isinstance(value, (int, float)):
                                writer.writerow([key, value])
                                
                print(f"Analysis results exported to: {file_path}")
                
            except Exception as e:
                self.show_error(f"Export failed: {str(e)}")
                
    def export_analysis_report(self):
        """Exports a PDF report of the current colocalization analysis."""
        if not self.current_results:
            self.show_error("No analysis results to export.")
            return

        from PyQt6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis Report",
            "colocalization_report.pdf",
            "PDF Files (*.pdf)"
        )

        if not file_path:
            return

        try:
            from matplotlib.figure import Figure
            import matplotlib.pyplot as plt
            report_fig = Figure(figsize=(8.5, 11))
            report_fig.suptitle('Colocalization Analysis Report', fontsize=16)

            gs = report_fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.5)

            ax1 = report_fig.add_subplot(gs[0, :])
            intensity_data = self.current_results['intensity_data']
            try:
                from matplotlib.colors import LogNorm
                ax1.hist2d(intensity_data['channel1'], intensity_data['channel2'], bins=100, norm=LogNorm(), cmap='viridis')
            except:
                ax1.scatter(intensity_data['channel1'], intensity_data['channel2'], alpha=0.1, s=1)
            ax1.set_title('Intensity Correlation Heatmap')
            ax1.set_xlabel('Channel 1 Intensity')
            ax1.set_ylabel('Channel 2 Intensity')

            ax2 = report_fig.add_subplot(gs[1, 0])
            ax2.hist(intensity_data['image1_full'].flatten(), bins=100, color='red', alpha=0.7)
            ax2.set_title('Channel 1 Histogram')
            ax2.set_yscale('log')

            ax3 = report_fig.add_subplot(gs[1, 1])
            ax3.hist(intensity_data['image2_full'].flatten(), bins=100, color='green', alpha=0.7)
            ax3.set_title('Channel 2 Histogram')
            ax3.set_yscale('log')

            # Add statistics text
            ax4 = report_fig.add_subplot(gs[2, :])
            ax4.axis('off')
            stats_text = []
            for key, value in self.current_results.items():
                if key != 'intensity_data':
                    stats_text.append(f"{key.replace('_', ' ').title()}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
            ax4.text(0.05, 0.95, '\n'.join(stats_text), transform=ax4.transAxes, va='top', fontfamily='monospace')

            report_fig.savefig(file_path, format='pdf')
            print(f"Report saved to {file_path}")

        except Exception as e:
            self.show_error(f"Failed to export PDF report: {e}")
                
    def get_selected_layer(self, combo_box):
        """Get the selected layer from combo box"""
        layer_name = combo_box.currentText()
        if layer_name == "No layers available":
            return None
            
        for layer in self.viewer.layers:
            if layer.name == layer_name and isinstance(layer, Image):
                return layer
        return None
        
    def update_layer_choices(self):
        """Update layer selection combo boxes"""
        for combo in [self.channel1_combo, self.channel2_combo]:
            combo.clear()
            
            image_layers = [layer for layer in self.viewer.layers if isinstance(layer, Image)]
            
            if image_layers:
                for layer in image_layers:
                    combo.addItem(layer.name)
            else:
                combo.addItem("No layers available")

        if hasattr(self, "curvature_labels_combo"):
            self.curvature_labels_combo.clear()
            label_layers = [layer for layer in self.viewer.layers if isinstance(layer, Labels)]
            if label_layers:
                for layer in label_layers:
                    self.curvature_labels_combo.addItem(layer.name)
            else:
                self.curvature_labels_combo.addItem("No labels available")

    def calculate_advanced_surface_curvatures(self):
        """Compute label-surface curvatures and map Gaussian values to napari Surface layers."""
        label_layer_name = self.curvature_labels_combo.currentText()
        if not label_layer_name or label_layer_name == "No labels available":
            self.show_error("Select a valid Labels layer for surface curvature analysis")
            return

        labels_layer = None
        for layer in self.viewer.layers:
            if layer.name == label_layer_name and isinstance(layer, Labels):
                labels_layer = layer
                break

        if labels_layer is None:
            self.show_error("Selected labels layer is unavailable")
            return

        spacing = tuple(float(v) for v in np.asarray(getattr(labels_layer, "scale", (1.0, 1.0, 1.0)))[-3:])
        labels_data = np.asarray(labels_layer.data)

        @thread_worker
        def _worker():
            return compute_mesh_morphometrics(
                labels_data,
                voxel_spacing=spacing,
                return_vertex_data=True,
            )

        self.surface_worker = _worker()
        self.surface_worker.returned.connect(
            lambda frame: self._add_curvature_surfaces_to_viewer(frame, label_layer_name)
        )
        self.surface_worker.errored.connect(
            lambda exc: self.show_error(
                f"Advanced surface curvature failed: {exc}. Install with: pip install pymaris[advanced]"
            )
        )
        self.surface_worker.start()

    def _add_curvature_surfaces_to_viewer(self, dataframe, source_layer_name: str):
        """Add per-label napari Surface layers colored by Gaussian curvature."""
        if dataframe is None or dataframe.empty:
            return

        for _, row in dataframe.iterrows():
            label_id = int(row.get("label", 0))
            vertices = row.get("mesh_vertices")
            faces = row.get("mesh_faces")
            values = row.get("gaussian_curvature_vertices")
            if vertices is None or faces is None or values is None:
                continue

            layer_name = f"{source_layer_name}_Label{label_id}_GaussianCurvature"
            surface_data = (np.asarray(vertices), np.asarray(faces), np.asarray(values, dtype=float))
            existing = [layer for layer in self.viewer.layers if layer.name == layer_name]
            if existing:
                existing[0].data = surface_data
                existing[0].colormap = "coolwarm"
            else:
                self.viewer.add_surface(surface_data, name=layer_name, colormap="coolwarm")
                
    def show_error(self, message):
        """Display error message"""
        print(f"ERROR: {message}")
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Analysis Error")
        msg.setText(message)
        msg.exec()
        
    def cleanup(self):
        """Cleanup resources"""
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.quit()
            self.analysis_thread.wait()
