# widget_manager.py
# GUI for managing widget loading and configuration

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QCheckBox, QLabel, QScrollArea,
                             QComboBox, QMessageBox)
from PyQt6.QtCore import Qt
import json
from pathlib import Path

class WidgetManagerWidget(QWidget):
    """Widget for managing which widgets are loaded"""
    
    WIDGET_INFO = {
        "file_io": {
            "name": "File I/O",
            "description": "Load and save microscopy files (TIFF, CZI, LIF, ND2, Imaris, Zarr)",
            "category": "Essential"
        },
        "processing": {
            "name": "Image Processing",
            "description": "Filters, transforms, and image enhancements",
            "category": "Essential"
        },
        "segmentation": {
            "name": "Segmentation",
            "description": "Object detection and cell segmentation",
            "category": "Essential"
        },
        "analysis": {
            "name": "Analysis & Plotting",
            "description": "Intensity measurements, colocalization, and plotting",
            "category": "Essential"
        },
        "visualization": {
            "name": "3D Visualization",
            "description": "MIP, volume rendering, orthogonal views, clipping",
            "category": "Essential"
        },
        "statistics": {
            "name": "Statistics",
            "description": "Statistical analysis and data export",
            "category": "Essential"
        },
        "filament_tracing": {
            "name": "Filament Tracing",
            "description": "Neuron and cytoskeleton tracing (Imaris FilamentTracer)",
            "category": "Advanced"
        },
        "tracking": {
            "name": "Cell Tracking",
            "description": "Cell tracking with lineage trees (Imaris Track)",
            "category": "Advanced"
        },
        "hca": {
            "name": "High-Content Analysis",
            "description": "Multi-well plate analysis and batch processing",
            "category": "Advanced"
        },
        "deconvolution": {
            "name": "Deconvolution",
            "description": "Richardson-Lucy and Wiener deconvolution",
            "category": "Optional"
        },
        "simple_threshold": {
            "name": "Simple Threshold",
            "description": "Basic threshold widget (magicgui)",
            "category": "Optional"
        },
        "adaptive_threshold": {
            "name": "Adaptive Threshold",
            "description": "Adaptive threshold widget (magicgui)",
            "category": "Optional"
        },
        "ai_segmentation": {
            "name": "AI Segmentation",
            "description": "Deep learning-based segmentation",
            "category": "Optional"
        },
        "biophysics": {
            "name": "Biophysics Analysis",
            "description": "FRAP, FRET, and dose-response analysis",
            "category": "Optional"
        },
        "interactive_plotting": {
            "name": "Interactive Plotting",
            "description": "Advanced interactive plots and charts",
            "category": "Optional"
        }
    }
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.config_file = Path(__file__).parent.parent / "config" / "widget_config.json"
        self.loaded_widgets = {}
        self.checkboxes = {}
        self.area_combos = {}
        self.init_ui()
        self.load_config()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("<h2>Widget Manager</h2>")
        layout.addWidget(header)
        
        # Instructions
        instructions = QLabel(
            "Select which widgets to load at startup. "
            "Changes take effect after restart."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Scroll area for widgets
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Group by category
        categories = {}
        for widget_id, info in self.WIDGET_INFO.items():
            category = info["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append((widget_id, info))
        
        # Create checkbox for each widget
        for category in ["Essential", "Advanced", "Optional"]:
            if category not in categories:
                continue
                
            group = QGroupBox(f"{category} Widgets")
            group_layout = QVBoxLayout()
            
            for widget_id, info in categories[category]:
                widget_layout = QHBoxLayout()
                
                # Enable checkbox
                checkbox = QCheckBox(info["name"])
                checkbox.setToolTip(info["description"])
                self.checkboxes[widget_id] = checkbox
                widget_layout.addWidget(checkbox, stretch=2)
                
                # Area selector
                area_combo = QComboBox()
                area_combo.addItems(["left", "right", "top", "bottom"])
                area_combo.setMaximumWidth(100)
                self.area_combos[widget_id] = area_combo
                widget_layout.addWidget(QLabel("Area:"))
                widget_layout.addWidget(area_combo, stretch=1)
                
                # Load now button
                load_btn = QPushButton("Load Now")
                load_btn.setMaximumWidth(100)
                load_btn.clicked.connect(lambda checked, wid=widget_id: self.load_widget_now(wid))
                widget_layout.addWidget(load_btn)
                
                group_layout.addLayout(widget_layout)
                
            group.setLayout(group_layout)
            scroll_layout.addWidget(group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save Configuration")
        save_btn.clicked.connect(self.save_config)
        button_layout.addWidget(save_btn)
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(reset_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def load_config(self):
        """Load configuration from file"""
        if not self.config_file.exists():
            return
            
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
            enabled = config.get("enabled_widgets", {})
            areas = config.get("widget_areas", {})
            
            for widget_id in self.checkboxes.keys():
                if widget_id in enabled:
                    self.checkboxes[widget_id].setChecked(enabled[widget_id])
                if widget_id in areas:
                    index = self.area_combos[widget_id].findText(areas[widget_id])
                    if index >= 0:
                        self.area_combos[widget_id].setCurrentIndex(index)
                        
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Could not load config: {e}")
            
    def save_config(self):
        """Save configuration to file"""
        try:
            config = {
                "enabled_widgets": {},
                "widget_areas": {},
                "load_on_startup": True,
                "show_welcome_message": True
            }
            
            for widget_id, checkbox in self.checkboxes.items():
                config["enabled_widgets"][widget_id] = checkbox.isChecked()
                config["widget_areas"][widget_id] = self.area_combos[widget_id].currentText()
            
            self.config_file.parent.mkdir(exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
                
            QMessageBox.information(
                self, 
                "Configuration Saved", 
                "Widget configuration saved. Restart the application for changes to take effect."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save config: {e}")
            
    def reset_to_defaults(self):
        """Reset to default configuration"""
        defaults = {
            "file_io": (True, "left"),
            "processing": (True, "left"),
            "segmentation": (True, "left"),
            "analysis": (True, "left"),
            "visualization": (True, "left"),
            "statistics": (True, "right"),
            "filament_tracing": (True, "right"),
            "tracking": (True, "right"),
            "hca": (True, "right"),
            "deconvolution": (False, "left"),
            "simple_threshold": (False, "left"),
            "adaptive_threshold": (False, "left"),
            "ai_segmentation": (False, "right"),
            "biophysics": (False, "right"),
            "interactive_plotting": (False, "right")
        }
        
        for widget_id, (enabled, area) in defaults.items():
            if widget_id in self.checkboxes:
                self.checkboxes[widget_id].setChecked(enabled)
                index = self.area_combos[widget_id].findText(area)
                if index >= 0:
                    self.area_combos[widget_id].setCurrentIndex(index)
                    
        QMessageBox.information(self, "Reset Complete", "Configuration reset to defaults")
        
    def load_widget_now(self, widget_id):
        """Load a widget immediately"""
        if widget_id in self.loaded_widgets:
            QMessageBox.information(
                self,
                "Already Loaded",
                f"{self.WIDGET_INFO[widget_id]['name']} is already loaded"
            )
            return
            
        area = self.area_combos[widget_id].currentText()
        
        try:
            # Import and load the widget
            from src.main_napari_configurable import load_widget
            
            config = {
                "enabled_widgets": {widget_id: True},
                "widget_areas": {widget_id: area}
            }
            
            widget = load_widget(self.viewer, widget_id, config)
            
            if widget is not None:
                self.loaded_widgets[widget_id] = widget
                QMessageBox.information(
                    self,
                    "Widget Loaded",
                    f"{self.WIDGET_INFO[widget_id]['name']} loaded successfully"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Load Failed",
                    f"Could not load {self.WIDGET_INFO[widget_id]['name']}"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Error loading widget: {str(e)}"
            )
