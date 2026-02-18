# widget_manager.py
# GUI for managing widget loading and configuration

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pymaris_napari.configurable import DEFAULT_CONFIG
from pymaris_napari.settings import (
    DEFAULT_PROJECT_STORE_SETTINGS,
    load_project_store_settings,
    load_widget_config,
    save_widget_config,
)


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
        },
        "distance_tools": {
            "name": "Distance Tools",
            "description": "Euclidean distance maps and pairwise distance queries",
            "category": "Optional",
        },
        "workflow_runner": {
            "name": "Workflow Runner (Core Backends)",
            "description": "Backend-driven workflows with progress and cancellation",
            "category": "Essential"
        }
    }

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.loaded_widgets = {}
        self.checkboxes = {}
        self.area_combos = {}
        self.session_value_label = None
        self.session_value_edit = None
        self.session_naming_combo = None
        self.project_base_edit = None
        self.provenance_enabled_check = None
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

        settings_group = QGroupBox("Project Store Settings")
        settings_form = QFormLayout()

        self.provenance_enabled_check = QCheckBox("Record workflow provenance")
        settings_form.addRow("Provenance", self.provenance_enabled_check)

        self.project_base_edit = QLineEdit()
        settings_form.addRow("Base Project Dir", self.project_base_edit)

        self.session_naming_combo = QComboBox()
        self.session_naming_combo.addItems(["none", "fixed", "timestamp"])
        settings_form.addRow("Session Naming", self.session_naming_combo)

        self.session_value_label = QLabel("Session Prefix")
        self.session_value_edit = QLineEdit()
        settings_form.addRow(self.session_value_label, self.session_value_edit)

        self.session_naming_combo.currentTextChanged.connect(self._on_session_naming_changed)
        settings_group.setLayout(settings_form)
        layout.addWidget(settings_group)

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
        try:
            config = load_widget_config()
            enabled = dict(DEFAULT_CONFIG.get("enabled_widgets", {}))
            enabled.update(config.get("enabled_widgets", {}))
            areas = dict(DEFAULT_CONFIG.get("widget_areas", {}))
            areas.update(config.get("widget_areas", {}))

            for widget_id in self.checkboxes.keys():
                if widget_id in enabled:
                    self.checkboxes[widget_id].setChecked(bool(enabled[widget_id]))
                if widget_id in areas:
                    index = self.area_combos[widget_id].findText(str(areas[widget_id]))
                    if index >= 0:
                        self.area_combos[widget_id].setCurrentIndex(index)

            self._apply_project_store_settings(load_project_store_settings(config))
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Could not load config: {e}")

    def save_config(self):
        """Save configuration to file"""
        try:
            config = load_widget_config()
            config["enabled_widgets"] = {}
            config["widget_areas"] = {}
            config["load_on_startup"] = bool(config.get("load_on_startup", True))
            config["show_welcome_message"] = bool(config.get("show_welcome_message", True))

            for widget_id, checkbox in self.checkboxes.items():
                config["enabled_widgets"][widget_id] = checkbox.isChecked()
                config["widget_areas"][widget_id] = self.area_combos[widget_id].currentText()
            config["project_store"] = self._collect_project_store_settings()
            save_widget_config(config)

            QMessageBox.information(
                self,
                "Configuration Saved",
                "Widget configuration saved. Restart the application for changes to take effect."
            )

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save config: {e}")

    def reset_to_defaults(self):
        """Reset to default configuration"""
        default_enabled = DEFAULT_CONFIG.get("enabled_widgets", {})
        default_areas = DEFAULT_CONFIG.get("widget_areas", {})
        for widget_id, enabled in default_enabled.items():
            area = default_areas.get(widget_id, "left")
            if widget_id in self.checkboxes:
                self.checkboxes[widget_id].setChecked(enabled)
                index = self.area_combos[widget_id].findText(area)
                if index >= 0:
                    self.area_combos[widget_id].setCurrentIndex(index)

        self._apply_project_store_settings(DEFAULT_PROJECT_STORE_SETTINGS)

        QMessageBox.information(self, "Reset Complete", "Configuration reset to defaults")

    def _collect_project_store_settings(self):
        """Collect project-store settings from UI controls."""
        naming = self.session_naming_combo.currentText().strip() or "timestamp"
        value = self.session_value_edit.text().strip()
        settings = {
            "base_project_dir": self.project_base_edit.text().strip() or ".pymaris_project",
            "session_naming": naming,
            "provenance_enabled": self.provenance_enabled_check.isChecked(),
            "session_name": "default",
            "session_prefix": "session",
        }
        if naming == "fixed":
            settings["session_name"] = value or "default"
        elif naming == "timestamp":
            settings["session_prefix"] = value or "session"
        return settings

    def _apply_project_store_settings(self, settings):
        """Apply project-store settings to UI controls."""
        merged = dict(DEFAULT_PROJECT_STORE_SETTINGS)
        merged.update(settings or {})
        self.provenance_enabled_check.setChecked(bool(merged.get("provenance_enabled", True)))
        self.project_base_edit.setText(str(merged.get("base_project_dir", ".pymaris_project")))
        naming = str(merged.get("session_naming", "timestamp"))
        index = self.session_naming_combo.findText(naming)
        if index >= 0:
            self.session_naming_combo.setCurrentIndex(index)
        if naming == "fixed":
            value = str(merged.get("session_name", "default"))
        else:
            value = str(merged.get("session_prefix", "session"))
        self.session_value_edit.setText(value)
        self._update_session_value_label(naming)

    def _on_session_naming_changed(self, naming):
        """Update session field label based on naming mode."""
        self._update_session_value_label(naming)

    def _update_session_value_label(self, naming):
        if naming == "fixed":
            self.session_value_label.setText("Session Name")
            self.session_value_edit.setPlaceholderText("default")
        elif naming == "timestamp":
            self.session_value_label.setText("Session Prefix")
            self.session_value_edit.setPlaceholderText("session")
        else:
            self.session_value_label.setText("Session Value")
            self.session_value_edit.setPlaceholderText("(unused)")

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
            try:
                from pymaris_napari.configurable import load_widget
            except Exception:
                from main_napari_configurable import load_widget

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
