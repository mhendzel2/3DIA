# main_napari.py
# PyMaris Scientific Image Analyzer - Napari Desktop Application
# Enhanced with configurable widget loading

import napari
import json
from pathlib import Path

# Widget configuration
DEFAULT_CONFIG = {
    "enabled_widgets": {
        "file_io": True,
        "processing": True,
        "segmentation": True,
        "analysis": True,
        "visualization": True,
        "deconvolution": False,
        "statistics": True,
        "filament_tracing": True,
        "tracking": True,
        "simple_threshold": False,
        "adaptive_threshold": False,
        "hca": True,
        "ai_segmentation": False,
        "biophysics": False,
        "interactive_plotting": False
    },
    "widget_areas": {
        "file_io": "left",
        "processing": "left",
        "segmentation": "left",
        "analysis": "left",
        "visualization": "left",
        "deconvolution": "left",
        "statistics": "right",
        "filament_tracing": "right",
        "tracking": "right",
        "simple_threshold": "left",
        "adaptive_threshold": "left",
        "hca": "right",
        "ai_segmentation": "right",
        "biophysics": "right",
        "interactive_plotting": "right"
    },
    "load_on_startup": True,
    "show_welcome_message": True
}

def load_config():
    """Load widget configuration from file"""
    config_file = Path(__file__).parent / "config" / "widget_config.json"
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠ Could not load config: {e}")
            return DEFAULT_CONFIG
    else:
        # Create default config
        config_file.parent.mkdir(exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG

def load_widget(viewer, widget_name, config):
    """Load a single widget on demand"""
    if not config["enabled_widgets"].get(widget_name, False):
        print(f"⚠ {widget_name} is disabled in config")
        return None
    
    area = config["widget_areas"].get(widget_name, "left")
    
    try:
        if widget_name == "file_io":
            from widgets.file_io_widget import FileIOWidget
            widget = FileIOWidget(viewer)
            viewer.window.add_dock_widget(widget, name="File I/O", area=area)
            
        elif widget_name == "processing":
            from widgets.processing_widget import ProcessingWidget
            widget = ProcessingWidget(viewer)
            viewer.window.add_dock_widget(widget, name="Image Processing", area=area)
            
        elif widget_name == "segmentation":
            from widgets.segmentation_widget import SegmentationWidget
            widget = SegmentationWidget(viewer)
            viewer.window.add_dock_widget(widget, name="Segmentation", area=area)
            
        elif widget_name == "analysis":
            from widgets.analysis_widget import AnalysisWidget
            widget = AnalysisWidget(viewer)
            viewer.window.add_dock_widget(widget, name="Analysis & Plotting", area=area)
            
        elif widget_name == "visualization":
            from widgets.visualization_widget import VisualizationWidget
            widget = VisualizationWidget(viewer)
            viewer.window.add_dock_widget(widget, name="3D Visualization", area=area)
            
        elif widget_name == "deconvolution":
            from widgets.deconvolution_widget import DeconvolutionWidget
            widget = DeconvolutionWidget(viewer)
            viewer.window.add_dock_widget(widget, name="Deconvolution", area=area)
            
        elif widget_name == "statistics":
            from widgets.statistics_widget import StatisticsWidget
            widget = StatisticsWidget(viewer)
            viewer.window.add_dock_widget(widget, name="Statistics", area=area)
            
        elif widget_name == "filament_tracing":
            from widgets.filament_tracing_widget import FilamentTracingWidget
            widget = FilamentTracingWidget(viewer)
            viewer.window.add_dock_widget(widget, name="Filament Tracing", area=area)
            
        elif widget_name == "tracking":
            from widgets.tracking_widget import AdvancedTrackingWidget
            widget = AdvancedTrackingWidget(viewer)
            viewer.window.add_dock_widget(widget, name="Cell Tracking & Lineage", area=area)
            
        elif widget_name == "simple_threshold":
            from widgets.magicgui_analysis_widget import simple_threshold_widget
            viewer.window.add_dock_widget(simple_threshold_widget, name="Simple Threshold", area=area)
            widget = simple_threshold_widget
            
        elif widget_name == "adaptive_threshold":
            from widgets.magicgui_analysis_widget import adaptive_threshold_widget
            viewer.window.add_dock_widget(adaptive_threshold_widget, name="Adaptive Threshold", area=area)
            widget = adaptive_threshold_widget
            
        elif widget_name == "hca":
            from widgets.hca_widget import HighContentAnalysisWidget
            widget = HighContentAnalysisWidget(viewer)
            viewer.window.add_dock_widget(widget, name="High-Content Analysis", area=area)
            
        elif widget_name == "ai_segmentation":
            from widgets.ai_segmentation_widget import AISegmentationWidget
            widget = AISegmentationWidget(viewer)
            viewer.window.add_dock_widget(widget, name="AI Segmentation", area=area)
            
        elif widget_name == "biophysics":
            from widgets.biophysics_widget import BiophysicsWidget
            widget = BiophysicsWidget(viewer)
            viewer.window.add_dock_widget(widget, name="Biophysics Analysis", area=area)
            
        elif widget_name == "interactive_plotting":
            from widgets.interactive_plotting_widget import InteractivePlottingWidget
            widget = InteractivePlottingWidget(viewer)
            viewer.window.add_dock_widget(widget, name="Interactive Plotting", area=area)
        
        else:
            print(f"⚠ Unknown widget: {widget_name}")
            return None
            
        print(f"✓ Loaded: {widget_name}")
        return widget
        
    except ImportError as e:
        print(f"⚠ Could not load {widget_name}: {e}")
        return None
    except Exception as e:
        print(f"✗ Error loading {widget_name}: {e}")
        return None

def main():
    """Main entry point for Napari application"""
    # Load configuration
    config = load_config()
    
    # Create Napari viewer
    viewer = napari.Viewer(title="PyMaris Scientific Image Analyzer")
    
    # Add Widget Manager first
    try:
        from widgets.widget_manager import WidgetManagerWidget
        manager = WidgetManagerWidget(viewer)
        viewer.window.add_dock_widget(manager, name="Widget Manager", area="right")
        print("✓ Widget Manager loaded")
    except Exception as e:
        print(f"⚠ Could not load Widget Manager: {e}")
    
    # Load enabled widgets
    if config["load_on_startup"]:
        print("\nLoading widgets...")
        loaded_count = 0
        
        for widget_name in config["enabled_widgets"].keys():
            if config["enabled_widgets"][widget_name]:
                result = load_widget(viewer, widget_name, config)
                if result is not None:
                    loaded_count += 1
        
        print(f"\n✓ Loaded {loaded_count} widgets")
    
    # Show welcome message
    if config["show_welcome_message"]:
        print("\nPyMaris application is ready with advanced widgets.")
        print("New Imaris-like features:")
        print("  • Volume Rendering (MIP, Alpha Blending, Orthogonal Views, Clipping)")
        print("  • Filament Tracing (Neuron/Cytoskeleton Analysis)")
        print("  • Advanced Cell Tracking (Lineage Trees, Gap Closing, Division Detection)")
        print("\nTo configure widgets: Edit config/widget_config.json")
        print("Go to File > Open... to load an image.")
    
    # Start Napari event loop
    napari.run()

if __name__ == "__main__":
    main()
