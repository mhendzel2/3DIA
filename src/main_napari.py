# main_napari.py
# PyMaris Scientific Image Analyzer - Napari Desktop Application
# Fixed to use advanced widgets from widgets/ directory

import napari

# Import the advanced widget classes from the widgets directory
from widgets.processing_widget import ProcessingWidget
from widgets.segmentation_widget import SegmentationWidget
from widgets.analysis_widget import AnalysisWidget
from widgets.visualization_widget import VisualizationWidget
from widgets.file_io_widget import FileIOWidget
from widgets.deconvolution_widget import DeconvolutionWidget
from widgets.statistics_widget import StatisticsWidget
from widgets.filament_tracing_widget import FilamentTracingWidget
from widgets.tracking_widget import AdvancedTrackingWidget
from widgets.magicgui_analysis_widget import simple_threshold_widget, adaptive_threshold_widget

def main():
    """Main entry point for Napari application"""
    # 1. Create a Napari viewer instance.
    viewer = napari.Viewer(title="PyMaris Scientific Image Analyzer")

    # 2. Add the advanced widgets to the viewer.
    viewer.window.add_dock_widget(
        FileIOWidget(viewer),
        name="File I/O",
        area='left'
    )
    viewer.window.add_dock_widget(
        ProcessingWidget(viewer),
        name="Image Processing",
        area='left'
    )
    viewer.window.add_dock_widget(
        SegmentationWidget(viewer),
        name="Segmentation",
        area='left'
    )
    viewer.window.add_dock_widget(
        AnalysisWidget(viewer),
        name="Analysis & Plotting",
        area='left'
    )
    viewer.window.add_dock_widget(
        VisualizationWidget(viewer),
        name="3D Visualization",
        area='left'
    )
    viewer.window.add_dock_widget(
        DeconvolutionWidget(viewer),
        name="Deconvolution",
        area='left'
    )
    viewer.window.add_dock_widget(
        StatisticsWidget(viewer),
        name="Statistics",
        area='right'
    )
    
    # Add new Imaris-like features
    viewer.window.add_dock_widget(
        FilamentTracingWidget(viewer),
        name="Filament Tracing",
        area='right'
    )
    
    viewer.window.add_dock_widget(
        AdvancedTrackingWidget(viewer),
        name="Cell Tracking & Lineage",
        area='right'
    )
    
    # Add magicgui widgets
    viewer.window.add_dock_widget(simple_threshold_widget, name="Simple Threshold", area="left")
    viewer.window.add_dock_widget(adaptive_threshold_widget, name="Adaptive Threshold", area="left")

    try:
        from widgets.hca_widget import HighContentAnalysisWidget
        hca_widget = HighContentAnalysisWidget(viewer)
        viewer.window.add_dock_widget(hca_widget, name="High-Content Analysis", area="right")
        print("✓ HCA widget loaded successfully")
    except ImportError as e:
        print(f"⚠ HCA widget not available: {e}")
    
    print("\nPyMaris application is ready with advanced widgets.")
    print("New Imaris-like features:")
    print("  • Volume Rendering (MIP, Alpha Blending, Orthogonal Views, Clipping)")
    print("  • Filament Tracing (Neuron/Cytoskeleton Analysis)")
    print("  • Advanced Cell Tracking (Lineage Trees, Gap Closing, Division Detection)")
    print("Go to File > Open... to load an image.")

    # 3. Start the Napari event loop.
    napari.run()

if __name__ == "__main__":
    main()
