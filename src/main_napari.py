# main_napari.py
# PyMaris Scientific Image Analyzer - Napari Desktop Application
# Fixed to use advanced widgets from widgets/ directory

import napari

# Import the correct advanced widget classes from the widgets directory
try:
    from widgets.processing_widget import ProcessingWidget
    from widgets.segmentation_widget import SegmentationWidget
    from widgets.analysis_widget import AnalysisWidget
    from widgets.visualization_widget import VisualizationWidget
    from widgets.file_io_widget import FileIOWidget
    ADVANCED_WIDGETS = True
except ImportError:
    # Fallback to basic widgets if advanced ones not available
    from processing_widget import processing_widget
    from segmentation_widget import segmentation_widget
    from analysis_widget import analysis_widget
    ADVANCED_WIDGETS = False
    print("Advanced widgets not found, using basic widgets")

def main():
    """Main entry point for Napari application"""
    # 1. Create a Napari viewer instance.
    viewer = napari.Viewer(title="PyMaris (Corrected Widgets)" if ADVANCED_WIDGETS else "PyMaris (Basic Widgets)")

    # 2. Add the custom widgets to the viewer.
    if ADVANCED_WIDGETS:
        # Use advanced threaded widgets from widgets/ directory
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
        print("\nPyMaris application is ready with advanced widgets.")
    else:
        # Use basic widgets as fallback
        viewer.window.add_dock_widget(
            processing_widget(),
            name="Image Processing",
            area='left'
        )
        viewer.window.add_dock_widget(
            segmentation_widget(),
            name="Segmentation",
            area='left'
        )
        viewer.window.add_dock_widget(
            analysis_widget(),
            name="Analysis & Plotting",
            area='left'
        )
        print("\nPyMaris application is ready with basic widgets.")

    print("Go to File > Open... to load an image.")

    # 3. Start the Napari event loop.
    napari.run()

if __name__ == "__main__":
    main()