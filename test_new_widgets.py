import numpy as np
import napari
import pytest
from PyQt6.QtWidgets import QApplication

from src.widgets.deconvolution_widget import DeconvolutionWidget
from src.widgets.statistics_widget import StatisticsWidget

# We need a QApplication instance to run Qt-based widgets
@pytest.fixture(scope="session")
def qt_application():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_deconvolution_widget(qt_application):
    """Test the DeconvolutionWidget."""
    viewer = napari.Viewer(show=False)
    widget = DeconvolutionWidget(viewer)
    viewer.window.add_dock_widget(widget)

    # Create a dummy image
    image = np.random.rand(100, 100)
    viewer.add_image(image, name="test_image")

    # Select the image in the widget
    widget.layer_combo.setCurrentText("test_image")

    # Test Richardson-Lucy
    widget.algo_combo.setCurrentText("richardson_lucy")
    widget.run_deconvolution()
    widget.thread.wait()  # Wait for the thread to finish

    # Check if a new layer was added
    assert len(viewer.layers) == 2
    assert "richardson_lucy_deconvolved" in [layer.name for layer in viewer.layers]

    # Test Wiener
    widget.algo_combo.setCurrentText("wiener")
    widget.run_deconvolution()
    widget.thread.wait()

    assert len(viewer.layers) == 3
    assert "wiener_deconvolved" in [layer.name for layer in viewer.layers]

    viewer.close()

def test_statistics_widget(qt_application):
    """Test the StatisticsWidget."""
    viewer = napari.Viewer(show=False)
    widget = StatisticsWidget(viewer)
    viewer.window.add_dock_widget(widget)

    # Create dummy image and labels
    image = np.random.rand(100, 100)
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[10:20, 10:20] = 1
    labels[30:40, 30:40] = 2

    viewer.add_image(image, name="test_image")
    viewer.add_labels(labels, name="test_labels")

    # Select layers in the widget
    widget.labels_combo.setCurrentText("test_labels")
    widget.intensity_combo.setCurrentText("test_image")

    # Calculate properties
    widget.calculate_properties()
    assert widget.dataframe is not None
    assert len(widget.dataframe) == 2

    # Generate plot
    widget.plot_column_combo.setCurrentText("area")
    widget.plot_type_combo.setCurrentText("Histogram")
    # In a non-interactive environment, plt.show() might block or fail.
    # We can't easily test the plot generation itself, but we can check if the method runs.
    try:
        widget.generate_plot()
    except Exception as e:
        pytest.fail(f"Plot generation failed with: {e}")

    viewer.close()
