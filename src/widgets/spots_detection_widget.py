
from magicgui import magicgui
import numpy as np
from .spots_detection_logic import detect_spots
from pymaris.advanced_analysis import compute_clark_evans_3d

@magicgui(
    auto_call=True,
    min_sigma={"widget_type": "FloatSlider", "min": 0.5, "max": 50},
    max_sigma={"widget_type": "FloatSlider", "min": 1, "max": 100},
    threshold={"widget_type": "FloatSlider", "min": 0, "max": 1, "step": 0.001},
    num_sigma={"widget_type": "Slider", "min": 1, "max": 20},
    result_widget=True,
)
def spots_detection_widget(
    image_layer: "napari.layers.Image",
    min_sigma: float = 1.0,
    max_sigma: float = 10.0,
    num_sigma: int = 10,
    threshold: float = 0.1,
) -> "napari.types.PointsData":
    return detect_spots(image_layer, min_sigma, max_sigma, num_sigma, threshold)


@magicgui(
    auto_call=False,
    call_button="Compute Spatial Statistics",
    volume_z={"widget_type": "FloatSpinBox", "value": 100.0, "min": 1e-6},
    volume_y={"widget_type": "FloatSpinBox", "value": 100.0, "min": 1e-6},
    volume_x={"widget_type": "FloatSpinBox", "value": 100.0, "min": 1e-6},
    result_widget=True,
)
def advanced_spatial_statistics_widget(
    points_layer: "napari.layers.Points",
    volume_z: float = 100.0,
    volume_y: float = 100.0,
    volume_x: float = 100.0,
) -> str:
    """Compute 3D Clark-Evans index and classify population structure."""
    if points_layer is None or points_layer.data is None or len(points_layer.data) < 2:
        return "Need at least two spot coordinates to compute Clark-Evans statistics."

    points = np.asarray(points_layer.data, dtype=float)
    if points.ndim != 2:
        return "Invalid points data shape."
    if points.shape[1] > 3:
        points = points[:, -3:]
    if points.shape[1] == 2:
        points = np.c_[np.zeros(points.shape[0], dtype=float), points]

    try:
        r_index = compute_clark_evans_3d(points, (float(volume_z), float(volume_y), float(volume_x)))
    except Exception as exc:
        return f"Spatial statistics failed: {exc}. Install with: pip install pymaris[advanced]"

    if r_index < 0.95:
        classification = "Clustered"
    elif r_index > 1.05:
        classification = "Dispersed"
    else:
        classification = "Random"

    message = f"Clark-Evans R={r_index:.4f} -> {classification}"
    print(message)
    return message
