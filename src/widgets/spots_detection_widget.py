
from magicgui import magicgui
from .spots_detection_logic import detect_spots

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
