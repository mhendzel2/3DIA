
import numpy as np
import pytest
from unittest.mock import Mock

def test_spots_detection_logic():
    """Test the spots_detection_logic's core logic."""
    from src.widgets.spots_detection_logic import detect_spots

    # Create a dummy image
    image = np.zeros((100, 100))
    image[25, 25] = 1
    image[50, 50] = 1
    image[75, 75] = 1

    # Create a mock image layer
    mock_layer = Mock()
    mock_layer.data = image

    # Run the function
    points_data = detect_spots(
        image_layer=mock_layer,
        min_sigma=1.0,
        max_sigma=2.0,
        num_sigma=5,
        threshold=0.1,
    )

    # Check that the function returns points data
    assert isinstance(points_data, np.ndarray)
    assert points_data.shape[1] == 2  # 2D coordinates
    assert len(points_data) > 0
