
from napari.layers import Image
from skimage.feature import blob_log
from napari.types import PointsData
import numpy as np

def detect_spots(
    image_layer: Image,
    min_sigma: float = 1.0,
    max_sigma: float = 10.0,
    num_sigma: int = 10,
    threshold: float = 0.1,
) -> PointsData:
    if image_layer is None:
        return None

    data = image_layer.data

    if data.ndim not in [2, 3]:
        print("This function only works on 2D or 3D images.")
        return None

    blobs = blob_log(
        data,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
    )

    if blobs.shape[0] > 0:
        coords = blobs[:, :data.ndim]
        return coords

    return None
