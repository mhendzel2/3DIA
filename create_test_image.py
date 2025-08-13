from PIL import Image
import numpy as np

# Create a black image
img_array = np.zeros((100, 100, 3), dtype=np.uint8)
# Add a white square in the middle
img_array[25:75, 25:75] = [255, 255, 255]

img = Image.fromarray(img_array)
img.save("test_image.png")
