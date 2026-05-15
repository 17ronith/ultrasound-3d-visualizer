"""
Ex 06 — Read a single ultrasound image and convert to numpy.
Loads US-2D.jpg, prints shape/dtype/range, then converts to grayscale.
"""
from _config import fast, DATA
import numpy as np

image = fast.ImageFileImporter\
    .create(DATA + "US/US-2D.jpg")\
    .runAndGetOutputData()

data = np.asarray(image)
print("Color image  :", data.shape, data.dtype, np.min(data), np.max(data))

image = fast.ColorToGrayscale.create().connect(image).runAndGetOutputData()
data = np.asarray(image)
print("Grayscale    :", data.shape, data.dtype, np.min(data), np.max(data))
