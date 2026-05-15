"""
Ex 23 — Custom Python process object.
Defines an Inverter class that flips pixel values (255 - x).
Optionally overlays text via OpenCV if available.
"""
from _config import fast, DATA
import numpy as np

try:
    import cv2
    use_opencv = True
except ImportError:
    use_opencv = False


class Inverter(fast.PythonProcessObject):
    def __init__(self):
        super().__init__()
        self.createInputPort(0)
        self.createOutputPort(0)

    def execute(self):
        image = self.getInputData()
        np_image = np.asarray(image)
        np_image = 255 - np_image

        if use_opencv:
            cv2.putText(np_image, 'OpenCV!', (40, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        new_image = fast.Image.createFromArray(np_image)
        new_image.setSpacing(image.getSpacing())
        self.addOutputData(0, new_image)


importer = fast.ImageFileStreamer.create(
    DATA + 'US/Heart/ApicalFourChamber/US-2D_#.mhd',
    loop=True,
    framerate=40,
)

inverter = Inverter.create().connect(importer)

fast.display2D(inverter)
