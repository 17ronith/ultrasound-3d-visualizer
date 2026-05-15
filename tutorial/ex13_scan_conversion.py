"""
Ex 13 — Scan conversion: convert polar beamspace data to Cartesian image.
Creates synthetic beamspace data, applies ScanConverter, displays with both FAST and matplotlib.
"""
from _config import fast, DATA
import numpy as np
import matplotlib.pyplot as plt

data = fast.Image.createFromArray(
    np.round(np.random.normal(size=(700, 256, 1)) * 255).astype(np.uint8)
)

scan_convert = fast.ScanConverter.create(
    width=1280,
    height=1024,
    startDepth=0,
    endDepth=120,
    startAngle=-0.785398,
    endAngle=0.785398,
).connect(data)

# Matplotlib preview (non-blocking)
result = np.asarray(scan_convert.runAndGetOutputData())
plt.figure(figsize=(6, 5))
plt.title("Scan-converted synthetic beamspace data")
plt.imshow(result[..., 0], cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

# Interactive FAST window
fast.display2D(scan_convert)
