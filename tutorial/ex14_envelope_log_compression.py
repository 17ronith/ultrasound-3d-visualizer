"""
Ex 14 — Envelope detection and log compression on synthetic IQ data.
"""
from _config import fast, DATA
import numpy as np
import matplotlib.pyplot as plt

iq_data = fast.Image.createFromArray(
    np.random.normal(size=(512, 512, 2)).astype(np.float32)
)

envelope = fast.EnvelopeAndLogCompressor.create().connect(iq_data)

scan_convert = fast.ScanConverter.create(
    width=1280,
    height=1024,
    startDepth=0,
    endDepth=120,
    startAngle=-0.785398,
    endAngle=0.785398,
).connect(envelope)

# Matplotlib preview
result = np.asarray(scan_convert.runAndGetOutputData())
plt.figure(figsize=(6, 5))
plt.title("Envelope + log-compressed synthetic IQ → scan converted")
plt.imshow(result[..., 0], cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

# Interactive FAST window
fast.display2D(scan_convert)
