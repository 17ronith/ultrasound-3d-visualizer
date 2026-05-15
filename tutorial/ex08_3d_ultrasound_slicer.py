"""
Ex 08 — Read 3D ultrasound (Ball phantom) and display with SlicerWindow.
Ball/ contains 84 volumetric MHD frames (276x249x200 voxels each).
"""
from _config import fast, DATA

streamer = fast.ImageFileStreamer.create(
    DATA + 'US/Ball/US-3Dt_#.mhd',
    framerate=5,
    loop=True,
)

fast.SlicerWindow.create()\
    .connectImage(streamer)\
    .run()
