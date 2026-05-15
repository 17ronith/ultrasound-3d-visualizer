"""
Ex 09 — Read 3D ultrasound (Ball phantom) and render with alpha-blending volume renderer.
"""
from _config import fast, DATA

streamer = fast.ImageFileStreamer.create(
    DATA + 'US/Ball/US-3Dt_#.mhd',
    framerate=5,
    loop=True,
)

renderer = fast.AlphaBlendingVolumeRenderer.create().connect(streamer)

fast.SimpleWindow3D.create().connect(renderer).run()
