"""
Ex 21 — Apply a predefined S-curve ultrasound colormap.
Side-by-side: grayscale (left) vs colormap (right).
"""
from _config import fast, DATA

streamer = fast.ImageFileStreamer.create(
    DATA + 'US/Heart/ApicalFourChamber/US-2D_#.mhd',
    loop=True,
    framerate=20,
)

colormap = fast.Colormap.Ultrasound()

apply = fast.ApplyColormap.create(colormap).connect(streamer)

renderer = fast.ImageRenderer.create().connect(streamer)
rendererColormap = fast.ImageRenderer.create().connect(apply)

fast.DualViewWindow2D.create(width=1024, height=512)\
    .connectLeft(renderer)\
    .connectRight(rendererColormap)\
    .run()
