"""
Ex 07 — Stream an image sequence from the Heart/ApicalFourChamber MHD series.
Use # as the frame index placeholder in the path.
"""
from _config import fast, DATA

streamer = fast.ImageFileStreamer.create(
    DATA + 'US/Heart/ApicalFourChamber/US-2D_#.mhd',
    framerate=20,
    loop=True,
)

fast.display2D(streamer)
