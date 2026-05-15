"""
Ex 19 — NonLocalMeans speckle denoising.
Side-by-side dual window: original (left) vs denoised (right).
Uses Heart/ApicalFourChamber MHD sequence.
"""
from _config import fast, DATA

streamer = fast.ImageFileStreamer.create(
    DATA + 'US/Heart/ApicalFourChamber/US-2D_#.mhd',
    framerate=2,
    loop=True,
)

nlm = fast.NonLocalMeans.create(
    filterSize=3,
    searchSize=11,
    smoothingAmount=0.2,
    inputMultiplicationWeight=0.5,
).connect(streamer)

renderer = fast.ImageRenderer.create().connect(streamer)
rendererNLM = fast.ImageRenderer.create().connect(nlm)

widget = fast.PlaybackWidget(streamer)
fast.DualViewWindow2D.create(width=1024, height=512)\
    .connectLeft(renderer)\
    .connectRight(rendererNLM)\
    .connect(widget)\
    .run()
