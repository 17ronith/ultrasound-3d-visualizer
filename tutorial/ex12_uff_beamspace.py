"""
Ex 12 — Read UFF data in beamspace (no scan conversion).
Turns off scan conversion to get raw polar/beamspace data, then converts dB to grayscale.
"""
from _config import fast, DATA

streamer = fast.UFFStreamer.create(
    DATA + "US/UFF/P4_2_PLAX.uff",
    framerate=5,
    loop=True,
    doScanConversion=False,
    convertToGrayscale=True,
    gain=0,
    dynamicRange=60,
)

widget = fast.PlaybackWidget(streamer)
fast.display2D(streamer, widgets=[widget])
