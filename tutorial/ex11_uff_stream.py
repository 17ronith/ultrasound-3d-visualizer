"""
Ex 11 — Read UFF (Ultrafast Ultrasound Format) file with scan conversion.
Uses P4_2_PLAX.uff (parasternal long-axis cardiac view).
"""
from _config import fast, DATA

streamer = fast.UFFStreamer.create(
    DATA + "US/UFF/P4_2_PLAX.uff",
    framerate=5,
    loop=True,
    scanConversionWidth=1024,
    scanConversionHeight=1024,
)

widget = fast.PlaybackWidget(streamer)
fast.display2D(streamer, widgets=[widget])
