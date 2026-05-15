"""
Ex 10 — Add a playback widget (play/pause/scrub) to a streamed sequence.
"""
from _config import fast, DATA

streamer = fast.ImageFileStreamer.create(
    DATA + 'US/Heart/ApicalFourChamber/US-2D_#.mhd',
    framerate=20,
)

widget = fast.PlaybackWidget(streamer)
fast.display2D(streamer, widgets=[widget])
