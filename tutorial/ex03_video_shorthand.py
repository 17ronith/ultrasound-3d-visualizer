"""
Ex 03 — Read ultrasound video, shorthand display.
Equivalent to ex02 using the convenience display2D function.
"""
from _config import fast, DATA

streamer = fast.MovieStreamer.create(DATA + "US/sagittal_spine.avi")

fast.display2D(streamer)
