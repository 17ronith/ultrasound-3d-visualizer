"""
Ex 02 — Read ultrasound video and display (full window API).
Opens an interactive FAST 2D window. Close the window to exit.
"""
from _config import fast, DATA

streamer = fast.MovieStreamer.create(DATA + "US/sagittal_spine.avi")

renderer = fast.ImageRenderer.create().connect(streamer)

fast.SimpleWindow2D.create().connect(renderer).run()
