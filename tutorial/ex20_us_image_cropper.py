"""
Ex 20 — Ultrasound image cropper.
Automatically crops the B-mode fan region from the sagittal spine video.
Side-by-side: original (left) vs cropped (right).
"""
from _config import fast, DATA

streamer = fast.MovieStreamer.create(DATA + "US/sagittal_spine.avi")

cropper = fast.UltrasoundImageCropper.create(
    staticCropping=False,
    thresholdVertical=30,
    thresholdHorizontal=10,
).connect(streamer)

renderer = fast.ImageRenderer.create().connect(streamer)
renderer2 = fast.ImageRenderer.create().connect(cropper)

fast.DualViewWindow2D.create()\
    .connectLeft(renderer)\
    .connectRight(renderer2)\
    .run()
