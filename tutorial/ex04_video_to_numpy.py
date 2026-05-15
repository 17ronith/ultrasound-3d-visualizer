"""
Ex 04 — Convert video frames to numpy arrays and display with matplotlib.
Samples every 20th frame, shows a 3x3 grid of 9 frames.
"""
from _config import fast, DATA
import numpy as np
import matplotlib.pyplot as plt

streamer = fast.MovieStreamer.create(
    DATA + "US/sagittal_spine.avi",
    grayscale=True,
)

frame_list = []
counter = 0
for frame in fast.DataStream(streamer):
    counter += 1
    if counter % 20 == 0:
        frame_list.append((np.asarray(frame), counter))
    if len(frame_list) == 9:
        f, axes = plt.subplots(3, 3, figsize=(10, 10))
        for i in range(3):
            for j in range(3):
                axes[j, i].set_title('Frame: ' + str(frame_list[i + j * 3][1]))
                axes[j, i].imshow(frame_list[i + j * 3][0][..., 0], cmap='gray')
                axes[j, i].axis('off')
        plt.tight_layout()
        plt.show()
        frame_list.clear()
        break
