"""
Ex 24 — Block matching speckle tracking on cardiac sequence.
Computes frame-to-frame motion vectors. Displays with matplotlib (quiver plot)
or FAST VectorFieldColorRenderer depending on the flag below.
"""
from _config import fast, DATA
import numpy as np
import matplotlib.pyplot as plt

VISUALIZE_WITH_MATPLOTLIB = True

streamer = fast.ImageFileStreamer.create(
    DATA + 'US/Heart/ApicalFourChamber/US-2D_#.mhd',
)

blockMatching = fast.BlockMatching.create(
    blockSize=13,
    searchSize=11,
    metric=fast.MatchingMetric_SUM_OF_ABSOLUTE_DIFFERENCES,
    timeLag=1,
    forwardBackwardTracking=False,
).connect(streamer)
blockMatching.setIntensityThreshold(75)

if VISUALIZE_WITH_MATPLOTLIB:
    frame_nr = 0
    for fast_image, vectorField in fast.DataStream(streamer, blockMatching):
        spacing = fast_image.getSpacing()
        image = np.asarray(fast_image)
        vf = np.asarray(vectorField)

        if frame_nr > 0:
            plt.figure(figsize=(8, 6))
            plt.imshow(image[..., 0], cmap='gray', aspect=spacing[1] / spacing[0])
            step = 8
            Y, X = np.mgrid[0:image.shape[0]:step, 0:image.shape[1]:step]
            plt.quiver(X, Y, vf[::step, ::step, 0], vf[::step, ::step, 1], color='r', scale=step * 10)
            plt.title(f"Block matching — frame {frame_nr}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        frame_nr += 1
        if fast_image.isLastFrame():
            break
else:
    imageRenderer = fast.ImageRenderer.create().connect(streamer)
    vectorRenderer = fast.VectorFieldColorRenderer.create().connect(blockMatching)
    fast.SimpleWindow2D.create()\
        .connect(imageRenderer)\
        .connect(vectorRenderer)\
        .run()
