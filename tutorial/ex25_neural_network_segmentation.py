"""
Ex 25 — Neural network segmentation of jugular vein.
Uses the bundled jugular_vein_segmentation.onnx model on JugularVein MHD sequences.
Renders artery (red) and vein (blue) overlays with labels.
"""
from _config import fast, DATA

streamer = fast.ImageFileStreamer.create(
    DATA + 'US/JugularVein/US-2D_#.mhd',
    loop=True,
)

segmentationNetwork = fast.SegmentationNetwork.create(
    DATA + 'NeuralNetworkModels/jugular_vein_segmentation.onnx',
    scaleFactor=1. / 255.,
).connect(streamer)

imageRenderer = fast.ImageRenderer.create().connect(streamer)

segmentationRenderer = fast.SegmentationRenderer.create(
    opacity=0.25,
    colors={1: fast.Color.Red(), 2: fast.Color.Blue()},
).connect(segmentationNetwork)

labelRenderer = fast.SegmentationLabelRenderer.create(
    labelNames={1: 'Artery', 2: 'Vein'},
    labelColors={1: fast.Color.Red(), 2: fast.Color.Blue()},
).connect(segmentationNetwork)

widget = fast.PlaybackWidget(streamer)

fast.SimpleWindow2D.create(bgcolor=fast.Color.Black())\
    .connect([imageRenderer, segmentationRenderer, labelRenderer])\
    .connect(widget)\
    .run()
