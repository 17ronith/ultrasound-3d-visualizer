"""
Ex 27 — Export segmentation visualization to video file.
Renders JugularVein segmentation frames, collects them, and saves as MP4 via imageio.
Output: segmentation_video.mp4 in the tutorial/ directory.
"""
from _config import fast, DATA

try:
    import imageio
except ImportError:
    print("imageio not installed. Run: pip install imageio[ffmpeg]")
    raise

streamer = fast.ImageFileStreamer.create(
    DATA + 'US/JugularVein/US-2D_#.mhd',
    loop=False,
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

renderToImage = fast.RenderToImage.create(bgcolor=fast.Color.Black())\
    .connect([imageRenderer, segmentationRenderer, labelRenderer])

frames = []
for image in fast.DataStream(renderToImage):
    frames.append(image)
    print(f"Collected frame {len(frames)}")

output_path = 'segmentation_video.mp4'
imageio.mimsave(output_path, frames, fps=20)
print(f"Saved {len(frames)} frames to {output_path}")
