"""
Ex 26 — Neural network image classification.

NOTE: No classification .onnx model is bundled in the FAST test dataset.
This example shows the API pattern using a placeholder model path.
To run for real, supply a trained classification model:
    classificationNetwork = fast.ImageClassificationNetwork.create(
        'path/to/classification_model.onnx',
        labels=['Normal', 'Artery', 'Vein'],
        scaleFactor=1./255.
    ).connect(streamer)

The wsi_classification.onnx in NeuralNetworkModels/ is for whole-slide images,
not ultrasound — it won't produce meaningful results on JugularVein data.
"""
print(__doc__)
