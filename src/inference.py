# src/inference.py
import fast
import numpy as np
from pathlib import Path

fast.Config.setTestDataPath('/Users/ronith/Documents/Projects/ultrasound/FAST/data/')
DATA = fast.Config.getTestDataPath()

_MODEL_ROUTES = {
    'JugularVein': DATA + 'NeuralNetworkModels/jugular_vein_segmentation.onnx',
    # LIDC uses a 3D sliding-window model handled by /analyze/nodule ORT session;
    # 2D dashboard preview falls back to pixel intensity.
}


def compute_confidence(probability_map: np.ndarray, binary_mask: np.ndarray) -> dict:
    detected = bool(binary_mask.any())
    total_pixels = binary_mask.size
    coverage = binary_mask.sum() / total_pixels * 100.0
    if detected:
        mean_conf = float(probability_map[binary_mask > 0].mean()) * 100.0
    else:
        mean_conf = 0.0
    return {
        'detected': str(detected),
        'coverage': f'{coverage:.2f}%',
        'mean_confidence': f'{mean_conf:.2f}%',
    }


def _fallback_inference(preprocessed: np.ndarray):
    prob = preprocessed.astype(np.float32)
    mask = (prob > 0.5).astype(np.uint8)
    return prob, mask, compute_confidence(prob, mask)


def run_inference(preprocessed, metadata: dict):
    anatomy = metadata.get('anatomy', 'unknown')

    if anatomy not in _MODEL_ROUTES:
        return _fallback_inference(preprocessed)

    return _run_onnx(preprocessed, anatomy)


def _run_onnx(preprocessed: np.ndarray, anatomy: str):
    model_path = _MODEL_ROUTES[anatomy]
    H, W = preprocessed.shape[:2]

    fast_img = fast.Image.createFromArray(preprocessed)
    net = fast.SegmentationNetwork.create(model_path, scaleFactor=1.0).connect(fast_img)
    seg_arr = np.asarray(net.runAndGetOutputData())   # (256, 256, 1) uint8

    import cv2
    seg_resized = cv2.resize(
        seg_arr.squeeze().astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST
    )[:, :, np.newaxis]

    binary_mask = (seg_resized > 0).astype(np.uint8)
    probability_map = binary_mask.astype(np.float32)

    return probability_map, binary_mask, compute_confidence(probability_map, binary_mask)
