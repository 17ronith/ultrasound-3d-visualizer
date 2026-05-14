# src/preprocessor.py
import fast
import numpy as np

fast.Config.setTestDataPath('/Users/ronith/Documents/Projects/ultrasound/FAST/data/')


def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert array to single-channel grayscale, preserving shape (H, W, 1)."""
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        gray = arr.mean(axis=2, keepdims=True).astype(arr.dtype)
        return gray
    return arr


def _nlm_denoise(arr: np.ndarray) -> np.ndarray:
    """Apply FAST NonLocalMeans denoising. Input must be (H, W, 1) uint8."""
    inp = arr if arr.dtype == np.uint8 else (arr * 255).clip(0, 255).astype(np.uint8)
    fast_img = fast.Image.createFromArray(inp)
    nlm = fast.NonLocalMeans.create(
        filterSize=3,
        searchSize=11,
        smoothingAmount=0.2,
        inputMultiplicationWeight=0.5,
    ).connect(fast_img)
    return np.asarray(nlm.runAndGetOutputData())


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize uint8 array to float32 in [0, 1]."""
    return arr.astype(np.float32) / 255.0


def _process_single(arr: np.ndarray) -> np.ndarray:
    """Grayscale → NLM denoise → normalize a single 2D frame."""
    arr = _to_grayscale(arr)
    arr = _nlm_denoise(arr)
    return _normalize(arr)


def preprocess(data, metadata: dict):
    """
    Preprocess image data: convert to grayscale, NLM denoise, normalize to [0, 1].

    Args:
        data: np.ndarray (H, W, C) for 2D_single; list of np.ndarray for 2D_sequence;
              np.ndarray (D, H, W, 1) for 3D_volume.
        metadata: dict with at least 'dimensionality' key.

    Returns:
        np.ndarray or list of np.ndarray, dtype float32, values in [0, 1].
    """
    dim = metadata.get('dimensionality', '2D_single')

    if dim == '2D_sequence' and isinstance(data, list):
        return [_process_single(frame) for frame in data]

    if dim == '3D_volume' and isinstance(data, np.ndarray) and data.ndim == 4:
        vol = data
        if vol.shape[-1] == 1:
            vol = vol[..., 0]   # (D, H, W)
        else:
            vol = vol.mean(axis=-1)  # (D, H, W)
        normalized = vol.astype(np.float32) / 255.0
        return normalized[..., np.newaxis]  # (D, H, W, 1)

    return _process_single(data)
