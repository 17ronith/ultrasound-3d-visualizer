# tests/test_preprocessor.py
import numpy as np
import pytest
from src.preprocessor import preprocess


def test_preprocess_normalizes_to_01(gray_array_uint8):
    result = preprocess(gray_array_uint8, {'dimensionality': '2D_single', 'anatomy': 'unknown'})
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_converts_rgb_to_gray():
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = preprocess(rgb, {'dimensionality': '2D_single', 'anatomy': 'unknown'})
    assert result.shape == (64, 64, 1)


def test_preprocess_keeps_grayscale_shape(gray_array_uint8):
    result = preprocess(gray_array_uint8, {'dimensionality': '2D_single', 'anatomy': 'unknown'})
    assert result.shape == (64, 64, 1)


def test_preprocess_handles_list_of_arrays():
    frames = [np.random.randint(0, 255, (32, 32, 1), dtype=np.uint8) for _ in range(3)]
    result = preprocess(frames, {'dimensionality': '2D_sequence', 'anatomy': 'unknown'})
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0].dtype == np.float32


def test_preprocess_3d_volume():
    vol = np.random.randint(0, 255, (10, 32, 32, 1), dtype=np.uint8)
    result = preprocess(vol, {'dimensionality': '3D_volume', 'anatomy': 'Ball'})
    assert result.dtype == np.float32
    assert result.shape == (10, 32, 32, 1)
    assert result.max() <= 1.0
