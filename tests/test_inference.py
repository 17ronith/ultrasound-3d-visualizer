# tests/test_inference.py
import numpy as np
import pytest
from src.inference import compute_confidence, run_inference


def test_confidence_all_detected(binary_mask):
    conf = compute_confidence(binary_mask.astype(np.float32), binary_mask)
    assert conf['detected'] == 'True'
    assert float(conf['coverage'].rstrip('%')) > 0
    assert float(conf['mean_confidence'].rstrip('%')) > 0


def test_confidence_nothing_detected():
    empty_mask = np.zeros((64, 64, 1), dtype=np.uint8)
    prob = np.zeros((64, 64, 1), dtype=np.float32)
    conf = compute_confidence(prob, empty_mask)
    assert conf['detected'] == 'False'
    assert conf['coverage'] == '0.00%'
    assert conf['mean_confidence'] == '0.00%'


def test_inference_fallback_returns_correct_shapes(gray_array_float):
    prob, mask, conf = run_inference(
        gray_array_float,
        {'anatomy': 'unknown', 'dimensionality': '2D_single'}
    )
    assert prob.shape == gray_array_float.shape
    assert mask.shape == gray_array_float.shape
    assert mask.dtype == np.uint8
    assert prob.dtype == np.float32


def test_inference_fallback_prob_equals_input(gray_array_float):
    prob, mask, conf = run_inference(
        gray_array_float,
        {'anatomy': 'unknown', 'dimensionality': '2D_single'}
    )
    np.testing.assert_array_almost_equal(prob, gray_array_float)


def test_inference_fallback_mask_thresholded(gray_array_float):
    prob, mask, conf = run_inference(
        gray_array_float,
        {'anatomy': 'unknown', 'dimensionality': '2D_single'}
    )
    expected_mask = (gray_array_float > 0.5).astype(np.uint8)
    np.testing.assert_array_equal(mask, expected_mask)


def test_inference_jugular_vein_returns_correct_shapes(data_path):
    import fast
    fast_img = fast.ImageFileImporter.create(data_path + 'US/JugularVein/US-2D_0.mhd').runAndGetOutputData()
    arr = np.asarray(fast_img).astype(np.float32) / 255.0
    prob, mask, conf = run_inference(arr, {'anatomy': 'JugularVein', 'dimensionality': '2D_single'})
    assert prob.shape == arr.shape
    assert mask.shape == arr.shape
    assert prob.dtype == np.float32
    assert mask.dtype == np.uint8


def test_inference_jugular_detects_structure(data_path):
    import fast
    fast_img = fast.ImageFileImporter.create(data_path + 'US/JugularVein/US-2D_0.mhd').runAndGetOutputData()
    arr = np.asarray(fast_img).astype(np.float32) / 255.0
    _, mask, conf = run_inference(arr, {'anatomy': 'JugularVein', 'dimensionality': '2D_single'})
    assert mask.any(), "Expected at least one segmented pixel"
    assert conf['detected'] == 'True'
