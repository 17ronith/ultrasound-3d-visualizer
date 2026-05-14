# tests/test_visualizer.py
import numpy as np
import pytest
from src.visualizer import visualize


@pytest.fixture
def single_inputs():
    H, W = 64, 64
    orig = np.random.randint(0, 255, (H, W, 1), dtype=np.uint8)
    prob = np.random.rand(H, W, 1).astype(np.float32)
    mask = (prob > 0.5).astype(np.uint8)
    conf = {'detected': 'True', 'coverage': '12.50%', 'mean_confidence': '74.10%'}
    meta = {'anatomy': 'JugularVein', 'dimensionality': '2D_single', 'format': 'mhd'}
    return orig, prob, mask, conf, meta


def test_visualize_mode1_returns_html_string(single_inputs):
    orig, prob, mask, conf, meta = single_inputs
    html = visualize(orig, prob, mask, conf, meta)
    assert isinstance(html, str)
    assert '<!DOCTYPE html>' in html


def test_visualize_mode1_contains_plotly(single_inputs):
    orig, prob, mask, conf, meta = single_inputs
    html = visualize(orig, prob, mask, conf, meta)
    assert 'plotly' in html.lower()


def test_visualize_mode1_contains_anatomy_title(single_inputs):
    orig, prob, mask, conf, meta = single_inputs
    html = visualize(orig, prob, mask, conf, meta)
    assert 'JugularVein' in html


def test_visualize_mode1_contains_confidence(single_inputs):
    orig, prob, mask, conf, meta = single_inputs
    html = visualize(orig, prob, mask, conf, meta)
    assert '12.50%' in html


def test_visualize_unknown_anatomy_uses_white_ring():
    H, W = 32, 32
    orig = np.zeros((H, W, 1), dtype=np.uint8)
    prob = np.ones((H, W, 1), dtype=np.float32) * 0.8
    mask = np.ones((H, W, 1), dtype=np.uint8)
    conf = {'detected': 'True', 'coverage': '100.00%', 'mean_confidence': '80.00%'}
    meta = {'anatomy': 'unknown', 'dimensionality': '2D_single', 'format': 'image'}
    html = visualize(orig, prob, mask, conf, meta)
    assert 'white' in html.lower() or '#ffffff' in html.lower() or 'rgb(255' in html.lower()
