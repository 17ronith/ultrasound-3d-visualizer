# tests/test_loader.py
import pytest
from src.loader import detect_format, detect_anatomy, detect_dimensionality


def test_detect_format_jpg():
    assert detect_format('US/US-2D.jpg') == 'image'


def test_detect_format_png():
    assert detect_format('US/US-2D.png') == 'image'


def test_detect_format_bmp():
    assert detect_format('US/US-2D.bmp') == 'image'


def test_detect_format_mhd():
    assert detect_format('US/JugularVein/US-2D_0.mhd') == 'mhd'


def test_detect_format_dcm():
    assert detect_format('CT/LIDC-IDRI-0072/000001.dcm') == 'dicom'


def test_detect_format_avi():
    assert detect_format('US/sagittal_spine.avi') == 'video'


def test_detect_format_mp4():
    assert detect_format('scan.mp4') == 'video'


def test_detect_anatomy_jugular():
    assert detect_anatomy('/data/US/JugularVein/US-2D_0.mhd') == 'JugularVein'


def test_detect_anatomy_carotid():
    assert detect_anatomy('/data/US/CarotidArtery/Left/US-2D_0.mhd') == 'CarotidArtery'


def test_detect_anatomy_femoral():
    assert detect_anatomy('/data/US/FemoralArtery/Right/US-2D_5.mhd') == 'FemoralArtery'


def test_detect_anatomy_lidc():
    assert detect_anatomy('/data/CT/LIDC-IDRI-0072/000001.dcm') == 'LIDC'


def test_detect_anatomy_ball():
    assert detect_anatomy('/data/US/Ball/US-3Dt_0.mhd') == 'Ball'


def test_detect_anatomy_heart():
    assert detect_anatomy('/data/US/Heart/ApicalFourChamber/US-2D_0.mhd') == 'Heart'


def test_detect_anatomy_unknown():
    assert detect_anatomy('/data/US/US-2D.jpg') == 'unknown'


def test_detect_dimensionality_single():
    assert detect_dimensionality('image', '/any/file.jpg') == '2D_single'


def test_detect_dimensionality_video():
    assert detect_dimensionality('video', '/any/file.avi') == '2D_sequence'


def test_detect_dimensionality_dicom():
    assert detect_dimensionality('dicom', '/any/file.dcm') == '2D_sequence'


import numpy as np
from src.loader import load


def test_load_single_jpg(single_jpg_path):
    data, meta = load(single_jpg_path)
    assert isinstance(data, np.ndarray)
    assert data.ndim == 3           # (H, W, C)
    assert meta['format'] == 'image'
    assert meta['anatomy'] == 'unknown'
    assert meta['dimensionality'] == '2D_single'


def test_load_single_mhd(jugular_mhd_path):
    data, meta = load(jugular_mhd_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (318, 492, 1)
    assert meta['format'] == 'mhd'
    assert meta['anatomy'] == 'JugularVein'
    assert meta['dimensionality'] == '2D_single'


def test_load_3d_mhd_sets_volume_dimensionality(ball_mhd_path):
    data, meta = load(ball_mhd_path)
    assert isinstance(data, np.ndarray)
    assert data.ndim == 4           # (D, H, W, C)
    assert meta['dimensionality'] == '3D_volume'
    assert meta['anatomy'] == 'Ball'
