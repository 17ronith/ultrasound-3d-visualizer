# src/loader.py
from pathlib import Path

import fast
import numpy as np

fast.Config.setTestDataPath('/Users/ronith/Documents/Projects/ultrasound/FAST/data/')


_FORMAT_MAP = {
    '.jpg': 'image', '.jpeg': 'image', '.png': 'image', '.bmp': 'image',
    '.mhd': 'mhd',
    '.dcm': 'dicom',
    '.avi': 'video', '.mp4': 'video',
}

_ANATOMY_KEYWORDS = [
    'JugularVein', 'CarotidArtery', 'FemoralArtery',
    'Heart', 'Ball', 'Axillary', 'LIDC',
]


def detect_format(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    return _FORMAT_MAP.get(ext, 'unknown')


def detect_anatomy(filepath: str) -> str:
    for keyword in _ANATOMY_KEYWORDS:
        if keyword in filepath:
            return keyword
    return 'unknown'


def detect_dimensionality(fmt: str, filepath: str) -> str:
    if fmt in ('video', 'dicom'):
        return '2D_sequence'
    return '2D_single'


def load(filepath: str):
    path = str(filepath)
    fmt = detect_format(path)
    anatomy = detect_anatomy(path)

    if fmt == 'image':
        fast_img = fast.ImageFileImporter.create(path).runAndGetOutputData()
        arr = np.asarray(fast_img)
        dim = detect_dimensionality(fmt, path)
        return arr, {'format': fmt, 'anatomy': anatomy, 'dimensionality': dim}

    if fmt == 'mhd':
        fast_img = fast.ImageFileImporter.create(path).runAndGetOutputData()
        arr = np.asarray(fast_img)
        dim = '3D_volume' if arr.ndim == 4 else '2D_single'
        return arr, {'format': fmt, 'anatomy': anatomy, 'dimensionality': dim}

    if fmt == 'video':
        streamer = fast.MovieStreamer.create(path, grayscale=True)
        frames = [np.asarray(f) for f in fast.DataStream(streamer)]
        return frames, {'format': fmt, 'anatomy': anatomy, 'dimensionality': '2D_sequence'}

    if fmt == 'dicom':
        streamer = fast.DICOMMultiFrameStreamer.create(path, loop=False, grayscale=True, cropToROI=False)
        frames = [np.asarray(f) for f in fast.DataStream(streamer)]
        return frames, {'format': fmt, 'anatomy': anatomy, 'dimensionality': '2D_sequence'}

    raise ValueError(f'Unsupported format for: {path}')
