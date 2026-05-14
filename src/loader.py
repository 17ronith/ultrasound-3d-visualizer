# src/loader.py
from pathlib import Path


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
