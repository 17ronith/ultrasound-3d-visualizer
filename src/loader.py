# src/loader.py
import re
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


def detect_sequence(filepaths: list) -> dict:
    """
    Check whether a list of files are sequential frames of the same scan.

    Returns:
        {'is_sequential': bool, 'ordered_paths': list[str], 'method': str}

    Signals checked in priority order:
      1. DICOM: all share the same SeriesInstanceUID  →  definitive
      2. Filename trailing-number sequence (step ≤ 2)  →  strong
      3. Same pixel dimensions + adjacent mean-intensity diff < 30/255  →  probabilistic
    """
    if len(filepaths) < 2:
        return {'is_sequential': False, 'ordered_paths': list(filepaths), 'method': 'single_file'}

    paths = [str(p) for p in filepaths]
    formats = [detect_format(p) for p in paths]

    # --- Signal 1: DICOM SeriesInstanceUID ---
    if all(f == 'dicom' for f in formats):
        try:
            import pydicom
            series_uids, instance_nums = [], []
            for p in paths:
                ds = pydicom.dcmread(p, stop_before_pixels=True)
                series_uids.append(str(getattr(ds, 'SeriesInstanceUID', '')))
                instance_nums.append(int(getattr(ds, 'InstanceNumber', 0)))
            if len(set(series_uids)) == 1 and series_uids[0]:
                ordered = [p for _, p in sorted(zip(instance_nums, paths))]
                return {'is_sequential': True, 'ordered_paths': ordered, 'method': 'dicom_series_uid'}
        except Exception:
            pass

    # --- Signal 2: Filename trailing-number sequence ---
    def _trailing_num(path):
        m = re.search(r'(\d+)$', Path(path).stem)
        return int(m.group(1)) if m else None

    nums = [_trailing_num(p) for p in paths]
    if all(n is not None for n in nums):
        pairs = sorted(zip(nums, paths))
        sorted_nums = [n for n, _ in pairs]
        sorted_paths = [p for _, p in pairs]
        gaps = [sorted_nums[i + 1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)]
        if all(1 <= g <= 2 for g in gaps):
            return {'is_sequential': True, 'ordered_paths': sorted_paths, 'method': 'filename_numbering'}

    # --- Signal 3: Same dimensions + similar mean intensity (images only) ---
    if all(f == 'image' for f in formats):
        try:
            from PIL import Image as _PIL
            sizes, means = [], []
            for p in paths:
                img = _PIL.open(p).convert('L')
                sizes.append(img.size)
                means.append(float(np.array(img, dtype=np.float32).mean()))
            if len(set(sizes)) == 1:
                name_order = sorted(range(len(paths)), key=lambda i: Path(paths[i]).name)
                ordered = [paths[i] for i in name_order]
                ordered_means = [means[i] for i in name_order]
                diffs = [abs(ordered_means[i + 1] - ordered_means[i]) for i in range(len(ordered_means) - 1)]
                if np.mean(diffs) < 30:
                    return {'is_sequential': True, 'ordered_paths': ordered, 'method': 'dimension_intensity'}
        except Exception:
            pass

    return {'is_sequential': False, 'ordered_paths': paths, 'method': 'no_pattern'}


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
        import pydicom
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array  # (rows, cols) or (frames, rows, cols)
        if arr.ndim == 2:
            frames = [arr]
        else:
            frames = [arr[i] for i in range(arr.shape[0])]
        return frames, {'format': fmt, 'anatomy': anatomy, 'dimensionality': '2D_sequence'}

    raise ValueError(f'Unsupported format for: {path}')
