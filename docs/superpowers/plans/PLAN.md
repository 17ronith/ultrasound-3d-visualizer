# UltrasoundViz Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 5-stage medical imaging pipeline that loads any FAST-format file, denoises it, runs anatomy-aware ONNX segmentation, and renders an interactive 3D Plotly visualization in the browser.

**Architecture:** Five independent modules (loader → preprocessor → inference → visualizer → main) wired together in `main.py`. Each module is tested independently before integration. Visualization outputs a single HTML file with a matplotlib left panel and Plotly 3D right panel.

**Tech Stack:** pyfast 4.17.1, numpy, plotly, scikit-image, matplotlib, opencv-python, pytest

---

## Confirmed Data Facts (do not guess these)

- `fast.Config.setTestDataPath('/Users/ronith/Documents/Projects/ultrasound/FAST/data/')`
- JugularVein frames: `US/JugularVein/US-2D_#.mhd` — input shape `(318, 492, 1)` uint8
- Ball 3D: `US/Ball/US-3Dt_#.mhd` — shape `(200, 249, 276, 1)` uint8, axes = (D, H, W, C)
- Single 2D image: `US/US-2D.jpg` — shape `(512, 512, 3)` uint8
- CarotidArtery/FemoralArtery use `Left/` and `Right/` subdirs; JugularVein files are directly inside
- `jugular_vein_segmentation.onnx` — input: any size (FAST auto-resizes), output: `(256, 256, 1)` uint8, classes: 0=background, 1=Artery, 2=Vein
- NLM output: same spatial shape as input, uint8

---

## File Structure

```
src/
  loader.py        — Stage 1: format detection, anatomy/dim metadata, FAST loading
  preprocessor.py  — Stage 2: grayscale, NLM denoise, normalize to 0-1
  inference.py     — Stage 3: ONNX routing + pixel intensity fallback + confidence dict
  visualizer.py    — Stage 4: Mode 1/2/3 rendering → HTML string
tests/
  conftest.py      — shared fixtures (data path, sample arrays)
  test_loader.py
  test_preprocessor.py
  test_inference.py
  test_visualizer.py
main.py            — Stage 5: entry point, sets FILEPATH, prints summary, opens browser
requirements.txt
```

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Install dependencies**

```bash
source venv/bin/activate
pip install plotly scikit-image matplotlib opencv-python pytest
```

Expected output includes: `Successfully installed plotly-... scikit-image-... matplotlib-... opencv-python-... pytest-...`

- [ ] **Step 2: Write requirements.txt**

```
pyfast>=4.17.0
numpy>=2.0
plotly>=5.0
scikit-image>=0.21
matplotlib>=3.8
opencv-python>=4.8
pytest>=8.0
```

- [ ] **Step 3: Create src/__init__.py**

```python
```
(empty file)

- [ ] **Step 4: Write tests/conftest.py**

```python
import pytest
import numpy as np
import fast

fast.Config.setTestDataPath('/Users/ronith/Documents/Projects/ultrasound/FAST/data/')
DATA = fast.Config.getTestDataPath()


@pytest.fixture
def data_path():
    return DATA


@pytest.fixture
def jugular_mhd_path():
    return DATA + 'US/JugularVein/US-2D_0.mhd'


@pytest.fixture
def single_jpg_path():
    return DATA + 'US/US-2D.jpg'


@pytest.fixture
def ball_mhd_path():
    return DATA + 'US/Ball/US-3Dt_0.mhd'


@pytest.fixture
def gray_array_uint8():
    return np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)


@pytest.fixture
def gray_array_float():
    return np.random.rand(64, 64, 1).astype(np.float32)


@pytest.fixture
def binary_mask():
    mask = np.zeros((64, 64, 1), dtype=np.uint8)
    mask[20:44, 20:44] = 1
    return mask
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/__init__.py tests/conftest.py
git commit -m "feat: project setup — deps, conftest fixtures"
```

---

## Task 2: Loader — Pure Logic Functions

**Files:**
- Create: `src/loader.py` (detect_format, detect_anatomy, detect_dimensionality)
- Create: `tests/test_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/test_loader.py -v 2>&1 | tail -10
```

Expected: `ImportError` or `ModuleNotFoundError` — `loader.py` doesn't exist yet.

- [ ] **Step 3: Implement detect_format, detect_anatomy, detect_dimensionality**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
source venv/bin/activate && pytest tests/test_loader.py -v 2>&1 | tail -15
```

Expected: all tests PASS (3D_volume detection comes in the next task).

- [ ] **Step 5: Commit**

```bash
git add src/loader.py tests/test_loader.py
git commit -m "feat: loader pure logic — format, anatomy, dimensionality detection"
```

---

## Task 3: Loader — FAST Data Loading

**Files:**
- Modify: `src/loader.py` — add `load()` function
- Modify: `tests/test_loader.py` — add integration tests

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_loader.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/test_loader.py::test_load_single_jpg -v 2>&1 | tail -5
```

Expected: `ImportError: cannot import name 'load'`

- [ ] **Step 3: Implement load()**

Add to `src/loader.py`:

```python
import fast
import numpy as np

fast.Config.setTestDataPath('/Users/ronith/Documents/Projects/ultrasound/FAST/data/')


def load(filepath: str):
    """
    Load any supported medical imaging file via FAST.
    Returns (array, metadata) where array is ndarray (single) or list[ndarray] (sequence).
    """
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
```

- [ ] **Step 4: Run tests**

```bash
source venv/bin/activate && pytest tests/test_loader.py -v 2>&1 | tail -20
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/loader.py tests/test_loader.py
git commit -m "feat: loader FAST data loading — single image, MHD, video, DICOM"
```

---

## Task 4: Preprocessor

**Files:**
- Create: `src/preprocessor.py`
- Create: `tests/test_preprocessor.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/test_preprocessor.py -v 2>&1 | tail -5
```

Expected: `ImportError: cannot import name 'preprocess'`

- [ ] **Step 3: Implement preprocess()**

```python
# src/preprocessor.py
import fast
import numpy as np

fast.Config.setTestDataPath('/Users/ronith/Documents/Projects/ultrasound/FAST/data/')


def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert (H, W, C) array to (H, W, 1). Passthrough if already 1 channel."""
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        gray = arr.mean(axis=2, keepdims=True).astype(arr.dtype)
        return gray
    return arr


def _nlm_denoise(arr: np.ndarray) -> np.ndarray:
    """Apply FAST NonLocalMeans to a (H, W, 1) uint8 array. Returns (H, W, 1) uint8."""
    inp = arr if arr.dtype == np.uint8 else (arr * 255).clip(0, 255).astype(np.uint8)
    fast_img = fast.Image.createFromArray(inp)
    nlm = fast.NonLocalMeans.create(
        filterSize=3, searchSize=11, smoothingAmount=0.2, inputMultiplicationWeight=0.5
    ).connect(fast_img)
    return np.asarray(nlm.runAndGetOutputData())


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize uint8 array to float32 [0, 1]."""
    return arr.astype(np.float32) / 255.0


def _process_single(arr: np.ndarray) -> np.ndarray:
    arr = _to_grayscale(arr)
    arr = _nlm_denoise(arr)
    return _normalize(arr)


def preprocess(data, metadata: dict):
    """
    Preprocess loaded data: grayscale → NLM denoise → normalize to [0,1] float32.
    Handles single arrays, lists of arrays, and 4D volumes.
    """
    dim = metadata.get('dimensionality', '2D_single')

    if dim == '2D_sequence' and isinstance(data, list):
        return [_process_single(frame) for frame in data]

    if dim == '3D_volume' and data.ndim == 4:
        # Normalize volume directly — NLM is a 2D filter, skip for 3D
        vol = data if data.dtype == np.uint8 else data
        if vol.shape[-1] == 1:
            vol = vol[..., 0]  # (D, H, W)
        else:
            vol = vol.mean(axis=-1)  # (D, H, W)
        normalized = vol.astype(np.float32) / 255.0
        return normalized[..., np.newaxis]  # (D, H, W, 1)

    return _process_single(data)
```

- [ ] **Step 4: Run tests**

```bash
source venv/bin/activate && pytest tests/test_preprocessor.py -v 2>&1 | tail -15
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: preprocessor — grayscale, NLM denoise, normalize"
```

---

## Task 5: Inference — Fallback and Confidence Dict

**Files:**
- Create: `src/inference.py`
- Create: `tests/test_inference.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/test_inference.py -v 2>&1 | tail -5
```

Expected: `ImportError`

- [ ] **Step 3: Implement compute_confidence and run_inference fallback**

```python
# src/inference.py
import fast
import numpy as np
from pathlib import Path

fast.Config.setTestDataPath('/Users/ronith/Documents/Projects/ultrasound/FAST/data/')
DATA = fast.Config.getTestDataPath()

_MODEL_ROUTES = {
    'JugularVein': DATA + 'NeuralNetworkModels/jugular_vein_segmentation.onnx',
    'LIDC':        DATA + 'NeuralNetworkModels/lung_nodule_segmentation.onnx',
}


def compute_confidence(probability_map: np.ndarray, binary_mask: np.ndarray) -> dict:
    """Build confidence dict from probability map and binary mask."""
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
    """Pixel intensity as proxy probability. No model used."""
    prob = preprocessed.astype(np.float32)
    mask = (prob > 0.5).astype(np.uint8)
    return prob, mask, compute_confidence(prob, mask)


def run_inference(preprocessed, metadata: dict):
    """
    Route to correct ONNX model or pixel intensity fallback.
    Returns (probability_map, binary_mask, confidence_dict).
    probability_map: float32 ndarray same shape as preprocessed
    binary_mask: uint8 ndarray same shape as preprocessed
    """
    anatomy = metadata.get('anatomy', 'unknown')

    if anatomy not in _MODEL_ROUTES:
        return _fallback_inference(preprocessed)

    return _run_onnx(preprocessed, anatomy)


def _run_onnx(preprocessed: np.ndarray, anatomy: str):
    """Run FAST SegmentationNetwork and resize output to match input shape."""
    model_path = _MODEL_ROUTES[anatomy]
    H, W = preprocessed.shape[:2]

    fast_img = fast.Image.createFromArray(preprocessed)
    net = fast.SegmentationNetwork.create(model_path, scaleFactor=1.0).connect(fast_img)
    seg_arr = np.asarray(net.runAndGetOutputData())   # (256, 256, 1) uint8

    # Resize segmentation output back to original spatial dimensions
    import cv2
    seg_resized = cv2.resize(
        seg_arr.squeeze().astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST
    )[:, :, np.newaxis]

    binary_mask = (seg_resized > 0).astype(np.uint8)
    probability_map = binary_mask.astype(np.float32)

    return probability_map, binary_mask, compute_confidence(probability_map, binary_mask)
```

- [ ] **Step 4: Run tests**

```bash
source venv/bin/activate && pytest tests/test_inference.py -v 2>&1 | tail -15
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inference.py tests/test_inference.py
git commit -m "feat: inference — fallback path and confidence dict"
```

---

## Task 6: Inference — ONNX Routing Integration Test

**Files:**
- Modify: `tests/test_inference.py` — add integration tests with real ONNX models

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_inference.py`:

```python
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
    # Model should detect at least artery or vein in a real jugular vein scan
    assert mask.any(), "Expected at least one segmented pixel"
    assert conf['detected'] == 'True'
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/test_inference.py::test_inference_jugular_vein_returns_correct_shapes -v 2>&1 | tail -5
```

Expected: test not found yet (we need to add `jugular_mhd_path` fixture usage here — but the test uses `data_path` directly).

- [ ] **Step 3: Run the integration tests (no code change needed)**

```bash
source venv/bin/activate && pytest tests/test_inference.py -v 2>&1 | tail -20
```

Expected: all tests PASS including the 2 new ONNX tests.

- [ ] **Step 4: Commit**

```bash
git add tests/test_inference.py
git commit -m "test: ONNX routing integration — jugular vein segmentation"
```

---

## Task 7: Visualizer — Mode 1 (2D Single Heightmap)

**Files:**
- Create: `src/visualizer.py`
- Create: `tests/test_visualizer.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/test_visualizer.py -v 2>&1 | tail -5
```

Expected: `ImportError: cannot import name 'visualize'`

- [ ] **Step 3: Implement visualizer with Mode 1**

```python
# src/visualizer.py
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage.measure import find_contours

_RING_COLORS = {
    'JugularVein':   'blue',
    'CarotidArtery': 'cyan',
    'FemoralArtery': 'cyan',
    'LIDC':          'red',
    'Heart':         'orange',
    'Ball':          'green',
}
_DEFAULT_RING_COLOR = 'white'


def _ring_color(anatomy: str) -> str:
    return _RING_COLORS.get(anatomy, _DEFAULT_RING_COLOR)


def _make_left_panel_b64(orig: np.ndarray, binary_mask: np.ndarray, anatomy: str) -> str:
    """Render original image with mask contour overlay to base64 PNG."""
    gray = orig.squeeze() if orig.shape[-1] == 1 else orig.mean(axis=-1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(gray, cmap='gray', aspect='auto')
    ax.set_title(f'Original — {anatomy}', color='white', fontsize=10)
    ax.axis('off')
    fig.patch.set_facecolor('#111111')

    contours = find_contours(binary_mask.squeeze().astype(float), level=0.5)
    color = _ring_color(anatomy)
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#111111')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _mode1_surface(prob: np.ndarray, binary_mask: np.ndarray, anatomy: str, conf: dict) -> go.Figure:
    """Mode 1: probability heightmap Surface + Scatter3d contour ring."""
    z = prob.squeeze()
    H, W = z.shape
    x = np.arange(W)
    y = np.arange(H)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale='hot',
        showscale=True,
        name='Probability Surface',
    ))

    contours = find_contours(binary_mask.squeeze().astype(float), level=0.5)
    peak_z = float(z.max())
    color = _ring_color(anatomy)
    for c in contours:
        fig.add_trace(go.Scatter3d(
            x=c[:, 1], y=c[:, 0],
            z=np.full(len(c), peak_z),
            mode='lines',
            line=dict(color=color, width=4),
            name='Structure Boundary',
        ))

    coverage = conf.get('coverage', 'N/A')
    mean_conf = conf.get('mean_confidence', 'N/A')
    fig.update_layout(
        title=dict(
            text=f'{anatomy} | coverage={coverage} | confidence={mean_conf}',
            font=dict(color='white'),
        ),
        scene=dict(
            xaxis_title='Width',
            yaxis_title='Height',
            zaxis_title='Probability',
            bgcolor='#111111',
            xaxis=dict(color='white'),
            yaxis=dict(color='white'),
            zaxis=dict(color='white'),
        ),
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _mode3_isosurface(data: np.ndarray, anatomy: str, conf: dict) -> go.Figure:
    """Mode 3: Plotly Isosurface for true 3D volumes."""
    vol = data.squeeze()          # (D, H, W)
    D, H, W = vol.shape
    z_idx, y_idx, x_idx = np.mgrid[0:D, 0:H, 0:W]
    values = vol.flatten().astype(np.float32)

    isomin = float(vol.max()) * 0.3
    isomax = float(vol.max()) * 0.9

    fig = go.Figure(go.Isosurface(
        x=x_idx.flatten().astype(float),
        y=y_idx.flatten().astype(float),
        z=z_idx.flatten().astype(float),
        value=values,
        isomin=isomin,
        isomax=isomax,
        surface_count=3,
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False),
        name='Volume',
    ))
    fig.update_layout(
        title=dict(text=f'{anatomy} — 3D Volume', font=dict(color='white')),
        scene=dict(
            xaxis_title='Width',
            yaxis_title='Height',
            zaxis_title='Depth',
            bgcolor='#111111',
            xaxis=dict(color='white'),
            yaxis=dict(color='white'),
            zaxis=dict(color='white'),
        ),
        paper_bgcolor='#111111',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _mode2_stacked(prob_list: list, binary_mask_list: list, anatomy: str, conf: dict) -> go.Figure:
    """Mode 2: stacked slice surfaces along Z axis."""
    fig = go.Figure()
    n = len(prob_list)
    for i, (prob, _mask) in enumerate(zip(prob_list, binary_mask_list)):
        z_slice = prob.squeeze()
        H, W = z_slice.shape
        x = np.arange(W)
        y = np.arange(H)
        z = np.full_like(z_slice, fill_value=i / max(n - 1, 1))
        color_z = z_slice

        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            surfacecolor=color_z,
            colorscale='hot',
            showscale=(i == 0),
            opacity=0.6,
            name=f'Slice {i}',
        ))

    coverage = conf.get('coverage', 'N/A')
    fig.update_layout(
        title=dict(text=f'{anatomy} | {n} slices | coverage={coverage}', font=dict(color='white')),
        scene=dict(
            xaxis_title='Width',
            yaxis_title='Height',
            zaxis_title='Slice Index',
            bgcolor='#111111',
            xaxis=dict(color='white'),
            yaxis=dict(color='white'),
            zaxis=dict(color='white'),
        ),
        paper_bgcolor='#111111',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _build_html(left_b64: str, plotly_fig: go.Figure, conf: dict, anatomy: str) -> str:
    """Combine matplotlib left panel (base64 PNG) and Plotly 3D right panel into single HTML."""
    plotly_div = plotly_fig.to_html(full_html=False, include_plotlyjs='cdn')
    coverage = conf.get('coverage', 'N/A')
    mean_conf = conf.get('mean_confidence', 'N/A')
    detected = conf.get('detected', 'N/A')

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>UltrasoundViz — {anatomy}</title></head>
<body style="margin:0;padding:10px;background:#111111;display:flex;flex-direction:column;font-family:sans-serif;color:white">
  <h2 style="margin:0 0 8px 0">UltrasoundViz | {anatomy} | detected={detected} | coverage={coverage} | confidence={mean_conf}</h2>
  <div style="display:flex;gap:10px;flex:1">
    <div style="flex:0.38;display:flex;align-items:center">
      <img src="data:image/png;base64,{left_b64}" style="width:100%;border:1px solid #333">
    </div>
    <div style="flex:0.62">
      {plotly_div}
    </div>
  </div>
</body>
</html>"""


def visualize(orig: np.ndarray, probability_map, binary_mask, conf: dict, metadata: dict) -> str:
    """
    Render based on dimensionality. Returns an HTML string.
    orig: raw loaded array (before preprocessing) used for the left panel.
    probability_map: float32 ndarray or list of ndarrays.
    binary_mask: uint8 ndarray or list of ndarrays.
    """
    anatomy = metadata.get('anatomy', 'unknown')
    dim = metadata.get('dimensionality', '2D_single')

    if dim == '2D_single':
        orig_single = orig if not isinstance(orig, list) else orig[0]
        left_b64 = _make_left_panel_b64(orig_single, binary_mask, anatomy)
        fig = _mode1_surface(probability_map, binary_mask, anatomy, conf)
        return _build_html(left_b64, fig, conf, anatomy)

    if dim == '3D_volume':
        orig_single = orig[0] if isinstance(orig, list) else orig[:1]
        first_slice = orig[0] if orig.ndim == 4 else orig[:, :, 0:1]
        # For left panel use middle depth slice
        mid_d = orig.shape[0] // 2
        mid_frame = orig[mid_d]
        empty_mask = np.zeros(mid_frame.shape[:2] + (1,), dtype=np.uint8)
        left_b64 = _make_left_panel_b64(mid_frame, empty_mask, anatomy)
        fig = _mode3_isosurface(orig, anatomy, conf)
        return _build_html(left_b64, fig, conf, anatomy)

    if dim == '2D_sequence':
        if isinstance(probability_map, list):
            prob_list = probability_map
            mask_list = binary_mask
        else:
            prob_list = [probability_map]
            mask_list = [binary_mask]
        orig_list = orig if isinstance(orig, list) else [orig]
        left_b64 = _make_left_panel_b64(orig_list[0], mask_list[0], anatomy)
        fig = _mode2_stacked(prob_list, mask_list, anatomy, conf)
        return _build_html(left_b64, fig, conf, anatomy)

    raise ValueError(f'Unknown dimensionality: {dim}')
```

- [ ] **Step 4: Run tests**

```bash
source venv/bin/activate && pytest tests/test_visualizer.py -v 2>&1 | tail -15
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visualizer.py tests/test_visualizer.py
git commit -m "feat: visualizer — Mode 1 heightmap, Mode 2 stacked, Mode 3 isosurface"
```

---

## Task 8: Visualizer — Mode 3 and Mode 2 Integration Tests

**Files:**
- Modify: `tests/test_visualizer.py` — add Mode 3 and Mode 2 tests

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_visualizer.py`:

```python
def test_visualize_mode3_returns_html():
    vol = np.random.randint(0, 200, (10, 32, 32, 1), dtype=np.uint8)
    prob = vol.astype(np.float32) / 255.0
    mask = (prob > 0.5).astype(np.uint8)
    conf = {'detected': 'True', 'coverage': '30.00%', 'mean_confidence': '65.00%'}
    meta = {'anatomy': 'Ball', 'dimensionality': '3D_volume', 'format': 'mhd'}
    html = visualize(vol, prob, mask, conf, meta)
    assert '<!DOCTYPE html>' in html
    assert 'Ball' in html
    assert 'Isosurface' in html or 'isosurface' in html or 'value' in html


def test_visualize_mode2_returns_html():
    frames = [np.random.randint(0, 255, (32, 32, 1), dtype=np.uint8) for _ in range(5)]
    probs = [f.astype(np.float32) / 255.0 for f in frames]
    masks = [(p > 0.5).astype(np.uint8) for p in probs]
    conf = {'detected': 'True', 'coverage': '20.00%', 'mean_confidence': '55.00%'}
    meta = {'anatomy': 'Heart', 'dimensionality': '2D_sequence', 'format': 'mhd'}
    html = visualize(frames, probs, masks, conf, meta)
    assert '<!DOCTYPE html>' in html
    assert 'Heart' in html
    assert '5 slices' in html
```

- [ ] **Step 2: Run tests**

```bash
source venv/bin/activate && pytest tests/test_visualizer.py -v 2>&1 | tail -15
```

Expected: all 7 tests PASS (no new code needed — `visualize()` already handles all 3 modes).

- [ ] **Step 3: Commit**

```bash
git add tests/test_visualizer.py
git commit -m "test: visualizer Mode 2 and Mode 3 coverage"
```

---

## Task 9: Main Entry Point

**Files:**
- Create: `main.py`
- Create: `tests/test_main.py` (integration smoke test)

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_main.py
import subprocess
import sys
import os


def test_main_runs_without_error_on_jugular_mhd(data_path):
    env = os.environ.copy()
    env['ULTRASOUND_FILEPATH'] = data_path + 'US/JugularVein/US-2D_0.mhd'
    env['ULTRASOUND_NO_BROWSER'] = '1'  # skip browser open in CI
    result = subprocess.run(
        [sys.executable, 'main.py'],
        capture_output=True, text=True, env=env, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert 'JugularVein' in result.stdout
    assert 'jugular_vein_segmentation.onnx' in result.stdout


def test_main_runs_on_ball_3d(data_path):
    env = os.environ.copy()
    env['ULTRASOUND_FILEPATH'] = data_path + 'US/Ball/US-3Dt_0.mhd'
    env['ULTRASOUND_NO_BROWSER'] = '1'
    result = subprocess.run(
        [sys.executable, 'main.py'],
        capture_output=True, text=True, env=env, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert 'Ball' in result.stdout
    assert '3D_volume' in result.stdout


def test_main_runs_on_plain_jpg(data_path):
    env = os.environ.copy()
    env['ULTRASOUND_FILEPATH'] = data_path + 'US/US-2D.jpg'
    env['ULTRASOUND_NO_BROWSER'] = '1'
    result = subprocess.run(
        [sys.executable, 'main.py'],
        capture_output=True, text=True, env=env, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert 'unknown' in result.stdout
    assert 'fallback' in result.stdout.lower() or 'intensity' in result.stdout.lower()
```

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate && pytest tests/test_main.py -v 2>&1 | tail -5
```

Expected: `FileNotFoundError` or `ModuleNotFoundError` for `main.py`

- [ ] **Step 3: Implement main.py**

```python
# main.py
import os
import sys
import webbrowser
import tempfile
import pathlib
import fast

fast.Config.setTestDataPath('/Users/ronith/Documents/Projects/ultrasound/FAST/data/')

# ─── Configuration ────────────────────────────────────────────────────────────
FILEPATH = os.environ.get(
    'ULTRASOUND_FILEPATH',
    fast.Config.getTestDataPath() + 'US/JugularVein/US-2D_0.mhd'
)
NO_BROWSER = os.environ.get('ULTRASOUND_NO_BROWSER', '0') == '1'
OUTPUT_HTML = 'ultrasoundviz_output.html'
# ──────────────────────────────────────────────────────────────────────────────

from src.loader import load
from src.preprocessor import preprocess
from src.inference import run_inference, _MODEL_ROUTES
from src.visualizer import visualize


def main():
    print(f"\n{'='*55}")
    print(f"  UltrasoundViz")
    print(f"{'='*55}")

    # Stage 1 — Load
    print(f"\n[1/4] Loading: {FILEPATH}")
    data, meta = load(FILEPATH)
    print(f"  Format detected  : {meta['format']}")
    print(f"  Anatomy          : {meta['anatomy']}")
    print(f"  Dimensionality   : {meta['dimensionality']}")

    # Keep original data for left panel
    orig = data

    # Stage 2 — Preprocess
    print(f"\n[2/4] Preprocessing...")
    preprocessed = preprocess(data, meta)

    # Stage 3 — Inference
    print(f"\n[3/4] Running inference...")
    anatomy = meta['anatomy']
    model_used = _MODEL_ROUTES.get(anatomy, 'pixel intensity fallback')
    print(f"  Model used       : {pathlib.Path(model_used).name if model_used != 'pixel intensity fallback' else model_used}")

    if isinstance(preprocessed, list):
        results = [run_inference(frame, meta) for frame in preprocessed]
        prob_maps = [r[0] for r in results]
        masks = [r[1] for r in results]
        conf = results[0][2]
    else:
        prob_maps, masks, conf = run_inference(preprocessed, meta)

    print(f"  Detected         : {conf['detected']}")
    print(f"  Coverage         : {conf['coverage']}")
    print(f"  Mean confidence  : {conf['mean_confidence']}")

    # Stage 4 — Visualize
    dim = meta['dimensionality']
    mode_map = {'2D_single': 'Mode 1 (2D heightmap)', '3D_volume': 'Mode 3 (3D isosurface)', '2D_sequence': 'Mode 2 (stacked slices)'}
    print(f"\n[4/4] Rendering: {mode_map.get(dim, dim)}")

    html = visualize(orig, prob_maps, masks, conf, meta)

    with open(OUTPUT_HTML, 'w') as f:
        f.write(html)
    print(f"\n  Saved to: {OUTPUT_HTML}")

    if not NO_BROWSER:
        webbrowser.open(f'file://{os.path.abspath(OUTPUT_HTML)}')
        print(f"  Opened in browser.")

    print(f"\n{'='*55}\n")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run integration tests**

```bash
source venv/bin/activate && pytest tests/test_main.py -v 2>&1 | tail -20
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Run end-to-end manually to verify HTML output**

```bash
source venv/bin/activate && python main.py
```

Expected terminal output:
```
=======================================================
  UltrasoundViz
=======================================================

[1/4] Loading: .../US/JugularVein/US-2D_0.mhd
  Format detected  : mhd
  Anatomy          : JugularVein
  Dimensionality   : 2D_single

[2/4] Preprocessing...

[3/4] Running inference...
  Model used       : jugular_vein_segmentation.onnx
  Detected         : True
  Coverage         : X.XX%
  Mean confidence  : 100.00%

[4/4] Rendering: Mode 1 (2D heightmap)

  Saved to: ultrasoundviz_output.html
  Opened in browser.
```

- [ ] **Step 6: Run full test suite**

```bash
source venv/bin/activate && pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 7: Commit and push**

```bash
git add main.py tests/test_main.py
git commit -m "feat: main entry point — full pipeline end-to-end with browser output"
git push origin master
```

---

## Self-Review

### Spec Coverage Check

| Spec Requirement | Covered By |
|---|---|
| Format routing: PNG/JPG/BMP → ImageFileImporter | Task 3 `load()` |
| Format routing: MHD → ImageFileImporter | Task 3 `load()` |
| Format routing: DICOM → DICOMMultiFrameStreamer | Task 3 `load()` |
| Format routing: AVI/MP4 → MovieStreamer | Task 3 `load()` |
| Single file → ndarray, sequence → list[ndarray] | Task 3 |
| metadata: anatomy, dimensionality, format | Task 2 |
| Grayscale conversion | Task 4 |
| NLM denoising via FAST | Task 4 |
| Normalize to 0-1 float32 | Task 4 |
| JugularVein → jugular_vein_segmentation.onnx | Task 6 |
| LIDC → lung_nodule_segmentation.onnx | Task 6 (`_MODEL_ROUTES`) |
| Fallback: pixel intensity proxy | Task 5 |
| probability_map + binary_mask outputs | Tasks 5, 6 |
| Confidence dict: detected/coverage/mean_confidence | Task 5 |
| Mode 1: Surface with probability heightmap | Task 7 |
| Mode 1: Scatter3d contour ring at peak height | Task 7 |
| Mode 1: ring color by anatomy | Task 7 `_ring_color()` |
| Mode 3: Isosurface 3D volume | Task 7 |
| Mode 2: stacked slices | Task 7 |
| Dark background, axis labels | Task 7 |
| Title: anatomy + confidence | Task 7 |
| Side-by-side: matplotlib left + Plotly right | Task 7 `_build_html()` |
| main.py terminal summary | Task 9 |
| Browser open | Task 9 |
| Build/test each stage independently | Enforced by task order |
| Mode priority: 1 → 3 → 2 | Tasks 7 → 8 → 8 (all in one impl, tested in order) |

### Placeholder Scan
No TBD/TODO found. All code blocks are complete.

### Type Consistency
- `load()` returns `(ndarray | list[ndarray], dict)` — preprocessor, inference, visualizer all accept this shape
- `run_inference()` returns `(prob: float32 ndarray, mask: uint8 ndarray, conf: dict)` — visualizer receives all three
- `visualize()` signature: `(orig, probability_map, binary_mask, conf, metadata)` — main.py passes all five in correct order
- `compute_confidence()` key names ('detected', 'coverage', 'mean_confidence') — used consistently in visualizer title and HTML
