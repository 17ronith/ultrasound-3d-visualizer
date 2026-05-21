# UltrasoundViz

**Medical imaging 3D visualizer.** Drop any ultrasound, CT, or MRI file — get anatomy-aware AI segmentation and an interactive 3D visualization in the browser. No configuration required.

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Three.js](https://img.shields.io/badge/Three.js-0.165-black?logo=threedotjs)](https://threejs.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16+-gray?logo=onnx)](https://onnxruntime.ai)
[![License](https://img.shields.io/badge/Data_License-ODC_Attribution-lightgray)](FAST/data/LICENSE.md)

---

## Overview

UltrasoundViz is a single-page web application backed by a FastAPI server. Upload any supported medical image and the pipeline automatically:

1. Detects the file format and anatomy from the file path
2. Denoises and normalizes the image with Non-Local Means filtering
3. Routes to the appropriate ONNX segmentation model (or pixel-intensity fallback)
4. Renders an interactive 3D visualization in the browser via Three.js

There is also a CLI mode that produces a self-contained Plotly HTML report without a server.

---

## Dashboards

### Single Frame — 3D Terrain

Renders any single image (JPG, PNG, MHD, DICOM) as a height-map terrain. Pixel intensity and the segmentation mask are blended into Z-values, producing smooth hills over anatomy and flat plains over background. A boundary ring marks the segmented region.

- Hot colormap vertex colors: black → red → yellow → white
- Terrain blend: `Z = 0.35 × grayscale + 0.65 × gaussian_blurred_mask (σ=15)`
- Rotate, zoom, and hover in real time

### Pulse Mode — Temporal Tube Visualization

Upload an MHD sequence or drop a folder of frames to see vessel pulsation animated as a 3D tube over time. Each frame contributes a ring to the tube; tube radius tracks vessel area per frame.

| Anatomy | Vessel color | Notes |
|---|---|---|
| JugularVein | Cyan | Both jugular (cyan) + carotid (orange) tubes rendered |
| CarotidArtery | Orange | Single tube; radius clamped to [0.6×, 1.4×] mean to prevent bulging |

Controls: frame scrubber, playback speed (0.25× – 4×), transfer-function presets (RAW / SOFT TISSUE / VASCULAR / BONE), opacity slider.

A waveform panel below the 3D view plots vessel area over time with detected peak frames and BPM.

### CT Nodule 3D

Upload a DICOM lung CT to enter the nodule mode. The server runs a 3D sliding-window ONNX model (64³ voxel patches) and streams progress via Server-Sent Events. The dashboard shows:

- **3D panel** — Three.js volume render of the nodule with a movable slice plane
- **Gallery** — axial slice thumbnails at every detected nodule layer
- **Stats** — volume (mm³), diameter, voxel spacing, bounding box per component
- **Waveform** — slice-by-slice segmentation area profile

---

## Generate Report

Every dashboard has a **Generate Report** button that opens a self-contained clinical-style HTML report in a new browser tab. Reports are generated entirely in the client — no server request is made. Each report includes canvas captures of the 3D view, a findings section, clinical interpretation, and a Print / Save as PDF button.

---

## Supported Formats

| Extension | Format | Handled by |
|---|---|---|
| `.jpg` `.jpeg` `.png` `.bmp` | Raster image | FAST `ImageFileImporter` |
| `.mhd` | MetaImage (2D or 3D volume) | FAST `ImageFileImporter` |
| `.mhd` (sequence `_#`) | MHD frame sequence | FAST `ImageFileStreamer` |
| `.dcm` | DICOM single slice | pydicom + CT lung windowing |
| `.dcm` (series) | DICOM multi-frame series | pydicom ordered by `InstanceNumber` |
| `.avi` `.mp4` | Video | FAST `MovieStreamer` |

Multiple files dropped at once are tested for sequential ordering via DICOM `SeriesInstanceUID`, trailing filename numbers, or pixel-dimension/intensity similarity.

---

## Anatomy Detection & AI Models

Anatomy is inferred from keywords in the file path. No manual selection is required.

| Keyword in path | Anatomy label | Segmentation model |
|---|---|---|
| `JugularVein` | JugularVein | `jugular_vein_segmentation.onnx` (labels: 1=Artery, 2=Vein) |
| `CarotidArtery` | CarotidArtery | `jugular_vein_segmentation.onnx` (reused; artery component) |
| `LIDC` or DICOM lung | LIDC | `lung_nodule_segmentation.onnx` (3D sliding-window) |
| `Heart` `FemoralArtery` `Axillary` `Ball` | respective label | Pixel-intensity fallback |
| anything else | unknown | Pixel-intensity fallback |

Both ONNX models are bundled with the FAST test-data set under the ODC Attribution License.

---

## Installation

**Prerequisites**

- Python 3.11 or newer (3.13 recommended)
- [FAST framework](https://github.com/smistad/FAST) — provides the `fast` Python package (`pyfast`)

**Setup**

```bash
git clone https://github.com/17ronith/ultrasound-3d-visualizer.git
cd ultrasound-3d-visualizer

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**FAST data path**

The ONNX models and test datasets ship with FAST. Set the path once in each script you run:

```python
import fast
fast.Config.setTestDataPath('/path/to/FAST/data/')
```

The default path expected by this repo is `./FAST/data/` relative to the project root.

---

## Usage

### Web App (recommended)

```bash
bash start.sh          # activates venv and starts uvicorn on http://localhost:8000
```

Then open [http://localhost:8000](http://localhost:8000) in a browser, drop any supported file onto the landing page, and wait for the dashboard to load.

**Manual start:**

```bash
source venv/bin/activate
uvicorn server:app --port 8000 --reload
```

### CLI — Plotly HTML Report

```bash
source venv/bin/activate

# Default: JugularVein MHD sequence
python main.py

# Custom file
ULTRASOUND_FILEPATH=/path/to/file.mhd python main.py

# Suppress automatic browser open
ULTRASOUND_NO_BROWSER=1 python main.py
```

Terminal output:

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
  Coverage         : 12.34%
  Mean confidence  : 74.10%

[4/4] Rendering: Mode 1 (2D heightmap)
  Saved to: ultrasoundviz_output.html
  Opened in browser.
=======================================================
```

---

## API Reference

The FastAPI server exposes three endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves `index.html` |
| `POST` | `/analyze` | Single-image or sequence analysis — returns JSON with terrain/mask data |
| `POST` | `/analyze/pulse` | Pulse-mode analysis for a sequence directory — returns JSON with waveform + vessel tracking |
| `POST` | `/analyze/nodule` | CT nodule analysis — streams SSE progress events, returns JSON with 3D volume data |

### `POST /analyze` — Response shape

```json
{
  "anatomy": "JugularVein",
  "dimensionality": "2D_single",
  "format": "mhd",
  "detected": "True",
  "coverage": "12.34%",
  "mean_confidence": "74.10%",
  "probability_map": [[...]],
  "binary_mask": [[...]],
  "original_image": "data:image/png;base64,..."
}
```

### `POST /analyze/pulse` — Response shape (excerpt)

```json
{
  "anatomy": "JugularVein",
  "frame_count": 171,
  "fps": 25,
  "duration_seconds": 6.84,
  "bpm": 68.4,
  "peak_frames": [12, 37, 62, ...],
  "vessel_area_waveform": [1200, 1350, ...],
  "vessel_components": [[{"centroid_x": 120, "centroid_y": 95, "radius": 18.3, ...}], ...]
}
```

### `POST /analyze/nodule` — SSE progress stream

```
data: {"status": "progress", "pct": 12, "msg": "Processing patch 3/24..."}
data: {"status": "progress", "pct": 48, "msg": "Processing patch 12/24..."}
data: {"status": "done", "result": {...}}
```

---

## Pipeline Architecture

```
File(s)
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  Stage 1 — Loader  (src/loader.py)                   │
│  detect_format → detect_anatomy → detect_dimensionality│
│  Routes: image / mhd / dicom / video / mhd-sequence  │
│  Returns: numpy array (or list of arrays) + metadata  │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  Stage 2 — Preprocessor  (src/preprocessor.py)       │
│  Grayscale conversion → NLM denoise → normalize [0,1]│
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  Stage 3 — Inference  (src/inference.py)             │
│  JugularVein / CarotidArtery  →  ONNX segmentation   │
│  LIDC (CT)                    →  ONNX 3D sliding-win │
│  Everything else              →  pixel-intensity proxy│
│  Returns: probability_map, binary_mask, confidence   │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  Stage 4 — Visualizer  (src/visualizer.py / frontend)│
│  Mode 1: 2D_single  →  Three.js terrain heightmap    │
│  Mode 2: 2D_sequence →  3D tube + waveform           │
│  Mode 3: 3D_volume  →  Plotly isosurface (CLI only)  │
└──────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ultrasound-3d-visualizer/
├── src/
│   ├── loader.py          # Stage 1: format/anatomy/dimensionality detection + load()
│   ├── preprocessor.py    # Stage 2: grayscale, NLM denoise, normalize
│   ├── inference.py       # Stage 3: ONNX routing + pixel-intensity fallback
│   └── visualizer.py      # Stage 4: Plotly visualization (used by CLI)
├── tests/
│   ├── conftest.py        # Shared fixtures
│   ├── test_loader.py     # 20 tests: format/anatomy/dimensionality + FAST loading
│   ├── test_preprocessor.py  # 5 tests
│   ├── test_inference.py  # 7 tests (includes ONNX integration)
│   ├── test_visualizer.py # 7 tests (Mode 1/2/3)
│   ├── test_pulse.py      # 7 tests: vessel tracking, BPM, waveform
│   └── test_main.py       # 3 end-to-end integration tests
├── FAST/
│   └── data/              # FAST test dataset (ODC Attribution License)
│       ├── US/            # Ultrasound MHD sequences (JugularVein, Heart, Ball…)
│       ├── CT/            # CT volumes (DICOM, MHD)
│       ├── MRI/           # MRI volumes
│       └── NeuralNetworkModels/  # Bundled ONNX models
├── server.py              # FastAPI backend (all three analysis endpoints)
├── index.html             # Single-file SPA — drop zone, loading, three dashboards
├── main.py                # CLI entry point
├── start.sh               # One-command web app launcher
├── requirements.txt       # Pinned Python dependencies
└── pytest.ini             # pythonpath = . (required for src imports)
```

---

## Testing

```bash
source venv/bin/activate
pytest tests/                    # run all 49 tests
pytest tests/test_loader.py -v   # single module, verbose
```

The test suite covers all five pipeline stages. Integration tests in `test_main.py` run the full pipeline end-to-end on real FAST test data. No internet connection required.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.13, FastAPI, Uvicorn |
| Medical imaging | FAST framework (pyfast 4.17), pydicom |
| AI inference | ONNX Runtime 1.16, two bundled ONNX models |
| Image processing | NumPy, OpenCV, scikit-image, SciPy, Pillow |
| 3D visualization (web) | Three.js 0.165 (importmap, no bundler) |
| 3D visualization (CLI) | Plotly |
| Typography | Lora (serif headings), DM Sans, DM Mono |
| Testing | pytest 9 |

---

## Data License

Test datasets are from the [FAST framework](https://github.com/smistad/FAST) and are licensed under the **ODC Attribution License**. See [`FAST/data/LICENSE.md`](FAST/data/LICENSE.md) for details.
