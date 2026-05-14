# UltrasoundViz — Project Context

## Overview
Medical imaging 3D visualization tool built on the FAST framework. Accepts any medical imaging file, runs anatomy-aware AI segmentation, and renders an interactive 3D visualization in the browser via Plotly.

**GitHub repo**: `ultrasound 3d visualizer` (ronithmenneni)

## Environment
- **Python**: 3.13.3
- **venv**: `venv/` in project root — always activate before running scripts
- **Installed packages**: `pyfast 4.17.1`, `numpy 2.4.4`
- **Activate**: `source venv/bin/activate`

## FAST Data Path Configuration
FAST's default test data path is `/Users/ronith/FAST/data/` which does NOT exist here.
Always set the path at the top of every script:

```python
import fast
fast.Config.setTestDataPath('/Users/ronith/Documents/Projects/ultrasound/FAST/data/')
DATA = fast.Config.getTestDataPath()
```

The shared `_config.py` in `tutorial/` does this for tutorial scripts.

## Dataset: FAST Test Data
Location: `FAST/data/`
License: ODC Attribution License (see `FAST/data/LICENSE.md`)

### Ultrasound (`FAST/data/US/`)
| File/Folder | Format | Description |
|---|---|---|
| `US-2D.jpg/.png/.bmp` | Image | Single 2D US frame (512×512) |
| `US-2D-compressed.mhd` | MHD | Compressed 2D MHD |
| `sagittal_spine.avi` | Video | Sagittal spine scan |
| `Heart/` | MHD sequences | 5 cardiac views (ApicalFourChamber, ApicalLongAxis, ApicalTwoChamber, ParasternalLongAxis, ParasternalShortAxis) |
| `JugularVein/Left/`, `/Right/` | MHD sequences | Jugular vein + carotid artery (492×318px, use `#` as frame index) |
| `CarotidArtery/Left/`, `/Right/` | MHD sequences | Carotid artery |
| `FemoralArtery/Left/`, `/Right/` | MHD sequences | Femoral artery |
| `Axillary/` | MHD sequences | Axillary region |
| `Ball/US-3Dt_#.mhd` | 3D MHD (84 volumes) | **True 3D** US phantom — 276×249×200 voxels |
| `UFF/P4_2_PLAX.uff`, `P4_2_A4C.uff` | UFF | Ultrafast Ultrasound Format cardiac |

### CT (`FAST/data/CT/`)
| File | Format | Description |
|---|---|---|
| `CT-Abdomen.mhd` + `.raw` | MHD | Abdominal CT volume |
| `CT-Thorax.mhd` + `.raw` | MHD | Thoracic CT volume |
| `LIDC-IDRI-0072/*.dcm` | DICOM | Lung cancer CT scan (multi-slice) |

### MRI (`FAST/data/MRI/`)
- `MR-Abdomen.mhd` + `.zraw`

### Neural Network Models (`FAST/data/NeuralNetworkModels/`)
| Model | Anatomy | Input shape |
|---|---|---|
| `jugular_vein_segmentation.onnx` | JugularVein | 1×1×256×256 FP32 — labels: 1=Artery, 2=Vein |
| `lung_nodule_segmentation.onnx` | CT/LIDC lung | — |

---

## UltrasoundViz Pipeline (5 Stages)

### Stage 1 — Format-Agnostic Loader
- **Input**: any file path or directory path
- **Format routing** (by extension):
  - `.png/.jpg/.bmp` → `fast.ImageFileImporter`
  - `.mhd` → `fast.ImageFileImporter`
  - `.dcm` (or directory of .dcm) → `fast.DICOMMultiFrameStreamer`
  - `.avi/.mp4` → `fast.MovieStreamer`
- **Single file** → numpy array; **directory/sequence** → list of numpy arrays
- **Returns**: `(array_or_list, metadata_dict)`
- **metadata_dict keys**:
  - `anatomy`: label derived from folder path (e.g. `"JugularVein"`, `"LIDC"`, `"Ball"`, `"unknown"`)
  - `dimensionality`: `"2D_single"` | `"2D_sequence"` | `"3D_volume"`
  - `format`: detected file format string

### Stage 2 — Preprocessing
- Convert to grayscale if needed
- Apply `fast.NonLocalMeans` denoising (filterSize=3, searchSize=11, smoothingAmount=0.2)
- Normalize to 0–1 float32
- Handles both single arrays and lists of arrays

### Stage 3 — AI Inference with Routing
Routing by `metadata["anatomy"]`:
- `"JugularVein"` → run `jugular_vein_segmentation.onnx` via FAST `SegmentationNetwork`
- `"LIDC"` → run `lung_nodule_segmentation.onnx` via FAST `SegmentationNetwork`
- Everything else → pixel intensity as proxy probability (no model)

**Both paths produce**:
- `probability_map`: per-pixel float32 0–1
- `binary_mask`: thresholded at 0.5

**Confidence dict** (from mask statistics):
```python
{
  "detected": "X%",      # bool — any pixel above threshold
  "coverage": "Y%",      # mask pixels / total pixels * 100
  "mean_confidence": "Z%" # mean probability inside mask region
}
```

### Stage 4 — 3D Visualization (3 Rendering Modes)
Mode selected automatically from `metadata["dimensionality"]`:

**Mode 1 — 2D single image** (`"2D_single"`): *Build first*
- Plotly `Surface`: Z = probability map (healthy tissue flat/dark, structure rises as peaks)
- `Scatter3d` contour ring from `skimage.measure.find_contours` at level 0.5, at peak height
- Ring color: blue=JugularVein, red=lung nodule, white=unknown

**Mode 3 — True 3D volume** (`"3D_volume"`): *Build second*
- Load Ball MHD as full 3D numpy array
- Plotly `Isosurface` at appropriate threshold to show internal structure

**Mode 2 — 2D sequence** (`"2D_sequence"`): *Build third*
- Stack each slice's probability map along Z axis
- Plotly `Volume` or series of `Surface` traces
- Slider to scrub through slices

**All modes share**:
- Dark background
- Axis labels: X=Width, Y=Height, Z=Depth/Probability
- Title: anatomy label + confidence dict values
- Full rotate/zoom/hover
- **Side-by-side panel**: left = original 2D image with mask outline (matplotlib), right = 3D Plotly figure

### Stage 5 — Entry Point (`main.py`)
- User sets one variable: `FILEPATH` (file or directory path)
- Runs all 4 stages sequentially
- Terminal summary printout:
  ```
  Format detected : MHD sequence
  Anatomy         : JugularVein
  Model used      : jugular_vein_segmentation.onnx
  Detection       : detected=True, coverage=12.3%, mean_confidence=74.1%
  Rendering mode  : Mode 1 (2D single heightmap)
  ```

---

## Build Order & Constraints
1. Build and test each stage independently before connecting
2. **Priority**: Mode 1 → Mode 3 → Mode 2
3. No BUSI dataset. No U-Net ResNet34.
4. Only FAST bundled ONNX models for segmentation; pixel intensity fallback for everything else
5. FAST GUI windows do not render in VSCode terminal — all output goes through Plotly (browser) and matplotlib (saved PNG or displayed)

---

## Key FAST API Patterns
```python
# Single image load
fast.ImageFileImporter.create(path).runAndGetOutputData()

# MHD sequence stream (# = frame index placeholder)
fast.ImageFileStreamer.create(path + 'US-2D_#.mhd', framerate=20, loop=True)

# DICOM
fast.DICOMMultiFrameStreamer.create(dcm_path, loop=True, grayscale=False, cropToROI=True)

# Video
fast.MovieStreamer.create(avi_path, grayscale=True)

# NLM denoising
fast.NonLocalMeans.create(filterSize=3, searchSize=11, smoothingAmount=0.2, inputMultiplicationWeight=0.5)

# Segmentation network
fast.SegmentationNetwork.create(model_path, scaleFactor=1./255.).connect(streamer)

# Convert to numpy
np.asarray(fast_image)   # shape: (H, W, C)

# DataStream iteration
for frame in fast.DataStream(streamer): ...
for img, seg in fast.DataStream(streamer, segNetwork): ...
```

---

## Tutorial Scripts (`tutorial/`)
29 scripts following the FAST Python Ultrasound Tutorial — used for framework exploration.
FAST GUI windows do not render in the VSCode terminal environment (exit code 0 but no window shown).
Headless-runnable scripts: ex01, ex04, ex06, ex13, ex14, ex22, ex24 (matplotlib mode).
