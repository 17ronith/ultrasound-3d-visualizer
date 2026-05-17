# Pulse Analysis Mode — Design Spec
**Date:** 2026-05-17
**Status:** Approved

## Overview

Add a Pulse Analysis Mode to UltrasoundViz that processes all 171 frames of the JugularVein cine-loop, extracts vessel pulsation data, detects heartbeat peaks, and renders a temporal variance terrain alongside a synchronized video/M-mode/waveform dashboard.

## Confirmed Data Facts

- **Frames:** 171 (US-2D_0.mhd – US-2D_170.mhd), all shape (318, 492, 1) uint8
- **Duration:** 5.226s, 32.53 fps, 30.74ms avg interval
- **Warm inference:** ~350ms/frame; estimated total: ~60s
- **Variance map:** 2.37× higher variance in center (vessel) vs edges — valid terrain signal
- **Memory:** uint8 stack = 25.5 MB, float32 stack = 102.1 MB — no downsampling needed

## State Machine

Five states: `drop | loading | mode_select | dashboard | pulse`

- Existing `drop → loading → dashboard` path unchanged
- New path: `drop → loading → mode_select → pulse` (via "Analyze Pulse" button)
- `mode_select` is a new intermediate state shown after `/analyze` returns

## Backend Changes

### 1. `POST /analyze` — add `has_sequence` + `sequence_dir`

After saving uploaded file, check:
- File is `.mhd`
- Same directory contains other `*_\d+.mhd` files
- `timestamps.fts` exists in same directory

Add to response:
```json
{ "has_sequence": true, "sequence_dir": "/abs/path/to/JugularVein" }
```

### 2. `POST /analyze/pulse` — streaming SSE endpoint

**Input:** Form field `sequence_dir` (absolute path)

**Pipeline (9 steps):**

| Step | Operation |
|---|---|
| 1 | Load all MHDs sorted by trailing number + parse timestamps.fts |
| 2 | preprocess + run_inference on every frame; emit SSE progress every 10 frames |
| 3 | `np.var(stack, axis=0)` → normalize → downsample ≤200×200 |
| 4 | Per-frame binary mask pixel count → normalize → `vessel_area_waveform[171]` |
| 5 | `np.mean(prob_maps, axis=0)` → downsample ≤200×200 → `mean_probability_map` |
| 6 | Peak detection: local maxima, value >0.3, min separation 15 frames → BPM |
| 7 | Centroid col of mean mask → `stack[:, :, centroid_col]` → `mmode_data (171, ≤200)` |
| 8 | Representative frame = median peak frame or frame 85 |
| 9 | Final SSE event with full JSON payload |

**SSE format:**
```
data: {"progress": 10, "total": 171, "stage": "inference"}\n\n
data: {"progress": 20, "total": 171, "stage": "inference"}\n\n
...
data: {"done": true, "mode": "pulse", "anatomy": "JugularVein", ...full payload}\n\n
```

**Response JSON fields:**
- `mode`, `anatomy`, `frame_count`, `fps`, `duration_seconds`
- `bpm` (null if <2 peaks), `peak_frames[]`, `centroid_col`
- `variance_map` (flattened), `variance_map_shape`
- `vessel_area_waveform[171]`
- `mmode_data` (flattened), `mmode_shape`
- `mean_probability_map` (flattened), `mean_probability_shape`
- `representative_frame` (flattened, full 318×492), `representative_frame_shape`
- `inference_time_ms`

## Frontend Changes

### State 3 — Mode Selection

Full-screen over linen `#f3efe8` background. Two cards side by side, centered.

**Card 1 — Single Frame:**
- Icon: single layer symbol
- Title: "Single Frame"
- Subtitle: "Analyze this frame with AI segmentation and explore as an interactive 3D probability terrain"
- Small canvas thumbnail from `original_image` base64
- Button: "Analyze Frame" (terracotta) — always active
- Action: call `showState('dashboard')` with already-loaded `/analyze` data

**Card 2 — Pulse Analysis:**
- Icon: waveform/pulse symbol
- Title: "Pulse Analysis"
- Subtitle: "Analyze all 171 frames as a cine-loop, extract vessel pulsation waveform, detect heartbeat, and visualize tissue motion as a 3D variance terrain"
- Button: "Analyze Pulse" (terracotta) — active only if `has_sequence=true`
- If `has_sequence=false`: greyed out, button disabled, label "No sequential frames detected"
- Active card hover: terracotta border glow
- Action: show loading state → POST SSE to `/analyze/pulse` → showState('pulse')

**"← Load different file"** link top-left resets to drop state.

### SSE Loading Progress

During `/analyze/pulse` fetch, loading state shows:
- Message: "Segmenting frame N / 171... X%"
- Progress bar fills proportionally
- Messages cycle: "Loading 171 frames..." → "Running AI segmentation (~60 seconds)..." → "Computing pulsation variance..." → "Detecting pulse peaks..." → "Rendering..."

### State 5 — Pulse Mode Layout

```
┌─────────────────────────────────────────────┐
│  TOP BAR: badges + "← Back" link           │
├──────────────────┬──────────────────────────┤
│  LEFT PANEL      │  RIGHT PANEL             │
│  [Video canvas]  │  [Three.js variance      │
│  [M-mode canvas] │   terrain]               │
├──────────────────┴──────────────────────────┤
│  WAVEFORM BAR (100px, full width)           │
├─────────────────────────────────────────────┤
│  CONTROL BAR                                │
├─────────────────────────────────────────────┤
│  STATS BAR (4 cards)                        │
└─────────────────────────────────────────────┘
```

**Top bar:** Anatomy badge "JugularVein", format badge "171 frames · 32.5 fps", "← Back" returns to mode_select.

**Left panel — split vertically:**
- Top: Video canvas. Current frame greyscale. Mean prob mask overlay (cyan). Vertical cyan line at `centroid_col`. Frame counter "Frame N / 171" bottom-right.
- Bottom: M-mode canvas. `mmode_data` rendered as greyscale image. Orange vertical cursor at current frame. Terracotta triangles at peak frames on top edge. Labels: X="Time (5.23s)", Y="Depth".

**Right panel:** Reuse `renderTerrain()` verbatim with `variance_map` as Z input. Colorbar bottom="Static tissue", top="High motion". Default preset: VASCULAR. Reconstruction + hardening animations run as-is.

**Waveform bar (full-width canvas, 100px):**
- Terracotta line chart of `vessel_area_waveform`
- Cyan vertical lines at `peak_frames`, orange playback cursor
- BPM top-right in terracotta, "N peaks detected" adjacent
- Clickable: click jumps playback to that frame
- X ticks every 30 frames with seconds label

**Control bar:**
- Row 1: Play/pause, scrubber (0–170), speed (0.25x/0.5x/1x/2x/4x), loop toggle
- Row 2: Opacity slider, transfer function presets (RAW/SOFT TISSUE/VASCULAR/BONE)

**Stats bar (4 cards):**
- Anatomy: "JugularVein" / "Temporal cine-loop"
- Heart Rate: "~N BPM" / "estimated from vessel area" (or "Insufficient peaks")
- Peak Count: "N peaks" / "detected over 5.23s"
- Mean Coverage: avg vessel area % / "mean detected area"

**Sync:** Single `_pulseFrame` index drives all four views (video, M-mode cursor, waveform cursor, scrubber) simultaneously.

## Unchanged

- State 1 (drop zone): no changes
- State 2 (loading): only loading messages updated for pulse flow
- State 4 (dashboard): completely unchanged — all existing features work as before

## Testing

Playwright tests:
1. Upload single MHD — mode_select shows, Single Frame card active, Pulse Analysis card active (has_sequence=true for JugularVein)
2. Click "Analyze Frame" — dashboard renders correctly with existing terrain
3. Click "Analyze Pulse" — loading bar updates with SSE progress; after ~60s pulse mode renders
4. Pulse mode sync: drag scrubber → all four views update
5. Waveform click → frame jumps correctly
6. "← Back" → returns to mode_select with both cards intact
7. "← Load different file" from mode_select → drop zone resets
