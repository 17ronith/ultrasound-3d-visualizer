import os
import io
import re
import base64
import shutil
import tempfile
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from skimage.filters import threshold_otsu

import fast
fast.Config.setTestDataPath('/Users/ronith/Documents/Projects/ultrasound/FAST/data/')

from src.loader import load, detect_sequence
from src.preprocessor import preprocess
from src.inference import run_inference


app = FastAPI(title='UltrasoundViz API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/', response_class=HTMLResponse)
async def index():
    with open('index.html', encoding='utf-8') as f:
        return f.read()


_OVERRIDE_TO_ROUTE = {
    'Jugular Vein': 'JugularVein',
    'Lung / CT':    'LIDC',
}

_FAST_DATA_ROOT = Path(fast.Config.getTestDataPath())

def _find_sequence_dir(filename: str, uploaded_path: str = '') -> tuple[bool, str]:
    """Search FAST data for a numbered MHD file and return its directory if it is
    part of a sequence (sibling numbered MHDs + timestamps.fts present).

    When multiple directories contain the same filename (e.g. US-2D_0.mhd), the
    uploaded file's content is compared byte-for-byte against each candidate to
    identify the correct source directory.
    """
    if not filename.lower().endswith('.mhd'):
        return False, ''

    uploaded_bytes: bytes | None = None
    if uploaded_path:
        try:
            uploaded_bytes = Path(uploaded_path).read_bytes()
        except OSError:
            pass

    for candidate in _FAST_DATA_ROOT.rglob(filename):
        d = candidate.parent
        if not (d / 'timestamps.fts').exists():
            continue
        siblings = [p for p in d.glob('*.mhd') if re.search(r'_(\d+)$', p.stem)]
        if len(siblings) <= 1:
            continue
        # If we have the uploaded bytes, confirm this is the right candidate
        if uploaded_bytes is not None:
            try:
                if candidate.read_bytes() != uploaded_bytes:
                    continue
            except OSError:
                continue
        return True, str(d)
    return False, ''

@app.post('/analyze')
async def analyze(files: List[UploadFile] = File(...), anatomy_override: Optional[str] = Form(None)):
    # Save all uploaded files to the same temp directory so that .mhd can
    # find its companion .raw file (both must live in the same folder).
    tmp_dir = tempfile.mkdtemp()
    mhd_path = None
    first_path = None
    saved_paths = []

    try:
        for f in files:
            fname = os.path.basename(f.filename or 'upload')
            if not fname or fname == '.':
                fname = 'upload'
            fpath = os.path.join(tmp_dir, fname)
            with open(fpath, 'wb') as out:
                out.write(await f.read())
            saved_paths.append(fpath)
            if first_path is None:
                first_path = fpath
            if fname.lower().endswith('.mhd'):
                mhd_path = fpath

        tmp_path = mhd_path or first_path

        # Sequence detection for mode_select UI
        _uploaded_mhd = mhd_path or first_path or ''
        _uploaded_fname = os.path.basename(_uploaded_mhd)
        _has_seq, _seq_dir = _find_sequence_dir(_uploaded_fname, _uploaded_mhd)

        # Companion extensions that pair with .mhd — not independent frames
        _companion = {'.mhd', '.raw', '.zraw'}
        data_paths = [p for p in saved_paths if Path(p).suffix.lower() not in _companion]

        seq_result = None
        if len(data_paths) > 1 and mhd_path is None:
            seq_result = detect_sequence(data_paths)
            if seq_result['is_sequential']:
                frames, ref_meta = [], None
                for p in seq_result['ordered_paths']:
                    frame_data, frame_meta = load(p)
                    if ref_meta is None:
                        ref_meta = frame_meta
                    frames.append(np.array(frame_data))
                data = frames
                meta = ref_meta
                meta['dimensionality'] = '2D_sequence'
            else:
                data, meta = load(tmp_path)
        else:
            data, meta = load(tmp_path)

        if anatomy_override and anatomy_override.strip():
            route_key = _OVERRIDE_TO_ROUTE.get(anatomy_override, anatomy_override)
            meta['anatomy'] = route_key
            display_anatomy = anatomy_override
        else:
            display_anatomy = meta['anatomy']
        preprocessed = preprocess(data, meta)

        if isinstance(preprocessed, list):
            mid_idx = len(preprocessed) // 2
            if meta['format'] == 'dicom':
                prob_map, bin_mask, conf = run_inference(preprocessed[mid_idx], meta)
                orig = data[mid_idx] if isinstance(data, list) else data
            else:
                results = [run_inference(frame, meta) for frame in preprocessed]
                prob_map, bin_mask, conf = results[0]
                orig = data[0] if isinstance(data, list) else data
        else:
            prob_map, bin_mask, conf = run_inference(preprocessed, meta)
            orig = data

        # Use middle depth slice for 3D volumes
        orig_arr = np.array(orig)
        if orig_arr.ndim == 4:
            orig_arr = orig_arr[orig_arr.shape[0] // 2]

        # Grayscale 2D
        if orig_arr.ndim == 3 and orig_arr.shape[-1] == 3:
            orig_gray = orig_arr.mean(axis=-1).astype(np.uint8)
        elif orig_arr.ndim == 3:
            orig_gray = orig_arr.squeeze(-1).astype(np.uint8)
        else:
            orig_gray = orig_arr.astype(np.uint8)

        # DICOM: Otsu foreground/background split + 85th-percentile structure detection
        if meta['format'] == 'dicom':
            otsu_val = threshold_otsu(orig_gray)
            foreground = orig_gray > otsu_val
            fg_pixels = orig_gray[foreground]
            p85 = float(np.percentile(fg_pixels, 85)) if len(fg_pixels) > 0 else 255.0
            dicom_prob = np.zeros(orig_gray.shape, dtype=np.float32)
            low_fg = foreground & (orig_gray <= p85)
            high_fg = foreground & (orig_gray > p85)
            dicom_prob[low_fg] = 0.1 + 0.2 * (orig_gray[low_fg].astype(np.float32) / p85)
            high_range = max(255.0 - p85, 1.0)
            dicom_prob[high_fg] = 0.7 + 0.3 * (
                (orig_gray[high_fg].astype(np.float32) - p85) / high_range
            )
            bin_mask = (dicom_prob > 0.5)[..., np.newaxis].astype(np.uint8)

        # Blended terrain Z (same formula as visualizer.py Mode 1)
        raw_norm = orig_gray.astype(np.float32) / 255.0
        mask_2d = bin_mask.squeeze().astype(np.float32)
        blurred = gaussian_filter(mask_2d, sigma=15)
        if blurred.max() > 0:
            blurred /= blurred.max()
        terrain_z = (0.35 * raw_norm + 0.65 * blurred).astype(np.float32)

        # Downsample to max 200×200
        H, W = terrain_z.shape
        MAX = 200
        if H > MAX or W > MAX:
            scale = MAX / max(H, W)
            new_H = max(1, int(H * scale))
            new_W = max(1, int(W * scale))

            def _resize_f32(arr, nH, nW):
                pil = PILImage.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
                return np.array(pil.resize((nW, nH), PILImage.BILINEAR)).astype(np.float32) / 255.0

            terrain_z = _resize_f32(terrain_z, new_H, new_W)
            mask_ds = (np.array(
                PILImage.fromarray((mask_2d * 255).astype(np.uint8)).resize((new_W, new_H), PILImage.NEAREST)
            ) > 127).astype(np.uint8)
            orig_gray = np.array(PILImage.fromarray(orig_gray).resize((new_W, new_H), PILImage.BILINEAR))
        else:
            mask_ds = mask_2d.astype(np.uint8)

        # Encode original as base64 PNG
        buf = io.BytesIO()
        PILImage.fromarray(orig_gray).save(buf, format='PNG')
        orig_b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            'anatomy': display_anatomy,
            'dimensionality': meta['dimensionality'],
            'format': meta['format'],
            'detected': conf['detected'] == 'True',
            'coverage': conf['coverage'],
            'mean_confidence': conf['mean_confidence'],
            'probability_map': terrain_z.tolist(),
            'binary_mask': mask_ds.tolist(),
            'original_image': orig_b64,
            'sequence_detected': seq_result['is_sequential'] if seq_result else False,
            'sequence_method': seq_result['method'] if seq_result else None,
            'frame_count': len(seq_result['ordered_paths']) if seq_result and seq_result['is_sequential'] else 1,
            'has_sequence': _has_seq,
            'sequence_dir': _seq_dir,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post('/analyze/pulse')
async def analyze_pulse(sequence_dir: str = Form(...)):
    """Stream SSE progress events during 171-frame inference, then emit full payload."""
    import asyncio

    async def generate():
        data_dir = Path(sequence_dir)
        if not data_dir.is_dir():
            yield f"data: {json.dumps({'error': 'Directory not found'})}\n\n"
            return

        # Step 1 — Load frame list + timestamps
        mhd_files = sorted(
            data_dir.glob('*.mhd'),
            key=lambda p: int(re.search(r'_(\d+)$', p.stem).group(1))
        )
        N = len(mhd_files)
        ts_file = data_dir / 'timestamps.fts'
        timestamps = [int(x.strip()) for x in ts_file.read_text().splitlines() if x.strip()]
        duration_s = (timestamps[-1] - timestamps[0]) / 1000.0
        fps = (N - 1) / duration_s

        # Step 2 — Inference on all frames (blocking; run each in thread to not block event loop)
        loop = asyncio.get_event_loop()
        t_infer_start = time.time()
        raw_frames: list[np.ndarray] = []
        prob_maps: list[np.ndarray] = []

        for i, mhd_path in enumerate(mhd_files):
            def _infer_frame(p=mhd_path):
                frame_arr, frame_meta = load(str(p))
                frame_meta['anatomy'] = 'JugularVein'
                frame_np = np.asarray(frame_arr).squeeze()          # (318, 492)
                prep = preprocess(frame_np, frame_meta)
                pm, bm, _ = run_inference(prep, frame_meta)
                return frame_np.astype(np.uint8), np.asarray(pm).squeeze()

            frame_np, pm = await loop.run_in_executor(None, _infer_frame)
            raw_frames.append(frame_np)
            prob_maps.append(pm)

            if (i + 1) % 10 == 0 or i == N - 1:
                yield f"data: {json.dumps({'progress': i + 1, 'total': N, 'stage': 'inference'})}\n\n"

        inference_time_ms = int((time.time() - t_infer_start) * 1000)

        # Step 3 — Temporal variance map (318, 492) → normalize → downsample ≤200×200
        stack = np.stack([f.astype(np.float32) for f in raw_frames], axis=0)  # (N,318,492)
        variance_map = np.var(stack, axis=0)                                   # (318, 492)
        v_min, v_max = variance_map.min(), variance_map.max()
        variance_map = (variance_map - v_min) / max(v_max - v_min, 1e-6)
        H, W = variance_map.shape
        MAX = 200
        if H > MAX or W > MAX:
            scale = MAX / max(H, W)
            nH, nW = max(1, int(H * scale)), max(1, int(W * scale))
            vm_img = PILImage.fromarray((variance_map * 255).clip(0, 255).astype(np.uint8))
            variance_map_ds = np.array(vm_img.resize((nW, nH), PILImage.BILINEAR)).astype(np.float32) / 255.0
        else:
            nH, nW = H, W
            variance_map_ds = variance_map.astype(np.float32)

        # Step 4 — Vessel area waveform (one value per frame, normalized 0-1)
        vessel_areas = [float((pm > 0.5).sum()) for pm in prob_maps]
        va_arr = np.array(vessel_areas, dtype=np.float32)
        va_min, va_max = va_arr.min(), va_arr.max()
        va_norm = (va_arr - va_min) / max(va_max - va_min, 1e-6)

        # Step 5 — Mean probability map → downsample ≤200×200
        mean_prob = np.mean(np.stack([pm.astype(np.float32) for pm in prob_maps], axis=0), axis=0)
        if H > MAX or W > MAX:
            mp_img = PILImage.fromarray((mean_prob * 255).clip(0, 255).astype(np.uint8))
            mean_prob_ds = np.array(mp_img.resize((nW, nH), PILImage.BILINEAR)).astype(np.float32) / 255.0
        else:
            mean_prob_ds = mean_prob.astype(np.float32)

        # Step 6 — Peak detection + BPM
        peaks, _ = find_peaks(va_norm, height=0.3, distance=15)
        peak_frames = peaks.tolist()
        bpm = round((len(peak_frames) / duration_s) * 60, 1) if len(peak_frames) >= 2 else None

        # Step 7 — M-mode: column at horizontal centroid of mean mask, shape (N, H)
        centroid_col = int(np.argmax(mean_prob.sum(axis=0)))
        mmode = stack[:, :, centroid_col].astype(np.float32)          # (N, 318)
        mm_min, mm_max = mmode.min(), mmode.max()
        mmode_norm = (mmode - mm_min) / max(mm_max - mm_min, 1e-6)
        if mmode_norm.shape[1] > 200:
            mm_img = PILImage.fromarray((mmode_norm * 255).astype(np.uint8))
            mm_ds = np.array(mm_img.resize((200, N), PILImage.BILINEAR)).astype(np.float32) / 255.0
        else:
            mm_ds = mmode_norm.astype(np.float32)

        # Step 8 — Representative frame (raw uint8, full 318×492)
        rep_idx = peak_frames[len(peak_frames) // 2] if peak_frames else min(85, N - 1)
        rep_frame = raw_frames[rep_idx]                                # (318, 492) uint8

        # Step 9 — Final SSE event with full payload
        payload = {
            'done': True,
            'mode': 'pulse',
            'anatomy': 'JugularVein',
            'frame_count': N,
            'fps': round(fps, 2),
            'duration_seconds': round(duration_s, 3),
            'bpm': bpm,
            'peak_frames': peak_frames,
            'centroid_col': centroid_col,
            'variance_map': variance_map_ds.ravel().tolist(),
            'variance_map_shape': [nH, nW],
            'vessel_area_waveform': va_norm.tolist(),
            'mmode_data': mm_ds.ravel().tolist(),
            'mmode_shape': list(mm_ds.shape),            # [N, D]
            'mean_probability_map': mean_prob_ds.ravel().tolist(),
            'mean_probability_shape': [nH, nW],
            'representative_frame': rep_frame.ravel().tolist(),
            'representative_frame_shape': [H, W],
            'inference_time_ms': inference_time_ms,
        }
        yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )
