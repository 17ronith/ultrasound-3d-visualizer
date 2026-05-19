import asyncio
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

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

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

_ort_session: ort.InferenceSession | None = None

def _get_ort_session() -> ort.InferenceSession:
    global _ort_session
    if _ort_session is None:
        model_path = str(_FAST_DATA_ROOT / 'NeuralNetworkModels/jugular_vein_segmentation.onnx')
        _ort_session = ort.InferenceSession(model_path)
    return _ort_session

def _ort_infer_jugular(frame_np: np.ndarray) -> np.ndarray:
    """Cached ORT inference — ~9 ms/frame vs ~1.7 s with FAST SegmentationNetwork."""
    session = _get_ort_session()
    inp_name = session.get_inputs()[0].name
    H, W = frame_np.shape[:2]
    norm = cv2.resize(frame_np.astype(np.float32) / 255.0, (256, 256))
    inp = norm[:, :, np.newaxis][np.newaxis]  # (1, 256, 256, 1)
    out = session.run(None, {inp_name: inp})[0][0]  # (256, 256, 3)
    seg_f32 = (np.argmax(out, axis=-1) > 0).astype(np.float32)  # 1=artery/vein
    return cv2.resize(seg_f32, (W, H), interpolation=cv2.INTER_NEAREST)

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


def _compute_pulse_results(raw_frames, prob_maps, N, duration_s, fps):
    MAX = 200

    # Step 3 — Variance map
    stack = np.stack([f.astype(np.float32) for f in raw_frames], axis=0)
    variance_map = np.var(stack, axis=0)
    v_min, v_max = variance_map.min(), variance_map.max()
    variance_map = (variance_map - v_min) / max(v_max - v_min, 1e-6)
    H, W = variance_map.shape
    if H > MAX or W > MAX:
        scale = MAX / max(H, W)
        nH, nW = max(1, int(H * scale)), max(1, int(W * scale))
        vm_img = PILImage.fromarray((variance_map * 255).clip(0, 255).astype(np.uint8))
        variance_map_ds = np.array(vm_img.resize((nW, nH), PILImage.BILINEAR)).astype(np.float32) / 255.0
    else:
        nH, nW = H, W
        variance_map_ds = variance_map.astype(np.float32)

    # Step 4 — Vessel area waveform
    vessel_areas = [float((pm > 0.5).sum()) for pm in prob_maps]
    va_arr = np.array(vessel_areas, dtype=np.float32)
    va_min, va_max = va_arr.min(), va_arr.max()
    va_norm = (va_arr - va_min) / max(va_max - va_min, 1e-6)

    # Step 5 — Mean probability map
    mean_prob = np.mean(np.stack([pm.astype(np.float32) for pm in prob_maps], axis=0), axis=0)
    if H > MAX or W > MAX:
        mp_img = PILImage.fromarray((mean_prob * 255).clip(0, 255).astype(np.uint8))
        mean_prob_ds = np.array(mp_img.resize((nW, nH), PILImage.BILINEAR)).astype(np.float32) / 255.0
    else:
        mean_prob_ds = mean_prob.astype(np.float32)

    # Step 6 — Peak detection
    peaks, _ = find_peaks(va_norm, height=0.3, distance=15)
    peak_frames = peaks.tolist()
    bpm = round((len(peak_frames) / duration_s) * 60, 1) if len(peak_frames) >= 2 else None

    # Step 7 — M-mode
    centroid_col = int(np.argmax(mean_prob.sum(axis=0)))
    mmode = stack[:, :, centroid_col].astype(np.float32)
    del stack  # free memory after extraction
    mm_min, mm_max = mmode.min(), mmode.max()
    mmode_norm = (mmode - mm_min) / max(mm_max - mm_min, 1e-6)
    if mmode_norm.shape[1] > MAX:
        mm_img = PILImage.fromarray((mmode_norm * 255).astype(np.uint8))
        mm_ds = np.array(mm_img.resize((MAX, N), PILImage.BILINEAR)).astype(np.float32) / 255.0
    else:
        mm_ds = mmode_norm.astype(np.float32)

    # Step 8 — Mask encoding shape for frontend scaling
    H_m, W_m = prob_maps[0].shape
    if H_m > MAX or W_m > MAX:
        scale_m = MAX / max(H_m, W_m)
        mask_enc_h = max(1, int(H_m * scale_m))
        mask_enc_w = max(1, int(W_m * scale_m))
    else:
        mask_enc_h, mask_enc_w = H_m, W_m

    # Step 9 — Vessel component extraction (per-frame connected components)
    vessel_components = []
    for pm in prob_maps:
        binary = pm > 0.5
        labeled = label(binary)
        props = regionprops(labeled)
        components = []
        for rp in props:
            if rp.area > 100:
                min_row, min_col, max_row, max_col = rp.bbox
                components.append({
                    'centroid_x': int(round(rp.centroid[1])),
                    'centroid_y': int(round(rp.centroid[0])),
                    'pixel_count': int(rp.area),
                    'bbox': [min_row, min_col, max_row, max_col],
                })
        components.sort(key=lambda c: c['pixel_count'], reverse=True)
        vessel_components.append(components)

    return {
        'variance_map_ds': variance_map_ds, 'nH': nH, 'nW': nW,
        'va_norm': va_norm, 'mean_prob_ds': mean_prob_ds,
        'peak_frames': peak_frames, 'bpm': bpm,
        'centroid_col': centroid_col, 'mm_ds': mm_ds,
        'H': H, 'W': W,
        'mask_enc_shape': [mask_enc_h, mask_enc_w],
        'vessel_components': vessel_components,
    }


@app.post('/analyze/pulse')
async def analyze_pulse(sequence_dir: str = Form(...)):
    """Stream SSE progress events during 171-frame inference, then emit full payload."""

    async def generate():
        data_dir = Path(sequence_dir)
        if not data_dir.is_dir():
            yield f"data: {json.dumps({'error': 'Directory not found'})}\n\n"
            return

        # Step 1 — Load frame list + timestamps
        mhd_files = sorted(
            [p for p in data_dir.glob('*.mhd') if re.search(r'_(\d+)$', p.stem)],
            key=lambda p: int(re.search(r'_(\d+)$', p.stem).group(1))
        )
        N = len(mhd_files)
        ts_file = data_dir / 'timestamps.fts'
        timestamps = []
        for x in ts_file.read_text().splitlines():
            x = x.strip()
            if x:
                try:
                    timestamps.append(int(x))
                except ValueError:
                    continue
        if len(timestamps) < 2:
            yield f"data: {json.dumps({'error': 'timestamps.fts has fewer than 2 entries'})}\n\n"
            return
        duration_s = (timestamps[-1] - timestamps[0]) / 1000.0
        fps = (N - 1) / duration_s

        # Step 2 — Load + ORT inference on all frames (~9 ms/frame vs ~1.7 s with FAST)
        loop = asyncio.get_running_loop()
        t_infer_start = time.time()
        raw_frames: list[np.ndarray] = []
        prob_maps: list[np.ndarray] = []

        for i, mhd_path in enumerate(mhd_files):
            def _load_and_ort_infer(p=mhd_path):
                frame_arr, _ = load(str(p))
                frame_np = np.asarray(frame_arr).squeeze()
                if frame_np.ndim == 3:
                    frame_np = frame_np.mean(axis=-1)
                frame_np = frame_np.astype(np.uint8)
                pm = _ort_infer_jugular(frame_np)
                return frame_np, pm

            try:
                frame_np, pm = await loop.run_in_executor(None, _load_and_ort_infer)
            except Exception as exc:
                yield f"data: {json.dumps({'error': f'Frame {i} failed: {exc}'})}\n\n"
                return
            raw_frames.append(frame_np)
            prob_maps.append(pm)

            if (i + 1) % 20 == 0 or i == N - 1:
                yield f"data: {json.dumps({'progress': i + 1, 'total': N, 'stage': 'inference'})}\n\n"

        inference_time_ms = int((time.time() - t_infer_start) * 1000)

        # Steps 3–8 — Post-loop computation offloaded to executor
        results = await loop.run_in_executor(
            None, _compute_pulse_results, raw_frames, prob_maps, N, duration_s, fps
        )

        # Done event — metadata only (no frame pixel data)
        variance_map_ds = results['variance_map_ds']
        nH, nW = results['nH'], results['nW']
        va_norm = results['va_norm']
        mean_prob_ds = results['mean_prob_ds']
        peak_frames = results['peak_frames']
        bpm = results['bpm']
        centroid_col = results['centroid_col']
        mm_ds = results['mm_ds']
        H, W = results['H'], results['W']
        mask_enc_shape = results['mask_enc_shape']
        vessel_components = results['vessel_components']

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
            'mmode_shape': list(mm_ds.shape),
            'mean_probability_map': mean_prob_ds.ravel().tolist(),
            'mean_probability_shape': [nH, nW],
            'representative_frame_shape': [H, W],
            'mask_enc_shape': mask_enc_shape,
            'inference_time_ms': inference_time_ms,
            'vessel_components': vessel_components,
        }
        yield f"data: {json.dumps(payload)}\n\n"

        # Stream individual frame events (~120 KB each instead of one 20 MB blob)
        mask_enc_h, mask_enc_w = mask_enc_shape
        H_m, W_m = prob_maps[0].shape
        needs_resize = H_m > 200 or W_m > 200
        for i, (frame_np, pm) in enumerate(zip(raw_frames, prob_maps)):
            _, fb = cv2.imencode('.jpg', frame_np, [cv2.IMWRITE_JPEG_QUALITY, 70])
            pm_u8 = (pm * 255).clip(0, 255).astype(np.uint8)
            if needs_resize:
                pm_u8 = np.array(
                    PILImage.fromarray(pm_u8).resize((mask_enc_w, mask_enc_h), PILImage.BILINEAR)
                )
            _, mb = cv2.imencode('.jpg', pm_u8, [cv2.IMWRITE_JPEG_QUALITY, 60])
            yield f"data: {json.dumps({'frame': i, 'frame_b64': base64.b64encode(fb.tobytes()).decode('ascii'), 'mask_b64': base64.b64encode(mb.tobytes()).decode('ascii')})}\n\n"

        yield f"data: {json.dumps({'frames_done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )
