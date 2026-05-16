import os
import io
import base64
import shutil
import tempfile
from typing import List

import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import fast
fast.Config.setTestDataPath('/Users/ronith/Documents/Projects/ultrasound/FAST/data/')

from src.loader import load
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


@app.post('/analyze')
async def analyze(files: List[UploadFile] = File(...)):
    # Save all uploaded files to the same temp directory so that .mhd can
    # find its companion .raw file (both must live in the same folder).
    tmp_dir = tempfile.mkdtemp()
    mhd_path = None
    first_path = None

    try:
        for f in files:
            fname = os.path.basename(f.filename or 'upload')
            if not fname or fname == '.':
                fname = 'upload'
            fpath = os.path.join(tmp_dir, fname)
            with open(fpath, 'wb') as out:
                out.write(await f.read())
            if first_path is None:
                first_path = fpath
            if fname.lower().endswith('.mhd'):
                mhd_path = fpath

        tmp_path = mhd_path or first_path

        data, meta = load(tmp_path)
        preprocessed = preprocess(data, meta)

        if isinstance(preprocessed, list):
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
            'anatomy': meta['anatomy'],
            'dimensionality': meta['dimensionality'],
            'format': meta['format'],
            'detected': conf['detected'] == 'True',
            'coverage': conf['coverage'],
            'mean_confidence': conf['mean_confidence'],
            'probability_map': terrain_z.tolist(),
            'binary_mask': mask_ds.tolist(),
            'original_image': orig_b64,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
