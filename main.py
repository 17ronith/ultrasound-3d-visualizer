# main.py
import os
import sys
import webbrowser
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

    orig = data

    # Stage 2 — Preprocess
    print(f"\n[2/4] Preprocessing...")
    preprocessed = preprocess(data, meta)

    # Stage 3 — Inference
    print(f"\n[3/4] Running inference...")
    anatomy = meta['anatomy']
    model_used = _MODEL_ROUTES.get(anatomy, 'pixel intensity fallback')
    model_name = pathlib.Path(model_used).name if model_used != 'pixel intensity fallback' else model_used
    print(f"  Model used       : {model_name}")

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
    mode_map = {
        '2D_single': 'Mode 1 (2D heightmap)',
        '3D_volume': 'Mode 3 (3D isosurface)',
        '2D_sequence': 'Mode 2 (stacked slices)'
    }
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
