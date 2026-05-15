# src/visualizer.py
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage.measure import find_contours
from scipy.ndimage import gaussian_filter

_RING_COLORS = {
    'JugularVein':   'blue',
    'CarotidArtery': 'cyan',
    'FemoralArtery': 'cyan',
    'LIDC':          'red',
    'Heart':         'orange',
    'Ball':          'green',
}
_DEFAULT_RING_COLOR = 'white'

_BADGE_COLORS = {
    'JugularVein':   '#3b82f6',
    'CarotidArtery': '#06b6d4',
    'FemoralArtery': '#06b6d4',
    'LIDC':          '#ef4444',
    'Heart':         '#f97316',
    'Ball':          '#10b981',
}
_DEFAULT_BADGE_COLOR = '#8b949e'


def _badge_color(anatomy: str) -> str:
    return _BADGE_COLORS.get(anatomy, _DEFAULT_BADGE_COLOR)


def _pct_color(pct: float) -> str:
    if pct >= 75:
        return '#10b981'
    elif pct >= 50:
        return '#f97316'
    return '#ef4444'


def _parse_pct(s: str) -> float:
    try:
        return float(s.rstrip('%'))
    except (ValueError, AttributeError):
        return 0.0


def _ring_color(anatomy: str) -> str:
    return _RING_COLORS.get(anatomy, _DEFAULT_RING_COLOR)


def _make_left_panel_b64(orig: np.ndarray, binary_mask: np.ndarray, anatomy: str) -> str:
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


def _mode1_surface(prob: np.ndarray, binary_mask: np.ndarray, anatomy: str, conf: dict, orig: np.ndarray = None) -> go.Figure:
    if orig is not None:
        o = orig.astype(np.float32) / 255.0
        raw = o.mean(axis=-1) if o.ndim == 3 else o.squeeze()
    else:
        raw = prob.squeeze()
    blurred = gaussian_filter(binary_mask.squeeze().astype(np.float32), sigma=15)
    if blurred.max() > 0:
        blurred /= blurred.max()
    z = 0.35 * raw + 0.65 * blurred
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
    for i, c in enumerate(contours):
        fig.add_trace(go.Scatter3d(
            x=c[:, 1], y=c[:, 0],
            z=np.full(len(c), peak_z),
            mode='lines',
            line=dict(color=color, width=4),
            name='Structure Boundary',
            legendgroup='boundary',
            showlegend=(i == 0),
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
            zaxis_title='Height',
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
    vol = data.squeeze()          # (D, H, W)
    # Downsample to max 64 voxels per axis to keep HTML under ~5MB
    step = max(1, max(vol.shape) // 64)
    vol = vol[::step, ::step, ::step]
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


def _build_html(left_b64: str, plotly_fig: go.Figure, conf: dict, anatomy: str, metadata: dict = None) -> str:
    plotly_div = plotly_fig.to_html(full_html=False, include_plotlyjs='cdn')
    coverage = conf.get('coverage', '0.00%')
    mean_conf = conf.get('mean_confidence', '0.00%')
    detected = conf.get('detected', 'False')
    fmt = (metadata or {}).get('format', '—').upper()
    dim = (metadata or {}).get('dimensionality', '—')

    badge_color = _badge_color(anatomy)
    detected_color = '#10b981' if detected == 'True' else '#ef4444'
    detected_text = '&#10003; Detected' if detected == 'True' else '&#10007; Not found'

    cov_pct = _parse_pct(coverage)
    conf_pct = _parse_pct(mean_conf)
    cov_color = _pct_color(cov_pct)
    conf_color = _pct_color(conf_pct)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UltrasoundViz &mdash; {anatomy}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0d1117;
      color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    .header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 24px;
      background: #161b2e;
      border-bottom: 1px solid #21262d;
    }}
    .brand {{
      font-family: 'Courier New', Courier, monospace;
      font-size: 17px;
      font-weight: 700;
      color: #58a6ff;
      letter-spacing: 0.4px;
    }}
    .brand-sub {{ color: #8b949e; font-weight: 400; }}
    .header-badges {{ display: flex; gap: 8px; align-items: center; }}
    .badge {{
      padding: 4px 11px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.2px;
    }}
    .badge-anatomy {{
      background: {badge_color}22;
      color: {badge_color};
      border: 1px solid {badge_color}55;
    }}
    .badge-format {{
      background: #21262d;
      color: #8b949e;
      border: 1px solid #30363d;
      font-family: 'Courier New', monospace;
      font-size: 11px;
    }}
    .stats-row {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      padding: 16px 24px;
    }}
    .stat-card {{
      background: #161b2e;
      border: 1px solid #21262d;
      border-radius: 10px;
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .stat-label {{
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #8b949e;
      font-weight: 600;
    }}
    .stat-value {{
      font-size: 26px;
      font-weight: 700;
      line-height: 1.1;
    }}
    .stat-sub {{ font-size: 11px; color: #8b949e; margin-top: 2px; }}
    .progress-bar {{
      height: 3px;
      background: #21262d;
      border-radius: 2px;
      margin-top: 10px;
      overflow: hidden;
    }}
    .progress-fill {{ height: 100%; border-radius: 2px; }}
    .main {{
      display: flex;
      gap: 16px;
      flex: 1;
      padding: 0 24px 16px;
      min-height: 520px;
    }}
    .panel-2d {{
      flex: 0 0 35%;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .panel-2d img {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid #21262d;
      display: block;
    }}
    .panel-caption {{
      font-size: 11px;
      color: #8b949e;
      text-align: center;
      padding: 7px 10px;
      background: #161b2e;
      border-radius: 6px;
      border: 1px solid #21262d;
    }}
    .panel-3d {{ flex: 1; min-height: 480px; }}
    .panel-3d > div {{ height: 480px !important; }}
    .footer {{
      padding: 10px 24px;
      background: #161b2e;
      border-top: 1px solid #21262d;
      font-size: 11px;
      color: #484f58;
      display: flex;
      justify-content: space-between;
    }}
  </style>
</head>
<body>
  <header class="header">
    <div class="brand">UltrasoundViz <span class="brand-sub">/ medical imaging</span></div>
    <div class="header-badges">
      <span class="badge badge-anatomy">{anatomy}</span>
      <span class="badge badge-format">{fmt} &middot; {dim}</span>
    </div>
  </header>

  <div class="stats-row">
    <div class="stat-card">
      <div class="stat-label">Anatomy</div>
      <div class="stat-value" style="font-size:20px;color:{badge_color}">{anatomy}</div>
      <div class="stat-sub">{dim}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Detection</div>
      <div class="stat-value" style="font-size:20px;color:{detected_color}">{detected_text}</div>
      <div class="stat-sub">structure in scan</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Coverage</div>
      <div class="stat-value" style="color:{cov_color}">{coverage}</div>
      <div class="stat-sub">of image area</div>
      <div class="progress-bar">
        <div class="progress-fill" style="width:{min(cov_pct,100):.1f}%;background:{cov_color}"></div>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Mean Confidence</div>
      <div class="stat-value" style="color:{conf_color}">{mean_conf}</div>
      <div class="stat-sub">model certainty</div>
      <div class="progress-bar">
        <div class="progress-fill" style="width:{min(conf_pct,100):.1f}%;background:{conf_color}"></div>
      </div>
    </div>
  </div>

  <div class="main">
    <div class="panel-2d">
      <img src="data:image/png;base64,{left_b64}" alt="Original scan with segmentation overlay">
      <div class="panel-caption">Original &middot; {anatomy} &middot; segmentation overlay</div>
    </div>
    <div class="panel-3d">
      {plotly_div}
    </div>
  </div>

  <footer class="footer">
    <span>UltrasoundViz &middot; {anatomy} &middot; {dim}</span>
    <span>Interactive 3D &mdash; rotate &middot; zoom &middot; hover for details</span>
  </footer>
</body>
</html>"""


def visualize(orig: np.ndarray, probability_map, binary_mask, conf: dict, metadata: dict) -> str:
    anatomy = metadata.get('anatomy', 'unknown')
    dim = metadata.get('dimensionality', '2D_single')

    if dim == '2D_single':
        orig_single = orig if not isinstance(orig, list) else orig[0]
        left_b64 = _make_left_panel_b64(orig_single, binary_mask, anatomy)
        fig = _mode1_surface(probability_map, binary_mask, anatomy, conf, orig_single)
        return _build_html(left_b64, fig, conf, anatomy, metadata)

    if dim == '3D_volume':
        mid_d = orig.shape[0] // 2
        mid_frame = orig[mid_d]
        empty_mask = np.zeros(mid_frame.shape[:2] + (1,), dtype=np.uint8)
        left_b64 = _make_left_panel_b64(mid_frame, empty_mask, anatomy)
        fig = _mode3_isosurface(orig, anatomy, conf)
        return _build_html(left_b64, fig, conf, anatomy, metadata)

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
        return _build_html(left_b64, fig, conf, anatomy, metadata)

    raise ValueError(f'Unknown dimensionality: {dim}')
