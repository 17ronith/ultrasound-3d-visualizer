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


def _build_html(left_b64: str, plotly_fig: go.Figure, conf: dict, anatomy: str) -> str:
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
    anatomy = metadata.get('anatomy', 'unknown')
    dim = metadata.get('dimensionality', '2D_single')

    if dim == '2D_single':
        orig_single = orig if not isinstance(orig, list) else orig[0]
        left_b64 = _make_left_panel_b64(orig_single, binary_mask, anatomy)
        fig = _mode1_surface(probability_map, binary_mask, anatomy, conf)
        return _build_html(left_b64, fig, conf, anatomy)

    if dim == '3D_volume':
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
