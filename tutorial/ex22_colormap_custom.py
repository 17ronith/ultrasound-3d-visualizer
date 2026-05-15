"""
Ex 22 — Create custom colormaps (grayscale, RGB, and non-interpolated).
This example is code-only (no window) — it just demonstrates the API.
"""
from _config import fast

# Grayscale colormap: [intensity_in, intensity_out, ...]
colormap_gray = fast.Colormap([0, 0, 100, 50, 200, 180, 255, 255], True)
print("Grayscale colormap created:", colormap_gray)

# RGB colormap: [intensity_in, R, G, B, ...]
colormap_rgb = fast.Colormap([0, 0, 0, 0, 100, 50, 50, 50, 200, 180, 180, 180, 255, 255, 255, 255], False)
print("RGB colormap created:", colormap_rgb)

# Without interpolation
colormap_no_interp = fast.Colormap([0, 0, 100, 50, 200, 180, 255, 255], True, False)
print("Non-interpolated colormap created:", colormap_no_interp)
