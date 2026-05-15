# UltrasoundViz Output Page Redesign

**Date:** 2026-05-15  
**Scope:** Redesign `_build_html()` in `src/visualizer.py` — no new files, no new dependencies.

---

## Goal

Transform the minimal dark-background dump into a professional shareable HTML report suitable for sending to colleagues.

---

## Layout (top to bottom)

### 1. Header bar
- Left: `UltrasoundViz / medical imaging` in monospace, blue accent
- Right: anatomy badge (colour-coded to match ring colour) + format/dimensionality badge

### 2. Stats cards row (4 equal cards)
| Card | Value | Extra |
|---|---|---|
| Anatomy | anatomy label, anatomy-coloured | dimensionality sub-label |
| Detection | True/False, green/red | "structure found" sub-label |
| Coverage | percentage, colour-scaled | progress bar fill |
| Mean Confidence | percentage, colour-scaled | progress bar fill |

Colour scale: ≥75% → green `#10b981`, 50–75% → orange `#f97316`, <50% → red `#ef4444`

### 3. Main content panel (two columns)
- Left (35%): 2D matplotlib PNG + caption strip
- Right (65%): Plotly 3D interactive figure

### 4. Footer strip
- Left: `UltrasoundViz · {anatomy} · {dimensionality}`
- Right: `Interactive 3D — rotate · zoom · hover for details`

---

## Colour palette
| Token | Value |
|---|---|
| Page background | `#0d1117` |
| Card surface | `#161b2e` |
| Border | `#21262d` |
| Primary text | `#e6edf3` |
| Muted text | `#8b949e` |
| Accent blue | `#58a6ff` |

### Anatomy badge colours
| Anatomy | Colour |
|---|---|
| JugularVein | `#3b82f6` |
| CarotidArtery / FemoralArtery | `#06b6d4` |
| LIDC | `#ef4444` |
| Heart | `#f97316` |
| Ball | `#10b981` |
| unknown | `#8b949e` |

---

## Constraints
- Pure inline CSS — no external fonts, no JS beyond Plotly CDN
- Single self-contained HTML file
- All 42 existing tests must continue to pass (tests check for `<!DOCTYPE html>`, anatomy name, coverage %, and `plotly` in HTML)
