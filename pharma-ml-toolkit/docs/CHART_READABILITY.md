# Chart readability and export

All plotting helpers share the renderer in `pharmaml.viz._compat`. The defaults are
intended to keep titles, labels and thin lines readable in notebooks, desktop cards,
HTML reports and document exports without requiring each plot to restyle itself.

## Defaults

| Setting | Default | Purpose |
|---|---:|---|
| `DEFAULT_FIGSIZE` | `(9.0, 6.0)` inches | More room for labels, legends and multiple series |
| display DPI | 120 | Clear interactive/notebook figures |
| `EXPORT_DPI` | 220 | High-resolution PNG and rasterized PDF/SVG artists |
| `HTML_DPI` | 160 | Crisp, zoomable embedded report images |
| `MAX_RASTER_PIXELS` | 40,000,000 | Memory safety for exceptionally large physical figures |
| layout | `tight` | Reserves room for labels and titles |
| export bounds | tight + 0.16-inch padding | Prevents clipped titles and tick labels |

The house style uses a white background, stronger title hierarchy, quieter grid lines,
subtle left/bottom spines and consistent label/tick spacing. Existing plotting function
signatures are unchanged.

## Physical scale versus export resolution

Use `scale` when a dense chart needs **more layout room**. It changes the physical figure
size, not the font size and not merely the number of pixels:

```python
from pharmaml.viz import new_figure, save_figure

fig, ax = new_figure(scale=1.35)  # 12.15 x 8.10 inches
ax.plot([0, 1, 2], [2.0, 4.5, 3.0], label="cohort A")
ax.set(title="Dose response", xlabel="Dose (mg)", ylabel="Response")
ax.legend(frameon=False)

save_figure(fig, "artifacts/dose-response.png")          # 220 DPI
save_figure(fig, "artifacts/dose-response.svg")          # vector
save_figure(fig, "artifacts/dose-response.pdf", dpi=300) # explicit override
```

Use `dpi` when the layout is already right but the output needs **more pixels**. Passing
`figsize=` still works exactly as before and is combined with `scale` when both are given.
Pass `layout="constrained"` for Matplotlib's constrained layout or `layout=None` to manage
spacing manually. Invalid, non-finite and non-positive sizes, scales and DPI values raise
`DataValidationError` instead of creating a corrupt or empty export.

For HTML, `figure_to_base64(fig)` uses `HTML_DPI`; it also accepts `dpi=` explicitly.
Exceptionally large physical figures are subject to `MAX_RASTER_PIXELS`: effective DPI
is reduced only when the base figure would exceed 40 million pixels, with extra margin
for tight-bound labels. This prevents a large similarity matrix from exhausting memory;
ordinary figures still use the full requested DPI, and SVG/PDF paths and text stay vector.
SVG/PDF metadata handling remains reproducible: SVG dates/IDs and PDF creation dates do
not make otherwise identical exports differ between runs.

## Reproducible example

```bash
python examples/chart_readability.py
```

This writes PNG, SVG, PDF and a self-contained HTML preview under
`artifacts/chart-readability/` (ignored by Git). The example uses only NumPy and the
reporting dependencies; it needs no display server or external data.

## Renderer-agnostic figure specs

`pharmaml.viz.figures.render_matplotlib()` uses the same house style as native plotting
helpers. Single-panel specs keep an 8×6 canvas. Specs built with `make_subplots()` retain
their declared grid cells, subplot titles, overall figure title and axis titles; unused
cells are hidden and empty legends are omitted. Scatter/line, horizontal or vertical bar,
and heat-map traces are supported. This keeps a spec's Matplotlib export visually aligned
with its interactive Canvas representation instead of collapsing every trace onto one
unlabelled axes.

## Playground presentation

Desktop figure hosts default to 390 pixels high (individual dense cards may request more).
The TkAgg path preserves aspect ratio as its host resizes and never intentionally crops
the centre of an oversized figure. The PNG fallback renders at `HTML_DPI`, fits in both
directions with high-quality resampling and permits at most a 1.35× display upscale.
Interactive toolbar, pan/zoom, crosshair and session-report behavior remain available.

Native GUI rendering still requires a working Tk/display environment; headless tests cover
the sizing policy and raster/vector exports, not physical window appearance.
