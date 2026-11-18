"""Round-246 probe: the §21.4 original-resolution (1:1) view mechanics.

Engine-printed FIRST (the probe-first rule).  Pinned facts:

* the view range that renders ONE image pixel at ONE screen pixel is
  exactly the view box's own (width, height) in px -- recoverable
  from the live ``viewRange()`` / ``viewPixelSize()`` pair (the
  circular arithmetic: ``box_px = range_span / data_units_per_px``),
  centred on the CURRENT view centre (the image-viewer convention);
* ``setRange(..., padding=0)`` under ``setAspectLocked(True)`` does
  NOT expand a range whose aspect already equals the box's -- the
  achieved ``viewPixelSize()`` lands at ~1.0 on BOTH axes (0.13.7,
  measured not assumed), including on a NON-SQUARE view with a
  NON-SQUARE matrix (the letterbox lives in the range, not the
  scale);
* the achieved scale is fresh WITHOUT an event pump (setRange
  invalidates the view cache synchronously -- the round-245 probe
  pinned the invalidation; this probe re-measures after a pump and
  prints both to confirm they agree);
* the 1:1 state is idempotent (re-apply stays ~1.0), reachable from
  a zoomed-in AND a sub-pixel zoomed-out start, and left again by
  ``autoRange()``;
* hidden widgets keep answering their default-geometry value (the
  panel-level gate for the round is ``isVisible()`` -- the engine
  probe only notes the default value here).

Run with the project venv's python (see README for the interpreter
path convention; the audit keeps absolute home paths out of the tree).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def apply_1to1(view) -> tuple[float, float] | None:
    """The round's algorithm at engine level (the panel wraps this)."""
    vb = view.getViewBox()
    pw, ph = vb.viewPixelSize()
    if pw <= 0.0 or ph <= 0.0:
        return None
    (x0, x1), (y0, y1) = vb.viewRange()
    box_w, box_h = (x1 - x0) / pw, (y1 - y0) / ph
    if box_w <= 0.0 or box_h <= 0.0:
        return None
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    vb.setRange(
        xRange=[cx - box_w / 2.0, cx + box_w / 2.0],
        yRange=[cy - box_h / 2.0, cy + box_h / 2.0],
        padding=0,
    )
    pw2, ph2 = vb.viewPixelSize()
    return 1.0 / pw2, 1.0 / ph2


def main() -> int:
    app = QApplication([])
    print("pyqtgraph", pg.__version__)

    glw = pg.GraphicsLayoutWidget()
    view = glw.addPlot(row=0, col=0)
    view.setAspectLocked(True)
    img = pg.ImageItem(axisOrder="row-major")
    view.addItem(img)
    glw.resize(400, 400)
    glw.show()
    app.processEvents()

    img.setImage(np.arange(10000, dtype=float).reshape(100, 100))
    view.autoRange()
    app.processEvents()
    vb = view.getViewBox()

    s_fit = vb.viewPixelSize()
    print(
        f"fit 100x100 into 400px: one image px = "
        f"{1 / s_fit[0]:.3f} x {1 / s_fit[1]:.3f} screen px"
    )

    a1 = apply_1to1(view)
    app.processEvents()
    s_pump = vb.viewPixelSize()
    print(
        f"1:1 from fit: achieved {a1[0]:.4f} x {a1[1]:.4f}; "
        f"after a pump {1 / s_pump[0]:.4f} x {1 / s_pump[1]:.4f} (fresh, no pump needed)"
    )

    a2 = apply_1to1(view)
    print(f"1:1 re-applied (idempotent): {a2[0]:.4f} x {a2[1]:.4f}")

    # from a hard zoom-in; centre preservation included
    vb.setRange(xRange=(10, 12), yRange=(40, 42))
    app.processEvents()
    s_in = vb.viewPixelSize()
    print(f"2x2 window: {1 / s_in[0]:.1f} x {1 / s_in[1]:.1f} screen px per image px")
    before_centre = tuple((r[0] + r[1]) / 2.0 for r in vb.viewRange())
    a3 = apply_1to1(view)
    after_centre = tuple((r[0] + r[1]) / 2.0 for r in vb.viewRange())
    after_span = tuple(r[1] - r[0] for r in vb.viewRange())
    print(
        f"1:1 from zoom-in: achieved {a3[0]:.4f} x {a3[1]:.4f}; "
        f"centre {before_centre} -> {after_centre}; "
        f"span {after_span[0]:.1f} x {after_span[1]:.1f} data units"
    )

    # sub-pixel start: 100x100 matrix in a 100px view
    glw.resize(100, 100)
    app.processEvents()
    vb.setRange(xRange=(0, 100), yRange=(0, 100))
    app.processEvents()
    s_out = vb.viewPixelSize()
    print(
        f"squeeze: {1 / s_out[0]:.3f} x {1 / s_out[1]:.3f} screen px per image px"
    )
    a4 = apply_1to1(view)
    print(f"1:1 from sub-pixel: achieved {a4[0]:.4f} x {a4[1]:.4f}")

    # non-square view + non-square matrix
    glw.resize(500, 300)
    app.processEvents()
    img.setImage(np.arange(4000, dtype=float).reshape(50, 80))
    view.autoRange()
    app.processEvents()
    s_ns = vb.viewPixelSize()
    print(f"fit 50x80 into 500x300: {1 / s_ns[0]:.3f} x {1 / s_ns[1]:.3f}")
    a5 = apply_1to1(view)
    print(f"1:1 non-square: achieved {a5[0]:.4f} x {a5[1]:.4f}")

    # autoRange leaves the 1:1 state
    view.autoRange()
    app.processEvents()
    s_back = vb.viewPixelSize()
    print(
        f"autoRange after 1:1: {1 / s_back[0]:.3f} x {1 / s_back[1]:.3f} "
        f"(left the unity state)"
    )

    # hidden default geometry note (the panel gates on isVisible)
    glw2 = pg.GraphicsLayoutWidget()
    view2 = glw2.addPlot(row=0, col=0)
    view2.setAspectLocked(True)
    img2 = pg.ImageItem(axisOrder="row-major")
    view2.addItem(img2)
    img2.setImage(np.zeros((10, 10)))
    view2.autoRange()
    s_hidden = view2.getViewBox().viewPixelSize()
    print(
        f"hidden default geometry: one image px = "
        f"{1 / s_hidden[0]:.2f} x {1 / s_hidden[1]:.2f} screen px "
        f"(the panel-level gate is isVisible())"
    )

    glw.close()
    glw2.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
