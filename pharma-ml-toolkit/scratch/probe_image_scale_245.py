"""Round-245 probe: ViewBox.viewPixelSize() drives the §23.10 scale.

Engine-printed FIRST (the probe-first rule).  Pinned facts:

* ``viewPixelSize()`` returns (w, h) -- the size of ONE screen pixel
  in view coordinates -- so the on-screen size of one image pixel is
  ``1 / viewPixelSize()`` per axis (the item transform is identity);
* the cache invalidates across ``setRange``/``autoRange`` (the fit,
  a 2x2 zoom-in window, and a squeeze-100x100-into-100px range all
  answer different values in one process);
* the regimes: fit 100x100 into a 400px view -> ~3.1 screen px per
  image px; a 2x2 window -> ~155.7; 100x100 into a 100px view ->
  0.375 (sub-pixel: one screen pixel straddles ~2.7 image pixels);
* hidden widgets answer a default-geometry value (no crash), so the
  tests that pin numbers show their panels explicitly.

Run with the project venv's python (see README for the interpreter
path convention; the audit keeps absolute home paths out of the tree).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


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

    data = np.arange(10000, dtype=float).reshape(100, 100)
    img.setImage(data)
    view.autoRange()
    app.processEvents()

    vb = view.getViewBox()
    s1 = vb.viewPixelSize()
    print(f"fit 100x100 into 400px view: view units/screen px = {s1}")
    print(f"  -> one image px = {1.0 / s1[0]:.4f} x {1.0 / s1[1]:.4f} screen px")

    vb.setRange(xRange=(10, 12), yRange=(10, 12))
    app.processEvents()
    s2 = vb.viewPixelSize()
    print(f"2x2 window: one image px = {1.0 / s2[0]:.2f} x {1.0 / s2[1]:.2f} screen px")

    glw.resize(100, 100)
    app.processEvents()
    vb.setRange(xRange=(0, 100), yRange=(0, 100))
    app.processEvents()
    s3 = vb.viewPixelSize()
    scale3 = (1.0 / s3[0], 1.0 / s3[1])
    print(
        f"100x100 into ~100px view: one image px = {scale3[0]:.4f} x "
        f"{scale3[1]:.4f} screen px (sub-pixel regime)"
    )
    print(f"  -> one screen px straddles <= {np.ceil(1 / scale3[0]):.0f} x "
          f"{np.ceil(1 / scale3[1]):.0f} image px")

    # hidden widget: default geometry, no crash (documented honesty)
    glw2 = pg.GraphicsLayoutWidget()
    view2 = glw2.addPlot(row=0, col=0)
    img2 = pg.ImageItem(axisOrder="row-major")
    view2.addItem(img2)
    img2.setImage(data)
    view2.autoRange()
    s4 = view2.getViewBox().viewPixelSize()
    print(f"hidden widget: one image px = {1.0 / s4[0]:.3f} screen px (default geometry)")
    print("PROBE245 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
