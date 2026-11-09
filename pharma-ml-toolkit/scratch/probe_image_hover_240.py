"""Round 240 probe -- Part-3 SS21.4 image pixel inspection feasibility.

Two questions, answered engine-first before any widget code changes:

1. SCENE->PIXEL MAPPING: does ``ImageItem.mapFromScene`` return data
   coordinates where ``x`` indexes columns and ``y`` indexes rows for
   a row-major matrix, so a hover handler can ``floor`` its way to the
   pixel under the cursor?  Verified by round-tripping
   ``mapToScene(col, row)`` through ``mapFromScene`` on a SHOWN panel
   (the transforms only exist after the first layout pass).

2. RGB FEASIBILITY: does an ``(h, w, 3)`` uint8 array survive
   ``ImageItem.setImage`` with a ``HistogramLUTItem`` attached (the
   panel's standing colormap control), and does ``autoRange`` keep
   the aspect-locked geometry?  Printed honestly; the widget accepts
   or refuses RGB based on what this probe shows.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_TK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TK))

from ui.headless import ensure_headless

ensure_headless()

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QEvent

print(f"[probe] pyqtgraph {pg.__version__}")
print(f"[probe] QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM', '<unset>')}")


def _settle(app, rounds: int = 8) -> None:
    for _ in range(rounds):
        app.processEvents()


def main() -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])

    # -- question 1: the scene -> pixel mapping round-trip -----------------
    from ui.widgets.image_panel import ImagePanel

    panel = ImagePanel()
    matrix = np.arange(35, dtype=float).reshape(5, 7)
    panel.set_matrix(matrix)
    panel.resize(420, 300)
    panel.show()
    _settle(app)

    img = panel.image_item()
    ok = True
    for col, row in [(0, 0), (3, 2), (6, 4), (5.5, 3.5)]:
        scene_pt = img.mapToScene(QPointF(float(col), float(row)))
        back = img.mapFromScene(scene_pt)
        drift = (abs(back.x() - col), abs(back.y() - row))
        status = "OK" if max(drift) < 1e-6 else "DRIFT"
        if status != "OK":
            ok = False
        print(
            f"[probe] map round-trip data=({col:g},{row:g}) "
            f"scene=({scene_pt.x():.3f},{scene_pt.y():.3f}) "
            f"back=({back.x():.4f},{back.y():.4f}) {status}"
        )

    # negative coordinates (left/below origin) must stay negative --
    # floor() then fails the bounds check honestly.
    below = img.mapFromScene(img.mapToScene(QPointF(-0.4, -0.4)))
    print(
        f"[probe] negative mapping back=({below.x():.4f},{below.y():.4f}) "
        f"(must be negative to refuse the pixel)"
    )

    # leaveEvent wiring smoke: a Leave event must be deliverable.
    leave = QEvent(QEvent.Type.Leave)
    try:
        panel.leaveEvent(leave)  # noqa: F841 - delivered, no crash is the point
        print("[probe] leaveEvent delivered OK (no crash)")
    except TypeError as exc:
        ok = False
        print(f"[probe] leaveEvent signature FAILS: {exc}")

    panel.close()

    # -- question 2: RGB arrays through the standing panel ----------------
    rgb_panel = ImagePanel()
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    try:
        img2 = rgb_panel.image_item()
        img2.setImage(rgb)
        rgb_panel.widget().resize(300, 240)
        rgb_panel.show()
        _settle(app)
        # does the HistogramLUT (already bound at construction) explode on
        # an RGB image?  does the histogram's gradient lookup still work?
        hist_ok = True
        try:
            rgb_panel.set_gradient("magma")
        except Exception as exc:  # noqa: BLE001 - probe prints, never dies
            hist_ok = False
            print(f"[probe] gradient switch on RGB FAILS: {type(exc).__name__}: {exc}")
        rendered = img2.pixmap() if hasattr(img2, "pixmap") else None
        print(
            f"[probe] RGB setImage renders OK, histogram gradient "
            f"{'OK' if hist_ok else 'BROKEN'}, pixmap={'yes' if rendered is not None else 'n/a'}"
        )
        # levels on an RGB image (what HistogramLUTItem does on gradient
        # drags) -- the honest compatibility answer:
        try:
            img2.setLevels([0.0, 255.0])
            print("[probe] RGB setLevels([0,255]) accepted")
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] RGB setLevels FAILS: {type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] RGB setImage FAILS: {type(exc).__name__}: {exc}")
    finally:
        rgb_panel.close()

    print(f"[probe] question-1 mapping verdict: {'USABLE' if ok else 'UNUSABLE'}")


if __name__ == "__main__":
    main()
