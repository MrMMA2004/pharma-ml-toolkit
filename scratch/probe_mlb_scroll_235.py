"""Round-235 probe: which ML Workbench tabs overflow vertically at the
app's minimum window size (860x560)?  Feeds the SS23 slice-2 decision
(wrap the overflowing tab contents in the responsive scroll policy)."""

from __future__ import annotations

import sys
from pathlib import Path

_TK = Path(__file__).resolve().parents[1] / "pharma-ml-toolkit"
sys.path.insert(0, str(_TK))

from ui.headless import ensure_headless

ensure_headless()

from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from ui.main_window import MainWindow

window = MainWindow()
window.resize(860, 560)
window.show()
app.processEvents()

from ui.pages.ml_workbench import MLWorkbenchPage

pages = window.findChildren(MLWorkbenchPage)
page = pages[0]
tabs = page._tabs
print(f"tab widget viewport-ish size: {tabs.size()}")
print(f"page size: {page.size()}")
print()
for i in range(tabs.count()):
    tabs.setCurrentIndex(i)
    app.processEvents()
    w = tabs.widget(i)
    min_hint = w.minimumSizeHint()
    avail = tabs.height() - tabs.tabBar().height()
    flag = "OVERFLOW" if min_hint.height() > avail else "fits"
    print(
        f"tab {i} '{tabs.tabText(i)}': min-height hint {min_hint.height()}px "
        f"(width {min_hint.width()}px) vs ~{avail}px available -> {flag}"
    )

# also: how much does the whole page need vs have?
page_min = page.minimumSizeHint()
print(f"\npage minimumSizeHint: {page_min.height()}px vs {page.height()}px actual")

window.close()
print("PROBE-DONE")
