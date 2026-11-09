"""Part-3 §23 responsive discipline: the shared layout policies.

§23.2 requires responsive Qt layouts -- ``QSplitter`` for adjustable
content areas, minimum and maximum widget dimensions, and the explicit
prevent-list: *no text overlapping images, no axis labels being
clipped, no tables overflowing their available space, no important
controls becoming inaccessible.*

The Qt defaults violate two of those outright:

* every ``QSplitter`` child is **collapsible** -- a careless drag
  hides the control column (or the whole chart) until the user finds
  the handle again;
* nothing below the splitter sets a minimum extent, so a squeeze
  clips form rows, table headers and axis labels long before the
  window itself refuses to shrink.

:func:`configure_splitter` applies the policy in one call, and every
page's splitter goes through it (the audit pass §23.2 asks for).
:func:`wrap_in_scroll` is the other half of the same discipline for
*tall* content: a form whose rows plus results board exceed the
viewport cannot shrink further -- without a scroll region the bottom
rows (the run button, the summary, the board) are simply clipped
away, which is §23.2's "important controls becoming inaccessible"
verbatim.  The wrapper is a visual no-op while the content fits
(no scrollbars, no frame, no inset background); it only engages when
the viewport is genuinely smaller than the content's minimum.
§23.3's export-legibility floor :data:`LEGIBILITY_FLOOR` lives here
too: the panels export at their own rendered size by contract, the
*user-facing* export buttons pass the floor so a chart squeezed into
a corner still exports legibly (see
:mod:`ui.widgets.image_export`).

§23.5 (scientific integrity): every helper here touches **display
geometry only** -- no data, no ranges, no normalisation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QFrame, QScrollArea, QWidget

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PySide6.QtWidgets import QSplitter

__all__ = [
    "LEGIBILITY_FLOOR",
    "MIN_CHILD_PX",
    "SCROLL_MIN_PX",
    "configure_splitter",
    "wrap_in_scroll",
]

#: §23.2: the smallest splitter child that still shows a form column
#: (label + input + margin) or a chart (axis + tick text) unclipped.
MIN_CHILD_PX = 320

#: §23.3 "generated plots remain legible when exported": exports from
#: user-facing buttons render at *at least* this size.  A 640×480 PNG
#: keeps tick labels and legend text readable in slides and papers;
#: smaller widgets are re-rendered from the vector scene at floor
#: scale (aspect preserved -- §23.4), never bitmap-stretched.
LEGIBILITY_FLOOR = QSize(640, 480)

#: §23.2: the smallest usable scroll region.  A wrapped form whose
#: viewport collapses below this is itself an inaccessible control
#: (a scrollbar slimmer than a row cannot be used), so the wrapper
#: holds a hard minimum height and the surrounding layout stops
#: squeezing there.
SCROLL_MIN_PX = 240


def configure_splitter(
    splitter: QSplitter,
    *,
    min_sizes: Sequence[int] | None = None,
    stretch: Sequence[int | None] | None = None,
) -> None:
    """Apply the §23.2 splitter policy to an already-populated splitter.

    * **No collapsible children** -- the drag handle can never hide a
      control column or a chart (§23.2 "important controls becoming
      inaccessible").
    * **Hard minimum extent per child** -- ``min_sizes`` (logical px,
      one per child) or the :data:`MIN_CHILD_PX` policy default, so a
      squeeze stops at readable, never clipped (axis labels, form
      rows, table headers).  Horizontal splitters clamp *width*,
      vertical ones *height*.
    * **Optional stretch factors** -- window growth flows to the
      flexible side (the chart), not the form column (§23.3
      "appropriate resizing during window changes").

    Display geometry only (§23.5): sizes, factors, minimums -- the
    underlying data and plot ranges are untouched.

    ``stretch`` entries may be ``None`` to leave a child's factor
    alone; children beyond the given sequences keep the defaults.
    """
    children = [splitter.widget(i) for i in range(splitter.count())]
    if not children:
        return

    splitter.setChildrenCollapsible(False)

    horizontal = splitter.orientation() == Qt.Orientation.Horizontal
    for i, child in enumerate(children):
        px = int(min_sizes[i]) if min_sizes is not None and i < len(min_sizes) else MIN_CHILD_PX
        px = max(px, 1)
        if horizontal:
            child.setMinimumWidth(px)
        else:
            child.setMinimumHeight(px)
        if stretch is not None and i < len(stretch):
            sf = stretch[i]
            if sf is not None:
                splitter.setStretchFactor(i, int(sf))


def wrap_in_scroll(content: QWidget) -> QScrollArea:
    """Wrap *content* in the §23.2 scroll policy and return the area.

    For tab/page bodies whose minimum height can exceed the viewport
    (the ML Workbench's forms: the round-235 probe measured the
    Leaderboards tab at a 775 px minimum against the ~450 px the
    860×560 minimum window leaves -- over 300 px of form, run buttons
    and results board unreachable below the fold).

    * **widget-resizable** -- the content tracks the viewport width,
      so rows reflow instead of needing sideways scrolling;
    * **horizontal scrollbars off** -- §23.2's "no tables overflowing
      their available space" is enforced by making horizontal overflow
      impossible, not by offering a scrollbar for it;
    * **vertical scrollbars as needed** -- the only escape a genuinely
      tall form is allowed, engaging solely when the content's own
      minimum no longer fits;
    * **frameless, palette-blended** -- while the content fits the
      wrapper is undetectable (no inset frame, no off-palette base),
      so pages keep their exact fitted look;
    * **hard minimum height** :data:`SCROLL_MIN_PX` -- the region
      itself never collapses to an unusable sliver.

    Display geometry only (§23.5): the content instance is carried
    over untouched (same object, same layout, same data), only where
    it is mounted changes.
    """
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setBackgroundRole(QPalette.ColorRole.Window)
    area.setMinimumHeight(SCROLL_MIN_PX)
    area.setWidget(content)
    return area
