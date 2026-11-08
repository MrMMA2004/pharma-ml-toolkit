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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PySide6.QtWidgets import QSplitter

__all__ = [
    "LEGIBILITY_FLOOR",
    "MIN_CHILD_PX",
    "configure_splitter",
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
        if stretch is not None and i < len(stretch) and stretch[i] is not None:
            splitter.setStretchFactor(i, int(stretch[i]))
