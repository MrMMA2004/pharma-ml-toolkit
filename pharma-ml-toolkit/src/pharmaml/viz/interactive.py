r"""Interactive figures as data — the bokeh/dash/plotly round, serverless.

The interactive-plotting round of the absorption pipeline reviewed three
stacks function by function — bokeh's ``ColumnDataSource``/``Figure``/
``save`` HTML export, plotly's ``Figure`` (data + layout JSON, traces
and display modes) and dash's ``@app.callback`` dependency graph — and
kept what survives without a browser stack:

* :class:`Figure` — the plotly figure-as-data model: typed traces
  (:meth:`~Figure.add_scatter` with lines/markers/lines+markers modes,
  :meth:`~Figure.add_bar`, :meth:`~Figure.add_heatmap`), the
  :meth:`~Figure.set_layout` record, a deterministic sorted-key
  :meth:`~Figure.to_json` / :meth:`~Figure.from_json` round-trip,
  :meth:`~Figure.to_ascii` unicode block previews and
  :meth:`~Figure.axis_ranges`;
* :class:`ColumnDataSource` — bokeh's streaming/patching data source
  semantics: :meth:`~ColumnDataSource.stream` with optional rolloff,
  :meth:`~ColumnDataSource.patch` with index-or-slice replacement and
  length validation, :meth:`~ColumnDataSource.to_frame`;
* :class:`CallbackRegistry` — dash's reactive callback graph without
  the server: outputs are unique, inputs form a dependency DAG, and
  :meth:`~CallbackRegistry.simulate_update` fires dependent callbacks
  in topological order with cycle detection (``ValueError``);
* :func:`save_interactive_html` — the self-contained offline HTML
  export: the figure JSON rides in a ``<script
  type="application/json">`` block and a pure-Python static SVG
  rendering (polyline for scatter/line traces, polygons for bars, a
  small labelled cell table for heatmaps) makes the file viewable
  anywhere.  No JavaScript is generated, no CDN is referenced, and the
  bytes are deterministic (no timestamps) — the same figure always
  hashes the same.

The honest scope, stated once: the JavaScript/WebGL browser layer —
the bokehjs and plotly.js rendering engines, dash's React frontend and
its WebSocket server, GPU shaders and million-point downsampling — is
what this toolkit does not vendor.  What transfers is exactly the data
model those engines consume: the figure-as-JSON contract, the
stream/patch data-source semantics, the callback dependency graph, and
the offline export that keeps a study's figure inspectable decades
after the vendor's CDN moved.  The ASCII preview and static SVG are
*previews*, not replacements for a WebGL canvas — pan, zoom and tooltip
interactivity stay with the vendor libraries.

Examples
--------
>>> from pathlib import Path
>>> import json
>>> from pharmaml.viz.interactive import (
...     CallbackRegistry, ColumnDataSource, Figure, figure_report,
...     save_interactive_html,
... )

The figure as data:
>>> fig = Figure()
>>> fig.add_scatter([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], name="parabola",
...                 mode="markers")
>>> fig.add_bar(["a", "b"], [3.0, -1.0], name="delta")
>>> fig.traces[0]["type"], fig.traces[1]["name"]
('scatter', 'delta')
>>> fig.axis_ranges()                     # bars anchor their baseline at zero
{'x': (0.0, 2.0), 'y': (-1.0, 4.0)}
>>> Figure().axis_ranges()
{'x': None, 'y': None}
>>> fig.set_layout(title="Study 42", x_title="dose", y_title="response", y_log=True)
>>> (fig.layout["title"], fig.layout["y_log"], fig.layout["x_log"])
('Study 42', True, False)
>>> fig.set_layout(title="Only title")    # a fresh call clears the unset fields
>>> (fig.layout["title"], fig.layout["x_title"], fig.layout["y_log"])
('Only title', None, False)

Deterministic JSON round-trip:
>>> bars = Figure()
>>> bars.add_bar(["a", "b", "c"], [3.0, 5.0, 2.0], name="yield")
>>> bars.set_layout(title="Batch yield", x_title="batch", y_title="mg")
>>> text = bars.to_json()
>>> payload = json.loads(text)
>>> sorted(payload)
['data', 'layout']
>>> trace = payload["data"][0]
>>> sorted(trace)
['name', 'type', 'x', 'y']
>>> (trace["type"], trace["name"], trace["x"])
('bar', 'yield', ['a', 'b', 'c'])
>>> payload["layout"]["title"]
'Batch yield'
>>> Figure.from_json(text).to_json() == text
True
>>> Figure.from_json("{oops}")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: from_json: not valid JSON — Expecting property name enclosed in double quotes
>>> Figure.from_json('{"data": [{"type": "pie"}]}')
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: from_json: unknown trace type 'pie' (expected scatter, bar or heatmap)

Unicode block previews (rows are right-stripped; y grows upward for
scatter/bars, heatmaps read like their printed table — first row on
top):
>>> vee = Figure()
>>> vee.add_scatter([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], name="vee")
>>> print(vee.to_ascii(width=11, height=5))
     █
    █ █
  ██   ██
 █       █
█         █
>>> bars.to_ascii(width=11, height=6).splitlines()[3]
'█    █    █'
>>> heat = Figure()
>>> heat.add_heatmap([[1.0, 2.0], [3.0, 4.0]])
>>> print(heat.to_ascii(width=5, height=2))
    ░
▓   █
>>> heat.axis_ranges()                    # cells span half-slots around coords
{'x': (-0.5, 1.5), 'y': (-0.5, 1.5)}
>>> Figure().to_ascii()
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: to_ascii: the figure has no traces to preview

Text reports:
>>> studied = Figure()
>>> studied.add_scatter([0.0, 1.0], [1.0, 2.0], name="dose-response",
...                     mode="markers")
>>> studied.add_bar(["a", "b"], [3.0, 5.0], name="yield")
>>> studied.set_layout(title="Study 42")
>>> print(figure_report(studied))
PharmaML interactive figure — 2 trace(s)
  trace 0: scatter 'dose-response' mode=markers n=2
  trace 1: bar 'yield' n=2
  layout: title='Study 42' x_title=None y_title=None x_log=False y_log=False
  axis ranges: x=(0.0, 1.0) y=(0.0, 5.0)
>>> print(figure_report(heat).splitlines()[1])
  trace 0: heatmap 2x2

Trace validation:
>>> bad = Figure()
>>> bad.add_scatter([0.0, 1.0], [1.0], name="mismatch")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: add_scatter: x (2) and y (1) must have the same length
>>> bad.add_scatter([0.0], [1.0], name="pie-mode", mode="pie")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: add_scatter: mode must be one of lines, markers, lines+markers — got 'pie'
>>> bad.add_heatmap([[1.0, 2.0], [3.0]])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: add_heatmap: z must be rectangular — row 1 has 1 column(s), expected 2
>>> bad.add_heatmap([[1.0, 2.0]], x=[0.0])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: add_heatmap: x needs 2 coordinate(s), got 1

Streaming and patching (bokeh's ``ColumnDataSource``):
>>> source = ColumnDataSource({"dose": [1.0, 2.0, 3.0], "resp": [0.1, 0.4, 0.9]})
>>> source.stream({"dose": [4.0], "resp": [1.6]}, rollover=4)
>>> source.data["dose"]
[1.0, 2.0, 3.0, 4.0]
>>> source.stream({"dose": [5.0], "resp": [2.5]}, rollover=4)
>>> source.data["dose"]                     # the oldest row rolled off
[2.0, 3.0, 4.0, 5.0]
>>> source.data["resp"]
[0.4, 0.9, 1.6, 2.5]
>>> source.patch({"dose": [(0, 20.0)]})
>>> source.data["dose"][0]
20.0
>>> source.patch({"resp": [(slice(1, 3), [0.5, 0.6])]})
>>> source.data["resp"]
[0.4, 0.5, 0.6, 2.5]
>>> frame = source.to_frame()
>>> frame.columns.tolist()
['dose', 'resp']
>>> frame["dose"].tolist()
[20.0, 3.0, 4.0, 5.0]
>>> source.patch({"dose": [(slice(1, 3), [1.0])]})    # wrong replacement length
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: patch for column 'dose': slice(1, 3) needs 2 value(s), got 1
>>> source.patch({"dose": [(17, 1.0)]})
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: patch for column 'dose': index 17 out of range for length 4
>>> source.patch({"missing": [(0, 1.0)]})
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: patch: column 'missing' not in the source (columns: ['dose', 'resp'])
>>> source.stream({"missing": [1.0]})
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: stream: column 'missing' not in the source (columns: ['dose', 'resp'])
>>> ColumnDataSource({"a": [1, 2], "b": [1]})
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: ColumnDataSource columns must have equal lengths: 'a' has 2, 'b' has 1

The callback dependency graph (dash's spirit, no server):
>>> registry = CallbackRegistry()
>>> events: list[str] = []
>>> def double(value):
...     events.append("double")
...     return value * 2
>>> def announce(value):
...     events.append("announce")
...     return f"value = {value}"
>>> registry.add("stats", "text", inputs=[("slider", "value")], function=double)
>>> registry.add("label", "children", inputs=[("stats", "text")], function=announce)
>>> registry.simulate_update("slider", "value", 21)
{('stats', 'text'): 42, ('label', 'children'): 'value = 42'}
>>> events                                  # topological order: producers first
['double', 'announce']
>>> registry.simulate_update("slider", "value", 1)
{('stats', 'text'): 2, ('label', 'children'): 'value = 2'}
>>> registry.simulate_update("unrelated", "prop", 0)   # nothing depends on it
{}
>>> registry.add("stats", "text", inputs=[("x", "v")], function=lambda v: v)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: add: output ('stats', 'text') is already registered — dash outputs are unique
>>> registry.add("sum", "v", inputs=[("a", "v"), ("b", "v")],
...              function=lambda x, y: x + y)
>>> registry.simulate_update("a", "v", 1)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: simulate_update: input ('b', 'v') of output ('sum', 'v') has no value — simulate_update it first
>>> registry.simulate_update("b", "v", 10)  # 'a' persisted from the call above
{('sum', 'v'): 11}
>>> CallbackRegistry().add("out", "v", inputs=[("bad",)], function=lambda v: v)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: add: inputs must be (component_id, prop) pairs, got ('bad',)
>>> CallbackRegistry().add("out", "v", function="nope")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: add: function must be callable, got 'nope'
>>> looped = CallbackRegistry()
>>> looped.add("a", "v", inputs=[("b", "v")], function=lambda v: v)
>>> looped.add("b", "v", inputs=[("a", "v")], function=lambda v: v)
>>> looped.simulate_update("a", "v", 1)
Traceback (most recent call last):
...
ValueError: cycle detected in the callback dependency graph

Self-contained offline HTML (no JavaScript, no CDN, deterministic bytes):
>>> wave = Figure()
>>> wave.add_scatter([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], name="wave")
>>> wave.set_layout(title="Dose wave", x_title="dose", y_title="response")
>>> out = save_interactive_html(wave, "/tmp/pharmaml_doctest_wave.html")
>>> out.as_posix()
'/tmp/pharmaml_doctest_wave.html'
>>> html = out.read_text(encoding="utf-8")
>>> "<svg" in html and 'type="application/json"' in html
True
>>> html.count("<svg")
1
>>> "<script>" not in html                 # JSON payload only — no executable JS
True
>>> 'src="http' not in html and 'href="http' not in html    # no CDN references
True
>>> "<polyline" in html and "<polygon" not in html
True
>>> again = save_interactive_html(wave, "/tmp/pharmaml_doctest_wave2.html")
>>> again.read_text(encoding="utf-8") == html      # byte-deterministic export
True
>>> two = save_interactive_html([wave, heat], "/tmp/pharmaml_doctest_two.html")
>>> two.read_text(encoding="utf-8").count("<svg")
2
>>> bar_html = save_interactive_html(bars, "/tmp/pharmaml_doctest_bars.html")
>>> "<polygon" in bar_html.read_text(encoding="utf-8")   # bars draw as polygons
True
>>> logfig = Figure()
>>> logfig.add_scatter([1.0, 10.0, 100.0], [1.0, 2.0, 3.0], name="log")
>>> logfig.set_layout(x_log=True)
>>> save_interactive_html(logfig, "/tmp/pharmaml_doctest_log.html").is_file()
True
>>> neg = Figure()
>>> neg.add_scatter([0.0, 1.0], [0.0, 1.0], name="neg")
>>> neg.set_layout(y_log=True)
>>> save_interactive_html(neg, "/tmp/pharmaml_doctest_neglog.html")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: y_log layout asks for a log axis but the y range includes non-positive values (0.0)
>>> save_interactive_html("not a figure", "/tmp/pharmaml_doctest_bad.html")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: save_interactive_html: expected Figure or a list of Figures, got str
>>> for stale in ("wave", "wave2", "two", "bars", "log"):
...     Path(f"/tmp/pharmaml_doctest_{stale}.html").unlink()
"""

from __future__ import annotations

import graphlib
import itertools
import json
import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "CallbackRegistry",
    "ColumnDataSource",
    "Figure",
    "figure_report",
    "save_interactive_html",
]

#: Trace stroke/fill colours (the plotly classic cycle, first six).
_TRACE_COLORS: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
)

#: Valid scatter display modes (the plotly vocabulary).
_SCATTER_MODES: tuple[str, ...] = ("lines", "markers", "lines+markers")

#: Layout fields with their defaults (``from_json`` backfills missing ones).
_LAYOUT_DEFAULTS: dict[str, Any] = {
    "title": None,
    "x_title": None,
    "y_title": None,
    "x_log": False,
    "y_log": False,
    "template": None,
}

#: Required keys per trace kind (``from_json`` validation).
_REQUIRED_TRACE_KEYS: dict[str, tuple[str, ...]] = {
    "scatter": ("type", "name", "mode", "x", "y"),
    "bar": ("type", "name", "x", "y"),
    "heatmap": ("type", "name", "z", "x", "y"),
}

#: The template registry (plotly's named themes, reduced to the knobs
#: the static SVG renderer actually consumes: the trace colour cycle,
#: the paper (page) fill and the axis ink).
TEMPLATES: dict[str, dict[str, Any]] = {
    "plotly": {
        "colors": _TRACE_COLORS,
        "paper": "#ffffff",
        "ink": "#64748b",
    },
    "plotly_white": {
        "colors": ("#636efa", "#EF553B", "#00cc96", "#ab63fa", "#19d3f3"),
        "paper": "#ffffff",
        "ink": "#b0b8c4",
    },
    "ggplot2": {
        "colors": ("#f8766d", "#00ba38", "#619cff", "#e58e00", "#b07aff"),
        "paper": "#f7f7f7",
        "ink": "#999999",
    },
    "seaborn": {
        "colors": ("#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"),
        "paper": "#eaeaf2",
        "ink": "#a0a0a0",
    },
}


def _hovertext_list(
    hovertext: Any, expected: int, caller: str
) -> list[str] | None:
    """Validate a per-point hover label sequence (plotly's hovertext)."""
    if hovertext is None:
        return None
    try:
        labels = [str(label) for label in hovertext]
    except TypeError as exc:
        raise DataValidationError(
            f"{caller}: hovertext must be a sequence of labels"
        ) from exc
    if len(labels) != expected:
        raise DataValidationError(
            f"{caller}: hovertext ({len(labels)}) must match the point count "
            f"({expected})"
        )
    return labels


def _numeric_list(values: Any, caller: str, axis: str) -> list[float]:
    """Validate a 1-D finite numeric sequence → plain Python floats."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"{caller}: {axis} must be numeric — conversion failed: {exc}"
        ) from exc
    if array.ndim != 1:
        raise DataValidationError(
            f"{caller}: {axis} must be 1-dimensional, got {array.ndim} dimension(s)"
        )
    if array.size == 0:
        raise DataValidationError(
            f"{caller}: {axis} is empty — a trace needs at least one point"
        )
    if not np.isfinite(array).all():
        raise DataValidationError(
            f"{caller}: {axis} contains NaN or infinite values"
        )
    return array.tolist()


def _category_list(values: Any, caller: str, axis: str) -> list[str | float]:
    """Validate bar categories: strings or numbers, at least one."""
    try:
        items = list(values)
    except TypeError as exc:
        raise DataValidationError(
            f"{caller}: {axis} must be a sequence, got {type(values).__name__}"
        ) from exc
    if not items:
        raise DataValidationError(
            f"{caller}: {axis} is empty — a trace needs at least one point"
        )
    out: list[str | float] = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, (int, float, np.integer, np.floating)) and not isinstance(
            item, bool
        ):
            out.append(float(item))
        else:
            raise DataValidationError(
                f"{caller}: {axis} must hold strings or numbers, got {item!r}"
            )
    return out


def _trace_positions(trace: Mapping[str, Any]) -> list[float]:
    """Numeric x positions of a trace (bar categories fall back to indices)."""
    xs = trace.get("x") or []
    if trace.get("type") == "bar":
        try:
            return [float(value) for value in xs]
        except (TypeError, ValueError):
            return [float(index) for index in range(len(xs))]
    return [float(value) for value in xs]


def _heatmap_coords(trace: Mapping[str, Any], axis: str) -> list[float]:
    """Heatmap cell coordinates (given coords, else slot indices)."""
    coords = trace.get(axis)
    if coords:
        return [float(value) for value in coords]
    z = trace.get("z") or []
    size = len(z[0]) if axis == "x" else len(z)
    return [float(index) for index in range(max(size, 1))]


# ---------------------------------------------------------------------------
# Figure — the plotly figure-as-data model
# ---------------------------------------------------------------------------


class Figure:
    """An interactive figure as pure data: typed traces + a layout record.

    The model mirrors plotly's ``{"data": [...], "layout": {...}}``
    contract, restricted to the three trace kinds the toolkit renders
    (scatter with lines/markers modes, bar, heatmap).  ``traces`` is a
    list of plain dicts of JSON-safe values, ``layout`` a dict with the
    keys ``title`` / ``x_title`` / ``y_title`` / ``x_log`` / ``y_log`` —
    everything serialises with :meth:`to_json` deterministically
    (sorted keys, no timestamps).

    Notes
    -----
    ``axis_ranges`` unions every trace; bar traces contribute the zero
    baseline, and heatmap cells span half-slots around their
    coordinates.  ``to_ascii`` is a linear-scale preview — it ignores
    the log flags (the SVG export honours them).
    """

    def __init__(self) -> None:
        self.traces: list[dict[str, Any]] = []
        self.layout: dict[str, Any] = dict(_LAYOUT_DEFAULTS)

    def add_scatter(
        self, x: Any, y: Any, name: str, mode: str = "lines",
        hovertext: Any = None,
    ) -> None:
        """Append a scatter trace (``mode``: lines, markers or both).

        ``hovertext`` (plotly's own trace field) carries one label per
        point -- the express layer fills it from a DataFrame column and
        the SVG export renders each as a native ``<title>`` tooltip.
        """
        xs = _numeric_list(x, "add_scatter", "x")
        ys = _numeric_list(y, "add_scatter", "y")
        if len(xs) != len(ys):
            raise DataValidationError(
                f"add_scatter: x ({len(xs)}) and y ({len(ys)}) must have the "
                "same length"
            )
        if mode not in _SCATTER_MODES:
            raise DataValidationError(
                f"add_scatter: mode must be one of {', '.join(_SCATTER_MODES)} "
                f"— got {mode!r}"
            )
        trace: dict[str, Any] = {
            "type": "scatter", "name": str(name), "mode": mode, "x": xs, "y": ys,
        }
        labels = _hovertext_list(hovertext, len(xs), "add_scatter")
        if labels is not None:
            trace["hovertext"] = labels
        self.traces.append(trace)

    def add_bar(self, x: Any, y: Any, name: str, hovertext: Any = None) -> None:
        """Append a bar trace (``x`` may be string categories)."""
        xs = _category_list(x, "add_bar", "x")
        ys = _numeric_list(y, "add_bar", "y")
        if len(xs) != len(ys):
            raise DataValidationError(
                f"add_bar: x ({len(xs)}) and y ({len(ys)}) must have the same "
                "length"
            )
        trace: dict[str, Any] = {"type": "bar", "name": str(name), "x": xs, "y": ys}
        labels = _hovertext_list(hovertext, len(xs), "add_bar")
        if labels is not None:
            trace["hovertext"] = labels
        self.traces.append(trace)

    def add_heatmap(self, z: Any, x: Any = None, y: Any = None) -> None:
        """Append a heatmap trace (``z``: rectangular numeric rows)."""
        try:
            rows = [list(row) for row in z]
        except TypeError as exc:
            raise DataValidationError(
                "add_heatmap: z must be 2-dimensional (a sequence of rows)"
            ) from exc
        if not rows or not rows[0]:
            raise DataValidationError("add_heatmap: z is empty")
        expected = len(rows[0])
        for index, row in enumerate(rows[1:], start=1):
            if len(row) != expected:
                raise DataValidationError(
                    f"add_heatmap: z must be rectangular — row {index} has "
                    f"{len(row)} column(s), expected {expected}"
                )
        try:
            grid = [[float(value) for value in row] for row in rows]
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"add_heatmap: z must be numeric — conversion failed: {exc}"
            ) from exc
        if not all(math.isfinite(value) for row in grid for value in row):
            raise DataValidationError("add_heatmap: z contains NaN or infinite values")
        x_coords = None
        y_coords = None
        if x is not None:
            x_coords = _numeric_list(x, "add_heatmap", "x")
            if len(x_coords) != expected:
                raise DataValidationError(
                    f"add_heatmap: x needs {expected} coordinate(s), "
                    f"got {len(x_coords)}"
                )
        if y is not None:
            y_coords = _numeric_list(y, "add_heatmap", "y")
            if len(y_coords) != len(grid):
                raise DataValidationError(
                    f"add_heatmap: y needs {len(grid)} coordinate(s), "
                    f"got {len(y_coords)}"
                )
        self.traces.append(
            {"type": "heatmap", "name": None, "z": grid, "x": x_coords, "y": y_coords}
        )

    def set_layout(
        self,
        title: str | None = None,
        x_title: str | None = None,
        y_title: str | None = None,
        x_log: bool = False,
        y_log: bool = False,
        template: str | None = None,
    ) -> None:
        """Set the whole layout in one call — unset arguments CLEAR their field.

        Every call replaces the layout record (the plotly-style
        assignment semantics), so ``set_layout(title="T")`` after
        ``set_layout(x_title="dose")`` leaves ``x_title`` ``None``.
        """
        self.layout = {
            "title": None if title is None else str(title),
            "x_title": None if x_title is None else str(x_title),
            "y_title": None if y_title is None else str(y_title),
            "x_log": bool(x_log),
            "y_log": bool(y_log),
            "template": None if template is None else str(template),
        }

    def axis_ranges(self) -> dict[str, tuple[float, float] | None]:
        """Data ranges over all traces (``None`` axes for an empty figure)."""
        xs: list[float] = []
        ys: list[float] = []
        for trace in self.traces:
            if trace.get("type") == "heatmap":
                x_coords = _heatmap_coords(trace, "x")
                y_coords = _heatmap_coords(trace, "y")
                xs.extend((min(x_coords) - 0.5, max(x_coords) + 0.5))
                ys.extend((min(y_coords) - 0.5, max(y_coords) + 0.5))
                continue
            xs.extend(_trace_positions(trace))
            ys.extend(float(value) for value in trace.get("y") or [])
            if trace.get("type") == "bar":
                ys.append(0.0)  # bars anchor their baseline at zero
        if not xs or not ys:
            return {"x": None, "y": None}
        return {"x": (min(xs), max(xs)), "y": (min(ys), max(ys))}

    def to_json(self) -> str:
        """Deterministic JSON (sorted keys, 2-space indent) of the figure."""
        return json.dumps(
            {"data": self.traces, "layout": self.layout}, sort_keys=True, indent=2
        )

    @classmethod
    def from_json(cls, text: str) -> Figure:
        """Rebuild a figure from :meth:`to_json` output (or compatible JSON).

        Raises
        ------
        DataValidationError
            On invalid JSON, a missing ``data`` list, or a trace with an
            unknown type or missing required keys.
        """
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DataValidationError(
                f"from_json: not valid JSON — {exc.msg}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise DataValidationError(
                "from_json: expected an object with a 'data' list of traces"
            )
        layout = payload.get("layout", {})
        if not isinstance(layout, dict):
            raise DataValidationError("from_json: 'layout' must be an object")
        figure = cls()
        for index, trace in enumerate(payload["data"]):
            if not isinstance(trace, dict):
                raise DataValidationError(
                    f"from_json: trace {index} must be an object, got "
                    f"{type(trace).__name__}"
                )
            kind = trace.get("type")
            if kind not in _REQUIRED_TRACE_KEYS:
                raise DataValidationError(
                    f"from_json: unknown trace type {kind!r} "
                    "(expected scatter, bar or heatmap)"
                )
            missing = [key for key in _REQUIRED_TRACE_KEYS[kind] if key not in trace]
            if missing:
                raise DataValidationError(
                    f"from_json: trace {index} is missing key(s) {missing}"
                )
            figure.traces.append(dict(trace))
        figure.layout = {
            key: layout.get(key, default) for key, default in _LAYOUT_DEFAULTS.items()
        }
        return figure

    def to_ascii(self, width: int = 60, height: int = 12) -> str:
        """Unicode block-character preview of the first trace.

        Scatter/bars are axis-oriented (y grows upward, bars filled from
        the zero baseline); heatmaps read like their printed table (the
        first row on top, five shading levels ``' ░▒▓█'``).  Rows are
        right-stripped; the scale is linear — log layout flags are
        ignored here (the SVG export honours them).

        Raises
        ------
        DataValidationError
            When the figure has no traces, or ``width``/``height`` are
            not integers >= 2.
        """
        for name, size in (("width", width), ("height", height)):
            if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
                raise DataValidationError(
                    f"to_ascii: {name} must be an integer >= 2, got {size!r}"
                )
            if int(size) < 2:
                raise DataValidationError(
                    f"to_ascii: {name} must be an integer >= 2, got {size!r}"
                )
        width, height = int(width), int(height)
        if not self.traces:
            raise DataValidationError("to_ascii: the figure has no traces to preview")
        trace = self.traces[0]
        grid: list[list[str]] = [[" "] * width for _ in range(height)]
        if trace.get("type") == "heatmap":
            z = trace.get("z") or [[]]
            rows = len(z)
            flat = [value for row in z for value in row]
            lo = min(flat) if flat else 0.0
            hi = max(flat) if flat else 0.0
            span = hi - lo
            for r, row in enumerate(z):
                for c, value in enumerate(row):
                    ai = round(r * (height - 1) / max(rows - 1, 1))
                    aj = round(c * (width - 1) / max(len(row) - 1, 1))
                    fraction = (value - lo) / span if span > 0 else 0.5
                    grid[ai][aj] = " ░▒▓█"[min(4, int(fraction * 5.0))]
        else:
            xs = _trace_positions(trace)
            ys = [float(value) for value in trace.get("y") or []]
            xmin, xmax = (min(xs), max(xs)) if xs else (0.0, 1.0)
            if trace.get("type") == "bar":
                ymin, ymax = min(0.0, min(ys)), max(0.0, max(ys))
            else:
                ymin, ymax = (min(ys), max(ys)) if ys else (0.0, 1.0)

            def col(x_value: float) -> int:
                if xmax > xmin:
                    return round((x_value - xmin) / (xmax - xmin) * (width - 1))
                return width // 2

            def row(y_value: float) -> int:
                if ymax > ymin:
                    return round((ymax - y_value) / (ymax - ymin) * (height - 1))
                return height // 2

            points = [(col(x), row(y)) for x, y in zip(xs, ys)]
            if trace.get("type") == "bar":
                zero_row = row(0.0)
                for c, r in points:
                    for rr in range(min(r, zero_row), max(r, zero_row) + 1):
                        grid[rr][c] = "█"
            else:
                mode = trace.get("mode", "lines")
                if "lines" in mode:
                    for (c0, r0), (c1, r1) in itertools.pairwise(points):
                        _ascii_segment(grid, c0, r0, c1, r1)
                if "markers" in mode:
                    for c, r in points:
                        grid[r][c] = "█"
        return "\n".join("".join(cells).rstrip() for cells in grid)


def _ascii_segment(
    grid: list[list[str]], c0: int, r0: int, c1: int, r1: int
) -> None:
    """Mark the cells a straight segment passes through (Bresenham-lite)."""
    steps = max(abs(c1 - c0), abs(r1 - r0))
    if steps == 0:
        grid[r0][c0] = "█"
        return
    for t in range(steps + 1):
        c = round(c0 + (c1 - c0) * t / steps)
        r = round(r0 + (r1 - r0) * t / steps)
        grid[r][c] = "█"


def figure_report(figure: Figure) -> str:
    """Deterministic text summary of a figure (traces, layout, ranges)."""
    lines = [f"PharmaML interactive figure — {len(figure.traces)} trace(s)"]
    for index, trace in enumerate(figure.traces):
        if trace.get("type") == "heatmap":
            z = trace.get("z") or []
            cols = len(z[0]) if z else 0
            lines.append(f"  trace {index}: heatmap {len(z)}x{cols}")
        elif trace.get("type") == "scatter":
            lines.append(
                f"  trace {index}: scatter {trace.get('name')!r} "
                f"mode={trace.get('mode', 'lines')} n={len(trace.get('x') or [])}"
            )
        else:
            lines.append(
                f"  trace {index}: bar {trace.get('name')!r} "
                f"n={len(trace.get('x') or [])}"
            )
    layout = figure.layout
    lines.append(
        f"  layout: title={layout.get('title')!r} "
        f"x_title={layout.get('x_title')!r} y_title={layout.get('y_title')!r} "
        f"x_log={layout.get('x_log')} y_log={layout.get('y_log')}"
    )
    ranges = figure.axis_ranges()
    lines.append(f"  axis ranges: x={ranges['x']} y={ranges['y']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ColumnDataSource — bokeh's streaming/patching semantics
# ---------------------------------------------------------------------------


class ColumnDataSource:
    """A dict of equal-length columns with bokeh's update semantics.

    ``data`` maps column names to lists (row *i* of every column forms
    row *i* of the source).  :meth:`stream` appends rows with an
    optional ``rollover`` that keeps only the newest values (bokeh's
    ring-buffer semantics); :meth:`patch` applies per-column
    ``(index_or_slice, value)`` replacements with length validation;
    :meth:`to_frame` exports to a pandas DataFrame.
    """

    def __init__(self, data: Mapping[str, Sequence[Any]] | None = None) -> None:
        columns = {str(key): list(column) for key, column in (data or {}).items()}
        lengths = {key: len(values) for key, values in columns.items()}
        if lengths and len(set(lengths.values())) > 1:
            detail = ", ".join(f"{key!r} has {n}" for key, n in lengths.items())
            raise DataValidationError(
                f"ColumnDataSource columns must have equal lengths: {detail}"
            )
        self.data: dict[str, list[Any]] = columns

    def patch(self, updates: Mapping[str, Sequence[tuple[Any, Any]]]) -> None:
        """Apply ``(index-or-slice, value)`` replacements per column.

        A slice target must receive exactly as many values as it
        replaces; an integer target must be in range.

        Raises
        ------
        DataValidationError
            On an unknown column, a bad target, a length mismatch, or
            an out-of-range index.
        """
        for key, entries in updates.items():
            if key not in self.data:
                raise DataValidationError(
                    f"patch: column {key!r} not in the source "
                    f"(columns: {sorted(self.data)})"
                )
            column = self.data[key]
            for target, value in entries:
                if isinstance(target, slice):
                    span = len(range(*target.indices(len(column))))
                    values = list(value)
                    if len(values) != span:
                        raise DataValidationError(
                            f"patch for column {key!r}: "
                            f"slice({target.start}, {target.stop}) needs {span} "
                            f"value(s), got {len(values)}"
                        )
                    column[target] = values
                else:
                    try:
                        index = operator.index(target)
                    except TypeError as exc:
                        raise DataValidationError(
                            f"patch for column {key!r}: target must be an index "
                            f"or slice, got {target!r}"
                        ) from exc
                    if not 0 <= index < len(column):
                        raise DataValidationError(
                            f"patch for column {key!r}: index {index} out of "
                            f"range for length {len(column)}"
                        )
                    column[index] = value

    def stream(
        self, new_data: Mapping[str, Sequence[Any]], rollover: int | None = None
    ) -> None:
        """Append rows; with ``rollover``, keep only the last values.

        Raises
        ------
        DataValidationError
            On unknown columns, unequal appended lengths, or a
            non-positive ``rollover``.
        """
        if rollover is not None and (
            isinstance(rollover, bool)
            or not isinstance(rollover, (int, np.integer))
            or int(rollover) < 1
        ):
            raise DataValidationError(
                f"stream: rollover must be a positive integer or None, "
                f"got {rollover!r}"
            )
        lengths = {key: len(list(values)) for key, values in new_data.items()}
        if lengths and len(set(lengths.values())) > 1:
            detail = ", ".join(f"{key!r} has {n}" for key, n in lengths.items())
            raise DataValidationError(
                f"stream: columns must have equal lengths: {detail}"
            )
        for key in new_data:
            if key not in self.data:
                raise DataValidationError(
                    f"stream: column {key!r} not in the source "
                    f"(columns: {sorted(self.data)})"
                )
        for key, values in new_data.items():
            self.data[key].extend(values)
        if rollover is not None:
            for column in self.data.values():
                excess = len(column) - int(rollover)
                if excess > 0:
                    del column[:excess]

    def to_frame(self) -> pd.DataFrame:
        """The source as a DataFrame (columns keep insertion order)."""
        return pd.DataFrame({key: list(values) for key, values in self.data.items()})


# ---------------------------------------------------------------------------
# CallbackRegistry — dash's reactive dependency graph, serverless
# ---------------------------------------------------------------------------


@dataclass
class _Callback:
    """One registered callback: its output, inputs and function."""

    output: tuple[str, str]
    inputs: tuple[tuple[str, str], ...]
    function: Callable[..., Any]


class CallbackRegistry:
    """Dash's ``@app.callback`` dependency graph without the server.

    Callbacks are registered with :meth:`add` (unique outputs, inputs
    as ``(component_id, prop)`` pairs) and fired by
    :meth:`simulate_update`, which stores the new component value and
    runs every callback transitively depending on it in topological
    order — producers before consumers, cycles rejected with
    ``ValueError``.  Component values persist across calls, so a
    callback's untouched inputs keep their last simulated value.
    """

    def __init__(self) -> None:
        self._callbacks: list[_Callback] = []
        self._state: dict[tuple[str, str], Any] = {}

    def add(
        self,
        output_id: str,
        output_prop: str,
        inputs: Sequence[tuple[str, str]] = (),
        function: Callable[..., Any] | None = None,
    ) -> None:
        """Register a callback producing ``output_id.output_prop``.

        Raises
        ------
        DataValidationError
            On a non-callable function, malformed input pairs, or an
            output that is already registered (dash outputs are
            unique).
        """
        if function is None or not callable(function):
            raise DataValidationError(
                f"add: function must be callable, got {function!r}"
            )
        pairs: list[tuple[str, str]] = []
        for pair in inputs:
            try:
                component_id, prop = pair
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"add: inputs must be (component_id, prop) pairs, got {pair!r}"
                ) from exc
            pairs.append((str(component_id), str(prop)))
        output = (str(output_id), str(output_prop))
        if any(callback.output == output for callback in self._callbacks):
            raise DataValidationError(
                f"add: output {output!r} is already registered — dash outputs "
                "are unique"
            )
        self._callbacks.append(
            _Callback(output=output, inputs=tuple(pairs), function=function)
        )

    def simulate_update(
        self, component_id: str, prop: str, value: Any
    ) -> dict[tuple[str, str], Any]:
        """Set one component value and fire the dependent callbacks.

        Returns
        -------
        dict
            ``{(output_id, output_prop): result}`` for every callback
            fired by this update, in execution (topological) order.

        Raises
        ------
        ValueError
            When the dependent callbacks form a dependency cycle.
        DataValidationError
            When a fired callback needs an input that was never given a
            value.
        """
        key = (str(component_id), str(prop))
        self._state[key] = value
        dependent: list[_Callback] = []
        seen: set[int] = set()
        queue = [callback for callback in self._callbacks if key in callback.inputs]
        while queue:
            callback = queue.pop()
            if id(callback) in seen:
                continue
            seen.add(id(callback))
            dependent.append(callback)
            for nxt in self._callbacks:
                if callback.output in nxt.inputs and id(nxt) not in seen:
                    queue.append(nxt)
        if not dependent:
            return {}
        by_id = {id(callback): callback for callback in dependent}
        predecessors: dict[int, set[int]] = {cid: set() for cid in by_id}
        for producer in dependent:
            for consumer in dependent:
                if producer.output in consumer.inputs:
                    predecessors[id(consumer)].add(id(producer))
        sorter = graphlib.TopologicalSorter(predecessors)
        try:
            order = list(sorter.static_order())
        except graphlib.CycleError as exc:
            raise ValueError(
                "cycle detected in the callback dependency graph"
            ) from exc
        results: dict[tuple[str, str], Any] = {}
        for cid in order:
            callback = by_id[cid]
            missing = [k for k in callback.inputs if k not in self._state]
            if missing:
                raise DataValidationError(
                    f"simulate_update: input {missing[0]!r} of output "
                    f"{callback.output!r} has no value — simulate_update it first"
                )
            result = callback.function(
                *(self._state[k] for k in callback.inputs)
            )
            self._state[callback.output] = result
            results[callback.output] = result
        return results


# ---------------------------------------------------------------------------
# Offline HTML export — figure JSON + a pure-Python static SVG
# ---------------------------------------------------------------------------


def _identity(value: float) -> float:
    return value


def _figure_to_svg(figure: Figure, width: float = 640.0, height: float = 400.0) -> str:
    """Static SVG rendering of a figure (pure Python, deterministic).

    Scatter traces draw a polyline (``lines``) and/or circles
    (``markers``); bar traces draw one polygon per bar anchored at the
    zero baseline; heatmaps draw a cell table (grayscale fills, value
    labels when the grid has at most 36 cells).  Log layout flags map
    the corresponding axis through ``log10`` — a non-positive range on
    a log axis raises instead of silently clipping.
    """
    left, right, top, bottom = 56.0, 18.0, 34.0, 42.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    ranges = figure.axis_ranges()
    x0, x1 = ranges["x"] or (0.0, 1.0)
    y0, y1 = ranges["y"] or (0.0, 1.0)
    template = TEMPLATES.get(str(figure.layout.get("template")), {}) if (
        figure.layout.get("template")
    ) else {}
    trace_colors = template.get("colors", _TRACE_COLORS)
    paper = template.get("paper", "#ffffff")
    ink = template.get("ink", "#64748b")
    x_log = bool(figure.layout.get("x_log"))
    y_log = bool(figure.layout.get("y_log"))
    if x_log and x0 <= 0.0:
        raise DataValidationError(
            f"x_log layout asks for a log axis but the x range includes "
            f"non-positive values ({x0})"
        )
    if y_log and y0 <= 0.0:
        raise DataValidationError(
            f"y_log layout asks for a log axis but the y range includes "
            f"non-positive values ({y0})"
        )
    x_map = math.log10 if x_log else _identity
    y_map = math.log10 if y_log else _identity
    x_lo, x_hi = x_map(x0), x_map(x1)
    y_lo, y_hi = y_map(y0), y_map(y1)

    def px(x_value: float) -> float:
        fraction = (x_map(x_value) - x_lo) / (x_hi - x_lo) if x_hi > x_lo else 0.5
        return left + fraction * plot_w

    def py(y_value: float) -> float:
        fraction = (y_map(y_value) - y_lo) / (y_hi - y_lo) if y_hi > y_lo else 0.5
        return top + (1.0 - fraction) * plot_h

    parts: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">'
        ),
        (
            f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" '
            f'fill="{paper}"/>'
        ),
        (
            f'<rect x="{left:.2f}" y="{top:.2f}" width="{plot_w:.2f}" '
            f'height="{plot_h:.2f}" fill="none" stroke="{ink}" '
            f'stroke-width="1"/>'
        ),
    ]
    for i in range(5):  # x ticks (even in transformed space, labels inverted)
        fraction = i / 4.0
        tick_t = x_lo + fraction * (x_hi - x_lo)
        tick_value = 10.0**tick_t if x_log else tick_t
        cx = left + fraction * plot_w
        parts.append(
            f'<line x1="{cx:.2f}" y1="{top + plot_h:.2f}" x2="{cx:.2f}" '
            f'y2="{top + plot_h + 4.0:.2f}" stroke="{ink}"/>'
        )
        parts.append(
            f'<text x="{cx:.2f}" y="{top + plot_h + 16.0:.2f}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="10" '
            f'fill="#334155">{tick_value:.3g}</text>'
        )
    for i in range(5):  # y ticks
        fraction = i / 4.0
        tick_t = y_lo + fraction * (y_hi - y_lo)
        tick_value = 10.0**tick_t if y_log else tick_t
        cy = top + (1.0 - fraction) * plot_h
        parts.append(
            f'<line x1="{left - 4.0:.2f}" y1="{cy:.2f}" x2="{left:.2f}" '
            f'y2="{cy:.2f}" stroke="{ink}"/>'
        )
        parts.append(
            f'<text x="{left - 7.0:.2f}" y="{cy + 3.0:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="10" '
            f'fill="#334155">{tick_value:.3g}</text>'
        )
    if figure.layout.get("title"):
        parts.append(
            f'<text x="{width / 2.0:.2f}" y="20" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14" font-weight="bold" '
            f'fill="#0f172a">{escape(str(figure.layout["title"]))}</text>'
        )
    if figure.layout.get("x_title"):
        parts.append(
            f'<text x="{width / 2.0:.2f}" y="{height - 8.0:.2f}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="12" '
            f'fill="#334155">{escape(str(figure.layout["x_title"]))}</text>'
        )
    if figure.layout.get("y_title"):
        parts.append(
            f'<text transform="translate(14 {top + plot_h / 2.0:.2f}) '
            f'rotate(-90)" text-anchor="middle" font-family="sans-serif" '
            f'font-size="12" '
            f'fill="#334155">{escape(str(figure.layout["y_title"]))}</text>'
        )

    for index, trace in enumerate(figure.traces):
        color = trace_colors[index % len(trace_colors)]
        kind = trace.get("type")
        hovers = trace.get("hovertext") or []
        if kind == "scatter":
            mode = trace.get("mode", "lines")
            xs = trace.get("x") or []
            ys = trace.get("y") or []
            if "lines" in mode and len(xs) >= 2:
                points = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in zip(xs, ys))
                parts.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" '
                    f'stroke-width="2"/>'
                )
            if "markers" in mode:
                for i, (x, y) in enumerate(zip(xs, ys)):
                    title = (
                        f"<title>{escape(str(hovers[i]))}</title>"
                        if i < len(hovers)
                        else ""
                    )
                    parts.append(
                        f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="3" '
                        f'fill="{color}">{title}</circle>'
                    )
        elif kind == "bar":
            positions = _trace_positions(trace)
            pixels = sorted(px(p) for p in positions)
            gaps = [b - a for a, b in itertools.pairwise(pixels)]
            min_gap = min(gaps) if gaps else plot_w
            half = min(0.4 * min_gap, 24.0)
            baseline = 0.0 if y0 <= 0.0 <= y1 else (y0 if y0 > 0.0 else y1)
            for i, (x, y) in enumerate(zip(positions, trace.get("y") or [])):
                cx = px(x)
                y_top = py(float(y))
                y_base = py(baseline)
                title = (
                    f"<title>{escape(str(hovers[i]))}</title>"
                    if i < len(hovers)
                    else ""
                )
                parts.append(
                    f'<polygon points="{cx - half:.2f},{y_base:.2f} '
                    f'{cx + half:.2f},{y_base:.2f} {cx + half:.2f},{y_top:.2f} '
                    f'{cx - half:.2f},{y_top:.2f}" fill="{color}" '
                    f'fill-opacity="0.85">{title}</polygon>'
                )
        elif kind == "heatmap":
            z = trace.get("z") or []
            x_coords = _heatmap_coords(trace, "x")
            y_coords = _heatmap_coords(trace, "y")
            flat = [value for row in z for value in row]
            lo = min(flat) if flat else 0.0
            hi = max(flat) if flat else 0.0
            span = hi - lo
            label_cells = sum(len(row) for row in z) <= 36
            for r, row in enumerate(z):
                for c, value in enumerate(row):
                    xc = x_coords[c] if c < len(x_coords) else float(c)
                    yc = y_coords[r] if r < len(y_coords) else float(r)
                    x_a, x_b = px(xc - 0.5), px(xc + 0.5)
                    y_a, y_b = py(yc + 0.5), py(yc - 0.5)
                    fraction = (value - lo) / span if span > 0 else 0.5
                    shade = 235 - round(fraction * 180.0)
                    fill = f"#{shade:02x}{shade:02x}{shade:02x}"
                    parts.append(
                        f'<rect x="{x_a:.2f}" y="{y_a:.2f}" '
                        f'width="{x_b - x_a:.2f}" height="{y_b - y_a:.2f}" '
                        f'fill="{fill}"/>'
                    )
                    if label_cells:
                        cx = (x_a + x_b) / 2.0
                        cy = (y_a + y_b) / 2.0
                        text_color = "#f8fafc" if fraction > 0.55 else "#1e293b"
                        parts.append(
                            f'<text x="{cx:.2f}" y="{cy + 3.5:.2f}" '
                            f'text-anchor="middle" font-family="sans-serif" '
                            f'font-size="9" fill="{text_color}">{value:.3g}</text>'
                        )
    parts.append("</svg>")
    return "\n".join(parts)


def save_interactive_html(
    figure: Figure | Sequence[Figure],
    path: str | Path,
    title: str = "PharmaML interactive figure",
) -> Path:
    """Write a self-contained offline HTML page for one figure or a list.

    Each figure gets a section with (1) the figure JSON embedded in a
    ``<script type="application/json">`` block — the exact data model a
    plotly/bokeh-style renderer would consume, with ``<`` escaped as
    ``\\u003c`` so no payload can close the tag; (2) a pure-Python
    static SVG rendering; and (3) a ``<pre>`` text report.  The file
    contains no JavaScript and references no CDN, and the bytes are
    deterministic — the same figures always produce the same file.

    Returns
    -------
    Path
        The resolved path written (parent directories are created).

    Raises
    ------
    DataValidationError
        When ``figure`` is not a :class:`Figure` or a list of them, or
        a log-axis layout meets non-positive data.
    """
    figures: list[Figure] = [figure] if isinstance(figure, Figure) else list(figure)
    for item in figures:
        if not isinstance(item, Figure):
            raise DataValidationError(
                f"save_interactive_html: expected Figure or a list of Figures, "
                f"got {type(item).__name__}"
            )
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{escape(str(title))}</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 2em; color: #0f172a; }",
        "section { margin-bottom: 2em; }",
        "pre { background: #f1f5f9; padding: 0.75em; overflow-x: auto; }",
        "svg { max-width: 100%; height: auto; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{escape(str(title))}</h1>",
    ]
    for index, fig in enumerate(figures):
        heading = fig.layout.get("title") or f"Figure {index}"
        parts.extend(
            [
                "<section>",
                f"<h2>{escape(str(heading))}</h2>",
                _figure_to_svg(fig),
                f'<script type="application/json" id="pharmaml-figure-{index}">',
                fig.to_json().replace("<", "\\u003c"),
                "</script>",
                f"<pre>{escape(figure_report(fig))}</pre>",
                "</section>",
            ]
        )
    parts.extend(["</body>", "</html>"])
    html_text = "\n".join(parts) + "\n"
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(html_text)
    return file_path


# --------------------------------------------------------------------------- #
# Express -- the plotly.express layer (round 186)
# --------------------------------------------------------------------------- #
#: The express chart kinds (one trace family each).
EXPRESS_KINDS = ("scatter", "line", "bar", "histogram")

#: Default bin strategy for ``express_histogram``.
_HISTOGRAM_BINS = 24


def _fd_bin_count(values: list[float]) -> int:
    """Freedman-Diaconis bin count (Sturges fallback when IQR collapses)."""
    import numpy as np

    array = np.asarray(values, dtype=float)
    q75, q25 = np.percentile(array, [75.0, 25.0])
    iqr = float(q75 - q25)
    n = array.size
    if iqr <= 0.0:
        return max(int(np.ceil(np.log2(n) + 1.0)), 1)
    width = 2.0 * iqr / max(n ** (1.0 / 3.0), 1e-12)
    if width <= 0.0:
        return max(int(np.ceil(np.log2(n) + 1.0)), 1)
    span = float(array.max() - array.min())
    count = int(np.ceil(span / width)) if span > 0 else 1
    return max(min(count, _HISTOGRAM_BINS), 1)


def _express_groups(
    frame: Any, color: str | None
) -> list[tuple[str | None, list[int]]]:
    """One ``(name, row positions)`` group per color level (or a single
    group when ``color`` is ``None``) -- plotly express's legend split."""
    rows = list(range(len(frame)))
    if color is None:
        return [(None, rows)]
    if color not in frame.columns:
        raise DataValidationError(
            f"express: color column {color!r} not in the frame"
        )
    levels: dict[str | None, list[int]] = {}
    for position, value in zip(rows, frame[color]):
        levels.setdefault(str(value), []).append(position)
    return sorted(levels.items(), key=lambda item: str(item[0]))


def _express_hover(
    frame: Any, hover: Sequence[str] | None, positions: Sequence[int]
) -> list[str] | None:
    """The hover label per row: ``field=value`` joined over the chosen
    columns (plotly's ``hover_data`` rendered as text)."""
    if not hover:
        return None
    for column in hover:
        if column not in frame.columns:
            raise DataValidationError(
                f"express: hover column {column!r} not in the frame"
            )
    labels = []
    for position in positions:
        pieces = [
            f"{column}={frame[column].iloc[position]}" for column in hover
        ]
        labels.append(" · ".join(pieces))
    return labels


def express_figure(
    frame: Any,
    kind: str,
    x: str,
    y: str | None = None,
    *,
    color: str | None = None,
    hover: Sequence[str] | None = None,
    labels: dict[str, str] | None = None,
    template: str = "plotly",
    title: str | None = None,
) -> Figure:
    """The ``plotly.express`` one-liner over the figure-as-data model.

    ``kind`` picks the trace family (``scatter`` / ``line`` / ``bar`` /
    ``histogram``); ``color`` splits the rows into one trace per level
    (the legend semantics); ``hover`` lists DataFrame columns whose
    values ride on the points as ``hovertext``; ``labels`` renames the
    axis titles; ``template`` picks one of :data:`TEMPLATES`.
    ``histogram`` bins ``x`` server-side (Freedman-Diaconis) into a bar
    trace -- the aggregation ``px.histogram`` does in JS, done here in
    numpy so the data model stays complete offline.
    """
    if kind not in EXPRESS_KINDS:
        raise DataValidationError(
            f"express: kind must be one of {EXPRESS_KINDS}, got {kind!r}"
        )
    if template not in TEMPLATES:
        raise DataValidationError(
            f"express: template must be one of {tuple(TEMPLATES)}, got {template!r}"
        )
    if x not in frame.columns:
        raise DataValidationError(f"express: x column {x!r} not in the frame")
    if y is not None and y not in frame.columns:
        raise DataValidationError(f"express: y column {y!r} not in the frame")
    if len(frame) == 0:
        raise DataValidationError("express: the frame is empty")
    renames = dict(labels or {})

    figure = Figure()
    if kind == "histogram":
        if y is not None:
            raise DataValidationError(
                "express: histogram takes only x (the value column to bin)"
            )
        values = [float(v) for v in frame[x]]
        n_bins = _fd_bin_count(values)
        lo, hi = min(values), max(values)
        if hi <= lo:
            hi = lo + 1.0
        import numpy as np

        edges = np.linspace(lo, hi, n_bins + 1)
        index = np.clip(
            ((np.asarray(values) - lo) / (hi - lo) * n_bins).astype(int),
            0,
            n_bins - 1,
        )
        counts = np.bincount(index, minlength=n_bins)
        centers = [
            float((edges[i] + edges[i + 1]) / 2.0) for i in range(n_bins)
        ]
        hovers = _express_hover(frame, hover, list(range(len(frame))))
        if hovers is not None:
            by_bin: list[list[str]] = [[] for _ in range(n_bins)]
            for label, slot in zip(hovers, index.tolist()):
                by_bin[slot].append(label)
            hovers = ["; ".join(items) for items in by_bin]
        figure.add_bar(
            centers, [int(c) for c in counts], name=x, hovertext=hovers
        )
        figure.set_layout(
            title=title or renames.get("title"),
            x_title=renames.get("x", x),
            y_title=renames.get("y", "count"),
            template=template,
        )
        return figure

    if y is None:
        raise DataValidationError(
            f"express: kind {kind!r} needs a y column"
        )
    for level, positions in _express_groups(frame, color):
        ys = [float(v) for v in frame[y].iloc[positions]]
        name = f"{color}={level}" if level is not None else f"{y} vs {x}"
        hovers = _express_hover(frame, hover, positions)
        if kind == "bar":
            bar_xs = [str(v) for v in frame[x].iloc[positions]]
            figure.add_bar(bar_xs, ys, name=name, hovertext=hovers)
        else:
            scatter_xs = [float(v) for v in frame[x].iloc[positions]]
            mode = "markers" if kind == "scatter" else "lines"
            figure.add_scatter(scatter_xs, ys, name=name, mode=mode, hovertext=hovers)
    figure.set_layout(
        title=title or renames.get("title"),
        x_title=renames.get("x", x),
        y_title=renames.get("y", y),
        template=template,
    )
    return figure


def express_scatter(
    frame: Any, x: str, y: str, **kwargs: Any
) -> Figure:
    """``px.scatter`` -- one marker trace per colour level."""
    return express_figure(frame, "scatter", x, y, **kwargs)


def express_line(frame: Any, x: str, y: str, **kwargs: Any) -> Figure:
    """``px.line`` -- one line trace per colour level."""
    return express_figure(frame, "line", x, y, **kwargs)


def express_bar(frame: Any, x: str, y: str, **kwargs: Any) -> Figure:
    """``px.bar`` -- one bar trace per colour level."""
    return express_figure(frame, "bar", x, y, **kwargs)


def express_histogram(frame: Any, x: str, **kwargs: Any) -> Figure:
    """``px.histogram`` -- Freedman-Diaconis bins as one bar trace."""
    return express_figure(frame, "histogram", x, **kwargs)


def apply_template(figure: Figure, name: str) -> Figure:
    """Switch a figure's template in place (plotly's named themes)."""
    if name not in TEMPLATES:
        raise DataValidationError(
            f"template must be one of {tuple(TEMPLATES)}, got {name!r}"
        )
    figure.layout["template"] = name
    return figure
