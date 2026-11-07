"""plotly.py's figure model and express transforms, plus bokeh's data
transforms, absorbed natively (100x-170).

The two spec-layer repositories were reviewed file by file; what lands
is everything that is *data and structure* rather than JavaScript:

From **plotly.py** (``basedatatypes.py``, ``_subplots.py``,
``express/_core.py``):

* :class:`Trace` / :class:`Figure` -- the ``data``/``layout`` figure
  spec with ``to_dict()`` and strict ``to_json()`` (NaN becomes null,
  NumPy arrays become lists -- plotly's ``PlotlyJSONEncoder``
  contract, minus the typed-array base64 encoding, which exists purely
  for plotly.js bandwidth);
* :func:`make_subplots` -- the subplot grid with plotly's exact axis
  naming (first subplot gets ``xaxis``/``yaxis`` referenced as
  ``"x"``/``"y"``, the n-th gets ``xaxis{n}`` referenced as
  ``"x{n}"``, numbered in creation order), domain boxes from the
  spacing defaults (``0.2/cols`` horizontal, ``0.3/rows`` vertical),
  ``matches`` sharing, and :meth:`Figure.add_trace(row, col)` stamping
  via the grid reference -- ``_set_trace_grid_reference`` semantics;
* the express transforms -- :func:`scatter_traces` /
  :func:`bar_traces` / :func:`histogram_traces` /
  :func:`heatmap_trace` with plotly express's grouping contract: one
  trace per crossed discrete mapping variable, each variable cycling
  its own value map (``sequence[len(map) % len(sequence)]``), the
  discrete/continuous color split on dtype (numeric columns go to a
  single ``marker.color`` array with a colorscale instead of
  grouping), ``sizeref = max(size)/size_max**2`` with
  ``sizemode="area"``, and the ``legendgroup``/``showlegend``
  de-duplication.

From **bokeh** (``transform.py``, evaluated for real instead of as
spec objects): :func:`cumsum` (with the ``include_zero`` length rule
-- ``[0, 1, 3, 6]`` for ``[1, 2, 3]``), :func:`stack` (the
elementwise multi-column sum behind stacked bars), :func:`factor_cmap`
cycling (re-exported from :mod:`pharmaml.viz.colors`),
:func:`dodge`/:func:`jitter` (uniform/normal scatter in data units),
and :func:`pie_angles` -- the canonical ``start = cumsum(angle,
include_zero=True)`` / ``end = cumsum(angle)`` wedge recipe.

Rendering stays external by design: a Figure is a renderer-agnostic
JSON contract (the playground's Canvas cards draw from it; plotly.js,
BokehJS and matplotlib can all consume the same spec).
:func:`render_matplotlib` bridges into the existing optional-matplotlib
figure layer when the ``reports`` extra is installed.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from pharmaml.viz.figures import (
...     Trace, bar_traces, cumsum, dodge, heatmap_trace, histogram_traces,
...     jitter, make_subplots, pie_angles, scatter_traces, stack,
... )
>>> fig = make_subplots(rows=2, cols=2, subplot_titles=["a", "b", "c", "d"])
>>> _trace = fig.add_trace(Trace(x=[1.0, 2.0], y=[3.0, 4.0], name="line"), row=2, col=1)
>>> [t["xaxis"] for t in fig.to_dict()["data"]]
['x3']
>>> sorted(fig.to_dict()["layout"].keys())
['annotations', 'template', 'xaxis', 'xaxis2', 'xaxis3', 'xaxis4', 'yaxis', 'yaxis2', 'yaxis3', 'yaxis4']
>>> frame = pd.DataFrame({
...     "x": np.arange(8.0),
...     "y": np.arange(8.0) ** 1.5,
...     "group": ["a", "b"] * 4,
... })
>>> traces = scatter_traces(frame, "x", "y", color="group",
...     color_discrete_sequence=("#111111", "#222222"))
>>> [(t.name, t.marker["color"]) for t in traces]
[('a', '#111111'), ('b', '#222222')]
>>> cumsum([1.0, 2.0, 3.0], include_zero=True).tolist()
[0.0, 1.0, 3.0]
>>> cumsum([1.0, 2.0, 3.0]).tolist()
[1.0, 3.0, 6.0]
>>> stack([1.0, 2.0], [10.0, 20.0]).tolist()
[11.0, 22.0]
>>> starts, ends = pie_angles([3.0, 1.0])
>>> round(float(ends[-1]), 6)
6.283185
>>> dodge([1.0, 2.0, 3.0], 0.25).tolist()
[1.25, 2.25, 3.25]
>>> jitter([0.0] * 3, width=0.5, seed=42).round(3).tolist()
[0.274, -0.061, 0.359]
>>> counts = bar_traces(pd.DataFrame({"kind": ["a", "a", "b"]}), "kind")
>>> counts[0].y
[2, 1]
>>> hist = histogram_traces(pd.DataFrame({"v": np.arange(12.0)}), "v", nbins=4)
>>> [int(c) for c in hist[0].y]
[3, 3, 3, 3]
>>> hm = heatmap_trace([[1.0, 2.0], [3.0, 4.0]])
>>> hm.z
[[1.0, 2.0], [3.0, 4.0]]
>>> hm.colorscale[0]
[0.0, '#440154']
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.viz.colors import (
    D3,
    PERCEPTUAL_CMAPS,
    factor_cmap,
    make_colorscale,
    to_hex,
)
from pharmaml.viz.statplots import histogram_stats

__all__ = [
    "Figure",
    "SubplotRef",
    "Trace",
    "bar_traces",
    "cumsum",
    "dodge",
    "factor_cmap",
    "heatmap_trace",
    "histogram_traces",
    "jitter",
    "make_subplots",
    "pie_angles",
    "render_matplotlib",
    "scatter_traces",
    "stack",
]


def _json_safe(obj: Any) -> Any:
    """plotly's encoder contract: NaN/inf -> null, arrays -> lists."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


@dataclass
class Trace:
    """One plotly trace: ``type`` + data + style props, dict-shaped.

    ``to_dict()`` emits exactly the keys that were set (plotly omits
    unset props rather than nulling them), with NumPy-safe JSON
    values; the ``type`` stamp lands last like the generated classes
    do.
    """

    x: Any = None
    y: Any = None
    z: Any = None
    name: str | None = None
    type: str = "scatter"
    mode: str | None = None
    text: Any = None
    orientation: str | None = None
    opacity: float | None = None
    marker: dict[str, Any] | None = None
    line: dict[str, Any] | None = None
    error_y: dict[str, Any] | None = None
    showscale: bool | None = None
    colorscale: Any = None
    xaxis: str | None = None
    yaxis: str | None = None
    legendgroup: str | None = None
    showlegend: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        bits = [f"type={self.type!r}"]
        if self.name:
            bits.append(f"name={self.name!r}")
        if self.xaxis:
            bits.append(f"xaxis={self.xaxis!r}")
            bits.append(f"yaxis={self.yaxis!r}")
        return f"Trace({', '.join(bits)})"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in (
            "x", "y", "z", "name", "mode", "text", "orientation", "opacity",
            "marker", "line", "error_y", "showscale", "colorscale", "xaxis",
            "yaxis", "legendgroup", "showlegend",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = _json_safe(value)
        out.update(_json_safe(self.extra))
        out["type"] = self.type
        return out


@dataclass(frozen=True)
class SubplotRef:
    """plotly's ``_subplots.SubplotRef``: where a grid cell's axes live."""

    layout_keys: tuple[str, str]
    trace_kwargs: dict[str, str]


class Figure:
    """The ``data``/``layout`` figure spec (plotly's ``go.Figure``).

    ``add_trace(trace, row, col)`` stamps subplot references onto the
    trace; ``update_layout`` accepts plotly's dotted-path kwargs
    (``xaxis_title="X"`` -> ``layout.xaxis.title``); ``to_dict`` /
    ``to_json`` are the renderer-agnostic serialization contract.
    """

    def __init__(self, data: Any = None, layout: dict[str, Any] | None = None) -> None:
        self.data: list[Trace] = []
        if data is not None:
            for item in data:
                self.data.append(item if isinstance(item, Trace) else Trace(**item))
        self.layout: dict[str, Any] = dict(layout or {})
        self._grid_ref: list[list[list[SubplotRef]]] | None = None

    def add_trace(self, trace: Trace, row: int | None = None, col: int | None = None) -> Trace:
        """Add a trace, optionally stamped onto a subplot grid cell."""
        if (row is None) != (col is None):
            raise DataValidationError("row and col must be given together")
        if row is not None and col is not None and self._grid_ref is not None:
            refs = self._grid_ref[row - 1][col - 1]
            trace.xaxis = refs[0].trace_kwargs["xaxis"]
            trace.yaxis = refs[0].trace_kwargs["yaxis"]
        self.data.append(trace)
        return trace

    def update_layout(self, **kwargs: Any) -> None:
        """Merge into ``layout``; ``xaxis_title`` -> ``layout.xaxis.title``."""
        for key, value in kwargs.items():
            path = key.split("_", 1)
            if len(path) == 2 and path[0] in ("xaxis", "yaxis", "coloraxis", "legend", "title"):
                self.layout.setdefault(path[0], {})[path[1]] = _json_safe(value)
            else:
                self.layout[key] = _json_safe(value)

    def to_dict(self) -> dict[str, Any]:
        return {"data": [t.to_dict() for t in self.data], "layout": _json_safe(self.layout)}

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Figure({len(self.data)} traces)"


def make_subplots(
    rows: int = 1,
    cols: int = 1,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
    horizontal_spacing: float | None = None,
    vertical_spacing: float | None = None,
    subplot_titles: list[str] | None = None,
) -> Figure:
    """plotly's ``make_subplots``: the axis grid, plotly's conventions.

    Axis naming follows ``_init_subplot_xy`` exactly (creation order,
    ``xaxis{n}`` referenced as ``"x{n}"`` with the first unstamped);
    domains come from the default spacings ``0.2/cols`` and
    ``0.3/rows``; sharing stamps ``matches`` plus hidden tick labels
    (per column for x, per row for y -- plotly's ``shared_xaxes``
    semantics); ``subplot_titles`` become paper-referenced annotations.
    """
    if rows < 1 or cols < 1:
        raise DataValidationError("make_subplots needs rows >= 1 and cols >= 1")
    h_space = 0.2 / cols if horizontal_spacing is None else float(horizontal_spacing)
    v_space = 0.3 / rows if vertical_spacing is None else float(vertical_spacing)
    total_w = 1.0 - h_space * (cols - 1)
    total_h = 1.0 - v_space * (rows - 1)
    cell_w = total_w / cols
    cell_h = total_h / rows
    figure = Figure(layout={"template": "plotly"})
    grid: list[list[list[SubplotRef]]] = []
    x_count = y_count = 0
    first_x_per_col: dict[int, str] = {}
    first_y_per_row: dict[int, str] = {}
    annotations: list[dict[str, Any]] = []
    for r_top in range(rows):
        grid_row: list[list[SubplotRef]] = []
        r_bottom = rows - 1 - r_top  # start_cell="top-left": rows walk downward
        for c in range(cols):
            x_count += 1
            y_count += 1
            x_label = f"x{x_count}" if x_count > 1 else "x"
            y_label = f"y{y_count}" if y_count > 1 else "y"
            x_key = f"xaxis{x_count}" if x_count > 1 else "xaxis"
            y_key = f"yaxis{y_count}" if y_count > 1 else "yaxis"
            x_domain = [c * (cell_w + h_space), c * (cell_w + h_space) + cell_w]
            y_domain = [r_bottom * (cell_h + v_space), r_bottom * (cell_h + v_space) + cell_h]
            x_axis: dict[str, Any] = {"domain": x_domain, "anchor": y_label}
            y_axis: dict[str, Any] = {"domain": y_domain, "anchor": x_label}
            first_x_per_col.setdefault(c, x_label)
            first_y_per_row.setdefault(r_bottom, y_label)
            if shared_xaxes and x_label != first_x_per_col[c]:
                x_axis["matches"] = first_x_per_col[c]
                x_axis["showticklabels"] = False
            if shared_yaxes and y_label != first_y_per_row[r_bottom]:
                y_axis["matches"] = first_y_per_row[r_bottom]
                y_axis["showticklabels"] = False
            figure.layout[x_key] = x_axis
            figure.layout[y_key] = y_axis
            grid_row.append(
                [SubplotRef(layout_keys=(x_key, y_key),
                            trace_kwargs={"xaxis": x_label, "yaxis": y_label})]
            )
            if subplot_titles is not None and (r_top * cols + c) < len(subplot_titles):
                annotations.append(
                    {
                        "text": subplot_titles[r_top * cols + c],
                        "xref": "paper", "yref": "paper",
                        "x": sum(x_domain) / 2, "y": sum(y_domain) + 0.03,
                        "showarrow": False,
                    }
                )
        grid.append(grid_row)
    if annotations:
        figure.layout["annotations"] = annotations
    figure._grid_ref = grid
    return figure


# ---------------------------------------------------------------------------
# plotly express transforms (express/_core.py semantics)
# ---------------------------------------------------------------------------


def _is_continuous(series: pd.Series) -> bool:
    return series.dtype.kind in "iufc"


def _cycled_map(values: list[Any], sequence: list[str]) -> dict[Any, str]:
    """plotly's ``val_map[val] = sequence[len(map) % len(sequence)]``."""
    mapping: dict[Any, str] = {}
    for value in values:
        if value not in mapping:
            mapping[value] = sequence[len(mapping) % len(sequence)]
    return mapping


def scatter_traces(
    data_frame: Any,
    x: str,
    y: str,
    color: str | None = None,
    size: str | None = None,
    symbol: str | None = None,
    color_discrete_sequence: Any = None,
    color_continuous_scale: Any = None,
    size_max: float = 20.0,
    category_orders: dict[str, list[Any]] | None = None,
) -> list[Trace]:
    """plotly express ``px.scatter``'s grouping and mapping semantics.

    Discrete mapping variables (object/categorical dtypes) cross into
    one trace per combination, each variable cycling its own value map
    independently; a *numeric* ``color`` column instead becomes a
    single trace with ``marker.color`` values plus a colorscale and
    colorbar (``showscale``); ``size`` maps through
    ``sizeref = max(size)/size_max**2`` with ``sizemode="area"``.
    ``legendgroup``/``showlegend`` de-duplicate exactly as px does.
    """
    frame = pd.DataFrame(data_frame)
    discrete_sequence = (
        [to_hex(c) for c in color_discrete_sequence]
        if color_discrete_sequence is not None else list(D3)
    )
    symbol_sequence = ["circle", "diamond", "square", "x", "cross"]
    orders = category_orders or {}
    group_cols = [
        c for c in (color, symbol) if c is not None and not _is_continuous(frame[c])
    ]

    def ordered_values(column: str) -> list[Any]:
        return list(orders.get(column, pd.unique(frame[column])))

    def size_marker(sub: pd.DataFrame) -> dict[str, Any]:
        if size is None:
            return {}
        max_size = float(frame[size].max())
        return {
            "size": sub[size].to_numpy(dtype=float),
            "sizemode": "area",
            "sizeref": max_size / float(size_max) ** 2 if max_size > 0 else 1.0,
        }

    if group_cols:
        color_map: dict[Any, str] = {}
        if color is not None and color in group_cols:
            color_map = _cycled_map(ordered_values(color), discrete_sequence)
        symbol_map: dict[Any, str] = {}
        if symbol is not None and symbol in group_cols:
            symbol_map = _cycled_map(ordered_values(symbol), symbol_sequence)
        traces: list[Trace] = []
        seen_names: set[str] = set()
        if len(group_cols) == 1:
            combos: list[tuple[Any, ...]] = [(v,) for v in ordered_values(group_cols[0])]
        else:
            assert color is not None and symbol is not None
            combos = [
                (cv, sv)
                for cv in ordered_values(color)
                for sv in ordered_values(symbol)
            ]
        for combo in combos:
            mask = pd.Series(True, index=frame.index)
            for column, value in zip(group_cols, combo):
                mask &= frame[column] == value
            sub = frame[mask]
            if sub.empty:
                continue
            trace_name = ", ".join(str(v) for v in combo)
            marker: dict[str, Any] = {}
            if color is not None and color in group_cols:
                marker["color"] = color_map.get(combo[group_cols.index(color)], discrete_sequence[0])
            if symbol is not None and symbol in group_cols:
                marker["symbol"] = symbol_map.get(
                    combo[group_cols.index(symbol)], symbol_sequence[0]
                )
            marker.update(size_marker(sub))
            traces.append(
                Trace(
                    x=sub[x].to_numpy(), y=sub[y].to_numpy(), name=trace_name,
                    mode="markers", marker=marker or None,
                    legendgroup=trace_name,
                    showlegend=trace_name not in seen_names,
                )
            )
            seen_names.add(trace_name)
        return traces

    marker = {}
    scale_colors: Any = None
    if color is not None:
        if _is_continuous(frame[color]):
            if isinstance(color_continuous_scale, (list, tuple)):
                hexes = [to_hex(c) for c in color_continuous_scale]
            else:
                scale_name = color_continuous_scale or "viridis"
                if scale_name not in PERCEPTUAL_CMAPS:
                    raise DataValidationError(
                        f"unknown colorscale {scale_name!r}; embedded: "
                        f"{', '.join(PERCEPTUAL_CMAPS)} -- or pass a color list"
                    )
                hexes = list(PERCEPTUAL_CMAPS[scale_name])
            scale_colors = make_colorscale(hexes)
            marker["color"] = frame[color].to_numpy(dtype=float)
            marker["colorscale"] = scale_colors
        else:
            mapping = _cycled_map(ordered_values(color), discrete_sequence)
            marker["color"] = [mapping[v] for v in frame[color]]
    marker.update(size_marker(frame))
    return [
        Trace(
            x=frame[x].to_numpy(), y=frame[y].to_numpy(), mode="markers",
            marker=marker or None, showscale=bool(scale_colors),
        )
    ]


def bar_traces(
    data_frame: Any,
    x: str,
    y: str | None = None,
    color: str | None = None,
    color_discrete_sequence: Any = None,
    orientation: str | None = None,
) -> list[Trace]:
    """plotly express ``px.bar``'s trace split (counts when ``y`` is None).

    One trace per discrete ``color`` level (cycling the sequence);
    without ``color`` a single trace.  Stacking is a layout concern
    (``layout.barmode = "stack"``) exactly as in plotly.
    """
    frame = pd.DataFrame(data_frame)
    if y is None:
        counts = frame[x].value_counts(sort=False)
        work = pd.DataFrame({x: counts.index.astype(str), "count": counts.to_numpy()})
        y_col = "count"
    else:
        work = frame
        y_col = y
    discrete_sequence = (
        [to_hex(c) for c in color_discrete_sequence]
        if color_discrete_sequence is not None else list(D3)
    )
    if color is None:
        return [
            Trace(
                x=work[x].astype(str).tolist(), y=work[y_col].tolist(),
                type="bar", orientation=orientation, name="count",
            )
        ]
    mapping = _cycled_map(list(pd.unique(frame[color])), discrete_sequence)
    traces: list[Trace] = []
    for value in pd.unique(frame[color]):
        sub = frame[frame[color] == value] if color in frame else work
        if y is None:
            sub_counts = sub[x].value_counts(sort=False)
            traces.append(
                Trace(
                    x=sub_counts.index.astype(str).tolist(), y=sub_counts.tolist(),
                    type="bar", name=str(value), legendgroup=str(value),
                    marker={"color": mapping[value]},
                )
            )
        else:
            traces.append(
                Trace(
                    x=sub[x].astype(str).tolist(), y=sub[y_col].tolist(),
                    type="bar", name=str(value), legendgroup=str(value),
                    marker={"color": mapping[value]},
                )
            )
    return traces


def histogram_traces(
    data_frame: Any,
    x: str,
    color: str | None = None,
    nbins: int = 10,
    color_discrete_sequence: Any = None,
) -> list[Trace]:
    """plotly express ``px.histogram``: binned counts as bar traces.

    Bins follow seaborn/numpy rules via :func:`histogram_stats` (the
    toolkit's own binning layer); one trace per discrete color level.
    """
    frame = pd.DataFrame(data_frame)
    discrete_sequence = (
        [to_hex(c) for c in color_discrete_sequence]
        if color_discrete_sequence is not None else list(D3)
    )

    def one_trace(sub: pd.DataFrame, name: str | None, fill: str | None) -> Trace:
        hist = histogram_stats(sub[x].to_numpy(dtype=float), bins=int(nbins))
        return Trace(
            x=hist["centers"].tolist(), y=hist["counts"].tolist(), type="bar",
            name=name, marker={"color": fill} if fill else None,
            legendgroup=name, showlegend=bool(name),
        )

    if color is None:
        return [one_trace(frame, None, None)]
    mapping = _cycled_map(list(pd.unique(frame[color])), discrete_sequence)
    out: list[Trace] = []
    for value in pd.unique(frame[color]):
        sub = frame[frame[color] == value]
        out.append(one_trace(sub, str(value), mapping[value]))
    return out


def heatmap_trace(
    matrix: Any,
    colorscale: Any = "viridis",
    text: Any = None,
    showscale: bool = True,
) -> Trace:
    """plotly's heatmap trace: ``z`` + colorscale (built from our tables)."""
    z = np.asarray(matrix, dtype=float)
    if z.ndim != 2:
        raise DataValidationError("heatmap_trace needs a 2-D matrix")
    if isinstance(colorscale, str):
        hexes = list(PERCEPTUAL_CMAPS.get(colorscale, ()))
        if not hexes:
            raise DataValidationError(
                f"unknown colorscale {colorscale!r}; embedded: "
                f"{', '.join(PERCEPTUAL_CMAPS)} or pass a color list"
            )
        scale = make_colorscale(hexes)
    else:
        scale = make_colorscale([to_hex(c) for c in colorscale])
    trace = Trace(type="heatmap", z=z.tolist(), colorscale=scale, showscale=showscale)
    if text is not None:
        trace.extra["text"] = _json_safe(text)
    return trace


# ---------------------------------------------------------------------------
# bokeh transforms (transform.py, evaluated for real)
# ---------------------------------------------------------------------------


def cumsum(values: Any, include_zero: bool = False) -> np.ndarray:
    """bokeh's ``CumSum`` expression: cumulative sum of one column.

    With ``include_zero=True`` the result is prefixed with 0 and the
    running total drops (length preserved) -- the exact rule bokeh's
    pie recipe relies on: ``start = cumsum(angle, include_zero=True)``
    and ``end = cumsum(angle)``.
    """
    arr = np.asarray(values, dtype=float).ravel()
    running = np.cumsum(arr)
    if include_zero:
        return np.concatenate([[0.0], running[:-1]])
    return running


def pie_angles(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Wedge ``(start, end)`` angles in radians for a pie of ``values``.

    The canonical bokeh recipe: fractions of the total times ``2π``,
    ``start = cumsum(fraction, include_zero=True)``,
    ``end = cumsum(fraction)``.
    """
    arr = np.asarray(values, dtype=float).ravel()
    total = float(arr.sum())
    if total <= 0:
        raise DataValidationError("pie_angles needs a positive total")
    fractions = arr / total * 2.0 * math.pi
    return cumsum(fractions, include_zero=True), cumsum(fractions)


def stack(*columns: Any) -> np.ndarray:
    """bokeh's ``Stack`` expression: elementwise sum across columns.

    The stacking arithmetic behind ``vbar_stack`` bottoms/tops (NOT a
    cumulative sum along one column -- bokeh's ``Stack`` adds the
    listed columns position-wise).
    """
    if not columns:
        raise DataValidationError("stack needs at least one column")
    arrays = [np.asarray(c, dtype=float).ravel() for c in columns]
    length = arrays[0].shape[0]
    for a in arrays:
        if a.shape[0] != length:
            raise DataValidationError("stack columns must have equal length")
    out = np.zeros(length, dtype=float)
    for a in arrays:
        out += a
    return out


def dodge(values: Any, value: float) -> np.ndarray:
    """bokeh's ``Dodge`` transform: shift by a fixed data-space offset."""
    return np.asarray(values, dtype=float) + float(value)


def jitter(
    values: Any,
    width: float,
    mean: float = 0.0,
    distribution: str = "uniform",
    seed: Any = None,
) -> np.ndarray:
    """bokeh's ``Jitter`` transform: random scatter around each value.

    ``width`` is the half-width for ``"uniform"`` and sigma for
    ``"normal"`` (bokeh's own parameterization).
    """
    if distribution not in ("uniform", "normal"):
        raise DataValidationError("distribution must be 'uniform' or 'normal'")
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        noise = rng.uniform(mean - float(width), mean + float(width), arr.shape)
    else:
        noise = rng.normal(mean, float(width), arr.shape)
    return arr + noise


# ---------------------------------------------------------------------------
# optional matplotlib bridge
# ---------------------------------------------------------------------------


def render_matplotlib(figure: Figure, path: Any = None) -> Any:
    """Draw a Figure spec through the toolkit's matplotlib layer.

    Single-panel specs retain their classic 8×6 canvas.  ``make_subplots``
    specs render onto their declared cells with subplot titles, axis labels
    and the overall figure title intact.  Scatter, line, bar and heat-map
    traces share the house style; unused cells are hidden and empty legends
    are never emitted.  Needs the ``reports`` extra, like every other
    plotting function in this package.
    """
    from pharmaml.viz._compat import get_pyplot, new_figure, save_figure

    def text(value: Any) -> str:
        if isinstance(value, dict):
            return str(value.get("text", ""))
        return "" if value is None else str(value)

    grid = figure._grid_ref
    rows = len(grid) if grid else 1
    columns = max((len(row) for row in grid), default=1) if grid else 1
    plt = get_pyplot()
    fig, raw_axes = new_figure(
        rows,
        columns,
        figsize=(max(8.0, 4.8 * columns), max(6.0, 3.8 * rows)),
        squeeze=False,
    )
    axes = np.asarray(raw_axes)
    references: dict[str, tuple[int, int]] = {}
    layout_keys: dict[tuple[int, int], tuple[str, str]] = {}
    if grid:
        for row_index, row in enumerate(grid):
            for column_index, grid_cell in enumerate(row):
                if not grid_cell:
                    continue
                reference = grid_cell[0]
                references[reference.trace_kwargs["xaxis"]] = (row_index, column_index)
                layout_keys[(row_index, column_index)] = reference.layout_keys

    used: set[tuple[int, int]] = set()
    labelled: set[tuple[int, int]] = set()
    for trace in figure.data:
        spec = trace.to_dict()
        row_index, column_index = references.get(spec.get("xaxis") or "x", (0, 0))
        cell_key = (row_index, column_index)
        used.add(cell_key)
        ax = axes[row_index, column_index]
        label = spec.get("name") if spec.get("showlegend", True) else None
        if label:
            labelled.add(cell_key)
        if spec.get("type") == "bar":
            color = (spec.get("marker") or {}).get("color")
            if spec.get("orientation") == "h":
                ax.barh(spec.get("y", []), spec.get("x", []), color=color, label=label)
            else:
                ax.bar(spec.get("x", []), spec.get("y", []), color=color, label=label)
        elif spec.get("type") == "heatmap":
            z = np.asarray(spec["z"], dtype=float)
            spec.get("colorscale") or make_colorscale(PERCEPTUAL_CMAPS["viridis"])
            ax.grid(False)
            image = ax.imshow(z, aspect="auto", cmap="viridis")
            if spec.get("showscale", True):
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        else:
            x = np.asarray(spec.get("x") or [], dtype=float)
            y = np.asarray(spec.get("y") or [], dtype=float)
            mode = str(spec.get("mode") or "markers")
            line = spec.get("line") or {}
            if "lines" in mode:
                ax.plot(
                    x,
                    y,
                    color=line.get("color"),
                    linewidth=line.get("width", 1.8),
                    linestyle="--" if line.get("dash") in {"dash", "dot"} else "-",
                    label=label,
                )
            if "markers" in mode:
                color = (spec.get("marker") or {}).get("color") or line.get("color")
                sizes = (spec.get("marker") or {}).get("size")
                if isinstance(color, (list, np.ndarray)) and color and isinstance(color[0], float):
                    color = None  # continuous color arrays ride the colorscale
                if sizes is not None and isinstance(sizes, (list, np.ndarray)):
                    sizes = np.sqrt(np.asarray(sizes, dtype=float))
                ax.scatter(x, y, c=color, s=sizes, label=label)

    annotations = figure.layout.get("annotations", [])
    for row_index in range(rows):
        for column_index in range(columns):
            cell_key = (row_index, column_index)
            ax = axes[row_index, column_index]
            if cell_key not in used:
                ax.set_visible(False)
                continue
            x_key, y_key = layout_keys.get(cell_key, ("xaxis", "yaxis"))
            ax.set_xlabel(text(figure.layout.get(x_key, {}).get("title")))
            ax.set_ylabel(text(figure.layout.get(y_key, {}).get("title")))
            annotation_index = row_index * columns + column_index
            if annotation_index < len(annotations):
                ax.set_title(text(annotations[annotation_index]))
            if cell_key in labelled:
                ax.legend(loc="best", frameon=False)

    title = text(figure.layout.get("title"))
    if title:
        fig.suptitle(title, fontsize=14, fontweight="semibold")
    if path is not None:
        save_figure(fig, path)
        plt.close(fig)
        return path
    return fig
