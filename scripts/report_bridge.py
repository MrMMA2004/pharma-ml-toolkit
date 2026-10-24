#!/usr/bin/env python3
"""Stdin -> stdout JSON bridge for the Next.js ``/api/report/session`` route.

Runs inside the toolkit virtualenv (real matplotlib + pharmaml) and wires
ONLY the public report surface the web twin needs:

    pharmaml.viz.HTMLReport + PALETTE + REPORT_THEMES

The web playground collects its session state client-side (last ML
training result, learning-curve run, PK/PD regimen, NCA rows, run
history) as a section payload; this bridge composes REAL matplotlib
figures from the point arrays (the same fill-between band semantics the
library's own plots use) and assembles them through HTMLReport — the
exact self-contained themed builder the desktop playground's
``File ▸ Save session report (HTML)…`` writes, theme toggle included.

Request (stdin, JSON)::

    {
      "theme": "auto" | "light" | "dark",
      "subtitle": "…",
      "sections": [
        {"kind": "metrics", "title": "ML training",
         "items": [["model", "ridge"], ["R²", 0.87], ...]},
        {"kind": "figure", "title": "Predicted vs observed",
         "xLabel": "observed", "yLabel": "predicted",
         "traces": [
           {"name": "model prediction", "style": "line",
            "points": [[x, y], ...]},
           {"name": "training score", "style": "band",
            "points": [[x, mean, lo, hi], ...]},
           {"name": "test observations", "style": "scatter",
            "points": [[x, y], ...],
            "classes": [0, 1, ...],        # optional color split
            "probas": [0.12, 0.98, ...]}   # optional opacity fade
         ]},
        {"kind": "table", "title": "NCA observed data",
         "columns": ["#", "time (h)", ...],
         "rows": [[1, "0.5", ...], ...]}
      ]
    }

Response (stdout, JSON)::

    {"html": "<!DOCTYPE html>…", "figures": 2, "tables": 1,
     "bytes": 412311, "theme": "auto"}

Errors print a one-line ``{"error": "..."}`` JSON on stdout with exit
code 2 so the route can surface them as HTTP 400/500 cleanly.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

EXIT_OK = 0
EXIT_ERROR = 2

MAX_SECTIONS = 16
MAX_TRACES = 8
MAX_POINTS = 2000
MAX_ROWS = 300
MAX_CELLS = 40
MAX_TEXT = 200


def _fail(message: str) -> int:
    print(json.dumps({"error": message}))
    return EXIT_ERROR


def _clean_number(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite point coordinate")
    return number


def _validate_section(section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise TypeError("each section must be an object")
    kind = section.get("kind")
    if kind not in {"metrics", "figure", "table"}:
        raise ValueError(f"unknown section kind {kind!r}")
    title = section.get("title")
    if not isinstance(title, str) or not title or len(title) > MAX_TEXT:
        raise ValueError("each section needs a title (1-200 chars)")
    if kind == "metrics":
        items = section.get("items")
        if not isinstance(items, list) or not items or len(items) > 40:
            raise ValueError("metrics sections need 1-40 items")
        clean = []
        for item in items:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("each metrics item is a [name, value] pair")
            name, value = item
            if not isinstance(name, str) or len(name) > MAX_TEXT:
                raise ValueError("metrics names are 1-200 chars")
            clean.append((name, value if isinstance(value, str) else str(value)))
        return {"kind": "metrics", "title": title, "items": clean}
    if kind == "table":
        columns = section.get("columns")
        rows = section.get("rows")
        if not isinstance(columns, list) or not columns or len(columns) > 12:
            raise ValueError("table sections need 1-12 columns")
        if not isinstance(rows, list) or not rows or len(rows) > MAX_ROWS:
            raise ValueError("table sections need 1-300 rows")
        for column in columns:
            if not isinstance(column, str) or len(column) > MAX_TEXT:
                raise ValueError("table column names are 1-200 chars")
        for row in rows:
            if not isinstance(row, list) or len(row) != len(columns):
                raise ValueError("every table row matches the column count")
            for cell in row:
                if isinstance(cell, str) and len(cell) > MAX_CELLS:
                    raise ValueError(f"table cells are capped at {MAX_CELLS} chars")
        return {"kind": "table", "title": title, "columns": columns, "rows": rows}
    # figure
    traces = section.get("traces")
    if not isinstance(traces, list) or not traces or len(traces) > MAX_TRACES:
        raise ValueError("figure sections need 1-8 traces")
    x_label = section.get("xLabel")
    y_label = section.get("yLabel")
    for label in (x_label, y_label):
        if not isinstance(label, str) or len(label) > MAX_TEXT:
            raise ValueError("figure labels are 0-200 chars")
    clean_traces = []
    for trace in traces:
        if not isinstance(trace, dict):
            raise TypeError("each trace must be an object")
        name = trace.get("name")
        style = trace.get("style")
        if not isinstance(name, str) or not name or len(name) > MAX_TEXT:
            raise ValueError("each trace needs a name (1-200 chars)")
        if style not in {"line", "band", "scatter"}:
            raise ValueError(f"unknown trace style {style!r}")
        points = trace.get("points")
        if not isinstance(points, list) or not points or len(points) > MAX_POINTS:
            raise ValueError(f"traces carry 1-{MAX_POINTS} points")
        width = 4 if style == "band" else 2
        for point in points:
            if not isinstance(point, list) or len(point) != width:
                raise ValueError(
                    f"{style} traces carry {'[x, y, lo, hi]' if style == 'band' else '[x, y]'} points"
                )
            for value in point:
                _clean_number(value)
        classes = trace.get("classes")
        probas = trace.get("probas")
        if classes is not None:
            if not isinstance(classes, list) or len(classes) != len(points):
                raise ValueError("classes must match the point count")
            for value in classes:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError("classes are numbers")
        if probas is not None:
            if not isinstance(probas, list) or len(probas) != len(points):
                raise ValueError("probas must match the point count")
            for value in probas:
                _clean_number(value)
        clean_traces.append(trace)
    return {
        "kind": "figure",
        "title": title,
        "xLabel": x_label,
        "yLabel": y_label,
        "traces": clean_traces,
    }


def _build_figure(plt: Any, palette: tuple[str, ...], section: dict[str, Any]) -> Any:
    """Compose one matplotlib figure from the trace list.

    Band traces mirror the library's own learning-curve style
    (``fill_between(mean ± std, alpha=0.15)`` + a mean line with
    markers); class scatters split teal/neutral exactly the way the web
    boundary dots do, ``probas`` fading opacity by ``|p − 0.5|``.
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for index, trace in enumerate(section["traces"]):
        color = palette[index % len(palette)]
        style = trace["style"]
        points = trace["points"]
        if style == "band":
            xs = [p[0] for p in points]
            mean = [p[1] for p in points]
            lo = [p[2] for p in points]
            hi = [p[3] for p in points]
            ax.fill_between(xs, lo, hi, alpha=0.15, color=color, linewidth=0)
            ax.plot(xs, mean, marker="o", lw=1.9, color=color, label=trace["name"])
        elif style == "line":
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, lw=1.9, color=color, label=trace["name"])
        else:  # scatter
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            classes = trace.get("classes")
            probas = trace.get("probas")
            if classes is not None:
                for class_value in (0, 1):
                    dot_color = color if class_value else palette[2]
                    members = [
                        (x, y, p)
                        for x, y, c, p in zip(xs, ys, classes, probas or [None] * len(xs))
                        if int(c) == class_value
                    ]
                    if not members:
                        continue
                    if probas is not None:
                        # per-point alpha needs individual dots; the legend
                        # gets one empty proxy so the trace stays nameable.
                        for x, y, proba in members:
                            alpha = max(0.15, min(1.0, abs(proba - 0.5) * 2))
                            ax.scatter(
                                [x], [y], s=14, color=dot_color,
                                alpha=alpha, linewidths=0,
                            )
                        if class_value == 1:
                            ax.scatter(
                                [], [], s=14, color=dot_color,
                                label=trace["name"],
                            )
                    else:
                        ax.scatter(
                            [m[0] for m in members], [m[1] for m in members],
                            s=16, color=dot_color, linewidths=0,
                            label=trace["name"] if class_value == 1 else None,
                        )
            else:
                ax.scatter(xs, ys, s=16, color=color, linewidths=0, label=trace["name"])
    ax.set_xlabel(section.get("xLabel") or "")
    ax.set_ylabel(section.get("yLabel") or "")
    ax.set_title(section["title"])
    if any(trace.get("name") for trace in section["traces"]):
        ax.legend(loc="best", fontsize=9)
    return fig


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 - any parse failure is ours
        return _fail(f"could not parse the request JSON: {exc}")
    if not isinstance(request, dict):
        return _fail("the request must be a JSON object")

    theme = request.get("theme", "auto")
    if not isinstance(theme, str) or theme not in ("auto", "light", "dark"):
        return _fail('theme must be one of "auto", "light", "dark"')
    subtitle = request.get("subtitle")
    if not isinstance(subtitle, str) or len(subtitle) > MAX_TEXT:
        subtitle = "web playground session"
    sections_in = request.get("sections")
    if not isinstance(sections_in, list) or not sections_in:
        return _fail("the report needs at least one section")
    if len(sections_in) > MAX_SECTIONS:
        return _fail(f"at most {MAX_SECTIONS} sections per report")

    sections = []
    try:
        for section in sections_in:
            sections.append(_validate_section(section))
    except (TypeError, ValueError) as exc:
        return _fail(str(exc))

    try:  # pragma: no cover - import guard for a broken environment
        from pharmaml.viz import PALETTE, HTMLReport, get_pyplot
    except ImportError as exc:
        return _fail(f"pharmaml.viz unavailable: {exc}")

    plt = get_pyplot()
    report = HTMLReport(
        "PharmaML web playground session",
        subtitle=subtitle,
        theme=theme,
    )
    figures = 0
    tables = 0
    composed: list[Any] = []
    try:
        for section in sections:
            if section["kind"] == "metrics":
                report.add_metrics(dict(section["items"]), heading=section["title"])
            elif section["kind"] == "table":
                import pandas as pd

                report.add_table(
                    pd.DataFrame(section["rows"], columns=section["columns"]),
                    heading=section["title"],
                )
                tables += 1
            else:
                figure = _build_figure(plt, PALETTE, section)
                composed.append(figure)
        report.add_text(
            "Every chart and table the web playground session still holds: the "
            "last ML training result (metrics + parity/decision figure), the "
            "learning-curve run, the PK/PD regimen and the NCA rows.  Figures "
            "are re-composed server-side with matplotlib and embedded as "
            "base64, so this file needs no network and no companion assets: "
            "move it, mail it, archive it.",
            heading="What this file contains",
        )
        for section, figure in zip(
            [s for s in sections if s["kind"] == "figure"], composed
        ):
            report.add_figure(figure, caption=section["title"])
            figures += 1
        html = report.render()
    except Exception as exc:  # noqa: BLE001 - library errors are ours to report
        return _fail(f"could not compose the report: {exc}")
    finally:
        for figure in composed:
            plt.close(figure)

    print(
        json.dumps(
            {
                "html": html,
                "figures": figures,
                "tables": tables,
                "bytes": len(html.encode("utf-8")),
                "theme": theme,
            }
        )
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
