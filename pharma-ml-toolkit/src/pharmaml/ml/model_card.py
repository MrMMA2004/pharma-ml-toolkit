"""Model cards: the provenance and honesty sheet for a saved model.

Mitchell et al. (2019), *Model Cards for Model Reporting*, proposed a
short structured document that travels with a model so a downstream user
knows what it is, how it was scored, and — crucially — where it should
**not** be trusted.  A ``.joblib`` bundle
(:mod:`pharmaml.ml.persistence`) already carries the raw material: the
estimator, the ordered feature names, the training target, the held-out
metrics, the best hyperparameters, and the stack versions that produced
it.  This module renders that material as a Markdown or HTML card.

The card is generated, not hand-written: every number comes from the
bundle's metadata (or from live introspection of the estimator), so it
can never drift from the artifact it describes.  Two sections are always
present because a QSAR card is incomplete without them:

* **Intended use / limitations** — a model trained on one assay does not
  transfer to another; the card says so instead of implying generality.
* **Applicability domain** — when the bundle records the training
  feature ranges, the card states the box the model was fitted inside,
  so a caller can see when a query falls outside it.

Examples
--------
>>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> from pharmaml.ml.persistence import save_model
>>> from pharmaml.ml.model_card import build_model_card, model_card_markdown
>>> model = LinearRegression().fit([[0.0], [1.0], [2.0]], [0.0, 2.0, 4.0])
>>> path = save_model(model, "/tmp/pharmaml_doctest_card/m.joblib", name="y=2x",
...                   task="regression", features=["x"],
...                   metrics={"rmse": 0.0, "r2": 1.0}, target="y")
>>> card = build_model_card(path)
>>> card["name"], card["task"]
('y=2x', 'regression')
>>> "rmse" in card["metrics"]
True
>>> "# Model card: y=2x" in model_card_markdown(card)
True
>>> import shutil
>>> shutil.rmtree("/tmp/pharmaml_doctest_card")
"""

from __future__ import annotations

import datetime as _dt
import html
import json
from pathlib import Path
from typing import Any

from pharmaml.ml.persistence import load_model

__all__ = [
    "build_model_card",
    "model_card_html",
    "model_card_markdown",
    "write_model_card",
]


def _estimator_name(model: Any) -> str:
    return type(model).__name__


def _estimator_params(model: Any) -> dict[str, Any]:
    """Best-effort ``get_params`` snapshot (JSON-safe scalars only)."""
    getter = getattr(model, "get_params", None)
    if getter is None:
        return {}
    try:
        params = getter(deep=False)
    except Exception:  # noqa: BLE001 - exotic estimators  # pragma: no cover
        return {}
    safe: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
    return safe


def build_model_card(model_path: str | Path, *, training_frame: Any = None) -> dict[str, Any]:
    """Assemble the card's data block from a saved bundle.

    Parameters
    ----------
    model_path:
        A ``.joblib`` bundle written by
        :func:`pharmaml.ml.persistence.save_model`.
    training_frame:
        Optional training DataFrame.  When given, the card records the
        per-feature min/max box (the applicability domain) and the row
        count; when omitted those fields are ``None`` and the card says
        so rather than inventing them.

    Returns
    -------
    dict
        The structured card: identity, provenance, metrics, best
        parameters, estimator hyperparameters, feature contract, the
        applicability box, and generated-at timestamp.
    """
    model, metadata = load_model(model_path)
    features = metadata.get("features")
    domain: dict[str, dict[str, float]] | None = None
    n_rows: int | None = None
    if training_frame is not None:
        if not hasattr(training_frame, "columns"):
            from pharmaml.exceptions import DataValidationError

            raise DataValidationError(
                "training_frame must be a DataFrame (with .columns), got "
                f"{type(training_frame).__name__}"
            )
        n_rows = len(training_frame)
        columns = list(features) if isinstance(features, list) and features else None
        from pharmaml.ml.persistence import feature_ranges_from_frame

        domain = feature_ranges_from_frame(training_frame, columns) or None
    elif metadata.get("feature_ranges"):
        # No training frame at hand, but the bundle recorded its box at
        # train time — report that rather than claiming it is missing.
        domain = {
            str(name): {"min": float(b["min"]), "max": float(b["max"])}
            for name, b in metadata["feature_ranges"].items()
            if isinstance(b, dict) and "min" in b and "max" in b
        } or None
    return {
        "name": metadata.get("name", "<unknown>"),
        "task": metadata.get("task", "<unknown>"),
        "target": metadata.get("target"),
        "estimator": _estimator_name(model),
        "estimator_params": _estimator_params(model),
        "features": list(features) if isinstance(features, list) else None,
        "n_features": len(features) if isinstance(features, list) else None,
        "metrics": dict(metadata.get("metrics", {})),
        "best_params": dict(metadata.get("best_params", {})),
        "provenance": {
            "pharmaml_version": metadata.get("pharmaml_version"),
            "sklearn_version": metadata.get("sklearn_version"),
            "numpy_version": metadata.get("numpy_version"),
            "trained_at": metadata.get("trained_at"),
        },
        "training_rows": n_rows,
        "applicability_domain": domain,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def model_card_markdown(card: dict[str, Any]) -> str:
    """Render the card dict as a Markdown document."""
    lines: list[str] = [f"# Model card: {card['name']}", ""]
    lines.append(
        "> Generated by `pharmaml.ml.model_card` — every value below is read "
        "from the saved bundle, never hand-entered."
    )
    lines.append("")

    lines += ["## Overview", ""]
    lines.append(f"- **Task:** {card.get('task', '<unknown>')}")
    target = card.get("target")
    lines.append(f"- **Target:** {target if target else '_(not recorded)_'}")
    lines.append(f"- **Estimator:** `{card.get('estimator', '<unknown>')}`")
    n_features = card.get("n_features")
    lines.append(
        f"- **Features:** {n_features} feature(s)"
        if n_features
        else "- **Features:** _(names not recorded — the count contract applies)_"
    )
    rows = card.get("training_rows")
    lines.append(
        f"- **Training rows:** {rows:,}" if rows else "- **Training rows:** _(not recorded)_"
    )
    lines.append("")

    lines += ["## Metrics (held-out)", ""]
    metrics = card.get("metrics") or {}
    if metrics:
        lines += ["| Metric | Value |", "| --- | --- |"]
        for key, value in metrics.items():
            lines.append(f"| {key} | {_format_metric(value)} |")
    else:
        lines.append("_No held-out metrics were recorded in the bundle._")
    lines.append("")

    best = card.get("best_params") or {}
    lines += ["## Hyperparameters", ""]
    if best:
        lines.append("Selected by tuning:")
        lines.append("")
        lines += ["| Parameter | Value |", "| --- | --- |"]
        for key, value in best.items():
            lines.append(f"| `{key}` | {value} |")
    else:
        lines.append("_Defaults (no tuning search recorded)._")
    lines.append("")

    params = card.get("estimator_params") or {}
    if params:
        lines += ["Estimator parameters:", ""]
        lines += ["| Parameter | Value |", "| --- | --- |"]
        for key, value in params.items():
            lines.append(f"| `{key}` | {value} |")
        lines.append("")

    features = card.get("features")
    if isinstance(features, list) and features:
        preview = ", ".join(f"`{name}`" for name in features[:12])
        more = f" … (+{len(features) - 12})" if len(features) > 12 else ""
        lines += ["## Feature contract", "", f"Ordered features: {preview}{more}", ""]

    domain = card.get("applicability_domain")
    lines += ["## Applicability domain", ""]
    if domain:
        lines.append(
            "The model was fitted inside these per-feature ranges; predictions "
            "for rows outside the box are extrapolations and should be treated "
            "with caution."
        )
        lines.append("")
        lines += ["| Feature | Min | Max |", "| --- | --- | --- |"]
        for name, bounds in domain.items():
            lines.append(f"| `{name}` | {bounds['min']:.6g} | {bounds['max']:.6g} |")
    else:
        lines.append(
            "_Not computed — regenerate the card with `training_frame=` to "
            "record the training feature ranges._"
        )
    lines.append("")

    lines += ["## Intended use and limitations", ""]
    lines.append(
        "- **Intended use:** QSAR/QSPR prediction for the assay and feature "
        "space it was trained on, as decision *support* alongside experimental "
        "measurement — not as a replacement for it."
    )
    lines.append(
        "- **Out-of-distribution inputs:** predictions for rows outside the "
        "applicability domain above are extrapolations; validate before use."
    )
    lines.append(
        "- **Transfer:** a model trained on one target/assay does not transfer "
        "to another without retraining and re-validation."
    )
    lines.append(
        "- **Metrics** are estimates on a held-out split; small test sets carry "
        "wide uncertainty around every number reported above."
    )
    lines.append("")

    prov = card.get("provenance") or {}
    lines += ["## Provenance", ""]
    lines.append(f"- **Trained at:** {prov.get('trained_at', '?')}")
    lines.append(f"- **pharmaml:** {prov.get('pharmaml_version', '?')}")
    lines.append(f"- **scikit-learn:** {prov.get('sklearn_version', '?')}")
    lines.append(f"- **numpy:** {prov.get('numpy_version', '?')}")
    lines.append(f"- **Card generated at:** {card.get('generated_at', '?')}")
    lines.append("")
    return "\n".join(lines)


_CARD_CSS = (
    "body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:52rem;"
    "margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1b1f23}"
    "h1{border-bottom:1px solid #d0d7de;padding-bottom:.3rem}"
    "table{border-collapse:collapse;margin:.5rem 0}"
    "th,td{border:1px solid #d0d7de;padding:.25rem .6rem;text-align:left}"
    "code{background:#f6f8fa;padding:.1rem .3rem;border-radius:4px}"
    "blockquote{color:#57606a;border-left:3px solid #d0d7de;margin:0;"
    "padding-left:.8rem}"
)


def _inline_html(text: str) -> str:
    """Escape ``text`` then render ``**bold**`` and `` `code` `` spans."""
    escaped = html.escape(text)
    parts = escaped.split("`")
    rebuilt = "".join(
        f"<code>{chunk}</code>" if index % 2 else chunk
        for index, chunk in enumerate(parts)
    )
    while "**" in rebuilt:
        rebuilt = rebuilt.replace("**", "<strong>", 1)
        if "**" in rebuilt:
            rebuilt = rebuilt.replace("**", "</strong>", 1)
    return rebuilt


def model_card_html(card: dict[str, Any]) -> str:
    """Render the card dict as a standalone, self-contained HTML page.

    A tiny, dependency-free renderer for exactly the Markdown subset
    :func:`model_card_markdown` emits (headings, a blockquote, bullet
    lists, pipe tables, bold and inline code) — so the HTML card needs no
    ``markdown`` package and stays a single self-contained file.
    """
    out: list[str] = []
    in_list = False
    in_table = False

    def close_blocks() -> None:
        nonlocal in_list, in_table
        if in_list:
            out.append("</ul>")
            in_list = False
        if in_table:
            out.append("</table>")
            in_table = False

    for raw in model_card_markdown(card).splitlines():
        line = raw.rstrip()
        if line.startswith("| "):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(set(cell) <= {"-", ":", " "} and cell for cell in cells):
                continue  # the markdown alignment row
            if not in_table:
                close_blocks()
                out.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"
            out.append(
                "<tr>" + "".join(f"<{tag}>{_inline_html(c)}</{tag}>" for c in cells) + "</tr>"
            )
            continue
        close_blocks()
        if line.startswith("## "):
            out.append(f"<h2>{_inline_html(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{_inline_html(line[2:])}</h1>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{_inline_html(line[2:])}</blockquote>")
        elif line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline_html(line[2:])}</li>")
        elif line:
            out.append(f"<p>{_inline_html(line)}</p>")
    close_blocks()

    title = html.escape(str(card["name"]))
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        f"<title>Model card: {title}</title>"
        f"<style>{_CARD_CSS}</style></head><body>\n"
        + "\n".join(out)
        + "\n</body></html>\n"
    )


def write_model_card(
    model_path: str | Path,
    out_path: str | Path,
    *,
    training_frame: Any = None,
    fmt: str | None = None,
) -> Path:
    """Build and write a card next to (or instead of) the model.

    ``fmt`` defaults from ``out_path``'s suffix (``.md`` → markdown,
    ``.html`` → HTML); anything else falls back to markdown.  Parent
    directories are created.
    """
    card = build_model_card(model_path, training_frame=training_frame)
    target = Path(out_path)
    if fmt:
        resolved = fmt
    elif target.suffix.lower() in {".html", ".htm"}:
        resolved = "html"
    elif target.suffix.lower() == ".json":
        resolved = "json"
    else:
        resolved = "md"
    if resolved in {"html", "htm"}:
        text = model_card_html(card)
    elif resolved in {"json"}:
        text = json.dumps(card, indent=2, sort_keys=True, default=str) + "\n"
    else:
        text = model_card_markdown(card)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
