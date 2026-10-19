#!/usr/bin/env python3
"""Stdin -> stdout JSON bridge for the Next.js ``/api/docking/rescore`` route.

Runs inside the toolkit virtualenv (real pharmaml) and wires ONLY the public
``pharmaml.docking.consensus_breakdown`` API -- the same shared core behind
``consensus_rescore`` and the ``pharmaml docking rescore`` CLI, so the web
playground can never drift from the library's math.

Request (stdin, JSON)::

    {
      "content": "ligand,best_lowest_energy,...\\naspirin,-4.91,...",
      "columns": ["best_lowest_energy", "best_num_in_cluster", ...],
      "weights": [1.0, 3.0, ...],          # optional, uniform when absent
      "higher": ["best_num_in_cluster"]    # optional direction overrides
    }

Response (stdout, JSON, camelCase keys for the TypeScript route)::

    {
      "identityColumn": "ligand",
      "columns": ["best_lowest_energy", ...],
      "rows": [{"ligand": "metformin", "best_lowest_energy_norm": 0.488,
                ..., "consensus": 0.782}, ...],       # sorted, consensus desc
      "recipe": "best_lowest_energy (lower=better, weight 1), ...",
      "nRows": 12,
      "verdict": {"identity": "metformin", "consensus": 0.782,
                  "driver": "ligand_efficiency_norm", "driverValue": 1.0},
      "pharmamlVersion": "1.2.33"
    }

Errors print a one-line ``{"error": "..."}`` JSON on stdout with exit code 2
so the route can surface them as HTTP 400/500 cleanly (the same contract as
docking_prep_bridge.py / chem_features_bridge.py).
"""

from __future__ import annotations

import io
import json
import math
import sys
from typing import Any

EXIT_OK = 0
EXIT_ERROR = 2

MAX_CONTENT_CHARS = 200_000
MAX_COLUMNS = 8
MAX_ROWS_RETURNED = 200


def _clean(value: Any) -> Any:
    """Recursively convert numpy scalars + NaN/inf to strict-JSON values."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _fail(message: str) -> int:
    print(json.dumps({"error": message}))
    return EXIT_ERROR


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _fail(f"bridge request is not valid JSON: {exc}")

    if not isinstance(request, dict):
        return _fail("bridge request must be a JSON object")

    content = request.get("content")
    if not isinstance(content, str) or not content.strip():
        return _fail("content must be a non-empty CSV string")

    columns = request.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or len(columns) > MAX_COLUMNS
        or not all(isinstance(c, str) and c.strip() for c in columns)
    ):
        return _fail(f"columns must be a list of 1-{MAX_COLUMNS} column names")

    weights_raw = request.get("weights")
    weights: list[float] | None = None
    if weights_raw is not None:
        if (
            not isinstance(weights_raw, list)
            or len(weights_raw) != len(columns)
            or not all(
                isinstance(w, (int, float)) and not isinstance(w, bool)
                and math.isfinite(float(w)) and float(w) >= 0.0
                for w in weights_raw
            )
        ):
            return _fail(
                "weights must be a list of numbers (0-100) matching the "
                f"{len(columns)} selected column(s)"
            )
        weights = [float(w) for w in weights_raw]
        if all(w == 0.0 for w in weights):
            return _fail("weights cannot all be zero")

    higher_raw = request.get("higher")
    higher: dict[str, bool] = {}
    if higher_raw is not None:
        if not isinstance(higher_raw, list) or not all(
            isinstance(h, str) for h in higher_raw
        ):
            return _fail("higher must be a list of column names")
        higher = {name: True for name in higher_raw if name}

    try:  # pragma: no cover - import guard for a broken environment
        import pharmaml
        from pharmaml.docking import consensus_breakdown
    except ImportError as exc:
        return _fail(f"pharmaml.docking unavailable: {exc}")

    import pandas as pd

    try:
        # sep=None + the python engine sniffs the delimiter, mirroring the
        # CSV arm of pharmaml.utils.io.read_table.
        frame = pd.read_csv(io.StringIO(content), sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001 - malformed CSV is user input
        return _fail(f"could not parse the CSV content: {exc}")

    if frame.empty:
        return _fail("the CSV content has no data rows")
    frame.columns = [str(column).strip() for column in frame.columns]
    identity_column = str(frame.columns[0])

    try:
        breakdown = consensus_breakdown(
            frame, columns, weights=weights, higher_is_better=higher or None
        )
    except Exception as exc:  # noqa: BLE001 - validation errors are user input
        return _fail(str(exc))

    rescored = frame.copy()
    for column in breakdown.columns:
        rescored[column] = breakdown[column].to_numpy()
    rescored = rescored.sort_values(
        "consensus", ascending=False, na_position="last"
    ).reset_index(drop=True)

    # The introspection line, byte-identical to the CLI's recipe: which
    # evidence, which direction, which weight.
    directions = higher or {}
    recipe = ", ".join(
        f"{col} ("
        + (
            "higher"
            if directions.get(col, not col.endswith(("energy", "affinity")))
            else "lower"
        )
        + "=better"
        + (f", weight {w:g}" if weights else "")
        + ")"
        for col, w in zip(columns, weights or [1.0] * len(columns))
    )

    norm_columns = [f"{column}_norm" for column in columns]
    rows: list[dict[str, Any]] = []
    for record in rescored.head(MAX_ROWS_RETURNED).to_dict("records"):
        row: dict[str, Any] = {identity_column: record.get(identity_column)}
        for column in norm_columns + ["consensus"]:
            row[column] = record.get(column)
        rows.append(row)

    verdict: dict[str, Any] | None = None
    valid = rescored["consensus"].dropna()
    if len(valid):
        # Column-wise access keeps each dtype: .loc[scalar] would upcast the
        # whole row Series to float64 and render integer identities as 4.0.
        top_index = int(valid.idxmax())
        top_norms = {column: rescored[column].iloc[top_index] for column in norm_columns}
        driver = max(top_norms, key=lambda name: top_norms[name])
        verdict = {
            "identity": rescored[identity_column].iloc[top_index],
            "consensus": float(rescored["consensus"].iloc[top_index]),
            "driver": driver,
            "driverValue": float(rescored[driver].iloc[top_index]),
        }

    response = {
        "identityColumn": identity_column,
        "columns": columns,
        "rows": rows,
        "recipe": recipe,
        "nRows": int(len(rescored)),
        "verdict": verdict,
        "pharmamlVersion": pharmaml.__version__,
    }
    json.dump(_clean(response), sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
