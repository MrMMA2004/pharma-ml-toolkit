#!/usr/bin/env python3
"""Stdin -> stdout JSON bridge for the Next.js ``/api/chem/features`` route.

Runs inside the toolkit virtualenv (real RDKit + pharmaml) and wires ONLY the
public ``pharmaml.chem.features`` API:

    compute_feature_frame -> rank_features -> select_top_features
    (+ family_of_column / selection_metadata)

Request (stdin, JSON)::

    {"smiles": ["CCO", ...], "families": ["descriptors", "ecfp"], "k": 200}

Response (stdout, JSON, camelCase keys for the TypeScript route)::

    {
      "nMolecules": 3, "dropped": [{"row": 1, "smiles": "not-a-smiles"}],
      "totalFeatures": 10795, "familiesComputed": {"descriptors": 43, ...},
      "k": 200, "survivors": 10546, "method": "variance",
      "selected": [{"rank": 1, "feature": "ecfp_bit_...", "family": "ecfp",
                     "importance": 12.34}, ...],
      "familyCounts": {"ecfp": 118, ...},
      "elapsedSeconds": 1.9, "rdkitVersion": "2026.09.1"
    }

Errors print a one-line ``{"error": "..."}`` JSON on stdout with exit code 2
so the route can surface them as HTTP 400/500 cleanly.
"""

from __future__ import annotations

import json
import math
import sys
import time
from typing import Any

EXIT_OK = 0
EXIT_ERROR = 2

# The web calculator is deliberately unsupervised: variance ranking only.
METHOD = "variance"


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
    return str(value)  # numpy scalars, Paths, etc.


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(json.dumps({"error": f"bridge request is not valid JSON: {exc}"}))
        return EXIT_ERROR

    if not isinstance(request, dict):
        print(json.dumps({"error": "bridge request must be a JSON object"}))
        return EXIT_ERROR

    smiles = request.get("smiles")
    families = request.get("families")
    k = request.get("k")

    if not isinstance(smiles, list) or not all(isinstance(s, str) for s in smiles):
        print(json.dumps({"error": "smiles must be a list of strings"}))
        return EXIT_ERROR
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        print(json.dumps({"error": "k must be a positive integer"}))
        return EXIT_ERROR
    if families is None:
        pass  # omitted in the request -> library default (all seven families)
    elif not isinstance(families, list) or not all(isinstance(f, str) for f in families):
        print(json.dumps({"error": "families must be a list of strings"}))
        return EXIT_ERROR

    try:  # pragma: no cover - import guard for a broken environment
        from pharmaml.chem.features import (
            compute_feature_frame,
            family_of_column,
            rank_features,
            select_top_features,
        )
    except ImportError as exc:
        print(json.dumps({"error": f"pharmaml.chem.features unavailable: {exc}"}))
        return EXIT_ERROR

    import rdkit

    started = time.perf_counter()
    try:
        if families is None:
            frame, report = compute_feature_frame(smiles)
        else:
            # An empty list intentionally falls through to the library's own
            # "families must not be empty" ValueError instead of silently
            # expanding to all seven families.
            frame, report = compute_feature_frame(smiles, families)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - surfaced to the web UI verbatim
        print(json.dumps({"error": f"feature computation failed: {exc}"}))
        return EXIT_ERROR

    if report.n_valid == 0:
        print(
            json.dumps(
                {
                    "error": "No valid SMILES in the request — every row failed to parse.",
                    "dropped": _clean(report.dropped),
                }
            )
        )
        return EXIT_ERROR
    if report.n_valid < 2:
        # Variance over a single row is undefined (pandas var -> NaN for every
        # column), which would silently yield an empty selection.
        print(
            json.dumps(
                {
                    "error": (
                        "Variance ranking needs at least 2 valid molecules — "
                        "paste more SMILES (one per line)."
                    )
                }
            )
        )
        return EXIT_ERROR

    try:
        ranked = rank_features(frame, method=METHOD)
        # select_top_features demands scores.index == frame.columns exactly
        # (order included) while rank_features returns descending order.
        scores = ranked.reindex(frame.columns)
        selection = select_top_features(frame, scores, k, method=METHOD)
        all_survivors = select_top_features(frame, scores, len(frame.columns), method=METHOD)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"feature selection failed: {exc}"}))
        return EXIT_ERROR

    selected_rows = [
        {
            "rank": rank,
            "feature": name,
            "family": family_of_column(name),
            "importance": float(selection.scores.loc[name]),
        }
        for rank, name in enumerate(selection.selected, start=1)
    ]

    response = {
        "nMolecules": report.n_valid,
        "dropped": [{"row": row, "smiles": smi} for row, smi in report.dropped],
        "totalFeatures": report.total_columns,
        "familiesComputed": dict(report.families),
        "k": k,
        "survivors": len(all_survivors.selected),
        "method": selection.method,
        "selected": selected_rows,
        "familyCounts": dict(selection.family_counts),
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "rdkitVersion": rdkit.__version__,
    }
    json.dump(_clean(response), sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
