#!/usr/bin/env python3
"""Stdin -> stdout JSON bridge for the Next.js ``/api/chem/sa-score`` route.

Runs inside the toolkit virtualenv (real RDKit + pharmaml) and wires ONLY
the public ``pharmaml.chem.sa_scoring`` API (MASTER_PROMPT part3 §24):

    sa_scores -> (optional) filter_by_sa -> sa_score_summary

Request (stdin, JSON)::

    {"smiles": ["CCO", ...],
     "minSa": 1.5,          // optional inclusive lower bound
     "maxSa": 4.0}          // optional inclusive upper bound

Response (stdout, JSON, camelCase keys for the TypeScript route)::

    {
      "nTotal": 3, "nScored": 2, "nInvalid": 1,
      "results": [{"smiles": "...", "score": 1.98, "valid": true,
                    "canonicalSmiles": "...", "error": null,
                    "direction": "lower = estimated easier to synthesise"}, ...],
      "sorted": [{"smiles": "...", "score": ...}, ...],   // easiest first
      "summary": {"min": ..., "max": ..., "mean": ..., "median": ...,
                   "std": ..., "method": "Ertl-Schuffenhauer ...",
                   "direction": "lower = ...",
                   "limitations": "The SA score is a computational estimate ..."},
      "kept": [...], "rejected": [...],      // present iff a bound was given
      "elapsedSeconds": 0.4, "rdkitVersion": "2026.09.1"
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
    # RDKit parse-error noise goes to stderr by default; the bridge contract
    # is strict JSON on stdout only, but the spam also pollutes dev.log.
    try:  # pragma: no cover - cosmetic logger guard
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.error")
    except Exception:  # noqa: BLE001 - never fail the bridge on cosmetics
        pass

    try:
        request = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(json.dumps({"error": f"bridge request is not valid JSON: {exc}"}))
        return EXIT_ERROR

    if not isinstance(request, dict):
        print(json.dumps({"error": "bridge request must be a JSON object"}))
        return EXIT_ERROR

    smiles = request.get("smiles")
    min_sa = request.get("minSa")
    max_sa = request.get("maxSa")

    if not isinstance(smiles, list) or not all(isinstance(s, str) for s in smiles):
        print(json.dumps({"error": "smiles must be a list of strings"}))
        return EXIT_ERROR
    if not smiles:
        print(json.dumps({"error": "pass at least one SMILES string"}))
        return EXIT_ERROR
    for bound, label in ((min_sa, "minSa"), (max_sa, "maxSa")):
        if bound is not None and (
            not isinstance(bound, (int, float))
            or isinstance(bound, bool)
            or not math.isfinite(bound)
        ):
            print(json.dumps({"error": f"{label} must be a finite number"}))
            return EXIT_ERROR
    if min_sa is not None and max_sa is not None and min_sa > max_sa:
        print(json.dumps({"error": f"minSa ({min_sa}) must not exceed maxSa ({max_sa})"}))
        return EXIT_ERROR

    try:  # pragma: no cover - import guard for a broken environment
        from pharmaml.chem.sa_scoring import (
            SA_SCORE_LIMITATIONS,
            SA_SCORE_METHOD,
            filter_by_sa,
            sa_scores,
            sa_score_summary,
        )
    except ImportError as exc:
        print(json.dumps({"error": f"pharmaml.chem.sa_scoring unavailable: {exc}"}))
        return EXIT_ERROR

    import rdkit

    started = time.perf_counter()
    try:
        batch = sa_scores(smiles)
        summary = sa_score_summary(smiles)
    except Exception as exc:  # noqa: BLE001 - surfaced to the web UI verbatim
        print(json.dumps({"error": f"SA scoring failed: {exc}"}))
        return EXIT_ERROR

    if batch.n_scored == 0:
        print(json.dumps({
            "error": "No valid SMILES in the request — every row failed to parse.",
            "results": _clean([r.as_dict() for r in batch.results]),
        }))
        return EXIT_ERROR

    payload: dict[str, Any] = {
        "nTotal": len(batch.results),
        "nScored": batch.n_scored,
        "nInvalid": batch.n_invalid,
        "results": [r.as_dict() for r in batch.results],
        "sorted": [
            {"smiles": r.smiles, "score": r.score}
            for r in batch.sorted_by_score()
        ],
        "summary": _clean(summary),
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "rdkitVersion": rdkit.__version__,
        "method": SA_SCORE_METHOD,
        "direction": "lower = estimated easier to synthesise",
        "limitations": SA_SCORE_LIMITATIONS,
    }

    if min_sa is not None or max_sa is not None:
        outcome = filter_by_sa(
            batch.results,
            min_sa=None if min_sa is None else float(min_sa),
            max_sa=None if max_sa is None else float(max_sa),
        )
        payload["kept"] = [r.as_dict() for r in outcome.kept]
        payload["rejected"] = [r.as_dict() for r in outcome.rejected]
        payload["nKept"] = outcome.n_kept
        payload["nRejected"] = outcome.n_rejected

    print(json.dumps(_clean(payload)))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
