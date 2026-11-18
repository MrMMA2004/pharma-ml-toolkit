#!/usr/bin/env python
"""Export the FastAPI OpenAPI contract to contracts/openapi.json.

part11 Phase Twelve §12.2 (API contracts) + Milestone C ("Generate
OpenAPI"): the TypeScript client and the frontend types are generated
or validated from THIS file — one DTO definition, never two.  The
committed copy is drift-gated by tests/api/test_contracts.py: a
schema change without a re-export fails the suite.

Usage (from pharma-ml-toolkit/):

    python tools/export_openapi.py            # write + report
    python tools/export_openapi.py --check     # exit 1 on drift
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONTRACT = REPO / "contracts" / "openapi.json"


def build_schema() -> dict:
    """Assemble the app and return its OpenAPI schema (deterministic)."""
    from pharmaml.api import create_app

    app = create_app(token="contract-export")
    return app.openapi()


def render(schema: dict) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    check = "--check" in argv
    CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    schema = build_schema()
    rendered = render(schema)
    if check:
        if not CONTRACT.exists():
            print("FAIL: no committed contract — run the export once")
            return 1
        committed = CONTRACT.read_text()
        if committed != rendered:
            print(
                "FAIL: contracts/openapi.json is stale — re-run "
                "tools/export_openapi.py"
            )
            return 1
        print(
            f"OK: committed contract is current "
            f"({len(schema['paths'])} paths)"
        )
        return 0
    CONTRACT.write_text(rendered)
    print(
        f"wrote {CONTRACT.relative_to(REPO)} "
        f"({len(schema['paths'])} paths, "
        f"{len(schema.get('components', {}).get('schemas', {}))} schemas)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
