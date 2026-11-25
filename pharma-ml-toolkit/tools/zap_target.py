#!/usr/bin/env python
"""The disposable penetration-test target (part17 §4.4, Round 356 P1).

Starts the REAL backend application — the same ``create_app()`` the
Tauri host ships — against a throwaway data directory, with a FIXED
SYNTHETIC session token, seeds it with clearly-labelled synthetic
data, and dumps the exact OpenAPI contract for ZAP's ``openapi``
job. It exists for one reason: ``.github/workflows/
penetration-test.yml`` must never scan anything except an isolated,
project-owned, ephemeral instance (part17's own standing rule), and
this launcher is what makes that instance.

Hard rules baked into the design (the workflow's documented lanes):

* loopback binding only (``127.0.0.1``), exactly like the product —
  the ephemeral instance never widens the attack surface it tests;
* ``ZAP_TEST_TOKEN`` is a synthetic literal for the ephemeral
  instance — NOT a credential, NOT derived from anything, and the
  only secret-shaped string this file ever contains by design;
* the seeded data is synthetic, deterministic, and labelled as such
  in every record name — no real datasets, no user data;
* never part of the installed runtime: no import from this module
  exists under ``src/`` (the supply-chain suite's tree scan keeps
  it that way — a product import would fail the workflow-hygiene
  pin).

Usage (from pharma-ml-toolkit/, exercised by the CI workflow):

    python tools/zap_target.py \\
        --data-dir "$RUNNER_TEMP/zap-data" \\
        --openapi-out "$RUNNER_TEMP/openapi.json" \\
        --seed-report-out "$RUNNER_TEMP/zap-seed.json" \\
        --port 8737

Prints one JSON status line when ready (mirroring the product's
handshake discipline: machine-readable stdout, diagnostics to
stderr), then serves until SIGTERM/SIGINT.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

#: The ephemeral instance's fixed synthetic token (part17 §4.4
#: "Configure authentication using ephemeral test credentials" — a
#: literal, restartable, disposable credential that never existed as
#: a real session on any machine).
ZAP_TEST_TOKEN = "zap-synthetic-session-token-fixed"

#: The default port the workflow targets (a fixed port so ZAP's
#: context URL is stable; collision risk on an ephemeral runner is
#: negligible, and the port is overridable).
DEFAULT_PORT = 8737

#: Deterministic synthetic seeding payload — three molecules of an
#: invented series. The CSV stays header-first (the loader is strict);
#: the "synthetic" label travels in the upload filename and every
#: record name the seed reports.
SYNTHETIC_CSV = (
    "smiles,logp\n"
    "CCO,1.1\n"
    "CCC,1.9\n"
    "CCCO,0.8\n"
)


def build_target_app(data_dir: Path | str | None = None):
    """Assemble the real app with the synthetic token.

    ``data_dir``: a throwaway directory (the workflow passes
    ``$RUNNER_TEMP/zap-data``) so the seeded registry is durable
    for the scan and never touches a real user-data home.
    """
    from pharmaml.api import create_app

    return create_app(token=ZAP_TEST_TOKEN, data_dir=data_dir)


def seed(client) -> dict[str, Any]:
    """Seed the synthetic dataset + one table artifact.

    Returns a small report the workflow archives next to the ZAP
    reports (the scan's own provenance: what the target carried).
    """
    dataset = client.post(
        "/api/v1/datasets",
        headers={"X-Session-Token": ZAP_TEST_TOKEN},
        files={
            "file": (
                "zap-synthetic.csv",
                SYNTHETIC_CSV.encode(),
                "text/csv",
            )
        },
        data={"target": "logp", "mode": "sync"},
    )
    dataset.raise_for_status()
    dataset_id = dataset.json()["dataset"]["dataset_id"]
    table = client.post(
        "/api/v1/artifacts/table",
        headers={"X-Session-Token": ZAP_TEST_TOKEN},
        json={
            "dataset_id": dataset_id,
            "format": "csv",
            "filename": "zap-synthetic-table",
        },
    )
    table.raise_for_status()
    return {
        "synthetic": True,
        "datasets": 1,
        "artifacts": 1,
        "dataset_id": dataset_id,
        "artifact_id": table.json()["artifact"]["artifact_id"],
    }


def dump_openapi(client, out: Path) -> dict:
    """Retrieve the exact OpenAPI contract for ZAP's openapi job."""
    response = client.get(
        "/api/openapi.json",
        headers={"X-Session-Token": ZAP_TEST_TOKEN},
    )
    response.raise_for_status()
    contract = response.json()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


def _ready_line(port: int, report: dict) -> str:
    return json.dumps(
        {"ready": True, "port": port, **report}, sort_keys=True
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the disposable ZAP target instance."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--openapi-out", type=Path, required=True)
    parser.add_argument("--seed-report-out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    import uvicorn
    from fastapi.testclient import TestClient

    app = build_target_app(data_dir=args.data_dir)

    # seed + contract through the real ASGI stack (the full
    # middleware/guard lanes included), synchronously, BEFORE the
    # socket serves its first ZAP request.
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        report = seed(tc)
        dump_openapi(tc, args.openapi_out)
    args.seed_report_out.parent.mkdir(parents=True, exist_ok=True)
    args.seed_report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda *_: setattr(server, "should_exit", True))
        except ValueError:  # pragma: no cover - non-main thread
            pass
    print(_ready_line(args.port, report), flush=True)
    server.run()
    return 0


if __name__ == "__main__":  # pragma: no cover - the workflow lane
    sys.exit(main())
