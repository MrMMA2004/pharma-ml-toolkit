"""part11 Phase Twelve §12.2 — the OpenAPI contract as the single DTO
source.

The committed ``contracts/openapi.json`` is the machine-readable API
contract the frontend's TypeScript client/types are generated or
validated from (openapi-typescript / Orval per the master prompt; NSwag
was the part10 WinUI artifact and is replaced by part11's §2 rules).
The drift gate keeps it honest: a route/schema change without a
re-export turns the suite red.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app

TEST_TOKEN = "test-session-token-0123456789"
CONTRACT = (
    Path(__file__).resolve().parent.parent.parent
    / "contracts" / "openapi.json"
)


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


class TestContractDrift:
    def test_committed_contract_matches_the_live_schema(self):
        assert CONTRACT.exists(), (
            "contracts/openapi.json missing — run tools/export_openapi.py"
        )
        app = create_app(token=TEST_TOKEN)
        live = json.loads(json.dumps(app.openapi(), sort_keys=True))
        committed = json.loads(CONTRACT.read_text())
        assert live == committed, (
            "contracts/openapi.json is stale: the routes changed "
            "without re-running tools/export_openapi.py"
        )

    def test_served_openapi_matches_the_committed_contract(self, client):
        committed = json.loads(CONTRACT.read_text())
        served = client.get(
            "/api/openapi.json", headers=auth_headers()
        ).json()
        assert served == committed


class TestContractShape:
    def test_every_path_is_versioned_under_api_v1(self):
        schema = json.loads(CONTRACT.read_text())
        assert schema["paths"]
        for path in schema["paths"]:
            assert path.startswith("/api/v1/"), (
                f"unversioned route escaped the contract: {path}"
            )

    def test_the_database_surface_is_in_the_contract(self):
        schema = json.loads(CONTRACT.read_text())
        for path in (
            "/api/v1/database",
            "/api/v1/database/jobs",
            "/api/v1/database/preferences",
            "/api/v1/database/preferences/{key}",
            "/api/v1/database/backup",
            "/api/v1/database/restore",
        ):
            assert path in schema["paths"], f"missing {path}"

    def test_the_nine_workspace_domains_are_represented(self):
        schema = json.loads(CONTRACT.read_text())
        domains = {
            "datasets": "/api/v1/datasets",
            "chemistry": "/api/v1/chemistry/descriptors",
            "ml": "/api/v1/ml/train",
            "dl": "/api/v1/dl/train",
            "semi-supervised": "/api/v1/semi-supervised/train",
            "generative": "/api/v1/generative/rnn/train",
            "docking": "/api/v1/docking/dock",
            "pkpd": "/api/v1/pkpd/catalog",
            "md": "/api/v1/md/analyze",
            "visualization": "/api/v1/visualization/catalog",
            "applicability-domain": "/api/v1/applicability-domain/analyze",
            "jobs": "/api/v1/jobs",
            "artifacts": "/api/v1/artifacts",
            "health": "/api/v1/health",
        }
        missing = [
            f"{name} ({path})"
            for name, path in domains.items()
            if path not in schema["paths"]
        ]
        assert not missing, f"workspace domains missing: {missing}"

    def test_health_schema_is_typed(self):
        schema = json.loads(CONTRACT.read_text())
        health = schema["paths"]["/api/v1/health"]["get"]
        assert "200" in health["responses"]
        ref = health["responses"]["200"].get(
            "content", {}
        ).get("application/json", {}).get("schema", {})
        assert ref.get("$ref") or ref.get("type"), (
            "the health response is untyped in the contract"
        )
