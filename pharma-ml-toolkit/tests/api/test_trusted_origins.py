"""part14 Phase 2/4 — the Tauri/WebView2 <-> FastAPI origin incident.

Round 326 reproduced the defect matrix live over the real
``create_app()`` (``scratch/p14_r326_origin_probe.py`` — the P1..P8
legs, all eight as predicted); this file pins the FIXED behavior as
the minimal regression tests part14 §15 Phase 2 demands ("Write
minimal regression tests for reproduced defects") and Phase 4 runs
after each fix ("Run the focused regression test").

The fixed contract (PART14_PHASE1_AUDIT.md §G.2 — D-1/D-2):

* **D-1 fixed**: the packaged Tauri 2 WebView2 origin
  ``http(s)://tauri.localhost`` is a trusted app origin — actual
  requests from it pass the guard when the Host is loopback and the
  session token is valid (P1/P2);
* **D-2 fixed**: the scoped CORS lane answers OPTIONS preflights for
  the trusted app origins and the loopback dev origins (any port)
  with the exact-origin allow headers (P3/P4) — while NO actual
  request escapes the guard: the token is still required (P5's 401
  lane) and non-loopback Hosts stay rejected (P7/P8);
* the attack surface stays shut: arbitrary origins, subdomains,
  lookalike hosts, port-bearing virtual-host variants, and
  non-loopback Hosts are all rejected, and disallowed origins get NO
  CORS allow headers (the browser keeps blocking them).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app
from pharmaml.api.security import TRUSTED_APP_ORIGINS

TEST_TOKEN = "test-session-token-0123456789"

#: The packaged WebView2 frontend's page origin (Tauri 2 on Windows).
PACKAGED_ORIGIN = "http://tauri.localhost"

#: The loopback Host the packaged app's requests actually target (the
#: handshake URL — never the virtual host itself).
LOOPBACK_HOST = "127.0.0.1:54127"


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers(token: str = TEST_TOKEN):
    return {"X-Session-Token": token}


def preflight_headers(origin: str, method: str = "GET") -> dict:
    return {
        "Host": LOOPBACK_HOST,
        "Origin": origin,
        "Access-Control-Request-Method": method,
        "Access-Control-Request-Headers": "content-type,x-session-token",
    }


# ------------------------------------------- D-1: trusted app origins


class TestTrustedAppOriginRequests:
    def test_trusted_app_origin_policy_is_exact(self):
        # the exported policy is exactly the two Tauri origins — no
        # wildcard, no port, nothing else
        assert TRUSTED_APP_ORIGINS == frozenset(
            {"http://tauri.localhost", "https://tauri.localhost"}
        )

    def test_p1_packaged_origin_with_token_passes(self, client):
        r = client.get(
            "/api/v1/health",
            headers={
                "Host": LOOPBACK_HOST,
                "Origin": PACKAGED_ORIGIN,
                **auth_headers(),
            },
        )
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_p2_packaged_https_origin_with_token_passes(self, client):
        r = client.get(
            "/api/v1/health",
            headers={
                "Host": LOOPBACK_HOST,
                "Origin": "https://tauri.localhost",
                **auth_headers(),
            },
        )
        assert r.status_code == 200

    def test_referer_from_trusted_app_origin_passes(self, client):
        # Referer values carry paths — the trust check parses, so the
        # path does not untrust the origin
        r = client.get(
            "/api/v1/health",
            headers={
                "Host": LOOPBACK_HOST,
                "Referer": "https://tauri.localhost/workspaces/ml",
                **auth_headers(),
            },
        )
        assert r.status_code == 200

    def test_trusted_origin_without_token_is_still_401(self, client):
        # the trusted-origin policy NEVER bypasses authentication
        r = client.get(
            "/api/v1/health",
            headers={"Host": LOOPBACK_HOST, "Origin": PACKAGED_ORIGIN},
        )
        assert r.status_code == 401

    def test_trusted_origin_non_loopback_host_is_still_403(self, client):
        # the Host policy is unchanged: loopback-only — the packaged
        # app's requests target 127.0.0.1, never the virtual host
        r = client.get(
            "/api/v1/health",
            headers={
                "Host": "tauri.localhost",
                "Origin": PACKAGED_ORIGIN,
                **auth_headers(),
            },
        )
        assert r.status_code == 403

    @pytest.mark.parametrize(
        "origin",
        [
            "http://evil.tauri.localhost",  # subdomain — not the app
            "https://sub.tauri.localhost",
            "http://tauri.localhost.evil.example",  # lookalike suffix
            "http://tauri.localhost:8080",  # port-bearing variant
            "tauri.localhost",  # no scheme — not an app origin
            "ftp://tauri.localhost",  # scheme-strict
            "http://localhost.evil.example",  # loopback lookalike
            "https://evil.example",  # arbitrary origin
        ],
    )
    def test_lookalike_and_arbitrary_origins_stay_403(self, client, origin):
        r = client.get(
            "/api/v1/health",
            headers={
                "Host": LOOPBACK_HOST,
                "Origin": origin,
                **auth_headers(),
            },
        )
        assert r.status_code == 403


# ------------------------------------------- D-2: the CORS preflight lane


class TestCorsPreflightLane:
    def test_p3_preflight_from_packaged_origin_is_answered(self, client):
        r = client.options("/api/v1/health", headers=preflight_headers(PACKAGED_ORIGIN))
        assert r.status_code == 200
        assert (
            r.headers.get("access-control-allow-origin") == PACKAGED_ORIGIN
        )
        methods = r.headers.get("access-control-allow-methods", "")
        assert "GET" in methods and "OPTIONS" in methods
        requested = r.headers.get("access-control-allow-headers", "")
        assert "x-session-token" in requested.lower()
        assert "content-type" in requested.lower()

    def test_p2_preflight_from_packaged_https_origin_is_answered(self, client):
        r = client.options(
            "/api/v1/health", headers=preflight_headers("https://tauri.localhost")
        )
        assert r.status_code == 200
        assert (
            r.headers.get("access-control-allow-origin") == "https://tauri.localhost"
        )

    @pytest.mark.parametrize(
        "dev_origin",
        [
            "http://localhost:5174",  # the Vite dev origin (devUrl)
            "http://localhost:5173",
            "http://127.0.0.1:5174",
            "http://[::1]:5174",
            "http://localhost",  # port-less loopback
        ],
    )
    def test_p4_preflight_from_loopback_dev_origins_is_answered(
        self, client, dev_origin
    ):
        r = client.options("/api/v1/health", headers=preflight_headers(dev_origin))
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == dev_origin

    def test_preflight_from_arbitrary_origin_gets_no_cors_grant(self, client):
        r = client.options(
            "/api/v1/health",
            headers=preflight_headers("https://evil.example"),
        )
        # the security property: NO allow headers — the browser keeps
        # blocking the actual request whatever the status is
        assert r.headers.get("access-control-allow-origin") is None

    @pytest.mark.parametrize(
        "origin",
        [
            "http://evil.tauri.localhost",
            "http://tauri.localhost.evil.example",
            "http://tauri.localhost:8080",
            "http://localhost.evil.example",
        ],
    )
    def test_preflight_from_lookalike_origins_gets_no_cors_grant(
        self, client, origin
    ):
        r = client.options("/api/v1/health", headers=preflight_headers(origin))
        assert r.headers.get("access-control-allow-origin") is None

    def test_preflight_never_bypasses_token_on_actual_requests(self, client):
        # the full browser sequence: the preflight passes, but the
        # actual request without the token is still 401 — the lane
        # answers policy probes only, never authenticates
        pre = client.options(
            "/api/v1/health", headers=preflight_headers(PACKAGED_ORIGIN)
        )
        assert pre.status_code == 200
        actual = client.get(
            "/api/v1/health",
            headers={"Host": LOOPBACK_HOST, "Origin": PACKAGED_ORIGIN},
        )
        assert actual.status_code == 401

    def test_actual_trusted_response_carries_cors_headers(self, client):
        r = client.get(
            "/api/v1/health",
            headers={
                "Host": LOOPBACK_HOST,
                "Origin": PACKAGED_ORIGIN,
                **auth_headers(),
            },
        )
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == PACKAGED_ORIGIN

    def test_guarded_error_response_carries_cors_headers(self, client):
        # the 401/403 answers stay READABLE to the trusted origin's
        # frontend — the precondition for honest client-side error
        # classification (the D-3 chain)
        r = client.get(
            "/api/v1/health",
            headers={"Host": LOOPBACK_HOST, "Origin": PACKAGED_ORIGIN},
        )
        assert r.status_code == 401
        assert r.headers.get("access-control-allow-origin") == PACKAGED_ORIGIN


# --------------------------------- the unchanged lanes (P5..P8 pinned)


class TestLoopbackLanesUnchanged:
    def test_p5_loopback_without_token_is_401(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 401

    def test_p6_loopback_with_token_is_200(self, client):
        r = client.get("/api/v1/health", headers=auth_headers())
        assert r.status_code == 200

    @pytest.mark.parametrize("host", ["10.0.0.5:9999", "tauri.localhost", "evil.example"])
    def test_p7_p8_non_loopback_host_is_403_even_with_token(self, client, host):
        r = client.get(
            "/api/v1/health",
            headers={
                "Host": host,
                "Origin": PACKAGED_ORIGIN,
                **auth_headers(),
            },
        )
        assert r.status_code == 403
