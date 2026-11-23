"""part15 §4.2 (D-6) — the WebSocket handshake Origin validation.

The REST lane has enforced the trusted-origin policy since part14
D-1/D-2 (``guard_request``: a present Origin/Referer must be loopback
or an exact trusted app origin); the WS lane verified only the session
token (close 4401) — the asymmetry the part14 register carried as
**D-6 OPEN ("design decision queued")** and part15 §20 elevates to the
headline recheck.

The designed policy (part15 §4.2's five investigation questions
answered; NOT a mechanical CORS-middleware copy — a pre-accept header
check on the handshake itself):

* **Is a malicious website able to open the WS?**  Browsers cannot
  omit or forge the WS handshake ``Origin`` header (RFC 6455 sends it
  as a plain HTTP header of the handshake request) — but they CAN open
  a socket to ``ws://127.0.0.1:<port>`` cross-origin, and the token
  was the only gate.  With a token leaked through a future XSS or a
  hostile browser extension, an evil page could have subscribed to the
  full event stream (job progress, dataset names, result payloads) —
  a real exfiltration channel.  Defense-in-depth is justified.
* **Is Origin validation needed in addition to the token?**  Yes —
  the token is secret-but-single-gate; the Origin check adds an
  independent browser-context boundary the page cannot choose.
* **Is the packaged Tauri origin supported?**  ``http(s)://
  tauri.localhost`` joins the trusted set exactly as in REST (the
  packaged WebView2 page's WS to the loopback sidecar is cross-origin,
  so it presents that Origin) — plus every loopback hostname (the dev
  Vite lane, any port).
* **Do non-browser local clients stay compatible?**  An ABSENT
  Origin stays the non-browser lane (the Rust host, ``curl``, the
  smoke lane's Python clients, the existing test suite) — allowed,
  token still required, zero breakage.
* **Nothing is weakened:** the change only ADDS a refusal lane
  (present-but-untrusted Origin → close 4403 before accept); every
  existing acceptance (absent Origin, loopback, trusted app origins,
  token on all lanes) is unchanged.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from pharmaml.api import create_app
from pharmaml.api.security import TRUSTED_APP_ORIGINS

TEST_TOKEN = "test-session-token-0123456789"

#: The packaged WebView2 frontend's page origin (Tauri 2 on Windows).
PACKAGED_ORIGIN = "http://tauri.localhost"
PACKAGED_ORIGIN_HTTPS = "https://tauri.localhost"

#: A loopback dev origin (the Vite dev server, any port).
DEV_ORIGIN = "http://localhost:5173"


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def ws_url(token: str = TEST_TOKEN) -> str:
    return f"/api/v1/events?token={token}"


# ------------------------------------------------ the acceptance lanes


class TestWsAcceptanceLanes:
    def test_trusted_app_origin_with_token_connects(self, client):
        # the packaged app's lane: WS from the WebView2 page is
        # cross-origin to 127.0.0.1 and presents tauri.localhost
        import json as _json

        with client.websocket_connect(
            ws_url(), headers={"Origin": PACKAGED_ORIGIN}
        ) as ws:
            hello = _json.loads(ws.receive_text())
            assert hello["payload"]["hello"] is True

    def test_packaged_https_origin_with_token_connects(self, client):
        with client.websocket_connect(
            ws_url(), headers={"Origin": PACKAGED_ORIGIN_HTTPS}
        ) as ws:
            assert "hello" in ws.receive_text()

    def test_loopback_dev_origin_any_port_with_token_connects(self, client):
        # the dev lane: hostname loopback, any port — same rule as REST
        with client.websocket_connect(
            ws_url(), headers={"Origin": DEV_ORIGIN}
        ) as ws:
            assert "hello" in ws.receive_text()

    def test_absent_origin_non_browser_lane_connects(self, client):
        # the Rust host / curl / test-suite lane: no Origin header at
        # all — unchanged by the D-6 policy
        with client.websocket_connect(ws_url()) as ws:
            assert "hello" in ws.receive_text()

    def test_trusted_origin_set_is_exact(self):
        assert TRUSTED_APP_ORIGINS == frozenset(
            {"http://tauri.localhost", "https://tauri.localhost"}
        )


# ------------------------------------------------ the refusal lanes


class TestWsRefusalLanes:
    def test_evil_origin_with_VALID_token_refused(self, client):
        # THE D-6 CORE: even a leaked/valid token must not open the
        # event stream from a hostile browser page
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                ws_url(), headers={"Origin": "https://evil.example"}
            ):
                pass

    def test_evil_origin_without_token_refused(self, client):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/api/v1/events",
                headers={"Origin": "https://evil.example"},
            ):
                pass

    def test_lookalike_subdomain_refused(self, client):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                ws_url(), headers={"Origin": "http://evil.tauri.localhost"}
            ):
                pass

    def test_lookalike_suffix_host_refused(self, client):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                ws_url(),
                headers={"Origin": "http://tauri.localhost.evil.example"},
            ):
                pass

    def test_port_bearing_trusted_host_refused(self, client):
        # trusted app origins are port-free EXACTLY (REST parity);
        # loopback-with-port is fine (test above), tauri.localhost:8080
        # is not the packaged origin
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                ws_url(),
                headers={"Origin": "http://tauri.localhost:8080"},
            ):
                pass

    def test_referer_lane_equally_enforced(self, client):
        # REST parity: a present Referer is held to the same policy —
        # a hostile page cannot dodge the Origin check by stripping
        # it and leaking a Referer instead
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                ws_url(),
                headers={"Referer": "https://evil.example/page"},
            ):
                pass

    def test_file_scheme_origin_refused(self, client):
        # non-http(s) schemes (file://, ws-only strings, garbage) are
        # never trusted app origins
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                ws_url(), headers={"Origin": "file://tauri.localhost"}
            ):
                pass
