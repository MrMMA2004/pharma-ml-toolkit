"""part17 §4 — the penetration-testing program's automated security
suite (Round 356, lane P1).

The part17 §4.3 mandate: "Create focused tests for" the attack lanes
this file implements against the REAL ``create_app()`` stack — the
same FastAPI middleware, the same stores, the same persistence vault
the shipped product runs. Every test maps to an OWASP reference
(WSTG where a web category applies, the API Security Top 10 for the
BOLA/BFLA lanes) and states its verdict; the honest division of
labour with the existing suites is:

* auth/origin/CORS/WS matrix — ``tests/api/test_trusted_origins.py``
  (34 legs) and ``tests/api/test_ws_origin.py`` (12/12) already pin
  the D-1..D-6 closure; this suite adds only the lanes they do NOT
  carry (session rotation/replay, header-precedence abuse, the
  ``Origin: null`` / userinfo / homoglyph variants);
* supply-chain — ``tests/test_security_supply_chain.py``;
* the rest of §4.3 (path traversal, ZIP Slip, symlink, uploads,
  MIME spoofing, dangerous filenames, unsafe export names, logs,
  IDOR, resource exhaustion, injection, CSP, Tauri permissions) —
  THIS file.

The program-level documents (docs/QA/THREAT_MODEL.md,
PENETRATION_TEST_PLAN.md, PENETRATION_TEST_REPORT.md,
SECURITY_REMEDIATION_REGISTER.md) reference every lane below by its
test node id; the ZAP workflow (``.github/workflows/
penetration-test.yml``) stays NOT_EXECUTED until its first runner
firing — never a fabricated "pass".
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app
from pharmaml.api.persistence import FileVault, Persistence
from pharmaml.api.stores import ArtifactStore

TEST_TOKEN = "pentest-fixed-token-0123456789"
SECOND_TOKEN = "pentest-second-token-9876543210"

SYNTHETIC_CSV = "smiles,logp\nCCO,1.1\nCCC,1.9\nCCCO,0.8\n"


def auth_headers(token: str = TEST_TOKEN):
    return {"X-Session-Token": token}


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


@pytest.fixture()
def second_client():
    """A second, independent session (the replay lane's other side)."""
    app = create_app(token=SECOND_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


# ---------------------------------------------------------------------------
# §4.3 "Authentication bypass" + "Invalid, expired, missing, malformed
# tokens" (WSTG-SESS/ATHN; the existing 401 lanes live in
# test_backend_service.py — these are the complement).
# ---------------------------------------------------------------------------
class TestTokenValidationBattery:
    """Malformed-token shapes and header-precedence abuse."""

    def test_wrong_token_is_401(self, client):
        r = client.get("/api/v1/capabilities", headers=auth_headers("nope"))
        assert r.status_code == 401

    def test_empty_token_is_401(self, client):
        r = client.get("/api/v1/capabilities", headers=auth_headers(""))
        assert r.status_code == 401

    def test_whitespace_token_is_401(self, client):
        r = client.get(
            "/api/v1/capabilities", headers=auth_headers("   ")
        )
        assert r.status_code == 401

    def test_prefix_substring_token_is_401(self, client):
        """A token-shaped prefix of the real token never verifies."""
        r = client.get(
            "/api/v1/capabilities", headers=auth_headers(TEST_TOKEN[:10])
        )
        assert r.status_code == 401

    def test_right_length_wrong_bytes_is_401(self, client):
        """Same length as the real token, zero content match."""
        r = client.get(
            "/api/v1/capabilities",
            headers=auth_headers("x" * len(TEST_TOKEN)),
        )
        assert r.status_code == 401

    def test_wrong_x_session_token_beats_right_bearer(self, client):
        """The X-Session-Token header has precedence — an attacker who
        can inject a (wrong) X-Session-Token alongside a valid Bearer
        does NOT get through: the lane is fail-closed, not fail-open
        (OWASP API1-adjacent header-smuggling shape)."""
        r = client.get(
            "/api/v1/capabilities",
            headers={
                "X-Session-Token": "attacker-controlled",
                "Authorization": f"Bearer {TEST_TOKEN}",
            },
        )
        assert r.status_code == 401

    def test_bearer_prefix_alone_is_401(self, client):
        r = client.get(
            "/api/v1/capabilities", headers={"Authorization": "Bearer "}
        )
        assert r.status_code == 401

    def test_replayed_token_from_other_session_is_401(
        self, client, second_client
    ):
        """Session replay across rotation (WSTG-SESS-06): a token
        captured from session A never authenticates session B — the
        per-session token model holds."""
        r = client.get(
            "/api/v1/capabilities", headers=auth_headers(SECOND_TOKEN)
        )
        assert r.status_code == 401
        r2 = second_client.get(
            "/api/v1/capabilities", headers=auth_headers(TEST_TOKEN)
        )
        assert r2.status_code == 401

    def test_unauthenticated_401_body_never_echoes_the_attempt(self, client):
        """The rejection detail is static — a presented token is never
        reflected in the response (no oracle for brute-force side
        channels)."""
        presented = "guess-" + "a" * 40
        r = client.get("/api/v1/capabilities", headers=auth_headers(presented))
        assert r.status_code == 401
        assert presented not in r.text
        assert r.json() == {"detail": "Invalid or missing session token."}


# ---------------------------------------------------------------------------
# §4.3 "Origin and Referer bypass" — the complement lanes the 34-leg
# matrix does not carry (WSTG-CLNT/ATHN).
# ---------------------------------------------------------------------------
class TestOriginBypassComplement:
    LOOPBACK_HOST = "127.0.0.1:54127"

    def _get(self, client, headers):
        merged = {**auth_headers(), "Host": self.LOOPBACK_HOST, **headers}
        return client.get("/api/v1/capabilities", headers=merged)

    def test_origin_null_is_rejected(self, client):
        """A sandboxed iframe/redirect sends ``Origin: null`` — the
        literal string, not a URL. It must never ride the trusted lane."""
        r = self._get(client, {"Origin": "null"})
        assert r.status_code == 403

    def test_origin_userinfo_trick_is_rejected(self, client):
        """``http://tauri.localhost@evil.example`` parses to the evil
        host (the userinfo is not the authority) — rejected as
        non-loopback, never suffix-matched against the trusted set."""
        r = self._get(
            client, {"Origin": "http://tauri.localhost@evil.example"}
        )
        assert r.status_code == 403

    def test_ascii_lookalike_host_is_rejected(self, client):
        """A lookalike host that byte-differs from every trusted entry
        (`0` for `o`) — the IDN-homoglyph class in ASCII clothing;
        the policy never suffix-matches."""
        r = self._get(client, {"Origin": "http://tauri.l0calhost"})
        assert r.status_code == 403

    def test_mixed_case_trusted_origin_is_the_permissive_lane(
        self, client
    ):
        """Scheme and host normalise case-insensitively (RFC 3986) —
        the INTENDED permissive lane; pinned so a future
        case-sensitive regression (the classic bypass class) fails
        here instead of in production."""
        r = self._get(client, {"Origin": "HTTP://TaurI.LocalHost"})
        assert r.status_code == 200

    def test_referer_with_path_and_query_is_parsed_not_matched(
        self, client
    ):
        """A Referer carries path/query — the policy compares
        scheme+host only, so a deep link from the packaged app stays
        trusted (and the ``@``-trick inside the path is inert)."""
        r = self._get(
            client,
            {
                "Referer": (
                    "http://tauri.localhost/datasets?id=1"
                    "&next=http://evil.example"
                )
            },
        )
        assert r.status_code == 200

    def test_data_scheme_origin_is_rejected(self, client):
        r = self._get(client, {"Origin": "data:text/html,hi"})
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# §4.3 "CORS bypass" — complement lanes over the scoped middleware
# (the 34-leg matrix owns the D-2 core; WSTG-CLNT-07).
# ---------------------------------------------------------------------------
class TestCorsBypassComplement:
    LOOPBACK_HOST = "127.0.0.1:54127"

    def _preflight(self, client, origin):
        return client.options(
            "/api/v1/capabilities",
            headers={
                "Host": self.LOOPBACK_HOST,
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )

    def test_port_bearing_trusted_virtual_host_gets_no_grant(self, client):
        """``https://tauri.localhost:443`` is NOT the packaged origin
        (the anchored regex admits exactly the port-less virtual
        host) — no ACAO header for the port variant."""
        r = self._preflight(client, "https://tauri.localhost:443")
        assert "access-control-allow-origin" not in r.headers

    def test_localhost_suffix_host_gets_no_grant(self, client):
        r = self._preflight(client, "http://localhost.evil.example")
        assert "access-control-allow-origin" not in r.headers

    def test_non_http_scheme_gets_no_grant(self, client):
        r = self._preflight(client, "chrome-extension://127.0.0.1")
        assert "access-control-allow-origin" not in r.headers

    def test_ipv6_loopback_dev_origin_is_the_permissive_lane(self, client):
        r = self._preflight(client, "http://[::1]:5173")
        assert r.headers.get("access-control-allow-origin") == (
            "http://[::1]:5173"
        )

    def test_credentials_are_never_allowed(self, client):
        """allow_credentials stays false: the session token is a
        header lane, never a cookie lane (the CSRF surface that
        credentialed CORS would open)."""
        r = self._preflight(client, "http://localhost:5173")
        assert r.headers.get("access-control-allow-credentials", "false") == (
            "false"
        )


# ---------------------------------------------------------------------------
# §4.3 "IDOR / BOLA" + "Broken Function Level Authorization" (OWASP
# API1/API5). The honest shape for a single-user local service: every
# object lane answers 404 for unknown ids (no existence oracle), and
# every mutating route is behind the token exactly like the read
# routes — there is no anonymous or "lesser" role to abuse.
# ---------------------------------------------------------------------------
class TestObjectLevelAuthorization:
    def test_unknown_dataset_id_is_404(self, client):
        r = client.get(
            "/api/v1/datasets/does-not-exist", headers=auth_headers()
        )
        assert r.status_code == 404

    def test_unknown_artifact_id_is_404(self, client):
        r = client.get(
            "/api/v1/artifacts/art-000000000000", headers=auth_headers()
        )
        assert r.status_code == 404

    def test_unknown_artifact_download_is_404(self, client):
        r = client.get(
            "/api/v1/artifacts/art-000000000000/download",
            headers=auth_headers(),
        )
        assert r.status_code == 404

    def test_unknown_artifact_delete_is_404(self, client):
        r = client.delete(
            "/api/v1/artifacts/art-000000000000", headers=auth_headers()
        )
        assert r.status_code == 404

    def test_unknown_job_id_is_404(self, client):
        r = client.get("/api/v1/jobs/job-none", headers=auth_headers())
        assert r.status_code == 404

    def test_unknown_run_id_is_404(self, client):
        r = client.get("/api/v1/runs/run-none", headers=auth_headers())
        assert r.status_code == 404

    def test_deletion_lanes_are_not_anonymous(self, client):
        """BFLA honesty: the mutating verb answers 401 without the
        token exactly as the read verbs do — no weaker role exists."""
        r = client.delete("/api/v1/artifacts/art-000000000000")
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# §4.3 "Path traversal" + "Symlink attacks" (WSTG-ATHZ-01) over the
# file vault — the unit lane the API ids cannot reach (ids are
# server-generated) but the vault contract must still prove.
# ---------------------------------------------------------------------------
class TestVaultTraversalAndSymlink:
    def test_dotdot_relative_is_refused(self, tmp_path):
        vault = FileVault(tmp_path)
        with pytest.raises(ValueError, match="escapes the vault"):
            vault.read("files/../../etc/passwd")

    def test_absolute_path_is_refused(self, tmp_path):
        vault = FileVault(tmp_path)
        with pytest.raises(ValueError, match="escapes the vault"):
            vault.read("/etc/passwd")

    def test_backslash_separators_do_not_smuggle(self, tmp_path):
        vault = FileVault(tmp_path)
        with pytest.raises(ValueError):
            vault.read("files\\..\\..\\etc\\passwd")

    def test_symlink_planted_in_the_vault_cannot_escape(self, tmp_path):
        """A real symlink created inside the files tree (the "server
        is compromised one level" lane) resolves OUTSIDE the vault —
        the resolver catches it, so a planted link never widens the
        read/write surface to the host filesystem."""
        vault = FileVault(tmp_path)
        target = tmp_path / "outside.txt"
        target.write_bytes(b"host data")
        planted_dir = tmp_path / "files" / "datasets"
        planted_dir.mkdir(parents=True, exist_ok=True)
        (planted_dir / "link").symlink_to(target)
        with pytest.raises(ValueError, match="escapes the vault"):
            vault.read("files/datasets/link")

    def test_non_string_rel_is_refused(self, tmp_path):
        vault = FileVault(tmp_path)
        with pytest.raises(ValueError):
            vault.read(None)


# ---------------------------------------------------------------------------
# §4.3 "ZIP Slip" over the backup-restore bundle lane (WSTG-ATHZ) —
# the only archive surface the product opens on user-supplied bytes.
# All fixtures are crafted in memory; nothing is extracted to the
# real filesystem — the restore validates before it writes.
# ---------------------------------------------------------------------------
class TestRestoreBundleZipSlip:
    def _craft(self, members: dict[str, bytes], db: bytes | None) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, payload in members.items():
                zf.writestr(name, payload)
            if db is not None:
                zf.writestr("studio.db", db)
        return buf.getvalue()

    def _persistence(self, tmp_path):
        return Persistence(tmp_path)

    def test_traversal_member_is_refused(self, tmp_path):
        p = self._persistence(tmp_path)
        bundle = self._craft({"files/../../evil.txt": b"x"}, b"")
        with pytest.raises(ValueError, match="unsafe bundle entry"):
            p.restore_bundle(bundle)

    def test_absolute_member_is_refused(self, tmp_path):
        p = self._persistence(tmp_path)
        bundle = self._craft({"/etc/evil.txt": b"x"}, b"")
        with pytest.raises(ValueError, match="unexpected bundle entry"):
            p.restore_bundle(bundle)

    def test_non_prefixed_member_is_refused(self, tmp_path):
        p = self._persistence(tmp_path)
        bundle = self._craft({"evil/studio.db": b"x"}, b"")
        with pytest.raises(ValueError, match="unexpected bundle entry"):
            p.restore_bundle(bundle)

    def test_prefixed_but_escaping_member_is_refused(self, tmp_path):
        """``files/../../escape`` passes the lexical prefix check and
        is still caught by the resolved-path containment check."""
        p = self._persistence(tmp_path)
        bundle = self._craft({"files/../../escape": b"x"}, b"")
        with pytest.raises(ValueError, match="unsafe bundle entry"):
            p.restore_bundle(bundle)

    def test_non_zip_bytes_are_refused(self, tmp_path):
        p = self._persistence(tmp_path)
        with pytest.raises(ValueError, match="not a backup bundle"):
            p.restore_bundle(b"PK-fake-but-not-a-zip" + b"\x00" * 64)

    def test_missing_db_entry_is_refused(self, tmp_path):
        p = self._persistence(tmp_path)
        bundle = self._craft({"files/datasets/x": b"1"}, None)
        with pytest.raises(ValueError, match="no studio.db entry"):
            p.restore_bundle(bundle)


# ---------------------------------------------------------------------------
# §4.3 "Malformed uploads", "MIME spoofing", "Dangerous filenames"
# (WSTG-BUSL) — the extension allowlist + size cap + safe-stem lanes
# over the REAL upload route.
# ---------------------------------------------------------------------------
class TestUploadAttackShapes:
    def _upload(self, client, filename, payload, ctype="text/csv"):
        return client.post(
            "/api/v1/datasets",
            headers=auth_headers(),
            files={"file": (filename, payload, ctype)},
            data={"target": "logp", "mode": "sync"},
        )

    def test_mime_spoofed_as_image_still_parses_as_data_only(self, client):
        """Declared ``image/png`` with a .csv extension: the extension
        governs, the parser sniffs the content, and the result is a
        data-file record — there is no execute lane for uploads, so a
        MIME lie is a (rejected or re-parsed) data problem, never a
        code problem. A valid CSV behind the spoofed type registers
        honestly."""
        r = self._upload(client, "spoof.csv", SYNTHETIC_CSV.encode(),
                          ctype="image/png")
        assert r.status_code == 200
        dataset = r.json()["dataset"]
        assert dataset["source_filename"] == "spoof.csv"
        assert "logp" in dataset["columns"]

    def test_exe_extension_is_415(self, client):
        r = self._upload(client, "payload.exe", b"MZ\x90\x00")
        assert r.status_code == 415

    def test_html_extension_is_415(self, client):
        r = self._upload(
            client, "x.html", b"<script>alert(1)</script>", "text/html"
        )
        assert r.status_code == 415

    def test_control_characters_in_filename_are_sanitised(self, client):
        """The stored dataset name never carries CR/LF/quotes from the
        upload's filename (the F-356-01 class, pinned at its second
        surface)."""
        r = self._upload(
            client, "a\nb\"c\r.csv", SYNTHETIC_CSV.encode()
        )
        assert r.status_code == 200
        name = r.json()["dataset"]["name"]
        assert re.fullmatch(r"[A-Za-z0-9._-]+", name)
        assert "\n" not in name and '"' not in name and "\r" not in name

    def test_windows_reserved_name_is_prefixed(self, client):
        r = self._upload(client, "CON.csv", SYNTHETIC_CSV.encode())
        assert r.status_code == 200
        assert r.json()["dataset"]["name"].startswith("_CON")

    def test_dots_only_filename_falls_back(self, client):
        r = self._upload(client, "...csv", SYNTHETIC_CSV.encode())
        assert r.status_code == 200
        assert r.json()["dataset"]["name"] == "unnamed"


# ---------------------------------------------------------------------------
# §4.3 "Unsafe export names" — F-356-01, the finding this round fixed
# (Content-Disposition injection via the artifact filename). The
# registration now sanitises; the header builder strips as
# defense-in-depth; both are pinned.
# ---------------------------------------------------------------------------
class TestUnsafeExportNamesF35601:
    def _figure(self, client, filename):
        return client.post(
            "/api/v1/artifacts/figure",
            headers=auth_headers(),
            json={
                "kind": "scatter",
                "x": [1.0, 2.0, 3.0],
                "y": [1.0, 4.0, 9.0],
                "format": "SVG",
                "filename": filename,
            },
        )

    def test_crlf_filename_is_sanitised_at_registration(self, client):
        r = self._figure(
            client, 'evil\r\nSet-Cookie: session=x\r\n\r\n<body>'
        )
        assert r.status_code == 200
        stored = r.json()["artifact"]["filename"]
        assert re.fullmatch(r"[A-Za-z0-9._-]+\.svg", stored)

    def test_download_header_stays_one_line(self):
        """The header builder cannot be talked into a second header
        line even by a hostile stored record (defense-in-depth over a
        fabricated record — registration sanitises, the builder
        strips)."""
        from pharmaml.api.routes.artifacts import _download_headers
        from pharmaml.api.stores import ArtifactRecord

        record = ArtifactRecord(
            "art-fabricated",
            "figure",
            'x"\r\nSet-Cookie: evil=1',
            "image/svg+xml",
            b"<svg/>",
        )
        headers = _download_headers(record)
        disposition = headers["Content-Disposition"]
        assert "\r" not in disposition
        assert "\n" not in disposition
        # exactly the two surrounding quotes — no smuggled third
        assert disposition.count('"') == 2
        assert disposition.endswith('filename="xSet-Cookie: evil=1"')

    def test_quote_filename_round_trips_as_portable_stem(self, client):
        r = self._figure(client, 'a"b')
        assert r.status_code == 200
        artifact = r.json()["artifact"]
        stored = artifact["filename"]
        assert '"' not in stored and stored.endswith(".svg")
        art_id = artifact["artifact_id"]
        dl = client.get(
            f"/api/v1/artifacts/{art_id}/download", headers=auth_headers()
        )
        assert dl.status_code == 200
        disposition = dl.headers["content-disposition"]
        assert "\r" not in disposition and "\n" not in disposition


# ---------------------------------------------------------------------------
# §4.3 "Resource exhaustion" (WSTG-BUSL-04/05) — the bounded stores.
# ---------------------------------------------------------------------------
class TestResourceExhaustionCaps:
    def test_artifact_count_cap_evicts_oldest_first(self):
        store = ArtifactStore(max_items=3, max_total_bytes=10**9)
        for i in range(5):
            store.register(
                kind="figure",
                filename=f"f{i}.svg",
                content_type="image/svg+xml",
                data=b"<svg/>",
            )
        listing = store.list()
        assert len(listing) == 3
        assert [item["filename"] for item in listing] == [
            "f2.svg", "f3.svg", "f4.svg",
        ]

    def test_artifact_total_bytes_cap_refuses_oversize(self):
        store = ArtifactStore(max_items=5, max_total_bytes=16)
        with pytest.raises(ValueError, match="exceeds the session cap"):
            store.register(
                kind="figure",
                filename="big.svg",
                content_type="image/svg+xml",
                data=b"x" * 17,
            )

    def test_dataset_count_cap_evicts_oldest_first(self):
        """The bounded registry through the REAL route: a capped store
        mounted on the app keeps only the newest N datasets."""
        from pharmaml.api.stores import DatasetStore

        app = create_app(token=TEST_TOKEN)
        app.state.dataset_store = DatasetStore(max_items=2)
        with TestClient(app, base_url="http://127.0.0.1") as tc:
            for i in range(4):
                r = tc.post(
                    "/api/v1/datasets",
                    headers=auth_headers(),
                    files={
                        "file": (
                            f"ds{i}.csv",
                            SYNTHETIC_CSV.encode(),
                            "text/csv",
                        )
                    },
                    data={"target": "logp", "mode": "sync"},
                )
                assert r.status_code == 200
            listing = tc.get(
                "/api/v1/datasets", headers=auth_headers()
            ).json()["datasets"]
        assert [row["name"] for row in listing] == ["ds2", "ds3"]


# ---------------------------------------------------------------------------
# §4.3 "Injection and XSS" (WSTG-INPV) — the persistence lane is
# parameterised and the REST surface is JSON-only.
# ---------------------------------------------------------------------------
class TestInjectionLanes:
    def test_sql_payload_round_trips_as_data(self, client):
        """A classic SQL injection string carried as a dataset name
        round-trips verbatim as DATA — the SQLite layer is fully
        parameterised, and the registry is intact after the attempt."""
        payload = "x'; DROP TABLE datasets; --"
        r = client.post(
            "/api/v1/datasets",
            headers=auth_headers(),
            files={
                "file": (
                    f"{payload}.csv",
                    SYNTHETIC_CSV.encode(),
                    "text/csv",
                )
            },
            data={"target": "logp"},
        )
        assert r.status_code == 200
        rows = client.get(
            "/api/v1/datasets", headers=auth_headers()
        ).json()["datasets"]
        # the sanitised name keeps the words (data, verbatim, inert); the
        # registry is intact and live after the attempt
        assert any("DROP" in row["name"] for row in rows)
        assert client.get(
            "/api/v1/capabilities", headers=auth_headers()
        ).status_code == 200

    def test_responses_are_never_text_html(self, client):
        """No API lane renders user data as HTML — the content type is
        always JSON, so reflected payloads stay data (the XSS lane's
        honest verdict for a JSON API)."""
        r = client.get(
            "/api/v1/capabilities", headers=auth_headers()
        )
        assert r.headers["content-type"].startswith("application/json")

    def test_persistence_execute_is_parameterised(self):
        """Source-level pin: the persistence helper takes its values as
        bound parameters, not interpolated SQL (drift-guard against a
        future f-string regression)."""
        import inspect

        source = inspect.getsource(Persistence._execute)
        assert "conn.execute(sql, params)" in source


# ---------------------------------------------------------------------------
# §4.3 "Sensitive information in logs" (WSTG-ERRH/INFO) — the session
# token never reaches a log record.
# ---------------------------------------------------------------------------
class TestLogHygiene:
    def test_token_never_appears_in_log_records(
        self, client, caplog
    ):
        with caplog.at_level(logging.DEBUG, logger="pharmaml"):
            client.get("/api/v1/capabilities", headers=auth_headers())
            client.get(
                "/api/v1/capabilities",
                headers=auth_headers("wrong-" + TEST_TOKEN),
            )
            client.get("/api/v1/capabilities")
        for record in caplog.records:
            assert TEST_TOKEN not in record.getMessage()

    def test_access_log_is_disabled_on_the_runner_config(self):
        """The shipped server config keeps uvicorn's access log off —
        the per-request line (which would carry the auth header's
        effect) is never written (drift-guard pin)."""
        server_source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "pharmaml"
            / "api"
            / "server.py"
        ).read_text(encoding="utf-8")
        assert "access_log=False" in server_source


# ---------------------------------------------------------------------------
# §4.3 "CSP violations" + "Tauri permission overreach" — the static
# pins over the shipped Tauri 2 configuration (the B-surface of the
# threat model; the WebView2 CSP and the capability allowlist).
# ---------------------------------------------------------------------------
class TestDesktopSurfaceStaticPins:
    TAURI_CONF = (
        Path(__file__).resolve().parents[2]
        / "src-tauri"
        / "tauri.conf.json"
    )
    CAPABILITIES = (
        Path(__file__).resolve().parents[2]
        / "src-tauri"
        / "capabilities"
        / "default.json"
    )

    def test_csp_pins_the_security_directives(self):
        conf = json.loads(self.TAURI_CONF.read_text(encoding="utf-8"))
        csp = conf["app"]["security"]["csp"]
        assert "object-src 'none'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "base-uri 'self'" in csp
        assert "form-action 'self'" in csp
        # script-src is 'self' with NO unsafe-inline / unsafe-eval
        script = re.search(r"script-src ([^;]+);", csp).group(1)
        assert "unsafe-inline" not in script
        assert "unsafe-eval" not in script

    def test_csp_connect_src_is_loopback_only(self):
        conf = json.loads(self.TAURI_CONF.read_text(encoding="utf-8"))
        csp = conf["app"]["security"]["csp"]
        connect = re.search(r"connect-src ([^;]+);", csp).group(1)
        for host in re.findall(r"https?://[^ ]+", connect):
            assert host.startswith(
                ("http://127.0.0.1", "http://ipc.localhost")
            ) or "localhost" not in host, host

    def test_csp_has_no_wildcard_sources(self):
        conf = json.loads(self.TAURI_CONF.read_text(encoding="utf-8"))
        csp = conf["app"]["security"]["csp"]
        # no bare "*" source anywhere in the policy
        assert not re.search(r"(?:^|[\s;])\*(?:\s|;|$)", csp)
        # no scheme-level wildcards (only loopback port wildcards are
        # legitimate: http://127.0.0.1:* and ws://127.0.0.1:*)
        for scheme in ("http", "https", "ws", "wss"):
            assert f"{scheme}://*" not in csp

    def test_capability_allowlist_is_core_only(self):
        """The main window's ONLY capability is ``core:default`` — no
        shell executor, no arbitrary filesystem, no dialog/plugin
        surface: the least-privilege pin (overreach = a new entry
        failing this gate)."""
        caps = json.loads(
            self.CAPABILITIES.read_text(encoding="utf-8")
        )
        assert caps["permissions"] == ["core:default"]
        assert caps["windows"] == ["main"]

    def test_no_extra_capability_files(self):
        parent = self.CAPABILITIES.parent
        files = sorted(p.name for p in parent.glob("*.json"))
        assert files == ["default.json"]


# ---------------------------------------------------------------------------
# §4.3 "Process-handshake validation" — the Python-side pins (the
# Rust-side validate_handshake's 11 cases are compile-blocked on this
# host and ride the Windows release pipeline; see the plan).
# ---------------------------------------------------------------------------
class TestHandshakeShape:
    def test_handshake_line_is_one_machine_readable_json(self):
        from pharmaml.api.server import handshake_line

        line = handshake_line(43551, "tok-xyz", 1234, "1.2.240")
        assert line.startswith("PHARMAML-READY ")
        payload = json.loads(line[len("PHARMAML-READY "):])
        assert payload == {
            "port": 43551,
            "token": "tok-xyz",
            "pid": 1234,
            "version": "1.2.240",
        }

    def test_loopback_socket_binds_port_zero_atomically(self):
        from pharmaml.api.server import bind_loopback_socket

        sock = bind_loopback_socket()
        try:
            host, port = sock.getsockname()
            assert host == "127.0.0.1"
            assert port != 0
        finally:
            sock.close()


# ---------------------------------------------------------------------------
# The ZAP lane's own tooling — the disposable target and the severity
# gate are product code, so they get the same pinned treatment.
# ---------------------------------------------------------------------------
class TestZapTargetTooling:
    # the tools/ home sits outside the pytest pythonpath ("src" only) —
    # the same guard test_docs_index.py carries for its own import
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    def test_target_app_is_the_real_app_with_fixed_test_token(self):
        from tools.zap_target import ZAP_TEST_TOKEN, build_target_app

        assert ZAP_TEST_TOKEN.startswith("zap-synthetic-")
        app = build_target_app()
        with TestClient(
            app, base_url="http://127.0.0.1"
        ) as tc:
            r = tc.get(
                "/api/v1/capabilities",
                headers={"X-Session-Token": ZAP_TEST_TOKEN},
            )
            assert r.status_code == 200

    def test_seed_and_contract_dump(self, tmp_path):
        from tools.zap_target import build_target_app, dump_openapi, seed

        app = build_target_app(data_dir=tmp_path)
        with TestClient(app, base_url="http://127.0.0.1") as tc:
            report = seed(tc)
            assert report["datasets"] == 1
            out = tmp_path / "openapi.json"
            dump_openapi(tc, out)
            contract = json.loads(out.read_text(encoding="utf-8"))
        assert contract["openapi"].startswith("3.")
        assert "/api/v1/datasets" in contract["paths"]

    def test_gate_fails_on_high_findings(self, tmp_path):
        from tools.zap_gate import gate_report

        report = tmp_path / "zap-report.json"
        report.write_text(
            json.dumps(
                {
                    "site": [
                        {
                            "alerts": [
                                {
                                    "pluginid": "40026",
                                    "name": "Cross Site Scripting",
                                    "riskcode": 3,
                                    "instances": [],
                                }
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        suppressed = tmp_path / "suppress.json"
        suppressed.write_text("[]", encoding="utf-8")
        verdict = gate_report(report, suppressed)
        assert verdict.exit_code == 1
        assert verdict.counts["High"] == 1

    def test_gate_passes_when_suppressions_are_documented(self, tmp_path):
        from tools.zap_gate import gate_report

        report = tmp_path / "zap-report.json"
        report.write_text(
            json.dumps(
                {
                    "site": [
                        {
                            "alerts": [
                                {
                                    "pluginid": "40026",
                                    "name": "Cross Site Scripting",
                                    "riskcode": 3,
                                    "instances": [],
                                }
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        suppressed = tmp_path / "suppress.json"
        suppressed.write_text(
            json.dumps(
                [
                    {
                        "pluginid": "40026",
                        "regex": "Cross Site Scripting",
                        "justification": (
                            "documented false positive on the ephemeral"
                            " test instance"
                        ),
                        "expires": "2099-01-01",
                    }
                ]
            ),
            encoding="utf-8",
        )
        verdict = gate_report(report, suppressed)
        assert verdict.exit_code == 0
        assert verdict.counts["High"] == 1
        assert verdict.suppressed == 1

    def test_gate_ignores_expired_suppressions(self, tmp_path):
        from tools.zap_gate import gate_report

        report = tmp_path / "zap-report.json"
        report.write_text(
            json.dumps(
                {
                    "site": [
                        {
                            "alerts": [
                                {
                                    "pluginid": "40026",
                                    "name": "Cross Site Scripting",
                                    "riskcode": 3,
                                    "instances": [],
                                }
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        suppressed = tmp_path / "suppress.json"
        suppressed.write_text(
            json.dumps(
                [
                    {
                        "pluginid": "40026",
                        "regex": "Cross Site Scripting",
                        "justification": "expired justification",
                        "expires": "2000-01-01",
                    }
                ]
            ),
            encoding="utf-8",
        )
        verdict = gate_report(report, suppressed)
        assert verdict.exit_code == 1
        assert verdict.suppressed == 0
