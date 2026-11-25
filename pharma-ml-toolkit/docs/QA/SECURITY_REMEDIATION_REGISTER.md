# SECURITY REMEDIATION REGISTER — part17 §4.5 deliverable (Round 356, lane P1)

The register of every security finding the program produces, with
severity, status, evidence, remediation, and retest — plus the
accepted-risk rows the threat model documents. Statuses: OPEN,
FIXED+PINNED, ACCEPTED (documented), SUPPRESSED (only via the
documented channel in `packaging/security/zap-suppressions.json`).

| ID | Finding | Severity | Status | Evidence / Reproduction | Remediation | Retest |
| --- | --- | --- | --- | --- | --- | --- |
| F-356-01 | Content-Disposition injection via artifact filenames (raw `Path(base).stem` kept CR/LF/quotes into the download header) | Low | **FIXED+PINNED** | pre-fix: `POST /api/v1/artifacts/figure` with `filename='evil\r\nSet-Cookie: x\r\n\r\n<body>'` stored the CRLF; the download header would carry the smuggled line | `routes/artifacts.py::_safe_stem` (`safe_filename` at all three registration sites) + `_download_headers` CR/LF/quote strip (defense-in-depth) | `tests/api/test_penetration_security.py::TestUnsafeExportNamesF35601` — 3/3 green (2026-11-25) |
| F-356-02 | The root CI push-trigger filter reads `branches: ain]` (the `[` of `[main]` lost) — cosmetic defect; pushes to `main` still fire CI in practice | Informational | **OPEN** | the ci.yml trigger block; the Actions API runs 371–376 (`push`, `success`, at the terminal SHAs) | proposed `branches: [main]`; deliberately not applied — trigger breadth is an Actions-minutes (billing-adjacent) surface on the owner's account | pending the owner's preference; the observed behavior re-verified this round |
| A-01 | No TLS on the backend transport | — | **ACCEPTED** | the product binds loopback only; the token never crosses a network segment it could be sniffed on | none (design) | the loopback-bind lanes |
| A-02 | Single-user product, no authorization model (the API5/BFLA lane degenerates to "every route equally token-gated") | — | **ACCEPTED** | a local single-user scientific workbench by design | none (documented) | `TestObjectLevelAuthorization` |
| A-03 | `style-src 'unsafe-inline'` in the desktop CSP | — | **ACCEPTED** | the Tauri inline-style requirement; `script-src` stays strict | none (documented) | `TestDesktopSurfaceStaticPins` |
| A-04 | The Rust-side handshake validation cases are compile-blocked on this host | — | **ACCEPTED (environment)** | no Windows toolchain in the sandbox; the cases ride the Windows release pipeline | the pipeline's `cargo test` lane | the release pipeline |

Standing verified rows (the lanes this program inherits as
evidence — see the plan's test-ID map): the D-1..D-6 origin/CORS/WS
closure (`test_trusted_origins.py` 34 legs, `test_ws_origin.py`
12/12), the supply-chain guard (8/8 + the CI security lanes), the
vault traversal lanes, the upload validation lanes, the secret-pattern
tree scan (the PAT stays disk-only, `.gitignore`-quarantined), and
the CodeQL analyze suite inside the 19/19 terminal checks.

## Suppression channel

The ONLY suppression mechanism is
`packaging/security/zap-suppressions.json` (ships empty). Entries
require `pluginid` + `regex` + `justification` + `expires` (ISO
date); an expired entry is silently ignored by `tools/zap_gate.py`
(the finding returns and the gate fails again). Blanket suppressions
are never valid.
