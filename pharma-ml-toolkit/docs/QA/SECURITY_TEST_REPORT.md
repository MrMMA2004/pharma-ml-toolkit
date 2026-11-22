# SECURITY TEST REPORT — part14 §13 deliverable #6

The part14 §11 security review of the desktop/backend boundary,
instantiated with the verified state. Every verified row names its
evidence; unverified rows stay open — least privilege throughout.

Seeded: round 329 (2026-11-22); the underlying fixes are the r327
D-1..D-5 incident close-out, re-pinned r329.

## Verified legs

| §11 item | Status | Evidence |
| --- | --- | --- |
| Loopback-only backend binding | **PASS** | `security.py` Host policy; P7/P8 legs (non-loopback Host 403); pinned by `test_trusted_origins.py` |
| Trusted Host validation | **PASS** | same matrix; lookalike/subdomain/port variants 403 |
| Trusted Origin validation (D-1) | **FIXED + REGRESSION-PINNED** | `TRUSTED_APP_ORIGINS` (scheme-strict, host-exact, port-less: `http(s)://tauri.localhost`); P1/P2 403→200; 34/34 legs r329 |
| CORS preflight handling (D-2) | **FIXED + REGRESSION-PINNED** | scoped `CORSMiddleware` (anchored regex; exact-origin echo; `allow_credentials=False`; token-header allowlist); preflight 405→200 trusted only; the lane never authenticates (preflight-then-401 proof) |
| Session token lifecycle | **PASS (REST)** | memory-only token; 401 lane pinned; delivered header-only |
| Secret redaction (D-4) | **FIXED (source; compile-blocked)** | `redacted_parse_failure()` — raw `PHARMAML-READY` never reaches an error surface (serde error + byte length only); `cargo test` rides the toolchain |
| Handshake validation (D-5) | **FIXED (source; compile-blocked)** | `validate_handshake()` (port ≠ 0; token 16..=128 urlsafe; pid ≠ 0; short numeric-semver); 11 `#[cfg(test)]` cases written |
| WebSocket authentication | **PASS (token) / OPEN (origin)** | token enforced at WS handshake (close 4401, consistent with REST); D-6: WS-side origin policy is a designed extension — OPEN in the register |
| CSP restrictions | **VERIFIED (audit)** | `connect-src` already permits only loopback http/ws (r326 audit: CSP was NOT the blocker) |
| Filesystem path validation / traversal | **PASS** | path-policy lanes in `tests/api/` (the 449+1s suite) |
| Error-reporting honesty (D-3) | **FIXED + TEST-PINNED** | `StartupFailureCause` classification (auth/network/http/protocol/handshake); 9-test suite; vitest 119/119; tsc clean |
| Supply-chain guard | **PASS** | CI-side `security-scan.yml` (pip-audit + npm audit + the local guard) — the r324 deliverable, CI-green at the terminal |
| Secret scanning (repo hygiene) | **ENFORCED LIVE** | GitHub push protection refused a machinery ghost commit that had swept the owner's token upload (r329); the ghost was reset away and the payload re-pushed clean; `upload/GitHub_token.*` + `.credentials/` now gitignored disk-only |

## Open / queued

| Item | Status |
| --- | --- |
| File upload/download restrictions deep-audit | queued (the r326 §G tail) |
| SQLite safety / FileVault integrity deep-audit | partial (suite lanes green); deep-audit queued |
| Installer contents + signing | BLOCKED — the signing lane (r325 lane #3) |
| Tauri capability permissions review | queued with the box (no broad FS/shell grants without documented need — §11 rule held; none added) |
| AgentShield findings manual review | no AgentShield in this harness (r326 §4.1 answer: the ECC/harness lane is dry-run-proposed, not installed) |

## Regression discipline

Security fixes carry regression tests before any FIXED status
(the D-1/D-2/D-3 rows above); D-4/D-5 stay compile-blocked and are
NEVER marked passed until a real `cargo test` (§16.14).

— seeded round 329; cross-reference `DEFECT_REGISTER.md` for D-1..D-6.
