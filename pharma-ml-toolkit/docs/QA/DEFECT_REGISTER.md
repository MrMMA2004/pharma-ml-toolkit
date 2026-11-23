# DEFECT REGISTER — part14 §13.2

The part14-era defect register, seeded from the Phase-1 audit
(`PART14_PHASE1_AUDIT.md` — the D-1…D-5 opening rows) and maintained
per round. Every row names its evidence; a fix without evidence is not
a fix. Severity per part14 §3.1's impact classes.

| ID | Severity | Title | Status | Fixed (round) | Evidence |
| --- | --- | --- | --- | --- | --- |
| D-1 | Critical | `LOOPBACK_HOSTS` lacked `tauri.localhost` — every packaged-app REST request 403-rejected even with a valid token (dev worked via the loopback Vite origin; production WebView2 origin differs) | **FIXED** | 327 | `src/pharmaml/api/security.py` `TRUSTED_APP_ORIGINS` (scheme-strict, host-exact: `http(s)://tauri.localhost`, no port); regression `tests/api/test_trusted_origins.py` (P1/P2 legs 403→200; lookalikes/subdomains/ports stay 403); probe re-run `scratch/p14_r327_origin_probe_postfix.txt` |
| D-2 | Critical | No CORS lane — OPTIONS preflight answered 405 from ANY origin, so the browser blocked the actual request and `fetch` rejected (dev survived only via the Vite same-origin proxy) | **FIXED** | 327 | `src/pharmaml/api/__init__.py` `create_app()` — scoped `CORSMiddleware` (anchored regex: the two Tauri origins + loopback-any-port; exact-origin echo; `allow_credentials=False`; token headers allowlisted); preflight 405→200 for trusted origins (P3/P4), NO ACAO for arbitrary/lookalike origins, the 401/403 auth lanes unchanged (the lane never authenticates) |
| D-3 | High | `startup.ts` wrapped every health-probe failure as "the backend health probe failed" → the gate rendered "The backend did not start / stage: health / Failed to fetch" — mislabeling CORS/403 chains as startup crashes (§3.1 item 12) | **FIXED** | 327 | `frontend/src/services/api/startup.ts` `StartupFailureCause` (auth/network/http/protocol/handshake) + `classifyHealthFailure`; `StartupGate.tsx` cause-aware headline; 9-test suite `frontend/src/services/api/__tests__/startup.test.ts` (vitest 119/119 incl. the 6 gate tests; tsc clean) |
| D-4 | Medium | Rust handshake parse-failure path formatted the RAW `PHARMAML-READY` line (session token inside) into the error surface — a §3.3 redaction violation if reached by a log/UI message | **FIXED (source; compile-blocked)** | 327 | `src-tauri/src/backend_manager.rs` `redacted_parse_failure()` — the raw line NEVER appears (serde error + byte length only); `#[cfg(test)]` redaction test written; **no Rust toolchain in this sandbox — `cargo test` BLOCKED, rides the first cargo-enabled environment** (never marked passed) |
| D-5 | Medium | No handshake field validation in the Rust host (port bounds, pid sanity, token presence/format, version shape — §3.3's list unenforced) | **FIXED (source; compile-blocked)** | 327 | `src-tauri/src/backend_manager.rs` `validate_handshake()` (port ≠ 0; token 16..=128 bytes urlsafe-alphabet; pid ≠ 0; version short numeric-semver) with redacted rejection reasons; 10 `#[cfg(test)]` cases written; **compile/run BLOCKED as D-4** |
| D-6 | Low (candidate) → Medium (part15) | The WS handshake (`routes/events.py`) enforced the session token (close 4401) but not the Origin header — the trusted-origin policy guarded REST only; with a token leaked through a future XSS or a hostile browser extension, an evil page could have subscribed to the full event stream (job progress, dataset names, result payloads) cross-origin to `127.0.0.1` | **FIXED** | 339 | `src/pharmaml/api/routes/events.py` — the pre-accept handshake guard (a present Origin/Referer held to the exact REST-lane policy via `security.origin_is_trusted`, untrusted → close 4403 BEFORE accept, valid token included; absent Origin = the non-browser lane, unchanged); `security.py` exported `origin_is_trusted` (the REST predicate, zero policy change); regression `tests/api/test_ws_origin.py` (12 tests: 5 acceptance lanes — packaged http/https, loopback dev any-port, absent-Origin non-browser; 7 refusal lanes — evil origin WITH a valid token, no-token, lookalike subdomain/suffix, port-bearing trusted host, Referer dodge, `file://` scheme), FAIL-FIRST witnessed (6 red pre-fix, 12/12 post); unchanged lanes re-pinned (the full `tests/api` suite exit 0, ruff + mypy clean, vitest 119/119, tsc clean) |

## Register conventions

- **FIXED** requires: the source change + the focused regression
  tests executed green + the unchanged lanes re-pinned (no auth
  bypass, no Host-policy weakening, lookalikes still rejected).
- **FIXED (source; compile-blocked)** means the change is in the
  tree with its tests written but the toolchain to compile/run them
  does not exist in this sandbox — the row converts to plain FIXED
  only after the first real `cargo test` pass. Never marked passed
  before then (§16.14).
- **OPEN** rows carry their next action; nothing silently closes.
- New findings append as D-N+1 with the discovery round named.

— seeded round 327 (the 119th recorded run); supersedes the
D-1…D-5 table inside `PART14_PHASE1_AUDIT.md` (which stands as the
Phase-1 historical record per the §2.4 historical-records rule).
D-6 closed round 339 (the 131st recorded run — part15 §4.2's
headline recheck; the register's first part15-era row conversion).
