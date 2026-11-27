# PENETRATION TEST REPORT — part17 §4.5 deliverable (Round 356, lane P1)

Seeded: Round 356 (2026-11-25), over the pushed r356 terminal
`dce99b1b`. Every verdict below names its execution evidence;
nothing is claimed for a lane that did not run.

## 1. Executed this round (the automated suite)

`python -m pytest tests/api/test_penetration_security.py` —
**67/67 passed, 0 failed** (the collection re-pin:
**8,870/0/488** = the r356 pin 8,803/0/487 + 67 + 1 file).

| Family | Tests | Verdict |
| --- | --- | --- |
| Token validation battery (bypass, malformed, header precedence, cross-session replay) | 9 | PASS |
| Origin/Referer bypass complement (`null`, userinfo, lookalike, mixed-case, Referer path, `data:`) | 6 | PASS |
| CORS bypass complement (port-bearing trusted host, suffix host, non-http scheme, IPv6 dev lane, credentials-false pin) | 5 | PASS |
| IDOR/BOLA + BFLA (the 404 lanes + the anonymous-mutation lane) | 7 | PASS |
| Vault traversal + symlink + non-string rel | 5 | PASS |
| ZIP Slip over the restore bundle (traversal, absolute, non-prefixed, prefixed-escaping, non-zip, missing-db) | 6 | PASS |
| Upload attack shapes (MIME spoof, exe/html 415, control-char sanitisation, reserved name, dots-only) | 6 | PASS |
| Unsafe export names (F-356-01: CRLF sanitised at registration, header stays one line, quote round-trip) | 3 | PASS |
| Resource exhaustion caps (artifact count/bytes, dataset count — oldest-first evict) | 3 | PASS |
| Injection + XSS (SQL payload round-trip, JSON-only responses, parameterized-execute pin) | 3 | PASS |
| Log hygiene (caplog token-absence, `access_log=False` source pin) | 2 | PASS |
| Desktop surface static pins (CSP directives, loopback-only connect-src, no wildcards, `core:default` only, no extra capability files) | 5 | PASS |
| Handshake shape (the one machine-readable line; the port-0 atomic loopback bind) | 2 | PASS |
| ZAP tooling (the target app/token lane, seed+contract dump, the gate's fail/pass/expiry verdicts) | 5 | PASS |

Standing suites re-verified as the complement set this round (their
lanes are part of this program's evidence map):
`test_trusted_origins.py` (the 34-leg origin/CORS/auth matrix),
`test_ws_origin.py` (12/12), `test_security_supply_chain.py` (8/8),
`test_datasets_ml_artifacts.py` + `test_persistence*.py` (the
upload/validation/restore lanes) — all green over the repaired venv.

## 2. Findings (this round)

### F-356-01 — Content-Disposition injection via artifact filenames

* **Severity**: Low (exploit path constrained: loopback-only bind,
  token-gated, single-user local product) — but a REAL
  response-header surface in every artifact kind.
* **What**: the artifact registration kept the raw
  `Path(base).stem` of the user-supplied `filename`; CR/LF, quotes
  and control characters flowed into the stored name and then into
  `Content-Disposition: attachment; filename="..."` at download —
  a header-injection shape (the second surface of the same class
  the dataset-upload lane already sanitised).
* **Reproduction**: `POST /api/v1/artifacts/figure` with
  `filename: 'evil\r\nSet-Cookie: session=x\r\n\r\n<body>'`
  (pre-fix: the stored name carried the CRLF; the download header
  would carry the smuggled second line).
* **Impact**: response-splitting-class header smuggling; under the
  gateway in front of the console a malformed header can break or
  mislead intermediaries.
* **Remediation (fixed this round)**:
  `routes/artifacts.py::_safe_stem` passes the stem through
  `safe_filename` (the datasets route's own lane) at all three
  registration sites, and `_download_headers` additionally strips
  `"` / CR / LF / NUL from any stored name before building the
  header (defense-in-depth for legacy records).
* **Retest**: `TestUnsafeExportNamesF35601` — 3/3 green (the
  sanitised stem is a portable `[A-Za-z0-9._-]+` token; the
  fabricated hostile record's header stays exactly one line with
  exactly two quotes).

### F-356-02 — the root CI push-trigger filter is cosmetically malformed

* **Severity**: Informational (hygiene).
* **What**: `.github/workflows/ci.yml` declares
  `push: branches: ain]` (and the same for `pull_request`) — the
  `[` of an intended `[main]` was lost at some historical edit.
* **Observed behavior (verified this round, honestly recorded)**:
  pushes to `main` STILL fire CI — the Actions API shows runs
  371–376 as `push`-event, `success`, at exactly the recent
  terminal SHAs (376 at `dce99b1b`), and the terminal's check suite
  is **19/19 ALL SUCCESS**. The defect is cosmetic-but-unexplained
  filter parsing, not a functional outage.
* **Remediation**: proposed (`branches: [main]`) and left OPEN in
  the register — changing trigger breadth is an Actions-minutes
  (billing-adjacent) surface on the owner's account, exactly the
  class the program records instead of touching unilaterally.

## 3. The ZAP workflow — NOT_EXECUTED

`.github/workflows/penetration-test.yml` is implemented, registered
at the repository root (the r293 registration lesson), scheduled
(Monday 04:30 UTC) + dispatch, with the disposable seeded target
(`tools/zap_target.py`), the Automation-Framework config, the
Replacer-rule token injection, the severity gate and the documented
suppression channel — all pinned by the tooling tests above. Its
first runner firing is the validation event; until then its status
in this report is **NOT_EXECUTED** (the honesty rule, part17 §4.5).
The web-console ZAP lane is likewise recorded as not practical on
the runner (the plan documents why).

## 4. Repository-state evidence gathered this round

* CI check-runs at the pushed r356 terminal `dce99b1b`:
  **19/19 ALL SUCCESS** (authenticated Actions API poll).
* Release `v1.2.240` assets intact: EXE 558,022,042 B,
  MSI 710,553,078 B, plus RELEASE_MANIFEST.json, the SBOM and
  SHA256SUMS.txt (authenticated API poll).
* The v1.2.240 zip parity across the in-tree homes re-verified
  byte-identical (`0fadb6e5` / 23,489,390 B).

## 5. Remaining limitations (honest)

* The Rust-side handshake validation cases are compile-blocked on
  this host (no Windows toolchain here); they ride the Windows
  release pipeline as they have since the part14 lane.
* The ZAP workflow's first run is pending (§3).
* The browser-based web-console attack surface is covered by the
  build gate + the local browser E2E walks; a ZAP-driven console
  scan remains a future lane, gated on the console being servable
  in CI.
* Windows clean-machine installation / installer signing stay with
  their long-standing BLOCKED_ENVIRONMENT entries.

---

## 6. Round 368 evidence (part19 Phase 7 — 2026-11-27)

**Executed locally this round:** the 67-test API penetration-security suite
(`tests/api/test_penetration_security.py`) re-run green on the pinned stack
(authn/authz, token replay, session isolation, IDOR/BOLA, function-level
authorization, Origin/Host validation, CORS, WebSocket authorization, upload
validation, path/symlink traversal, ZIP Slip, SQL injection, XSS, response
headers, resource exhaustion, log leakage, SQLite/FileVault access — the
coverage matrix above in §1).

**The ZAP campaign status is unchanged and honest:** the scheduled weekly
Monday 04:30 UTC slot (`penetration-test.yml`, the standing 360-minute deep
scan per the r364 close-out) did not fire during this round — the round ran
Friday 2026-11-27; the next slot is Monday 2026-11-30 04:30 UTC. No scan
results exist to review for this round; per the honesty rule the campaign is
NOT labelled complete. The synthetic isolated target (`tools/zap_target.py`
+ the Automation-Framework config + the severity gate) remains pinned by the
ZAP tooling tests (§1) and ready for the slot's next firing. The Web Console
scan limitation (not practical in CI) stands as documented in the plan.

**Adjacent supply-chain lane (historical, last verified r365):**
security-scan run 45873026401 green (pip-audit + npm-audit + the
supply-chain guard) — reported by the round ledger, not re-verified this
round (no workflow-run reader available in this session).
