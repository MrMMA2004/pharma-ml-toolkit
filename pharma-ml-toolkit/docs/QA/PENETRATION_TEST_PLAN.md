# PENETRATION TEST PLAN — part17 §4.5 deliverable (Round 356, lane P1)

The project-specific test plan derived from the mandated official
references — the OWASP Web Security Testing Guide
(https://owasp.org/www-project-web-security-testing-guide/), the OWASP
API Security Top 10 (2023 edition), and the OWASP ZAP Automation
Framework / Docker documentation — applied to THIS product's four
surfaces (the threat model's A–D).

## 1. Methodology

The program has exactly two legs, and neither is ever reported as the
other:

1. **The focused automated security suite** —
   `tests/api/test_penetration_security.py`, executed in the
   repository's own test runner (this round: **67 tests, all green**).
   These are deterministic, repeatable, regression-pinning probes of
   the attack classes against the REAL `create_app()` stack.
2. **The OWASP ZAP workflow** —
   `.github/workflows/penetration-test.yml`, which imports the exact
   OpenAPI contract of a disposable, seeded, project-owned instance
   (`tools/zap_target.py`) and runs passive scanning plus a
   controlled active API scan against that instance only.

Standing rule (part17 §4.4, verbatim in spirit): **ZAP Baseline is
passive scanning, not a complete penetration test** — nothing in this
program's reporting ever claims otherwise.

## 2. The test-ID map (what exists, where, with what verdict source)

| §4.3 mandate item | OWASP mapping | Evidence (file :: class) | Verdict source |
| --- | --- | --- | --- |
| Authentication bypass | WSTG-ATHN/SESS | `test_penetration_security.py :: TestTokenValidationBattery` (9 tests) | executed this round |
| Invalid / expired / missing / malformed tokens | WSTG-SESS | same battery (shapes: wrong, empty, whitespace, prefix, right-length-wrong-bytes, wrong-X-token-beats-right-Bearer, stale cross-session replay, no-echo oracle) | executed this round |
| IDOR / BOLA | API1 | `:: TestObjectLevelAuthorization` (7 tests: the 404 lanes across datasets/artifacts/jobs/runs + download/delete) | executed this round |
| Broken Function Level Authorization | API5 | same class (`test_deletion_lanes_are_not_anonymous`) — the honest single-user shape | executed this round |
| CORS bypass | WSTG-CLNT-07 | `:: TestCorsBypassComplement` (5 tests) + the standing 34-leg origin matrix | executed this round |
| Origin and Referer bypass | WSTG-CLNT | `:: TestOriginBypassComplement` (6 tests: `Origin: null`, userinfo trick, ASCII-lookalike host, mixed-case permissive pin, Referer path/query, `data:` scheme) | executed this round |
| WebSocket hijacking | WSTG-CLNT | `tests/api/test_ws_origin.py` (12/12 — the standing r339 lane) | standing green |
| Session replay | WSTG-SESS-06 | `:: test_replayed_token_from_other_session_is_401` | executed this round |
| Path traversal | WSTG-ATHZ-01 | `:: TestVaultTraversalAndSymlink` (5 tests) + the standing vault lanes in `test_persistence*.py` | executed this round |
| ZIP Slip | WSTG-ATHZ | `:: TestRestoreBundleZipSlip` (6 tests — the only archive surface, the backup-restore bundle) | executed this round |
| Symlink attacks | WSTG-ATHZ | `:: test_symlink_planted_in_the_vault_cannot_escape` | executed this round |
| Malformed uploads | WSTG-BUSL | the standing 415/413/422 lanes (`test_datasets_ml_artifacts.py`) + `:: TestUploadAttackShapes` (exe/html extension refusals) | standing + this round |
| MIME spoofing | WSTG-BUSL | `:: test_mime_spoofed_as_image_still_parses_as_data_only` | executed this round |
| Oversized uploads | WSTG-BUSL | the standing 413 lane + `:: TestResourceExhaustionCaps` | standing + this round |
| Dangerous filenames | WSTG-BUSL | `:: test_control_characters_in_filename_are_sanitised`, `test_windows_reserved_name_is_prefixed`, `test_dots_only_filename_falls_back` | executed this round |
| Unsafe export names | WSTG-BUSL | `:: TestUnsafeExportNamesF35601` (3 tests — finding F-356-01's fix, pinned) | executed this round |
| Sensitive information in logs | WSTG-INFO | `:: TestLogHygiene` (2 tests: caplog token-absence + the `access_log=False` source pin) | executed this round |
| Unauthorized artifact / dataset access | API1 | the 404 lanes of `TestObjectLevelAuthorization` | executed this round |
| Resource exhaustion | WSTG-BUSL-04/05 | `:: TestResourceExhaustionCaps` (3 tests: artifact count cap, byte cap, dataset count cap — oldest-first evict) | executed this round |
| Injection | WSTG-INPV | `:: TestInjectionLanes` (3 tests: the SQL payload round-trip, JSON-only responses, the parameterized-execute source pin) | executed this round |
| XSS | WSTG-INPV | `:: test_responses_are_never_text_html` — the honest verdict for a JSON API (no HTML rendering lane exists) | executed this round |
| CSP violations | — | `:: TestDesktopSurfaceStaticPins` (3 tests over `src-tauri/tauri.conf.json`) | executed this round |
| Process-handshake validation | — | `:: TestHandshakeShape` (2 Python-side pins; the Rust-side 11 cases are compile-blocked on this host and ride the Windows release pipeline — honestly OPEN there) | executed this round |
| Tauri permission overreach | — | `:: test_capability_allowlist_is_core_only` + `test_no_extra_capability_files` (`core:default` ONLY) | executed this round |
| Supply chain (part17 §4.2D) | — | `tests/test_security_supply_chain.py` (8/8 standing) + `security-scan.yml` CI lanes + CodeQL | standing green |

## 3. The ZAP workflow's scope and standing boundaries

`.github/workflows/penetration-test.yml` — weekly (Monday 04:30 UTC)
plus manual dispatch; the disposable target is launched by
`tools/zap_target.py` (loopback-bound, seeded with the labelled
synthetic dataset + one table artifact, contract dumped live), ZAP
runs `-cmd` via the Automation Framework config
(`packaging/security/zap-automation.yaml`), the session token rides
ZAP's documented Replacer rule, and JSON/HTML/Markdown reports are
archived as workflow artifacts together with the target's seed
provenance. The severity gate (`tools/zap_gate.py`) enforces §5's
policy and consults `packaging/security/zap-suppressions.json`.

Boundaries that are part of the plan, not limitations of it:

* active scanning touches ONLY the isolated ephemeral instance;
* no ZAP API port is ever exposed;
* no application control (auth, Origin validation, CSP, Host
  validation, loopback binding) is ever disabled to make a scan pass;
* the frontend/web-console surface is not ZAP-scanned here (the
  console is not servable on the runner) — its lanes are the
  build gate plus the local browser E2E walks, recorded honestly in
  the report;
* pull-request execution (the mandate's optional lane) is not
  enabled: the repository works by direct pushes with no PR lane, and
  the CI-minutes discipline is a recorded, deliberate choice.

## 4. Test data

All fixtures are deterministic synthetic data (the three-molecule
CSV, the in-memory crafted ZIPs, the fabricated records); nothing
derived from real user data, nothing destructive to real user data
(part17 §4.3's own rule).

## 5. Severity policy and failure thresholds

| Severity | Meaning here | Action |
| --- | --- | --- |
| Critical | authenticated-code-execution or data-destruction class reachable in the shipped topology | workflow FAIL, immediate remediation |
| High | auth bypass / data exfiltration in the shipped topology | workflow FAIL (`zap_gate.py` exit 1) |
| Medium | bounded or attacker-local-only impact | recorded in the register, remediated on the next lane |
| Low / Info | hygiene, defense-in-depth gaps | recorded in the register with justification |

## 6. False-positive suppression policy

Suppressions live ONLY in `packaging/security/zap-suppressions.json`
(the file ships empty). Every entry must carry `pluginid`, `regex`,
`justification` (why it is a false positive — never blanket), and
`expires` (an ISO date). **An expired entry is silently ignored** —
the finding returns, the gate fails again, and the suppression must
be re-justified or the finding remediated. `tools/zap_gate.py`
implements this exactly and is itself pinned by tests
(`TestZapTargetTooling::test_gate_*`).

## 7. Honesty rules

* A workflow that has not run stays **NOT_EXECUTED** in every
  document (the ZAP workflow's first firing validates its
  Automation-Framework config; any schema mismatch is itself a
  register finding).
* No test result is ever claimed for a lane that did not execute.
* Windows-only lanes stay with their BLOCKED_ENVIRONMENT entries.
