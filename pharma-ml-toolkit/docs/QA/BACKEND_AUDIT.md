# BACKEND AUDIT — part15 §13 deliverable #2

The part15 Phase-Two module-by-module backend inspection record, over the
audited commit `0132d409` (round r340). Findings carry their evidence;
nothing is marked verified by inspection alone. The audit re-reads the
Phase-One verification matrix (COMPREHENSIVE_REPOSITORY_AUDIT.md §2) as its
baseline.

## 4.1 FastAPI architecture

| Area | State | Evidence |
| --- | --- | --- |
| Application factory + lifecycle | `src/pharmaml/api/__init__.py` `create_app()` — factory pattern; startup/shutdown managed | full `tests/api` suite exit 0 (r339); the server boots in every gate round |
| Router registration | 27 route modules under `routes/` (health, datasets, chemistry, ml, dl, semi_supervised, generative, docking, pkpd, md, admet, dta, drugresponse, pharmacogenomics, repurposing, multiobjective, applicability_domain, model_evaluation, visualization, workflow, jobs, events, database, capabilities, system, artifacts + `__init__`) | contract guards + drift tests in the suite |
| Dependency injection + middleware ordering | session-token dependency chain; scoped `CORSMiddleware` (the r327 D-2 fix: anchored regex, exact-origin echo, `allow_credentials=False`, token headers allowlisted) | `test_trusted_origins.py` 34 legs green |
| Pydantic validation + response models | typed request/response contracts per route; the one-DTO rule (§12.2) guarded | drift guards (`test_workflow_run_request_fields_mirror_the_backend`) |
| Exception handling | typed errors, safe messages (never stack traces/paths/secrets — §4.1 rule) | API error lanes in the suite |
| OpenAPI schema generation | exported contract **109 paths / 71 schemas** (r324 pin, CI-guarded byte-identical re-export) | contract check green in CI (19/19 at HEAD) |
| Background tasks / startup-shutdown | JobManager cooperative cancellation + progress reporting | job-lifecycle lanes in the suite |

**Finding**: no incorrect status codes, unhandled-exception leaks, or
duplicate implementations found in the current inspection; the known
defect history (D-1..D-7) is closed in the DEFECT_REGISTER.

## 4.2 Backend security (post-D-6)

| Control | State | Evidence |
| --- | --- | --- |
| Loopback-only binding | server binds loopback only | `server.py` + boot lanes in every suite run |
| Host validation | loopback-host policy | security suite lanes |
| Origin/Referer validation (REST) | scheme-strict, host-exact `TRUSTED_APP_ORIGINS` policy; lookalikes/subdomains/ports rejected | `test_trusted_origins.py` (34 legs, the r335 pin) |
| Origin validation (WS — D-6, r339) | pre-accept handshake guard: present Origin/Referer held to the same predicate, untrusted → close **4403 BEFORE accept** (valid token included — defense-in-depth); absent Origin = the non-browser lane (Rust host, curl, tests) | `tests/api/test_ws_origin.py` 12/12, FAIL-FIRST (6 red witnessed pre-fix) |
| Session token generation/verification | token handshake, close 4401 unauthenticated (WS), 401 (REST) | security lanes |
| CORS | scoped to trusted origins only; arbitrary origins receive no permissive response | P-lane evidence (r327/r329/r339) |
| File upload/download validation | size/type validation on upload lanes | api suite |
| Path traversal protection | `security.py` path policy + `resolveToolkitPath`-class guards (traversal, null-byte, absolute-path rejection) | FS lanes in tests/api |
| Temp-file handling | isolated-temp-DB discipline for tests (never user data — §4.5 rule) | suite fixtures |
| Secrets in logs | D-4 redaction (source) + strict pattern scan zero-hit | supply-chain suite (r324) |

The Tauri/WebView2 incident history (tauri.localhost rejected / 405
preflight / misclassified startup errors) is closed at D-1/D-2/D-3 with
their regression suites; the packaged-origin lanes are re-verified each
security round.

## 4.3 Rust handshake and process supervision (D-4/D-5 — the blocked lane)

`src-tauri/src/backend_manager.rs` carries the structured handshake
parsing, `validate_handshake()` (port bounds, pid sanity, token
16..=128-byte urlsafe shape, version semver shape) and
`redacted_parse_failure()` (raw line + token never formatted into error
surfaces), with 11 `#[cfg(test)]` cases. **Rust toolchain is absent in
this sandbox** — `cargo fmt --check` / `cargo clippy` / `cargo test` are
honestly BLOCKED (never marked run). The carrier is
`scripts/windows/test-rust.ps1` (SS11), whose green run on the owner's
Windows box converts D-4/D-5 to plain FIXED per the register convention.

## 4.4 WebSocket reliability

The events route (`routes/events.py` + `events.py`): connection
establishment, token auth (4401), origin guard (4403, r339), heartbeats
and the replay/sequence discipline; the REST surface remains the source
of truth for job state (§4.4 rule — the WS stream is a progress channel,
never the only answer; the frontend JobRunCard relay reads the REST
result). The interactive desktop reconnect legs (backend restart during
active work, abrupt termination) ride the Windows box — honestly BLOCKED
here; the API-level lanes are pinned by `test_ws_origin.py`.

## 4.5 SQLite and FileVault

`api/persistence.py`: sqlite3 connections with row factory, WAL-mode
discipline, the schema/migrations for job history, dataset and model
metadata, artifact registration; `FileVault` for artifact bytes with
integrity validation. Concurrency: the JobManager persists state
transitions through the store; concurrent read/write lanes are covered by
the persistence suites under isolated temp directories (§4.5 rule:
user data never touched). Backup/restore and corrupted-metadata lanes
are exercised in the persistence suites; disk-full handling is a
documented BLOCKED environmental lane (see RELEASE_BLOCKERS).

## 4.6 JobManager and resource management

`api/jobs.py` `JobManager`: typed `JobRecord` + `JobState` machine,
cooperative cancellation (`CancelledJob` + cancel-event injection between
workflow nodes, r322), progress reporting through the WS channel, queue
limits, and SQLite persistence of job history (restart-recovery lanes in
the suite). Long-running scientific work executes off the API event loop
(thread pool + job mode). Pause/resume is NOT claimed for tasks that
cannot safely persist execution state (§4.6 rule honored). The read-only
`/api/v1/system/resources` snapshot (r324) exposes CPU/memory/queue depth
without changing manager semantics.

## 4.7 Scientific backend correctness

The scientific modules behind the 27 route families cover the §4.7 list
(dataset handling, standardization, RDKit descriptors, fingerprints,
feature selection, splitting, classical ML, DL, semi-supervised,
generative, docking, PK/PD, MD, visualization, ADMET, DTA, repurposing,
drug response, pharmacogenomics, multi-objective optimization, the
workflow engine). The correctness properties (input validation, output
shapes, NaN/Inf handling, seed handling, leakage discipline — split
before selection in the guided template, cross-validation, serialization,
provenance, disclaimers) are pinned by the scientific suites in the
canonical collection (8,601 / 0 errors / 472 files live; 8,555 at the
shipped v1.2.240 freeze). No fabricated datasets/checkpoints/results —
the synthetic stand-ins are labeled as such in the route scope blocks.

— audited round 340 (the 132nd recorded run) at `0132d409`.
