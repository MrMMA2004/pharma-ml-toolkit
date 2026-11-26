# MIGRATION_TAURI2 — historical part11 campaign ledger

**Status:** This is a versioned planning/evidence ledger, not the current
install guide or release-status page. Its tables describe their recorded
baselines and may predate the shipped desktop app. The supported desktop is
React + TypeScript hosted by Tauri 2; the separate Next.js console is a
separate application, and the older PySide6 UI is retired. See
[`frontend/README.md`](../frontend/README.md) for current desktop scope.

**Master prompt part11**: "Complete Migration to an Offline React +
TypeScript Desktop Application Using Tauri 2, Python + FastAPI +
WebSocket, SQLite, Complete Backend–GUI Integration, Nine-Workspace
Architecture, Legacy PySide6 Retirement, and Standalone Windows
EXE/MSI Deployment."

This ledger supersedes `MIGRATION_WINUI3.md` as the campaign plan of
record: part11 §2 ("Mandatory Architecture Replacement Rules")
explicitly removes every requirement that existed solely for the
former C#/WinUI 3 frontend — C# as the frontend language, WinUI 3,
XAML, MVVM, CommunityToolkit.Mvvm, .NET ViewModels/Commands, NSwag
C# client generation, ScottPlot/LiveCharts, Inno Setup, .NET
self-contained publishing. The queued "C# solution specification"
row of the part10 ledger is therefore **cancelled**.

Nothing scientific is discarded: part11 §2 preserves every backend,
workflow, visualization, security, testing and standalone-deployment
requirement, and part11 Phase Seven mandates **exactly the FastAPI +
REST + WebSocket backend layer the part10 campaign already built and
tested**. The migration continues on top of it with a different —
and in one decisive way more honest — frontend host.

---

## 0. The environment boundary (re-stated for part11)

The interactive development sandbox is **Linux-only**. A GitHub Actions
Windows runner has built and verified the v1.2.230 EXE/MSI release
artifacts. They are attached to a **draft** GitHub release, not the
latest public release. A runner build is not a substitute for the
separate clean-machine installation and upgrade ritual.

| part11 deliverable | Current status |
|---|---|
| React + TypeScript + Vite frontend (Phase Three) | **BUILT** — eight scientific workflows connect to the real API, alongside the Introduction start page; PR CI type-checks/builds the offline SPA and runs contract/workspace drift guards. |
| FastAPI backend integration layer (Phase Seven) | **BUILT** — the versioned API surface is exercised by the API/database test suite. |
| SQLite persistence + file vault (Phase Eleven) | **BUILT** — `src/pharmaml/api/persistence.py` + `/api/v1/database`; see §2. |
| Backend process lifecycle + handshake (Phase Nine) | **BUILT** — port-0 loopback bind, `PHARMAML-READY` handshake, graceful shutdown, and the packaged sidecar's live Windows-runner probe. |
| Background job architecture (Phase Thirteen) | **BUILT** — JobManager state machine, cooperative cancellation, `n_jobs` passthrough including `-1`, never `0`. |
| Tauri 2 host and Windows packaging (Phase Four / part11_2) | **PARTIAL** — the v1.2.230 Windows runner generated the EXE/MSI, manifest and checksums and attached them to a draft release; clean-machine install/upgrade and runtime verification remain pending. |
| PySide6 retirement (Phase Sixteen) | **NOT DONE** — PySide6 remains the production Studio until the remaining migration/release gates pass. |

The remaining BLOCKED items are named explicitly; completed builds and
workflows below carry their own evidence and are not described as
queued.

---

## 1. What part11 adopts from the part10 campaign (already REAL)

The complete backend side of part11 Phases Seven / Eight / Nine /
Thirteen / Fifteen is the existing `src/pharmaml/api/` package —
one codebase, **251 api tests green** this round:

| part11 requirement | Delivered as | Test evidence |
|---|---|---|
| §12.1 REST route groups (versioned `/api/v1`) | health, capabilities, chemistry, datasets (+profile/split), ml, dl, semi-supervised, generative, docking, pkpd, md, visualization, applicability-domain, jobs, artifacts, models, **database** (this round) | `tests/api/` — 6 files, 251 tests |
| §12.2 Pydantic schemas → OpenAPI → *TypeScript* client | `/api/openapi.json` served (docs disabled per §15) + **`contracts/openapi.json` exported** (`tools/export_openapi.py`, drift-gated by `tests/api/test_contracts.py`); `openapi-typescript` generates the committed frontend types, and service request bodies use those types (one DTO definition, never two) | contract exported + committed + drift-gated; generated types and clients integrated |
| §13 WebSocket events | typed `EventEnvelope` (single wire DTO), 20-event vocabulary, per-bus monotonic sequence, replay-since dedup, bounded-queue backpressure + `bus.overflow`, heartbeats, REST stays authoritative | TestEventBus, TestWebSocketEvents, TestDLTrain live-stream |
| §14.1 startup handshake | port-0 socket owned by the server, fd handed to uvicorn (no check-then-bind race); single `PHARMAML-READY {port, token, pid, version}` stdout line; machine-readable line separate from logs | TestStartupHandshake (real serve probe) |
| §15 local API security | per-startup `secrets.token_urlsafe(32)`, memory-only, constant-time verify, loopback-only bind, Host/Origin guard on every route, upload size/extension guards, docs disabled | TestSessionSecurity, TestLoopbackGuard |
| §13/n_jobs | manual `n_jobs` verbatim incl. `-1`; `0` rejected by the backend's own validation; expensive-slot semaphore caps concurrent heavy jobs | TestMLTrain, TestJobManager |
| §16.1–16.4 SQLite | **this round** — see §2 below | `tests/api/test_persistence_database.py` (37) + `tests/api/test_persistence.py` (30) |

The part10 Phase-One audit (requirement matrix, 62-row capability
inventory, packaging-readiness audit) carries over verbatim — part11
Phase One inspects the same project; its results are recorded in
`MIGRATION_WINUI3.md` §1 and `docs/CAPABILITY_MAP.md`.

---

## 2. Phase Eleven — SQLite persistence (built this round)

The part10 stores were session-scoped by design (the WinUI host
re-uploaded everything after a restart). part11 §16 replaces that
contract; implemented in `src/pharmaml/api/persistence.py` (959
lines) + `src/pharmaml/api/routes/database.py`:

* **Ownership** (§16 intro): the Python backend owns the database;
  the frontend never touches the file — there is no JS or Rust
  database implementation, SQLite access exists only behind the API.
* **Large data** (§16.2): the `FileVault` holds dataset source
  bytes, pickled models, priors and artifact payloads under
  `<data_dir>/files/<category>/`; SQLite stores ids, relative paths,
  sha256, metadata — never the payloads themselves.
* **Location** (§16.5): `default_data_dir()` →
  `%LOCALAPPDATA%\PharmaML Studio` on Windows, XDG data home
  elsewhere, `PHARMAML_STUDIO_DATA` override for tests/hosts; never
  inside the installation directory; user data survives upgrades.
* **Connections** (§16.4): per-operation connections with
  `foreign_keys=ON`, `journal_mode=WAL`, `busy_timeout=5000` — no
  connection shared across the event loop and job threads.
* **Migrations** (§16.3–16.4): the `schema_migrations` ledger
  (v1 base registry, v2 indexes + generative priors), idempotent,
  transactional (`BEGIN IMMEDIATE`), re-applied after restore.
* **Activation contract**: `create_app()` stays **ephemeral** (the
  documented library/test default — nothing on disk, status reports
  `persistent: false`); `create_app(data_dir=...)` or
  `PHARMAML_STUDIO_DATA` turns durability on; the shipped host entry
  (`server.serve()`) always passes the platform user-data directory.
* **Job history** (§13/§16.1): every JobManager state transition
  upserts its row (failure-tolerant — a failed history write never
  fails the live job); `prune_jobs` enforces the retention rule.
* **Backup/restore** (§16.4): the online-backup snapshot +
  **referenced vault payloads** as one zip bundle, integrity- and
  schema-version-validated before swap, atomic replace with WAL
  sidecar cleanup, orphan sweep reports honest degradation; corrupt
  or future-schema images are refused (ValueError → 422).
* **Recovery** (§16.1 "preserve user data across restarts"):
  `recover_registry()` rebuilds the live stores on startup — dataset
  frames re-parsed through the REAL loader, models unpickled; rows
  whose payloads fail are dropped from DB + registry and *reported*
  in the status route's `startup_recovery.dropped`, never silently
  skipped.

Two honest bugs the new suites caught and fixed on arrival:
`FileVault.exists()` reported resolvable-but-absent paths as present
(which would have silently defeated the orphan sweep); garbage
restore images raise `sqlite3.DatabaseError` at execute time — now
normalised to the documented `ValueError` → 422.

The contract export (§2's second half) surfaced its own bug class:
the four colliding request-model names across route modules
(`AnalyzeRequest` ×2, `TableRequest` ×2) made FastAPI's collision
fallback name components nondeterministically across processes
(hash-seed-dependent set order — the r275 selfies flake class);
all four are renamed uniquely and the export is byte-stable across
nine hash seeds.

### The §12.1 database route group

| Route | Contract |
|---|---|
| `GET /api/v1/database` | status: persistent flag, data dir, schema version, WAL, integrity, per-table counts, startup-recovery report |
| `GET /api/v1/database/jobs` | persisted job history (workspace filter, limit) |
| `GET /api/v1/database/preferences` | list `{key, value, updated_at}` |
| `GET/PUT/DELETE /api/v1/database/preferences/{key}` | durable preference CRUD (strict-JSON values) |
| `POST /api/v1/database/backup` | full bundle registered as a `database` artifact (downloaded via the standard artifact contract) |
| `POST /api/v1/database/restore` | restore by `artifact_id`; `database`-kind enforced (422 otherwise); rebuilds the live registry from the restored database |

All behind the same session token + loopback guard as every route
(401 without token; foreign Host rejected).

---

## 3. The part11 campaign ledger (multi-round plan)

| Round | Scope | Status |
|---|---|---|
| 278 | Phase Eleven SQLite + file vault + `/api/v1/database` + durability/recovery (67 persistence/database tests across the two lanes, api suite 251 green) + §12.2 contract tooling (`contracts/openapi.json` + `tools/export_openapi.py` + the drift gate + the four colliding request-model renames for byte-stable exports) + this ledger (architecture replacement recorded, C# spec row cancelled) | done (v1.2.222) |
| 279 | Phase Three foundation: the `frontend/` Vite + React + TS workspace (strict tsconfig, Tailwind v4 + the Mineral tokens, shadcn/ui primitives on the Radix heads, Motion for React) with the horizontal nine-workspace navigation shell (the exact §10.1 order; Introduction the §10.2 start page; the eight queued workspaces render honest shells) + the §12.2 client half (openapi-typescript generation wired to the committed contract — the 5,476-line `api.d.ts`, every schema covered) + the typed REST client (handshake-only endpoint/token) + the WS envelope mirror + 6 Python-side drift guards (`tests/test_frontend_foundation.py`); `tsc --noEmit` clean, the production build green (2,085 modules) | done (v1.2.223) |
| queued | §12.2 client half: openapi-typescript type generation wired to the committed `contracts/openapi.json` (the export + drift gate are done — see row 278) | done in 279 |
| 280 | Phase Three services, part two: the WS client service (`websocket/client.ts` — the real hello frame, client-side duplicate-event protection, the `since` replay cursor, 40 s heartbeat liveness, bounded backoff reconnection, shutdown-never-reconnect) + the job manager (`jobs/types+client+tracker` — JobView field-for-field + runtime validation, the tracker keeps REST authoritative with 1.5 s polls, reconnection + `bus.overflow` reconciliation) + the dataset client (multipart upload with the backend's exact Form fields, list/view/preview/delete/profile/split, requests from the GENERATED types) + the artifact client (list/view/delete + the token-header `raw` download lane consuming `X-Artifact-SHA256`/ETag) + the `useSyncExternalStore` hooks + 14 drift guards (`tests/test_frontend_services.py`, incl. the vocabulary guard the 279 counter claimed but never shipped); the round-279 fictional `{kind:"hello"}` wire shape corrected to the backend's real heartbeat-payload hello; **a REAL E2E rode the live backend** (`scratch/e2e280/`: handshake→WS→multipart upload→tracked job to COMPLETED on live events→result→registry/preview/profile/split→artifact download with digest→SIGTERM→the shutdown envelope reached the Node client) and it CAUGHT A REAL BACKEND BUG: stock uvicorn closes connections before the lifespan handlers fire so the `shutdown` publish never reached a socket — `StudioServer.shutdown()` now pushes the envelope + one sender beat BEFORE uvicorn's connection close, regression-tested on a real socket (`TestGracefulShutdownPush`); the canonical pins' surfaced lint debt swept (mypy src/ clean after 13 fixes, ruff all-green after the 364 sweep); tsc + production build green | done (v1.2.224) |
| queued | Phase Three services, part two: the WS client service (hello/replay/heartbeats/reconnection) + the job manager client + dataset/artifact clients on the generated types (§5.2) — the typed REST client + envelope mirror shipped in 279 | done in 280 |
| queued | Datasets workspace workflow on the real API (§11.1), then ML (§11.2), then the remaining seven workspaces row by row | done in 281 (Datasets) |
| 281 | The Datasets workspace workflow (§11.1) + the part11_2 Windows release pipeline addendum: the `features/datasets/DatasetsPage` — the complete sequential chain on the REAL backend (multipart import with live job tracking → the registry/paginated-preview Inspect → the profile-frame Validate evidence → the loader-policy Clean/Select display → the Chemistry Lab engines through the NEW bounded column bridge → the stratified split report → the statplots Visualize → the digest-verified Export); the chemistry client (`services/chemistry/client.ts`, requests from the generated types) + the column bridge (`GET /datasets/{id}/column/{column}` — bounded by the chemistry engine's own 500 cap, `truncated` explicit) + the split route's sklearn boundary fixed to a clean 422 (the live E2E caught the raw 500); **E2E-DATASETS-PASS** (`scratch/e2e281/`: handshake→WS→upload→column bridge→descriptors(job)→fingerprints(sync)→profile→split→the digest-verified export→shutdown envelope); part11_2: `.github/workflows/windows-release.yml` (23 stages, pinned actions, least-privilege, the honest BLOCKED closing note) + `src-tauri/` (the conf with offline WebView2 + NSIS/MSI + the loopback CSP; the least-privilege capability; the Rust host — backend_manager implements the TESTED PHARMAML-READY handshake contract with the startup timeout + graceful window, security.rs's loopback assertion, the single backend_endpoint IPC command) + `packaging/windows/pharmaml_backend.spec` (the sidecar entry, the Qt stack excluded) + `backend_entry.py`/`verify_sidecar.py` + `TAURI_RELEASE_PIPELINE.md` (the §12 honest labels) + 8 pipeline drift guards (`tests/test_windows_pipeline.py`); the lost r280 artifacts client restored (a rollback had eaten it — the guard was red) | done (v1.2.224) |
| 282 | §11.2 Machine Learning workflow on the real API: dataset/model selection, training jobs, evaluation, comparison, applicability-domain analysis, semi-supervised learning and export; the live backend driver completed **E2E-ML-PASS**. | done |
| 283–287 | §11.3–§11.8: Deep Learning, Generative AI, Docking, PK/PD, Molecular Dynamics Analysis and Visualization workflows completed against their real backend services; the nine-workspace reconstruction completed. | done; round-specific E2E and browser evidence is recorded in the repository work log |
| 288–291 | 3D visualization, startup/integration audit, SHAP explainability and Plotly adoption across the workflows; live browser verification includes ML parity/sample inspection, applicability-domain and SHAP charts. | done; all nine workspaces usable, plotting integration closed |
| 298–299 | v1.2.230 Windows release ritual: fresh notebooks and source archive, Windows sidecar verification, Tauri/NSIS EXE + WiX MSI build, validation, manifest/checksums and GitHub Release upload. | build and uploads succeeded; assets are attached to a draft release, not yet public |
| Windows clean-machine gate | Install, offline launch, WebView2/runtime behavior, upgrade/data retention and uninstall on a genuinely clean Windows 10/11 machine; only after this gate may PySide6 retirement be considered. | **BLOCKED / not yet verified**; the hosted release runner is build evidence, not the clean-machine ritual |

The React workflows and Windows release artifacts are implemented and
have execution evidence. The remaining desktop-migration boundary is
the clean-machine gate and the resulting retirement decision; do not
label either as complete until that evidence exists.

---

## part12 addendum — the Phase-1 audit campaign (2026-11-20, round 302)

The part12 MASTER_PROMPT ("Comprehensive Repository Audit, Architecture
Hardening, Scientific Feature Expansion, GUI–Backend Verification,
Performance Optimization, and Production Release Validation") opened
against baseline v1.2.231 / `7bbc41ed`. Its Phase-1 deliverables live in
`docs/part12_audit/` (nine documents: compliance matrix, integration
matrix, architecture review, bug report, scientific/performance/security
reports, feature gap + GitHub references, clean-install report). Ledger
changes this round:

1. **Database persistence: GUI status flipped from "no consumer" to
   BUILT+WIRED.** The `/api/v1/database` group (status/job history/
   preferences/backup/restore — always backend-BUILT with 67 tests) gained
   its first product consumer: the typed `services/database/client.ts` +
   the Introduction "studio database" card (B-2). The §10.2 Introduction
   claim "backup and restore built in" is now substantiated.
2. **Source-archive integrity defect closed (B-1):** the any-depth
   `artifacts` exclusion dropped `frontend/src/services/artifacts/
   client.ts` from the shipped v1.2.231 zip AND hid it from the explorer +
   census. Scoped relative-path exclusions fixed walk + streamer + builder;
   this round's archive re-ships complete (composition audit re-run at
   close-out).
3. **Latent `frontend/dist` leak closed (B-1b)** — build output can no
   longer ride the census/archive after local builds.
4. **Hygiene (B-3):** the malformed, unregistered nested
   `pharma-ml-toolkit/.github/workflows/ci.yml` (`branches: ain]`) is
   removed; root registration remains the single CI truth.
5. **The honest lanes are unchanged**: clean-machine smoke test
   (C-3/C-5) and the §28 Stage-5 PySide6 retirement stay BLOCKED behind
   it; the new part12 audit documents them per-row
   (WINDOWS_CLEAN_INSTALL_REPORT.md).
