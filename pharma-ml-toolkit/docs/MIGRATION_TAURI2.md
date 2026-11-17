# MIGRATION_TAURI2 — the part11 campaign ledger

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

The development container is **Linux-only**. The boundary MOVES with
part11 — in part11's favour:

| part11 deliverable | Linux-container status |
|---|---|
| React + TypeScript + Vite frontend (Phase Three) | **REAL (feasible)** — a Vite/React/TS build is a plain Node toolchain; unlike WinUI 3 it needs no Windows host. The part10 frontend BLOCKED line is lifted. |
| FastAPI backend integration layer (Phase Seven) | **REAL (built)** — part10 rows 1–4 delivered the complete API surface; part11 adopts it unchanged. |
| SQLite persistence + file vault (Phase Eleven) | **REAL (built this round)** — `src/pharmaml/api/persistence.py` + `/api/v1/database`; see §2. |
| Backend process lifecycle + handshake (Phase Nine) | **REAL (backend side)** — port-0 loopback bind, `PHARMAML-READY` handshake, graceful shutdown; the *Tauri-side* supervisor moves to the host. |
| Background job architecture (Phase Thirteen) | **REAL (built)** — JobManager state machine, cooperative cancellation, `n_jobs` passthrough incl. `-1`, never `0`. |
| Tauri 2 Rust host (Phase Four) | **PARTIAL (feasible)** — Tauri compiles on Linux (webkit2gtk) for host-side logic and contract verification; the *Windows* packaging legs stay host-bound. |
| Windows EXE/MSI bundling, WebView2 runtime verification, clean-machine §14-equivalent | **BLOCKED** — unchanged: no Windows host (same boundary as part10 and part9's `packaging/windows/BLOCKED` reports). |
| PySide6 retirement (Phase Sixteen) | **BLOCKED** — rule unchanged: parity first; PySide6 stays production. |

No BLOCKED line is claimed as done. The feasible-but-unbuilt lines
are queued rows, not claims.

---

## 1. What part11 adopts from the part10 campaign (already REAL)

The complete backend side of part11 Phases Seven / Eight / Nine /
Thirteen / Fifteen is the existing `src/pharmaml/api/` package —
one codebase, **251 api tests green** this round:

| part11 requirement | Delivered as | Test evidence |
|---|---|---|
| §12.1 REST route groups (versioned `/api/v1`) | health, capabilities, chemistry, datasets (+profile/split), ml, dl, semi-supervised, generative, docking, pkpd, md, visualization, applicability-domain, jobs, artifacts, models, **database** (this round) | `tests/api/` — 6 files, 251 tests |
| §12.2 Pydantic schemas → OpenAPI → *TypeScript* client | `/api/openapi.json` served (docs disabled per §15) + **`contracts/openapi.json` exported this round** (`tools/export_openapi.py`, drift-gated by `tests/api/test_contracts.py`); part11 replaces NSwag with openapi-typescript/Orval — the generated TS client is a queued frontend row, generated FROM the committed contract (one DTO definition, never two) | contract exported + committed + drift-gated; no hand-written duplicate |
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
| queued | Phase Three foundation: the `frontend/` Vite + React + TS workspace (strict tsconfig, Tailwind + shadcn/ui tokens, feature-oriented skeleton) with the horizontal nine-workspace navigation shell (§10.1 order: Introduction → Datasets → ML → DL → Generative AI → Docking → PK/PD → MD Analysis → Visualization; Introduction-only start page) | queued |
| queued | §12.2 client half: openapi-typescript type generation wired to the committed `contracts/openapi.json` (the export + drift gate are done — see row 278) | queued |
| queued | Phase Three services: typed REST/WS clients, job manager, dataset/artifact clients (§5.2 frontend application services); Introduction page content (§10.2) | queued |
| queued | Datasets + ML workspace workflows on the real API (§11.1–11.2), then the remaining seven workspaces row by row | queued |
| Windows host | Tauri 2 host + sidecar supervision (§14.2), WebView2 packaging, EXE/MSI installer, clean-machine verification, PySide6 retirement | **BLOCKED** (unchanged boundary) |

The queued rows are honest plans, not claims. The frontend rows are
NEW-feasible (Node toolchain); the Windows rows stay blocked until a
Windows host exists.
