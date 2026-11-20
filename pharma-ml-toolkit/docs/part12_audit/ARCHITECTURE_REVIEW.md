# Architecture Review (part12 Phase 1)

Baseline v1.2.231 / `7bbc41ed`. This review verifies the architecture
against part12 PART 2's mandatory stack and PART 4's boundary rules, and
records the findings that carry engineering consequences.

## 1. Layer map (verified)

```
React 19 + TS + Vite (frontend/)      UI · workspace orchestration · typed
   │  OpenAPI-generated types (api.d.ts) + runtime-narrowed clients
   ▼  REST (JSON/form/raw lanes) + WS /api/v1/events (token header/query)
FastAPI (src/pharmaml/api)            17 routers, 88 ops + WS · security
   │  JobManager (thread pool, cooperative cancel) · EventBus · stores
   ▼
Python engines (src/pharmaml/*)       chem · ml · generative · docking ·
   │                                 pharma/pkpd · md · biology · viz
   ▼
SQLite (studio.db, WAL) + FileVault   persistence.py · artifacts · backups
Tauri 2 host (src-tauri/, Rust)       lifecycle · handshake · loopback
                                      assert · one IPC command
```

- **The frontend never touches SQLite directly** — verified: no sqlite
  import in `frontend/`; all persistence flows through REST.
- **The backend depends on neither React nor Tauri** — verified: no
  frontend imports anywhere in `src/pharmaml`.
- **PySide6 legacy GUI (ui/)** runs the same engines **in-process** (no
  HTTP) — a parallel surface, not a client of the API; retirement gated
  (Stage-5).

## 2. Strengths (preserved, evidence-backed)

1. **Contract-first**: OpenAPI 3.1 export + `--check` drift gate +
   generated `api.d.ts`; typed clients validate payloads at runtime —
   drifted contracts fail loudly, not silently.
2. **Job architecture**: one JobManager, real backend callables on a
   thread pool, cooperative `JobCancelled`, SQLite-upserted history,
   REST-authoritative state with WS progress — never duplicated per
   workspace.
3. **Security posture**: per-startup random token (memory-only),
   constant-time verify, loopback Host/Origin assert, strict CSP,
   single IPC command in the host — no fs/shell/dialog commands exposed.
4. **Offline-first**: Vite `assetsInlineLimit: 0` + manualChunks,
   WebView2 offline installer, loopback-only connect-src.
5. **Test depth**: 8,590 collected (0 errors) including src doctests;
   drift guards pin workspace order, WS vocabulary, startup gate,
   version carriers, workflow registration.

## 3. Findings (ranked by engineering consequence)

| # | Finding | Consequence | Status |
| --- | --- | --- | --- |
| A-1 | Any-depth `artifacts` dir exclusion in tree walk + ZIP patterns swallowed `frontend/src/services/artifacts/` | Real source hidden from explorer, census undercount, **shipped v1.2.231 zip missing exactly that file** | **Fixed this round** (B-1, scoped relative-path exclusions) |
| A-2 | `frontend/dist` (Vite output) not excluded anywhere | 5.7 MB build output leaks into census/archive the moment a local build exists (latent — first fired this session) | **Fixed this round** (B-1b) |
| A-3 | `/database` route group had no GUI consumer while Introduction claimed backup/restore | Unsubstantiated user-facing claim; persistence surface unreachable from the product | **Fixed this round** (B-2 wiring) |
| A-4 | Nested `pharma-ml-toolkit/.github/workflows/ci.yml` unregistered + corrupted trigger (`branches: ain]`) | Dead config, misleads audits; root workflow is the real one | **Removed this round** (B-3) |
| A-5 | Census LOC convention counts `py/toml/txt/md/yml/json/ipynb` only | React/Rust layers (`~19.6k` + `~1k` lines) count as files, never as LOC; `.github` yml files DO count (A-4 dropped the census by 52 lines) | Documented (methodology note for the next census decision) |
| A-6 | 9 library-surface modules imported nowhere (fih, pkstats, odesim, association, admet, lime, dataset, surrogate, ebm) + 3 duplicate-symbol families (gdsc_anova ×2, mabel_dose ×3, scaffold_split ×3) | Doctest-only coverage; capability implemented in multiple places invites drift | Documented → FEATURE_GAP_AND_GITHUB_REFERENCES.md (remediation needs an owner decision per "don't rewrite working subsystems") |
| A-7 | Tk playground (`playground.py`, 30,047 lines) declared retired (r228 studio docstring) yet still shipped with `test_playground.py` (16,370 lines, the largest test file) | Dead-weight in archive + installer; README says retired at v1.2.141 | Documented → BUG_REPORT B-6 (retirement is a Phase-8 gate decision, not a this-round deletion) |
| A-8 | One IPC command + no file dialogs in Rust; uploads are paste/browser-lane only | No native file picker UX | Accepted design (browser upload lanes); noted for backlog |

## 4. Boundary hygiene checks

- Version-carrier lockstep (8 carriers): guarded + green this session.
- Workflow registration: root-level (round-293 rule) + 9 drift guards;
  the nested copy is gone (A-4).
- Archive composition: this round's builder carries the scoped exclusion
  set (A-1/A-2) + the standing five-home byte-parity ritual.

## 5. Verdict

The architecture is **sound and consistent with the mandatory stack** —
no technology migration needed. The one systemic weakness this audit
found (any-depth exclusion patterns in the explorer/ZIP layer) is now
fixed at its root with scoped, path-anchored rules; the residual lanes
(A-5..A-8) are documented with owners rather than silently rewritten —
per part12's primary engineering principle.
