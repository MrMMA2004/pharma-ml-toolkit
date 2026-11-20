# Backend–GUI Integration Matrix (part12 Phase 3)

Machine-verifiable mapping: **backend module → public API → TypeScript
client → workspace → user flow → verdict**. 89 endpoints audited
(88 REST ops + 1 WebSocket) across 17 routers under `/api/v1`; the
OpenAPI 3.1 contract lives at `contracts/openapi.json` (83 paths / 88 ops /
55 schemas) and the generated client types at `frontend/src/types/api.d.ts`.

Verdict legend: **WIRED** = a React workspace consumes it end-to-end;
**PARTIAL** = consumed by a thinner lane than the route offers;
**NON-GUI (documented)** = real, tested backend surface with no GUI
consumer — its status is explicitly recorded here per part12's acceptance
criterion ("verified GUI workflow **or** an explicitly documented non-GUI
status"); **LEGACY** = served to the PySide6/CLI surfaces instead.

## A. Route-level matrix (by router)

| Route | Backend module | TS client | Consumer workspace | Verdict |
| --- | --- | --- | --- | --- |
| `GET /health` | `api/server.py` | `services/api/startup.ts` | StartupGate §14.1 | WIRED |
| `GET /capabilities`(+`/{id}`) | `studio/capability_registry.py` (62 rows) | — | none | NON-GUI (documented): capability registry is an audit surface (docs/CAPABILITY_*.md consume it) |
| `GET /jobs`, `/{id}`, `/result`, `POST /{id}/cancel` | `api/jobs.py` | `services/jobs/client.ts` + `tracker.ts` | every workspace (JobRunCard) | WIRED |
| `POST /chemistry/descriptors` | `chem/descriptors.py` | `services/chemistry/client.ts` | Datasets (stage: Descriptor/Fingerprint) | WIRED |
| `POST /chemistry/fingerprints` | `chem/descriptors.py` | same | same | WIRED (morgan/rdkit/atom_pair/avalon; radius/nBits/counts) |
| `POST /datasets` + 6 read/split/profile ops | `data.py` + `api/stores.py` + `ml/splits.py` | `services/datasets/client.ts` | Datasets | WIRED |
| `GET /ml/models` (+ trained registry, predict, explain) | `ml/models.py`, `training.py`, `explain.py` | `services/ml/client.ts` | Machine Learning | WIRED |
| `POST /ml/train` | `training.py` | same | ML stage 7 (job mode) | WIRED |
| `GET /dl/networks`, `POST /dl/train` | `ml/numpy_neural.py` | `services/dl/client.ts` | Deep Learning | WIRED (+ live `training.metrics` WS) |
| `POST /semi-supervised/train` | `ml/semi_supervised.py` | `services/ml/client.ts` | ML semi-supervised stage | WIRED |
| `POST /semi-supervised/evaluate` | `ml/semi_supervised.py` | the /evaluate GUI consumer (r303) | ML semi-supervised stage | WIRED (r303; the wiring exposed + fixed the job-mode TypeError — the COMPLETION pin; NON-GUI at this audit, trued r307) |
| `GET /generative/objectives` (+ priors, rnn train/sample, optimize) | `generative/*` | `services/generative/client.ts` | Generative AI | WIRED |
| `GET /docking/engines`, dockstring catalog/objective, receptor/ligand prep, parse vina/dlg, scoring×2, contacts, pose-rmsd, `POST /dock` | `docking/*` | `services/docking/client.ts` | Molecular Docking | WIRED (3D via Molecule3D; dock execution honestly 422 off-Windows) |
| `GET /pkpd/catalog` + nca/steady-state/iv-bolus/bateman/superposition/pbpk/pd/population | `pharma/pk.py`, `pkpd/*` | `services/pkpd/client.ts` | PK/PD | WIRED |
| `GET /md/catalog` + synthetic/analyze/xvg/enm/pca/study | `md/*` | `services/md/client.ts` | MD Analysis | WIRED (+ 3D playback) |
| `GET /visualization/catalog` + stat/concentration-profile/dose-accumulation/md-series/table/report | `viz/*` | `services/visualization/client.ts` | Visualization | WIRED (report = job + per-section WS progress) |
| `GET /applicability-domain/catalog` | `ml/applicability.py` | — | none | NON-GUI (documented): catalog listing; the ML page pins kNN AD via `/analyze` |
| `POST /applicability-domain/knn` | `ml/applicability.py` | — | none | NON-GUI (documented): the single-shot kNN lane; `/analyze` is the composite consumed by the GUI |
| `POST /applicability-domain/analyze` | `ml/applicability.py` | `services/ml/client.ts` | ML AD stage | WIRED (kNN distance + bootstrap uncertainty plots) |
| `POST /applicability-domain/domain-box` | `ml/applicability.py` | — | none | NON-GUI (documented): metadata-box audit lane (DomainBox) |
| `POST /artifacts/figure|table|model`, list/view/delete/download | `api/stores.py` ArtifactStore | `services/artifacts/client.ts` | every workspace (exports + digest-verified downloads) | WIRED — **this client was the B-1 casualty: the shipped v1.2.231 zip omitted it; scoped exclusion fixed this round** |
| `POST /artifacts/table` (creation lane) | ArtifactStore | — (workspaces post tables via `/visualization/table`) | none direct | NON-GUI (documented): creation lane superseded by the visualization table route |
| `GET /database`, `/database/jobs`, preferences CRUD, `POST /backup`, `POST /restore` | `api/persistence.py` | `services/database/client.ts` (**new this round**) | Introduction (studio database card) | WIRED (was NON-GUI before the B-2 fix) |
| **WS** `/events` | `api/events.py` | `services/websocket/client.ts` | StartupGate + all workspaces | WIRED (hello envelope, since-cursor replay, heartbeat, backoff, dedupe) |

## B. Focus-area chains (part12 PART 4's machine-verifiable five)

1. **Descriptors**: `chem/descriptors.py` → `POST /chemistry/descriptors`
   → `chemistry/client.ts` → DatasetsPage stage (job mode) → table + export. **COMPLETE**.
2. **ML training**: `ml/training.py` → `POST /ml/train` → `ml/client.ts`
   → MLPage stage 7 → JobRunCard → parity Plotly scatter + export. **COMPLETE**.
3. **Docking prep**: `docking/preparation.py`/`receptor.py` → prep routes →
   `docking/client.ts` → DockingPage stages → PDBQT + 3D + digest. **COMPLETE**.
4. **NCA**: `pharma/pk.py`/`pkpd/nca_ext.py` → `POST /pkpd/nca` →
   `pkpd/client.ts` → PKPDPage stage 2 → summary/terminal-phase panels +
   concentration curve + export. **COMPLETE** (desktop). Web-explorer note:
   the Next.js `nca-panel` card is a parity-pinned TS mirror, not this route
   — documented as the explorer's design, not the product lane.
5. **MD xvg**: `md/xvg.py` → `POST /md/xvg` → `md/client.ts` → MDPage →
   PlotlyFigure series + stats + export. **COMPLETE**.

## C. Orphan/coverage findings (all verified this session)

- **GUI controls with no real backend**: none found (every control traces
  to a route or an explicitly client-side calculator in the web explorer).
- **Backend routes with no GUI consumer**: 6 as of the r303 follow-up
  (the semi-supervised/evaluate lane gained its consumer); the remaining
  six carry explicit NON-GUI (documented) status in section A
  (capabilities, AD catalog/knn/domain-box, artifacts/table creation
  lane) — re-verified at the r307 reconciliation.
- **Mock-only integrations**: none in the desktop app. The Next.js web
  explorer intentionally mirrors some calculators client-side (pk-simulator,
  pd-effect, pharma-calcs, ml/train mirror) with parity probes
  (`tests/test_web_parity.py`); real-Python lanes there: chem/features,
  chem/sa-score, docking/prep, docking/rescore, gen/mcts, gen/surgery,
  md/*, ml/shap, ml/drugresponse, report/session.
- **Incorrect request/response mappings**: none found; typed clients are
  runtime-narrowed (§8.1 style) and contract-drift-guarded
  (`tools/export_openapi.py --check`).
- **The Introduction "backup and restore" claim**: was unsubstantiated at
  GUI level → **wired this round** (B-2).

## D. Web explorer (repo root `src/`) — separate surface, documented

35 playground cards across 6 sub-tabs; 29 API routes. The explorer is the
web console (not the desktop product): its TS-mirror lanes are documented
above; the archive/download routes serve the shipped zip (this round's
B-1/B-1b scoping fixes keep the fallback byte-parity with the release
artifact).
