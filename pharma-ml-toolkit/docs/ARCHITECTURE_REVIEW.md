# ARCHITECTURE REVIEW — the §8 Scientific Workflow Engine evaluation

Part13 §21 deliverable #3 (ARCHITECTURE_REVIEW.md), opened by the
Phase-6 round-322 lane: the §8 "ARCHITECTURE IMPROVEMENT: SCIENTIFIC
WORKFLOW ENGINE" evaluation, executed exactly as the directive
orders — *"Do not introduce a large workflow engine automatically.
First assess whether the existing JobManager and capability registry
can support a lightweight implementation. Keep the application
fully local and offline-capable."*

- **Audit date**: 2026-11-22 (round 322, the 114th recorded run;
  the interactive part13 continuation over the thirteenth sandbox
  rollback, recovered per the r306–r313 playbook before any lane
  work: PAT re-spliced and API-validated masked, origin re-attached,
  fetched, the tree reset onto the r321 terminal `c9df2a5d`, the
  rollback's 14 ghost-residue paths swept — the tree 0/0 clean — the
  venv re-pinned foreground, the frontend `npm ci` after its
  snapshot-wipe).
- **Audited base**: the r321 terminal `c9df2a5d` (origin/main,
  verified 0/0 after reset); this round's lane rides the round-322
  commit over it.
- **Scope**: §8's workflow-engine requirement list (the graph, the
  typed input/output contracts, dependency validation, workflow
  serialization, saved workflow templates, execution history,
  progress reporting, cancellation, partial failure handling,
  artifact lineage, reproducibility metadata) + the four directed
  candidate reviews + the decision and its execution (the
  lightweight in-tree engine `pharmaml.workflow` + the
  `/api/v1/workflow` REST lane).
- **Commands executed** (this session, verbatim anchors): `pytest
  tests src/pharmaml --doctest-modules --collect-only -q` (8,523 /
  0 errors / 467 files, the re-trued pin); `pytest
  tests/test_workflow_engine.py tests/api/test_workflow.py -q`
  (39/39); `pytest tests/api/ tests/test_web_parity.py
  tests/test_architecture_hygiene.py` (976 passed / 2 standard
  skips); `ruff check` + `mypy` over the new modules (both clean);
  `python tools/export_openapi.py` (108 paths / 71 schemas);
  `npm --prefix frontend run generate:types` (byte-lockstep with the
  exported contract).
- **Test results**: see above; every number was produced by the
  commands named, none inherited.
- **Known limitations**: named inline below and in the engine's own
  honest notes — the mutual-info selector member is unseeded in the
  backend (run-to-run column-set drift); the report PDF embeds
  fpdf's creation timestamp (byte drift at equal content); the
  in-session run ledger lives as long as the process (job-mode runs
  persist through the JobManager's SQLite history); the REST run
  lane is bounded (sync ≤ 12 nodes / 80 records, job ≤ 24 nodes /
  200 records) — the honest bound, not a hidden cap.
- **Remaining blockers** (unchanged from PROJECT_STATUS §4):
  clean-machine smoke (C-3/C-5), Stage-5 PySide6 retirement,
  installer signing (certificate-gated).

## 1. The §8 requirement list → the delivered surface

| §8 requirement | Where it lives (round 322) | Evidence |
| --- | --- | --- |
| Workflow graph | `pharmaml.workflow.engine.WorkflowSpec` (nodes + typed wiring) | `tests/test_workflow_engine.py` TestTemplates/TestValidation |
| Typed input/output contracts | `pharmaml.workflow.contracts` (PortSpec/StepSpec, the closed `PORT_KINDS` vocabulary; kind EQUALITY is compatibility) | TestStepRegistry; `GET /api/v1/workflow/steps` |
| Dependency validation | `WorkflowSpec.validate()` — unknown steps/nodes/ports, kind mismatches, duplicate node ids (parse-level), cycles — every finding at once, pre-execution | `POST /api/v1/workflow/validate` (422 with the full error list) |
| Workflow serialization | `to_dict` / `from_dict` (JSON-safe round trip) + `canonical_digest` (sha-256 over the sorted canonical form) | TestSerializationStability |
| Saved workflow templates | `pharmaml.workflow.templates` — §8's own example chain (`guided-qsar`, 10 nodes) + the minimal core (6 nodes) | `GET /api/v1/workflow/templates` |
| Execution history | `WorkflowEngine`'s bounded in-session ledger + the JobManager's SQLite job records (job mode) | `GET /api/v1/workflow/runs[/{id}]` |
| Progress reporting | the per-node `progress` callback, wired to `JobManager.report_progress` (job mode: `node i/n: <id> <state>`) | `POST /run {mode: job}` → `/api/v1/jobs/{id}` |
| Cancellation | the cooperative `cancel_event`, checked BETWEEN nodes (the JobManager's own injection) | TestCancellation; the API cancel lane |
| Partial failure handling | a failed node records the backend's error VERBATIM; downstream nodes SKIPPED with the reason; completed artifacts SURVIVE; run state PARTIAL | TestPartialFailure; the API partial-failure lane |
| Artifact lineage | every output artifact carries producer node/port + the TRANSITIVE upstream chain + a content digest (honest `digest_kind`: dataframe-shape-rowhash / estimator-params / json) | TestGoldenPath lineage asserts |
| Reproducibility metadata | the run block: toolkit version, python, platform, seed, spec digest, the honest determinism note | TestGoldenPath; every run answer |

## 2. The four directed candidates — reviewed honestly

The directive names four references. Reviewed against THIS
application's own constraints (fully local, offline-capable, no
heavy new dependencies, the existing JobManager already providing
jobs/progress/cancellation/history):

- **PrefectHQ/prefect** — a full orchestration platform: its happy
  path is a server/orion database + flows/deployments + the
  prefect-client runtime; adopting it means a new service process,
  a schema-bearing state database, and a dependency tree far beyond
  the toolkit's pinned scientific stack. It solves multi-team
  scheduled orchestration, not a single-user offline studio. What
  transfers is the DESIGN vocabulary (flow/task typing, state
  machines, run-state ledgering) — all of which the in-tree engine
  adopts in miniature.
- **knime** — a Java-based visual analytics PLATFORM, not an
  embeddable Python engine; its nodes/marketplace paradigm cannot
  ride inside a PySide6/Python-local application. What transfers is
  the node-graph UX idea (the GUI lane's stage chain, queued for
  the next continuation).
- **jupyterlab** — a notebook workbench: its DAG is a document
  cell graph, not a typed contract graph; it carries no
  cancellation/lineage/reproducibility contract for saved
  multi-stage workflows, and embedding a Jupyter server violates
  the offline-single-process posture.
- **marimo-team/marimo** — the closest philosophical relative
  (reactive, reproducible, git-friendly Python notebooks) — but it
  is a notebook application with its own editor/runtime, and §8
  asks for a workflow ENGINE inside THIS application, not a second
  application to host it. What transfers is the reproducibility
  discipline (the run's metadata block mirrors marimo's own
  "how this document was computed" stance).

## 3. The decision

**The existing JobManager and capability registry CAN support the
lightweight implementation — and now do.** The engine
(`pharmaml.workflow`: contracts + runners + engine + templates) is
pure in-tree Python, zero new dependencies; every step delegates to
a REAL backend callable that already existed (standardize_smiles,
compute_descriptors, make_train_test_split, select_kbest_union,
build_model, evaluate_model, leverage_williams, shap_summary,
save_pdf_report); job-mode runs ride the EXISTING JobManager for
progress, cooperative cancellation, and the SQLite history. The
capability-registry discipline (one frozen record per step, typed
ports, explicit notes) is applied verbatim to the step registry.

Honest engineering notes captured during the lane (each now pinned
by a test):

- **The leakage reorder**: §8's illustrative chain lists Feature
  Selection before Data Split; the toolkit's own leakage rule
  (selection fitted on training data only — the look.ipynb fix)
  wins, so the guided template splits FIRST and the template's
  metadata says so explicitly.
- **The rank hygiene**: the leverage/Williams backend's own
  full-rank guard ("drop the collinear descriptors") fires on
  exactly-collinear descriptor columns (e.g. MaxAbsEStateIndex vs
  MaxEStateIndex coincide on all-positive E-State sets); the
  selection step now executes the guard's own prescription — a kept
  column survives only if it adds rank to `[1 | X]`, every dropped
  name + reason recorded in the artifact.
- **The determinism scope**: seeded splits/estimators reproduce;
  the f_regression selector is deterministic while the mutual-info
  union member is unseeded in the backend; the report PDF embeds
  fpdf's creation timestamp. All three stated in the run's own
  determinism_note — never hidden behind a fake determinism claim.
- **The honest guard firing**: a 12-row frame still fails the
  leverage/Williams `p+2` rows requirement (rank hygiene cannot
  conjure rows) — the partial-failure lane pins it as the honest
  outcome, not a bug.

## 4. Next (the Phase-6 remainder, in order)

- The **GUI lane EXECUTED (round 323, the interactive part13
  continuation, trace 1a2022c2d26dbc05)**: the Workflow Builder
  stage (stage 17 of the Machine Learning workspace, the later
  stages renumbered 18/19/20) — the construct/save/reproduce
  surface §8's objective names, over this engine: the template
  select seeding the editable spec JSON canvas, the validate gate,
  the sync + job run lanes (the JobManager relay), the run record's
  lineage + reproducibility rendering, and the bounded runs ledger;
  8 component tests + the browser-E2E golden path over the REAL
  backend (zero page/console errors, the r323 evidence pair under
  `docs/gui/screenshots/r323-*`).
- Resource monitoring, security scanning lanes, release-provenance
  and documentation-automation sub-lanes follow (Phase 6's own
  list), each small, reversible, tested.
