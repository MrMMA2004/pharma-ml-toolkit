# BACKEND–FRONTEND TRACEABILITY — MASTER_PROMPT part20 deliverable B (round 370)

The standing per-operation matrix is ROUND368_API_TRACEABILITY_MATRIX.md
(148 rows, r368 phase3: runtime app re-derived live via
`create_app().openapi()` and byte-compared to contracts/openapi.json).
This document is the part20 §15-B deliverable: the operation census
re-derived, the consumer classification revalidated against live code
(this round), the nine-workspace control-to-route coverage statement, and
the honest corrections to the r368 matrix discovered this round.

## 1. Operation census (re-derived this round)

* `contracts/openapi.json`: **142 paths / 149 operations** — 54 GET,
  89 POST, 4 DELETE, 1 PUT, **1 HEAD** (`HEAD /api/v1/artifacts/{artifact_id}/download`,
  `routes/artifacts.py:328`). The r368 matrix's "148" counts every
  operation except the HEAD leg — the correction is D-14 (P3, doc-only):
  the census is 149 with HEAD included.
* Runtime == contract: carried from the r368 phase3 byte-level
  reconciliation (`create_app().openapi()` == contracts/openapi.json after
  normalization; `api.d.ts` regenerates byte-identically via
  openapi-typescript 7.13.0); no drift either way.
* 148 unique operationIds + the HEAD leg; 32 route files under
  `src/pharmaml/api/routes/` mounted at `/api/v1`.

## 2. Consumer classification (revalidated against live code this round)

**Real runtime frontend consumers (typed service → page/component call —
verified by grep + the r368 phase4 per-workspace audit):** every workspace
family (datasets, ml, dl, docking, generative, pkpd, md, viz, quality, runs,
jobs, settings, system, websocket, chemistry, artifacts, database,
molecular3d, samples, admet/AD-analyze + leverage-williams) is wired
end-to-end: React control → hook/service → typed client.ts call →
`X-Session-Token` REST or authenticated WS → FastAPI route → scientific
module → DTO → typed response → state → rendered result. The heavy
families re-verified by direct service-grep this round: mlApi (42 methods —
all consumed, incl. dta :1842, drugresponse :1956, pharmacogenomics :2058
in MLPage.tsx), dlApi (10), dockingApi (15), pkpdApi (12), datasetsApi (8).

**Intentionally API-only routes (documented, 10 — revalidated this round):**

| Route | Backend | Runtime GUI consumer | Documented reason (r368 matrix) |
| --- | --- | --- | --- |
| GET /capabilities | routes/capabilities.py | none (types only) | API/CLI introspection lane |
| GET /capabilities/{id} | same | none | same |
| GET /applicability-domain/catalog | routes/applicability_domain.py | none | catalog introspection; GUI uses /analyze + /leverage-williams (services/ml/client.ts:2159/:2190) |
| POST /applicability-domain/knn | same | none | API-only analysis lane |
| POST /applicability-domain/domain-box | same | none | API-only analysis lane |
| GET /model-evaluation/catalog | routes/model_evaluation.py:122 | none | catalog introspection |
| POST /model-evaluation/conformal-regression | model_evaluation.py:256 | none | **the real GUI-coverage gap** (registered; recommendation below) |
| POST /model-evaluation/conformal-classification | model_evaluation.py:349 | none | same |
| POST /model-evaluation/coverage-diagnostics | model_evaluation.py:436 | none | same |
| GET /workflow/runs/{run_id} | routes/workflow.py | none (ml service implements list/steps/templates/validate/run only, client.ts:2775-2838) | detail introspection lane |

The r368 matrix's prose said "nine" while enumerating these 10 routes —
corrected here (D-14, same doc-only family).

**Dead frontend service methods (D-13, P3 — defined, exported, never called
from any page/component or test):**

| Method | File:line | Backend op |
| --- | --- | --- |
| unsupervisedApi.methods | frontend/src/services/ml/unsupervised.ts:335 | GET /api/v1/unsupervised/methods |
| reinforceApi.methods | unsupervised.ts:384 | GET /api/v1/reinforce/methods |
| reinforceApi.evaluate | unsupervised.ts:417 | POST /api/v1/reinforce/evaluate (matrix row 111's consumer is service-only) |
| generativeApi.priorView | frontend/src/services/generative/client.ts:271 | GET /api/v1/generative/priors/{prior_id} |

Decision (part20 §12 minimal-change): **kept + documented** — they are
thin typed wrappers giving the frontend the same typed surface the API
documents (the API-only lanes stay reachable from the typed client without
a GUI); removal buys nothing and shrinks the typed parity. Documented here
as the honest D-13.

**Correctly NOT flagged** (verified this round): drugresponse, dta,
pharmacogenomics, and the reinforce train lane all have full GUI wiring
(MLPage.tsx :1786/:1842/:1956/:2058+ + the ReinforceLane); `runsApi.detail`
is used (RunLineage.tsx:329).

## 3. Nine-workspace coverage statement (r368 phase4 + this round's re-checks)

| Workspace | Page | Verified wiring (controls → routes) | Placeholder/dead controls found |
| --- | --- | --- | --- |
| Introduction | features/introduction | info panels + system health via system service | none |
| Datasets | features/datasets | import/preview/validate/preprocess; chemistry lanes (pull-column/descriptors/fingerprints with in-flight busy guards — the r368 F-1 fix) | none |
| Machine Learning | features/ml | registry/train/predict/inference; AD analyze + leverage-williams; dta/drugresponse/pharmacogenomics/reinforce/RSM/multiresponse/unsupervised/semi-supervised lanes | none |
| Deep Learning | features/dl | catalog/train (6 tabular families incl. transformer/cnn-predict lane)/predict/evaluation | none |
| Generative AI | features/generative | generate/validate/SA; priors list (create/read wired; priorView service-only as D-13) | none |
| Molecular Docking | features/docking | receptor/ligand prep; engine census; campaign lanes | none |
| PK/PD | features/pkpd | catalog/NCA/parameter lanes + plots | none |
| Molecular Dynamics Analysis | features/md | catalog + analysis lanes | none |
| Visualization | features/visualization | stat figures, exports digest-verified | none |

Production frontend greps (this round): zero TODO/FIXME/not-implemented/
coming-soon/alert()/confirm() in `frontend/src` production files; every
"placeholder" hit is an HTML input attribute; every "stub" hit is test-only.
The r368 phase4 conclusion ("no fake progress, no hardcoded results, no
placeholder buttons anywhere") re-verified.

## 4. Contract hygiene (part20 §4.2)

* OpenAPI == runtime (byte-level, r368 phase3; HEAD leg counted this round).
* api.d.ts is generated (`bun run generate:types`), never hand-edited; no
  manual type patching to conceal mismatch exists (verified by the
  regeneration identity).
* Auth/session: `X-Session-Token` header on REST (client.ts:57/87/115);
  WS token-as-query is the documented browser limitation (loopback-only
  mitigates; r339 D-6/D-8 origin + policy-close lanes ride the backend and
  client).

## 5. Known gaps + recommendations (honest register)

1. **Conformal GUI coverage gap (P2, OPEN)** — the conformal family
   (regression/classification/coverage-diagnostics) is backend-complete
   (tests/api/test_model_evaluation.py, 23 conformal refs) with zero runtime
   GUI consumer. Recommendation: a Model Evaluation stage in the Machine
   Learning workspace (the established lane pattern: typed service method →
   form + progress + results card) — deferred as a feature increment, never
   silently claimed as covered.
2. **D-13 dead service methods** — documented above; revisit only if the
   corresponding GUI lanes are built (they become the consumers).
3. **D-14 census corrections** — this document carries the corrected counts
   (149 ops incl. HEAD / 10 API-only routes); the ROUND368 matrix stays as
   the historical 148-row artifact with its own conventions.
