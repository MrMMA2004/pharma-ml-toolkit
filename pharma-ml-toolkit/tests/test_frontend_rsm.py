"""part17 §6 (the R1 lane) — the Response Surface Methodology drift
guards.

The Python gate cannot execute the React section, but every contract
the lane rides CAN be pinned against the real frontend + backend
source — a rename, a dropped honesty note, a second registration
path, or an invented null turns a guard red before the surface
drifts:

* the backend module + the route + both test suites exist (the design
  generators, the second-order fit with its ANOVA partition, the
  desirability optimiser, the persistence bundle);
* the route is registered EXACTLY once in the aggregator, tagged, and
  carrying the seven paths the contract export grew (128 paths / 83
  schemas, the seven rsm paths named);
* the design honesty: the rotatable CCD's axial runs OUTSIDE the
  declared ranges is stated in the module source, the Box-Behnken /
  full-factorial blocking refusal is typed, the 512-run bound and the
  factor-range validation are pinned;
* the fit honesty: the pure error / lack-of-fit nulls where no
  replicated design points exist (the honest-null source), the
  hierarchy notes flag-never-remove, the aliasing disclosure, the
  machine-precision diagnostics guard;
* the optimisation honesty: the Derringer-Suich goals vocabulary, the
  region confinement note (cannot extrapolate by construction), the
  bounds-source disclosure, the Pareto sample note;
* the frontend typed client: the eight service methods + the runtime
  parsers + the export lane exist in ``services/ml/client.ts``;
* the workspace integration: the Experimental Design & Response
  Surface section rides the EXISTING Machine Learning workspace
  (stage index 21, the Export stage renumbered 22 — no tenth
  workspace, the mandate's own rule), and its contour visualisations
  stay in the Machine Learning workspace, NOT the Visualization one;
* the client-side gates: the factor validation + the goal validation
  fire BEFORE any request;
* the honest renders: the coded AND actual run table, the
  assumptions, the zone vocabulary, the fixed-factors display, the
  SYNTHETIC demo labelling;
* the vitest suite exists with the mandated lanes (9 tests, counted
  green this round — the suite-level pin lives in the QA surface).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
ML_FEATURES = SRC / "features" / "ml"
ML_CLIENT = SRC / "services" / "ml" / "client.ts"
MLPAGE = ML_FEATURES / "MLPage.tsx"
VITEST = ML_FEATURES / "__tests__" / "RsmLane.test.tsx"
MODULE = ROOT / "src" / "pharmaml" / "ml" / "rsm.py"
ROUTE = ROOT / "src" / "pharmaml" / "api" / "routes" / "rsm.py"
ROUTE_INIT = ROOT / "src" / "pharmaml" / "api" / "routes" / "__init__.py"
API_TESTS = ROOT / "tests" / "api" / "test_rsm.py"
CONTRACT = ROOT / "contracts" / "openapi.json"
API_TYPES = SRC / "types" / "api.d.ts"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the lane's own files exist


def test_the_module_route_and_both_suites_exist() -> None:
    assert MODULE.is_file(), "the part17 §6 rsm module is missing"
    assert ROUTE.is_file(), "the /api/v1/rsm route is missing"
    assert API_TESTS.is_file(), "the API test suite is missing"
    assert VITEST.is_file(), "the vitest component suite is missing"
    assert ML_CLIENT.is_file()


def test_the_route_is_registered_exactly_once() -> None:
    init = _read(ROUTE_INIT)
    assert init.count("rsm,") == 1, "the import drifted"
    assert init.count("include_router(rsm.router)") == 1, (
        "the rsm router must register exactly once"
    )
    route = _read(ROUTE)
    assert 'prefix="/rsm"' in route
    assert 'tags=["rsm"]' in route


def test_the_contract_carries_the_seven_new_paths() -> None:
    schema = json.loads(CONTRACT.read_text())
    paths = schema["paths"]
    # 134 -> 142 by the r367 part18 §5 DL architecture expansion (the
    # CNN pair, the autoencoder pair, the GAN pair, the diffusion pair);
    # 142 -> 143 by the r378 GEN-2 metrics path (the pin left stale at
    # 142 in that round — caught and trued by the r379 DS-2 round);
    # 143 -> 144 by the r379 DS-2 standardization path; 144 -> 145 by
    # the r380 DS-3 frames/register path; 145 -> 146 by the r382 ML-4
    # unsupervised pca sibling path
    assert len(paths) == 146, f"the contract drifted: {len(paths)} paths"
    # 87 by the r365 part18 lanes; 87 -> 96 by the r367 part18 §5
    # DL architecture expansion (the four pairs' request DTOs)
    # 96 -> 97 by the r375 ML-3 train-lane split section (the
    # TrainSplitSpec request DTO, one schema; no new paths)
    # 97 -> 98 by the r378 GEN-2 GenerationMetricsRequest DTO (this
    # pin left stale at 97 in that round — caught and trued by the
    # r379 DS-2 round); 98 -> 99 by the r379 DS-2 StandardizeRequest;
    # 99 -> 100 by the r380 DS-3 FramesRegisterRequest; 100 -> 101 by
    # the r382 ML-4 PcaRequest DTO
    assert len(schema["components"]["schemas"]) == 101
    for path in (
        "/api/v1/rsm/registry",
        "/api/v1/rsm/design",
        "/api/v1/rsm/fit",
        "/api/v1/rsm/predict",
        "/api/v1/rsm/contour",
        "/api/v1/rsm/optimize",
        "/api/v1/rsm/models",
    ):
        assert path in paths, f"{path} missing from the exported contract"
    # and the regenerated types carry the request schemas
    types = _read(API_TYPES)
    for schema_name in (
        "DesignRsmRequest",
        "FitRsmRequest",
        "PredictRsmRequest",
        "ContourRsmRequest",
        "OptimizeRsmRequest",
        "GoalSpec",
        "ConstraintSpec",
    ):
        assert schema_name in types, f"{schema_name} drifted from api.d.ts"


# ---------------------------------------------------------------------------
# the design honesty (§6.3)


def test_the_rotatable_alpha_disclosure_is_pinned() -> None:
    module = _read(MODULE)
    # the axial runs sit OUTSIDE the declared ranges — disclosed
    assert "OUTSIDE the declared factor ranges" in module
    # the rotatable alpha formula itself
    assert "(2.0**n_factors) ** 0.25" in module
    # the faced alpha stays inside
    assert 'alpha_type == "faced"' in module


def test_the_design_refusals_are_typed() -> None:
    module = _read(MODULE)
    # the Box-Behnken minimum factor count
    assert "a Box-Behnken design needs at least 3 factors" in module
    # the full factorial cap
    assert "capped at 8 factors" in module
    # the blocking honesty: the CCD two-block split only
    assert "honestly refused, never faked" in module
    # the 512-run bound
    assert "MAX_DESIGN_RUNS = 512" in module
    # the factor contract: low < high, unique names, 2..10
    assert "factor names must be unique" in module
    assert "needs low < high" in module
    # the coded/actual dual table (the experiment sheet contract)
    assert "coded" in module and "actual" in module


# ---------------------------------------------------------------------------
# the fit honesty (§6.4)


def test_the_anova_nulls_are_honest() -> None:
    module = _read(MODULE)
    # no replicated design points -> the pure error / LOF rows are
    # nulls, never invented
    assert "nulls, never invented" in module
    # the noise-free pure-error F is UNDEFINED -- reported null
    assert "UNDEFINED" in module
    # the lack-of-fit needs genuinely replicated design points
    assert "no replicated design points" in module


def test_the_hierarchy_notes_flag_never_remove() -> None:
    module = _read(MODULE)
    assert "NEVER auto-removed" in module
    assert "keep the parents anyway" in module
    # the aliasing disclosure exists at source
    assert "rank deficient" in module
    assert "separably" not in module  # the honest spelling, never the typo
    assert "separately interpretable" in module


def test_the_machine_precision_diagnostics_guard_exists() -> None:
    module = _read(MODULE)
    # the noise-free demo class reports its scaled diagnostics as
    # exact zeros -- never flags floating-point dust as outliers
    assert "machine precision" in module
    assert "never the floating-point" in module


def test_the_demo_experiment_is_labelled_synthetic() -> None:
    module = _read(MODULE)
    assert "the seeded SYNTHETIC experiment frame" in module
    route = _read(ROUTE)
    assert "SYNTHETIC" in route
    assert '"synthetic": True' in route


# ---------------------------------------------------------------------------
# the optimisation honesty (§6.6)


def test_the_desirability_goals_and_region_notes() -> None:
    module = _read(MODULE)
    # the Derringer-Suich goal vocabulary
    assert 'GOALS = ("maximize", "minimize", "target", "range")' in module
    # the region confinement -- cannot extrapolate by construction
    assert "cannot extrapolate" in module or "CANNOT extrapolate" in module
    # the bounds fall back to the observed design min/max, stated
    assert "observed" in module
    assert "never silent" in module
    # the Pareto alternatives are a SAMPLE, honestly not the front
    assert "NOT the exhaustive Pareto front" in module or "NOT the" in module


# ---------------------------------------------------------------------------
# the frontend typed client


def test_the_typed_client_carries_the_eight_methods() -> None:
    client = _read(ML_CLIENT)
    for method in (
        "rsmRegistry",
        "rsmDesign",
        "rsmFit",
        "rsmPredict",
        "rsmContour",
        "rsmOptimize",
        "rsmModels",
        "rsmExport",
    ):
        assert f"async {method}(" in client, f"the {method} method drifted"
    # the runtime parsers exist (the REAL ones, never re-declared)
    for parser in (
        "parseRsmRegistryResponse",
        "parseRsmDesignResponse",
        "parseRsmFitResponse",
        "parseRsmPredictResponse",
        "parseRsmContourResponse",
        "parseRsmOptimizeResponse",
        "parseRsmModelView",
    ):
        assert f"function {parser}(" in client
    # the typed response interfaces
    for iface in (
        "RsmRegistryResponse",
        "RsmDesignResponse",
        "RsmFitResponse",
        "RsmPredictResponse",
        "RsmContourResponse",
        "RsmOptimizeResponse",
        "RsmModelView",
    ):
        assert f"interface {iface}" in client


# ---------------------------------------------------------------------------
# the workspace integration (no tenth workspace)


def test_the_section_rides_the_existing_ml_workspace() -> None:
    page = _read(MLPAGE)
    assert 'title="Experimental Design & Response Surface"' in page
    # stage 21 inside the SAME workspace, Export renumbered 22
    m = re.search(
        r'index=\{21\}\s*\n\s*title="Experimental Design & Response Surface"',
        page,
    )
    assert m is not None, "the section must hold stage index 21"
    m = re.search(r'index=\{22\}\s*\n\s*title="Export"', page)
    assert m is not None, "the Export stage must renumber to 22"
    # the mandate's own vocabulary rides the section prose
    assert "part17 §6 (R1) lane" in page or "part17 §6, the R1 lane" in page


def test_the_visualisations_stay_in_the_ml_workspace() -> None:
    page = _read(MLPAGE)
    # the contour lane renders inside the Machine Learning workspace's
    # own section -- the §6.7 rule (NOT the Visualization workspace)
    assert 'ariaLabel="the ${rsmContourResult.z_name} contour' in page or (
        "rsmContourResult.z_name} contour" in page
    )
    # and the Visualization workspace carries no rsm lane
    viz_pages = sorted(
        (SRC / "features").rglob("*.tsx")
    )
    for candidate in viz_pages:
        if "MLPage" in str(candidate) or "__tests__" in str(candidate):
            continue
        text = _read(candidate)
        assert "rsmContour" not in text, (
            f"{candidate.name} carries an rsm contour lane -- the §6.7 rule "
            "keeps the RSM visualisations in the Machine Learning workspace"
        )


def test_the_client_side_gates_fire_before_any_request() -> None:
    page = _read(MLPAGE)
    # the factor validation gate
    assert "at least 2 valid factors" in page
    assert "every factor needs low < high" in page
    # the response column gate for the dataset mode
    assert "at least one response column selected" in page
    # the goal validation gate
    assert "every model response needs a goal" in page


def test_the_honest_renders_are_pinned() -> None:
    page = _read(MLPAGE)
    # the coded AND actual run table (both units ride the cells)
    assert "run.coded[f.name]" in page
    assert "run.actual[f.name]" in page
    # the assumptions list rides the generated design
    assert "rsmDesignResult.assumptions.map" in page
    # the zone vocabulary + the extrapolation flag
    assert "row.zone" in page
    assert "extrapolation flagged on" in page
    # the fixed factors display (both units)
    assert "other factors held fixed" in page
    # the SYNTHETIC demo labelling
    assert "KNOWN surfaces" in page
    # the Pareto / region / bounds notes ride the optimisation answer
    assert "rsmOptimizeResult.pareto_note" in page
    assert "rsmOptimizeResult.region_note" in page
    # the export note: NO duplicate export endpoint
    assert "rsmDesignResult.export_note" in page


def test_the_vitest_suite_carries_the_mandated_lanes() -> None:
    suite = _read(VITEST)
    for lane in (
        "renders the stage with the registry vocabulary and the factor editor",
        "generates a design and renders the run table with the assumptions",
        "runs the demo fit round trip and renders the coefficient surface",
        "renders the prediction round trip with the zone vocabulary",
        "renders the contour round trip through the Plotly lane",
        "renders the desirability optimisation round trip",
        "surfaces the backend's own hard 422",
        "gates an invalid factor row client-side",
        "exports the fitted bundle through the joblib artifact lane",
    ):
        assert lane in suite, f"the vitest suite lost its {lane!r} lane"
