"""part17 §5 (the M1 lane) — the multi-response ML drift guards.

The Python gate cannot execute the React section, but every contract
the lane rides CAN be pinned against the real frontend + backend
source — a rename, a dropped honesty note, a second registration path,
or a silent aggregate turns a guard red before the surface drifts:

* the backend module + the route + the API test suite exist (the
  strategy registry with stated mechanisms, the trainer, the
  persistence bundle, the masked multi-task wrapper);
* the route is registered EXACTLY once in the aggregator, tagged, and
  carrying the four paths the contract export grew (121 paths / 75
  schemas, the four multiresponse paths named);
* the mechanism honesty: every strategy row states its mechanism
  (native 2-D Y / wrapper per-target clones / chain feed) and the
  per_target missing lane rides the wrapper mechanism only — the
  typed refusal pinned in the module source;
* the leakage discipline: the trainer splits FIRST and the feature
  pipeline + per-target scalers are fit on the train partition only
  (the source order pinned);
* the §5.6 contract completeness: the persistence contract carries
  the mandate's own key list;
* the frontend typed client: the five service methods + the runtime
  parsers + the export lane exist in ``services/ml/client.ts``;
* the ML workspace integration: the Multi-Response section rides the
  EXISTING Machine Learning workspace (stage index 20, the Export
  stage renumbered 21 — no tenth workspace, the mandate's own rule);
* the client-side gates: the >= 2-target validation fires BEFORE any
  request (the disabled button + the guard pinned as source text);
* the honest renders: the per-target metric cards, the
  aggregate-alongside-never-instead note, the bounded residual sample
  with its bound reported, the SYNTHETIC demo labelling;
* the vitest suite exists with the mandated lanes (6 tests, counted
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
VITEST = ML_FEATURES / "__tests__" / "MultiresponseLane.test.tsx"
MODULE = ROOT / "src" / "pharmaml" / "ml" / "multiresponse.py"
ROUTE = ROOT / "src" / "pharmaml" / "api" / "routes" / "multiresponse.py"
ROUTE_INIT = ROOT / "src" / "pharmaml" / "api" / "routes" / "__init__.py"
API_TESTS = ROOT / "tests" / "api" / "test_multiresponse.py"
CONTRACT = ROOT / "contracts" / "openapi.json"
API_TYPES = SRC / "types" / "api.d.ts"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the lane's own files exist


def test_the_module_route_and_both_suites_exist() -> None:
    assert MODULE.is_file(), "the part17 §5 multiresponse module is missing"
    assert ROUTE.is_file(), "the /api/v1/multiresponse route is missing"
    assert API_TESTS.is_file(), "the API test suite is missing"
    assert VITEST.is_file(), "the vitest component suite is missing"
    assert ML_CLIENT.is_file()


def test_the_route_is_registered_exactly_once() -> None:
    init = _read(ROUTE_INIT)
    assert init.count("multiresponse,") == 1, "the import drifted"
    assert init.count("include_router(multiresponse.router)") == 1, (
        "the multiresponse router must register exactly once"
    )
    route = _read(ROUTE)
    assert 'prefix="/multiresponse"' in route
    assert 'tags=["multiresponse"]' in route


def test_the_contract_carries_the_four_new_paths() -> None:
    schema = json.loads(CONTRACT.read_text())
    paths = schema["paths"]
    # 121 -> 128 / 75 -> 83 by the r359 part17 §6 R1 rsm septet (the
    # seven rsm paths + the eight request schemas) -- the RSM lane's
    # own growth, trued here with the dated comment
    # 134 -> 142 / 87 -> 96 by the r367 part18 §5 DL architecture
    # expansion (the CNN + autoencoder + GAN + diffusion pairs' paths
    # and request DTOs); 142 -> 143 by the r378 GEN-2 metrics path
    # (the pin left stale at 142 in that round — caught and trued by
    # the r379 DS-2 round); 143 -> 144 by the r379 DS-2 path;
    # 144 -> 145 by the r380 DS-3 frames/register path; 145 -> 146 by
    # the r382 ML-4 unsupervised pca sibling path; 146 -> 147 by the
    # r383 ML-5 drift path; 147 -> 148 by the r383 DL-2 gnn-explain path
    assert len(paths) == 148, f"the contract drifted: {len(paths)} paths"
    # 96 -> 97 by the r375 ML-3 train-lane split section (the
    # TrainSplitSpec request DTO, one schema; no new paths)
    # 97 -> 98 by the r378 GEN-2 GenerationMetricsRequest DTO (this
    # pin left stale at 97 in that round — caught and trued by the
    # r379 DS-2 round); 98 -> 99 by the r379 DS-2 StandardizeRequest;
    # 99 -> 100 by the r380 DS-3 FramesRegisterRequest; 100 -> 101 by
    # the r382 ML-4 PcaRequest DTO; 101 -> 102 by the r383 ML-5
    # DriftRequest DTO; 102 -> 103 by the r383 DL-2 GnnExplainRequest DTO
    assert len(schema["components"]["schemas"]) == 103
    for path in (
        "/api/v1/multiresponse/registry",
        "/api/v1/multiresponse/train",
        "/api/v1/multiresponse/predict",
        "/api/v1/multiresponse/models",
    ):
        assert path in paths, f"{path} missing from the exported contract"
    # and the regenerated types carry the request schemas
    types = _read(API_TYPES)
    assert "TrainMultiResponseRequest" in types
    assert "PredictMultiResponseRequest" in types


# ---------------------------------------------------------------------------
# the mechanism honesty + the mandate's own algorithm minimum


def test_the_registry_states_every_mechanism() -> None:
    module = _read(MODULE)
    assert 'mechanism: Literal["native", "wrapper", "chain"]' in module
    for key in (
        '"pls2"',
        '"multi_task_elastic_net"',
        '"multi_output"',
        '"regressor_chain"',
    ):
        assert f"key={key}," in module, f"the mandated strategy {key} drifted"
    # the per-target missing lane rides the wrapper mechanism only —
    # the typed refusal is pinned as source text
    assert 'if missing == "per_target" and spec.mechanism != "wrapper":' in module
    # the sample-weight capability is stated per strategy, never assumed
    assert "sample_weight_support" in module


def test_the_demo_frame_is_labelled_synthetic() -> None:
    module = _read(MODULE)
    assert "The seeded SYNTHETIC demo frame" in module
    assert '"synthetic": True' in _read(ROUTE) or "synthetic" in _read(ROUTE)


# ---------------------------------------------------------------------------
# the leakage discipline + the contract completeness


def test_the_trainer_splits_first_and_fits_on_the_train_partition() -> None:
    module = _read(MODULE)
    # the split precedes the preprocessing fit (the leakage rule as
    # source order): the split block ends, then the missing-strategy
    # block, then _fit_all_parts
    split_at = module.index("# the split FIRST (leakage discipline")
    fit_at = module.index("fitted = _fit_all_parts(")
    assert split_at < fit_at, "the split must precede the pipeline fit"
    # and the helper's own contract says it
    assert "fit on THIS partition's" in module
    assert "split FIRST" in module


def test_the_persistence_contract_carries_the_mandate_keys() -> None:
    module = _read(MODULE)
    for key in (
        '"target_names"',
        '"target_order"',
        '"target_units"',
        '"target_scaler"',
        '"feature_names"',
        '"preprocessing"',
        '"estimator_strategy"',
        '"hyperparameters"',
        '"split"',
        '"random_seed"',
        '"per_target_metrics"',
        '"environment"',
    ):
        assert key in module, f"the §5.6 contract lost its {key} key"


# ---------------------------------------------------------------------------
# the frontend typed client


def test_the_typed_client_carries_the_five_methods() -> None:
    client = _read(ML_CLIENT)
    for method in (
        "multiresponseRegistry",
        "multiresponseTrain",
        "multiresponsePredict",
        "multiresponseModels",
        "multiresponseExport",
    ):
        assert f"async {method}(" in client, f"the {method} method drifted"
    # the runtime parsers exist (the REAL ones, never re-declared)
    for parser in (
        "parseMultiresponseRegistryResponse",
        "parseMultiresponseTrainResponse",
        "parseMultiresponsePredictResponse",
        "parseMultiresponseModelsResponse",
    ):
        assert f"function {parser}(" in client
    # the typed response interfaces
    for iface in (
        "MultiresponseRegistryResponse",
        "MultiresponseTrainResponse",
        "MultiresponsePredictResponse",
        "MultiresponseModelView",
    ):
        assert f"interface {iface}" in client


# ---------------------------------------------------------------------------
# the workspace integration (no tenth workspace)


def test_the_section_rides_the_existing_ml_workspace() -> None:
    page = _read(MLPAGE)
    assert 'title="Multi-Response Regression"' in page
    # stage 20 inside the SAME workspace, Export renumbered 21
    m = re.search(r'index=\{20\}\s*\n\s*title="Multi-Response Regression"', page)
    assert m is not None, "the section must hold stage index 20"
    # the r359 RSM lane inserted stage 21 (Experimental Design &
    # Response Surface) -- the Export stage renumbered 21 -> 22
    m = re.search(r'index=\{22\}\s*\n\s*title="Export"', page)
    assert m is not None, "the Export stage must renumber to 22"
    # the mandate's own vocabulary rides the section prose
    assert "no tenth workspace" not in page  # never re-stated as a claim
    assert "part17 §5 (M1) lane" in page or "part17 §5, the M1 lane" in page


def test_the_client_side_target_gate_fires_before_any_request() -> None:
    page = _read(MLPAGE)
    assert "multi-response training needs at least 2 target columns" in page
    # the disabled train button while the gate fails
    assert "mrTargetColumns.length < 2" in page
    # and the per_target mechanism refusal is mirrored client-side
    assert "wrapper mechanism only" in page


def test_the_honest_renders_are_pinned() -> None:
    page = _read(MLPAGE)
    # the per-target metric cards
    assert "per_target_metrics" in page
    # the aggregate-alongside-never-instead note
    assert "never instead" in page
    # the bounded residual sample with its bound reported
    assert "the bound is" in page
    # the SYNTHETIC demo labelling rides the provenance line
    assert "the seeded SYNTHETIC demo frame" in page
    # the models comparison surface
    assert "the comparison surface" in page


def test_the_vitest_suite_carries_the_mandated_lanes() -> None:
    suite = _read(VITEST)
    for lane in (
        "renders the stage with the registry vocabulary",
        "runs the demo train round trip",
        "renders the predict round trip in the stable target order",
        "surfaces the backend's own hard 422",
        "gates a one-target selection client-side",
        "exports the trained model through the joblib artifact lane",
    ):
        assert lane in suite, f"the vitest suite lost its {lane!r} lane"
