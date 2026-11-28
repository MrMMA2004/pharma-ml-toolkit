"""part16 §15.2 (Phase 12) — the experiment-tracking drift guards.

The Python gate cannot execute the React lineage card, but every
contract the lane rides CAN be pinned against the real frontend +
backend source — a rename, a dropped §15.2 field, a renumbered stage,
or a second monitoring-style parallel architecture turns a guard red
before the surface drifts:

* the service pair (client/types) + the component + the two suites
  exist under ``services/runs`` / ``components/runs``;
* the integration rule: the client's ONLY data sources are
  ``GET /api/v1/runs`` and ``GET /api/v1/runs/{run_id}`` (the lane
  composes with the training points — no parallel tracking
  architecture, nothing hand-POSTed);
* the §15.2 vocabulary is mirrored field-for-field in the frontend
  types (the checklist the directive names);
* the status vocabulary is the same three states on both sides;
* the schema pin ``pharmaml-experiment-run/1`` stands in both
  sources;
* the ML workspace mounts the lineage card inside the Compare stage
  without renumbering any stage;
* the generated types were regenerated over the 114-path contract;
* the backend records at BOTH training exits (the completed run and
  the failed/cancelled run) — the honest-lane rule;
* the durable migration stands (MIGRATIONS v3, the experiment_runs
  table + its recovery block).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
RUNS = SRC / "services" / "runs"
COMPONENTS = SRC / "components" / "runs"
API = ROOT / "src" / "pharmaml" / "api"
ROUTES = API / "routes"
ML_PAGE = SRC / "features" / "ml" / "MLPage.tsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the service pair + the component + the suites


def test_the_runs_service_pair_and_component_exist() -> None:
    for name in ("client.ts", "types.ts"):
        assert (RUNS / name).is_file(), f"the runs service lost {name}"
    assert (RUNS / "__tests__" / "client.test.ts").is_file()
    assert (COMPONENTS / "RunLineage.tsx").is_file()
    assert (COMPONENTS / "__tests__" / "RunLineage.test.tsx").is_file()


def test_the_client_pins_the_two_integration_points() -> None:
    client = _read(RUNS / "client.ts")
    assert '"/api/v1/runs"' in client, (
        "the lineage table must consume the backend's GET /api/v1/runs — "
        "a second listing source would be the parallel architecture §8 forbids"
    )
    assert "/api/v1/runs/${" in client.replace("runId", "runId"), (
        "the detail fetch must target the backend's run_id path"
    )
    assert "parseRunSummary" in client or "parseExperimentRun" in client, (
        "the raw answers must be narrowed, never any"
    )
    # the backend surface stands (the consumed contract)
    runs_route = _read(ROUTES / "runs.py")
    assert 'APIRouter(prefix="/runs"' in runs_route
    assert '@router.get("")' in runs_route
    assert '@router.get("/{run_id}")' in runs_route


def test_the_detail_path_is_properly_encoded() -> None:
    client = _read(RUNS / "client.ts")
    assert "encodeURIComponent" in client, (
        "run ids travel through encodeURIComponent per the house rule"
    )


# ---------------------------------------------------------------------------
# the §15.2 vocabulary mirror


def test_the_frontend_types_mirror_the_1512_checklist() -> None:
    types = _read(RUNS / "types.ts")
    for field in (
        "dataset",
        "feature_configuration",
        "preprocessing",
        "algorithm",
        "hyperparameters",
        "random_seed",
        "cross_validation",
        "evaluation",
        "model_artifact",
        "environment",
        "logs",
    ):
        assert field in types, (
            f"§15.2 names '{field}' — the frontend record must mirror it"
        )
    # the §15.2 sub-fields that make the lineage traceable
    for sub in (
        "source_filename",  # dataset identifiers
        "toolkit_version",  # algorithm and version
        "best_params",  # hyperparameters (tuned)
        "requested",  # hyperparameters (as-asked)
        "cv_folds",  # cross-validation configuration
        "cv_mean",  # evaluation metrics
        "model_id",  # model artifact references
        "started_at",  # execution timestamps
        "completed_at",
    ):
        assert sub in types, f"§15.2 lineage lost the '{sub}' field"


def test_the_status_vocabulary_is_the_same_three_states() -> None:
    types = _read(RUNS / "types.ts")
    for state in ("completed", "failed", "cancelled"):
        assert state in types, f"the frontend lost the '{state}' status"
    runs_route = _read(ROUTES / "runs.py")
    for state in ("completed", "failed", "cancelled"):
        assert state in runs_route, (
            f"the backend route lost the '{state}' status validation"
        )


def test_the_schema_pin_stands_on_both_sides() -> None:
    assert "pharmaml-experiment-run/1" in _read(RUNS / "types.ts")
    assert "pharmaml-experiment-run/1" in _read(API / "runs.py")


def test_the_backend_record_carries_the_checklist() -> None:
    runs = _read(API / "runs.py")
    for field in (
        '"dataset"',
        '"feature_configuration"',
        '"preprocessing"',
        '"algorithm"',
        '"hyperparameters"',
        '"random_seed"',
        '"cross_validation"',
        '"evaluation"',
        '"model_artifact"',
        '"environment"',
        '"logs"',
    ):
        assert field in runs, f"the backend §15.2 record lost {field}"


# ---------------------------------------------------------------------------
# the honest-lane rules


def test_the_failed_and_cancelled_exits_record_their_runs() -> None:
    ml = _read(ROUTES / "ml.py")
    assert "build_ml_train_run(" in ml, "the completed exit must record"
    assert "build_ml_train_failed_run(" in ml, (
        "the failed/cancelled exit must record — a doomed configuration "
        "is scientific information, never silently dropped"
    )
    assert '"cancelled"' in ml and '"failed"' in ml


def test_the_ml_route_passes_the_run_store_in_both_modes() -> None:
    ml = _read(ROUTES / "ml.py")
    assert ml.count("run_store=request.app.state.run_store") == 2, (
        "both the sync and the job lane must carry the run store"
    )


def test_the_store_is_wired_into_the_app_factory() -> None:
    init = _read(API / "__init__.py")
    assert "RunStore(persistence=persistence)" in init, (
        "the run store must ride the app.state singleton set"
    )
    assert "app.state.run_store = " in init


# ---------------------------------------------------------------------------
# the durable lane


def test_the_v3_migration_stands() -> None:
    persistence = _read(API / "persistence.py")
    assert "CREATE TABLE experiment_runs" in persistence
    assert "(3, _V3_SCHEMA)" in persistence, (
        "the migration ledger must carry the runs table as v3"
    )
    assert "def record_run" in persistence
    assert "def list_runs" in persistence
    assert "def load_run" in persistence
    assert "from .runs import ExperimentRunRecord" in persistence, (
        "the recovery rehydrates through the record's own constructor"
    )


# ---------------------------------------------------------------------------
# the ML workspace mount


def test_the_lineage_card_mounts_inside_the_compare_stage() -> None:
    page = _read(ML_PAGE)
    assert "<RunLineage" in page, "the lineage card is not mounted"
    assert 'from "@/components/runs/RunLineage"' in page or (
        'from "@/components/runs/RunLineage.jsx"' in page
    )
    # no stage renumbering: the Compare stage keeps its index 9 title
    compare = re.search(
        r'index=\{9\}\s*\n\s*title="Compare"', page
    )
    assert compare is not None, "the Compare stage lost its index-9 pin"


def test_the_generated_types_carry_the_runs_paths() -> None:
    generated = _read(FRONTEND / "src" / "types" / "api.d.ts")
    assert '"/api/v1/runs"' in generated
    assert '"/api/v1/runs/{run_id}"' in generated


def test_the_contract_carries_the_two_new_paths() -> None:
    import json

    contract = json.loads(
        (ROOT / "contracts" / "openapi.json").read_text(encoding="utf-8")
    )
    paths = contract["paths"]
    assert "/api/v1/runs" in paths
    assert "/api/v1/runs/{run_id}" in paths
    # 121 -> 128 by the r359 part17 §6 R1 rsm septet (the seven rsm
    # paths), trued with the dated comment; 128 -> 134 by the r365
    # part18 lanes (the unsupervised trio + the RL trio); 134 -> 142
    # by the r367 part18 §5 DL architecture expansion (the CNN pair,
    # the autoencoder pair, the GAN pair, the diffusion pair);
    # 142 -> 143 by the r378 GEN-2 metrics path (the pin left stale
    # at 142 in that round — caught and trued by the r379 DS-2 round);
    # 143 -> 144 by the r379 DS-2 standardization path; 144 -> 145 by
    # the r380 DS-3 frames/register path; 145 -> 146 by the r382 ML-4
    # unsupervised pca sibling path; 146 -> 147 by the r383 ML-5
    # drift path; 147 -> 148 by the r383 DL-2 gnn-explain path
    assert len(paths) == 148, (
        "the contract must pin 148 paths (112 + the runs pair + the r351 "
        "quality path + the r352 diagnostics pair + the r358 part17 §5 "
        "multiresponse quartet + the r359 part17 §6 rsm septet + the "
        "r365 part18 unsupervised trio + the r365 part18 RL trio + the "
        "r367 part18 DL architecture octet + the r382 ML-4 pca sibling + "
        "the r383 ML-5 drift path + the r383 DL-2 gnn-explain path)"
    )
