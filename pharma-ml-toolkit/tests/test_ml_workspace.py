"""part11 §11.2, the ML workspace row — the drift guards.

The Python gate cannot execute the React page, but every contract the
ML workspace rides CAN be pinned against the backend source that
defines it, so a backend-side rename turns into a red test before the
frontend drifts:

* the ml client's REST surface vs ``pharmaml.api.routes.ml`` /
  ``routes/applicability_domain`` / ``routes/semi_supervised`` /
  ``routes/admet`` / ``routes/dta``;
* the request bodies' field parity — the four request objects the
  client builds vs the pydantic request models those routes actually
  define (§12.2's one-DTO-source rule, enforced at the key level);
* every ``/api/v1`` literal the page + client carry must exist in the
  exported ``contracts/openapi.json`` (the anti-fabrication guard: no
  call to a route the contract never declared);
* the catalog entry keys vs ``pharmaml.ml.models.list_models``'s own
  documented output;
* the export lane: rows-based ``POST /visualization/table`` with the
  backend's own 5000-row cap mirrored client-side + the
  digest-verified download labels;
* the workspace registry state (machine-learning ``workflow``) + the
  App wiring;
* the §11.2 stage order (the page's StageHeader sequence) and the
  honest capability boundaries (real SHAP explainability, the
  classification-only gate, the bounded column bridge) — the §12
  labels, pinned.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SERVICES = FRONTEND / "src" / "services"
FEATURES = FRONTEND / "src" / "features"
APP = FRONTEND / "src" / "app"
API = ROOT / "src" / "pharmaml" / "api"
CONTRACTS = ROOT / "contracts"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _route_module(name: str) -> str:
    return _read(API / "routes" / name)


def _api_prefix() -> str:
    src = _route_module("__init__.py")
    match = re.search(r'api_router\s*=\s*APIRouter\(prefix="(/api/v[0-9]+)"', src)
    assert match is not None, "the api_router prefix literal vanished"
    return match.group(1)


def _router_prefix(module_src: str) -> str:
    match = re.search(r'APIRouter\(\s*prefix="([^"]+)"', module_src, re.DOTALL)
    assert match is not None, "a route module lost its APIRouter prefix"
    return match.group(1)


def _backend_routes(module_name: str) -> dict[str, set[str]]:
    """``{route_path: {methods}}`` for one routes module."""
    src = _route_module(module_name)
    prefix = _api_prefix() + _router_prefix(src)
    routes: dict[str, set[str]] = {}
    for match in re.finditer(
        r'@router\.(get|post|put|delete|head|patch)\(\s*"([^"]*)"', src
    ):
        method, path = match.group(1).upper(), match.group(2)
        full = prefix + path
        routes.setdefault(full, set()).add(method)
    return routes


def _pydantic_fields(module_src: str, class_name: str) -> set[str]:
    """The field names of one BaseModel class block (comments and the
    multi-line Field(...) defaults are tolerated; method ``def`` lines
    never match a ``name:`` line start)."""
    start = re.search(
        r"^class " + re.escape(class_name) + r"\(BaseModel\):",
        module_src,
        re.MULTILINE,
    )
    assert start is not None, f"{class_name} vanished from the backend source"
    rest = module_src[start.end() :]
    end = re.search(r"^(?:class |@router\.|def )", rest, re.MULTILINE)
    body = rest[: end.start()] if end else rest
    return set(re.findall(r"^\s+(\w+)\s*:", body, re.MULTILINE))


def _client_request_keys(client_src: str, type_name: str) -> set[str]:
    """The keys of the ``request: <Type> = { ... }`` block in the client."""
    pattern = re.compile(
        r"request:\s*" + re.escape(type_name) + r"\s*=\s*\{(.*?)\n\s*\};",
        re.DOTALL,
    )
    match = pattern.search(client_src)
    assert match is not None, f"the {type_name} request block vanished"
    return set(re.findall(r"^\s{6}(\w+):", match.group(1), re.MULTILINE))


def _frontend_api_paths(source: str) -> set[str]:
    """Normalize the literals (query suffixes stripped, interpolations
    become ``{param}``) to the backend's route-path shapes."""
    normalized = source.replace(
        "${encodeURIComponent(modelId)}", "{model_id}"
    )
    normalized = normalized.replace(
        "${encodeURIComponent(datasetId)}", "{dataset_id}"
    )
    paths: set[str] = set()
    for match in re.finditer(r"[\"'`](/api/v[0-9]+/[^\"'`?]*)", normalized):
        paths.add(match.group(1))
    return paths


def test_ml_rest_surface_covered() -> None:
    """Every path the ml client carries is a route the backend defines."""
    client = _read(SERVICES / "ml" / "client.ts")
    carried = _frontend_api_paths(client)
    assert carried, "the ml client lost every path literal"
    real: dict[str, set[str]] = {}
    for module in ("ml.py", "applicability_domain.py", "semi_supervised.py", "admet.py", "dta.py", "repurposing.py", "drugresponse.py", "pharmacogenomics.py"):
        real.update(_backend_routes(module))
    unknown = carried - set(real)
    assert not unknown, f"ml client calls undeclared routes: {sorted(unknown)}"
    # the workflow's own calls are covered, method-for-method
    for path in ("/api/v1/ml/models", "/api/v1/ml/trained", "/api/v1/ml/train",
                 "/api/v1/ml/predict", "/api/v1/applicability-domain/analyze",
                 "/api/v1/semi-supervised/train"):
        assert path in carried, f"the workflow stopped calling {path}"
        assert path in real, f"the backend stopped declaring {path}"


def test_ml_page_calls_only_contract_routes() -> None:
    """The anti-fabrication guard: every /api/v1 literal in the page +
    the client exists in the exported openapi contract."""
    page = _read(FEATURES / "ml" / "MLPage.tsx")
    client = _read(SERVICES / "ml" / "client.ts")
    carried = _frontend_api_paths(page) | _frontend_api_paths(client)
    assert carried, "the ML workspace lost every path literal"
    contract = json.loads(_read(CONTRACTS / "openapi.json"))
    declared = set(contract["paths"])
    unknown = carried - declared
    assert not unknown, (
        f"the ML workspace calls routes the contract never declared: "
        f"{sorted(unknown)}"
    )


def test_train_request_fields_mirror_the_backend() -> None:
    client = _read(SERVICES / "ml" / "client.ts")
    backend_fields = _pydantic_fields(_route_module("ml.py"), "TrainRequest")
    assert backend_fields >= {
        "dataset_id", "model", "task", "tune", "cv", "n_jobs",
        "preprocess", "random_state", "model_params", "mode",
    }, "the backend TrainRequest changed shape (update this guard + the page)"
    assert _client_request_keys(client, "TrainRequest") == backend_fields


def test_predict_request_fields_mirror_the_backend() -> None:
    client = _read(SERVICES / "ml" / "client.ts")
    backend_fields = _pydantic_fields(_route_module("ml.py"), "PredictRequest")
    # the workflow's dataset-mode lane: rows stays the backend's own
    # inline path, never sent from this page
    assert backend_fields >= {
        "model_id", "rows", "dataset_id", "predict_proba", "mode",
    }
    assert _client_request_keys(client, "PredictRequest") == {
        "model_id", "dataset_id", "predict_proba", "mode",
    }


def test_ad_analyze_fields_mirror_the_backend() -> None:
    client = _read(SERVICES / "ml" / "client.ts")
    backend_fields = _pydantic_fields(
        _route_module("applicability_domain.py"), "ADAnalyzeRequest"
    )
    assert backend_fields >= {
        "model_id", "dataset_id", "x_test", "k", "percentile", "n_models",
        "uncertainty", "random_state", "figure", "mode",
    }
    assert _client_request_keys(client, "ADAnalyzeRequest") == backend_fields


def test_semi_supervised_fields_mirror_the_backend() -> None:
    client = _read(SERVICES / "ml" / "client.ts")
    backend_fields = _pydantic_fields(
        _route_module("semi_supervised.py"), "SemiSupervisedTrainRequest"
    )
    assert backend_fields >= {
        "dataset_id", "method", "unlabeled_marker", "mask_fraction",
        "random_state", "base_estimator", "threshold", "k_best", "max_iter",
        "kernel", "n_neighbors", "gamma", "mode",
    }
    assert _client_request_keys(client, "SemiSupervisedTrainRequest") == (
        backend_fields
    )


def test_catalog_fields_mirror_list_models() -> None:
    """The client's catalog entry keys are exactly list_models' own."""
    client = _read(SERVICES / "ml" / "client.ts")
    models_src = _read(ROOT / "src" / "pharmaml" / "ml" / "models.py")
    doc = re.search(
        r"Returns a list of dicts with keys (.{0,160}?)--", models_src, re.DOTALL
    )
    assert doc is not None, "list_models' documented output keys vanished"
    documented = set(re.findall(r"``([a-z_]+)``", doc.group(1)))
    assert documented == {
        "name", "task", "available", "requires", "notes"
    }
    for key in documented:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"the catalog parser dropped the '{key}' key list_models emits"
        )


def test_export_rides_the_rows_table_lane() -> None:
    """The export stage: rows-based /visualization/table + the backend's
    own 5000-row cap mirrored client-side + the digest-verified labels."""
    page = _read(FEATURES / "ml" / "MLPage.tsx")
    assert "VisualizationTableRequest" in page, (
        "the export stopped using the generated table request type"
    )
    cap = re.search(
        r"TABLE_EXPORT_MAX_ROWS\s*=\s*(\d+)", page
    )
    assert cap is not None and cap.group(1) == "5000", (
        "the client-side row cap drifted from visualization.py's _MAX_ROWS"
    )
    viz_src = _route_module("visualization.py")
    backend_cap = re.search(r"_MAX_ROWS\s*=\s*(\d+)", viz_src)
    assert backend_cap is not None, "visualization.py lost its _MAX_ROWS pin"
    assert int(cap.group(1)) == int(backend_cap.group(1)), (
        "the mirrored cap and the backend cap disagree"
    )
    assert "digest_verified" in page, "the export lost its digest labels"
    assert re.search(r"/api/v1/visualization/table", page) is not None, (
        "the export stopped riding the rows-based table lane"
    )


def test_workspace_flipped_to_workflow() -> None:
    """machine-learning: shell → workflow, and the App renders the page."""
    workspaces = _read(APP / "workspaces.ts")
    match = re.search(
        r'id:\s*"machine-learning",.*?state:\s*"(\w+)"', workspaces,
        re.DOTALL,
    )
    assert match is not None, "the machine-learning entry vanished"
    assert match.group(1) == "workflow", (
        "the ML workspace is still a shell after the §11.2 build"
    )
    app = _read(APP / "App.tsx")
    assert 'MLPage' in app and 'from "@/features/ml/MLPage"' in app, (
        "the App stopped importing the ML page"
    )
    assert re.search(
        r'entry\.id === "machine-learning"', app
    ) is not None, "the App lost the machine-learning render branch"


def test_stage_order_matches_the_spec_chain() -> None:
    """The page's StageHeader sequence = §11.2's fixed workflow order."""
    page = _read(FEATURES / "ml" / "MLPage.tsx")
    titles = re.findall(r'<StageHeader\b[^>]*?title="([^"]+)"', page, re.DOTALL)
    expected = [
        "Select Dataset",
        "Target & Features",
        "Preprocess",
        "Configure Split / CV",
        "Select Algorithm",
        "Configure Hyperparameters",
        "Train",
        "Evaluate — Inspect Predictions",
        "Compare",
        "Applicability Domain",
        "ADMET Profile",
        "Drug–Target Interaction",
        "Drug Repurposing",
        "Cancer Cell-Line Drug Response",
        "Pharmacogenomics",
        "Semi-supervised Learning",
        "Explainability & Uncertainty",
        "Export",
    ]
    assert titles == expected, (
        "the §11.2 stage order drifted "
        f"(got {titles}, expected {expected})"
    )


def test_honesty_blocks_present() -> None:
    """§12's labels: the real-SHAP explainability note, the
    classification gate, the bounded column bridge note. (r295: r290
    closed the §12.2 queue end-to-end — POST /api/v1/ml/explain + the
    GUI stage-12 card — and retired the queued placeholder this test
    still pinned; the pin now guards the CURRENT honest state, the
    real-lane hint, instead of the retired one.)"""
    page = _read(FEATURES / "ml" / "MLPage.tsx")
    assert re.search(r"the real SHAP lane", page) is not None, (
        "the explainability card lost its honest real-SHAP-lane note"
    )
    assert re.search(r"classification-only", page) is not None, (
        "the semi-supervised card lost its classification-only gate note"
    )
    assert re.search(r"bounded column bridge", page) is not None, (
        "the AD stage lost its bounded column bridge note"
    )
    # the bounded bridge rides the datasets client's own 500 cap
    chemistry = _read(SERVICES / "chemistry" / "client.ts")
    assert re.search(r"CHEMISTRY_MAX_SMILES\s*=\s*500", chemistry) is not None
