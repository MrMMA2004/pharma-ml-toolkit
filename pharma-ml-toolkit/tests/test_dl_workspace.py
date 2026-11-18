"""part11 §11.3, the Deep Learning workspace row — the drift guards.

The Python gate cannot execute the React page, but every contract the
DL workspace rides CAN be pinned against the backend source that
defines it, so a backend-side rename turns into a red test before the
frontend drifts:

* the dl client's REST surface vs ``pharmaml.api.routes.dl`` plus the
  shared lanes the page rides (``routes/ml.py``'s predict path, the
  artifacts figure/model lanes, the visualization table lane);
* the train request body's field parity — the request object the
  client builds vs the pydantic ``DLTrainRequest`` the route actually
  defines (§12.2's one-DTO-source rule, enforced at the key level);
* the network catalog entry keys vs ``routes/dl.py``'s own
  ``_network_catalog`` output dict;
* the per-epoch ``training.metrics`` payload keys vs the
  ``_training_progress_hook`` metrics dict the Monitor stage's live
  loss/validation curves ride;
* every ``/api/v1`` literal the page + client carry must exist in the
  exported ``contracts/openapi.json`` (the anti-fabrication guard: no
  call to a route the contract never declared);
* the checkpoint lane: the model artifact joblib bundle + the
  digest-verified download labels;
* the figure lane: the loss-curve artifact + its two-epoch minimum;
* the export lane: rows-based ``POST /visualization/table`` with the
  backend's own 5000-row cap mirrored client-side;
* the workspace registry flip (deep-learning ``shell`` →
  ``workflow``) + the App wiring;
* the §11.3 stage order (the page's StageHeader sequence) and the
  honesty blocks (the estimator's internal standardisation note, the
  no-CV note, the single prediction path note, the task-mismatch
  gate) — the §12 labels, pinned.
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
    """The field names of one BaseModel class block."""
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
        "${encodeURIComponent(datasetId)}", "{dataset_id}"
    )
    normalized = normalized.replace(
        "${encodeURIComponent(artifactId)}", "{artifact_id}"
    )
    paths: set[str] = set()
    for match in re.finditer(r"[\"'`](/api/v[0-9]+/[^\"'`?]*)", normalized):
        paths.add(match.group(1))
    return paths


def test_dl_rest_surface_covered() -> None:
    """Every path the dl client carries is a route the backend defines."""
    client = _read(SERVICES / "dl" / "client.ts")
    carried = _frontend_api_paths(client)
    assert carried, "the dl client lost every path literal"
    real: dict[str, set[str]] = {}
    for module in ("dl.py", "artifacts.py", "visualization.py"):
        real.update(_backend_routes(module))
    unknown = carried - set(real)
    assert not unknown, f"dl client calls undeclared routes: {sorted(unknown)}"
    # the workflow's own calls are covered, method-for-method
    for path in ("/api/v1/dl/networks", "/api/v1/dl/train",
                 "/api/v1/artifacts/figure", "/api/v1/artifacts/model"):
        assert path in carried, f"the workflow stopped calling {path}"
        assert path in real, f"the backend stopped declaring {path}"
    # the predict + table lanes the page rides (not the dl client's
    # own contract — the page carries those literals)
    page = _read(FEATURES / "dl" / "DLPage.tsx")
    page_paths = _frontend_api_paths(page)
    assert "/api/v1/visualization/table" in page_paths, (
        "the export stage stopped riding the rows-based table lane"
    )
    ml_client = _read(SERVICES / "ml" / "client.ts")
    assert "/api/v1/ml/predict" in _frontend_api_paths(ml_client), (
        "the single prediction path vanished from the ml client"
    )


def test_dl_page_calls_only_contract_routes() -> None:
    """The anti-fabrication guard: every /api/v1 literal in the page +
    the dl client exists in the exported openapi contract."""
    page = _read(FEATURES / "dl" / "DLPage.tsx")
    client = _read(SERVICES / "dl" / "client.ts")
    carried = _frontend_api_paths(page) | _frontend_api_paths(client)
    assert carried, "the DL workspace lost every path literal"
    contract = json.loads(_read(CONTRACTS / "openapi.json"))
    declared = set(contract["paths"])
    unknown = carried - declared
    assert not unknown, (
        f"the DL workspace calls routes the contract never declared: "
        f"{sorted(unknown)}"
    )


def test_dl_train_request_fields_mirror_the_backend() -> None:
    client = _read(SERVICES / "dl" / "client.ts")
    backend_fields = _pydantic_fields(_route_module("dl.py"), "DLTrainRequest")
    assert backend_fields >= {
        "dataset_id", "network", "hyperparams", "mode",
    }, "the backend DLTrainRequest changed shape (update this guard + the page)"
    assert _client_request_keys(client, "DLTrainRequest") == backend_fields


def test_network_catalog_fields_mirror_the_backend() -> None:
    """The client's catalog entry keys are exactly _network_catalog's
    own output dict — the live signature defaults, never re-declared."""
    client = _read(SERVICES / "dl" / "client.ts")
    route_src = _route_module("dl.py")
    block = re.search(
        r"out\.append\(\s*\{(.*?)\}\s*\)", route_src, re.DOTALL
    )
    assert block is not None, "the _network_catalog append block vanished"
    emitted = set(re.findall(r'"([a-z_]+)":', block.group(1)))
    assert emitted == {
        "network", "class", "module", "task", "engine", "defaults",
        "train_url",
    }, f"the catalog dict changed shape: {sorted(emitted)}"
    for key in emitted:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"the catalog parser dropped the '{key}' key the route emits"
        )
    # the hyperparameter vocabulary mirror is the route's own spec
    spec = re.search(r"_HYPERPARAM_SPEC\s*=\s*\{(.*?)\}", route_src, re.DOTALL)
    assert spec is not None, "the _HYPERPARAM_SPEC block vanished"
    backend_keys = set(
        re.findall(r'"(\w+)":\s*(?:list|int|float|str|bool)', spec.group(1))
    )
    mirror = re.search(
        r"DL_HYPERPARAM_KEYS:\s*readonly string\[\]\s*=\s*\[(.*?)\]\s*as const",
        client, re.DOTALL,
    )
    assert mirror is not None, "the DL_HYPERPARAM_KEYS mirror vanished"
    mirrored = set(re.findall(r'"(\w+)"', mirror.group(1)))
    assert mirrored == backend_keys, (
        f"the mirrored hyperparameter vocabulary drifted "
        f"(client {sorted(mirrored)} vs backend {sorted(backend_keys)})"
    )
    # the activation tokens mirror the estimator's own registry
    neural = _read(ROOT / "src" / "pharmaml" / "ml" / "neural.py")
    registry = re.search(r"_ACTIVATIONS:.*?=\s*\((.*?)\)", neural, re.DOTALL)
    assert registry is not None, "neural.py lost its _ACTIVATIONS registry"
    backend_tokens = set(re.findall(r'"(\w+)"', registry.group(1)))
    token_mirror = re.search(
        r"DL_ACTIVATIONS:\s*readonly string\[\]\s*=\s*\[(.*?)\]\s*as const",
        client, re.DOTALL,
    )
    assert token_mirror is not None, "the DL_ACTIVATIONS mirror vanished"
    mirrored_tokens = set(re.findall(r'"(\w+)"', token_mirror.group(1)))
    assert mirrored_tokens == backend_tokens, (
        f"the mirrored activation vocabulary drifted "
        f"(client {sorted(mirrored_tokens)} vs "
        f"neural.py {sorted(backend_tokens)})"
    )


def test_training_metrics_fields_mirror_the_backend() -> None:
    """The Monitor stage's telemetry parser keys are exactly the
    metrics dict _training_progress_hook publishes per epoch."""
    client = _read(SERVICES / "dl" / "client.ts")
    route_src = _route_module("dl.py")
    block = re.search(
        r"def _training_progress_hook.*?metrics\s*=\s*\{(.*?)\}",
        route_src, re.DOTALL,
    )
    assert block is not None, "the _training_progress_hook metrics dict vanished"
    emitted = set(re.findall(r'"([a-z_]+)":', block.group(1)))
    assert emitted == {
        "epoch", "epochs", "train_loss", "val_loss", "best_val", "stale",
    }, f"the per-epoch metrics dict changed shape: {sorted(emitted)}"
    for key in emitted:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"parseTrainingMetricsPayload dropped the '{key}' telemetry key"
        )
    # the vocabulary rides the drift-guarded event bus
    events = _read(SERVICES / "websocket" / "events.ts")
    assert '"training.metrics"' in events and '"training.progress"' in events, (
        "the WS event vocabulary lost the training telemetry events"
    )


def test_checkpoint_rides_the_model_artifact_lane() -> None:
    """The Save Checkpoint stage: POST /artifacts/model (the joblib
    bundle + the metadata sidecar) + the digest-verified labels."""
    client = _read(SERVICES / "dl" / "client.ts")
    page = _read(FEATURES / "dl" / "DLPage.tsx")
    assert "saveCheckpoint" in client, "the checkpoint lane vanished from the client"
    assert re.search(r"/api/v1/artifacts/model", client) is not None, (
        "the checkpoint stopped riding the model artifact lane"
    )
    assert re.search(r"/api/v1/artifacts/model", page) is not None or (
        "saveCheckpoint" in page
    ), "the page lost the checkpoint stage's lane note"
    assert "digest_verified" in page, "the checkpoint lost its digest labels"
    # the persistence engine the lane rides is the backend's own
    persistence = _read(ROOT / "src" / "pharmaml" / "ml" / "persistence.py")
    assert "def save_model" in persistence, (
        "the backend's save_model engine vanished — the lane has no engine"
    )
    artifacts_route = _route_module("artifacts.py")
    assert "save_model" in artifacts_route, (
        "the artifacts route stopped riding the persistence engine"
    )


def test_figure_visualization_rides_the_artifact_lane() -> None:
    """The Visualize stage: the loss-curve figure artifact (the REAL
    §17 export stack) with its two-epoch minimum."""
    client = _read(SERVICES / "dl" / "client.ts")
    assert "lossCurveFigure" in client, "the figure lane vanished from the client"
    assert re.search(r"/api/v1/artifacts/figure", client) is not None, (
        "the loss curve stopped riding the figure artifact lane"
    )
    # a curve needs at least two epochs — the honest minimum
    assert re.search(r"at least two epochs", client) is not None, (
        "the figure lane lost its two-epoch minimum"
    )
    # the figure request body rides the generated contract types
    assert "FigureRequest" in client and (
        "interface FigureRequest" not in client
    ), "the figure request stopped using the generated contract type"


def test_export_rides_the_rows_table_lane() -> None:
    """The export stage: rows-based /visualization/table + the backend's
    own 5000-row cap mirrored client-side + the digest-verified labels."""
    page = _read(FEATURES / "dl" / "DLPage.tsx")
    assert "VisualizationTableRequest" in page, (
        "the export stopped using the generated table request type"
    )
    cap = re.search(r"TABLE_EXPORT_MAX_ROWS\s*=\s*(\d+)", page)
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
    """deep-learning: shell → workflow, and the App renders the page."""
    workspaces = _read(APP / "workspaces.ts")
    match = re.search(
        r'id:\s*"deep-learning",.*?state:\s*"(\w+)"', workspaces,
        re.DOTALL,
    )
    assert match is not None, "the deep-learning entry vanished"
    assert match.group(1) == "workflow", (
        "the DL workspace is still a shell after the §11.3 build"
    )
    app = _read(APP / "App.tsx")
    assert 'DLPage' in app and 'from "@/features/dl/DLPage"' in app, (
        "the App stopped importing the DL page"
    )
    assert re.search(r'entry\.id === "deep-learning"', app) is not None, (
        "the App lost the deep-learning render branch"
    )


def test_stage_order_matches_the_spec_chain() -> None:
    """The page's StageHeader sequence = §11.3's fixed workflow order."""
    page = _read(FEATURES / "dl" / "DLPage.tsx")
    titles = re.findall(r'<StageHeader\b[^>]*?title="([^"]+)"', page, re.DOTALL)
    expected = [
        "Select Dataset",
        "Validate",
        "Preprocess",
        "Select Architecture",
        "Configure Model",
        "Configure Training",
        "Train",
        "Monitor",
        "Evaluate — Inspect Predictions",
        "Visualize",
        "Save Checkpoint / Export",
    ]
    assert titles == expected, (
        "the §11.3 stage order drifted "
        f"(got {titles}, expected {expected})"
    )


def test_honesty_blocks_present() -> None:
    """§12's labels: the estimator's internal standardisation note (no
    preprocess toggle), the no-CV note, the single prediction path
    note, the task-mismatch gate."""
    page = _read(FEATURES / "dl" / "DLPage.tsx")
    assert "no preprocess toggle" in page, (
        "the Preprocess stage lost its honest internal-standardisation note"
    )
    assert re.search(r"no CV scores|carries none — honest, not faked", page), (
        "the page lost its honest no-CV note"
    )
    assert re.search(r"SINGLE session prediction path", page) is not None, (
        "the Evaluate stage lost its single-prediction-path note"
    )
    assert re.search(r"task mismatch", page) is not None, (
        "the architecture catalog lost its task-mismatch gate"
    )
    # the backend's own refusal reasons ride the catalog rows
    route_src = _route_module("dl.py")
    assert "422" in route_src or "task" in route_src, (
        "the backend stopped refusing task mismatches"
    )


def test_monitor_rides_the_real_event_stream() -> None:
    """The Monitor stage subscribes to the live WS bus (exactly the
    envelope vocabulary the backend's per-epoch hook publishes) and
    parses each payload at runtime."""
    page = _read(FEATURES / "dl" / "DLPage.tsx")
    assert "getEventStream()" in page, (
        "the Monitor stage stopped subscribing to the event stream"
    )
    assert 'envelope.event_type === "training.metrics"' in page, (
        "the Monitor lost its training.metrics filter"
    )
    assert "parseTrainingMetricsPayload" in page, (
        "the Monitor stopped parsing the telemetry at runtime"
    )
    # the job-scoped filter: only THIS job's envelopes accumulate
    assert re.search(r"envelope\.job_id !== trainJobId", page) is not None, (
        "the Monitor lost its job-scoped envelope filter"
    )
