"""part11 §11.4, the Generative AI workspace row — the drift guards.

The Python gate cannot execute the React page, but every contract the
Generative workspace rides CAN be pinned against the backend source
that defines it, so a backend-side rename turns into a red test before
the frontend drifts:

* the generative client's REST surface vs
  ``pharmaml.api.routes.generative`` plus the shared lanes the page
  rides (the chemistry descriptors engine, the artifacts figure lane,
  the visualization table lane);
* the three request bodies' field parity — the request objects the
  client builds vs the pydantic ``PriorTrainRequest`` /
  ``SampleRequest`` / ``OptimizeRequest`` the route actually defines
  (§12.2's one-DTO-source rule, enforced at the key level);
* the objective catalog: the mirrored kind vocabulary vs the route's
  own ``_OBJECTIVE_CATALOG`` rows, and the parser's keys vs the row
  dict the route emits;
* the prior view keys vs ``PriorRecord.view()``'s own output dict;
* the ``training.metrics`` payload keys vs the publish dict the prior
  lane sends when training ends, and the ``iteration.update`` payload
  keys vs ``JobManager.report_progress``'s own envelope (the per-
  generation telemetry the Generate stage accumulates);
* the optimize result keys vs ``_optimize_body``'s own answer (and the
  ranked ``top`` row keys);
* every ``/api/v1`` literal the page + client carry must exist in the
  exported ``contracts/openapi.json`` (the anti-fabrication guard: no
  call to a route the contract never declared);
* the figure lanes: the prior loss curve + the ranked score profile
  through ``POST /artifacts/figure`` with their honest minimums;
* the export lane: rows-based ``POST /visualization/table`` with the
  backend's own 5000-row cap mirrored client-side;
* the workspace registry flip (generative-ai ``shell`` →
  ``workflow``) + the App wiring;
* the §11.4 stage order (the page's StageHeader sequence) and the
  honesty blocks (the sync sample lane, the ≥ 5 corpus rule, the
  validity-verdict note, the native-cancel note) — the §12 labels,
  pinned.
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
        "${encodeURIComponent(priorId)}", "{prior_id}"
    )
    paths: set[str] = set()
    for match in re.finditer(r"[\"'`](/api/v[0-9]+/[^\"'`?]*)", normalized):
        paths.add(match.group(1))
    return paths


def test_generative_rest_surface_covered() -> None:
    """Every path the generative client carries is a route the backend
    defines; the workflow's own calls are covered method-for-method."""
    client = _read(SERVICES / "generative" / "client.ts")
    carried = _frontend_api_paths(client)
    assert carried, "the generative client lost every path literal"
    real: dict[str, set[str]] = {}
    for module in ("generative.py", "artifacts.py", "visualization.py"):
        real.update(_backend_routes(module))
    unknown = carried - set(real)
    assert not unknown, (
        f"generative client calls undeclared routes: {sorted(unknown)}"
    )
    for path in (
        "/api/v1/generative/objectives",
        "/api/v1/generative/priors",
        "/api/v1/generative/priors/{prior_id}",
        "/api/v1/generative/rnn/train",
        "/api/v1/generative/rnn/sample",
        "/api/v1/generative/optimize",
        "/api/v1/artifacts/figure",
        "/api/v1/visualization/table",
    ):
        assert path in carried, f"the workflow stopped calling {path}"
        assert path in real, f"the backend stopped declaring {path}"
    # the Validate/Properties stage rides the chemistry engine's own
    # client — never re-declared in the generative client
    chemistry = _read(SERVICES / "chemistry" / "client.ts")
    assert "/api/v1/chemistry/descriptors" in _frontend_api_paths(chemistry), (
        "the chemistry descriptors lane vanished from the chemistry client"
    )
    assert "/api/v1/chemistry/descriptors" not in _frontend_api_paths(client), (
        "the generative client re-declared the chemistry lane "
        "(a duplicate contract, not its own surface)"
    )


def test_generative_page_calls_only_contract_routes() -> None:
    """The anti-fabrication guard: every /api/v1 literal in the page +
    the generative client exists in the exported openapi contract."""
    page = _read(FEATURES / "generative" / "GenerativePage.tsx")
    client = _read(SERVICES / "generative" / "client.ts")
    carried = _frontend_api_paths(page) | _frontend_api_paths(client)
    assert carried, "the Generative workspace lost every path literal"
    contract = json.loads(_read(CONTRACTS / "openapi.json"))
    declared = set(contract["paths"])
    unknown = carried - declared
    assert not unknown, (
        f"the Generative workspace calls routes the contract never "
        f"declared: {sorted(unknown)}"
    )


def test_prior_train_request_fields_mirror_the_backend() -> None:
    client = _read(SERVICES / "generative" / "client.ts")
    backend_fields = _pydantic_fields(
        _route_module("generative.py"), "PriorTrainRequest"
    )
    assert backend_fields >= {
        "smiles", "dataset_id", "column", "name", "embedding_dim",
        "hidden_dim", "n_layers", "epochs", "batch_size", "lr",
        "grad_clip", "seed", "mode",
    }, "the backend PriorTrainRequest changed shape (update this guard + the page)"
    assert _client_request_keys(client, "PriorTrainRequest") == backend_fields


def test_sample_and_optimize_request_fields_mirror_the_backend() -> None:
    client = _read(SERVICES / "generative" / "client.ts")
    route_src = _route_module("generative.py")
    sample_fields = _pydantic_fields(route_src, "SampleRequest")
    assert sample_fields >= {
        "prior_id", "n", "temperature", "top_k", "seed", "max_length",
    }, "the backend SampleRequest changed shape (update this guard + the page)"
    assert _client_request_keys(client, "SampleRequest") == sample_fields
    optimize_fields = _pydantic_fields(route_src, "OptimizeRequest")
    assert optimize_fields >= {
        "objective", "seed_smiles", "dataset_id", "column",
        "population_size", "generations", "mutation_rate",
        "crossover_rate", "elitism", "tournament_k", "max_smiles_length",
        "seed", "top_n", "mode",
    }, "the backend OptimizeRequest changed shape (update this guard + the page)"
    assert _client_request_keys(client, "OptimizeRequest") == optimize_fields


def test_objective_catalog_fields_mirror_the_backend() -> None:
    """The mirrored objective-kind vocabulary is exactly the route's own
    ``_OBJECTIVE_CATALOG`` kinds, and the parser keeps every row key the
    catalog emits."""
    client = _read(SERVICES / "generative" / "client.ts")
    route_src = _route_module("generative.py")
    catalog = re.search(
        r"_OBJECTIVE_CATALOG:\s*list\[dict\[str,\s*Any\]\]\s*=\s*\[(.*?)\n\]",
        route_src,
        re.DOTALL,
    )
    assert catalog is not None, "the _OBJECTIVE_CATALOG block vanished"
    backend_kinds = set(re.findall(r'"kind":\s*"(\w+)"', catalog.group(1)))
    assert backend_kinds == {
        "qed", "sas", "logp_band", "mw_band", "tpsa_band", "similarity",
        "substructure", "novelty", "multi",
    }, f"the objective catalog changed shape: {sorted(backend_kinds)}"
    mirror = re.search(
        r"OBJECTIVE_KINDS:\s*readonly string\[\]\s*=\s*\[(.*?)\]\s*as const",
        client,
        re.DOTALL,
    )
    assert mirror is not None, "the OBJECTIVE_KINDS mirror vanished"
    mirrored = set(re.findall(r'"(\w+)"', mirror.group(1)))
    assert mirrored == backend_kinds, (
        f"the mirrored objective vocabulary drifted "
        f"(client {sorted(mirrored)} vs backend {sorted(backend_kinds)})"
    )
    # the catalog row keys are the route's own dict keys
    row_keys = set(re.findall(r'"(kind|class|params|description)":', catalog.group(1)))
    assert row_keys == {"kind", "class", "params", "description"}, (
        f"the catalog row dict changed shape: {sorted(row_keys)}"
    )
    for key in row_keys:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"the catalog parser dropped the '{key}' key the route emits"
        )


def test_prior_view_fields_mirror_the_backend() -> None:
    """The prior view parser keeps every key ``PriorRecord.view()``
    emits (the registry's own dict in stores.py)."""
    client = _read(SERVICES / "generative" / "client.ts")
    stores = _read(API / "stores.py")
    block = re.search(
        r"class PriorRecord.*?def view\(self\).*?return\s*\{(.*?)\n\s*\}",
        stores,
        re.DOTALL,
    )
    assert block is not None, "the PriorRecord.view block vanished"
    emitted = set(re.findall(r'"([a-z_]+)":', block.group(1)))
    assert emitted == {
        "prior_id", "name", "kind", "params", "n_train_smiles",
        "n_epochs", "final_loss", "losses", "corpus_source",
        "created_at", "sample_url",
    }, f"the prior view dict changed shape: {sorted(emitted)}"
    for key in emitted:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"parsePriorView dropped the '{key}' key the registry emits"
        )


def test_training_metrics_fields_mirror_the_backend() -> None:
    """The prior lane's loss-curve parser keys are exactly the publish
    dict the route sends when the prior's training ends."""
    client = _read(SERVICES / "generative" / "client.ts")
    route_src = _route_module("generative.py")
    block = re.search(
        r"bus\.publish\(\s*EventType\.TRAINING_METRICS,\s*\{(.*?)\}",
        route_src,
        re.DOTALL,
    )
    assert block is not None, "the TRAINING_METRICS publish block vanished"
    emitted = set(re.findall(r'"([a-z_]+)":', block.group(1)))
    assert emitted == {"stage", "epochs", "losses", "final_loss"}, (
        f"the prior loss-curve dict changed shape: {sorted(emitted)}"
    )
    for key in emitted:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"parseTrainingMetricsPayload dropped the '{key}' telemetry key"
        )
    # the vocabulary rides the drift-guarded event bus
    events = _read(SERVICES / "websocket" / "events.ts")
    assert '"training.metrics"' in events and '"iteration.update"' in events, (
        "the WS event vocabulary lost the generative telemetry events"
    )


def test_iteration_update_fields_mirror_the_backend() -> None:
    """The per-generation telemetry parser keys are exactly the envelope
    dict ``JobManager.report_progress`` publishes (jobs.py's own)."""
    client = _read(SERVICES / "generative" / "client.ts")
    jobs = _read(API / "jobs.py")
    block = re.search(
        r"self\._bus\.publish\(event,\s*\{(.*?)\}", jobs, re.DOTALL
    )
    assert block is not None, "the report_progress publish block vanished"
    emitted = set(re.findall(r'"([a-z_]+)"\s*:', block.group(1)))
    assert emitted == {"progress", "stage"}, (
        f"the iteration envelope changed shape: {sorted(emitted)}"
    )
    for key in emitted:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"parseIterationUpdatePayload dropped the '{key}' telemetry key"
        )
    # the route drives iteration.update through report_progress itself
    route_src = _route_module("generative.py")
    assert "EventType.ITERATION_UPDATE" in route_src, (
        "the optimize lane stopped riding the per-generation event"
    )


def test_optimize_result_fields_mirror_the_backend() -> None:
    """The optimization parser keeps every key ``_optimize_body``'s own
    answer carries (and the ranked ``top`` row keys)."""
    client = _read(SERVICES / "generative" / "client.ts")
    route_src = _route_module("generative.py")
    block = re.search(
        r'return\s*\{\s*"optimization":\s*\{(.*?)\}\s*\}',
        route_src,
        re.DOTALL,
    )
    assert block is not None, "the _optimize_body answer block vanished"
    emitted = set(re.findall(r'"([a-z_]+)":', block.group(1)))
    assert emitted == {
        "best", "best_score", "generations", "history", "metadata", "top",
    }, f"the optimization answer changed shape: {sorted(emitted)}"
    for key in emitted:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"parseOptimizationView dropped the '{key}' key the route emits"
        )
    top = re.search(
        r'top\s*=\s*\[\s*\{(.*?)\}\s*for', route_src, re.DOTALL
    )
    assert top is not None, "the ranked top block vanished"
    top_keys = set(re.findall(r'"([a-z_]+)":', top.group(1)))
    assert top_keys == {"smiles", "score", "breakdown"}, (
        f"the ranked row dict changed shape: {sorted(top_keys)}"
    )
    for key in top_keys:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"the ranked-row parser dropped the '{key}' key"
        )


def test_figure_lanes_ride_the_artifact_lane() -> None:
    """The Visualize stage: the prior loss-curve figure + the ranked
    score profile through POST /artifacts/figure (the REAL §17 export
    stack) with their honest minimums, riding the generated contract
    type."""
    client = _read(SERVICES / "generative" / "client.ts")
    page = _read(FEATURES / "generative" / "GenerativePage.tsx")
    assert "priorLossFigure" in client and "rankFigure" in client, (
        "a figure lane vanished from the client"
    )
    assert re.search(r"/api/v1/artifacts/figure", client) is not None, (
        "the figures stopped riding the figure artifact lane"
    )
    # the honest minimums, stated before the requests fire
    assert re.search(r"at least two epochs", client) is not None, (
        "the loss-curve lane lost its two-epoch minimum"
    )
    assert re.search(r"at least two scored molecules", client) is not None, (
        "the score-profile lane lost its two-molecule minimum"
    )
    # the figure request bodies ride the generated contract types
    assert "FigureRequest" in client and (
        "interface FigureRequest" not in client
    ), "the figure requests stopped using the generated contract type"
    # the ranked profile names its real axis — the score label the
    # Rank stage selected, never an invented one
    assert "scoreLabel" in client, (
        "the score profile stopped naming its real ranking key"
    )
    assert "digest_verified" in page, "the figures lost their digest labels"


def test_export_rides_the_rows_table_lane() -> None:
    """The export stage: rows-based /visualization/table + the backend's
    own 5000-row cap mirrored client-side + the digest-verified labels."""
    page = _read(FEATURES / "generative" / "GenerativePage.tsx")
    client = _read(SERVICES / "generative" / "client.ts")
    assert "VisualizationTableRequest" in client, (
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
    assert re.search(r"/api/v1/visualization/table", client) is not None, (
        "the export stopped riding the rows-based table lane"
    )


def test_validate_rides_the_chemistry_engine() -> None:
    """The Validate + Calculate Properties stages ride the REAL descriptor
    engine through the chemistry client, with the engine's own 500 cap
    mirrored client-side and the frame's own index as the verdict."""
    page = _read(FEATURES / "generative" / "GenerativePage.tsx")
    chemistry = _read(SERVICES / "chemistry" / "client.ts")
    cap = re.search(r"CHEMISTRY_MAX_SMILES\s*=\s*(\d+)", chemistry)
    assert cap is not None and cap.group(1) == "500", (
        "the mirrored chemistry cap drifted from chemistry.py's max_length"
    )
    route_src = _route_module("chemistry.py")
    backend_cap = re.search(
        r'Field\(min_length=1,\s*max_length=(\d+)\)', route_src
    )
    assert backend_cap is not None, "chemistry.py lost its smiles cap pin"
    assert int(cap.group(1)) == int(backend_cap.group(1)), (
        "the mirrored chemistry cap and the backend cap disagree"
    )
    assert "chemistryApi" in page and "descriptors" in page, (
        "the Validate stage stopped riding the chemistry engine"
    )
    assert "rdkit_all" in page, (
        "the Properties stage stopped using the rdkit_all descriptor set"
    )
    # the verdict note: the frame's own index IS the answer
    assert re.search(r"never a\s+client-side guess|never a client-side guess", page), (
        "the Validate stage lost its honest verdict note"
    )


def test_workspace_flipped_to_workflow() -> None:
    """generative-ai: shell → workflow, and the App renders the page."""
    workspaces = _read(APP / "workspaces.ts")
    match = re.search(
        r'id:\s*"generative-ai",.*?state:\s*"(\w+)"', workspaces,
        re.DOTALL,
    )
    assert match is not None, "the generative-ai entry vanished"
    assert match.group(1) == "workflow", (
        "the Generative workspace is still a shell after the §11.4 build"
    )
    app = _read(APP / "App.tsx")
    assert 'GenerativePage' in app and (
        'from "@/features/generative/GenerativePage"' in app
    ), "the App stopped importing the Generative page"
    assert re.search(r'entry\.id === "generative-ai"', app) is not None, (
        "the App lost the generative-ai render branch"
    )


def test_stage_order_matches_the_spec_chain() -> None:
    """The page's StageHeader sequence = §11.4's fixed workflow order."""
    page = _read(FEATURES / "generative" / "GenerativePage.tsx")
    titles = re.findall(r'<StageHeader\b[^>]*?title="([^"]+)"', page, re.DOTALL)
    expected = [
        "Select Generation Method",
        "Define Starting Molecules",
        "Configure Constraints/Objectives",
        "Generate",
        "Validate",
        "Calculate Properties",
        "Filter",
        "Rank",
        "Visualize",
        "Export",
    ]
    assert titles == expected, (
        "the §11.4 stage order drifted "
        f"(got {titles}, expected {expected})"
    )


def test_honesty_blocks_present() -> None:
    """§12's labels: the sync sample lane note (no pretend tracking),
    the ≥ 5 corpus rule (the route's own), the native-cancel note
    (CANCELLED, never FAILED), the multi-nesting note."""
    page = _read(FEATURES / "generative" / "GenerativePage.tsx")
    assert re.search(r"sync by design", page) is not None, (
        "the sample lane lost its honest sync-by-design note"
    )
    assert re.search(r"at least 5", page) is not None, (
        "the corpus stage lost the route's own ≥ 5 rule note"
    )
    assert re.search(r"CANCELLED, never FAILED", page) is not None, (
        "the optimize lane lost its native-cancel note"
    )
    client = _read(SERVICES / "generative" / "client.ts")
    assert "the route declares no job mode" in client, (
        "the client lost the sample lane's honest sync note"
    )


def test_telemetry_rides_the_real_event_stream() -> None:
    """The Generate stage subscribes to the live WS bus (exactly the
    envelope vocabulary the backend's own hooks publish — the prior
    lane's training.metrics, the optimizer's iteration.update) and
    parses each payload at runtime, job-scoped."""
    page = _read(FEATURES / "generative" / "GenerativePage.tsx")
    assert "getEventStream()" in page, (
        "the Generate stage stopped subscribing to the event stream"
    )
    assert 'envelope.event_type === "training.metrics"' in page, (
        "the prior lane lost its training.metrics filter"
    )
    assert 'envelope.event_type === "iteration.update"' in page, (
        "the optimize lane lost its iteration.update filter"
    )
    assert "parseTrainingMetricsPayload" in page and (
        "parseIterationUpdatePayload" in page
    ), "the telemetry stopped parsing at runtime"
    # the job-scoped filters: only THIS job's envelopes accumulate
    assert re.search(r"envelope\.job_id !== descriptorJobId", page) is not None, (
        "the descriptor lane lost its job-scoped envelope filter"
    )
