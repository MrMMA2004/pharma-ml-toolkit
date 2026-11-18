"""part11 §11.8, the Visualization workspace row — the drift guards.

The Python gate cannot execute the React page, but every contract the
Visualization workspace rides CAN be pinned against the backend
source that defines it, so a backend-side rename turns into a red
test before the frontend drifts:

* the visualization client's REST surface vs
  ``pharmaml.api.routes.visualization`` (six routes: catalog / stat /
  concentration-profile / dose-accumulation / md-series / table /
  report);
* the request bodies' field parity — the request objects the client
  builds vs the pydantic ``StatRequest`` /
  ``ConcentrationProfileRequest`` / ``DoseAccumulationRequest`` /
  ``MdSeriesRequest`` / ``VisualizationTableRequest`` /
  ``ReportRequest`` the route actually defines (§12.2's
  one-DTO-source rule, enforced at the key level — the types come
  from the generated ``api.d.ts``, never hand-duplicated beside it);
* the response view keys vs the route's own answer dicts — the
  catalog view vs the ``catalog`` route's own families block, the
  stat view vs the ``stat`` route's answer, the pharma-figure lanes'
  answers, the md-series answer, the table answer, and the report
  body vs ``_report_body``'s own return dict;
* every ``/api/v1`` literal the page + client carry must exist in the
  exported ``contracts/openapi.json`` (the anti-fabrication guard: no
  call to a route the contract never declared);
* §11.8's cross-workspace demand — the page reads the session's
  artifact registry through the shared ``artifactsApi.list()``
  (``GET /api/v1/artifacts``), never a duplicated client;
* the export lane: rows-based ``POST /visualization/table`` with the
  backend's own 5000-row cap mirrored client-side;
* the workspace registry flip (visualization ``shell`` →
  ``workflow``) + the App wiring + the unchanged §10.1 order;
* the §11.8 stage order (the page's StageHeader sequence) and the
  honesty blocks (the backend's own validator as the authority, the
  no-guess catalog, the native-cancel note, the job/sync lane
  honesty, the honest compare pane);
* the report job lane rides the REAL tracker: the JobRunCard +
  ``useJob`` machinery, the COMPLETED job's REST result relayed
  through the jobs client (never the WS event payload alone), and
  the WS ``file.progress`` envelopes parsed at runtime, job-scoped.
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
    src = _route_module(
        module_name if module_name.endswith(".py") else f"{module_name}.py"
    )
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
        r"^class " + re.escape(class_name) + r"\((?:BaseModel|[A-Za-z_][\w]*)\):",
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


def _interface_keys(client_src: str, type_name: str) -> set[str]:
    """The prop names of one exported interface block in the client."""
    pattern = re.compile(
        r"export interface " + re.escape(type_name) + r"[^{]*\{(.*?)\n\}",
        re.DOTALL,
    )
    match = pattern.search(client_src)
    assert match is not None, f"the {type_name} interface vanished"
    return set(re.findall(r"^\s+readonly (\w+)", match.group(1), re.MULTILINE))


def _frontend_api_paths(source: str) -> set[str]:
    """Normalize the literals (query suffixes stripped, interpolations
    become ``{param}``) to the backend's route-path shapes."""
    normalized = source.replace("${encodeURIComponent(name)}", "{name}")
    paths: set[str] = set()
    for match in re.finditer(r"[\"'`](/api/v[0-9]+/[^\"'`?]*)", normalized):
        paths.add(match.group(1))
    return paths


def test_visualization_rest_surface_covered() -> None:
    """Every path the visualization client carries is a route the
    backend defines; all seven visualization routes are covered
    method-for-method (the registry lane is the artifacts client's
    own — never re-declared here)."""
    client = _read(SERVICES / "visualization" / "client.ts")
    carried = _frontend_api_paths(client)
    assert carried, "the visualization client lost every path literal"
    real = _backend_routes("visualization")
    unknown = carried - set(real)
    assert not unknown, (
        f"the visualization client calls routes the backend never "
        f"defined: {sorted(unknown)}"
    )
    # the visualization surface itself, method-for-method
    viz_routes = _backend_routes("visualization")
    assert viz_routes == {
        "/api/v1/visualization/catalog": {"GET"},
        "/api/v1/visualization/stat": {"POST"},
        "/api/v1/visualization/concentration-profile": {"POST"},
        "/api/v1/visualization/dose-accumulation": {"POST"},
        "/api/v1/visualization/md-series": {"POST"},
        "/api/v1/visualization/table": {"POST"},
        "/api/v1/visualization/report": {"POST"},
    }, f"the visualization route surface drifted: {sorted(viz_routes)}"
    for path, methods in viz_routes.items():
        assert path in carried, f"{path} is not carried by the viz client"
        assert real[path] == methods


def test_visualization_page_calls_only_contract_routes() -> None:
    """The anti-fabrication guard: every /api/v1 literal in the page +
    the visualization client exists in the exported openapi
    contract."""
    page = _read(FEATURES / "visualization" / "VisualizationPage.tsx")
    client = _read(SERVICES / "visualization" / "client.ts")
    carried = _frontend_api_paths(page) | _frontend_api_paths(client)
    assert carried, "the Visualization workspace lost every path literal"
    contract = json.loads(_read(CONTRACTS / "openapi.json"))
    declared = set(contract["paths"])
    unknown = carried - declared
    assert not unknown, (
        f"the Visualization workspace calls routes the contract never "
        f"declared: {sorted(unknown)}"
    )


def test_request_bodies_mirror_the_backend() -> None:
    """The request bodies: the client's keys vs the pydantic DTOs the
    route actually defines (§12.2's one-DTO-source rule, enforced at
    the key level — the types come from the generated api.d.ts)."""
    client = _read(SERVICES / "visualization" / "client.ts")
    route = _route_module("visualization.py")
    pairs = [
        ("StatRequest", "StatRequest"),
        ("ConcentrationProfileRequest", "ConcentrationProfileRequest"),
        ("DoseAccumulationRequest", "DoseAccumulationRequest"),
        ("MdSeriesRequest", "MdSeriesRequest"),
        ("VisualizationTableRequest", "VisualizationTableRequest"),
        ("ReportRequest", "ReportRequest"),
    ]
    for dto, request_type in pairs:
        backend_fields = _pydantic_fields(route, dto)
        assert backend_fields, f"the {dto} DTO vanished from the backend source"
        assert _client_request_keys(client, request_type) <= backend_fields, (
            f"the {dto} request invented keys the route never declared"
        )
        assert f'components["schemas"]["{request_type}"]' in client, (
            f"the client stopped using the generated {request_type} type"
        )
    # the stat DTO's own vocabulary (the statplots kinds)
    stat_fields = _pydantic_fields(route, "StatRequest")
    assert stat_fields == {
        "kind", "values", "x", "y", "bins", "whis", "n_boot", "seed",
        "color", "with_artifact", "format",
    }, f"the stat DTO drifted: {sorted(stat_fields)}"
    # the report DTO's own vocabulary
    report_fields = _pydantic_fields(route, "ReportRequest")
    assert report_fields == {
        "title", "subtitle", "theme", "metrics", "tables",
        "concentration_profile", "dose_accumulation", "stat", "mode",
    }, f"the report DTO drifted: {sorted(report_fields)}"


def test_response_view_keys_mirror_the_backend() -> None:
    """The response views: the client's own keys vs the backend's own
    answer dicts — the stat view vs the stat route's answer, the
    pharma-figure lanes' answers, the md-series answer, the table
    answer, the report body vs `_report_body`'s return, and the
    catalog rows vs the catalog route's own families block."""
    client = _read(SERVICES / "visualization" / "client.ts")
    route = _route_module("visualization.py")

    # the stat answer vs the stat route's own answer keys
    stat_block = re.search(
        r'@router\.post\("/stat"\).*?return \{(.*?)\n    \}', route, re.DOTALL
    )
    assert stat_block is not None, "the stat route's answer vanished"
    stat_keys = set(
        re.findall(r'^\s{8}"(\w+)":', stat_block.group(1), re.MULTILINE)
    )
    assert stat_keys == {"kind", "n_points", "stats", "artifact"}, (
        f"the stat answer keys drifted: {sorted(stat_keys)}"
    )
    assert _interface_keys(client, "StatResponse") == stat_keys, (
        "the stat view drifted from the route's own answer"
    )

    # the concentration-profile answer vs the route's own answer
    # (a flat single-line return)
    cp_block = re.search(
        r'@router\.post\("/concentration-profile"\).*?return \{(.*?)\}',
        route, re.DOTALL,
    )
    assert cp_block is not None, "the concentration-profile answer vanished"
    cp_keys = set(re.findall(r'"(\w+)":', cp_block.group(1)))
    assert _interface_keys(client, "ConcentrationProfileResponse") == cp_keys, (
        "the concentration-profile view drifted from the route's own answer"
    )

    # the dose-accumulation answer vs the route's own answer
    # (a flat single-line return)
    da_block = re.search(
        r'@router\.post\("/dose-accumulation"\).*?return \{(.*?)\}',
        route, re.DOTALL,
    )
    assert da_block is not None, "the dose-accumulation answer vanished"
    da_keys = set(re.findall(r'"(\w+)":', da_block.group(1)))
    assert _interface_keys(client, "DoseAccumulationResponse") == da_keys, (
        "the dose-accumulation view drifted from the route's own answer"
    )

    # the md-series answer vs the route's own answer
    ms_block = re.search(
        r'@router\.post\("/md-series"\).*?return \{(.*?)\n    \}',
        route, re.DOTALL,
    )
    assert ms_block is not None, "the md-series answer vanished"
    ms_keys = set(
        re.findall(r'^\s{8}"(\w+)":', ms_block.group(1), re.MULTILINE)
    )
    assert _interface_keys(client, "VizMdSeriesResponse") == ms_keys, (
        "the md-series view drifted from the route's own answer"
    )
    # the md-series per-series stats row vs the route's own series block
    series_row = _interface_keys(client, "VizSeriesStatView")
    assert {"n_points", "label", "stats"} <= series_row, (
        "the md-series stats row lost the route's own series keys"
    )

    # the table answer vs the route's own answer (a flat single-line
    # return)
    table_block = re.search(
        r'@router\.post\("/table"\).*?return \{(.*?)\}',
        route, re.DOTALL,
    )
    assert table_block is not None, "the table answer vanished"
    table_keys = set(re.findall(r'"(\w+)":', table_block.group(1)))
    assert _interface_keys(client, "VizTableResponse") == table_keys, (
        "the table view drifted from the route's own answer"
    )

    # the report body vs _report_body's own return keys (the function
    # has two returns — the artifact-store lane is the one the routes
    # ride; the union of both return dicts is the contract)
    body_span = re.search(
        r"def _report_body(.*?)@router\.post\(\"/report\"\)", route, re.DOTALL
    )
    assert body_span is not None, "_report_body vanished"
    report_keys: set[str] = set()
    for body_return in re.finditer(
        r"return \{(.*?)\}", body_span.group(1), re.DOTALL
    ):
        report_keys.update(re.findall(r'"(\w+)":', body_return.group(1)))
    assert report_keys == {"title", "stages", "figures", "artifact"}, (
        f"the report body keys drifted: {sorted(report_keys)}"
    )
    assert _interface_keys(client, "ReportBodyView") == report_keys, (
        "the report body view drifted from _report_body's return"
    )

    # the catalog view vs the catalog route's own answer keys
    catalog_block = re.search(
        r'@router\.get\("/catalog"\).*?return \{(.*?)\n    \}', route, re.DOTALL
    )
    assert catalog_block is not None, "the catalog answer vanished"
    catalog_keys = set(
        re.findall(r'^\s{8}"(\w+)":', catalog_block.group(1), re.MULTILINE)
    )
    assert catalog_keys == {
        "families", "palette", "export_formats", "report_themes",
    }, f"the catalog answer keys drifted: {sorted(catalog_keys)}"
    catalog_view = _interface_keys(client, "VizCatalogResponse")
    assert catalog_view == catalog_keys, (
        "the catalog view drifted from the route's own answer"
    )
    family_view = _interface_keys(client, "VizCatalogFamily")
    assert {"family", "kinds", "formats", "backend"} == family_view, (
        "the catalog family view drifted from the route's own row keys"
    )


def test_cross_workspace_outputs_ride_the_shared_registry() -> None:
    """§11.8's own demand: the workspace accesses the outputs created
    in other workspaces through the SHARED registry client — the page
    rides `artifactsApi.list()` (GET /api/v1/artifacts), never a
    duplicated client of its own."""
    page = _read(FEATURES / "visualization" / "VisualizationPage.tsx")
    client = _read(SERVICES / "visualization" / "client.ts")
    artifacts_client = _read(SERVICES / "artifacts" / "client.ts")
    assert re.search(r"artifactsApi\s*\.\s*list\(\)", page) is not None, (
        "the page stopped reading the session's artifact registry"
    )
    assert "/api/v1/artifacts" not in client, (
        "the visualization client re-declared the registry lane — the "
        "artifacts client owns it"
    )
    assert "/api/v1/artifacts" in artifacts_client, (
        "the shared artifacts client lost its registry lane"
    )
    # the page states the cross-workspace provenance honestly
    assert "outputs other workspaces" in page or "other workspaces" in page, (
        "the page lost its cross-workspace provenance note"
    )


def test_export_rides_the_rows_table_lane() -> None:
    """The export stage: rows-based /visualization/table + the backend's
    own 5000-row cap mirrored client-side + the digest-verified
    labels."""
    page = _read(FEATURES / "visualization" / "VisualizationPage.tsx")
    client = _read(SERVICES / "visualization" / "client.ts")
    assert "exportTable" in client and "tableResult" in page, (
        "the table export lane vanished"
    )
    assert "/api/v1/visualization/table" in client, (
        "the export stopped riding the rows-based table lane"
    )
    assert "the backend's own cap is 5000" in client, (
        "the client lost the backend's own 5000-row cap (a silent "
        "truncation would hide it)"
    )
    assert 'components["schemas"]["VisualizationTableRequest"]' in client
    route = _route_module("visualization.py")
    table_fields = _pydantic_fields(route, "VisualizationTableRequest")
    assert _client_request_keys(client, "VisualizationTableRequest") == table_fields, (
        "the table export request keys drifted from the route's own DTO"
    )
    assert "digest_verified" in page, "the exports lost their digest labels"
    # the report theme is the catalog's own, never invented client-side
    assert re.search(r"report_themes", page) is not None, (
        "the page stopped rendering the catalog's own report themes"
    )


def test_workspace_flipped_to_workflow() -> None:
    """visualization: shell → workflow, the App renders the page, and
    the §10.1 registry order is unchanged."""
    workspaces = _read(APP / "workspaces.ts")
    match = re.search(
        r'id:\s*"visualization",.*?state:\s*"(\w+)"', workspaces, re.DOTALL
    )
    assert match is not None, "the visualization entry vanished"
    assert match.group(1) == "workflow", (
        "the Visualization workspace is still a shell after the §11.8 build"
    )
    app = _read(APP / "App.tsx")
    assert 'from "@/features/visualization/VisualizationPage"' in app, (
        "the App stopped importing the Visualization page"
    )
    assert re.search(r'entry\.id === "visualization"', app) is not None, (
        "the App lost the visualization render branch"
    )
    assert (FEATURES / "visualization" / "VisualizationPage.tsx").is_file(), (
        "the Visualization page file vanished"
    )
    # the §10.1 sequence stays exactly as test_frontend_foundation pins it
    labels = re.findall(r'label:\s*"([^"]+)"', workspaces)
    assert labels == [
        "Introduction", "Datasets", "Machine Learning", "Deep Learning",
        "Generative AI", "Molecular Docking", "PK/PD",
        "Molecular Dynamics Analysis", "Visualization",
    ], f"the workspace registry order drifted: {labels}"
    assert "WORKSPACE_ORDER" in workspaces, (
        "the exported order constant vanished from the registry"
    )


def test_stage_order_matches_the_spec_chain() -> None:
    """The page's StageHeader sequence = §11.8's fixed workflow order
    (Select Dataset/Analysis Result → Select Visualization → Configure
    Variables → Generate → Inspect → Filter/Compare → Export)."""
    page = _read(FEATURES / "visualization" / "VisualizationPage.tsx")
    titles = re.findall(r'<StageHeader\b[^>]*?title="([^"]+)"', page, re.DOTALL)
    expected = [
        "Select Dataset/Analysis Result",
        "Select Visualization",
        "Configure Variables",
        "Generate",
        "Inspect",
        "Filter/Compare",
        "Export",
    ]
    assert titles == expected, f"the §11.8 stage order drifted: {titles}"


def test_honesty_blocks_present() -> None:
    """§12's labels: the backend's own validator as the authority (the
    scalar ranges are its honest 422s — scientific verdicts, not
    transport errors), the catalog's no-guess note, the no-fake
    discipline, the native-cancel note, the job/sync lane honesty,
    and the honest compare pane (only over what actually arrived)."""
    page = _read(FEATURES / "visualization" / "VisualizationPage.tsx")
    client = _read(SERVICES / "visualization" / "client.ts")
    assert re.search(r"honest 422", page) is not None, (
        "the page stopped surfacing the route's honest 422s"
    )
    assert re.search(r"never a client-side guess", page) is not None, (
        "the catalog's no-guess note vanished from the page"
    )
    assert re.search(r"CANCELLED, never FAILED", page) is not None, (
        "the report lane lost its native-cancel note"
    )
    assert re.search(
        r"never the\s*\*?\s*WS[- ]event[- ]payload[- ]alone", client
    ) is not None, (
        "the client lost the WS-payload-only relay honesty note"
    )
    assert re.search(r"is authoritative", page) is not None, (
        "the job lane lost its REST-authoritative note"
    )
    assert "parseReportBody" in page and "parseReportBody" in client, (
        "the report REST result stopped being narrowed at runtime"
    )
    # the compare pane is honest — only over what actually arrived
    assert re.search(r"only the artifacts that\s*\*?\s*actually arrived", page) is not None, (
        "the compare pane lost its only-what-arrived honesty note"
    )
    # the stats are the backend's own verdicts, nothing invented
    assert "Nothing is client-side imputed" in page, (
        "the inspect stage lost its nothing-imputed note"
    )


def test_report_job_lane_rides_the_real_tracker() -> None:
    """The Generate report lane rides the REAL job machinery: the
    tracker-driven JobRunCard (WS state + REST reconciliation, the
    cancellation lane), the COMPLETED job's REST result relayed
    through the jobs client (never the WS event payload alone), and
    the WS file.progress envelopes parsed at runtime, job-scoped."""
    page = _read(FEATURES / "visualization" / "VisualizationPage.tsx")
    client = _read(SERVICES / "visualization" / "client.ts")
    assert "JobRunCard" in page and "useJob" in page, (
        "the report lane stopped riding the tracker's job card"
    )
    assert "jobsApi" in page and ".result(jobId)" in page, (
        "the job lane stopped relaying the authoritative REST result"
    )
    assert "getEventStream()" in page, (
        "the report lane stopped subscribing to the event stream"
    )
    assert 'envelope.event_type === "file.progress"' in page, (
        "the report lane lost its file.progress filter"
    )
    assert "parseJobProgressPayload" in page, (
        "the report telemetry stopped parsing at runtime"
    )
    assert re.search(r"envelope\.job_id !== reportJobId", page) is not None, (
        "the report lane lost its job-scoped envelope filter"
    )
    assert 'raw["mode"] === "sync"' in client, (
        "the report lane lost the route's sync branch"
    )
    assert "parseJobProgressPayload" in client, (
        "the client stopped narrowing the file.progress envelopes"
    )
    # the report relay narrows the REAL body at runtime
    assert "parseReportBody" in client, (
        "the report body parser vanished from the client"
    )
