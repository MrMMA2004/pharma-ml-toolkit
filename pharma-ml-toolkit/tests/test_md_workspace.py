"""part11 §11.7, the Molecular Dynamics Analysis workspace row — the
drift guards.

The Python gate cannot execute the React page, but every contract the
MD workspace rides CAN be pinned against the backend source that
defines it, so a backend-side rename turns into a red test before the
frontend drifts:

* the md client's REST surface vs
  ``pharmaml.api.routes.md`` plus the shared lanes the page rides
  (the visualization md-series figure lane, the visualization table
  lane);
* the request bodies' field parity — the request objects the client
  builds vs the pydantic ``SyntheticRequest`` / ``MDAnalyzeRequest``
  / ``XvgRequest`` / ``EnmRequest`` / ``PcaRequest`` /
  ``StudyRequest`` the route actually defines (§12.2's
  one-DTO-source rule, enforced at the key level — the types come
  from the generated ``api.d.ts``, never hand-duplicated beside it);
* the response view keys vs the route's own answer dicts — the
  metric bundle view vs ``_metric_bundle``'s own return keys (plus
  the contact-map occupancy block and the PCA block), the xvg view
  vs the ``xvg`` route's own answer, the ENM view, the PCA view, the
  study body vs ``_study_body``'s own return dict, and the catalog
  groups vs the ``catalog`` route's own rows;
* every ``/api/v1`` literal the page + client carry must exist in the
  exported ``contracts/openapi.json`` (the anti-fabrication guard: no
  call to a route the contract never declared);
* the figure lane: the md-series figure through
  ``POST /visualization/md-series`` over the pasted ``.xvg`` text;
* the export lane: rows-based ``POST /visualization/table`` with the
  backend's own 5000-row cap mirrored client-side;
* the workspace registry flip (molecular-dynamics ``shell`` →
  ``workflow``) + the App wiring + the unchanged §10.1 order;
* the §11.7 stage order (the page's StageHeader sequence) and the
  honesty blocks (the analysis-vs-simulation distinction, the
  backend's own validator rendered as the authority, the no-guess
  catalog, the native-cancel note, the job/sync lane honesty);
* the study job lane rides the REAL tracker: the JobRunCard +
  ``useJob`` machinery, the COMPLETED job's REST result relayed
  through the jobs client (never the WS event payload alone), and
  the WS ``job.progress`` envelopes parsed at runtime, job-scoped.
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


def test_md_rest_surface_covered() -> None:
    """Every path the md client carries is a route the backend defines;
    all seven md routes are covered method-for-method, and the shared
    figure/table lanes are real routes too (the jobs client owns the
    job surface — never re-declared here)."""
    client = _read(SERVICES / "md" / "client.ts")
    carried = _frontend_api_paths(client)
    assert carried, "the md client lost every path literal"
    real: dict[str, set[str]] = {}
    for module in ("md", "visualization"):
        for path, methods in _backend_routes(module).items():
            real.setdefault(path, set()).update(methods)
    shared = {
        "/api/v1/visualization/md-series": {"POST"},
        "/api/v1/visualization/table": {"POST"},
    }
    for path, methods in shared.items():
        assert real.get(path, set()) >= methods, f"{path} left the backend"
    unknown = carried - set(real)
    assert not unknown, (
        f"the md client calls routes the backend never defined: {sorted(unknown)}"
    )
    # the md surface itself, method-for-method
    md_routes = _backend_routes("md")
    assert md_routes == {
        "/api/v1/md/catalog": {"GET"},
        "/api/v1/md/synthetic": {"POST"},
        "/api/v1/md/analyze": {"POST"},
        "/api/v1/md/xvg": {"POST"},
        "/api/v1/md/enm": {"POST"},
        "/api/v1/md/pca": {"POST"},
        "/api/v1/md/study": {"POST"},
    }, f"the md route surface drifted: {sorted(md_routes)}"
    for path, methods in md_routes.items():
        assert path in carried, f"{path} is not carried by the md client"
        assert real[path] == methods
    # the shared lanes the page rides are carried too
    for path in shared:
        assert path in carried, f"{path} is not carried by the md client"


def test_md_page_calls_only_contract_routes() -> None:
    """The anti-fabrication guard: every /api/v1 literal in the page +
    the md client exists in the exported openapi contract."""
    page = _read(FEATURES / "md" / "MDPage.tsx")
    client = _read(SERVICES / "md" / "client.ts")
    carried = _frontend_api_paths(page) | _frontend_api_paths(client)
    assert carried, "the MD workspace lost every path literal"
    contract = json.loads(_read(CONTRACTS / "openapi.json"))
    declared = set(contract["paths"])
    unknown = carried - declared
    assert not unknown, (
        f"the MD workspace calls routes the contract never declared: "
        f"{sorted(unknown)}"
    )


def test_trajectory_lane_requests_mirror_the_backend() -> None:
    """The trajectory lanes' request bodies: the client's keys vs the
    pydantic DTOs the route actually defines (the optional
    flexible_residues / masses ride the route's own optional keys)."""
    client = _read(SERVICES / "md" / "client.ts")
    route = _route_module("md.py")
    synthetic_fields = _pydantic_fields(route, "SyntheticRequest")
    assert synthetic_fields == {
        "n_frames", "seed", "n_residues", "noise", "drift_radians",
        "flexible_residues",
    }, f"the synthetic DTO drifted: {sorted(synthetic_fields)}"
    assert _client_request_keys(client, "SyntheticRequest") == synthetic_fields, (
        "the synthetic request keys drifted from SyntheticRequest"
    )
    analyze_fields = _pydantic_fields(route, "MDAnalyzeRequest")
    assert analyze_fields == {"coords", "masses"}, (
        f"the analyze DTO drifted: {sorted(analyze_fields)}"
    )
    assert _client_request_keys(client, "MDAnalyzeRequest") == analyze_fields, (
        "the analyze request keys drifted from MDAnalyzeRequest"
    )


def test_xvg_enm_pca_study_requests_mirror_the_backend() -> None:
    """The xvg / enm / pca / study lanes' request bodies, key-for-key
    against the route's own pydantic DTOs."""
    client = _read(SERVICES / "md" / "client.ts")
    route = _route_module("md.py")
    assert _client_request_keys(client, "XvgRequest") == _pydantic_fields(
        route, "XvgRequest"
    ), "the xvg request keys drifted from XvgRequest"
    assert _pydantic_fields(route, "XvgRequest") == {
        "text", "moving_average_window",
    }, "the xvg DTO drifted"
    assert _client_request_keys(client, "EnmRequest") == _pydantic_fields(
        route, "EnmRequest"
    ), "the enm request keys drifted from EnmRequest"
    assert _pydantic_fields(route, "EnmRequest") == {
        "coords", "cutoff", "gamma",
    }, "the enm DTO drifted"
    assert _client_request_keys(client, "PcaRequest") == _pydantic_fields(
        route, "PcaRequest"
    ), "the pca request keys drifted from PcaRequest"
    assert _pydantic_fields(route, "PcaRequest") == {
        "coords", "n_components",
    }, "the pca DTO drifted"
    assert _client_request_keys(client, "StudyRequest") == _pydantic_fields(
        route, "StudyRequest"
    ), "the study request keys drifted from StudyRequest"
    assert _pydantic_fields(route, "StudyRequest") == {
        "n_frames", "seed", "n_residues", "noise", "drift_radians",
        "title", "mode",
    }, "the study DTO drifted"
    # the generated contract types carry all five request DTOs
    generated = _read(FRONTEND / "src" / "types" / "api.d.ts")
    for dto in ("SyntheticRequest", "MDAnalyzeRequest", "XvgRequest",
                "EnmRequest", "PcaRequest", "StudyRequest"):
        assert f"{dto}:" in generated, f"the generated types lost {dto}"
        assert f'components["schemas"]["{dto}"]' in client, (
            f"the client stopped using the generated {dto} type"
        )


def test_response_view_keys_mirror_the_backend() -> None:
    """The response views: the client's own keys vs the backend's own
    answer dicts — the metric bundle vs `_metric_bundle`'s return
    (plus the contact-map occupancy block and the PCA block), the xvg
    view vs the xvg route's answer, the ENM view, the PCA view, the
    study body vs `_study_body`'s return, and the catalog rows vs the
    catalog route's own."""
    client = _read(SERVICES / "md" / "client.ts")
    route = _route_module("md.py")

    # the metric bundle vs _metric_bundle's own return keys
    bundle_block = re.search(
        r"def _metric_bundle.*?return \{(.*?)\n    \}", route, re.DOTALL
    )
    assert bundle_block is not None, "_metric_bundle vanished"
    bundle_keys = set(
        re.findall(r'^\s{8}"(\w+)":', bundle_block.group(1), re.MULTILINE)
    )
    assert bundle_keys == {
        "n_frames", "n_atoms", "rmsd", "rmsf", "radius_of_gyration",
        "native_contact_fraction", "contact_map", "pca",
    }, f"the metric bundle keys drifted: {sorted(bundle_keys)}"
    bundle_view = _interface_keys(client, "MetricBundleView")
    assert bundle_keys <= bundle_view, (
        "the metric bundle view drifted from _metric_bundle's return"
    )
    assert "source" in bundle_view, (
        "the bundle view lost the source discriminator the route carries"
    )

    # the contact-map occupancy block vs the route's own block
    contact_block = re.search(
        r'"contact_map":\s*\{(.*?)\},', route, re.DOTALL
    )
    assert contact_block is not None, "the route lost its contact_map block"
    contact_keys = set(re.findall(r'"(\w+)":', contact_block.group(1)))
    assert _interface_keys(client, "ContactMapSummaryView") == contact_keys, (
        "the contact-map view drifted from the route's own block"
    )

    # the trajectory PCA block vs the route's own block
    pca_block = re.search(r'"pca":\s*\{(.*?)\},\n', route, re.DOTALL)
    assert pca_block is not None, "the route lost its pca block"
    pca_keys = set(re.findall(r'"(\w+)":', pca_block.group(1)))
    assert _interface_keys(client, "TrajectoryPcaView") == pca_keys, (
        "the trajectory PCA view drifted from the route's own block"
    )

    # the xvg answer vs the route's own answer keys
    xvg_block = re.search(
        r'@router\.post\("/xvg"\).*?return \{(.*?)\n    \}', route, re.DOTALL
    )
    assert xvg_block is not None, "the xvg route's answer vanished"
    xvg_keys = set(
        re.findall(r'^\s{8}"(\w+)":', xvg_block.group(1), re.MULTILINE)
    )
    assert {
        "source", "label", "n_points", "time", "values", "stats",
        "moving_average",
    } == xvg_keys, f"the xvg answer keys drifted: {sorted(xvg_keys)}"
    assert _interface_keys(client, "XvgResponse") == xvg_keys, (
        "the xvg view drifted from the route's own answer"
    )
    # metric_stats' own keys (mean/std/cv, null-aware)
    xvg_stats = _interface_keys(client, "XvgStatsView")
    assert xvg_stats == {"mean", "std", "cv"}, (
        "the metric_stats view drifted from the module's own dict"
    )

    # the enm answer vs the route's own answer keys
    enm_block = re.search(
        r'@router\.post\("/enm"\).*?return \{(.*?)\n    \}', route, re.DOTALL
    )
    assert enm_block is not None, "the enm route's answer vanished"
    enm_keys = set(
        re.findall(r'^\s{8}"(\w+)":', enm_block.group(1), re.MULTILINE)
    )
    assert _interface_keys(client, "EnmResponse") == enm_keys, (
        "the enm view drifted from the route's own answer"
    )

    # the pca answer vs the route's own answer keys
    pca_route = re.search(
        r'@router\.post\("/pca"\).*?return \{(.*?)\n    \}', route, re.DOTALL
    )
    assert pca_route is not None, "the pca route's answer vanished"
    pca_route_keys = set(
        re.findall(r'^\s{8}"(\w+)":', pca_route.group(1), re.MULTILINE)
    )
    assert _interface_keys(client, "PcaResponse") == pca_route_keys, (
        "the pca view drifted from the route's own answer"
    )

    # the study body vs _study_body's own return keys (a flat dict —
    # the route's return is single-line)
    study_block = re.search(
        r"def _study_body.*?return \{(.*?)\}", route, re.DOTALL
    )
    assert study_block is not None, "_study_body vanished"
    study_keys = set(re.findall(r'"(\w+)":', study_block.group(1)))
    assert study_keys == {"title", "seed", "metrics", "artifacts"}, (
        f"the study body keys drifted: {sorted(study_keys)}"
    )
    body_view = _interface_keys(client, "StudyBodyView")
    assert study_keys <= body_view, (
        "the study body view drifted from _study_body's return"
    )
    # the study metrics block vs the route's own metrics dict
    metrics_block = re.search(
        r"metrics = \{(.*?)\n    \}", route, re.DOTALL
    )
    assert metrics_block is not None, "the study metrics dict vanished"
    metrics_keys = set(
        re.findall(r'^\s{8}"(\w+)":', metrics_block.group(1), re.MULTILINE)
    )
    assert _interface_keys(client, "StudyMetricsView") == metrics_keys, (
        "the study metrics view drifted from the route's own metrics dict"
    )

    # the catalog rows vs the catalog route's own group dict
    catalog_block = re.search(
        r'"group":\s*"trajectory".*?"backend":\s*"[^"]*"', route, re.DOTALL
    )
    assert catalog_block is not None, "the catalog route lost its group rows"
    group_view = _interface_keys(client, "MdCatalogGroup")
    assert group_view == {"group", "routes", "backend"}, (
        "the catalog group view drifted from the route's own row keys"
    )


def test_figure_lane_rides_the_md_series_lane() -> None:
    """The Visualize stage: the md-series figure through
    POST /visualization/md-series over the pasted .xvg text (the REAL
    §17 export stack), riding the generated contract type
    field-for-field."""
    client = _read(SERVICES / "md" / "client.ts")
    page = _read(FEATURES / "md" / "MDPage.tsx")
    assert "mdSeries" in client and "handleFigure" in page, (
        "the md-series figure lane vanished"
    )
    assert "/api/v1/visualization/md-series" in client, (
        "the figure stopped riding the visualization md-series lane"
    )
    assert 'components["schemas"]["MdSeriesRequest"]' in client, (
        "the md-series request stopped using the generated contract type"
    )
    viz_route = _route_module("visualization.py")
    md_series_fields = _pydantic_fields(viz_route, "MdSeriesRequest")
    assert _client_request_keys(client, "MdSeriesRequest") <= md_series_fields, (
        "the md-series request invented keys visualization.py never declared"
    )
    # the honest minimum, stated before the request fires
    assert "at least one named series" in client, (
        "the md-series lane lost its one-series minimum"
    )
    assert "digest_verified" in page, "the figures lost their digest labels"
    # the study lane's own artifacts arrive with the body (never re-fetched)
    assert "studyBody.artifacts" in page, (
        "the study artifacts stopped rendering with the body"
    )


def test_export_rides_the_rows_table_lane() -> None:
    """The export stage: rows-based /visualization/table + the backend's
    own 5000-row cap mirrored client-side + the digest-verified
    labels."""
    page = _read(FEATURES / "md" / "MDPage.tsx")
    client = _read(SERVICES / "md" / "client.ts")
    assert "exportTable" in client and "handleExport" in page, (
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
    viz_route = _route_module("visualization.py")
    table_fields = _pydantic_fields(viz_route, "VisualizationTableRequest")
    assert _client_request_keys(client, "VisualizationTableRequest") == table_fields, (
        "the table export request keys drifted from the route's own DTO"
    )
    assert "rows the backend actually computed" in page, (
        "the export lost its honest-rows note"
    )
    assert "digest_verified" in page


def test_workspace_flipped_to_workflow() -> None:
    """molecular-dynamics: shell → workflow, the App renders the page,
    and the §10.1 registry order is unchanged."""
    workspaces = _read(APP / "workspaces.ts")
    match = re.search(
        r'id:\s*"molecular-dynamics",.*?state:\s*"(\w+)"', workspaces, re.DOTALL
    )
    assert match is not None, "the molecular-dynamics entry vanished"
    assert match.group(1) == "workflow", (
        "the Molecular Dynamics Analysis workspace is still a shell "
        "after the §11.7 build"
    )
    app = _read(APP / "App.tsx")
    assert 'from "@/features/md/MDPage"' in app, (
        "the App stopped importing the MD page"
    )
    assert re.search(r'entry\.id === "molecular-dynamics"', app) is not None, (
        "the App lost the molecular-dynamics render branch"
    )
    assert (FEATURES / "md" / "MDPage.tsx").is_file(), (
        "the MD page file vanished"
    )
    # the §10.1 sequence stays exactly as test_frontend_foundation pins it
    labels = re.findall(r'label:\s*"([^"]+)"', workspaces)
    assert labels == [
        "Introduction", "Datasets", "Machine Learning", "Deep Learning",
        "Generative AI", "Molecular Docking", "PK/PD",
        "Molecular Dynamics Analysis", "Visualization",
    ], f"the workspace registry order drifted: {labels}"
    # §11.7's own distinction stays visible in the registry summary
    assert "distinct from simulation" in workspaces, (
        "the registry lost the analysis-vs-simulation distinction"
    )


def test_stage_order_matches_the_spec_chain() -> None:
    """The page's StageHeader sequence = §11.7's fixed workflow order
    (Load Topology/Trajectory → Validate → Select Analysis → Execute →
    Inspect Results → Visualize → Export)."""
    page = _read(FEATURES / "md" / "MDPage.tsx")
    titles = re.findall(r'<StageHeader\b[^>]*?title="([^"]+)"', page, re.DOTALL)
    expected = [
        "Load Topology/Trajectory",
        "Validate",
        "Select Analysis",
        "Execute",
        "Inspect Results",
        "Visualize",
        "Export",
    ]
    assert titles == expected, f"the §11.7 stage order drifted: {titles}"


def test_honesty_blocks_present() -> None:
    """§12's labels: the analysis-vs-simulation distinction (§11.7's own
    demand), the backend's own validator as the authority (a bad
    stack is its honest 422, surfaced verbatim — never a client-side
    guess), the catalog's no-guess note, the native-cancel note, the
    job/sync lane honesty."""
    page = _read(FEATURES / "md" / "MDPage.tsx")
    client = _read(SERVICES / "md" / "client.ts")
    # the analysis-vs-simulation distinction, stated in the page header
    assert "never simulation" in page or "not run simulations" in page, (
        "the page lost the trajectory-analysis-vs-simulation distinction"
    )
    assert "no MD engine is bundled or launched" in page, (
        "the page lost its no-MD-engine note (the §11.7 distinction)"
    )
    assert "never simulation" in client, (
        "the client lost the analysis-not-simulation note"
    )
    assert re.search(r"honest 422", page) is not None, (
        "the page stopped surfacing the route's honest 422s"
    )
    assert re.search(r"never a client-side guess", page) is not None, (
        "the catalog's no-guess note vanished from the page"
    )
    assert re.search(r"CANCELLED, never FAILED", page) is not None, (
        "the study lane lost its native-cancel note"
    )
    assert re.search(
        r"never the\s*\*?\s*WS[- ]event[- ]payload[- ]alone", client
    ) is not None, (
        "the client lost the WS-payload-only relay honesty note"
    )
    assert re.search(r"is authoritative", page) is not None, (
        "the job lane lost its REST-authoritative note"
    )
    assert "parseStudyBody" in page and "parseStudyBody" in client, (
        "the study REST result stopped being narrowed at runtime"
    )
    # the curves are drawn only from points that actually arrived
    assert re.search(r"points that actually arrived", page)
    assert re.search(
        r"points\s*\*?\s*that actually arrived", client
    ) is not None, (
        "the client lost its real-points figure note"
    )
    # the preview bound is a preview, never the payload
    assert "a preview, never the payload" in page, (
        "the preview stopped stating it is never the exported payload"
    )


def test_study_job_lane_rides_the_real_tracker() -> None:
    """The Execute study lane rides the REAL job machinery: the
    tracker-driven JobRunCard (WS state + REST reconciliation, the
    cancellation lane), the COMPLETED job's REST result relayed
    through the jobs client (never the WS event payload alone), and
    the WS job.progress envelopes parsed at runtime, job-scoped."""
    page = _read(FEATURES / "md" / "MDPage.tsx")
    client = _read(SERVICES / "md" / "client.ts")
    assert "JobRunCard" in page and "useJob" in page, (
        "the study lane stopped riding the tracker's job card"
    )
    assert "jobsApi" in page and ".result(jobId)" in page, (
        "the job lane stopped relaying the authoritative REST result"
    )
    assert "getEventStream()" in page, (
        "the study lane stopped subscribing to the event stream"
    )
    assert 'envelope.event_type === "job.progress"' in page, (
        "the study lane lost its job.progress filter"
    )
    assert "parseJobProgressPayload" in page, (
        "the study telemetry stopped parsing at runtime"
    )
    assert re.search(r"envelope\.job_id !== studyJobId", page) is not None, (
        "the study lane lost its job-scoped envelope filter"
    )
    assert 'raw["mode"] === "sync"' in client, (
        "the study lane lost the route's sync branch"
    )
    assert "parseJobProgressPayload" in client, (
        "the client stopped narrowing the job.progress envelopes"
    )
    # the WS is progress-only; the body rides the REST relay
    assert re.search(
        r"never the\s*\*?\s*WS[- ]event[- ]payload[- ]alone", client
    ) is not None, (
        "the client lost the WS-is-progress-only honesty note"
    )
    # the study relay narrows the REAL body at runtime
    assert "parseStudyBody" in client, (
        "the study body parser vanished from the client"
    )
