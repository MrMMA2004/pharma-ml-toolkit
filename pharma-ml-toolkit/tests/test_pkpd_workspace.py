"""part11 §11.6, the PK/PD workspace row — the drift guards.

The Python gate cannot execute the React page, but every contract the
PK/PD workspace rides CAN be pinned against the backend source that
defines it, so a backend-side rename turns into a red test before the
frontend drifts:

* the pkpd client's REST surface vs
  ``pharmaml.api.routes.pkpd`` plus the shared lanes the page rides
  (the artifacts figure lane, the visualization table lane);
* the request bodies' field parity — the request objects the client
  builds vs the pydantic ``TimesConcRequest`` / ``SteadyStateRequest``
  / ``IvBolusRequest`` / ``BatemanRequest`` /
  ``SuperpositionRequest`` / ``PbpkRequest`` / ``PdRequest`` /
  ``PopulationRequest`` the route actually defines (§12.2's
  one-DTO-source rule, enforced at the key level — the types come from
  the generated ``api.d.ts``, never hand-duplicated beside it);
* the response view keys vs the route's own answer dicts — the NCA
  summary view vs ``PKSummary.as_dict()``'s keys, the interval metrics
  view vs ``SteadyStateMetrics.as_dict()``'s keys (both nullables
  included), the terminal-phase view vs the ``nca`` route's own
  terminal block, the PD view vs ``effect_table``'s own parallel
  lists, the population body vs ``_population_body``'s own return
  dict, and the catalog groups vs the ``catalog`` route's own rows;
* the PBPK frame columns vs ``MinimalPBPK.COMPARTMENTS`` + the
  frame's own inserted ``C_plasma`` column (rendered dynamically off
  the response, never a hardcoded client column list);
* the population lane's exact vector shapes (``theta`` exactly 3,
  ``omega_diag`` exactly 1, the grid 2…2000) and the mode lane (job
  default / sync), mirrored in the client's own pre-flight;
* every ``/api/v1`` literal the page + client carry must exist in the
  exported ``contracts/openapi.json`` (the anti-fabrication guard: no
  call to a route the contract never declared);
* the figure lanes: the exposure profile and the sigmoid Emax
  response curve through ``POST /artifacts/figure`` with their honest
  two-point minimums;
* the export lane: rows-based ``POST /visualization/table`` with the
  backend's own 5000-row cap mirrored client-side;
* the workspace registry flip (pkpd ``shell`` → ``workflow``) + the
  App wiring + the unchanged §10.1 order;
* the §11.6 stage order (the page's StageHeader sequence) and the
  honesty blocks (the backend's own validator rendered as the
  authority, the no-guess catalog, the native-cancel note, the
  job/sync lane honesty);
* the population job lane rides the REAL tracker: the JobRunCard +
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
PHARMAML = ROOT / "src" / "pharmaml"
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


def _class_block(module_src: str, class_name: str) -> str:
    """The whole source block of one class (fields + validators)."""
    start = re.search(
        r"^class " + re.escape(class_name) + r"\((?:BaseModel|[A-Za-z_][\w]*)\):",
        module_src,
        re.MULTILINE,
    )
    assert start is not None, f"{class_name} vanished from the backend source"
    rest = module_src[start.end() :]
    end = re.search(r"^(?:class |@router\.|def )", rest, re.MULTILINE)
    return rest[: end.start()] if end else rest


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


def test_pkpd_rest_surface_covered() -> None:
    """Every path the pkpd client carries is a route the backend
    defines; all nine pkpd routes are covered method-for-method, and
    the job surface is never re-declared here (the jobs client owns
    it)."""
    client = _read(SERVICES / "pkpd" / "client.ts")
    carried = _frontend_api_paths(client)
    assert carried, "the pkpd client lost every path literal"
    real: dict[str, set[str]] = {}
    for module in ("pkpd", "artifacts", "visualization"):
        for path, methods in _backend_routes(module).items():
            real.setdefault(path, set()).update(methods)
    # the shared lanes the page rides are real routes too
    shared = {
        "/api/v1/artifacts/figure": {"POST"},
        "/api/v1/visualization/table": {"POST"},
    }
    for path, methods in shared.items():
        assert real.get(path, set()) >= methods, f"{path} left the backend"
    unknown = carried - set(real)
    assert not unknown, (
        f"the pkpd client calls routes the backend never defined: {sorted(unknown)}"
    )
    # the pkpd surface itself, method-for-method
    pkpd_routes = _backend_routes("pkpd")
    assert pkpd_routes == {
        "/api/v1/pkpd/catalog": {"GET"},
        "/api/v1/pkpd/nca": {"POST"},
        "/api/v1/pkpd/steady-state": {"POST"},
        "/api/v1/pkpd/iv-bolus": {"POST"},
        "/api/v1/pkpd/bateman": {"POST"},
        "/api/v1/pkpd/superposition": {"POST"},
        "/api/v1/pkpd/pbpk": {"POST"},
        "/api/v1/pkpd/pd": {"POST"},
        "/api/v1/pkpd/population": {"POST"},
    }, f"the pkpd route surface drifted: {sorted(pkpd_routes)}"
    for path, methods in pkpd_routes.items():
        assert path in carried, f"{path} is not carried by the pkpd client"
        assert real[path] == methods


def test_pkpd_page_calls_only_contract_routes() -> None:
    """The anti-fabrication guard: every /api/v1 literal in the page +
    the pkpd client exists in the exported openapi contract."""
    page = _read(FEATURES / "pkpd" / "PKPDPage.tsx")
    client = _read(SERVICES / "pkpd" / "client.ts")
    carried = _frontend_api_paths(page) | _frontend_api_paths(client)
    assert carried, "the PK/PD workspace lost every path literal"
    contract = json.loads(_read(CONTRACTS / "openapi.json"))
    declared = set(contract["paths"])
    unknown = carried - declared
    assert not unknown, (
        f"the PK/PD workspace calls routes the contract never "
        f"declared: {sorted(unknown)}"
    )


def test_nca_and_steady_state_requests_mirror_the_backend() -> None:
    """The NCA + steady-state request bodies: the client's keys vs the
    pydantic DTOs the route actually defines (TimesConcRequest is the
    shared series base; SteadyStateRequest adds tau/dose/lambda_z)."""
    client = _read(SERVICES / "pkpd" / "client.ts")
    route = _route_module("pkpd.py")
    series_fields = _pydantic_fields(route, "TimesConcRequest")
    assert series_fields == {"times", "conc"}, (
        f"the series DTO drifted: {sorted(series_fields)}"
    )
    ss_fields = _pydantic_fields(route, "SteadyStateRequest")
    assert ss_fields == {"tau", "dose", "lambda_z"}, (
        f"the steady-state DTO drifted: {sorted(ss_fields)}"
    )
    assert _client_request_keys(client, "NcaRequest") == series_fields, (
        "the NCA request keys drifted from TimesConcRequest"
    )
    assert (
        _client_request_keys(client, "SteadyStateRequest")
        == series_fields | ss_fields
    ), "the steady-state request keys drifted from the route DTOs"
    # the generated contract types, never hand-duplicated
    assert 'components["schemas"]["NcaRequest"]' in client, (
        "the NCA request stopped riding the generated type"
    )
    assert "interface NcaRequest" not in client
    assert "interface SteadyStateRequest" not in client
    # the list bounds the client pre-flights: >= 2, <= the route cap
    max_points = re.search(r"_MAX_POINTS\s*=\s*(\d+)", route)
    assert max_points is not None, "pkpd.py lost its _MAX_POINTS pin"
    assert "at least 2 required" in client, (
        "the series pre-flight lost its two-point minimum"
    )
    assert f"over the {max_points.group(1)} cap" in client, (
        "the series pre-flight cap drifted from the route's _MAX_POINTS"
    )
    assert max_points.group(1) == "10000"


def test_model_lane_requests_mirror_the_backend() -> None:
    """The iv-bolus / bateman / superposition request bodies: the
    client's keys vs the route's own pydantic DTOs (Superposition
    extends Bateman with tau + n_doses)."""
    client = _read(SERVICES / "pkpd" / "client.ts")
    route = _route_module("pkpd.py")
    iv = _pydantic_fields(route, "IvBolusRequest")
    bat = _pydantic_fields(route, "BatemanRequest")
    sup = _pydantic_fields(route, "SuperpositionRequest")
    assert iv == {"c0", "ke", "times"}, f"the IV DTO drifted: {sorted(iv)}"
    assert bat == {"dose_per_volume", "ka", "ke", "times"}, (
        f"the bateman DTO drifted: {sorted(bat)}"
    )
    assert sup == {"tau", "n_doses"}, (
        f"the superposition DTO drifted: {sorted(sup)} (it inherits "
        f"the bateman fields)"
    )
    assert _client_request_keys(client, "IvBolusRequest") == iv
    assert _client_request_keys(client, "BatemanRequest") == bat
    assert _client_request_keys(client, "SuperpositionRequest") == bat | sup
    for name in ("IvBolusRequest", "BatemanRequest", "SuperpositionRequest"):
        assert f'components["schemas"]["{name}"]' in client, (
            f"the {name} request stopped riding the generated type"
        )
        assert f"interface {name}" not in client


def test_pbpk_and_pd_requests_mirror_the_backend() -> None:
    """The PBPK + PD request bodies: the client's keys vs the route's
    own pydantic DTOs (the physiology set + the sigmoid Emax
    scalars)."""
    client = _read(SERVICES / "pkpd" / "client.ts")
    route = _route_module("pkpd.py")
    pb = _pydantic_fields(route, "PbpkRequest")
    pd = _pydantic_fields(route, "PdRequest")
    expected_pb = {
        "times", "dose_mg", "route", "ka", "f", "bw", "clint", "fu",
        "bp", "kp_lung", "kp_liver", "kp_kidney", "kp_rest",
    }
    expected_pd = {"concentrations", "e0", "emax", "ec50", "hill"}
    assert pb == expected_pb, f"the PBPK DTO drifted: {sorted(pb)}"
    assert pd == expected_pd, f"the PD DTO drifted: {sorted(pd)}"
    assert _client_request_keys(client, "PbpkRequest") == pb
    assert _client_request_keys(client, "PdRequest") == pd
    for name in ("PbpkRequest", "PdRequest"):
        assert f'components["schemas"]["{name}"]' in client, (
            f"the {name} request stopped riding the generated type"
        )
        assert f"interface {name}" not in client
    # the route literal the PBPK route itself pins
    assert 'route: "iv" | "oral"' in client, (
        "the PBPK route literal stopped being a client union"
    )
    assert re.search(r'route:\s*Literal\["iv",\s*"oral"\]', route) is not None


def test_population_request_and_shapes_mirror_the_backend() -> None:
    """The population lane: the request body's field parity, the exact
    vector shapes the route pins (theta 3, omega_diag 1, the grid
    2…2000), and the job/sync mode literal."""
    client = _read(SERVICES / "pkpd" / "client.ts")
    page = _read(FEATURES / "pkpd" / "PKPDPage.tsx")
    route = _route_module("pkpd.py")
    pop = _pydantic_fields(route, "PopulationRequest")
    expected = {
        "theta", "omega_diag", "times", "n_subjects", "seed",
        "sigma_add", "sigma_prop", "n_profiles", "mode",
    }
    assert pop == expected, f"the population DTO drifted: {sorted(pop)}"
    assert _client_request_keys(client, "PopulationRequest") == pop
    assert 'components["schemas"]["PopulationRequest"]' in client
    assert "interface PopulationRequest" not in client
    # the exact vector shapes, pinned in the route's own Field bounds
    pop_block = _class_block(route, "PopulationRequest")
    theta_len = re.search(
        r"theta:.*?min_length=(\d+),\s*max_length=(\d+)", pop_block, re.DOTALL
    )
    assert theta_len is not None and theta_len.group(1) == "3" == theta_len.group(2), (
        "the route lost its theta exactly-3 pin"
    )
    omega_len = re.search(
        r"omega_diag:.*?min_length=(\d+),\s*max_length=(\d+)", pop_block, re.DOTALL
    )
    assert omega_len is not None and omega_len.group(1) == "1" == omega_len.group(2), (
        "the route lost its omega_diag exactly-1 pin"
    )
    times_len = re.search(
        r"times:.*?min_length=(\d+),\s*max_length=(\d+)", pop_block, re.DOTALL
    )
    assert times_len is not None and times_len.group(1) == "2", (
        "the population grid lost its two-point minimum"
    )
    # the client's own pre-flight mirrors the exact shapes
    assert "exactly 3 required" in client, (
        "the client pre-flight lost its theta exactly-3 mirror"
    )
    assert "exactly 1 required" in client, (
        "the client pre-flight lost its omega exactly-1 mirror"
    )
    assert re.search(r"over the 2000 cap", client) is not None, (
        "the client pre-flight lost the grid cap mirror"
    )
    # the mode lane: job is the route's default; both in the client
    assert re.search(
        r'mode:\s*Literal\["job",\s*"sync"\]\s*=\s*"job"', pop_block
    )
    assert 'mode: "job" | "sync"' in client, (
        "the population mode literal stopped being a client union"
    )
    assert page.count('checked={popMode === mode}') >= 1, (
        "the page lost its job/sync mode radios"
    )


def test_response_view_keys_mirror_the_backend() -> None:
    """The response views: the client's own keys vs the backend's own
    answer dicts — the NCA summary vs PKSummary.as_dict(), the
    interval metrics vs SteadyStateMetrics.as_dict(), the terminal
    block vs the nca route's own answer, the PD view vs effect_table's
    parallel lists, the population body vs _population_body's return,
    the catalog rows vs the catalog route's own."""
    client = _read(SERVICES / "pkpd" / "client.ts")
    route = _route_module("pkpd.py")
    pk_src = _read(PHARMAML / "pharma" / "pk.py")
    nca_ext = _read(PHARMAML / "pkpd" / "nca_ext.py")
    pd_src = _read(PHARMAML / "pharma" / "pd.py")

    # the NCA summary view vs PKSummary.as_dict()'s own keys
    as_dict = re.search(
        r"class PKSummary.*?def as_dict.*?return \{(.*?)\}", pk_src, re.DOTALL
    )
    assert as_dict is not None, "PKSummary.as_dict vanished"
    backend_keys = set(re.findall(r'"(\w+)":', as_dict.group(1)))
    assert _interface_keys(client, "NcaSummaryView") == backend_keys, (
        "the NCA summary view drifted from PKSummary.as_dict()"
    )

    # the interval metrics view vs SteadyStateMetrics.as_dict()'s keys
    ss_dict = re.search(
        r"class SteadyStateMetrics.*?def as_dict.*?return \{(.*?)\}", nca_ext,
        re.DOTALL,
    )
    assert ss_dict is not None, "SteadyStateMetrics.as_dict vanished"
    ss_keys = set(re.findall(r'"(\w+)":', ss_dict.group(1)))
    assert _interface_keys(client, "SteadyStateMetricsView") == ss_keys, (
        "the interval metrics view drifted from SteadyStateMetrics.as_dict()"
    )

    # the terminal block vs the nca route's own answer keys
    terminal_block = re.search(r'"terminal":\s*\{(.*?)\},', route, re.DOTALL)
    assert terminal_block is not None, "the nca route lost its terminal block"
    terminal_keys = set(re.findall(r'"(\w+)":', terminal_block.group(1)))
    assert _interface_keys(client, "TerminalPhaseView") == terminal_keys, (
        "the terminal-phase view drifted from the route's own answer"
    )

    # the PD view vs effect_table's own parallel lists + the route echo
    effect = re.search(
        r'def effect_table.*?return \{(.*?)\}', pd_src, re.DOTALL
    )
    assert effect is not None, "effect_table's answer dict vanished"
    effect_keys = set(re.findall(r'"(\w+)":', effect.group(1)))
    pd_view = _interface_keys(client, "PdResponse")
    assert effect_keys <= pd_view, (
        "the PD view lost effect_table's own parallel lists"
    )
    pd_route_echo = {"e0", "emax", "ec50", "hill", "n_points"}
    assert pd_route_echo <= pd_view, "the PD view lost the route's echo keys"

    # the population body vs _population_body's own return keys
    pop_block_src = re.search(
        r"def _population_body.*?return \{(.*?)\n    \}", route, re.DOTALL
    )
    assert pop_block_src is not None, "_population_body vanished"
    pop_keys = set(
        re.findall(r'^\s{8}"(\w+)":', pop_block_src.group(1), re.MULTILINE)
    )
    assert pop_keys == {
        "n_subjects", "n_times", "seed", "summary", "subject_profiles",
    }, f"the population body keys drifted: {sorted(pop_keys)}"
    body_view = _interface_keys(client, "PopulationBodyView")
    assert pop_keys <= body_view, (
        "the population body view drifted from _population_body's return"
    )
    summary_view = _interface_keys(client, "PopulationSummaryView")
    assert summary_view == {"mean", "p5", "p95"}, (
        "the population summary view drifted from the route's own curves"
    )

    # the catalog rows vs the catalog route's own group dict
    catalog_block = re.search(r'"group":\s*"nca".*?"backend":.*?"[^"]*"', route, re.DOTALL)
    assert catalog_block is not None, "the catalog route lost its group rows"
    group_view = _interface_keys(client, "PkpdCatalogGroup")
    assert group_view == {"group", "routes", "backend"}, (
        "the catalog group view drifted from the route's own row keys"
    )
    assert re.search(r'group.*?routes.*?backend', route, re.DOTALL) is not None


def test_pbpk_columns_mirror_the_backend() -> None:
    """The PBPK frame: the client's row view requires the frame's own
    inserted C_plasma column, and the page renders the compartments
    dynamically off the response — never a hardcoded column list."""
    client = _read(SERVICES / "pkpd" / "client.ts")
    page = _read(FEATURES / "pkpd" / "PKPDPage.tsx")
    pbpk_src = _read(PHARMAML / "pkpd" / "pbpk.py")
    compartments = re.search(
        r"COMPARTMENTS\s*=\s*\(([^)]*)\)", pbpk_src
    )
    assert compartments is not None, "MinimalPBPK.COMPARTMENTS vanished"
    names = [name.strip().strip('"') for name in compartments.group(1).split(",")]
    assert names == ["venous", "lung", "liver", "kidney", "rest"], (
        f"the PBPK compartment tuple drifted: {names}"
    )
    assert re.search(r'frame\["C_plasma"\]', pbpk_src) is not None, (
        "the frame stopped inserting its C_plasma column"
    )
    # the client row view carries the frame's own columns
    row_view = _interface_keys(client, "PbpkRow")
    assert "time" in row_view and "C_plasma" in row_view, (
        "the PBPK row view lost the frame's own columns"
    )
    # the client parser requires every compartment + C_plasma, labeled
    assert "compartment ${compartment} is not a finite number" in client.replace(
        "${compartment}", "${compartment}"
    ) or re.search(r"compartment .* is not a finite number", client), (
        "the PBPK parser stopped requiring the compartment columns"
    )
    assert re.search(r"C_plasma is not a finite number", client) is not None, (
        "the PBPK parser stopped requiring the C_plasma column"
    )
    # the page renders the compartments the response carried
    assert "pbResult.compartments.map" in page, (
        "the PBPK table stopped rendering the response's own compartments"
    )
    assert "venous" not in page, (
        "a hardcoded compartment name crept into the page — the columns "
        "must come from the response, never the client"
    )


def test_figure_lanes_ride_the_artifact_lane() -> None:
    """The Exposure/Response Visualization stage: the concentration
    profile + the sigmoid Emax curve through POST /artifacts/figure
    (the REAL §17 export stack) with their honest two-point minimums,
    riding the generated contract type field-for-field."""
    client = _read(SERVICES / "pkpd" / "client.ts")
    page = _read(FEATURES / "pkpd" / "PKPDPage.tsx")
    assert "exposureFigure" in client and "exposureFigure" in page, (
        "the exposure figure lane vanished"
    )
    assert "responseFigure" in client and "responseFigure" in page, (
        "the response figure lane vanished"
    )
    assert re.search(r"/api/v1/artifacts/figure", client) is not None, (
        "the figures stopped riding the figure artifact lane"
    )
    assert client.count('kind: "scatter"') == 2, (
        "the two pkpd figures stopped being the scatter charts the "
        "concentration/response points carry"
    )
    # the honest minimums, stated before the requests fire
    assert "at least two finite time/concentration points" in client, (
        "the exposure figure lost its two-point minimum"
    )
    assert "at least two finite concentration/effect points" in client, (
        "the response figure lost its two-point minimum"
    )
    # the figure request bodies ride the generated contract types
    assert 'components["schemas"]["FigureRequest"]' in client
    assert "interface FigureRequest" not in client, (
        "the figure requests stopped using the generated contract type"
    )
    artifacts_src = _route_module("artifacts.py")
    figure_fields = _pydantic_fields(artifacts_src, "FigureRequest")
    assert _client_request_keys(client, "FigureRequest") <= figure_fields, (
        "the figure request invented keys artifacts.py never declared"
    )
    kinds = re.search(r"kind:\s*Literal\[([^\]]*)\]", artifacts_src)
    assert kinds is not None and '"scatter"' in kinds.group(1), (
        "the backend figure kinds lost the scatter chart the pkpd lanes ride"
    )
    assert "digest_verified" in page, "the figures lost their digest labels"


def test_export_rides_the_rows_table_lane() -> None:
    """The export stage: rows-based /visualization/table + the backend's
    own 5000-row cap mirrored client-side + the digest-verified labels."""
    page = _read(FEATURES / "pkpd" / "PKPDPage.tsx")
    client = _read(SERVICES / "pkpd" / "client.ts")
    assert "VisualizationTableRequest" in client, (
        "the export stopped using the generated table request type"
    )
    assert "interface VisualizationTableRequest" not in client
    assert 'components["schemas"]["VisualizationTableRequest"]' in client
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
    assert re.search(r"/api/v1/visualization/table", client) is not None, (
        "the export stopped riding the rows-based table lane"
    )
    assert _client_request_keys(
        client, "VisualizationTableRequest"
    ) == _pydantic_fields(viz_src, "VisualizationTableRequest"), (
        "the table request keys drifted from visualization.py's own"
    )
    assert "digest_verified" in page, "the export lost its digest labels"


def test_workspace_flipped_to_workflow() -> None:
    """pkpd: shell → workflow, the App renders the page, and the
    §10.1 registry order is unchanged."""
    workspaces = _read(APP / "workspaces.ts")
    match = re.search(
        r'id:\s*"pkpd",.*?state:\s*"(\w+)"', workspaces, re.DOTALL
    )
    assert match is not None, "the pkpd entry vanished"
    assert match.group(1) == "workflow", (
        "the PK/PD workspace is still a shell after the §11.6 build"
    )
    app = _read(APP / "App.tsx")
    assert 'from "@/features/pkpd/PKPDPage"' in app, (
        "the App stopped importing the PK/PD page"
    )
    assert re.search(r'entry\.id === "pkpd"', app) is not None, (
        "the App lost the pkpd render branch"
    )
    assert (FEATURES / "pkpd" / "PKPDPage.tsx").is_file(), (
        "the PK/PD page file vanished"
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
    """The page's StageHeader sequence = §11.6's fixed workflow order
    (Import Data → Inspect → Select Model → Configure Parameters →
    Fit/Simulate → Diagnostics → Exposure/Response Visualization →
    Export)."""
    page = _read(FEATURES / "pkpd" / "PKPDPage.tsx")
    titles = re.findall(r'<StageHeader\b[^>]*?title="([^"]+)"', page, re.DOTALL)
    expected = [
        "Import Data",
        "Inspect",
        "Select Model",
        "Configure Parameters",
        "Fit/Simulate",
        "Diagnostics",
        "Exposure/Response Visualization",
        "Export",
    ]
    assert titles == expected, (
        f"the §11.6 stage order drifted: {titles}"
    )


def test_honesty_blocks_present() -> None:
    """§12's labels: the backend's own validator as the authority (a
    bad series is its honest 422, surfaced verbatim — never a
    client-side guess), the catalog's no-guess note, the no-fake
    discipline, the native-cancel note, the job/sync lane honesty."""
    page = _read(FEATURES / "pkpd" / "PKPDPage.tsx")
    client = _read(SERVICES / "pkpd" / "client.ts")
    assert re.search(r"honest 422", page) is not None, (
        "the page stopped surfacing the route's honest 422s"
    )
    assert re.search(r"never a client-side guess", page) is not None, (
        "the catalog's no-guess note vanished from the page"
    )
    assert re.search(r"the engine&apos;s own verdict", client) is not None or (
        "verdict" in client
    ), "the client lost its engine-verdict note"
    assert re.search(r"CANCELLED, never FAILED", page) is not None, (
        "the population lane lost its native-cancel note"
    )
    assert re.search(r"never the WS event payload alone", client) is not None, (
        "the client lost the WS-payload-only relay honesty note"
    )
    assert re.search(r"is authoritative", page) is not None, (
        "the job lane lost its REST-authoritative note"
    )
    assert "parsePopulationBody" in page and "parsePopulationBody" in client, (
        "the population REST result stopped being narrowed at runtime"
    )
    # the diagnostics are the backend's own outputs, nothing invented
    assert re.search(r"nothing is invented here", page) is not None, (
        "the diagnostics stage lost its nothing-invented note"
    )
    assert re.search(r"span_low", page) is not None, (
        "the terminal-phase diagnostics stopped rendering the backend's "
        "own span warning"
    )
    # the curve honesty: drawn only from points that actually arrived
    assert re.search(r"points that actually arrived|only from points that", page)
    assert re.search(r"points that actually arrived", client) is not None, (
        "the client lost its real-points figure note"
    )


def test_population_job_lane_rides_the_real_tracker() -> None:
    """The Fit/Simulate population lane rides the REAL job machinery:
    the tracker-driven JobRunCard (WS state + REST reconciliation, the
    cancellation lane), the COMPLETED job's REST result relayed
    through the jobs client (never the WS event payload alone), and
    the WS job.progress envelopes parsed at runtime, job-scoped."""
    page = _read(FEATURES / "pkpd" / "PKPDPage.tsx")
    client = _read(SERVICES / "pkpd" / "client.ts")
    assert "JobRunCard" in page and "useJob" in page, (
        "the population lane stopped riding the tracker's job card"
    )
    assert "jobsApi" in page and ".result(jobId)" in page, (
        "the job lane stopped relaying the authoritative REST result"
    )
    assert "getEventStream()" in page, (
        "the population lane stopped subscribing to the event stream"
    )
    assert 'envelope.event_type === "job.progress"' in page, (
        "the population lane lost its job.progress filter"
    )
    assert "parseJobProgressPayload" in page, (
        "the population telemetry stopped parsing at runtime"
    )
    assert re.search(r"envelope\.job_id !== popJobId", page) is not None, (
        "the population lane lost its job-scoped envelope filter"
    )
    assert 'raw["mode"] === "sync"' in client, (
        "the population lane lost the route's sync branch"
    )
    assert "parseJobProgressPayload" in client, (
        "the client stopped narrowing the job.progress envelopes"
    )
    # the WS is progress-only; the body rides the REST relay
    assert "never the WS event payload alone" in client, (
        "the client lost the WS-is-progress-only honesty note"
    )
    # the population relay narrows the REAL body at runtime
    assert "parsePopulationBody" in client, (
        "the population body parser vanished from the client"
    )
