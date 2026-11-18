"""part11 §11.5, the Molecular Docking workspace row — the drift guards.

The Python gate cannot execute the React page, but every contract the
Docking workspace rides CAN be pinned against the backend source that
defines it, so a backend-side rename turns into a red test before the
frontend drifts:

* the docking client's REST surface vs
  ``pharmaml.api.routes.docking`` plus the shared lanes the page rides
  (the artifacts figure lane, the visualization table lane);
* the request bodies' field parity — the request objects the client
  builds vs the pydantic ``ReceptorPrepareRequest`` /
  ``LigandPdbqtRequest`` / ``ParseTextRequest`` / ``ConsensusRequest``
  / ``LigandEfficiencyRequest`` / ``ContactAtomSpec`` /
  ``ContactsRequest`` / ``PoseRmsdRequest`` / ``DockRequest`` /
  ``ObjectiveRequest`` the route actually defines (§12.2's
  one-DTO-source rule, enforced at the key level — the types come from
  the generated ``api.d.ts``, never hand-duplicated beside it);
* the engine catalog: the row keys the parser keeps vs
  ``_engine_catalog``'s own entry dict, and the response envelope's
  keys vs the ``engines`` route's own answer (platform + the
  DOCKSTRING inventory);
* the DOCKSTRING target catalog: the parser's columns vs
  ``target_frame``'s real columns, and the single-target view keys vs
  ``dockstring_target_view``'s own answer;
* the parse/scoring/contacts view keys vs the route responses
  (``vina_parse``'s pose rows, ``dlg_parse``'s summary + clusters —
  including ``parse_dlg_summary``'s own dict — ``contacts``'s rows +
  PLIF, ``pose_rmsd``'s answer, ``_dock_body``'s rows answer, and the
  receptor report's keys vs ``ProteinReport.as_dict()``);
* the dock request's mode lane (job default / sync), the numeric
  bounds and defaults the page's inputs mirror, and the engine
  literal;
* every ``/api/v1`` literal the page + client carry must exist in the
  exported ``contracts/openapi.json`` (the anti-fabrication guard: no
  call to a route the contract never declared);
* the figure lane: the parsed poses' affinity profile through
  ``POST /artifacts/figure`` with its honest minimum;
* the export lane: rows-based ``POST /visualization/table`` with the
  backend's own 5000-row cap mirrored client-side;
* the workspace registry flip (molecular-docking ``shell`` →
  ``workflow``) + the App wiring + the unchanged §10.1 order;
* the §11.5 stage order (the page's StageHeader sequence) and the
  honesty blocks (the engine boundary rendered from the endpoint's
  own ``boundary_note`` — never a hardcoded client guess, the
  no-fake-results discipline, the native-cancel note, the job/sync
  lane honesty);
* the dock job lane rides the REAL tracker: the JobRunCard +
  ``useJob`` machinery, the COMPLETED job's REST result relayed
  through the jobs client (never the WS event payload alone), and the
  WS ``job.progress`` envelopes parsed at runtime, job-scoped.
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


def _client_object_keys(client_src: str, type_name: str) -> set[str]:
    """The keys of every ``const x: <Type> = { ... }`` literal block."""
    pattern = re.compile(
        r":\s*" + re.escape(type_name) + r"\s*=\s*\{(.*?)\n\s*\};",
        re.DOTALL,
    )
    keys: set[str] = set()
    for match in pattern.finditer(client_src):
        keys |= set(re.findall(r"^\s+(\w+):", match.group(1), re.MULTILINE))
    assert keys, f"the {type_name} literal blocks vanished"
    return keys


def _frontend_api_paths(source: str) -> set[str]:
    """Normalize the literals (query suffixes stripped, interpolations
    become ``{param}``) to the backend's route-path shapes."""
    normalized = source.replace("${encodeURIComponent(name)}", "{name}")
    paths: set[str] = set()
    for match in re.finditer(r"[\"'`](/api/v[0-9]+/[^\"'`?]*)", normalized):
        paths.add(match.group(1))
    return paths


def test_docking_rest_surface_covered() -> None:
    """Every path the docking client carries is a route the backend
    defines; all thirteen docking routes are covered method-for-
    method, and the job surface is never re-declared here."""
    client = _read(SERVICES / "docking" / "client.ts")
    carried = _frontend_api_paths(client)
    assert carried, "the docking client lost every path literal"
    real: dict[str, set[str]] = {}
    for module in ("docking.py", "artifacts.py", "visualization.py"):
        real.update(_backend_routes(module))
    unknown = carried - set(real)
    assert not unknown, (
        f"docking client calls undeclared routes: {sorted(unknown)}"
    )
    for path in (
        "/api/v1/docking/engines",
        "/api/v1/docking/dockstring/targets",
        "/api/v1/docking/dockstring/targets/{name}",
        "/api/v1/docking/dockstring/objective",
        "/api/v1/docking/receptor/prepare",
        "/api/v1/docking/ligand/pdbqt",
        "/api/v1/docking/parse/vina",
        "/api/v1/docking/parse/dlg",
        "/api/v1/docking/scoring/consensus",
        "/api/v1/docking/scoring/ligand-efficiency",
        "/api/v1/docking/contacts",
        "/api/v1/docking/pose-rmsd",
        "/api/v1/docking/dock",
        "/api/v1/artifacts/figure",
        "/api/v1/visualization/table",
    ):
        assert path in carried, f"the workflow stopped calling {path}"
        assert path in real, f"the backend stopped declaring {path}"
    # the whole docking module is carried — no route left behind
    docking_routes = {
        path for path in real if path.startswith("/api/v1/docking/")
    }
    assert docking_routes <= carried, (
        f"the docking client lost backend routes: "
        f"{sorted(docking_routes - carried)}"
    )
    # the dock job lane rides the jobs client + the tracker — never a
    # re-declared job surface here
    assert "/api/v1/jobs" not in carried, (
        "the docking client re-declared the job surface "
        "(the jobs client owns it)"
    )


def test_docking_page_calls_only_contract_routes() -> None:
    """The anti-fabrication guard: every /api/v1 literal in the page +
    the docking client exists in the exported openapi contract."""
    page = _read(FEATURES / "docking" / "DockingPage.tsx")
    client = _read(SERVICES / "docking" / "client.ts")
    carried = _frontend_api_paths(page) | _frontend_api_paths(client)
    assert carried, "the Docking workspace lost every path literal"
    contract = json.loads(_read(CONTRACTS / "openapi.json"))
    declared = set(contract["paths"])
    unknown = carried - declared
    assert not unknown, (
        f"the Docking workspace calls routes the contract never "
        f"declared: {sorted(unknown)}"
    )


def test_receptor_prepare_request_fields_mirror_the_backend() -> None:
    client = _read(SERVICES / "docking" / "client.ts")
    route_src = _route_module("docking.py")
    backend_fields = _pydantic_fields(route_src, "ReceptorPrepareRequest")
    assert backend_fields >= {
        "pdbqt_text", "filename", "remove_waters", "keep_hetatm",
        "chain", "strip_hydrogens",
    }, "the backend ReceptorPrepareRequest changed shape (update this guard + the page)"
    assert _client_request_keys(client, "ReceptorPrepareRequest") == backend_fields
    # §12.2: one DTO source — the request body comes from api.d.ts
    assert 'components["schemas"]["ReceptorPrepareRequest"]' in client
    assert "interface ReceptorPrepareRequest" not in client


def test_ligand_pdbqt_and_parse_requests_mirror_the_backend() -> None:
    client = _read(SERVICES / "docking" / "client.ts")
    route_src = _route_module("docking.py")
    ligand_fields = _pydantic_fields(route_src, "LigandPdbqtRequest")
    assert ligand_fields >= {
        "smiles", "name", "charge", "seed",
    }, "the backend LigandPdbqtRequest changed shape (update this guard + the page)"
    assert _client_request_keys(client, "LigandPdbqtRequest") == ligand_fields
    assert 'components["schemas"]["LigandPdbqtRequest"]' in client
    assert "interface LigandPdbqtRequest" not in client
    # the two pasted-log parse lanes share the one ParseTextRequest
    parse_fields = _pydantic_fields(route_src, "ParseTextRequest")
    assert parse_fields == {"text"}, (
        f"the backend ParseTextRequest changed shape: {sorted(parse_fields)}"
    )
    assert _client_request_keys(client, "ParseTextRequest") == parse_fields
    assert 'components["schemas"]["ParseTextRequest"]' in client
    assert "interface ParseTextRequest" not in client


def test_consensus_and_ligand_efficiency_requests_mirror_the_backend() -> None:
    client = _read(SERVICES / "docking" / "client.ts")
    route_src = _route_module("docking.py")
    consensus_fields = _pydantic_fields(route_src, "ConsensusRequest")
    assert consensus_fields >= {
        "rows", "columns", "weights", "higher_is_better",
    }, "the backend ConsensusRequest changed shape (update this guard + the page)"
    assert _client_request_keys(client, "ConsensusRequest") == consensus_fields
    assert 'components["schemas"]["ConsensusRequest"]' in client
    assert "interface ConsensusRequest" not in client
    efficiency_fields = _pydantic_fields(route_src, "LigandEfficiencyRequest")
    assert efficiency_fields >= {"affinity_kcal_mol", "num_heavy_atoms"}, (
        "the backend LigandEfficiencyRequest changed shape (update this guard + the page)"
    )
    assert (
        _client_request_keys(client, "LigandEfficiencyRequest")
        == efficiency_fields
    )
    assert 'components["schemas"]["LigandEfficiencyRequest"]' in client
    assert "interface LigandEfficiencyRequest" not in client


def test_contacts_and_pose_rmsd_requests_mirror_the_backend() -> None:
    client = _read(SERVICES / "docking" / "client.ts")
    route_src = _route_module("docking.py")
    contacts_fields = _pydantic_fields(route_src, "ContactsRequest")
    assert contacts_fields >= {"ligand", "receptor", "cutoff"}, (
        "the backend ContactsRequest changed shape (update this guard + the page)"
    )
    assert _client_request_keys(client, "ContactsRequest") == contacts_fields
    assert 'components["schemas"]["ContactsRequest"]' in client
    assert "interface ContactsRequest" not in client
    rmsd_fields = _pydantic_fields(route_src, "PoseRmsdRequest")
    assert rmsd_fields >= {"smiles", "coords_ref", "poses"}, (
        "the backend PoseRmsdRequest changed shape (update this guard + the page)"
    )
    assert _client_request_keys(client, "PoseRmsdRequest") == rmsd_fields
    assert 'components["schemas"]["PoseRmsdRequest"]' in client
    assert "interface PoseRmsdRequest" not in client
    # every ContactAtomSpec field the route defines is built verbatim
    atom_fields = _pydantic_fields(route_src, "ContactAtomSpec")
    assert atom_fields >= {
        "serial", "name", "element", "resname", "chain", "resseq",
        "x", "y", "z", "autodock_type",
    }, "the backend ContactAtomSpec changed shape (update this guard + the page)"
    assert _client_object_keys(client, "ContactAtomSpec") == atom_fields
    assert 'components["schemas"]["ContactAtomSpec"]' in client
    assert "interface ContactAtomSpec" not in client


def test_dock_request_fields_mirror_the_backend() -> None:
    client = _read(SERVICES / "docking" / "client.ts")
    page = _read(FEATURES / "docking" / "DockingPage.tsx")
    route_src = _route_module("docking.py")
    backend_fields = _pydantic_fields(route_src, "DockRequest")
    assert backend_fields >= {
        "smiles", "receptor_pdbqt", "engine", "center", "size",
        "exhaustiveness", "num_modes", "workers", "seed", "timeout_s",
        "mode",
    }, "the backend DockRequest changed shape (update this guard + the page)"
    assert _client_request_keys(client, "DockRequest") == backend_fields
    assert 'components["schemas"]["DockRequest"]' in client
    assert "interface DockRequest" not in client
    # the mode lane: job is the default, sync the route's other literal
    assert 'mode: options.mode ?? "job"' in client, (
        "the dock lane stopped defaulting to job mode"
    )
    assert 'raw["mode"] === "sync"' in client, (
        "the dock lane lost the route's sync branch"
    )
    assert '<option value="job">' in page and '<option value="sync">' in page, (
        "the page lost the job/sync mode toggle"
    )
    # the engine literal is the route's own
    engine = re.search(r'engine:\s*Literal\["(\w+)",\s*"(\w+)"\]', route_src)
    assert engine is not None, "the engine literal vanished from the route"
    assert (engine.group(1), engine.group(2)) == ("vina", "autodock4"), (
        "the engine vocabulary moved (update this guard + the page)"
    )
    assert 'engine: "vina" | "autodock4"' in client, (
        "the client's engine vocabulary drifted from the route's literal"
    )
    # the numeric bounds the page's labels mirror, straight off the route
    label_bounds: dict[str, tuple[str, str]] = {}
    for field, label in (
        ("exhaustiveness", "exhaustiveness"),
        ("num_modes", "num modes"),
        ("workers", "workers"),
    ):
        match = re.search(
            rf"{field}:\s*int\s*=\s*Field\(default=(\d+),\s*ge=(\d+),\s*le=(\d+)\)",
            route_src,
        )
        assert match is not None, f"the {field} bounds vanished from the route"
        label_bounds[label] = (match.group(2), match.group(3))
    timeout = re.search(
        r"timeout_s:\s*float\s*=\s*Field\(default=([\d.]+),\s*gt=([\d.]+),\s*le=([\d.]+)\)",
        route_src,
    )
    assert timeout is not None, "the timeout bounds vanished from the route"
    timeout_le = float(timeout.group(3))
    timeout_le_label = (
        str(int(timeout_le)) if timeout_le.is_integer() else str(timeout_le)
    )
    label_bounds["timeout s"] = ("0", timeout_le_label)
    for label, (ge, le) in label_bounds.items():
        assert f"{label} ({ge}–{le})" in page, (
            f"the page's {label} bound drifted from the route's ge/le ({ge}/{le})"
        )
    # the box defaults the page's grid inputs carry, verbatim
    center = re.search(
        r"center:\s*list\[float\]\s*=\s*Field\(\s*default=\[([^\]]*)\]",
        route_src,
        re.DOTALL,
    )
    size = re.search(
        r"size:\s*list\[float\]\s*=\s*Field\(\s*default=\[([^\]]*)\]",
        route_src,
        re.DOTALL,
    )
    assert center is not None and size is not None, (
        "the box defaults vanished from the route"
    )
    assert [v.strip() for v in center.group(1).split(",")] == [
        "0.0", "0.0", "0.0",
    ], "the route's center default moved (update this guard + the page)"
    assert [v.strip() for v in size.group(1).split(",")] == [
        "22.5", "22.5", "22.5",
    ], "the route's size default moved (update this guard + the page)"
    assert "center 0, 0, 0" in page and "size 22.5, 22.5, 22.5" in page, (
        "the page stopped stating the route's own box defaults"
    )
    assert page.count("useState(0)") >= 3, (
        "the page's center inputs stopped mirroring the route default"
    )
    assert page.count("useState(22.5)") >= 3, (
        "the page's size inputs stopped mirroring the route default"
    )
    # the scalar defaults the page's inputs carry
    for field, default in (
        ("exhaustiveness", "8"),
        ("num_modes", "9"),
        ("workers", "1"),
        ("seed", "0"),
        ("timeout_s", "3600"),
    ):
        route_default = re.search(
            rf"{field}:\s*(?:int|float)\s*=\s*Field\(default=([\d.]+)", route_src
        )
        assert route_default is not None, f"the {field} default vanished"
        assert float(route_default.group(1)) == float(default), (
            f"the route's {field} default moved (update this guard + the page)"
        )
        assert f"useState({default})" in page, (
            f"the page's {field} input drifted from the route default {default}"
        )


def test_engine_catalog_fields_mirror_the_backend() -> None:
    """The mirrored engine catalog rows keep every key
    ``_engine_catalog`` emits, and the response envelope keeps the
    ``engines`` route's own answer keys."""
    client = _read(SERVICES / "docking" / "client.ts")
    page = _read(FEATURES / "docking" / "DockingPage.tsx")
    route_src = _route_module("docking.py")
    entry = re.search(
        r"entry:\s*dict\[str,\s*Any\]\s*=\s*\{(.*?)\}", route_src, re.DOTALL
    )
    assert entry is not None, "the engine catalog entry dict vanished"
    entry_keys = set(re.findall(r'"([a-z_]+)":', entry.group(1)))
    assert re.search(r'entry\["boundary_note"\]', route_src) is not None, (
        "the engine catalog lost its boundary_note assignment"
    )
    entry_keys.add("boundary_note")
    assert entry_keys == {
        "name", "bundled", "binary", "launchable", "boundary_note",
    }, f"the engine catalog entry changed shape: {sorted(entry_keys)}"
    for key in entry_keys:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"the catalog parser dropped the '{key}' key the route emits"
        )
    # the response envelope: engines + platform + the dockstring inventory
    engines_block = re.search(
        r'@router\.get\("/engines"\).*?return\s*\{(.*?)\n    \}',
        route_src,
        re.DOTALL,
    )
    assert engines_block is not None, "the engines answer block vanished"
    emitted = set(re.findall(r'"([a-z_]+)":', engines_block.group(1)))
    assert emitted == {
        "engines", "platform", "dockstring", "n_targets", "objectives",
    }, f"the engines answer changed shape: {sorted(emitted)}"
    for key in ("engines", "platform", "dockstring"):
        assert re.search(r'r\["' + key + r'"\]', client) is not None, (
            f"the engines parser dropped the '{key}' key the route emits"
        )
    for key in ("n_targets", "objectives"):
        assert re.search(r'dockstring\["' + key + r'"\]', client) is not None, (
            f"the inventory parser dropped the '{key}' key the route emits"
        )
    # the page renders the endpoint's own boundary verdict — never a
    # hardcoded client guess
    assert ".boundary_note" in page, (
        "the engine stage stopped rendering the route's boundary_note"
    )


def test_dockstring_target_fields_mirror_the_backend() -> None:
    """The target-catalog parser keeps every column ``target_frame``
    emits, and the single-target view keys match the route's answer."""
    client = _read(SERVICES / "docking" / "client.ts")
    route_src = _route_module("docking.py")
    dockstring_src = _read(PHARMAML / "docking" / "dockstring.py")
    frame = re.search(
        r"def target_frame\(\).*?\{(.*?)\}\s*for\s+target",
        dockstring_src,
        re.DOTALL,
    )
    assert frame is not None, "the target_frame block vanished"
    columns = set(re.findall(r'"([a-z_]+)":', frame.group(1)))
    assert columns == {
        "target", "center_x", "center_y", "center_z",
        "size_x", "size_y", "size_z",
    }, f"the target frame columns changed: {sorted(columns)}"
    for key in columns:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"the catalog parser dropped the '{key}' column target_frame emits"
        )
    view_block = re.search(
        r"def dockstring_target_view.*?return\s*\{(.*?)\n    \}",
        route_src,
        re.DOTALL,
    )
    assert view_block is not None, "the single-target view block vanished"
    view_keys = set(re.findall(r'"([a-z_]+)":', view_block.group(1)))
    assert view_keys == {"name", "center", "size", "config"}, (
        f"the single-target view changed shape: {sorted(view_keys)}"
    )
    for key in view_keys:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"the view parser dropped the '{key}' key the route emits"
        )
    # the benchmark objective lane's request rides the same rule
    objective_fields = _pydantic_fields(route_src, "ObjectiveRequest")
    assert objective_fields >= {"name", "scores", "qed"}, (
        "the backend ObjectiveRequest changed shape (update this guard + the page)"
    )
    assert _client_request_keys(client, "ObjectiveRequest") == objective_fields
    assert 'components["schemas"]["ObjectiveRequest"]' in client
    assert "interface ObjectiveRequest" not in client
    # the objectives ride the live engines response — not a mirrored
    # hardcoded vocabulary (the route's own 422 names the real ones)
    assert "OBJECTIVE_KINDS" not in client, (
        "the docking client hardcoded an objective vocabulary — the "
        "names come from the engines response, never a mirror"
    )


def test_pose_cluster_contacts_view_fields_mirror_the_backend() -> None:
    """The parse/scoring/contacts view parsers keep every key the
    route responses actually carry (pose rows, the DLG summary + its
    cluster rows, the contacts rows + PLIF, the pose-rmsd answer, the
    dock rows body, the receptor report)."""
    client = _read(SERVICES / "docking" / "client.ts")
    route_src = _route_module("docking.py")
    parsing_src = _read(PHARMAML / "docking" / "parsing.py")

    vina_block = re.search(
        r"def vina_parse.*?return\s*\{(.*?)\n    \}", route_src, re.DOTALL
    )
    assert vina_block is not None, "the vina parse answer block vanished"
    vina_keys = set(re.findall(r'"([a-z_]+)":', vina_block.group(1)))
    assert vina_keys == {
        "n_poses", "poses", "mode", "affinity", "rmsd_ub", "rmsd_lb",
    }, f"the vina parse answer changed shape: {sorted(vina_keys)}"
    for key in vina_keys:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"the vina parser dropped the '{key}' key the route emits"
        )

    dlg_block = re.search(
        r"def dlg_parse.*?return\s*\{(.*?)\n    \}", route_src, re.DOTALL
    )
    assert dlg_block is not None, "the dlg parse answer block vanished"
    dlg_keys = set(re.findall(r'"([a-z_]+)":', dlg_block.group(1)))
    assert dlg_keys == {
        "summary", "clusters", "rank", "lowest_energy", "mean_energy",
        "num_in_cluster",
    }, f"the dlg parse answer changed shape: {sorted(dlg_keys)}"
    for key in dlg_keys:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"the dlg parser dropped the '{key}' key the route emits"
        )
    summary_block = re.search(
        r"def parse_dlg_summary.*?return\s*\{(.*?)\n    \}",
        parsing_src,
        re.DOTALL,
    )
    assert summary_block is not None, "the DLG summary dict vanished"
    summary_keys = set(re.findall(r'"([a-z_]+)":', summary_block.group(1)))
    assert summary_keys == {
        "n_clusters", "top_rank", "top_lowest_energy", "top_mean_energy",
        "top_num_in_cluster", "best_rank", "best_lowest_energy",
        "best_mean_energy", "best_num_in_cluster",
    }, f"the DLG summary dict changed shape: {sorted(summary_keys)}"
    for key in summary_keys:
        pattern = r'summary\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"the summary parser dropped the '{key}' key the parser emits"
        )

    contacts_block = re.search(
        r"def contacts\(.*?return\s*\{(.*?)\n    \}", route_src, re.DOTALL
    )
    assert contacts_block is not None, "the contacts answer block vanished"
    contacts_keys = set(re.findall(r'"([a-z_]+)":', contacts_block.group(1)))
    assert contacts_keys == {
        "cutoff", "n_contacts", "interaction_classes", "contacts", "plif",
        "ligand_serial", "ligand_atom", "receptor_residue", "receptor_atom",
        "distance", "interaction",
    }, f"the contacts answer changed shape: {sorted(contacts_keys)}"
    for key in contacts_keys:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"the contacts parser dropped the '{key}' key the route emits"
        )
    # the PLIF rows: residue + total (the class columns ride
    # interaction_classes, the answer's own legend)
    assert re.search(r'r\["residue"\]', client) is not None, (
        "the PLIF parser dropped the residue key"
    )
    assert re.search(r'r\["total"\]', client) is not None, (
        "the PLIF parser dropped the total key"
    )

    rmsd_block = re.search(
        r"def pose_rmsd\(.*?return\s*\{(.*?)\n    \}", route_src, re.DOTALL
    )
    assert rmsd_block is not None, "the pose-rmsd answer block vanished"
    rmsd_keys = set(re.findall(r'"([a-z_]+)":', rmsd_block.group(1)))
    assert rmsd_keys == {"smiles", "n_atoms", "n_poses", "rmsd"}, (
        f"the pose-rmsd answer changed shape: {sorted(rmsd_keys)}"
    )
    for key in rmsd_keys:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"the pose-rmsd parser dropped the '{key}' key the route emits"
        )

    dock_block = re.search(
        r"def _dock_body.*?return\s*\{(.*?)\n    \}", route_src, re.DOTALL
    )
    assert dock_block is not None, "the dock rows answer block vanished"
    dock_keys = set(re.findall(r'"([a-z_]+)":', dock_block.group(1)))
    assert dock_keys == {"n_ligands", "n_ok", "n_failed", "rows"}, (
        f"the dock rows answer changed shape: {sorted(dock_keys)}"
    )
    for key in dock_keys:
        pattern = r'r\["' + re.escape(key) + r'"\]'
        assert re.search(pattern, client) is not None, (
            f"parseDockRows dropped the '{key}' key the route emits"
        )

    # the receptor report's displayed keys ⊆ ProteinReport.as_dict()
    receptor_src = _read(PHARMAML / "docking" / "receptor.py")
    report_block = re.search(
        r"def as_dict\(self\).*?return\s*\{(.*?)\n        \}",
        receptor_src,
        re.DOTALL,
    )
    assert report_block is not None, "ProteinReport.as_dict() vanished"
    report_keys = set(re.findall(r'"([a-z_]+)":', report_block.group(1)))
    parser_block = re.search(
        r"function parseReceptorReport\(.*?\n\}", client, re.DOTALL
    )
    assert parser_block is not None, "parseReceptorReport vanished"
    client_keys = set(
        re.findall(r'r\["([a-z_]+)"\]', parser_block.group(0))
    )
    assert client_keys <= report_keys, (
        f"the receptor report parser invented keys the audit never "
        f"emits: {sorted(client_keys - report_keys)}"
    )


def test_figure_lanes_ride_the_artifact_lane() -> None:
    """The Visualize stage: the parsed poses' REAL affinity profile
    (a bar per mode) through POST /artifacts/figure (the REAL §17
    export stack) with its honest minimum, riding the generated
    contract type field-for-field."""
    client = _read(SERVICES / "docking" / "client.ts")
    page = _read(FEATURES / "docking" / "DockingPage.tsx")
    assert "poseAffinityFigure" in client and "poseAffinityFigure" in page, (
        "the pose affinity figure lane vanished"
    )
    assert re.search(r"/api/v1/artifacts/figure", client) is not None, (
        "the figure stopped riding the figure artifact lane"
    )
    # the natural pose chart: one bar per parsed mode
    assert 'kind: "bar"' in client, (
        "the pose figure stopped being the bar chart the poses carry"
    )
    # the honest minimum, stated before the request fires
    assert re.search(r"at least one parsed pose", client) is not None, (
        "the figure lane lost its one-pose minimum"
    )
    # the figure request bodies ride the generated contract types
    assert 'components["schemas"]["FigureRequest"]' in client
    assert "interface FigureRequest" not in client, (
        "the figure requests stopped using the generated contract type"
    )
    # the shorthand props (x, y, color, format) ride TS's own field
    # checking — the named keys must never invent one the route lacks
    artifacts_src = _route_module("artifacts.py")
    figure_fields = _pydantic_fields(artifacts_src, "FigureRequest")
    assert _client_request_keys(client, "FigureRequest") <= figure_fields, (
        "the figure request invented keys artifacts.py never declared"
    )
    kinds = re.search(r"kind:\s*Literal\[([^\]]*)\]", artifacts_src)
    assert kinds is not None and '"bar"' in kinds.group(1), (
        "the backend figure kinds lost the bar chart the pose lane rides"
    )
    assert "digest_verified" in page, "the figures lost their digest labels"


def test_export_rides_the_rows_table_lane() -> None:
    """The export stage: rows-based /visualization/table + the backend's
    own 5000-row cap mirrored client-side + the digest-verified labels."""
    page = _read(FEATURES / "docking" / "DockingPage.tsx")
    client = _read(SERVICES / "docking" / "client.ts")
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
    """molecular-docking: shell → workflow, the App renders the page,
    and the §10.1 registry order is unchanged."""
    workspaces = _read(APP / "workspaces.ts")
    match = re.search(
        r'id:\s*"molecular-docking",.*?state:\s*"(\w+)"', workspaces,
        re.DOTALL,
    )
    assert match is not None, "the molecular-docking entry vanished"
    assert match.group(1) == "workflow", (
        "the Docking workspace is still a shell after the §11.5 build"
    )
    app = _read(APP / "App.tsx")
    assert 'from "@/features/docking/DockingPage"' in app, (
        "the App stopped importing the Docking page"
    )
    assert re.search(r'entry\.id === "molecular-docking"', app) is not None, (
        "the App lost the molecular-docking render branch"
    )
    assert (FEATURES / "docking" / "DockingPage.tsx").is_file(), (
        "the Docking page file vanished"
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
    """The page's StageHeader sequence = §11.5's fixed workflow order."""
    page = _read(FEATURES / "docking" / "DockingPage.tsx")
    titles = re.findall(r'<StageHeader\b[^>]*?title="([^"]+)"', page, re.DOTALL)
    expected = [
        "Select Ligands",
        "Prepare Ligands",
        "Prepare Receptor",
        "Define Binding Site/Grid",
        "Configure Engine",
        "Run Docking",
        "Inspect Poses",
        "Analyze Interactions",
        "Visualize",
        "Export",
    ]
    assert titles == expected, (
        "the §11.5 stage order drifted "
        f"(got {titles}, expected {expected})"
    )


def test_honesty_blocks_present() -> None:
    """§12's labels: the engine-boundary honesty (the endpoint's own
    boundary note, never a hardcoded client guess — the dock button
    still submits and the route's 422 surfaces as the stage's error
    state), the no-fake-results discipline, the native-cancel note,
    the job/sync lane honesty."""
    page = _read(FEATURES / "docking" / "DockingPage.tsx")
    client = _read(SERVICES / "docking" / "client.ts")
    assert ".boundary_note" in page, (
        "the engine boundary stopped riding the endpoint's own note"
    )
    assert "launchable" in page, (
        "the engine stage stopped stating the launchable verdict"
    )
    assert "Windows" not in page, (
        "a hardcoded engine-boundary string crept in — the note must "
        "come from the engines endpoint, never the client"
    )
    assert re.search(r"never faked", page) is not None, (
        "the no-fake-results discipline vanished from the page"
    )
    assert "never faked" in client, (
        "the client lost its no-fake-results note"
    )
    assert re.search(r"honest 422", page) is not None, (
        "the dock lane stopped surfacing the route's honest 422"
    )
    assert re.search(r"CANCELLED, never FAILED", page) is not None, (
        "the dock lane lost its native-cancel note"
    )
    assert re.search(r"no pretend job tracking", page) is not None, (
        "the sync lane lost its honest no-pretend-tracking note"
    )
    assert "REST-authoritative" in page, (
        "the job lane lost its REST-authoritative note"
    )
    assert "never the WS event payload" in client, (
        "the client lost the WS-payload-only relay honesty note"
    )


def test_dock_job_lane_rides_the_real_tracker() -> None:
    """The Run Docking stage rides the REAL job machinery: the
    tracker-driven JobRunCard (WS state + REST reconciliation, the
    cancellation lane), the COMPLETED job's REST result relayed
    through the jobs client (never the WS event payload alone), and
    the WS job.progress envelopes parsed at runtime, job-scoped."""
    page = _read(FEATURES / "docking" / "DockingPage.tsx")
    client = _read(SERVICES / "docking" / "client.ts")
    assert "JobRunCard" in page and "useJob" in page, (
        "the dock lane stopped riding the tracker's job card"
    )
    assert "jobsApi" in page and ".result(jobId)" in page, (
        "the job lane stopped relaying the authoritative REST result"
    )
    assert "getEventStream()" in page, (
        "the dock lane stopped subscribing to the event stream"
    )
    assert 'envelope.event_type === "job.progress"' in page, (
        "the dock lane lost its job.progress filter"
    )
    assert "parseJobProgressPayload" in page, (
        "the dock telemetry stopped parsing at runtime"
    )
    assert re.search(r"envelope\.job_id !== dockJobId", page) is not None, (
        "the dock lane lost its job-scoped envelope filter"
    )
    assert "parseDockRows" in page and "parseDockRows" in client, (
        "the REST result stopped being narrowed by parseDockRows"
    )
    assert 'raw["mode"] === "sync"' in client, (
        "the dock lane lost the route's sync branch"
    )
    # the WS is progress-only; the rows ride the REST relay
    assert "never the WS event payload" in client, (
        "the client lost the WS-is-progress-only honesty note"
    )
