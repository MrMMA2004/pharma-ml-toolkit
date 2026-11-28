"""part11 Phase Three, services part two — the service-layer drift guards.

The frontend SPA cannot be executed by the Python gate, but every
contract the four new services ride CAN be pinned against the backend
source that defines it — these guards parse the REAL route modules and
turn a backend-side rename into a red test before the frontend drifts:

* the WS client's socket path, token/since params, hello frame keys and
  ``ping`` text vs ``pharmaml.api.routes.events``;
* the event vocabulary vs ``pharmaml.api.events.EventType`` (the
  round-279 counter claimed this guard; it ships here — no claim
  without the test);
* the job REST surface + ``JobRecord.public()`` fields + the JobState
  vocabulary vs ``pharmaml.api.routes.jobs`` / ``pharmaml.api.jobs``;
* the dataset upload multipart field names (Form/File parameter names)
  + the full route surface + the generated-type-only request bodies vs
  ``pharmaml.api.routes.datasets`` (§12.2: the requests come from the
  generated ``api.d.ts``, never hand-duplicated);
* the artifact registry/download surface + the token-header raw lane +
  the digest header vs ``pharmaml.api.routes.artifacts``;
* §13's client-side guarantees: duplicate-event protection, the
  replay cursor, heartbeats/liveness, reconnection backoff, graceful
  shutdown, and the tracker's REST-authoritative reconciliation.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SERVICES = FRONTEND / "src" / "services"
HOOKS = FRONTEND / "src" / "hooks"
API = ROOT / "src" / "pharmaml" / "api"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _route_module(name: str) -> str:
    return _read(API / "routes" / name)


def _api_prefix() -> str:
    """``/api/v1`` — the versioned mount every router lives under."""
    src = _read(API / "routes" / "__init__.py")
    match = re.search(r'api_router\s*=\s*APIRouter\(prefix="(/api/v[0-9]+)"', src)
    assert match is not None, "the api_router prefix literal vanished"
    return match.group(1)


def _router_prefix(module_src: str) -> str:
    match = re.search(r'APIRouter\(prefix="([^"]+)"', module_src)
    assert match is not None, "a route module lost its APIRouter prefix"
    return match.group(1)


def _backend_routes(module_name: str) -> dict[str, set[str]]:
    """``{route_path: {methods}}`` for one routes module (path → set)."""
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


def _frontend_path_set(source: str) -> set[str]:
    """Normalize the client's literals (quoted + backticked) to path shapes."""
    normalized = source.replace("${encodeURIComponent(jobId)}", "{job_id}")
    normalized = normalized.replace("${encodeURIComponent(datasetId)}", "{dataset_id}")
    normalized = normalized.replace("${encodeURIComponent(artifactId)}", "{artifact_id}")
    normalized = normalized.replace("${encodeURIComponent(column)}", "{column}")
    normalized = normalized.replace("${encodeURIComponent(sampleId)}", "{sample_id}")
    # quoted strings AND backtick template literals carry the paths
    hits = re.findall(r"/api/v1/[A-Za-z0-9_/{}\-]+", normalized)
    # strip query strings the client appends (?limit=…, ?workspace=…)
    return {h.split("?", 1)[0] for h in hits}


# -- the event vocabulary -------------------------------------------------


def test_event_vocabulary_mirrors_the_backend() -> None:
    backend = _read(API / "events.py")
    enum_src = backend.split("class EventType", 1)[1].split("class ", 1)[0]
    backend_values = re.findall(r'=\s*"([^"]+)"', enum_src)
    assert len(backend_values) >= 20, "the backend vocabulary shrank?"

    frontend = _read(SERVICES / "websocket" / "events.ts")
    block = frontend.split("export const EVENT_TYPES", 1)[1]
    block = block.split("] as const;", 1)[0]
    frontend_values = re.findall(r'"([^"]+)"', block)
    assert frontend_values == backend_values, (
        "frontend EVENT_TYPES drifted from pharmaml.api.events.EventType: "
        f"frontend={frontend_values} backend={backend_values}"
    )


# -- the WebSocket socket -------------------------------------------------


def test_ws_client_path_params_and_ping_match_the_backend() -> None:
    events_src = _route_module("events.py")
    prefix = _api_prefix() + _router_prefix(events_src)
    assert prefix == "/api/v1/events", f"the socket route moved: {prefix}"

    client = _read(SERVICES / "websocket" / "client.ts")
    assert '"/api/v1/events"' in client, (
        "the WS client must open the backend's socket route"
    )
    # the backend reads the token + the replay cursor from query_params
    assert 'searchParams.set("token"' in client
    assert 'searchParams.set("since"' in client
    # the protocol's keepalive text (the backend answers a literal ping)
    assert '"ping"' in client


def test_hello_frame_contract_matches_the_backend_welcome() -> None:
    events_src = _route_module("events.py")
    hello_block = events_src.split("hello = EventEnvelope(", 1)[1]
    hello_block = hello_block.split(")", 1)[0]
    backend_keys = set(re.findall(r'"([a-z_]+)":', hello_block))
    assert backend_keys == {"hello", "last_sequence", "replayed"}, (
        f"the backend hello payload changed: {sorted(backend_keys)}"
    )

    events_ts = _read(SERVICES / "websocket" / "events.ts")
    block = events_ts.split("export interface HelloPayload", 1)[1]
    block = block.split("}", 1)[0]
    frontend_keys = set(re.findall(r"readonly\s+([a-z_]+):", block))
    assert frontend_keys == backend_keys, (
        f"HelloPayload drifted: frontend={sorted(frontend_keys)} "
        f"backend={sorted(backend_keys)}"
    )
    # the hello is an ordinary heartbeat envelope distinguished by its
    # payload — never a separate message shape
    assert 'envelope.payload["hello"] === true' in events_ts


def test_ws_client_implements_the_thirteen_guarantees() -> None:
    client = _read(SERVICES / "websocket" / "client.ts")
    # duplicate-event protection (client-side cursor)
    assert "envelope.sequence <= this._lastSequence" in client
    # heartbeats / dead-peer detection (§13: the backend beats every 15 s)
    assert "heartbeatTimeoutMs" in client
    # reconnection with bounded backoff
    assert "maxReconnectAttempts" in client
    assert "reconnectBaseMs" in client
    assert "2 ** this._attempts" in client
    # graceful shutdown: the backend's shutdown envelope never reconnects
    assert "shutdownSeen" in client
    # the REST-authoritative rule is the documented contract
    assert "authoritative" in client


def test_ws_client_treats_policy_close_codes_as_terminal() -> None:
    """part15 §4.2 (D-6), the frontend lane (r341).

    The backend refuses untrusted Origins (close 4403, before accept,
    valid token included) and refused tokens (close 4401); both are
    PERMANENT — the browser's Origin and the same memory-only token
    cannot change between attempts — so the client must go terminal
    with a surfaced cause instead of a reconnect storm. This guard
    pins the code pair on BOTH sides of the wire so either drifting
    (a backend code change, a client policy-table change) trips it.
    """
    events_src = _route_module("events.py")
    assert "code=4403" in events_src, "the D-6 origin-policy close moved"
    assert "code=4401" in events_src, "the token-refusal close moved"

    client = _read(SERVICES / "websocket" / "client.ts")
    # the terminal-reason lane: the named cause surfaced on the status
    assert "terminalReason" in client
    # the policy close codes held in the client's own table
    assert "4401: \"session token rejected (close 4401)\"" in client
    assert "4403: \"origin policy rejection (close 4403)\"" in client
    # terminal handling precedes the reconnect scheduler
    terminal_block = client.split("POLICY_CLOSE_REASONS[event.code]", 1)[1]
    assert "scheduleReconnect" in terminal_block.split("private scheduleReconnect", 1)[0], (
        "the policy lane must reach its terminal return BEFORE the "
        "reconnect scheduler (a 4401/4403 retry is a guaranteed storm)"
    )
    # the surfacing: the shell banner names the terminal cause
    gate = _read(FRONTEND / "src" / "app" / "StartupGate.tsx")
    assert "ws.terminalReason" in gate, "the banner must name the policy cause"


def test_tracker_reconciles_through_rest() -> None:
    tracker = _read(SERVICES / "jobs" / "tracker.ts")
    # §13: recovery of job status after reconnection
    assert "refreshNonTerminal" in tracker
    assert '"open"' in tracker
    # §13 backpressure: bus.overflow -> REST is authoritative
    assert '"bus.overflow"' in tracker
    # the periodic authoritative reconciliation
    assert "pollMs" in tracker
    assert "jobsApi.status" in tracker


# -- the job surface ------------------------------------------------------


def test_job_rest_surface_covered() -> None:
    backend = _backend_routes("jobs.py")
    expected = {
        "/api/v1/jobs": {"GET"},
        "/api/v1/jobs/{job_id}": {"GET"},
        "/api/v1/jobs/{job_id}/result": {"GET"},
        "/api/v1/jobs/{job_id}/cancel": {"POST"},
    }
    assert backend == expected, f"the backend job routes moved: {backend}"

    client = _read(SERVICES / "jobs" / "client.ts")
    frontend_paths = _frontend_path_set(client)
    missing = expected.keys() - frontend_paths
    assert not missing, (
        f"the jobs client lost backend routes: {sorted(missing)} "
        f"(has {sorted(frontend_paths)})"
    )


def test_job_state_vocabulary_mirrors_the_backend() -> None:
    jobs_src = _read(API / "jobs.py")
    enum_src = jobs_src.split("class JobState", 1)[1].split("class ", 1)[0]
    backend_states = re.findall(r'=\s*"([A-Z_]+)"', enum_src)

    types_ts = _read(SERVICES / "jobs" / "types.ts")
    block = types_ts.split("export const JOB_STATES", 1)[1]
    block = block.split("] as const;", 1)[0]
    frontend_states = re.findall(r'"([A-Z_]+)"', block)
    assert frontend_states == backend_states, (
        f"JOB_STATES drifted: frontend={frontend_states} "
        f"backend={backend_states}"
    )


def test_job_view_fields_mirror_the_backend_public() -> None:
    jobs_src = _read(API / "jobs.py")
    public_block = jobs_src.split("def public(self)", 1)[1]
    public_block = public_block.split("}", 1)[0]
    backend_fields = set(re.findall(r'"([a-z_]+)":', public_block))

    types_ts = _read(SERVICES / "jobs" / "types.ts")
    block = types_ts.split("export interface JobView", 1)[1]
    block = block.split("}", 1)[0]
    frontend_fields = set(re.findall(r"readonly\s+([a-z_]+)\??:", block))
    assert frontend_fields == backend_fields, (
        f"JobView drifted: frontend-only={sorted(frontend_fields - backend_fields)} "
        f"backend-only={sorted(backend_fields - frontend_fields)}"
    )

    # the route-added wrappers must keep their documented flags
    status_block = types_ts.split("export interface JobStatusView", 1)[1]
    status_block = status_block.split("}", 1)[0]
    assert "result_ready: boolean" in status_block
    cancel_block = types_ts.split("export interface JobCancelledView", 1)[1]
    cancel_block = cancel_block.split("}", 1)[0]
    assert "cancel_acknowledged: true" in cancel_block


# -- the dataset surface --------------------------------------------------


def test_dataset_rest_surface_covered() -> None:
    backend = _backend_routes("datasets.py")
    # the §15.4 quality lane's route (r351) lives in its own service
    # module — services/quality/client.ts — not the datasets client;
    # the expected set keeps it separated so each client's coverage
    # stays pinned against the routes it owns.
    quality_lane = {"/api/v1/datasets/{dataset_id}/quality": {"POST"}}
    expected = {
        "/api/v1/datasets": {"POST", "GET"},
        "/api/v1/datasets/{dataset_id}": {"GET", "DELETE"},
        "/api/v1/datasets/{dataset_id}/preview": {"GET"},
        "/api/v1/datasets/{dataset_id}/profile": {"POST"},
        "/api/v1/datasets/{dataset_id}/split": {"POST"},
        # the §11.1 chemistry-stage bridge (bounded by the chemistry
        # engine's own request cap — see routes/datasets.py)
        "/api/v1/datasets/{dataset_id}/column/{column}": {"GET"},
    }
    assert backend == expected | quality_lane, (
        f"the backend dataset routes moved: {backend}"
    )

    client = _read(SERVICES / "datasets" / "client.ts")
    frontend_paths = _frontend_path_set(client)
    missing = expected.keys() - frontend_paths
    assert not missing, (
        f"the datasets client lost backend routes: {sorted(missing)} "
        f"(has {sorted(frontend_paths)})"
    )
    # the quality lane's own client covers its route
    quality_paths = _frontend_path_set(
        _read(SERVICES / "quality" / "client.ts")
    )
    assert quality_lane.keys() <= quality_paths, (
        f"the quality client lost its route: {sorted(quality_lane.keys())} "
        f"not in {sorted(quality_paths)}"
    )


def test_sample_rest_surface_covered() -> None:
    """part16 §8 (Phase 5): the demo-data lane's route surface is pinned.

    The backend module is parsed for its real route table; the samples
    client must carry every route (and nothing the backend lost).
    """
    backend = _backend_routes("samples.py")
    expected = {
        "/api/v1/samples": {"GET"},
        "/api/v1/samples/{sample_id}/preview": {"GET"},
        "/api/v1/samples/{sample_id}/load": {"POST"},
    }
    assert backend == expected, f"the backend sample routes moved: {backend}"

    client = _read(SERVICES / "samples" / "client.ts")
    frontend_paths = _frontend_path_set(client)
    missing = expected.keys() - frontend_paths
    assert not missing, (
        f"the samples client lost backend routes: {sorted(missing)} "
        f"(has {sorted(frontend_paths)})"
    )


def test_dataset_upload_form_fields_match_the_backend() -> None:
    datasets_src = _route_module("datasets.py")
    upload_block = datasets_src.split("async def upload_dataset", 1)[1]
    upload_block = upload_block.split("@router.get", 1)[0]
    backend_fields = set(re.findall(r"(\w+):\s*[^\n=]+=\s*(?:File|Form)\(", upload_block))
    assert backend_fields == {"file", "target", "task", "mode"}, (
        f"the backend upload form fields moved: {sorted(backend_fields)}"
    )

    client = _read(SERVICES / "datasets" / "client.ts")
    for field in sorted(backend_fields):
        assert f'form.append("{field}"' in client, (
            f"the upload lost the form field {field!r} the backend expects"
        )
    # the browser owns the multipart boundary — no JSON content-type
    assert "postForm" in client
    assert 'api.postForm("/api/v1/datasets"' in client


def test_dataset_requests_use_the_generated_contract_types() -> None:
    client = _read(SERVICES / "datasets" / "client.ts")
    # §12.2: one DTO source — the request bodies come from api.d.ts
    assert 'from "@/types/api.d"' in client
    assert "ProfileRequest" in client and "SplitRequest" in client
    # …and nothing hand-duplicates them beside the generated import
    assert "interface ProfileRequest" not in client
    assert "interface SplitRequest" not in client


# -- the artifact surface -------------------------------------------------


def test_artifact_rest_surface_covered() -> None:
    backend = _backend_routes("artifacts.py")
    registry_routes = {
        "/api/v1/artifacts": {"GET"},
        "/api/v1/artifacts/{artifact_id}": {"GET", "DELETE"},
        "/api/v1/artifacts/{artifact_id}/download": {"GET", "HEAD"},
    }
    # the registry surface the generic client rides must exist as-is
    for path, methods in registry_routes.items():
        assert backend.get(path, set()) >= methods, (
            f"the backend artifact registry routes moved: {path} -> "
            f"{backend.get(path)}"
        )
    # the creation routes (figure/table/model POST) are the workspace
    # rows' own surfaces — NOT the generic registry client's contract

    client = _read(SERVICES / "artifacts" / "client.ts")
    frontend_paths = _frontend_path_set(client)
    missing = registry_routes.keys() - frontend_paths
    assert not missing, (
        f"the artifacts client lost backend routes: {sorted(missing)} "
        f"(has {sorted(frontend_paths)})"
    )


def test_artifact_download_rides_the_token_header_lane() -> None:
    client = _read(SERVICES / "artifacts" / "client.ts")
    # §15: the browser cannot put the session header on a bare <a href> —
    # the download must ride the client's raw lane (token applied)
    assert "api.raw(" in client
    assert "/download" in client
    # the route's verification headers are consumed, not discarded
    assert "X-Artifact-SHA256" in client
    assert "ETag" in client


# -- the chemistry surface (§11.1's featurization stage) ------------------


def test_chemistry_rest_surface_covered() -> None:
    backend = _backend_routes("chemistry.py")
    expected = {
        "/api/v1/chemistry/descriptors": {"POST"},
        "/api/v1/chemistry/fingerprints": {"POST"},
        "/api/v1/chemistry/standardize": {"POST"},  # DS-2 (row 6)
    }
    assert backend == expected, f"the backend chemistry routes moved: {backend}"

    client = _read(SERVICES / "chemistry" / "client.ts")
    frontend_paths = _frontend_path_set(client)
    missing = expected.keys() - frontend_paths
    assert not missing, (
        f"the chemistry client lost backend routes: {sorted(missing)} "
        f"(has {sorted(frontend_paths)})"
    )


def test_chemistry_requests_use_the_generated_contract_types() -> None:
    client = _read(SERVICES / "chemistry" / "client.ts")
    # §12.2: one DTO source — the request bodies come from api.d.ts
    assert 'from "@/types/api.d"' in client
    assert "DescriptorsRequest" in client and "FingerprintsRequest" in client
    assert "StandardizeRequest" in client
    assert "interface DescriptorsRequest" not in client
    assert "interface FingerprintsRequest" not in client
    assert "interface StandardizeRequest" not in client
    # the engine's own cap is mirrored once, beside the route
    assert "CHEMISTRY_MAX_SMILES = 500" in client


# -- the React binding ----------------------------------------------------


def test_hooks_bind_the_services_through_react_stores() -> None:
    hooks = _read(HOOKS / "useJobTracking.ts")
    # §8.2: the store binding is React's own, no extra dependency
    assert "useSyncExternalStore" in hooks
    # the two documented actions reach the real API
    assert "jobsApi.cancel" in hooks
    assert "tracker.refresh" in hooks
    # the singleton stream/tracker (§5.2's application services)
    assert "getEventStream" in hooks
    assert "getJobTracker" in hooks
