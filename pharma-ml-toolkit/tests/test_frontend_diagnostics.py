"""part16 §15.5 (Phase 12) — the diagnostics/recovery drift guards.

The Python gate cannot execute the React diagnostics card, but every
contract the lane rides CAN be pinned against the real frontend +
backend source — a rename, a dropped §15.5 block, a second mutation
control smuggled into the surface, or a cache target outside the
enumerated list turns a guard red before the surface drifts:

* the service pair (client/diagnostics) + the component + the two
  suites exist under ``services/system`` / ``components/system``;
* the integration rule: the client's ONLY data sources are
  ``GET /api/v1/system/diagnostics`` and
  ``POST /api/v1/system/cache/cleanup`` (composing the existing
  system service — no parallel monitoring architecture);
* the §15.5 vocabulary is mirrored on both sides of the wire (the
  dependency probe table, the scientific-tool functional probes, the
  crash/log/recovery block names);
* the no-user-data rule rides both sources (the backend's preserved
  list + the card's verbatim guarantee line);
* the card offers exactly ONE mutation control — the safe cache
  cleanup (the diagnostics are read-only by design);
* the Introduction page mounts the card beside the studio-database
  section (the backup/restore half of §15.5 it completes);
* the cleanup lane targets the enumerated derived cache only (the
  samples stats cache) through the module's own public function.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
SYSTEM = SRC / "services" / "system"
COMPONENTS = SRC / "components" / "system"
ROUTES = ROOT / "src" / "pharmaml" / "api" / "routes"
SYSTEM_ROUTE = ROUTES / "system.py"
SAMPLES_ROUTE = ROUTES / "samples.py"
INTRO_PAGE = SRC / "features" / "introduction" / "IntroductionPage.tsx"
GENERATED = FRONTEND / "src" / "types" / "api.d.ts"

#: the dependency probe vocabulary the route probes (the
#: requirements.txt inventory — the frontend's tier split mirrors it).
PROBE_NAMES = (
    "numpy", "pandas", "scipy", "sklearn", "joblib", "openpyxl",
    "rdkit", "torch", "matplotlib", "xgboost", "lightgbm", "catboost",
    "shap", "selfies", "mordred", "meeko", "gemmi",
    "fastapi", "uvicorn", "pydantic", "psutil",
)

#: the scientific tools carrying functional probes.
FUNCTIONAL_TOOLS = ("rdkit", "torch", "selfies")

#: the §15.5 answer's block vocabulary (the parser's field-for-field
#: mirror — a backend rename turns this guard red first).
BLOCK_KEYS = (
    "dependencies",
    "scientific_tools",
    "storage",
    "crash",
    "logs",
    "recovery",
    "scope",
)


def _read(path: Path) -> str:
    assert path.is_file(), f"missing file: {path}"
    return path.read_text(encoding="utf-8")


def _read_raw(path: Path) -> bytes:
    assert path.is_file(), f"missing file: {path}"
    return path.read_bytes()


# ---------------------------------------------------------------------------
# the lane's files


def test_the_service_pair_and_component_exist() -> None:
    for name in ("client.ts", "types.ts", "diagnostics.ts"):
        assert (SYSTEM / name).is_file(), f"the system service lost {name}"
    assert (SYSTEM / "__tests__" / "diagnostics.test.ts").is_file()
    assert (COMPONENTS / "DiagnosticsCard.tsx").is_file()
    assert (COMPONENTS / "__tests__" / "DiagnosticsCard.test.tsx").is_file()


def test_the_client_pins_the_two_integration_points() -> None:
    client = _read(SYSTEM / "client.ts")
    assert "/api/v1/system/diagnostics" in client, (
        "the diagnostics must consume the backend's route — a second "
        "diagnostics source would be the parallel architecture §8 forbids"
    )
    assert "/api/v1/system/cache/cleanup" in client, (
        "the cleanup must consume the backend's safe-cleanup route"
    )
    assert "parseDiagnosticsSnapshot" in client, (
        "the raw diagnostics answer must be narrowed, never any"
    )
    assert "parseCacheCleanupAnswer" in client, (
        "the cleanup answer must be narrowed, never any"
    )
    # the backend surface stands (the consumed contract)
    route = _read(SYSTEM_ROUTE)
    assert '@router.get("/diagnostics")' in route
    assert '@router.post("/cache/cleanup")' in route


def test_the_lane_composes_the_existing_system_service() -> None:
    # the diagnostics lane extends the SAME service module the §8
    # resource monitor owns — no separate system service tree
    client = _read(SYSTEM / "client.ts")
    assert "systemApi" in client, (
        "the diagnostics operations ride the existing systemApi object"
    )
    assert "/api/v1/system/resources" in client, (
        "the §8 resource lane must stay mounted alongside"
    )


def test_the_probe_vocabulary_is_mirrored_both_sides() -> None:
    route = _read(SYSTEM_ROUTE)
    for name in PROBE_NAMES:
        assert f'"{name}"' in route, (
            f"the backend probe table lost '{name}'"
        )
    diagnostics = _read(SYSTEM / "diagnostics.ts")
    assert '"required" | "optional"' in diagnostics, (
        "the tier split must mirror the requirements.txt inventory"
    )


def test_the_functional_probe_vocabulary_is_pinned() -> None:
    route = _read(SYSTEM_ROUTE)
    for tool in FUNCTIONAL_TOOLS:
        assert f'"{tool}"' in route
    assert "functional_probe" in route
    # the parser carries the field-for-field mirror (the tool names
    # themselves are the backend's dynamic answer, not a frontend enum)
    diagnostics = _read(SYSTEM / "diagnostics.ts")
    assert "functional_probe" in diagnostics
    for probe_word in ("version", "importable", "probe"):
        assert probe_word in diagnostics


def test_the_block_vocabulary_is_mirrored_field_for_field() -> None:
    route = _read(SYSTEM_ROUTE)
    diagnostics = _read(SYSTEM / "diagnostics.ts")
    for key in BLOCK_KEYS:
        assert f'"{key}"' in route, f"the backend lost the {key} block"
    for key in BLOCK_KEYS:
        assert key in diagnostics, (
            f"the frontend parser lost the {key} block"
        )
    # the honest-split vocabulary of the recovery block
    for field in ("persisted_by_state", "in_session_by_state"):
        assert field in route
        assert field in diagnostics


def test_the_no_user_data_rule_rides_both_sources() -> None:
    route = _read(SYSTEM_ROUTE)
    assert "Do not delete user data during diagnostic or cleanup" in route, (
        "§15.5's own rule must ride the backend answer verbatim"
    )
    card = _read(COMPONENTS / "DiagnosticsCard.tsx")
    assert "Do not delete user data during diagnostic or cleanup" in card, (
        "the card must render the guarantee verbatim"
    )
    # the preserved list names every user-data surface
    for surface in (
        "datasets", "trained models", "generative priors", "artifacts",
        "user preferences", "job history",
    ):
        assert surface in route, (
            f"the backend preserved list lost: {surface}"
        )


def test_the_card_offers_exactly_one_mutation_control() -> None:
    card = _read(COMPONENTS / "DiagnosticsCard.tsx")
    assert "cleanupCache" in card, (
        "the single mutation control is the safe cache cleanup"
    )
    for word in ("deleteDataset", "evict", "purge", "drop table"):
        assert word not in card, (
            f"a foreign mutation control crept in: {word}"
        )
    # the mutation vocabulary stays the §15.5 enumeration
    samples = _read(SAMPLES_ROUTE)
    assert "def clear_stats_cache()" in samples, (
        "the cleanup must ride the samples module's own public function"
    )
    route = _read(SYSTEM_ROUTE)
    assert "clear_stats_cache" in route


def test_the_cleanup_target_is_the_enumerated_cache_only() -> None:
    route = _read(SYSTEM_ROUTE)
    # the route body must not reference the stores or the persistence —
    # the ONLY callable it makes is the samples clear fn (the prose
    # docstring names the surfaces it does NOT touch; the executable
    # references are what this guard scans)
    cleanup_body = route.split("@router.post(\"/cache/cleanup\")")[1]
    cleanup_body = cleanup_body.split("@router")[0]
    # strip the docstring (the prose names the preserved surfaces)
    cleanup_body = cleanup_body.split('"""', 2)[-1]
    for forbidden in (
        "dataset_store", "model_store", "artifact_store", "prior_store",
        "run_store", "persistence", "delete_dataset", "delete_model",
    ):
        assert forbidden not in cleanup_body, (
            f"the cleanup must never reference {forbidden} — §15.5's rule"
        )


def test_the_introduction_mounts_the_card() -> None:
    page = _read(INTRO_PAGE)
    assert "DiagnosticsCard" in page, (
        "the Introduction workspace must mount the diagnostics card"
    )
    assert "StudioDatabaseSection" in page, (
        "the database/backup half of §15.5 must stay mounted beside it"
    )
    card = _read(COMPONENTS / "DiagnosticsCard.tsx")
    assert 'id="diagnostics-card"' in card


def test_the_generated_types_carry_the_diagnostics_paths() -> None:
    generated = _read_raw(GENERATED)
    assert b"/api/v1/system/diagnostics" in generated, (
        "api.d.ts must be regenerated over the diagnostics path"
    )
    assert b"/api/v1/system/cache/cleanup" in generated, (
        "api.d.ts must carry the cache-cleanup path"
    )
