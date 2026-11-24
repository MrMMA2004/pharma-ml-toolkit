"""part16 §15.4 (Phase 12) — the data-quality drift guards.

The Python gate cannot execute the React quality card, but every
contract the lane rides CAN be pinned against the real frontend +
backend source — a rename, a dropped §15.4 finding code, a mutation
control smuggled into the advisory surface, or a renumbered stage
turns a guard red before the surface drifts:

* the service pair (client/types) + the component + the two suites
  exist under ``services/quality`` / ``components/quality``;
* the integration rule: the client's ONLY data source is
  ``POST /api/v1/datasets/{id}/quality`` (the lane composes with the
  loader's registered frames — no parallel validation architecture);
* the §15.4 finding vocabulary is mirrored code-for-code on both
  sides of the wire (the fourteen codes the engine emits);
* the severity + category vocabularies match;
* the no-mutation rule: the card offers no fix/repair/apply control
  and the read-only advisory rides both sources;
* the Datasets workspace mounts the card inside the Validate stage
  without renumbering any stage;
* the generated types were regenerated over the 115-path contract;
* the backend route carries the honest 404 + the sync/job pair.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
QUALITY = SRC / "services" / "quality"
COMPONENTS = SRC / "components" / "quality"
API = ROOT / "src" / "pharmaml"
ROUTES = API / "api" / "routes"
DATASETS_PAGE = SRC / "features" / "datasets" / "DatasetsPage.tsx"
ENGINE = ROOT / "src" / "pharmaml" / "data_quality.py"
DATASETS_ROUTE = ROUTES / "datasets.py"
GENERATED = FRONTEND / "src" / "types" / "api.d.ts"
CONTRACT = ROOT / "contracts" / "openapi.json"

#: the fourteen §15.4 finding codes the engine emits (the pinned
#: vocabulary — the frontend's QUALITY_CODES mirrors this list).
FINDING_CODES = (
    "invalid_smiles",
    "unsanitizable_molecules",
    "duplicate_molecules",
    "duplicate_rows",
    "duplicate_columns",
    "missing_values",
    "non_finite_values",
    "constant_columns",
    "id_like_column",
    "target_leakage",
    "class_imbalance",
    "train_test_overlap_risk",
    "out_of_range_values",
    "non_numeric_column",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the service pair + the component + the suites


def test_the_quality_service_pair_and_component_exist() -> None:
    for name in ("client.ts", "types.ts"):
        assert (QUALITY / name).is_file(), f"the quality service lost {name}"
    assert (QUALITY / "__tests__" / "types.test.ts").is_file()
    assert (COMPONENTS / "DataQualityCard.tsx").is_file()
    assert (COMPONENTS / "__tests__" / "DataQualityCard.test.tsx").is_file()


def test_the_client_pins_the_single_integration_point() -> None:
    client = _read(QUALITY / "client.ts")
    assert "/api/v1/datasets/${encodeURIComponent(datasetId)}/quality" in client, (
        "the scan must consume the backend's quality route — a second "
        "validation source would be the parallel architecture §8 forbids"
    )
    assert "encodeURIComponent" in client, (
        "dataset ids travel through encodeURIComponent per the house rule"
    )
    assert "parseDataQualityReport" in client, (
        "the raw answer must be narrowed, never any"
    )
    assert "QualityRequest" in client, (
        "the request body must come from the generated types (the one-DTO rule)"
    )
    # the backend surface stands (the consumed contract)
    route = _read(DATASETS_ROUTE)
    assert '@router.post("/{dataset_id}/quality")' in route
    assert "data_quality_report" in route
    assert (ENGINE).is_file()


# ---------------------------------------------------------------------------
# the §15.4 vocabulary mirror


def test_the_finding_vocabulary_is_mirrored_code_for_code() -> None:
    types = _read(QUALITY / "types.ts")
    engine = _read(ENGINE)
    for code in FINDING_CODES:
        assert f'"{code}"' in types, (
            f"the frontend lost the '{code}' finding code"
        )
        assert f'"{code}"' in engine, (
            f"the backend engine lost the '{code}' finding code"
        )


def test_the_severity_and_category_vocabularies_match() -> None:
    types = _read(QUALITY / "types.ts")
    for severity in ("error", "warning", "info"):
        assert f'"{severity}"' in types
    for category in (
        "chemistry",
        "completeness",
        "distribution",
        "integrity",
        "leakage",
        "range",
    ):
        assert f'"{category}"' in types, (
            f"the frontend lost the '{category}' finding category"
        )


# ---------------------------------------------------------------------------
# the no-mutation rule (§15.4's hard rule)


def test_the_card_offers_no_mutation_control() -> None:
    card = _read(COMPONENTS / "DataQualityCard.tsx")
    for word in ("\"Fix\"", "\"Repair\"", "\"Apply\"", "\"Delete\""):
        assert word not in card, (
            f"the quality card must never offer a '{word}' control — "
            "§15.4: never silently alter user-provided scientific data"
        )
    assert "advisory" in card.lower()


def test_the_read_only_contract_rides_both_sources() -> None:
    engine = _read(ENGINE)
    types = _read(QUALITY / "types.ts")
    assert '"read_only": True' in engine
    assert "read_only" in types
    assert "never modifies" in engine
    # the card renders the advisory line from the backend payload
    card = _read(COMPONENTS / "DataQualityCard.tsx")
    assert "report.advisory" in card


# ---------------------------------------------------------------------------
# the mount + the generated contract


def test_the_datasets_workspace_mounts_the_card_in_the_validate_stage() -> (
    None
):
    page = _read(DATASETS_PAGE)
    assert "DataQualityCard" in page, (
        "the Datasets workspace must mount the quality card"
    )
    assert 'datasetId={selectedId}' in page
    # no stage renumbering: the five stage headers stand
    for index, title in (
        (1, "Inspect"),
        (2, "Preview"),
        (3, "Validate"),
        (4, "Clean / Standardize / Select Columns"),
    ):
        assert f'title="{title}"' in page or title in page, (
            f"the DatasetsPage lost the '{title}' stage"
        )


def test_the_generated_types_carry_the_quality_request() -> None:
    generated = _read(GENERATED)
    assert "QualityRequest" in generated, (
        "api.d.ts must be regenerated over the re-exported contract"
    )
    contract = _read(CONTRACT)
    assert "/api/v1/datasets/{dataset_id}/quality" in contract, (
        "the OpenAPI contract must carry the quality path"
    )


def test_the_route_carries_the_honest_lanes() -> None:
    route = _read(DATASETS_ROUTE)
    assert "Unknown dataset_id." in route, "the honest 404 stands"
    assert '"job", "sync"' in route or 'Literal["job", "sync"]' in route, (
        "the sync/job pair stands on the quality request"
    )
    assert "CancelledJob" in route, "the cooperative cancel stands"
