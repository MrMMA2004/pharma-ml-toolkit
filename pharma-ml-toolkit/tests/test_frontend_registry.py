"""part16 §15.3 (Phase 12) — the model-registry drift guards.

The Python gate cannot execute the React registry card, but every
contract the lane rides CAN be pinned against the real frontend +
backend source — a rename, a dropped §15.3 field, a renumbered stage,
or a second parallel registry architecture turns a guard red before
the surface drifts:

* the registry component + its vitest suite exist under
  ``components/registry`` and the client operations live in the ML
  client (the one-DTO rule — no duplicate registry DTOs);
* the integration rule: the card's ONLY data sources are the ml
  client's own operations plus the artifacts client's verified
  download (§8 — no parallel registry architecture, no bespoke
  deletion path client-side);
* the §15.3 vocabulary is mirrored on both sides (the catalog's five
  keys; the trained view's twelve);
* the store bound stated in the card == ``MAX_TRAINED_MODELS``;
* the version column is derived (created_at order within each
  model+dataset identity) and stated as derived, never invented;
* the backend's deletion lane stands (``DELETE /ml/trained/{id}``,
  the honest 404, the scope rails in the docstring);
* the ML workspace mounts the card inside the Compare stage without
  renumbering any stage;
* the generated types were regenerated over the 114-path contract
  (the DELETE rides the existing trained-model path);
* the eviction scope rails stand on the backend (the run lineage
  outlives the model — the record keeps its reference verbatim).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
COMPONENTS = SRC / "components" / "registry"
ML_SERVICE = SRC / "services" / "ml"
API = ROOT / "src" / "pharmaml" / "api"
ROUTES = API / "routes"
ML_PAGE = SRC / "features" / "ml" / "MLPage.tsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the component + the suites + the client operations


def test_the_registry_component_and_its_suites_exist() -> None:
    assert (COMPONENTS / "ModelRegistry.tsx").is_file(), (
        "the registry card is missing"
    )
    assert (COMPONENTS / "__tests__" / "ModelRegistry.test.tsx").is_file()
    assert (ML_SERVICE / "__tests__" / "registry.test.ts").is_file(), (
        "the client operations' own suite is missing"
    )


def test_the_client_carries_the_three_registry_operations() -> None:
    client = _read(ML_SERVICE / "client.ts")
    assert "async evictTrained(" in client, (
        "the deletion lane must ride the ml client (§15.3: deletion)"
    )
    assert "async exportTrained(" in client, (
        "the export lane must ride the ml client (§15.3: export)"
    )
    assert "async catalog(" in client and "task?: CatalogTaskFilter" in client, (
        "the catalog's full listing (task omitted) must stay available"
    )


def test_the_card_pins_its_data_sources_to_the_existing_surface() -> None:
    card = _read(COMPONENTS / "ModelRegistry.tsx")
    # §8: the only data sources are the ml client's own operations…
    assert 'from "@/services/ml/client"' in card
    assert "mlApi.catalog(undefined, false)" in card, (
        "the card must ask for the full catalog (availability per row)"
    )
    assert "mlApi.trained()" in card
    assert "mlApi.trainedView(" in card
    assert "mlApi.evictTrained(" in card
    assert "mlApi.exportTrained(" in card
    # …plus the artifacts client's verified download (never a bespoke lane)
    assert 'from "@/services/artifacts/client"' in card
    assert "artifactsApi.download(" in card
    # no parallel fetch layer, no raw fetch
    assert "fetch(" not in card


# ---------------------------------------------------------------------------
# the §15.3 vocabulary mirror


def test_the_frontend_mirrors_the_catalog_and_trained_vocabularies() -> None:
    client = _read(ML_SERVICE / "client.ts")
    for field in (
        "name",
        "task",
        "available",
        "requires",
        "notes",
    ):
        assert f'"{field}"' in client or field in client, (
            f"§15.3 names the catalog's '{field}' — the client must mirror it"
        )
    for field in (
        "model_id",
        "name",
        "task",
        "dataset_id",
        "target",
        "features",
        "n_train_rows",
        "best_params",
        "cv_scores",
        "cv_mean",
        "cv_std",
        "fit_seconds",
        "created_at",
    ):
        assert f'"{field}"' in client, (
            f"the trained view lost the '{field}' field"
        )


def test_the_backend_catalog_and_trained_surfaces_stand() -> None:
    ml = _read(ROUTES / "ml.py")
    assert '@router.get("/models")' in ml
    assert '@router.get("/trained")' in ml
    assert '@router.get("/trained/{model_id}")' in ml
    artifacts = _read(ROUTES / "artifacts.py")
    assert '@router.post("/model")' in artifacts, (
        "the export lane (§15.3: export) must stand on the artifacts route"
    )


# ---------------------------------------------------------------------------
# the honest-lane rules


def test_the_store_bound_is_stated_where_it_acts() -> None:
    card = _read(COMPONENTS / "ModelRegistry.tsx")
    stores = _read(API / "stores.py")
    match = re.search(r"TRAINED_STORE_BOUND = (\d+)", card)
    assert match is not None, "the card must state the store's own bound"
    bound_card = int(match.group(1))
    match_store = re.search(r"MAX_TRAINED_MODELS = (\d+)", stores)
    assert match_store is not None
    assert bound_card == int(match_store.group(1)), (
        "the card's bound must mirror MAX_TRAINED_MODELS, never invent one"
    )


def test_the_version_column_is_derived_and_stated_as_derived() -> None:
    card = _read(COMPONENTS / "ModelRegistry.tsx")
    assert "versionIndex(" in card, "the version derivation must exist"
    assert "created_at.localeCompare" in card, (
        "the derivation rides the registry's own created_at order"
    )
    assert "derived from the created_at order" in card, (
        "the version column must be stated as derived, never invented"
    )


def test_the_deletion_lane_is_scoped_on_the_backend() -> None:
    ml = _read(ROUTES / "ml.py")
    assert '@router.delete("/trained/{model_id}")' in ml, (
        "the registry's deletion lane must stand on the ml route"
    )
    assert "model_store.evict(model_id)" in ml, (
        "the deletion must ride the store's own evict, never a bespoke path"
    )
    assert 'status_code=404, detail="Unknown model_id."' in ml, (
        "an unknown id must be the honest 404, never a silent no-op"
    )
    # the scope rails: the run lineage outlives the evicted model
    assert "outlives the model" in ml


# ---------------------------------------------------------------------------
# the ML workspace mount + the generated contract


def test_the_registry_card_mounts_inside_the_compare_stage() -> None:
    page = _read(ML_PAGE)
    assert "<ModelRegistry" in page, "the registry card is not mounted"
    assert 'from "@/components/registry/ModelRegistry"' in page
    assert re.search(
        r"<RunLineage[^>]*/>\s*\n\s*<ModelRegistry", page
    ), "the registry card rides beside the lineage card in the Compare stage"
    # no stage renumbering: the Compare stage keeps its index 9 title
    compare = re.search(r'index=\{9\}\s*\n\s*title="Compare"', page)
    assert compare is not None, "the Compare stage lost its index-9 pin"


def test_the_generated_types_carry_the_deletion_path() -> None:
    generated = _read(FRONTEND / "src" / "types" / "api.d.ts")
    assert '"/api/v1/ml/trained/{model_id}"' in generated
    assert "api.delete<" in _read(ML_SERVICE / "client.ts"), (
        "the evict lane rides the transport's DELETE method"
    )


def test_the_contract_carries_the_registry_deletion_path() -> None:
    contract = json.loads(
        (ROOT / "contracts" / "openapi.json").read_text(encoding="utf-8")
    )
    paths = contract["paths"]
    assert "/api/v1/ml/trained/{model_id}" in paths
    assert "delete" in paths["/api/v1/ml/trained/{model_id}"], (
        "the trained-model path must carry the DELETE operation"
    )
    # the DELETE rides the EXISTING path — the path count stays 114
    # (the 112 + runs pair contract, unchanged by this lane)
    assert len(paths) == 114, (
        "the contract must pin 114 paths (the registry's DELETE rides the "
        "existing trained-model path — no new path)"
    )
