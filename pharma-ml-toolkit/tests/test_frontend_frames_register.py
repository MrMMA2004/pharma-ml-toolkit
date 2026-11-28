"""The frames-register drift guard (DS-3, the roadmap's row 7).

The lane spans four sources that must never drift apart:

* the engines' callable surfaces -- ``pharmaml.chem.descriptors``'s
  ``compute_descriptors`` (the three descriptor sets) +
  ``compute_fingerprints`` (the four kinds) -- the only real
  featurization units the route may offer;
* the route's request Literals -- ``routes/chemistry.py:
  FramesRegisterRequest`` (the featurization branch, the descriptor
  sets, the fingerprint kinds);
* the generated contract the service client types ride
  (``contracts/openapi.json`` -> ``src/types/api.d.ts``);
* the frontend's copies -- the client's options + the page's
  stage-5 register block (the register button, the featurization
  select, the navigate hint).

Plus the roadmap's own acceptance clauses: the route never writes to
the source record (the same no-mutation scan the DS-2 guard runs),
the registration rides the store's public ``derived`` lane with the
``derived-descriptors|fingerprints`` provenance kinds, and the
navigate hint hands the workflow to the real ML workspace id.

Pins (the roadmap's DS-3 acceptance, the drift-guard clause):
* the route Literal sets == the engine's real vocabularies;
* the contract's FramesRegisterRequest schema carries the branch
  enum + both branches' knobs (generated-types lane);
* the client's request mirrors the generated DTO and the parser
  narrows counts (n_rows/kept/rejected/features_appended), the
  rejected triple, provenance, and the derived identity on both ends
  of the JSON boundary;
* the page offers the register button + the featurization select +
  the derived select + the navigate hint + the honest notes;
* the route never writes to the source record (the scan below);
* the provenance kinds ride the store's own field, and the
  fingerprints branch's kind key is ``fingerprint_kind`` (the
  duplicate-dict-key lesson the first draft taught: a second
  ``"kind"`` key silently overwrote ``derived-fingerprints``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROUTES = REPO / "src" / "pharmaml" / "api" / "routes" / "chemistry.py"
CONTRACT = REPO / "contracts" / "openapi.json"
PAGE = REPO / "frontend" / "src" / "features" / "datasets" / "DatasetsPage.tsx"
CLIENT = REPO / "frontend" / "src" / "services" / "chemistry" / "client.ts"
APP = REPO / "frontend" / "src" / "app" / "App.tsx"
WORKSPACES = REPO / "frontend" / "src" / "app" / "workspaces.ts"

#: The engines' own vocabularies -- the only real featurization units
#: the route may offer (``chem/descriptors.py``'s own Literal sets).
ENGINE_DESCRIPTOR_SETS = {"rdkit_all", "morgan", "mordred"}
ENGINE_FINGERPRINT_KINDS = {"morgan", "rdkit", "atom_pair", "avalon"}
ENGINE_FEATURIZATIONS = {"descriptors", "fingerprints"}


def _route_literals(field: str) -> set[str]:
    src = ROUTES.read_text(encoding="utf-8")
    m = re.search(
        rf"{field}: Literal\[(.*?)]", src, re.DOTALL
    )
    assert m, f"FramesRegisterRequest no longer declares {field}"
    return set(re.findall(r'"([a-z_]+)"', m.group(1)))


class TestFeaturizationVocabulary:
    """One engine vocabulary, three carriers -- never two sets."""

    def test_route_featurization_set_equals_the_engine(self):
        assert _route_literals("featurization") == ENGINE_FEATURIZATIONS

    def test_route_descriptor_set_equals_the_engine(self):
        assert _route_literals("descriptor_set") == ENGINE_DESCRIPTOR_SETS

    def test_route_fingerprint_kind_set_equals_the_engine(self):
        assert _route_literals("kind") == ENGINE_FINGERPRINT_KINDS

    def test_contract_schema_carries_the_branch_enum_and_knobs(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        schema = contract["components"]["schemas"]["FramesRegisterRequest"]
        props = schema["properties"]
        assert set(props["featurization"]["enum"]) == ENGINE_FEATURIZATIONS
        assert set(props["descriptor_set"]["enum"]) == ENGINE_DESCRIPTOR_SETS
        assert set(props["kind"]["enum"]) == ENGINE_FINGERPRINT_KINDS
        # both branches' knobs ride the generated DTO
        assert "dataset_id" in schema["required"]
        assert "target" in props and "column" in props
        assert "radius" in props and "n_bits" in props
        assert "as_counts" in props and "include_chirality" in props


class TestNoWriteToTheSourceRecord:
    """The roadmap's own clause: the route never writes the source.

    Scans ``routes/chemistry.py`` for every pattern that would mutate
    a registered record in place: store internals (``_items``), evict,
    in-place frame edits (``inplace=``, ``.loc[...] =``, ``.at[``,
    ``.iat[``), or a write into ``record.frame``. The register body
    must stage its output through a positional ``.copy()`` and
    register only the derived frame through the public
    ``store.register``.
    """

    def test_route_source_has_no_record_mutation_patterns(self):
        src = ROUTES.read_text(encoding="utf-8")
        for forbidden in (
            "._items",
            ".evict(",
            "inplace=True",
            "inplace = True",
        ):
            assert forbidden not in src, (
                f"the chemistry route must never carry {forbidden!r} "
                "(a registered record is read-only server-side)"
            )
        for pattern in (
            r"\.(loc|iloc)\[[^\]]*\]\s*=(?!=)",
            r"\.(at|iat)\[[^\]]*\]\s*=",
        ):
            assert re.search(pattern, src) is None, (
                f"the chemistry route must never assign through {pattern!r} "
                "(a registered record is read-only server-side)"
            )
        # the derived registration rides the public lane only
        assert "store.register(" in src
        # the featurized frame is a staged positional copy + concat,
        # never the source
        assert "frame.iloc[kept].copy().reset_index(drop=True)" in src
        assert "pd.concat(" in src

    def test_route_reads_the_source_frame_only_through_loads(self):
        src = ROUTES.read_text(encoding="utf-8")
        # the one legitimate read: frame[column].tolist() (a load)
        assert "frame[column].tolist()" in src
        # and the source dataset id is pinned as read-only metadata
        assert 'f"Unknown dataset_id {req.dataset_id!r}."' in src


class TestPageAndParserContract:
    """The stage-5 register block and the report parser stay in the
    sources (the roadmap's acceptance: the register button + the
    navigate hint)."""

    def test_page_pins_the_register_action_and_the_honest_notes(self):
        src = PAGE.read_text(encoding="utf-8")
        # the action button + the featurization select
        assert "Register as derived dataset" in src
        assert 'setRegFeaturization' in src
        assert "descriptors (the set picked above)" in src
        assert "fingerprints (the knobs picked above)" in src
        # the honest notes: no mutation, no silent drop
        assert "the source dataset is never modified" in src
        assert "never silently dropped" in src
        # the report's readouts: counts + rejected + derived select
        assert "Select the derived dataset" in src
        assert "rejected" in src and "reason" in src
        assert "the source is untouched" in src

    def test_page_pins_the_navigate_hint(self):
        src = PAGE.read_text(encoding="utf-8")
        # the shell's own switcher, handed down as a prop
        assert "onNavigate?: (id: WorkspaceId) => void" in src
        assert 'onNavigate("machine-learning")' in src
        assert "Continue in Machine Learning" in src

    def test_app_wires_the_navigate_lane_and_the_id_is_real(self):
        app = APP.read_text(encoding="utf-8")
        assert "<DatasetsPage onNavigate={handleSelect} />" in app
        workspaces = WORKSPACES.read_text(encoding="utf-8")
        assert '"machine-learning"' in workspaces

    def test_client_parser_narrows_the_report(self):
        src = CLIENT.read_text(encoding="utf-8")
        assert "parseFramesRegisterReport" in src
        for field in (
            "source_dataset_id",
            "counts",
            "rejected",
            "provenance",
            "dataset_id",
            "features_appended",
            "kept",
        ):
            assert field in src, f"the parser lost {field}"
        # the rejected rows keep their honest triple
        assert '"row"' in src and '"input"' in src and '"reason"' in src

    def test_route_response_carries_the_report_fields(self):
        src = ROUTES.read_text(encoding="utf-8")
        assert '"counts"' in src
        assert '"rejected"' in src
        assert '"provenance"' in src
        assert '"dataset"' in src
        assert '"kept"' in src and '"features_appended"' in src
        assert '"featurization"' in src


class TestDerivedProvenanceRidesTheStore:
    """The provenance block is the store's own field, not route prose."""

    def test_store_records_and_views_the_derived_block(self):
        stores = (
            REPO / "src" / "pharmaml" / "api" / "stores.py"
        ).read_text(encoding="utf-8")
        assert "derived: dict | None = None" in stores
        assert 'payload["derived"] = json_safe(self.derived)' in stores

    def test_derived_kinds_and_the_fingerprint_kind_key(self):
        # the route registers with derived= (the provenance block)
        src = ROUTES.read_text(encoding="utf-8")
        assert "derived=provenance" in src
        assert '"kind": "derived-descriptors"' in src
        assert '"kind": "derived-fingerprints"' in src
        # the duplicate-dict-key lesson: the fingerprints branch's kind
        # rides its own key, never a second "kind" that overwrites the
        # provenance kind
        assert '"fingerprint_kind": kind,' in src
        assert src.count('"fingerprint_kind"') == 1
