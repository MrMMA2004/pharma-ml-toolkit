"""The standardization-action drift guard (DS-2, the roadmap's row 6).

The lane spans four sources that must never drift apart:

* the engine's callable surface — ``pharmaml.chem.standardize``'s
  ``canonicalize_tautomer`` + ``reinvent_standardize_smiles`` (the
  only two passes the route may offer, the REINVENT-style chain being
  the roadmap entry's own named engine);
* the route's request Literals — ``routes/chemistry.py:StandardizeRequest``;
* the generated contract the service client types ride
  (``contracts/openapi.json`` -> ``src/types/api.d.ts``);
* the frontend's copies — the client's ``STANDARDIZE_OP_LITERALS`` +
  the page's stage-4 checkboxes.

Plus the roadmap's own acceptance clause, "a drift guard scanning
that the route never writes to the source record": the route source
is scanned for every mutation pattern that would touch a registered
record in place (store internals, evict, in-place frame edits) — the
standardization body must build a ``.copy()`` and register only the
derived frame.

Pins (the roadmap's DS-2 acceptance, the drift-guard clause):
* the route Literal set == the engine's two real callables;
* the contract's StandardizeRequest schema carries the ops enum, the
  column override, and the neutralize toggle (generated-types lane);
* the client's literal list == the same set, and the parser narrows
  counts/rejected/provenance on both ends of the JSON boundary;
* the page offers both passes + the uncharge toggle + the derived
  dataset's select action + the honest "source untouched" note;
* the route never writes to the source record (the scan below).
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

#: The engine's own callable passes — the REINVENT-style chain and the
#: tautomer canonicalizer are the ONLY real units the route may offer
#: (``chem/standardize.py``'s public vocabulary for the Clean stage).
ENGINE_OPS = {"canonical_tautomer", "reinvent"}


def _route_ops_literals() -> set[str]:
    src = ROUTES.read_text(encoding="utf-8")
    m = re.search(
        r"ops: list\[Literal\[(.*?)]\]", src, re.DOTALL
    )
    assert m, "StandardizeRequest no longer declares the ops Literal"
    return set(re.findall(r'"([a-z_]+)"', m.group(1)))


def _client_ops_literals() -> set[str]:
    src = CLIENT.read_text(encoding="utf-8")
    m = re.search(
        r"export const STANDARDIZE_OP_LITERALS[^=]*=\s*\[(.*?)\] as const",
        src,
        re.DOTALL,
    )
    assert m, "client.ts no longer declares STANDARDIZE_OP_LITERALS"
    return set(re.findall(r'"([a-z_]+)"', m.group(1)))


class TestOpsVocabulary:
    """One engine vocabulary, three carriers — never two sets."""

    def test_route_literal_set_equals_the_engine(self):
        assert _route_ops_literals() == ENGINE_OPS

    def test_client_literal_set_equals_the_engine(self):
        assert _client_ops_literals() == ENGINE_OPS

    def test_contract_enum_equals_the_engine(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        schema = contract["components"]["schemas"]["StandardizeRequest"]
        items = schema["properties"]["ops"]["items"]
        assert set(items["enum"]) == ENGINE_OPS
        # the honest knobs ride the schema too
        assert "column" in schema["properties"]
        assert "neutralize" in schema["properties"]
        assert "dataset_id" in schema["required"]


class TestNoWriteToTheSourceRecord:
    """The roadmap's own clause: the route never writes the source.

    Scans ``routes/chemistry.py`` for every pattern that would mutate
    a registered record in place: store internals (``_items``), evict,
    in-place frame edits (``inplace=``, ``.loc[...] =``, ``.at[``,
    ``.iat[``), or a write into ``record.frame``. The standardization
    body must stage its output through a ``.copy()`` and register only
    the derived frame through the public ``store.register``.
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
        # positional/label ASSIGNMENTS on a frame are the mutation
        # patterns (a bare .iloc[...] READ + .copy() is the staged
        # output, the opposite of a write)
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
        # the standardized frame is a staged copy, never the source
        assert ".copy()" in src

    def test_route_reads_the_source_frame_only_through_loads(self):
        src = ROUTES.read_text(encoding="utf-8")
        # the one legitimate read: frame[column].tolist() (a load)
        assert "frame[column].tolist()" in src
        # the staged output is a positional read + copy, never the source
        assert "frame.iloc[keep].copy()" in src
        # and the source dataset id is pinned as read-only metadata
        assert 'f"Unknown dataset_id {req.dataset_id!r}."' in src


class TestPageAndParserContract:
    """The stage-4 action and the report parser stay in the sources."""

    def test_page_pins_the_action_and_the_honest_notes(self):
        src = PAGE.read_text(encoding="utf-8")
        # the action button and the pass checkboxes
        assert "Standardize → derived dataset" in src
        assert "canonical tautomer pass (applied first)" in src
        assert "REINVENT4 chain" in src
        assert 'setStdNeutralize' in src  # the uncharge toggle
        # the honest notes: no mutation, no silent drop
        assert "the source dataset is never modified" in src
        assert "never silently dropped" in src
        # the report's readouts: counts + rejected + derived select
        assert "Select the derived dataset" in src
        assert "rejected" in src and "reason" in src
        assert "the source is untouched" in src

    def test_client_parser_narrows_the_report(self):
        src = CLIENT.read_text(encoding="utf-8")
        assert "parseStandardizeReport" in src
        for field in (
            "source_dataset_id",
            "counts",
            "rejected",
            "provenance",
            "dataset_id",
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
        assert '"changed"' in src and '"unchanged"' in src


class TestDerivedProvenanceRidesTheStore:
    """The provenance block is the store's own field, not route prose."""

    def test_store_records_and_views_the_derived_block(self):
        stores = (
            REPO / "src" / "pharmaml" / "api" / "stores.py"
        ).read_text(encoding="utf-8")
        assert "derived: dict | None = None" in stores
        assert 'payload["derived"] = json_safe(self.derived)' in stores

    def test_derived_view_reaches_the_contractless_json_lane(self):
        # the route registers with derived= (the provenance block)
        src = ROUTES.read_text(encoding="utf-8")
        assert "derived=provenance" in src
        assert '"kind": "standardized"' in src
