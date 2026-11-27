"""The DS-1 split-strategy drift guard (the roadmap's first P0 row).

The strategy lane spans three sources that must never drift apart:

* the engine's vocabulary — ``pharmaml.ml.splits.SPLIT_STRATEGIES``;
* the route's request Literal — ``routes/datasets.py:SplitRequest``;
* the frontend's two copies — the generated OpenAPI contract the
  service client types ride, and the page's own selector options /
  the client parser's ``STRATEGY_LITERALS``.

This guard parses the REAL sources (the house pattern — no fabricated
fixtures): a strategy added to the engine without the route, the
contract, the parser, and the page selector turning red here before
the frontend silently drifts.

Pins (the roadmap's DS-1 acceptance, the drift-guard clause):
* the route Literal set == the engine's ``SPLIT_STRATEGIES``;
* the contract's SplitRequest schema carries the enum + the column
  fields (the generated-types lane);
* the page offers every engine strategy + the group-column gate +
  the shared-groups leakage verdict;
* the client parser's literal list == the engine's set;
* the census response fields (``group_census`` / ``shared_groups`` /
  ``top_groups``) exist on both ends of the JSON boundary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pharmaml.ml.splits import SPLIT_STRATEGIES

REPO = Path(__file__).resolve().parent.parent
ROUTES = REPO / "src" / "pharmaml" / "api" / "routes" / "datasets.py"
CONTRACT = REPO / "contracts" / "openapi.json"
PAGE = REPO / "frontend" / "src" / "features" / "datasets" / "DatasetsPage.tsx"
CLIENT = REPO / "frontend" / "src" / "services" / "datasets" / "client.ts"


def _route_strategy_literals() -> set[str]:
    src = ROUTES.read_text(encoding="utf-8")
    m = re.search(
        r"strategy:\s*Literal\[(.*?)]\s*\|\s*None", src, re.DOTALL
    )
    assert m, "SplitRequest no longer declares the strategy Literal"
    return set(re.findall(r'"([a-z_]+)"', m.group(1)))


def _client_strategy_literals() -> set[str]:
    src = CLIENT.read_text(encoding="utf-8")
    m = re.search(
        r"const STRATEGY_LITERALS[^=]*=\s*\[(.*?)];", src, re.DOTALL
    )
    assert m, "client.ts no longer declares STRATEGY_LITERALS"
    return set(re.findall(r'"([a-z_]+)"', m.group(1)))


def _page_option_values() -> set[str]:
    src = PAGE.read_text(encoding="utf-8")
    m = re.search(
        r"value=\{splitStrategy\}(.*?)</select>", src, re.DOTALL
    )
    assert m, "the page's split strategy selector is gone"
    return set(re.findall(r'value="([a-z_]+)"', m.group(1)))


class TestStrategyVocabulary:
    """One engine vocabulary, four carriers — never two sets."""

    def test_route_literal_set_equals_the_engine(self):
        assert _route_strategy_literals() == set(SPLIT_STRATEGIES)

    def test_client_parser_literals_equal_the_engine(self):
        assert _client_strategy_literals() == set(SPLIT_STRATEGIES)

    def test_page_selector_offers_the_engine_set(self):
        options = _page_option_values()
        assert set(SPLIT_STRATEGIES) <= options
        assert "auto" in options  # the legacy stratify-toggle lane


class TestContractCarriesTheFields:
    """The generated-types lane: the contract is the single DTO source."""

    def test_split_request_schema_has_the_strategy_fields(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        schema = contract["components"]["schemas"]["SplitRequest"]
        props = schema["properties"]
        values = set()
        for part in props["strategy"]["anyOf"]:
            if "enum" in part:
                values |= set(part["enum"])
        assert values == set(SPLIT_STRATEGIES)
        # Optional[str] rides anyOf(string, null) — both column fields
        assert {"type": "string"} in props["smiles_column"]["anyOf"]
        assert {"type": "string"} in props["group_column"]["anyOf"]


class TestPageAndParserContract:
    """The census verdict and the honest gate stay in the sources."""

    def test_page_pins_the_leakage_verdict_and_the_group_gate(self):
        src = PAGE.read_text(encoding="utf-8")
        assert "shared groups across sides" in src  # the verdict line
        assert "group labels are never guessed" in src  # the gate hint
        # the run button stays gated without a group column
        assert 'splitStrategy === "group" && splitGroupColumn === ""' in src
        # the census readout renders per-side group counts
        assert "group_census" in src and "n_groups" in src

    def test_route_response_carries_the_census_fields(self):
        src = ROUTES.read_text(encoding="utf-8")
        assert '"group_census"' in src
        assert '"shared_groups"' in src
        assert '"top_groups"' in src

    def test_client_parser_narrows_the_census(self):
        src = CLIENT.read_text(encoding="utf-8")
        assert "parseGroupCensus" in src
        assert "shared_groups" in src
        assert "top_groups" in src
