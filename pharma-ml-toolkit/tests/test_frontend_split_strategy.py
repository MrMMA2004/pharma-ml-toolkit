"""The split-strategy drift guard (DS-1 + ML-3, the roadmap's first rows).

The strategy lane spans three sources that must never drift apart:

* the engine's vocabulary — ``pharmaml.ml.splits.SPLIT_STRATEGIES``;
* the route's request Literals — ``routes/datasets.py:SplitRequest``
  (DS-1) and ``routes/ml.py:TrainSplitSpec`` (ML-3, the train lane's
  ``split`` section);
* the frontend's copies — the generated OpenAPI contract the service
  client types ride, and the page's own selector options / the client
  parser's literal lists.

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

ML-3 pins (the train lane's own acceptance, same clause):
* ``TrainSplitSpec``'s strategy Literal == the engine's set;
* the contract's TrainRequest.split -> TrainSplitSpec schema carries
  the same enum (the generated-types lane);
* the ML page's Configure Split/CV selector offers every engine
  strategy + the legacy "none" lane + the group-column gate;
* the ml client parser's ``TRAIN_SPLIT_STRATEGY_LITERALS`` == the
  engine's set, and the carve block's census fields narrow on both
  ends of the JSON boundary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pharmaml.ml.splits import SPLIT_STRATEGIES

REPO = Path(__file__).resolve().parent.parent
ROUTES = REPO / "src" / "pharmaml" / "api" / "routes" / "datasets.py"
ML_ROUTES = REPO / "src" / "pharmaml" / "api" / "routes" / "ml.py"
CONTRACT = REPO / "contracts" / "openapi.json"
PAGE = REPO / "frontend" / "src" / "features" / "datasets" / "DatasetsPage.tsx"
CLIENT = REPO / "frontend" / "src" / "services" / "datasets" / "client.ts"
ML_PAGE = REPO / "frontend" / "src" / "features" / "ml" / "MLPage.tsx"
ML_CLIENT = REPO / "frontend" / "src" / "services" / "ml" / "client.ts"


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


def _ml_route_strategy_literals() -> set[str]:
    src = ML_ROUTES.read_text(encoding="utf-8")
    m = re.search(
        r"class TrainSplitSpec\(BaseModel\):.*?strategy:\s*Literal\[(.*?)]",
        src,
        re.DOTALL,
    )
    assert m, "TrainSplitSpec no longer declares the strategy Literal"
    return set(re.findall(r'"([a-z_]+)"', m.group(1)))


def _ml_client_train_literals() -> set[str]:
    src = ML_CLIENT.read_text(encoding="utf-8")
    m = re.search(
        r"const TRAIN_SPLIT_STRATEGY_LITERALS[^=]*=\s*\[(.*?)];",
        src,
        re.DOTALL,
    )
    assert m, "ml/client.ts no longer declares TRAIN_SPLIT_STRATEGY_LITERALS"
    return set(re.findall(r'"([a-z_]+)"', m.group(1)))


def _ml_page_option_values() -> set[str]:
    src = ML_PAGE.read_text(encoding="utf-8")
    m = re.search(
        r"value=\{splitStrategy\}(.*?)</select>", src, re.DOTALL
    )
    assert m, "the ML page's split strategy selector is gone"
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


class TestTrainLaneVocabulary:
    """ML-3 — the same engine vocabulary on the train lane's carriers."""

    def test_train_route_literal_set_equals_the_engine(self):
        assert _ml_route_strategy_literals() == set(SPLIT_STRATEGIES)

    def test_train_client_literals_equal_the_engine(self):
        assert _ml_client_train_literals() == set(SPLIT_STRATEGIES)

    def test_train_page_selector_offers_the_engine_set(self):
        options = _ml_page_option_values()
        assert set(SPLIT_STRATEGIES) <= options
        assert "none" in options  # the legacy full-frame lane


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

    def test_train_request_split_section_carries_the_enum(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        schemas = contract["components"]["schemas"]
        split_prop = schemas["TrainRequest"]["properties"]["split"]
        refs = {
            part["$ref"]
            for part in split_prop["anyOf"]
            if "$ref" in part
        }
        assert refs == {"#/components/schemas/TrainSplitSpec"}
        spec = schemas["TrainSplitSpec"]
        assert set(spec["properties"]["strategy"]["enum"]) == set(
            SPLIT_STRATEGIES
        )
        assert spec["required"] == ["strategy"]
        assert "smiles_column" in spec["properties"]
        assert "group_column" in spec["properties"]
        assert "test_size" in spec["properties"]


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

    def test_ml_page_pins_the_leakage_verdict_and_the_group_gate(self):
        src = ML_PAGE.read_text(encoding="utf-8")
        assert "shared groups across the sides" in src  # the verdict line
        assert "group labels are never guessed" in src  # the gate hint
        # the train button stays gated without a group column
        assert (
            'splitStrategy === "group" && splitGroupColumn === ""' in src
        )
        # the carve readout renders the census + per-side group counts
        assert "group_census" in src and "n_groups" in src

    def test_train_route_response_carries_the_census_fields(self):
        src = ML_ROUTES.read_text(encoding="utf-8")
        assert '"group_census"' in src
        assert '"shared_groups"' in src
        assert '"top_groups"' in src

    def test_ml_client_parser_narrows_the_carve_block(self):
        src = ML_CLIENT.read_text(encoding="utf-8")
        assert "parseTrainSplitBlock" in src
        assert "shared_groups" in src
        assert "top_groups" in src
        # the honest sides readout: the fit saw the train side only
        assert "n_train_rows" in ML_PAGE.read_text(encoding="utf-8")
