"""The model-evaluation consumer drift guard (ML-1, the roadmap's row 3).

The four finished conformal routes finally gained their frontend
consumer (the roadmap's own problem statement for ML-1 was "zero
consumers"); this guard keeps the lane from drifting back apart:

* the four paths stay consumed — the ml client carries them and the ML
  page renders the panel that calls them;
* the three POST wire bodies mirror the backend's pydantic request
  fields exactly (the field-mirror rule the ml-workspace family
  already pins for train / predict / AD analyze);
* the conditional-mode Literal set pins both ends of the boundary;
* the page renders the honest bits: the finite-sample caveat, the
  "empirical, never the guarantee" coverage line, the bounded
  interval / label-set tables, and the Wilson-bounded coverage table;
* the exported contract declares the family (the one-DTO rule).

Parses the REAL sources (the house pattern — no fabricated fixtures):
a field added to the backend request without the client's wire body
turns this guard red before the frontend silently stops sending it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROUTES = REPO / "src" / "pharmaml" / "api" / "routes" / "model_evaluation.py"
CLIENT = REPO / "frontend" / "src" / "services" / "ml" / "client.ts"
PAGE = REPO / "frontend" / "src" / "features" / "ml" / "MLPage.tsx"
CONTRACT = REPO / "contracts" / "openapi.json"

FOUR_PATHS = (
    "/api/v1/model-evaluation/catalog",
    "/api/v1/model-evaluation/conformal-regression",
    "/api/v1/model-evaluation/conformal-classification",
    "/api/v1/model-evaluation/coverage-diagnostics",
)


def _pydantic_fields(class_name: str) -> set[str]:
    """The field names of one BaseModel block (comment-aware)."""
    src = ROUTES.read_text(encoding="utf-8")
    start = re.search(
        r"^class " + re.escape(class_name) + r"\(BaseModel\):",
        src,
        re.MULTILINE,
    )
    assert start is not None, f"{class_name} vanished from the backend"
    rest = src[start.end() :]
    end = re.search(r"^(?:class |@router\.|def )", rest, re.MULTILINE)
    body = rest[: end.start()] if end else rest
    return set(re.findall(r"^\s+(\w+)\s*:", body, re.MULTILINE))


def _post_wire_keys(path: str) -> set[str]:
    """The wire keys of the client's api.post(path, ...) body.

    The path is anchored to the api.post CALL (not the JSDoc mention
    that also carries the literal); the generic parameter is skipped
    with a non-paren run so the nested angle brackets
    (<Record<string, unknown>>) are tolerated.
    """
    src = CLIENT.read_text(encoding="utf-8")
    m = re.search(
        "api\\.post<[^(]*>\\(\\s*[\"']" + re.escape(path),
        src,
    )
    assert m is not None, f"the {path} post call vanished from the client"
    tail = src[m.end() :]
    brace = tail.index("{")
    end = re.search(r"\n\s*\},?\s*\n\s*\);", tail)
    assert end is not None, f"the {path} post body vanished from the client"
    body = tail[brace : end.start()]
    return set(re.findall(r"^\s+(\w+):", body, re.MULTILINE))


def test_the_four_paths_stay_consumed() -> None:
    """The client carries the family; the page renders its panel."""
    client = CLIENT.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    for path in FOUR_PATHS:
        assert path in client, f"the ml client stopped calling {path}"
    assert "modelEvaluationApi" in page, (
        "the ML page stopped importing modelEvaluationApi"
    )
    assert "Run the conformal calibration" in page, (
        "the conformal panel's run button vanished"
    )


def test_conformal_regression_wire_mirror() -> None:
    """The regression lane's wire body == the backend request fields."""
    assert _post_wire_keys(FOUR_PATHS[1]) == _pydantic_fields(
        "ConformalRegressionRequest"
    )


def test_conformal_classification_wire_mirror() -> None:
    """The classification lane's wire body == the backend request."""
    assert _post_wire_keys(FOUR_PATHS[2]) == _pydantic_fields(
        "ConformalClassificationRequest"
    )


def test_coverage_diagnostics_wire_mirror() -> None:
    """The standalone coverage lane's wire body == the backend request."""
    assert _post_wire_keys(FOUR_PATHS[3]) == _pydantic_fields(
        "CoverageDiagnosticsRequest"
    )


def test_conditional_literals_pinned_both_ends() -> None:
    """The conditional-mode Literal rides both ends unchanged."""
    routes = ROUTES.read_text(encoding="utf-8")
    m = re.search(r"conditional:\s*Literal\[(.*?)]", routes, re.DOTALL)
    assert m, "the route's conditional Literal vanished"
    backend = set(re.findall(r'"([a-z_]+)"', m.group(1)))
    client = CLIENT.read_text(encoding="utf-8")
    c = re.search(
        r"MODEL_EVALUATION_CONDITIONAL_LITERALS[^=]*=\s*\[(.*?)\] as const",
        client,
        re.DOTALL,
    )
    assert c, "the client's conditional literal list vanished"
    frontend = set(re.findall(r'"([a-z_]+)"', c.group(1)))
    assert backend == frontend == {"split", "label"}, (
        f"the conditional literals drifted: backend={backend} "
        f"frontend={frontend}"
    )


def test_the_page_renders_the_honest_panel() -> None:
    """The panel's honest bits: the caveat, the never-the-guarantee
    lines, the bounded tables, the catalog notes, the ratio cap.

    The page source is whitespace-normalized before matching — the
    JSX wraps long copy across lines (the rendered text is one run).
    """
    page = re.sub(r"\s+", " ", PAGE.read_text(encoding="utf-8"))
    for marker in (
        # the finite-sample caveat (the exchangeability honesty)
        "finite-sample guarantee needs exchangeable held-out rows",
        # the empirical line renders "never the guarantee" on both lanes
        "empirical coverage (never the guarantee)",
        # the bounded tables' aria labels (interval / set / coverage)
        'aria-label="conformal interval rows"',
        'aria-label="conformal coverage table"',
        'aria-label="conformal label-set rows"',
        'aria-label="conformal per-class coverage table"',
        # the family catalog's honesty notes render beside the panel
        "cfCatalog.notes.join",
        # the bridge caps the slice sizes to the assembled rows
        "Math.max(2, Math.floor((cfCalRows / requested) * returned))",
    ):
        assert marker in page, f"the conformal panel lost: {marker!r}"


def test_the_contract_declares_the_family() -> None:
    """The exported openapi contract carries all four paths."""
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    declared = set(contract["paths"])
    for path in FOUR_PATHS:
        assert path in declared, f"the contract stopped declaring {path}"
