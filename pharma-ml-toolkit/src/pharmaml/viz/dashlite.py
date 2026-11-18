"""dash's reactive core, absorbed as a headless engine (100x-170).

The dash repository was reviewed file by file; everything of dash that
is *pure Python structure* lands here as a browser-free "DashLite":

* the **component spec** -- :class:`Component` follows
  ``dash/development/base_component.py``'s contract: keyword-argument
  props validated against a per-class list, the ``{"props", "type",
  "namespace"}`` serialization of ``to_plotly_json()`` (unset props
  omitted, never null), ids that refuse dots (upstream's
  ``IDsCantContainPeriods``), and tree traversal over ``children``.
  :class:`Div`/:class:`H3`/:class:`P` mirror the ``html`` namespace,
  :class:`Graph`/:class:`Slider`/:class:`Dropdown`/:class:`Button`/
  :class:`Store`/:class:`Interval`/:class:`DataTable` the ``dcc`` one
  (``n_clicks``/``n_intervals`` are plain counters a headless runtime
  can synthesize, exactly how the React components drive them).
* the **callback graph** -- :class:`Output`/:class:`Input`/:class:`State`
  dependencies, the ``@app.callback`` decorator with upstream's
  validation rules (Outputs first, then Inputs, then States;
  duplicate outputs raise :class:`DuplicateCallback`), a
  serializable ``callback_list`` (the ``_dash-dependencies`` payload),
  and :meth:`DashApp.trigger` -- the headless dispatch the React
  renderer performs in the browser: a changed prop fires every
  not-yet-run callback that reads it, outputs propagate, rounds
  repeat until nothing new fires.  One deliberate *addition* over
  upstream: registration-time cycle detection (dash's server side has
  none -- the renderer catches it; a headless engine must not hang).
* the **DataTable engine** -- :func:`filter_rows` implements the
  ``filter_query`` grammar verbatim from
  ``dash-table/src/core/syntax-tree``: the ``{column}`` field syntax,
  quoted strings vs bare tokens (bare numeric tokens become numbers),
  the relational operators ``= eq  != ne  > gt  >= ge  < lt  <= le
  contains datestartswith`` with the ``i``/``s`` case-sensitivity
  prefixes, the unary ``is blank/nil/num/str/bool/odd/even/prime``,
  ``!``/``&&``/``||`` with upstream's precedence (not > and > or,
  parentheses over all), and upstream's invalid-query behavior --
  **an unparseable filter leaves the data unfiltered**.
  :func:`parse_filter_query` exposes the AST in the
  ``derived_filter_query_structure`` shape.  :func:`sort_rows`
  implements ``sort_by`` (multi-key, in click order, nully last) and
  :func:`paginate_rows` the ``page_action`` slicing.
* :func:`run_scenario` -- a scripted sequence of triggers with the
  intermediate prop snapshots, for testing reactive wiring without a
  browser (dash's own precedent is its 4.x MCP layer, which drives
  the same ``callback_map`` headlessly).

Stated once, honestly: the React renderer, the Flask/Quart/FastAPI
servers, hot reload, background/long callbacks (Diskcache/Celery) and
the browser DataTable UI are external -- this module is the *engine*
a desktop app or test harness needs, not a web server.

Examples
--------
>>> from pharmaml.viz.dashlite import (
...     DashApp, DataTable, Dropdown, Graph, Input, Output, Slider,
...     State, Button, filter_rows, parse_filter_query, sort_rows,
... )
>>> rows = [
...     {"name": "aspirin", "ic50": 12.5, "class": "nsaid"},
...     {"name": "ibuprofen", "ic50": 45.0, "class": "nsaid"},
...     {"name": "metformin", "ic50": 3.2, "class": "biguanide"},
... ]
>>> kept = filter_rows(rows, "{ic50} < 20 && {class} eq nsaid")
>>> [r["name"] for r in kept]
['aspirin']
>>> [r["name"] for r in filter_rows(rows, "{name} icontains ASP")]
['aspirin']
>>> [r["name"] for r in filter_rows(rows, "{name} contains ASP")]
[]
>>> [r["name"] for r in filter_rows(rows, "{ic50} is blank")]
[]
>>> filter_rows(rows, "{ic50} < 5 || is blank") is rows  # invalid: unfiltered
True
>>> parse_filter_query("{ic50} < 5 || is blank")
Traceback (most recent call last):
    ...
ValueError: DataTable filtering syntax is invalid for query: {ic50} < 5 || is blank
>>> filter_rows(rows, "{ic50} << broken") is rows
True
>>> sorted_r = sort_rows(rows, [{"column_id": "ic50", "direction": "desc"}])
>>> [r["name"] for r in sorted_r]
['ibuprofen', 'aspirin', 'metformin']
>>> ast = parse_filter_query("{ic50} > 10")
>>> ast["type"], ast["subType"], ast["right"]["value"]
('relational-operator', '>', 10.0)
>>> app = DashApp("Mini screen")
>>> app.layout = Div(id="root", children=[
...     Dropdown(id="column", options=["ic50", "ic50"], value="ic50"),
...     Slider(id="bins", min=2, max=8, step=1, value=4),
...     Graph(id="hist", figure=None),
... ])
>>> @app.callback(Output("hist", "figure"), Input("bins", "value"),
...               State("column", "value"))
... def rebin(n_bins, column):
...     return {"data": [{"type": "bar", "y": [n_bins], "name": column}]}
>>> changed = app.trigger("bins.value", 6)
>>> sorted(changed)
['bins.value', 'hist.figure']
>>> app.trigger("bins.value", 6)
{}
>>> edges = app.callback_graph()
>>> [("bins.value" in edge[0]) and (edge[1] == "hist.figure") for edge in edges]
[True]
>>> snapshot = run_scenario(app, [("bins.value", 8)])[0]
>>> snapshot["changed"]
['bins.value', 'hist.figure']
>>> snapshot["props"]["hist.figure"]["data"][0]["y"]
[8]
>>> table = DataTable(id="t", columns=[
...     {"id": "name", "name": "Name"}, {"id": "ic50", "name": "IC50"}],
...     data=rows, filter_query="{ic50} < 20", sort_action="native",
...     filter_action="native", page_size=2,
...     sort_by=[{"column_id": "ic50", "direction": "asc"}])
>>> state = derive_table_state(table)
>>> [r["name"] for r in state["derived_virtual_data"]]
['metformin', 'aspirin']
>>> from pharmaml.viz.dashlite import Button  # the counter a headless runtime drives
>>> Button(id="go", children="Go").n_clicks
0
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any, ClassVar

from pharmaml.exceptions import DataValidationError

__all__ = [
    "H3",
    "Button",
    "Component",
    "DashApp",
    "DashLiteError",
    "DataTable",
    "Div",
    "Dropdown",
    "DuplicateCallback",
    "DuplicateIdError",
    "Graph",
    "Input",
    "Interval",
    "Output",
    "P",
    "Slider",
    "State",
    "Store",
    "callback_graph_edges",
    "derive_table_state",
    "filter_rows",
    "paginate_rows",
    "parse_filter_query",
    "run_scenario",
    "sort_rows",
]


# ---------------------------------------------------------------------------
# exceptions (dash/exceptions.py)
# ---------------------------------------------------------------------------


class DashLiteError(DataValidationError):
    """Base of the DashLite validation errors (dash's DashException)."""


class IncorrectTypeException(DashLiteError):
    """Arguments not Output/Input/State or out of order."""


class DuplicateCallback(DashLiteError):
    """Two callbacks claiming the same output."""


class DuplicateIdError(DashLiteError):
    """The same component id twice in one layout."""


class CircularDependency(DashLiteError):
    """A callback chain that feeds its own inputs (our addition)."""


class InvalidCallbackReturnValue(DashLiteError):
    """Wrong number of return values for the declared outputs."""


class ClientsideNotExecutable(DashLiteError):
    """A clientside (JavaScript) callback was triggered headlessly."""


# ---------------------------------------------------------------------------
# dependency objects (dash/dependencies.py)
# ---------------------------------------------------------------------------


class DashDependency:
    """One ``(component_id, component_property)`` dependency."""

    def __init__(self, component_id: str, component_property: str) -> None:
        if not isinstance(component_id, str) or not component_id:
            raise IncorrectTypeException("component_id must be a non-empty string")
        if "." in component_id:
            raise DashLiteError(
                f"component id {component_id!r} may not contain a period "
                "(dash's IDsCantContainPeriods -- the dot separates id from prop)"
            )
        if not isinstance(component_property, str) or not component_property:
            raise IncorrectTypeException("component_property must be a non-empty string")
        self.component_id = component_id
        self.component_property = component_property

    @property
    def prop_id(self) -> str:
        return f"{self.component_id}.{self.component_property}"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.component_id, "property": self.component_property}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.prop_id!r})"


class Output(DashDependency):
    pass


class Input(DashDependency):
    pass


class State(DashDependency):
    pass


def _prop_id_of(dep: Any) -> str:
    if isinstance(dep, DashDependency):
        return dep.prop_id
    raise IncorrectTypeException(
        f"callback arguments must be Output/Input/State objects, got {dep!r}"
    )


# ---------------------------------------------------------------------------
# the component spec (dash/development/base_component.py)
# ---------------------------------------------------------------------------


class Component:
    """Base component: validated kwargs -> ``{"props", "type", "namespace"}``."""

    _namespace = "dashlite"
    _type = "Component"
    _prop_names: tuple[str, ...] = ("id", "children")

    def __init__(self, **kwargs: Any) -> None:
        unknown = [k for k in kwargs if k not in self._prop_names]
        if unknown:
            raise DashLiteError(
                f"The {self._namespace}.{self._type} component received an "
                f"unexpected keyword argument: {unknown[0]!r} (valid: "
                f"{', '.join(self._prop_names)})"
            )
        if "id" in kwargs and "." in str(kwargs["id"]):
            raise DashLiteError(f"component id {kwargs['id']!r} may not contain a period")
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, "id"):
            self.id = None  # type: ignore[assignment]
        if not hasattr(self, "children"):
            self.children = None  # type: ignore[assignment]

    def to_plotly_json(self) -> dict[str, Any]:
        props = {
            name: getattr(self, name)
            for name in self._prop_names
            if hasattr(self, name) and getattr(self, name) is not None
        }
        return {"props": props, "type": self._type, "namespace": self._namespace}

    def _traverse(self) -> list[Component]:
        out = [self]
        children = getattr(self, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                if isinstance(child, Component):
                    out.extend(child._traverse())
        elif isinstance(children, Component):
            out.extend(children._traverse())
        return out

    def find(self, component_id: str) -> Component | None:
        for component in self._traverse():
            if getattr(component, "id", None) == component_id:
                return component
        return None

    def __len__(self) -> int:
        return len(self._traverse())

    def __iter__(self):  # pragma: no cover - convenience
        return iter(c.id for c in self._traverse() if getattr(c, "id", None))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        ident = getattr(self, "id", None)
        return f"{self._type}(id={ident!r})"


class _HTMLComponent(Component):
    _namespace = "dash_html_components"


class Div(_HTMLComponent):
    _type = "Div"
    _prop_names = ("id", "children", "style")


class H3(_HTMLComponent):
    _type = "H3"
    _prop_names = ("id", "children", "style")


class P(_HTMLComponent):
    _type = "P"
    _prop_names = ("id", "children", "style")


class _DCCComponent(Component):
    _namespace = "dash_core_components"


class Graph(_DCCComponent):
    _type = "Graph"
    _prop_names = ("id", "figure", "config", "style")


class Slider(_DCCComponent):
    _type = "Slider"
    _prop_names = ("id", "min", "max", "step", "marks", "value", "vertical")

    def __init__(self, id: str | None = None, **kwargs: Any) -> None:
        super().__init__(id=id, **kwargs)
        if not hasattr(self, "marks"):
            self.marks = None


class Dropdown(_DCCComponent):
    _type = "Dropdown"
    _prop_names = ("id", "options", "value", "multi", "clearable", "placeholder")

    def __init__(self, id: str | None = None, **kwargs: Any) -> None:
        super().__init__(id=id, **kwargs)
        if not hasattr(self, "multi"):
            self.multi = False
        if not hasattr(self, "clearable"):
            self.clearable = True


class Button(_DCCComponent):
    _type = "Button"
    _prop_names = ("id", "children", "n_clicks", "disabled")

    def __init__(self, id: str | None = None, **kwargs: Any) -> None:
        super().__init__(id=id, **kwargs)
        if not hasattr(self, "n_clicks"):
            self.n_clicks = 0


class Store(_DCCComponent):
    _type = "Store"
    _prop_names = ("id", "data", "storage_type")

    def __init__(self, id: str | None = None, **kwargs: Any) -> None:
        super().__init__(id=id, **kwargs)
        if not hasattr(self, "storage_type"):
            self.storage_type = "memory"


class Interval(_DCCComponent):
    _type = "Interval"
    _prop_names = ("id", "interval", "n_intervals", "max_intervals", "disabled")

    def __init__(self, id: str | None = None, **kwargs: Any) -> None:
        super().__init__(id=id, **kwargs)
        if not hasattr(self, "n_intervals"):
            self.n_intervals = 0
        if not hasattr(self, "interval"):
            self.interval = 1000
        if not hasattr(self, "max_intervals"):
            self.max_intervals = -1


class DataTable(_DCCComponent):
    """The DataTable spec plus the derived-state engine attached."""

    _type = "DataTable"
    # declared for the type checker: the props are set through the
    # generic Component.__init__ (and defaulted just below)
    columns: list[dict[str, Any]]
    data: list[dict[str, Any]]
    filter_query: str
    sort_by: list[dict[str, Any]]
    page_action: str
    page_current: int
    page_size: int
    filter_action: str
    sort_action: str
    sort_as_null: list[Any]
    _prop_names = (
        "id", "columns", "data", "filter_query", "sort_by", "page_action",
        "page_current", "page_size", "filter_action", "sort_action",
        "sort_as_null",
    )

    def __init__(self, id: str | None = None, **kwargs: Any) -> None:
        super().__init__(id=id, **kwargs)
        if not hasattr(self, "filter_query"):
            self.filter_query = ""
        if not hasattr(self, "sort_by"):
            self.sort_by = []
        if not hasattr(self, "page_action"):
            self.page_action = "none"
        if not hasattr(self, "page_current"):
            self.page_current = 0
        if not hasattr(self, "page_size"):
            self.page_size = 250
        if not hasattr(self, "filter_action"):
            self.filter_action = "none"
        if not hasattr(self, "sort_action"):
            self.sort_action = "none"
        if not hasattr(self, "sort_as_null"):
            self.sort_as_null = []


# ---------------------------------------------------------------------------
# the DataTable query engine (dash-table's syntax tree, ported)
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_DATE_RE = re.compile(
    r"^\s*(-?\d{4}|\d{2})(-(\d{1,2})(-(\d{1,2})"
    r"([ Tt]([01]?\d|2[0-3])(:([0-5]\d)(:([0-5]\d(\.\d+)?))?"
    r"(Z|z|[+\-]\d{2}:?\d{2})?)?)?)?)?\s*$"
)

_RELATIONAL = ("contains", "eq", "ge", "gt", "le", "lt", "ne", "datestartswith")
_UNARY = (
    "is blank", "is nil", "is bool", "is num", "is str", "is object",
    "is odd", "is even", "is prime",
)


def _lex(query: str) -> list[tuple[str, Any]] | None:
    """Tokenize a filter query; ``None`` when the text is not well-formed.

    Tokens are ``(kind, value)``: ``block-open``/``block-close``,
    ``logical``/``relational``/``unary`` operators, or ``expression``
    (a ``(value, is_field)`` pair -- dash's two expression subtypes).
    Operators are matched before bare words exactly as dash's lexicon
    orders them, with word-boundary lookaheads so values like
    ``andromeda`` stay values.
    """
    unary_re = re.compile(
        r"is\s+(blank|nil|bool|num|str|object|odd|even|prime)(?=[\s)]|$)",
        re.IGNORECASE,
    )
    word_re = re.compile(
        r"(?:(i|s))?(eq|ne|gt|ge|lt|le|contains|datestartswith)(?=[\s)]|$)|^(and|or)(?=[\s)]|$)",
        re.IGNORECASE,
    )
    symbols = ((">=", "ge"), ("<=", "le"), ("!=", "ne"), ("<", "lt"), (">", "gt"), ("=", "eq"))
    tokens: list[tuple[str, Any]] = []
    i, n = 0, len(query)
    while i < n:
        ch = query[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(("block-open", "("))
            i += 1
            continue
        if ch == ")":
            tokens.append(("block-close", ")"))
            i += 1
            continue
        if query.startswith("&&", i):
            tokens.append(("logical", "and"))
            i += 2
            continue
        if query.startswith("||", i):
            tokens.append(("logical", "or"))
            i += 2
            continue
        if ch == "!":
            if query.startswith("!=", i):
                tokens.append(("relational", "ne"))
                i += 2
            else:
                tokens.append(("unary", "!"))
                i += 1
            continue
        match = unary_re.match(query, i)
        if match:
            tokens.append(("unary", "is " + match.group(1).lower()))
            i = match.end()
            continue
        match = word_re.match(query, i)
        if match:
            if match.group(3):  # and/or carry no case prefix
                tokens.append(("logical", match.group(3).lower()))
            else:
                fold = bool(match.group(1) and match.group(1).lower() == "i")
                tokens.append(("relational", (match.group(2).lower(), fold)))
            i = match.end()
            continue
        fold = False
        pos = i
        if query[pos] in "iIsS" and pos + 1 < n and query[pos + 1] in "=<>!":
            fold = query[pos].lower() == "i"
            pos += 1
        for symbol, name in symbols:
            if query.startswith(symbol, pos):
                tokens.append(("relational", (name, fold)))
                i = pos + len(symbol)
                break
        else:
            if ch == "{":
                end = query.find("}", i)
                if end < 0:
                    return None
                field = query[i + 1 : end]
                if not field:
                    return None
                tokens.append(("expression", (field, True)))
                i = end + 1
                continue
            if ch in "'\"`":
                end = query.find(ch, i + 1)
                if end < 0:
                    return None
                tokens.append(("expression", (query[i + 1 : end], False)))
                i = end + 1
                continue
            j = i
            while j < n and not query[j].isspace() and query[j] not in "(){}'\"`":
                j += 1
            word = query[i:j]
            i = j
            if not word:
                return None
            # bare value: numeric tokens become numbers (dash's isNumeric rule)
            if _NUMERIC_RE.match(word):
                tokens.append(("expression", (float(word), False)))
            else:
                tokens.append(("expression", (word, False)))
    return tokens or None


class _QueryParser:
    """Recursive-descent parser over the lexed query (dash's priorities).

    The pivot is the token at nesting 0 with the highest priority
    (``or`` 3, ``and`` 2, ``!`` 1.5, block-open 1, everything else 0);
    a top-level parenthesis pair pivots as a block and is stripped.
    """

    _PRIORITY: ClassVar[dict[str, float]] = {"or": 3.0, "and": 2.0, "!": 1.5}

    def __init__(self, tokens: list[tuple[str, Any]]) -> None:
        self.tokens = tokens

    def parse(self) -> dict[str, Any] | None:
        if not self.tokens:
            return None
        nesting = 0
        pivot: tuple[str, Any] | None = None
        pivot_priority = -1.0
        pivot_idx = -1
        for k, (kind, value) in enumerate(self.tokens):
            if kind == "block-open":
                if nesting == 0 and 1.0 > pivot_priority:
                    pivot, pivot_priority, pivot_idx = (kind, value), 1.0, k
                nesting += 1
            elif kind == "block-close":
                nesting -= 1
                if nesting < 0:
                    return None
            elif nesting == 0 and kind in ("logical", "unary", "relational"):
                priority = self._PRIORITY.get(value, 0.0) if kind == "logical" else (
                    1.5 if kind == "unary" and value == "!" else 0.0
                )
                if priority > pivot_priority:
                    pivot, pivot_priority, pivot_idx = (kind, value), priority, k
        if nesting != 0:
            return None
        if pivot is None:
            return self._leaf()
        kind, value = pivot
        if kind == "block-open":
            # a wrapping parenthesis: the close must be the last token
            if self.tokens[-1][0] != "block-close":
                return None
            return _QueryParser(self.tokens[1:-1]).parse()
        if kind == "logical":
            left = _QueryParser(self.tokens[:pivot_idx]).parse()
            right = _QueryParser(self.tokens[pivot_idx + 1 :]).parse()
            if left is None or right is None:
                return None
            return {
                "type": "logical-operator",
                "subType": "&&" if value == "and" else "||",
                "left": left,
                "right": right,
            }
        if value == "!":
            right = _QueryParser(self.tokens[pivot_idx + 1 :]).parse()
            if right is None:
                return None
            return {"type": "unary-operator", "subType": "!", "block": right}
        if kind == "relational":
            name, fold = value if isinstance(value, tuple) else (value, False)
            left = _QueryParser(self.tokens[:pivot_idx]).parse()
            right = _QueryParser(self.tokens[pivot_idx + 1 :]).parse()
            if left is None or right is None:
                return None
            return {
                "type": "relational-operator", "subType": self._symbol(str(name)),
                "caseInsensitive": fold, "left": left, "right": right,
            }
        # a field-attached unary operator (is blank / is num / ...)
        left = _QueryParser(self.tokens[:pivot_idx]).parse()
        if left is None:
            return None
        return {"type": "unary-operator", "subType": value, "left": left}

    def _leaf(self) -> dict[str, Any] | None:
        if len(self.tokens) != 1:
            return None
        kind, value = self.tokens[0]
        if kind == "expression":
            text, is_field = value
            return {
                "type": "expression",
                "subType": "field" if is_field else "value",
                "value": text,
            }
        return None

    @staticmethod
    def _symbol(name: str) -> str:
        return {
            "eq": "=", "ne": "!=", "ge": ">=", "gt": ">", "le": "<=", "lt": "<",
            "contains": "contains", "datestartswith": "datestartswith",
        }[name]


def parse_filter_query(query: str) -> dict[str, Any]:
    """Parse ``query`` into dash's ``derived_filter_query_structure`` AST.

    Raises the upstream error message for invalid queries (dash's
    DataTable raises when the AST is *evaluated* on an invalid tree;
    :func:`filter_rows` instead follows the app-level behavior of
    leaving data unfiltered).
    """
    tokens = _lex(query.strip())
    if tokens is None:
        raise ValueError(
            f"DataTable filtering syntax is invalid for query: {query}"
        )
    tree = _QueryParser(tokens).parse()
    if tree is None:
        raise ValueError(
            f"DataTable filtering syntax is invalid for query: {query}"
        )
    return tree


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return bool(_NUMERIC_RE.match(value.strip()))
    return False


def _normalize_date(value: Any) -> str | None:
    if not isinstance(value, str) or not _DATE_RE.match(value):
        return None
    parts = value.strip().split("-")
    year = parts[0]
    if len(year) == 2:
        century = 2000 if int(year) <= 68 else 1900
        year = str(century + int(year))
    out = year
    if len(parts) > 1:
        out += f"-{int(parts[1]):02d}"
        if len(parts) > 2:
            day = parts[2]
            out += f"-{int(day[:2]):02d}"
            if len(day) > 2:
                out += day[2:]
    return out


def _eval_node(node: dict[str, Any], row: dict[str, Any]) -> bool:
    kind = node["type"]
    if kind == "logical-operator":
        left = _eval_node(node["left"], row)
        right = _eval_node(node["right"], row)
        return (left and right) if node["subType"] == "&&" else (left or right)
    if kind == "unary-operator":
        sub = node["subType"]
        if sub == "!":
            return not _eval_node(node["block"], row)
        target = _resolve(node.get("left") or {}, row)
        if sub == "is blank":
            return target is None or target == ""
        if sub == "is nil":
            return target is None
        if sub == "is bool":
            return isinstance(target, bool)
        if sub == "is num":
            return isinstance(target, (int, float)) and not isinstance(target, bool)
        if sub == "is str":
            return isinstance(target, str)
        if sub == "is object":
            return isinstance(target, dict)
        if sub in ("is odd", "is even"):
            if not isinstance(target, (int, float)) or isinstance(target, bool):
                return False
            return int(target) % 2 == (1 if sub == "is odd" else 0)
        if sub == "is prime":
            return _is_prime(target)
        return False
    if kind == "relational-operator":
        left = _resolve(node["left"], row)
        right = _resolve(node["right"], row)
        sub = node["subType"]
        fold = bool(node.get("caseInsensitive"))

        def _text(value: Any) -> str:
            text = str(value)
            return text.upper() if fold else text

        if sub == "contains":
            if left is None or right is None:
                return False
            if not isinstance(left, str) and not isinstance(right, str):
                return False
            return _text(right) in _text(left)
        if sub == "datestartswith":
            lo = _normalize_date(left)
            hi = _normalize_date(right)
            if lo is None or hi is None:
                return False
            return lo.startswith(hi)
        if sub == "=":
            if _is_numeric(left) and _is_numeric(right):
                return float(left) == float(right)
            return _text(left) == _text(right)
        if sub == "!=":
            if _is_numeric(left) and _is_numeric(right):
                return float(left) != float(right)
            return _text(left) != _text(right)
        if left is None or right is None:
            return False
        lhs: Any
        rhs: Any
        if _is_numeric(left) and _is_numeric(right):
            lhs, rhs = float(left), float(right)
        else:
            lhs, rhs = str(left), str(right)
        if sub == ">":
            return lhs > rhs
        if sub == ">=":
            return lhs >= rhs
        if sub == "<":
            return lhs < rhs
        if sub == "<=":
            return lhs <= rhs
    return False


def _resolve(node: dict[str, Any], row: dict[str, Any]) -> Any:
    if node.get("type") == "expression":
        if node.get("subType") == "field":
            return row.get(str(node.get("value")))
        return node.get("value")
    return None


def _is_prime(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    n = int(value)
    if n < 2 or n != value:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    limit = math.isqrt(n) + 1
    for d in range(3, limit, 2):
        if n % d == 0:
            return False
    return True


def filter_rows(rows: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Apply a ``filter_query`` to row dicts (dash's evaluation rules).

    An invalid query returns the rows unchanged -- the exact behavior
    of dash's ``derived/data/virtual`` layer; a *structurally valid*
    query whose evaluation hits a type mismatch simply selects no
    rows for that comparison.
    """
    try:
        tree = parse_filter_query(query.strip())
    except ValueError:
        return rows
    return [row for row in rows if _eval_node(tree, row)]


def sort_rows(
    rows: list[dict[str, Any]],
    sort_by: list[dict[str, Any]],
    sort_as_null: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Multi-key sort in click order; nully values sort last both ways.

    Comparators chain left-to-right (dash's ``R.sortWith``), nully is
    ``None`` or a value listed in that column's ``sort_as_null``.
    """
    if not sort_by:
        return list(rows)
    null_values = set(sort_as_null or [])

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        parts: list[Any] = []
        for spec in sort_by:
            column = str(spec.get("column_id"))
            value = row.get(column)
            descending = str(spec.get("direction")) == "desc"
            nully = value is None or value in null_values
            numeric = _is_numeric(value)
            rank = 1 if nully else 0
            body: Any
            if descending:
                # invert the value while keeping nully last
                if nully:
                    body = 0
                elif numeric and value is not None:
                    body = -float(value)
                else:
                    body = _ReverseStr(str(value))
            else:
                body = 0 if nully or value is None else (
                    float(value) if numeric else str(value)
                )
            parts.append((rank, body))
        return tuple(parts)

    return sorted(rows, key=sort_key)


class _ReverseStr:
    """A wrapper that sorts descending for string columns."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __lt__(self, other: _ReverseStr) -> bool:
        return self.text > other.text

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ReverseStr) and self.text == other.text


def paginate_rows(
    rows: list[dict[str, Any]], page_current: int = 0, page_size: int = 250
) -> list[dict[str, Any]]:
    """The ``page_action='native'`` slice (with clamped page_current)."""
    if page_size <= 0:
        return list(rows)
    last_page = max(0, (len(rows) - 1) // page_size)
    page = min(max(int(page_current), 0), last_page)
    return rows[page * page_size : (page + 1) * page_size]


def derive_table_state(table: DataTable) -> dict[str, Any]:
    """Compute a DataTable's derived props (the front-end's own pipeline).

    Applies ``filter_query`` (when ``filter_action`` is ``native``),
    then ``sort_by`` (when ``sort_action`` is ``native``), then the
    pagination slice -- the order dash's ``derived/data/virtual``
    layer documents.
    """
    rows = list(table.data or [])
    if str(table.filter_action) == "native" and table.filter_query:
        rows = filter_rows(rows, table.filter_query)
    if str(table.sort_action) == "native" and table.sort_by:
        rows = sort_rows(rows, list(table.sort_by), list(table.sort_as_null or []))
    indices = [i for i, row in enumerate(table.data or []) if row in rows]
    indices = _stable_indices(table.data or [], rows)
    viewport = paginate_rows(rows, int(table.page_current), int(table.page_size)) \
        if str(table.page_action) == "native" else rows
    return {
        "derived_virtual_data": rows,
        "derived_virtual_indices": indices,
        "derived_viewport_data": viewport,
        "n_rows": len(table.data or []),
        "n_filtered": len(rows),
    }


def _stable_indices(data: list[dict[str, Any]], kept: list[dict[str, Any]]) -> list[int]:
    out: list[int] = []
    pool = {i: row for i, row in enumerate(data)}
    for row in kept:
        for i, source in pool.items():
            if i not in out and source == row:
                out.append(i)
                break
    return out


# ---------------------------------------------------------------------------
# the app (dash/dash.py + _callback.py, headless)
# ---------------------------------------------------------------------------


class DashApp:
    """A headless dash application: layout + the callback graph.

    ``trigger`` is the browser renderer's dispatch loop, minus the
    browser: changed props fire every not-yet-run callback that reads
    them, outputs propagate round by round until nothing new can run.
    Cycle detection happens at registration time (dash's server side
    has none -- the React renderer catches cycles; a headless engine
    must not hang on them).
    """

    def __init__(self, title: str = "DashLite app") -> None:
        self.title = title
        self._layout: Component | None = None
        self.callback_map: dict[str, dict[str, Any]] = {}
        self.callback_list: list[dict[str, Any]] = []

    # -- layout ---------------------------------------------------------- #

    @property
    def layout(self) -> Component | None:
        return self._layout

    @layout.setter
    def layout(self, value: Component) -> None:
        if not isinstance(value, Component):
            raise DashLiteError("layout must be a Component tree")
        self._layout = value

    def validate_layout(self) -> list[str]:
        """Check duplicate ids; return the id list (dash's validate_layout)."""
        if self._layout is None:
            raise DashLiteError("no layout set")
        ids: list[str] = []
        for component in self._layout._traverse():
            ident = getattr(component, "id", None)
            if ident is not None:
                if ident in ids:
                    raise DuplicateIdError(f"duplicate component id {ident!r}")
                ids.append(ident)
        return ids

    # -- callback registration ------------------------------------------- #

    def callback(self, *args: Any, prevent_initial_call: bool = False) -> Callable:
        """The ``@app.callback`` decorator, upstream's validation rules.

        Outputs first, then Inputs, then States (the exact ordering
        rule from ``dash/_validate.py``); at least one Output and one
        Input; duplicate outputs raise :class:`DuplicateCallback`.
        """
        deps = [a for a in args if isinstance(a, (list, tuple))]
        flat: list[DashDependency] = []
        for a in args:
            if isinstance(a, (list, tuple)):
                flat.extend(a)
            else:
                flat.append(a)  # type: ignore[arg-type]
        if deps and not flat:
            raise IncorrectTypeException("callback needs at least one Output and one Input")
        kinds = [type(d).__name__ for d in flat]
        order = {"Output": 0, "Input": 1, "State": 2}
        ranks = [order.get(k, 9) for k in kinds]
        if 9 in ranks:
            raise IncorrectTypeException(
                "callback arguments must be Output, Input or State objects "
                "(you must provide all Outputs first, then all Inputs, then all States)"
            )
        if ranks != sorted(ranks):
            raise IncorrectTypeException(
                "callback arguments out of order: all Outputs first, then all "
                "Inputs, then all States"
            )
        outputs = [d for d in flat if isinstance(d, Output)]
        inputs = [d for d in flat if isinstance(d, Input)]
        states = [d for d in flat if isinstance(d, State)]
        if not outputs or not inputs:
            raise IncorrectTypeException(
                "callback needs at least one Output and one Input"
            )
        callback_id = ".." + "...".join(d.prop_id for d in outputs) + ".." \
            if len(outputs) > 1 else outputs[0].prop_id
        if any(callback_id in entry["output"] for entry in self.callback_list):
            raise DuplicateCallback(
                f"the output {callback_id!r} is already claimed by another callback"
            )

        def register(fn: Callable) -> Callable:
            self.callback_map[callback_id] = {
                "inputs": [d.to_dict() for d in inputs],
                "state": [d.to_dict() for d in states],
                "outputs": [d.to_dict() for d in outputs],
                "fn": fn,
                "prevent_initial_call": prevent_initial_call,
            }
            self.callback_list.append(
                {
                    "output": callback_id,
                    "inputs": [d.to_dict() for d in inputs],
                    "state": [d.to_dict() for d in states],
                    "clientside_function": None,
                    "prevent_initial_call": bool(prevent_initial_call),
                }
            )
            self._check_cycles()
            return fn

        return register

    def clientside_callback(self, clientside_function: Any, *args: Any) -> None:
        """Register a JS callback spec -- honest: it cannot run headless."""
        outputs = [a for a in args if isinstance(a, Output)]
        inputs = [a for a in args if isinstance(a, Input)]
        states = [a for a in args if isinstance(a, State)]
        if not outputs or not inputs:
            raise IncorrectTypeException(
                "clientside_callback needs at least one Output and one Input"
            )
        callback_id = outputs[0].prop_id
        self.callback_map[callback_id] = {
            "inputs": [d.to_dict() for d in inputs],
            "state": [d.to_dict() for d in states],
            "outputs": [d.to_dict() for d in outputs],
            "clientside_function": {
                "namespace": getattr(clientside_function, "namespace", "dash_clientside"),
                "function_name": getattr(clientside_function, "function_name", str(clientside_function)),
            },
            "prevent_initial_call": False,
        }
        self.callback_list.append(
            {
                "output": callback_id,
                "inputs": [d.to_dict() for d in inputs],
                "state": [d.to_dict() for d in states],
                "clientside_function": {
                    "namespace": getattr(clientside_function, "namespace", "dash_clientside"),
                    "function_name": getattr(clientside_function, "function_name", str(clientside_function)),
                },
                "prevent_initial_call": False,
            }
        )
        self._check_cycles()

    def _check_cycles(self) -> None:
        """Topo-check the input->output edges (our addition over dash).

        Only *Input* dependencies propagate (States never fire a
        callback), so cycles are computed over the input edges alone;
        Kahn's algorithm -- any node left with positive in-degree sits
        on a cycle.
        """
        edge_set: set[tuple[str, str]] = set()
        nodes: set[str] = set()
        for entry in self.callback_list:
            outputs = self._expand_outputs(entry)
            for dep in entry["inputs"]:
                src = dep["id"] + "." + dep["property"]
                nodes.add(src)
                for out in outputs:
                    nodes.add(out)
                    edge_set.add((src, out))
        indegree = {node: 0 for node in nodes}
        adjacency: dict[str, list[str]] = {node: [] for node in nodes}
        for src, dst in edge_set:
            adjacency[src].append(dst)
            indegree[dst] += 1
        queue = [node for node in nodes if indegree[node] == 0]
        seen = 0
        while queue:
            node = queue.pop()
            seen += 1
            for nxt in adjacency[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        if seen != len(nodes):
            cyclic = sorted(node for node in nodes if indegree[node] > 0)
            raise CircularDependency(
                "Circular Dependencies: " + ", ".join(cyclic)
            )

    @staticmethod
    def _expand_outputs(entry: dict[str, Any]) -> list[str]:
        output = entry["output"]
        if isinstance(output, str) and output.startswith("..") and output.endswith(".."):
            body = output[2:-2]
            return [part for part in body.split("...") if part]
        return [output]

    # -- the headless dispatch ------------------------------------------- #

    def _component(self, component_id: str) -> Component | None:
        if self._layout is None:
            return None
        return self._layout.find(component_id)

    def get_prop(self, prop_id: str) -> Any:
        """Read a live prop from the layout (``id.property`` key)."""
        ident, _, prop = prop_id.partition(".")
        component = self._component(ident)
        if component is None:
            raise DashLiteError(f"no component with id {ident!r} in the layout")
        if not hasattr(component, prop):
            raise DashLiteError(
                f"component {ident!r} has no prop {prop!r} "
                f"(valid: {', '.join(type(component)._prop_names)})"
            )
        return getattr(component, prop)

    def _set_prop(self, prop_id: str, value: Any) -> None:
        ident, _, prop = prop_id.partition(".")
        component = self._component(ident)
        if component is None:
            raise DashLiteError(f"no component with id {ident!r} in the layout")
        if prop not in type(component)._prop_names:
            raise DashLiteError(
                f"component {ident!r} has no prop {prop!r} "
                f"(valid: {', '.join(type(component)._prop_names)})"
            )
        setattr(component, prop, value)

    def trigger(self, prop_id: str, value: Any) -> dict[str, Any]:
        """Dispatch one prop change through the callback graph.

        Fires every not-yet-run callback whose *input* (not State)
        changed, propagating outputs round by round -- the renderer's
        semantics, including its no-op rule: assigning a value equal
        to the current one fires nothing.  Returns every changed prop
        (including the seed).
        """
        if self.get_prop(prop_id) == value:
            return {}
        self._set_prop(prop_id, value)
        changed = {prop_id: value}
        executed: set[str] = set()
        while True:
            fired = False
            for entry in self.callback_list:
                callback_id = entry["output"]
                if callback_id in executed:
                    continue
                input_ids = [
                    d["id"] + "." + d["property"] for d in entry["inputs"]
                ]
                if not any(pid in changed for pid in input_ids):
                    continue
                info = self.callback_map.get(callback_id)
                if info is None:  # pragma: no cover - lists stay in sync
                    continue
                if info.get("clientside_function"):
                    raise ClientsideNotExecutable(
                        f"callback {callback_id!r} is a clientside (JavaScript) "
                        "function -- a headless runtime cannot execute it "
                        "(dash runs it in the browser)"
                    )
                args = [changed.get(pid, self.get_prop(pid)) for pid in input_ids]
                state_ids = [d["id"] + "." + d["property"] for d in entry["state"]]
                args.extend(self.get_prop(pid) for pid in state_ids)
                result = info["fn"](*args)
                outputs = self._expand_outputs(entry)
                values = result if isinstance(result, (list, tuple)) else [result]
                if len(values) != len(outputs):
                    raise InvalidCallbackReturnValue(
                        f"callback {callback_id!r} returned {len(values)} values "
                        f"for {len(outputs)} declared outputs"
                    )
                for out, out_value in zip(outputs, values):
                    self._set_prop(out, out_value)
                    changed[out] = out_value
                executed.add(callback_id)
                fired = True
            if not fired:
                break
        return changed

    def run_initial(self) -> dict[str, Any]:
        """Fire initial callbacks (``prevent_initial_call=False`` ones)."""
        changed = {
            d["id"] + "." + d["property"]: self.get_prop(d["id"] + "." + d["property"])
            for entry in self.callback_list
            for d in entry["inputs"]
        }
        executed: set[str] = set()
        while True:
            fired = False
            for entry in self.callback_list:
                callback_id = entry["output"]
                if callback_id in executed or entry.get("prevent_initial_call"):
                    continue
                info = self.callback_map.get(callback_id)
                if info is None or info.get("clientside_function"):  # pragma: no cover
                    continue
                input_ids = [d["id"] + "." + d["property"] for d in entry["inputs"]]
                args = [self.get_prop(pid) for pid in input_ids]
                state_ids = [d["id"] + "." + d["property"] for d in entry["state"]]
                args.extend(self.get_prop(pid) for pid in state_ids)
                result = info["fn"](*args)
                outputs = self._expand_outputs(entry)
                values = result if isinstance(result, (list, tuple)) else [result]
                for out, out_value in zip(outputs, values):
                    self._set_prop(out, out_value)
                    changed[out] = out_value
                executed.add(callback_id)
                fired = True
            if not fired:
                break
        return changed

    def callback_graph(self) -> list[tuple[str, str]]:
        """The dependency edges as ``(input prop, output prop)`` pairs."""
        return callback_graph_edges(self)

    def describe_graph(self) -> str:
        edges = self.callback_graph()
        if not edges:
            return "no callbacks registered"
        lines = [f"{src}  ->  {dst}" for src, dst in edges]
        return "\n".join(lines)


def callback_graph_edges(app: DashApp) -> list[tuple[str, str]]:
    """Extract ``(input, output)`` edges from an app's callback list."""
    edges: list[tuple[str, str]] = []
    for entry in app.callback_list:
        outputs = DashApp._expand_outputs(entry)
        for dep in entry["inputs"]:
            src = dep["id"] + "." + dep["property"]
            for out in outputs:
                edges.append((src, out))
    return edges


def run_scenario(
    app: DashApp, events: list[tuple[str, Any]]
) -> list[dict[str, Any]]:
    """Drive a sequence of triggers, snapshotting after each one."""
    snapshots: list[dict[str, Any]] = []
    for prop_id, value in events:
        changed = app.trigger(prop_id, value)
        snapshots.append(
            {
                "event": (prop_id, value),
                "changed": sorted(changed),
                "props": {pid: app.get_prop(pid) for pid in sorted(changed)},
            }
        )
    return snapshots
