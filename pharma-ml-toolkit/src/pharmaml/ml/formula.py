r"""The statsmodels/patsy formula language, absorbed natively.

``pydata/patsy`` + ``statsmodels/statsmodels`` (reviewed together in
this round's original pass and re-derived here from their published
grammar) give R its ``y ~ x1 + x2`` interface: a small declarative
language over a data frame that expands into a **design matrix** with
documented column names, categorical contrasts and term slices.  The
R/C integration, the formula *compiler cache* and the CRAN-scale
model zoo are what this toolkit does not vendor; the language and
its semantics transfer natively:

* the **grammar**, parsed by a real recursive-descent parser:
  ``~`` separates response and predictors; ``+``/``-`` add and remove
  terms; ``:`` interacts; ``*`` crosses (``a*b`` = ``a + b + a:b``);
  ``/`` nests (``a/b`` = ``a + a:b``); ``(a + b)**2`` crosses to a
  depth; ``-1``/``+0`` drop the intercept; ``C(x)`` marks a
  categorical with its contrast; ``I(expr)`` splices a numeric
  Python expression in verbatim.
* the **coding layer**: treatment (reference-level dummies, patsy's
  default), sum and Helmert contrasts (R's ``contr.helmert``
  convention), auto-categorical detection for string columns,
  boolean columns coded ``[T.True]``, interaction columns as
  pairwise products with ``a:b``-style names.
* :func:`design_matrices` — the matrix itself, as a named-column
  frame plus :class:`DesignInfo` with ``term_slices`` (which columns
  belong to which term — the thing ANOVA needs and one-hot helpers
  never give you).
* :func:`fit_ols_formula` — least squares through the formula, with
  standard errors, t values and **two-sided p-values from a
  self-contained incomplete-beta t-distribution** (no scipy needed).
* :func:`anova_ii` — Type-II sums of squares from the term slices
  (drop-each-term against every model that does not *contain* it,
  the containment rule that makes Type-II the honest default).

The honest scope, stated once: no Pandas back-ends beyond the frame
passed in, no codegen/caching (formulas are parsed per call —
teaching scale), ``I()`` trusts its author exactly like patsy does,
and orthogonal-polynomial contrasts are left out (the numeric path
covers the analysis; ``C(x, Poly)`` raises instead of lying).

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.formula import design_matrices, fit_ols_formula
>>> data = {
...     "y": [2.0, 4.1, 6.1, 8.0],
...     "x": [1.0, 2.0, 3.0, 4.0],
...     "g": ["a", "a", "b", "b"],
... }
>>> info = design_matrices("y ~ x * g", data)
>>> info.columns
['Intercept', 'x', 'g[T.b]', 'g[T.b]:x']
>>> design = info.design
>>> design.shape
(4, 4)
>>> result = fit_ols_formula("y ~ x", {"y": [1.0, 3.0, 5.0], "x": [0.0, 1.0, 2.0]})
>>> [round(v, 3) for v in result.params]
[1.0, 2.0]
>>> float(round(result.r_squared, 6))
1.0
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "ANOVARow",
    "DesignInfo",
    "Factor",
    "Term",
    "anova_ii",
    "contrast_matrix",
    "design_matrices",
    "fit_ols_formula",
    "parse_formula",
]


# --------------------------------------------------------------------------
# the term algebra
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Factor:
    """One predictor inside a term: numeric, categorical or intercept.

    Attributes
    ----------
    kind : {"numeric", "categorical", "intercept"}
    expr : str
        The data column (or ``"1"`` for the intercept).
    contrast : {"Treatment", "Sum", "Helmert"}
    reference : str | None
        The treatment coding's reference level (default: first level).
    """

    kind: str
    expr: str
    contrast: str = "Treatment"
    reference: str | None = None
    raw_expr: str | None = field(default=None, compare=False)

    def __repr__(self) -> str:
        if self.kind == "intercept":
            return "1"
        if self.kind == "categorical":
            if self.reference is not None:
                return f"C({self.expr}, Treatment({self.reference!r}))"
            return f"C({self.expr}, {self.contrast})" if self.contrast != "Treatment" else (
                f"C({self.expr})"
            )
        return self.expr


@dataclass(frozen=True)
class Term:
    """A product of factors (``a:b``); a single factor is a main effect."""

    factors: frozenset[Factor]

    @property
    def degree(self) -> int:
        return len(self.factors)

    def __repr__(self) -> str:
        return ":".join(sorted(repr(f) for f in self.factors))

    def contains(self, other: Term) -> bool:
        """Whether this term's factors include all of ``other``'s."""

        return other.factors <= self.factors


INTERCEPT = Term(frozenset({Factor("intercept", "1")}))


# --------------------------------------------------------------------------
# the tokenizer
# --------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(
    r"""
    \s*(?:
        (?P<op>\*\*|[-+*/:~^(),])          |
        (?P<number>\d+\.?\d*)              |
        (?P<quoted>`[^`]+`)                |
        (?P<string>'[^']*'|"[^"]*")        |
        (?P<name>[A-Za-z_][A-Za-z0-9_.]*)
    )
    """,
    re.VERBOSE,
)


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(text):
        match = _TOKEN_PATTERN.match(text, position)
        if match is None or match.end() == position:
            if text[position:].strip() == "":
                break
            raise DataValidationError(
                f"formula: cannot tokenize from {text[position:position+12]!r}"
            )
        position = match.end()
        kind = match.lastgroup or ""
        value = match.group(kind)
        if kind == "quoted":
            tokens.append(("name", value[1:-1]))
        elif kind == "string":
            tokens.append(("string", value[1:-1]))
        else:
            tokens.append((kind, value))
    return tokens


# --------------------------------------------------------------------------
# the parser
# --------------------------------------------------------------------------


class _FormulaParser:
    """Recursive-descent parser for the patsy grammar."""

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> tuple[str, str] | None:
        token = self.peek()
        if token is not None:
            self.pos += 1
        return token

    def expect_op(self, op: str) -> None:
        token = self.next()
        if token is None or token[0] != "op" or token[1] != op:
            raise DataValidationError(f"formula: expected {op!r}, got {token!r}")

    def parse(self) -> tuple[str | None, list[Term]]:
        lhs: str | None = None
        first = self.peek()
        if first is not None and first[0] == "name":
            following = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else None
            if following is not None and following[0] == "op" and following[1] == "~":
                lhs = first[1]
                self.next()
                self.next()
        terms = self.parse_sum()
        tail = self.peek()
        if tail is not None:
            raise DataValidationError(f"formula: trailing tokens from {tail!r}")
        if not terms:
            raise DataValidationError("formula: the right-hand side is empty")
        return lhs, terms

    def parse_sum(self) -> list[Term]:
        """``+``/``-`` accumulation over the implicit-intercept model."""

        terms = [INTERCEPT]
        sign = "+"
        if self.peek() == ("op", "-"):
            self.next()
            sign = "-"
        terms = self._apply(terms, sign, self.parse_piece())
        while True:
            token = self.peek()
            if token is None or token[0] != "op" or token[1] not in ("+", "-"):
                return terms
            sign = token[1]
            self.next()
            terms = self._apply(terms, sign, self.parse_piece())

    @staticmethod
    def _apply(terms: list[Term], sign: str, chunk: list[Term]) -> list[Term]:
        # "+0" / "-1" (and "+0" inside a chunk) drop the intercept
        if any(
            f.kind == "intercept" and f.expr == "0"
            for t in chunk
            for f in t.factors
        ):
            return [t for t in terms if t != INTERCEPT]
        if sign == "+":
            existing = {t.factors for t in terms}
            return terms + [t for t in chunk if t.factors not in existing]
        drop = {t.factors for t in chunk}
        return [t for t in terms if t.factors not in drop]

    def parse_piece(self) -> list[Term]:
        """``*`` (full crossing) and ``/`` (nesting) over segments."""

        terms = self.parse_segment()
        while True:
            token = self.peek()
            if token is None or token[0] != "op" or token[1] not in ("*", "/"):
                return terms
            operator = token[1]
            self.next()
            rhs = self.parse_segment()
            if operator == "*":
                merged = list(terms) + list(rhs)
                terms = terms + rhs + [
                    Term(frozenset().union(*(t.factors for t in combo)))
                    for size in range(2, len(merged) + 1)
                    for combo in _combinations(merged, size)
                ]
                terms = self._dedup(terms)
            else:  # nesting: a/b = a + a:b (right-associative chain)
                left = terms[-1] if terms else INTERCEPT
                terms = terms[:-1] if terms else []
                nested = [left]
                for right in rhs:
                    nested.append(Term(left.factors | right.factors))
                terms = terms + nested

    def parse_segment(self) -> list[Term]:
        """A parenthesised sum (with optional ``**depth``) or one term."""

        if self.peek() == ("op", "("):
            self.next()
            inner = self.parse_sum()
            self.expect_op(")")
            token = self.peek()
            if token is not None and token[0] == "op" and token[1] in ("**", "^"):
                self.next()
                depth = self.next()
                if depth is None or depth[0] != "number":
                    raise DataValidationError("formula: ** needs a numeric depth")
                power = int(float(depth[1]))
                if power < 1:
                    raise DataValidationError("formula: ** depth must be >= 1")
                factors = [
                    f for t in inner if t != INTERCEPT for f in t.factors
                ]
                return [
                    Term(frozenset(combo))
                    for size in range(1, power + 1)
                    for combo in _combinations_factors(factors, size)
                ]
            return [t for t in inner if t != INTERCEPT] or [INTERCEPT]
        return [self.parse_term()]

    def parse_term(self) -> Term:
        """``:``-joined factors (an interaction is their product)."""

        factors = [self.parse_factor()]
        while self.peek() == ("op", ":"):
            self.next()
            factors.append(self.parse_factor())
        return Term(frozenset(factors))

    def parse_factor(self) -> Factor:
        token = self.peek()
        if token is None:
            raise DataValidationError("formula: a factor was expected, input ended")
        if token[0] == "op" and token[1] in ("+", "-", "~", "*", "/"):
            raise DataValidationError(f"formula: a factor was expected, got {token[1]!r}")
        if token[0] == "number":
            self.next()
            if token[1] in ("0", "1"):
                return Factor("intercept", token[1])
            return Factor("numeric", token[1])
        if token[0] != "name":
            raise DataValidationError(f"formula: unexpected token {token!r}")
        name = token[1]
        self.next()
        if name == "C":
            self.expect_op("(")
            column = self.next()
            if column is None or column[0] not in ("name", "string"):
                raise DataValidationError("formula: C() needs a column name")
            contrast = "Treatment"
            reference: str | None = None
            if self.peek() == ("op", ","):
                self.next()
                spec = self.next()
                if spec is None or spec[0] != "name":
                    raise DataValidationError("formula: C() contrast expected")
                contrast = spec[1]
                if self.peek() == ("op", "("):
                    self.next()
                    ref = self.next()
                    if ref is None or ref[0] not in ("name", "string", "number"):
                        raise DataValidationError("formula: reference level expected")
                    reference = ref[1]
                    self.expect_op(")")
                if contrast not in ("Treatment", "Sum", "Helmert"):
                    raise DataValidationError(
                        f"formula: unknown contrast {contrast!r} "
                        "(Treatment, Sum, Helmert)"
                    )
            self.expect_op(")")
            return Factor("categorical", column[1], contrast, reference)
        if name == "I":
            self.expect_op("(")
            depth = 1
            pieces: list[str] = []
            while True:
                inner = self.next()
                if inner is None:
                    raise DataValidationError("formula: unterminated I(...)")
                if inner[0] == "op":
                    if inner[1] == "(":
                        depth += 1
                    elif inner[1] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                pieces.append(inner[1])
            return Factor("numeric", "I(" + " ".join(pieces) + ")", raw_expr=" ".join(pieces))
        return Factor("numeric", name)

    @staticmethod
    def _dedup(terms: list[Term]) -> list[Term]:
        seen: set[frozenset[Factor]] = set()
        unique: list[Term] = []
        for term in terms:
            if term.factors not in seen:
                seen.add(term.factors)
                unique.append(term)
        return unique


def _combinations(items: list[Term], size: int) -> list[list[Term]]:
    if size > len(items):
        return []
    if size == 1:
        return [[item] for item in items]
    out: list[list[Term]] = []
    for i in range(len(items) - size + 1):
        for rest in _combinations(items[i + 1 :], size - 1):
            out.append([items[i]] + rest)
    return out


def _combinations_factors(factors: list[Factor], size: int) -> list[frozenset[Factor]]:
    if size > len(factors):
        return []
    if size == 1:
        return [frozenset({f}) for f in factors]
    out: list[frozenset[Factor]] = []
    for i in range(len(factors) - size + 1):
        for rest in _combinations_factors(factors[i + 1 :], size - 1):
            out.append(frozenset({factors[i]}) | rest)
    return out


def parse_formula(formula: str) -> tuple[str | None, list[Term]]:
    """Parse a formula into ``(response, terms)``.

    The grammar (patsy precedence — ``:`` tightest, then ``*``/``/``,
    then ``+``/``-``)::

        formula := [name "~"] sum
        sum     := ["-"] prod ( ("+" | "-") prod )*
        prod    := power ( ("*" | "/") power )*
        power   := term [("**" | "^") depth]
        term    := factor ( ":" factor )*
        factor  := "0" | "1" | number | name
                |  "C" "(" name ["," contrast ["(" level ")"]] ")"
                |  "I" "(" python-expression ")"

    Examples
    --------
    >>> response, terms = parse_formula("y ~ a * b")
    >>> [repr(t) for t in terms]
    ['1', 'a', 'b', 'a:b']
    >>> _, terms2 = parse_formula("y ~ (a + b + c) ** 2")
    >>> [repr(t) for t in terms2]
    ['1', 'a', 'b', 'c', 'a:b', 'a:c', 'b:c']
    >>> _, terms3 = parse_formula("y ~ a / b")
    >>> [repr(t) for t in terms3]
    ['1', 'a', 'a:b']
    >>> _, terms4 = parse_formula("y ~ x + C(g) - 1")
    >>> [repr(t) for t in terms4]
    ['x', 'C(g)']
    """

    return _FormulaParser(_tokenize(formula)).parse()


# --------------------------------------------------------------------------
# the coding layer
# --------------------------------------------------------------------------


def _column(data: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in data:
        raise DataValidationError(
            f"formula: column {name!r} is not in the data "
            f"(available: {sorted(data.keys())})"
        )
    values = data[name]
    if isinstance(values, pd.Series):
        return values.to_numpy()
    return np.asarray(values)


def _is_categorical(values: np.ndarray) -> bool:
    if values.dtype.kind in "USb":
        return True
    return bool(np.issubdtype(values.dtype, np.str_) or np.issubdtype(values.dtype, np.object_))


def contrast_matrix(levels: Sequence[Any], contrast: str, reference: Any = None) -> pd.DataFrame:
    """The coding matrix for one categorical factor.

    Rows = levels, columns = design columns.  ``Treatment`` drops the
    reference level (default: the first), ``Sum`` codes the last level
    as ``-1`` (weights sum to zero across levels), ``Helmert`` follows
    R's ``contr.helmert`` (column *j* contrasts level *j+1* with the
    mean of the levels before it).

    Parameters
    ----------
    levels : sequence
        The observed levels, first-appearance order (patsy semantics).
    contrast : {"Treatment", "Sum", "Helmert"}
    reference : Any, optional
        The Treatment reference level.

    Returns
    -------
    pandas.DataFrame
        Index = levels, columns = coded names (``[T.x]`` / ``[S.x]`` /
        ``[H.j]`` suffixes).

    Examples
    --------
    >>> coding = contrast_matrix(["low", "mid", "high"], "Treatment")
    >>> [coding.loc["high"].tolist(), coding.columns.tolist()]
    [[0.0, 1.0], ['[T.mid]', '[T.high]']]
    >>> summed = contrast_matrix(["a", "b", "c"], "Sum")
    >>> summed.loc["c"].tolist()
    [-1.0, -1.0]
    """

    k = len(levels)
    if k < 2:
        raise DataValidationError(
            f"contrast_matrix: need >= 2 levels, got {list(levels)!r}"
        )
    if contrast == "Treatment":
        ref = reference if reference is not None else levels[0]
        if ref not in list(levels):
            raise DataValidationError(
                f"contrast_matrix: reference {ref!r} is not among the levels"
            )
        kept = [level for level in levels if level != ref]
        columns = {
            f"[T.{level}]": [1.0 if lvl == level else 0.0 for lvl in levels]
            for level in kept
        }
    elif contrast == "Sum":
        kept = list(levels[:-1])
        columns = {
            f"[S.{level}]": [
                1.0 if lvl == level else (-1.0 if lvl == levels[-1] else 0.0)
                for lvl in levels
            ]
            for level in kept
        }
    elif contrast == "Helmert":
        columns = {}
        for j in range(1, k):
            name = f"[H.{j}]"
            row = []
            for i, _level in enumerate(levels):
                one_based = i + 1
                if one_based == j + 1:
                    row.append(float(j))
                elif one_based <= j:
                    row.append(-1.0)
                else:
                    row.append(0.0)
            columns[name] = row
    else:
        raise DataValidationError(f"contrast_matrix: unknown contrast {contrast!r}")
    return pd.DataFrame(columns, index=[str(level) for level in levels])


_I_NAMESPACE: dict[str, Any] = {
    "log": np.log, "log10": np.log10, "log2": np.log2, "exp": np.exp,
    "sqrt": np.sqrt, "abs": np.abs, "sin": np.sin, "cos": np.cos,
    "tan": np.tan, "floor": np.floor, "ceil": np.ceil, "square": np.square,
    "minimum": np.minimum, "maximum": np.maximum, "where": np.where,
}


def _evaluate_factor(factor: Factor, data: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    """One factor → (column names, matrix block)."""

    if factor.kind == "intercept":
        raise DataValidationError("the intercept is handled by the builder")
    if factor.expr.startswith("I("):
        source = factor.raw_expr if factor.raw_expr is not None else factor.expr[2:-1]
        try:
            values = eval(source, {"__builtins__": {}}, {**_I_NAMESPACE, **dict(data)})
            block = np.asarray(values, dtype=float).reshape(-1, 1)
        except Exception as exc:
            raise DataValidationError(
                f"formula: I({source}) failed to evaluate: {exc}"
            ) from exc
        return [factor.expr], block
    values = _column(data, factor.expr)
    categorical = factor.kind == "categorical" or _is_categorical(values)
    if categorical:
        level_values = [str(v) for v in values.tolist()]
        levels: list[str] = []
        for level in level_values:
            if level not in levels:
                levels.append(level)
        contrast = factor.contrast if factor.kind == "categorical" else "Treatment"
        reference = factor.reference
        coding = contrast_matrix(levels, contrast, reference)
        names = [f"{factor.expr}{col}" for col in coding.columns]
        block = coding.loc[level_values].to_numpy(dtype=float)
        return names, block
    numeric = np.asarray(values, dtype=float).reshape(-1, 1)
    return [factor.expr], numeric


@dataclass
class DesignInfo:
    """The design matrix and everything about how it was built.

    Attributes
    ----------
    design : pandas.DataFrame
        The matrix, named columns, intercept first.
    term_slices : dict[str, slice]
        Which columns belong to which term (ANOVA's map).
    terms : list[Term]
        The parsed, ordered terms.
    response : str | None
        The left-hand side column, if given.
    formula : str
        The formula as written.
    """

    design: pd.DataFrame
    term_slices: dict[str, slice] = field(default_factory=dict)
    terms: list[Term] = field(default_factory=list)
    response: str | None = None
    formula: str = ""

    @property
    def columns(self) -> list[str]:
        return list(self.design.columns)

    @property
    def matrix(self) -> np.ndarray:
        return self.design.to_numpy(dtype=float)

    def summary(self) -> str:
        lines = [f"formula: {self.formula}", f"columns ({len(self.columns)}):"]
        for name, block in self.term_slices.items():
            columns = ", ".join(self.columns[block])
            lines.append(f"  {name:<28} -> {columns}")
        return "\n".join(lines)


def design_matrices(formula: str, data: Mapping[str, Any] | pd.DataFrame) -> DesignInfo:
    """Expand a formula against a data frame into the design matrix.

    Parameters
    ----------
    formula : str
        The patsy-style formula (``"y ~ x1 + C(x2)"``); the response is
        parsed but only recorded — call :func:`fit_ols_formula` to use it.
    data : mapping or DataFrame
        The columns the formula names.

    Returns
    -------
    DesignInfo
        The matrix (``.design``/``.matrix``), the term slices and the
        parsed term list.

    Examples
    --------
    >>> info = design_matrices("y ~ a + b + a:b", {
    ...     "y": [1, 2, 3, 4], "a": [1.0, 2.0, 3.0, 4.0],
    ...     "b": [2.0, 1.0, 2.0, 1.0],
    ... })
    >>> info.columns
    ['Intercept', 'a', 'b', 'a:b']
    >>> info.term_slices['a:b']
    slice(3, 4, None)
    """

    frame = dict(data.items()) if isinstance(data, pd.DataFrame) else dict(data)
    response, terms = parse_formula(formula)
    has_intercept = any(term == INTERCEPT for term in terms)
    ordered: list[Term] = []
    seen: set[frozenset[Factor]] = set()
    for term in terms:
        if term == INTERCEPT:
            continue
        if term.factors and all(f.kind == "intercept" for f in term.factors):
            continue
        if term.factors in seen:
            continue
        seen.add(term.factors)
        ordered.append(term)
    n_rows = _n_rows(frame)
    columns: list[str] = []
    blocks: list[np.ndarray] = []
    slices: dict[str, slice] = {}
    if has_intercept:
        columns.append("Intercept")
        blocks.append(np.ones((n_rows, 1)))
    for term in ordered:
        names: list[str] = []
        block = np.ones((n_rows, 1))
        for factor in sorted(term.factors, key=repr):
            factor_names, factor_block = _evaluate_factor(factor, frame)
            if not names:
                names = list(factor_names)
                block = factor_block
            else:
                names = [f"{a}:{b}" for a in names for b in factor_names]
                block = (
                    block[:, :, None] * factor_block[:, None, :]
                ).reshape(n_rows, -1)
        start = len(columns)
        columns.extend(names)
        blocks.append(block)
        slices[repr(term)] = slice(start, len(columns))
    if not columns:
        raise DataValidationError("formula: the design matrix is empty")
    design = pd.DataFrame(
        np.hstack([b.reshape(n_rows, -1) for b in blocks]), columns=columns
    )
    return DesignInfo(design, slices, ordered, response, formula)


def _n_rows(frame: Mapping[str, Any]) -> int:
    for value in frame.values():
        if isinstance(value, pd.Series):
            return len(value)
        if isinstance(value, np.ndarray):
            return int(value.shape[0])
        return len(value)
    raise DataValidationError("formula: the data frame has no columns")


# --------------------------------------------------------------------------
# the t distribution, self-contained
# --------------------------------------------------------------------------


def _log_gamma(x: float) -> float:
    return math.lgamma(x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) via the Lentz continued fraction."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        _log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x) / b


def _beta_cf(a: float, b: float, x: float) -> float:
    """The continued fraction core (Numerical Recipes' betacf)."""

    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 1e-12:
            return result
    return result


def _t_two_sided_p(t_stat: float, df: int) -> float:
    """Two-sided p-value of Student's t (incomplete-beta identity)."""

    if df <= 0:
        raise DataValidationError("t p-value: degrees of freedom must be positive")
    t_squared = float(t_stat) ** 2
    return float(_betainc(df / 2.0, 0.5, df / (df + t_squared)))


# --------------------------------------------------------------------------
# the fit + ANOVA layer
# --------------------------------------------------------------------------


@dataclass
class OLSResult:
    """Ordinary least squares through a formula.

    Attributes
    ----------
    params, se, t_values, p_values : list[float]
        Per design column (aligned with ``columns``).
    r_squared, adj_r_squared : float
    residuals, fitted : np.ndarray
    columns : list[str]
    n, k : int
    formula, design : str, DesignInfo
    """

    params: list[float]
    se: list[float]
    t_values: list[float]
    p_values: list[float]
    r_squared: float
    adj_r_squared: float
    residuals: np.ndarray
    fitted: np.ndarray
    columns: list[str]
    n: int
    k: int
    formula: str
    design: DesignInfo

    def summary(self) -> str:
        header = (
            f"OLS: {self.formula}   (n={self.n}, k={self.k}, "
            f"R2={self.r_squared:.4f}, adj R2={self.adj_r_squared:.4f})"
        )
        lines = [header, f"{'column':<24}{'coef':>10}{'se':>10}{'t':>9}{'p':>10}"]
        for i, column in enumerate(self.columns):
            lines.append(
                f"{column:<24}{self.params[i]:>10.4f}{self.se[i]:>10.4f}"
                f"{self.t_values[i]:>9.3f}{self.p_values[i]:>10.4f}"
            )
        return "\n".join(lines)


def fit_ols_formula(formula: str, data: Mapping[str, Any] | pd.DataFrame) -> OLSResult:
    """Least squares through the formula language.

    The design comes from :func:`design_matrices`; the response is the
    formula's left-hand side; the solution is ``lstsq`` with the
    classical variance ``σ²(XᵀX)⁻¹`` and two-sided t p-values from the
    incomplete-beta identity (no scipy anywhere).

    Examples
    --------
    >>> result = fit_ols_formula("y ~ x", {
    ...     "y": [1.0, 3.1, 4.9, 7.2], "x": [0.0, 1.0, 2.0, 3.0],
    ... })
    >>> [round(v, 2) for v in result.params]
    [0.99, 2.04]
    >>> bool(result.p_values[1] < 0.01)    # the slope is no accident
    True
    """

    frame = dict(data.items()) if isinstance(data, pd.DataFrame) else dict(data)
    info = design_matrices(formula, frame)
    if info.response is None:
        raise DataValidationError(
            "fit_ols_formula: the formula needs a response ('y ~ x')"
        )
    y = _column(frame, info.response).astype(float)
    matrix = info.matrix
    if y.shape[0] != matrix.shape[0]:
        raise DataValidationError(
            f"fit_ols_formula: response has {y.shape[0]} rows, design {matrix.shape[0]}"
        )
    beta, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    fitted = matrix @ beta
    residuals = y - fitted
    n, k = matrix.shape
    dof = n - k
    if dof <= 0:
        raise DataValidationError(
            f"fit_ols_formula: no residual degrees of freedom (n={n}, k={k})"
        )
    sigma2 = float(residuals @ residuals) / dof
    try:
        covariance = sigma2 * np.linalg.inv(matrix.T @ matrix)
    except np.linalg.LinAlgError as exc:
        raise DataValidationError(
            "fit_ols_formula: singular X'X — the design has collinear columns"
        ) from exc
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.where(se > 0, beta / np.where(se > 0, se, 1.0), 0.0)
    p_values = [_t_two_sided_p(float(t), dof) for t in t_values]
    ss_total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - float(residuals @ residuals) / ss_total if ss_total > 0 else 1.0
    adj = 1.0 - (1.0 - r_squared) * (n - 1) / dof if dof > 0 else r_squared
    return OLSResult(
        params=[float(v) for v in beta],
        se=[float(v) for v in se],
        t_values=[float(v) for v in t_values],
        p_values=p_values,
        r_squared=float(r_squared),
        adj_r_squared=float(adj),
        residuals=residuals,
        fitted=fitted,
        columns=info.columns,
        n=n,
        k=k,
        formula=formula,
        design=info,
    )


@dataclass
class ANOVARow:
    """One Type-II row: the term, its SS, F and p."""

    term: str
    sum_squares: float
    df: int
    f_statistic: float
    p_value: float


def anova_ii(formula: str, data: Mapping[str, Any] | pd.DataFrame) -> list[ANOVARow]:
    """Type-II ANOVA table from the term slices.

    Type-II rule: each term is tested by dropping it (and every term
    that *contains* it) from the model and comparing residual sums of
    squares — the containment rule that never tests a main effect
    inside an interaction it participates in.

    Examples
    --------
    >>> rows = anova_ii("y ~ g + x", {
    ...     "y": [1.0, 2.0, 5.0, 6.0], "g": ["a", "a", "b", "b"],
    ...     "x": [1.0, 2.0, 3.0, 4.0],
    ... })
    >>> [row.term for row in rows]
    ['g', 'x']
    >>> bool(rows[1].p_value < 0.05)     # the slope carries the signal
    True
    """

    frame = dict(data.items()) if isinstance(data, pd.DataFrame) else dict(data)
    info = design_matrices(formula, frame)
    if info.response is None:
        raise DataValidationError("anova_ii: the formula needs a response")
    y = _column(frame, info.response).astype(float)
    n = y.shape[0]
    matrix = info.matrix
    dof_full = n - matrix.shape[1]
    if dof_full <= 0:
        raise DataValidationError("anova_ii: no residual degrees of freedom")
    beta_full, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    rss_full = float(((y - matrix @ beta_full) ** 2).sum())
    rows: list[ANOVARow] = []
    for term in info.terms:
        contained = [other for other in info.terms if other.contains(term) and other != term]
        drop_terms = {term, *contained}
        keep = [
            i
            for i, other in enumerate(info.terms)
            if other not in drop_terms
        ]
        columns_kept = [0]  # intercept unless it was dropped
        for i in keep:
            block = info.term_slices[repr(info.terms[i])]
            columns_kept.extend(range(block.start, block.stop))
        reduced = matrix[:, columns_kept]
        beta_red, *_ = np.linalg.lstsq(reduced, y, rcond=None)
        rss_red = float(((y - reduced @ beta_red) ** 2).sum())
        ss = max(rss_red - rss_full, 0.0)
        df = 1  # numeric terms are single columns
        block = info.term_slices[repr(term)]
        df = block.stop - block.start
        ms = ss / df if df > 0 else 0.0
        mse = rss_full / dof_full
        f_stat = ms / mse if mse > 0 else 0.0
        # F ~ F(df, dof_full): p via the incomplete beta as well
        p = _f_p_value(f_stat, df, dof_full)
        rows.append(ANOVARow(repr(term), ss, df, f_stat, p))
    return rows


def _f_p_value(f_stat: float, df1: int, df2: int) -> float:
    """Upper-tail F p-value via the incomplete beta."""

    if df1 <= 0 or df2 <= 0 or f_stat <= 0:
        return 1.0
    x = df2 / (df2 + df1 * f_stat)
    return float(_betainc(df2 / 2.0, df1 / 2.0, x))
