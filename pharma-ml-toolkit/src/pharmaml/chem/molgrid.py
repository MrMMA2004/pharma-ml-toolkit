r"""The mols2grid concept, absorbed natively: the grid data model and its
selection semantics, with ASCII depictions standing in for the SVG cards.

The instruction: *"Check the contents and files in this GitHub link
completely and one by one and add the missing items to playground.py:
github.com/cbouy/mols2grid."*  The repository was cloned and reviewed
file by file -- ``molgrid.py`` (the ``MolGrid`` class: the DataFrame
card model, ``subset``/``tooltip`` column DSL, ``mols2grid-id``,
``sort_by``, the SMARTS filter, the cached-checkbox selection restore),
``select.py`` (the ``SelectionRegister``: named grids, the
most-recently-updated pointer, ``get_selection``/``list_grids``),
``callbacks.py`` (the ``info()`` popup computing molecular weight, HBA,
HBD, TPSA per molecule), ``utils.py`` (``tooltip_formatter`` with its
per-column ``style``/``transform`` hooks, ``sdf_to_dataframe``),
``dispatch.py``, ``widget.py`` and the ``src/`` TypeScript + templates
(the Bootstrap grid, DataTables search, popovers, the on-the-fly
RDKit-JS SVG renderer).

What mols2grid *is* is a browser rendering of a molecule table -- and
the rendering layer (HTML/CSS/JS, ipywidgets, RDKit-JS) is exactly what
this toolkit does not vendor.  What transfers natively is everything
*around* the rendering, and all of it landed here:

* **The card model** (:class:`MoleculeGrid`) -- a pandas DataFrame plus
  a SMILES column becomes a grid of cards: the ``subset``/``tooltip``
  column DSL (visible fields below the depiction, hover fields
  formatted as ``key: value`` pairs exactly like ``tooltip_formatter``,
  with ``transform`` hooks applied first), the injected
  ``mols2grid-id``, ``sort_by`` with stable numeric-aware ordering,
  and the case-insensitive text search across subset + tooltip fields
  (the DataTables search box's semantics, without DataTables).
* **The SMARTS filter** (:meth:`MoleculeGrid.filter_smarts`) --
  substructure filtering over the parsed molecules, returning both the
  surviving ids and the matched atom indices per molecule (what the
  JS-side highlight would circle).  Honest scope: the query language is
  :func:`pharmaml.chem.molgraph.substructure_matches`' query-SMILES
  subset -- element, aromaticity, bracket H counts, charge -- not full
  SMARTS (no wildcards, no topology operators).
* **The selection register** (:class:`GridSelections`, the module-level
  :data:`register`) -- named grids, the most-recently-updated pointer,
  ``get_selection``/``list_grids``, and the cached-checkbox restore
  (``set_selection`` repopulates from the register, mols2grid's
  ``cache_selection`` parameter as a method).
* **The ``info()`` popup as a card** (:func:`descriptor_card`) --
  molecular weight, hydrogen-bond acceptors and donors (Lipinski's own
  counts), TPSA, the formula, ring and rotatable-bond counts, computed
  by :func:`pharmaml.chem.molgraph.molecule_profile` -- the same
  descriptor set mols2grid's popup fetches from RDKit-JS.
* **The depiction** (:func:`ascii_depiction`) -- where mols2grid embeds
  an SVG from RDKit's 2D depictor, this module rasterises
  :func:`pharmaml.chem.molgraph.generate_2d_coordinates` into a
  character grid: bonds as ``- | / \`` chosen by slope, hetero atoms
  labelled with their element symbol, carbons implicit (RDKit's
  depiction convention).  Deterministic, no rendering library anywhere.

Attribution: cbouy/mols2grid, MIT license; the data model follows its
public API, and every piece of browser machinery it drives is stated
here as deliberately absent rather than quietly re-implemented.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.chem.molgraph import (
    Molecule,
    generate_2d_coordinates,
    molecule_profile,
    parse_smiles,
    substructure_matches,
)
from pharmaml.exceptions import DataValidationError

__all__ = [
    "GridSelections",
    "MoleculeGrid",
    "ascii_depiction",
    "descriptor_card",
    "register",
    "tooltip_line",
]


#: Bond characters by screen direction bucket (math-convention angle of the
#: on-screen delta; y grows downward in the character grid).
_BOND_CHARS = {
    "h": "-",
    "v": "|",
    "diag_up": "/",
    "diag_down": "\\",
}


def _bond_char(dx: float, dy: float) -> str:
    """Pick the bond glyph for one on-screen step (y grows downward)."""
    if abs(dx) >= 3.0 * abs(dy):
        return _BOND_CHARS["h"]
    if abs(dy) >= 3.0 * abs(dx):
        return _BOND_CHARS["v"]
    return _BOND_CHARS["diag_up"] if dx * dy < 0 else _BOND_CHARS["diag_down"]


def ascii_depiction(
    mol: Molecule,
    *,
    width: int = 46,
    height: int = 18,
    show_carbons: bool = False,
) -> str:
    r"""Rasterise the 2D depiction of ``mol`` into a character grid.

    The concept of mols2grid's embedded SVG -- RDKit's 2D coordinates
    rendered per card -- in plain text: the layout from
    :func:`pharmaml.chem.molgraph.generate_2d_coordinates` is scaled
    into ``width`` x ``height`` cells, each bond is walked with
    sub-cell sampling and drawn as ``-``, ``|``, ``/`` or ``\\`` by its
    screen direction, and every hetero atom is labelled with its
    element symbol (carbons stay implicit, the RDKit depiction
    convention; ``show_carbons=True`` labels them too).  Lines carry no
    trailing whitespace and the frame is deterministic, so the output
    is a doctest-stable contract.  Not an RDKit-quality depiction --
    bond orders are not doubled and stereo wedges are absent -- and
    said so.

    >>> from pharmaml.chem.molgraph import parse_smiles
    >>> art = ascii_depiction(parse_smiles("CCO"), width=26, height=9)
    >>> print(art)
    <BLANKLINE>
      O
       \\
         \\
           \
            \\
              \\
                -------------
    <BLANKLINE>
    """
    if width < 8 or height < 5:
        raise DataValidationError(
            f"depiction canvas must be at least 8x5 cells, got {width}x{height}"
        )
    if not mol.atoms:
        return ""
    coords = np.asarray(generate_2d_coordinates(mol), dtype=float)
    grid: list[list[str]] = [[" "] * width for _ in range(height)]
    if len(mol.atoms) == 1:
        symbol = mol.atoms[0].symbol
        row, col = height // 2, (width - len(symbol)) // 2
        grid[row][col : col + len(symbol)] = list(symbol)
        return "\n".join("".join(line).rstrip() for line in grid)

    lo = coords.min(axis=0)
    span = coords.max(axis=0) - lo
    span = np.where(span < 1e-9, 1.0, span)
    margin = 1.0
    scale = np.array(
        [
            (width - 2 * margin - 2.0) / span[0],
            (height - 2 * margin - 1.0) / span[1],
        ]
    )
    # (100x-169) a fresh name for the scalar: `scale` was reused with
    # two types (ndarray, then float) and mypy 2.4 refuses the min()
    # overload across them.
    sx, sy = float(scale[0]), float(scale[1])
    grid_scale = min(sx, sy)
    centre = (np.array([width, height], dtype=float) - 1.0) / 2.0
    points = (coords - (lo + span / 2.0)) * grid_scale + centre

    def cell(x: float, y: float) -> tuple[int, int]:
        return round(float(y)), round(float(x))

    for bond in mol.bonds:
        a = points[bond.begin]
        b = points[bond.end]
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        glyph = _bond_char(dx, dy)
        # dominant-axis walk: one glyph per row (steep) or per column
        # (shallow), so bonds stay one character thick
        if abs(dy) >= abs(dx):
            row_a, row_b = round(float(a[1])), round(float(b[1]))
            span = row_b - row_a
            for row in range(min(row_a, row_b), max(row_a, row_b) + 1):
                t = 0.0 if span == 0 else (row - row_a) / span
                col = round(float(a[0]) + t * dx)
                if 0 <= row < height and 0 <= col < width and grid[row][col] == " ":
                    grid[row][col] = glyph
        else:
            col_a, col_b = round(float(a[0])), round(float(b[0]))
            span = col_b - col_a
            for col in range(min(col_a, col_b), max(col_a, col_b) + 1):
                t = 0.0 if span == 0 else (col - col_a) / span
                row = round(float(a[1]) + t * dy)
                if 0 <= row < height and 0 <= col < width and grid[row][col] == " ":
                    grid[row][col] = glyph

    for atom, point in zip(mol.atoms, points):
        symbol = atom.symbol if (show_carbons or atom.symbol != "C") else ""
        if not symbol:
            continue
        row, col = cell(*point)
        col -= len(symbol) // 2
        for offset, char in enumerate(symbol):
            if 0 <= row < height and 0 <= col + offset < width:
                grid[row][col + offset] = char
    return "\n".join("".join(line).rstrip() for line in grid)


def tooltip_line(
    record: Mapping[str, Any],
    columns: Sequence[str],
    *,
    transform: Mapping[str, Callable[[Any], Any]] | None = None,
    fmt: str = "{key}: {value}",
) -> str:
    """Format one record's ``columns`` as tooltip key/value pairs.

    mols2grid's ``tooltip_formatter`` verbatim in concept: the
    ``transform`` hooks are applied to each value before formatting
    (mols2grid's per-column ``transform`` dict), and the pairs are
    joined -- with ``<br>`` in the original, ``; `` here, because the
    consumer is a text card, not HTML.

    >>> tooltip_line({"name": "ethanol", "mw": 46.07}, ["name", "mw"])
    'name: ethanol; mw: 46.07'
    >>> tooltip_line(
    ...     {"mw": 46.07}, ["mw"], transform={"mw": lambda v: round(v, 1)}
    ... )
    'mw: 46.1'
    """
    transform = transform or {}
    pairs = []
    for key in columns:
        value = record.get(key)
        hook = transform.get(key)
        if hook is not None:
            value = hook(value)
        pairs.append(fmt.format(key=key, value=value))
    return "; ".join(pairs)


def descriptor_card(smiles: str) -> str:
    """The ``info()`` popup as a text card: the descriptor set mols2grid
    computes through RDKit-JS (molecular weight, H-bond acceptors and
    donors, TPSA), plus the formula, ring and rotatable-bond counts --
    every number from :func:`pharmaml.chem.molgraph.molecule_profile`.

    >>> print(descriptor_card("CCO"))
    SMILES   CCO
    formula  C2H6O
    MW       46.07
    HBA      1
    HBD      1
    TPSA     20.23
    rings    0
    rot.     0
    <BLANKLINE>
    """
    profile = molecule_profile(smiles)
    lines = [
        f"SMILES   {profile['smiles']}",
        f"formula  {profile['formula']}",
        f"MW       {profile['molecular_weight']:.2f}",
        f"HBA      {profile['hba']}",
        f"HBD      {profile['hbd']}",
        f"TPSA     {profile['tpsa']:.2f}",
        f"rings    {profile['rings']}",
        f"rot.     {profile['rotatable_bonds']}",
    ]
    return "\n".join(lines) + "\n"


class GridSelections:
    """Register for grid selections -- mols2grid's ``SelectionRegister``.

    Each named grid stores its selection as ``{id: smiles}`` (the
    payload mols2grid ships to callbacks); ``current_selection`` tracks
    the most recently updated grid so a bare ``get_selection()`` after
    a GUI interaction returns what the user just touched.

    >>> register = GridSelections()
    >>> register.update("library", {0: "CCO", 2: "c1ccccc1"})
    >>> register.get_selection("library")
    {0: 'CCO', 2: 'c1ccccc1'}
    >>> register.current_selection
    'library'
    >>> register.list_grids()
    ['library']
    """

    def __init__(self) -> None:
        self._selections: dict[str, dict[int, str]] = {}
        self.current_selection: str | None = None

    def update(self, name: str, payload: Mapping[int, str]) -> None:
        """Store one grid's selection (the checkbox state as id->SMILES)."""
        self._selections[name] = dict(payload)
        self.current_selection = name

    def get_selection(self, name: str | None = None) -> dict[int, str]:
        """The selection of ``name`` (the most recent grid when ``None``)."""
        if name is None:
            name = self.current_selection
        if name is None or name not in self._selections:
            return {}
        return dict(self._selections[name])

    def list_grids(self) -> list[str]:
        """The registered grid names, in first-touch order."""
        return list(self._selections)

    def clear(self) -> None:
        """Drop every selection (mols2grid's ``_clear``)."""
        self._selections.clear()
        self.current_selection = None


#: Module-level register, mirroring mols2grid's module-level ``register``.
register = GridSelections()


@dataclass
class _Entry:
    """One grid row: the parsed molecule (or its parse failure)."""

    id: int
    smiles: str
    molecule: Molecule | None
    error: str | None


class MoleculeGrid:
    """A DataFrame of molecules as a grid of cards -- mols2grid's
    ``MolGrid``, without the browser.

    Parameters
    ----------
    frame : pandas.DataFrame, dict of lists, or list of dicts
        The molecule table (mols2grid accepts the same three shapes).
    smiles_col : str
        The SMILES column (mols2grid's ``smiles_col``).
    name : str
        Grid name -- the register key under which selections live.
    subset : sequence of str or None
        Columns displayed on each card below the depiction; ``None``
        keeps every column except the SMILES one.
    tooltip : sequence of str, or False
        Columns reachable through :meth:`tooltip` (the hover fields);
        ``None`` moves every non-subset, non-SMILES column there, as
        mols2grid does when only ``subset`` is given.
    rename : mapping or None
        Column renaming before anything else (mols2grid's ``rename``).

    >>> grid = MoleculeGrid(
    ...     {"SMILES": ["CCO", "c1ccccc1", "CCCCO"],
    ...      "name": ["ethanol", "benzene", "butanol"]},
    ...     subset=["name"],
    ... )
    >>> grid.n_molecules, grid.n_valid
    (3, 3)
    >>> grid.filter("enz")              # DataTables search, in text
    [1]
    >>> grid.filter_smarts("c1ccccc1")  # the SMARTS filter box
    [1]
    >>> grid.set_selection([0, 1])
    >>> grid.get_selection()
    {0: 'CCO', 1: 'c1ccccc1'}
    >>> print(grid.tooltip(0))
    SMILES: CCO
    """

    def __init__(
        self,
        frame: pd.DataFrame | Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
        smiles_col: str = "SMILES",
        *,
        name: str = "default",
        subset: Sequence[str] | None = None,
        tooltip: Sequence[str] | bool | None = None,
        rename: Mapping[str, str] | None = None,
    ) -> None:
        if isinstance(frame, pd.DataFrame):
            data = frame.copy()
        else:
            data = pd.DataFrame(frame)
        if rename:
            data = data.rename(columns=dict(rename))
        if smiles_col not in data.columns:
            raise DataValidationError(
                f"SMILES column {smiles_col!r} not found; columns are "
                f"{list(data.columns)}"
            )
        if not isinstance(name, str):
            raise DataValidationError(
                f"grid name must be a string, got {type(name).__name__}"
            )
        data = data.reset_index(drop=True)
        self.frame = data
        self.smiles_col = smiles_col
        self.name = name
        columns = [c for c in data.columns if c != smiles_col]
        if subset is None:
            subset = list(columns)
        missing = [c for c in subset if c not in data.columns]
        if missing:
            raise DataValidationError(
                f"subset columns not in the frame: {missing}"
            )
        self.subset: list[str] = list(subset)
        if tooltip is None or tooltip is True:
            self.tooltip_columns = [
                c for c in columns if c not in self.subset
            ] + [smiles_col]
        elif tooltip is False:
            self.tooltip_columns = []
        else:
            missing = [c for c in tooltip if c not in data.columns]  # type: ignore[union-attr]
            if missing:
                raise DataValidationError(
                    f"tooltip columns not in the frame: {missing}"
                )
            self.tooltip_columns = list(tooltip)  # type: ignore[arg-type]
        self._entries: list[_Entry] = []
        for row_id, smiles in enumerate(data[smiles_col]):
            text = "" if pd.isna(smiles) else str(smiles)
            try:
                self._entries.append(_Entry(row_id, text, parse_smiles(text), None))
            except DataValidationError as exc:
                self._entries.append(_Entry(row_id, text, None, str(exc)))
        register.update(name, register.get_selection(name))

    # -- counts -------------------------------------------------------- #

    @property
    def n_molecules(self) -> int:
        """Rows in the frame (valid or not)."""
        return len(self._entries)

    @property
    def n_valid(self) -> int:
        """Rows whose SMILES parsed."""
        return sum(1 for entry in self._entries if entry.molecule is not None)

    @property
    def invalid_smiles(self) -> list[tuple[int, str, str]]:
        """``(id, smiles, error)`` for every unparsable row."""
        return [
            (entry.id, entry.smiles, entry.error or "")
            for entry in self._entries
            if entry.molecule is None
        ]

    def ids(self) -> list[int]:
        """Every row id, in frame order."""
        return [entry.id for entry in self._entries]

    def molecule(self, row_id: int) -> Molecule:
        """The parsed molecule of one row (raises for unparsable rows)."""
        entry = self._entries[row_id]
        if entry.molecule is None:
            raise DataValidationError(
                f"row {row_id} does not parse: {entry.error}"
            )
        return entry.molecule

    # -- filters ------------------------------------------------------- #

    def filter(self, text: str) -> list[int]:
        """Case-insensitive search over subset + tooltip + SMILES fields.

        The DataTables search box semantics: a row survives when any of
        its searchable fields contains the query, case-insensitively.

        >>> grid = MoleculeGrid(
        ...     {"SMILES": ["CCO", "CCN"], "name": ["ethanol", "ethylamine"]},
        ...     subset=["name"],
        ... )
        >>> grid.filter("eth")
        [0, 1]
        >>> grid.filter("amine")
        [1]
        """
        needle = text.strip().lower()
        if not needle:
            return self.ids()
        hits: list[int] = []
        search_columns = list(
            dict.fromkeys(
                self.subset + self.tooltip_columns + [self.smiles_col]
            )
        )
        for entry in self._entries:
            record = self.frame.iloc[entry.id]
            for column in search_columns:
                if column not in self.frame.columns:
                    continue
                value = record[column]
                if pd.isna(value):
                    continue
                if needle in str(value).lower():
                    hits.append(entry.id)
                    break
        return hits

    def filter_smarts(self, query: str) -> list[int]:
        """Rows whose molecule matches the substructure ``query``.

        The SMARTS filter box (mols2grid runs it through RDKit-JS on
        the browser side); the query language here is the query-SMILES
        subset of :func:`pharmaml.chem.molgraph.substructure_matches`:
        element, aromaticity, bracket H counts and charge -- no
        wildcards or topology operators.
        """
        return [
            entry.id
            for entry in self._entries
            if entry.molecule is not None
            and substructure_matches(entry.molecule, query)
        ]

    def substructure_atoms(self, row_id: int, query: str) -> list[int]:
        """The matched atom indices of one row (the highlight circle set)."""
        mol = self.molecule(row_id)
        matches = substructure_matches(mol, query)
        atoms: set[int] = set()
        for match in matches:
            atoms.update(match)
        return sorted(atoms)

    def sort(
        self,
        ids: Iterable[int] | None = None,
        *,
        by: str | None = None,
        descending: bool = False,
    ) -> list[int]:
        """Stable sort of ``ids`` (all rows when ``None``) by a column.

        Numeric-looking values sort numerically before lexicographic
        ones (mols2grid delegates to DataTables' type detection).
        """
        rows = list(ids) if ids is not None else self.ids()
        if by is None:
            return rows

        def key_of(row_id: int) -> tuple[int, float | str]:
            value = self.frame.iloc[row_id][by]
            if pd.isna(value):
                return (2, str(value))
            text = str(value)
            try:
                return (0, float(text))
            except ValueError:
                return (1, text.lower())

        return sorted(rows, key=key_of, reverse=descending)

    # -- cards --------------------------------------------------------- #

    def card(
        self,
        row_id: int,
        *,
        width: int = 46,
        height: int = 18,
        with_depiction: bool = True,
        highlight: Iterable[int] = (),
    ) -> str:
        """One text card: id header, depiction, subset fields.

        ``highlight`` is the set of atom indices to circle -- the
        matched atoms of a SMARTS query in mols2grid; here the marked
        atoms are listed on one line (the honest text rendering of the
        highlight).
        """
        entry = self._entries[row_id]
        # the cached checkbox: mols2grid draws one per card when
        # selection is on, and restores it from the register
        selection = register.get_selection(self.name)
        mark = " [x]" if entry.id in selection else " [ ]"
        lines = [f"#{entry.id}  {entry.smiles}{mark}"]
        if entry.molecule is None:
            lines.append(f"  unparsable: {entry.error}")
            return "\n".join(lines)
        if with_depiction:
            art = ascii_depiction(
                entry.molecule, width=width, height=height
            )
            lines.extend("  " + art_line for art_line in art.splitlines() or [""])
        record = self.frame.iloc[row_id]
        for column in self.subset:
            if column == self.smiles_col:
                continue
            lines.append(f"  {column}: {record[column]}")
        marks = sorted(int(a) for a in highlight)
        if marks:
            lines.append(f"  highlight: atoms {marks}")
        return "\n".join(lines)

    def tooltip(self, row_id: int) -> str:
        """The hover fields of one row, as ``key: value`` pairs."""
        entry = self._entries[row_id]
        record = self.frame.iloc[row_id]
        columns = [
            c
            for c in self.tooltip_columns
            if c != self.smiles_col or c in self.subset
        ]
        return tooltip_line(
            {**{c: record[c] for c in columns}, self.smiles_col: entry.smiles},
            columns + [self.smiles_col] if self.smiles_col in self.tooltip_columns else columns,
        )

    def grid_text(
        self,
        ids: Iterable[int] | None = None,
        *,
        columns: int = 2,
        width: int = 46,
        height: int = 18,
        highlight_query: str | None = None,
    ) -> str:
        """Render the grid as text -- the HTML document as a plain layout.

        Cards are laid out ``columns``-wide, side by side, separated by
        a vertical rule; a SMARTS ``highlight_query`` marks the matched
        atoms of every card (the filter's highlight, carried through).
        """
        rows = list(ids) if ids is not None else self.ids()
        cards = []
        for row_id in rows:
            highlight: Iterable[int] = ()
            if highlight_query:
                try:
                    highlight = self.substructure_atoms(row_id, highlight_query)
                except DataValidationError:
                    highlight = ()
            cards.append(
                self.card(row_id, width=width, height=height, highlight=highlight)
            )
        if not cards:
            return "(empty grid)"
        blocks = [card.splitlines() for card in cards]
        out_lines: list[str] = []
        for start in range(0, len(blocks), columns):
            chunk = blocks[start : start + columns]
            tallest = max(len(block) for block in chunk)
            for block in chunk:
                block.extend([""] * (tallest - len(block)))
            card_width = max(
                max(len(line) for line in block) for block in chunk
            )
            for line_index in range(tallest):
                pieces = []
                for block in chunk:
                    pieces.append(block[line_index].ljust(card_width))
                out_lines.append(" | ".join(pieces).rstrip())
            out_lines.append("")
        return "\n".join(out_lines).rstrip() + "\n"

    # -- selection ----------------------------------------------------- #

    def set_selection(self, ids: Iterable[int]) -> None:
        """Store the checkbox state: ``ids`` become the id->SMILES map."""
        payload: dict[int, str] = {}
        for row_id in ids:
            if 0 <= row_id < self.n_molecules:
                payload[row_id] = self._entries[row_id].smiles
        register.update(self.name, payload)

    def get_selection(self) -> dict[int, str]:
        """The registered selection of this grid (id -> SMILES)."""
        return register.get_selection(self.name)

    def selection_size(self) -> int:
        """How many cards are checked."""
        return len(self.get_selection())

    # -- reporting ----------------------------------------------------- #

    def summary(self) -> str:
        """One-line state: rows, parsable, selection, name."""
        return (
            f"grid {self.name!r}: {self.n_molecules} molecules, "
            f"{self.n_valid} parsable, {self.selection_size()} selected"
        )


def grid_demo() -> str:
    """A self-contained demonstration: filter, sort, select, tooltip.

    >>> print(grid_demo().splitlines()[0])
    grid 'demo': 4 molecules, 4 parsable, 2 selected
    """
    frame = pd.DataFrame(
        {
            "SMILES": ["CCO", "c1ccccc1", "CCN", "CC(=O)Oc1ccccc1C(=O)O"],
            "name": ["ethanol", "benzene", "ethylamine", "aspirin"],
        }
    )
    grid = MoleculeGrid(frame, subset=["name"], name="demo")
    grid.set_selection([1, 3])
    lines = [grid.summary()]
    lines.append(f"text filter 'en': {grid.filter('en')}")
    lines.append(f"SMARTS filter 'c1ccccc1': {grid.filter_smarts('c1ccccc1')}")
    lines.append(f"sorted by name: {grid.sort(by='name')}")
    lines.append(f"tooltip of row 3: {grid.tooltip(3)}")
    lines.append(f"aspirin phenyl atoms: {grid.substructure_atoms(3, 'c1ccccc1')}")
    return "\n".join(lines)
