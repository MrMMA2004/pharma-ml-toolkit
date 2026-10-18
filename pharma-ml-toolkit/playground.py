"""PharmaML Playground -- the whole toolkit behind one window.

Run this file from the toolkit root (what you get when you unzip the
downloaded archive)::

    python playground.py

A tabbed graphical interface opens with every calculation domain of the
:mod:`pharmaml` package, no command-line knowledge required:

* **Chemistry**     -- IC50/Ki -> pIC50/pKi conversion, molar mass, the
  full RDKit descriptor table for any SMILES, Tanimoto similarity, and
  the feature factory: the rar's own 7-family, 10,000+-column
  descriptor + fingerprint matrix, ranked and cut to a top-K selection.
* **PK / PD**       -- half-life <-> k(e), accumulation, IV bolus decay
  and Emax curves (plotted), oral Cmax/tmax, Cockcroft-Gault creatinine
  clearance with renal dose adjustment, loading/maintenance dosing,
  BSA/weight-based and pediatric dose rules.
* **Cell Lines**    -- GDSC-style drug-response screens: load and audit
  a screen, pivot the sensitivity matrix, call sensitive vs resistant
  lines, convert ``LN_IC50`` <-> pIC50, and discover expression
  biomarkers for any compound (the EGFR / erlotinib story).
* **Docking**       -- molecular docking end to end: prepare a ligand
  library and a receptor into PDBQT files, compute the grid box and
  write the GPF/DPF configs, parse DLG / Vina results or scan a whole
  campaign, and launch batch AutoDock4 or single Vina runs.
* **Data & ML**     -- open any CSV (or one of the bundled example
  datasets), pick a target column, train a registry model with the
  toolkit's own preprocessing pipeline, inspect the metrics and the
  parity/confusion figure, cross-validate a model leaderboard.
* **Generative**    -- run the genetic algorithm on a SMILES corpus with
  any bundled objective (QED, LogP window, ...); adaptive runs render
  the operator-mix learning curve (``plot_operator_trajectory``), and
  multi-objective runs fill the Pareto front card: the audit line, the
  non-dominated table (with the deepest hidden trade-off rank), the
  objective-pair scatter and a CSV export of the front.

Design notes
------------
* The file needs nothing but the standard library to *start*: every
  heavy dependency is probed at launch and each feature explains the
  exact ``pip install`` line it wants when missing.
* ``pharmaml`` itself resolves two ways: a normal installed package, or
  the ``src/`` tree sitting next to this file -- so the playground
  works straight after unzipping, before any ``pip install -e .``.
* The playground version follows the toolkit's own version (read from
  ``pyproject.toml`` / ``VERSION`` at import), so the two can never
  silently drift apart.
* Long calculations (descriptor batches, model training, evolution,
  biomarker discovery, docking preparation runs, feature-factory
  selection) run in a worker thread; the window never freezes and shows
  a progress bar plus a live status line.
* Every input survives a restart: on close the window writes a session
  file (``~/.pharmaml/playground-session.json``) with all entries,
  toggles, pasted corpora, the active tab, the window geometry and the
  theme, and restores it on the next launch (``File ▸ Save session
  now`` / ``Clear saved session`` control it manually).
* Sessions are shareable: ``File ▸ Export session`` writes the exact
  same payload to any path (mail it to a colleague) and ``File ▸
  Import session`` applies such a file onto the live window --
  including the theme.  ``File ▸ Reset current tab to defaults``
  (Ctrl+Shift+R) restores just the active tab's inputs to their
  freshly-launched values, leaving every other tab untouched -- and
  every content tab ALSO carries its own quiet ``Reset this tab to
  defaults`` button pinned in a non-scrolling footer strip below the
  cards (Home, which owns no inputs, carries none), so the reset is
  always one click away without scrolling back up.  Next to it sits
  the footer's second affordance: ``Copy inputs as JSON`` puts the
  SAME tab's live inputs on the clipboard as a small JSON snippet --
  a share-lite for issue reports and quick hand-offs.
* ``View ▸ Follow system / Light / Dark`` recolours the whole window
  (two named palettes drive every shade); the choice is part of the
  saved session.  Embedded figures keep the toolkit's white
  published-figure background on purpose -- plot legibility first.
* Embedded figures reuse the toolkit's own plotting helpers, so the
  playground looks exactly like the package's published figures.

Command line helpers (useful for scripting and CI)::

    python playground.py --probe        # dependency report, then exit
    python playground.py --self-test    # build the UI headlessly, then exit
    python playground.py --version      # print versions, then exit
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import queue
import re
import shutil
import sys
import threading
import traceback
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter import font as tkfont
except ImportError:  # pragma: no cover -- a Python built without tkinter
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    tkfont = None  # type: ignore[assignment]

TOOLKIT_ROOT = Path(__file__).resolve().parent
SRC_DIR = TOOLKIT_ROOT / "src"
EXAMPLES_DIR = TOOLKIT_ROOT / "examples" / "data"


def toolkit_version() -> str:
    """The toolkit's own version: pyproject ``[project]`` -> ``VERSION``.

    A deliberate line scan (no ``tomllib`` import) so the value is
    available on every Python that can start this file: the
    ``version = "x.y.z"`` entry of the ``[project]`` section wins, a
    bare ``VERSION`` file at the toolkit root is the fallback, and
    ``"0.0.0"`` covers a tree with neither (never happens in the
    shipped archive).
    """
    pyproject = TOOLKIT_ROOT / "pyproject.toml"
    if pyproject.is_file():
        in_project = False
        try:
            lines = pyproject.read_text(encoding="utf-8").splitlines()
        except OSError:  # pragma: no cover - unreadable pyproject
            lines = []
        for raw_line in lines:
            line = raw_line.strip()
            if line.startswith("[") and line.endswith("]"):
                in_project = line == "[project]"
                continue
            if not in_project or not line.startswith("version"):
                continue
            key, sep, value = line.partition("=")
            if not sep or key.strip() != "version":
                continue
            declared = value.split("#", 1)[0].strip().strip('"').strip("'")
            if declared:
                return declared
    version_file = TOOLKIT_ROOT / "VERSION"
    if version_file.is_file():
        try:
            text = version_file.read_text(encoding="utf-8").strip()
        except OSError:  # pragma: no cover - unreadable VERSION
            text = ""
        if text:
            return text
    return "0.0.0"


#: The playground is versioned WITH the toolkit (see ``toolkit_version``).
__version__ = toolkit_version()

# -- visual language (a calm pharma green, no framework required) ------------
#
# Two named palettes drive every colour in the window.  The module-level
# names below (``BG``, ``CARD``, ...) are *rebound in place* by
# :func:`apply_theme`, so all widget code keeps reading plain module
# globals while the View menu flips the whole window between light and
# dark.  Keys beyond the original ten exist only for the inline shades
# ``_build_style`` used to hardcode (tabs, headings, status bar, button
# faces) so the dark theme can recolour them too.
_LIGHT_PALETTE: dict[str, str] = {
    "BG": "#eef1ee",
    "CARD": "#ffffff",
    "BORDER": "#d9ded7",
    "INK": "#1c2321",
    "MUTED": "#5f6b64",
    "ACCENT": "#0e7c66",
    "ACCENT_DARK": "#0a5c4b",
    "ACCENT_SOFT": "#dceee8",
    "GOOD": "#157f3d",
    "BAD": "#b3261e",
    "TAB": "#dde3de",
    "HEADING": "#edf1ee",
    "STATUS": "#e3e7e2",
    "BUTTON_FACE": "#ffffff",
    "BUTTON_DISABLED": "#9fbcb4",
    "BUTTON_DISABLED_INK": "#e8f0ee",
}
_DARK_PALETTE: dict[str, str] = {
    "BG": "#151a18",
    "CARD": "#202724",
    "BORDER": "#39423e",
    "INK": "#e7ece9",
    "MUTED": "#9daba4",
    "ACCENT": "#16a98a",
    "ACCENT_DARK": "#0f8a70",
    "ACCENT_SOFT": "#2c4139",
    "GOOD": "#37c46f",
    "BAD": "#e5766b",
    "TAB": "#242c29",
    "HEADING": "#2b3531",
    "STATUS": "#111514",
    "BUTTON_FACE": "#2b3531",
    "BUTTON_DISABLED": "#3a4642",
    "BUTTON_DISABLED_INK": "#8d9b95",
}
_PALETTES: dict[str, dict[str, str]] = {"light": _LIGHT_PALETTE, "dark": _DARK_PALETTE}

#: The palette currently bound to the module-level colour names.
_ACTIVE_THEME = "light"

BG = _LIGHT_PALETTE["BG"]
CARD = _LIGHT_PALETTE["CARD"]
BORDER = _LIGHT_PALETTE["BORDER"]
INK = _LIGHT_PALETTE["INK"]
MUTED = _LIGHT_PALETTE["MUTED"]
ACCENT = _LIGHT_PALETTE["ACCENT"]
ACCENT_DARK = _LIGHT_PALETTE["ACCENT_DARK"]
ACCENT_SOFT = _LIGHT_PALETTE["ACCENT_SOFT"]
GOOD = _LIGHT_PALETTE["GOOD"]
BAD = _LIGHT_PALETTE["BAD"]
TAB = _LIGHT_PALETTE["TAB"]
HEADING = _LIGHT_PALETTE["HEADING"]
STATUS = _LIGHT_PALETTE["STATUS"]
BUTTON_FACE = _LIGHT_PALETTE["BUTTON_FACE"]
BUTTON_DISABLED = _LIGHT_PALETTE["BUTTON_DISABLED"]
BUTTON_DISABLED_INK = _LIGHT_PALETTE["BUTTON_DISABLED_INK"]


def apply_theme(name: str) -> str:
    """Rebind the module-level colour names to the named palette.

    ``"light"`` and ``"dark"`` are the two shipped palettes; anything
    else falls back to light.  Returns the palette actually applied (the
    caller resolves ``"auto"`` itself -- detecting the OS preference
    needs a live Tk interpreter, which this pure helper must not).

    Widget code reads ``BG`` / ``CARD`` / ... at call time, so a live
    window only needs :meth:`PlaygroundApp._build_style` re-run plus a
    walk over the plain ``tk`` widgets (see ``_retint_plain_widgets``)
    after calling this.
    """
    global _ACTIVE_THEME, BG, CARD, BORDER, INK, MUTED, ACCENT, ACCENT_DARK
    global ACCENT_SOFT, GOOD, BAD, TAB, HEADING, STATUS
    global BUTTON_FACE, BUTTON_DISABLED, BUTTON_DISABLED_INK
    _ACTIVE_THEME = name if name in _PALETTES else "light"
    palette = _PALETTES[_ACTIVE_THEME]
    BG = palette["BG"]
    CARD = palette["CARD"]
    BORDER = palette["BORDER"]
    INK = palette["INK"]
    MUTED = palette["MUTED"]
    ACCENT = palette["ACCENT"]
    ACCENT_DARK = palette["ACCENT_DARK"]
    ACCENT_SOFT = palette["ACCENT_SOFT"]
    GOOD = palette["GOOD"]
    BAD = palette["BAD"]
    TAB = palette["TAB"]
    HEADING = palette["HEADING"]
    STATUS = palette["STATUS"]
    BUTTON_FACE = palette["BUTTON_FACE"]
    BUTTON_DISABLED = palette["BUTTON_DISABLED"]
    BUTTON_DISABLED_INK = palette["BUTTON_DISABLED_INK"]
    return _ACTIVE_THEME


# -- session persistence (inputs survive restarts) --------------------------- #

#: Bumped whenever the on-disk session layout changes in a way old
#: playgrounds must not silently mis-read.
SESSION_FORMAT = 1

_THEMES = ("light", "dark", "auto")


def session_path() -> Path:
    """Where the playground keeps its session file (``~/.pharmaml``)."""
    return Path.home() / ".pharmaml" / "playground-session.json"


def parse_geometry(text: str) -> tuple[int, int, int, int] | None:
    """Split a Tk geometry string into ``(w, h, x, y)`` (or ``None``).

    Only the full ``WxH+X+Y`` form is accepted -- a bare size or a bare
    position is not enough to restore a window deterministically.
    """
    match = re.match(r"^(\d+)x(\d+)([+-]\d+)([+-]\d+)$", str(text).strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))


def dump_session(
    *,
    theme: str,
    geometry: str,
    tab: int,
    variables: dict[str, Any],
    families: dict[str, bool],
    texts: dict[str, str],
) -> dict[str, Any]:
    """Assemble the JSON-serialisable session payload (pure normaliser).

    Values are coerced to their session types here so the file can never
    carry a half-typed payload, whatever the caller collected.
    """
    return {
        "format": SESSION_FORMAT,
        "theme": theme if theme in _THEMES else "light",
        "geometry": str(geometry),
        "tab": int(tab),
        "variables": {str(name): value for name, value in variables.items()},
        "families": {str(name): bool(checked) for name, checked in families.items()},
        "texts": {str(name): str(content) for name, content in texts.items()},
    }


def load_session(raw: Any) -> dict[str, Any] | None:
    """Validate a parsed session file into a trusted dict (or ``None``).

    Malformed files are rejected wholesale (``None``); structurally fine
    files get every field sanitised back to its session type, so the
    restore path never has to defend itself twice.
    """
    if not isinstance(raw, dict) or raw.get("format") != SESSION_FORMAT:
        return None
    theme = raw.get("theme")
    if theme not in _THEMES:
        theme = "light"
    geometry = raw.get("geometry")
    geometry = geometry if isinstance(geometry, str) else ""
    tab = raw.get("tab")
    if isinstance(tab, bool) or not isinstance(tab, int) or tab < 0:
        tab = 0
    variables = raw.get("variables")
    if not isinstance(variables, dict):
        variables = {}
    variables = {str(name): value for name, value in variables.items()}
    families = raw.get("families")
    if not isinstance(families, dict):
        families = {}
    families = {str(name): checked for name, checked in families.items()
                if isinstance(checked, bool)}
    texts = raw.get("texts")
    if not isinstance(texts, dict):
        texts = {}
    texts = {str(name): content for name, content in texts.items()
             if isinstance(content, str)}
    return {
        "format": SESSION_FORMAT,
        "theme": theme,
        "geometry": geometry,
        "tab": tab,
        "variables": variables,
        "families": families,
        "texts": texts,
    }


def export_session(payload: dict[str, Any], path: str | Path) -> bool:
    """Write a session payload to ``path`` as shareable JSON.

    The file uses the exact same format as the auto-saved session
    (``format`` marker included), so an exported session can be mailed
    to a colleague and imported on their machine.  ``False`` on any
    write failure (unwritable directory, non-serialisable payload) --
    never an exception.
    """
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return True
    except (OSError, TypeError, ValueError):
        return False


def read_json_file(path: str | Path) -> Any | None:
    """Read parsed JSON from ``path`` (``None`` on any failure)."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None

#: (import name, display label, why it matters, pip hint)
_DEPENDENCIES: tuple[tuple[str, str, str, str], ...] = (
    ("numpy", "numpy", "array math behind every domain", "numpy"),
    ("pandas", "pandas", "descriptor / dataset / leaderboard tables", "pandas"),
    ("sklearn", "scikit-learn", "QSAR model training + evaluation", "scikit-learn"),
    ("scipy", "SciPy", "statistics behind biomarker discovery", "scipy"),
    ("matplotlib", "matplotlib", "embedded figures (parity, profiles, curves)", "matplotlib"),
    ("rdkit", "RDKit", "chemistry: descriptors, fingerprints, QED, the GA", "rdkit"),
    ("torch", "PyTorch", "optional: the REINVENT-style RL optimiser", "torch"),
)

_OBJECTIVE_KINDS = (
    "QED drug-likeness",
    "LogP window",
    "Molecular weight window",
    "TPSA window",
    "QED + LogP blend",
    "Multi-objective (QED \u00d7 MW \u00d7 TPSA)",
)

#: Resolved once on first figure render: the TkAgg canvas class, or None when
#: the ``_tkagg`` acceleration cannot load on this Python build (e.g. a Tcl 9
#: build without the legacy ``Tcl_SetVar`` symbol).  None switches figure
#: embedding to an Agg-rendered PNG -- visually equivalent for results panes
#: and portable across every Tcl/Tk combination.
_FIGURE_CANVAS: Any = None
_FIGURE_CANVAS_PROBED = False

_PHARMAML_HINT = (
    "pharmaml could not be imported.\n\n"
    "Either run the playground from the toolkit root (the folder holding\n"
    "src/ and README.md), or install the package first:\n\n"
    "    pip install -e ."
)


# --------------------------------------------------------------------------- #
# environment probing
# --------------------------------------------------------------------------- #
def probe_dependencies() -> dict[str, bool]:
    """Report which optional heavy dependencies are importable.

    Uses :func:`importlib.util.find_spec` (no actual import, so probing
    is instant and side-effect free).
    """
    import importlib.util

    status: dict[str, bool] = {}
    for module, _label, _purpose, _pip in _DEPENDENCIES:
        try:
            status[module] = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):  # pragma: no cover - exotic setups
            status[module] = False
    return status


def load_pharmaml() -> Any:
    """Import :mod:`pharmaml`, falling back to the bundled ``src/`` tree.

    The playground ships inside the toolkit archive next to ``src/``, so
    it works straight after ``unzip`` with no installation -- and keeps
    working unchanged once the package is properly installed.
    """
    try:
        import pharmaml
    except ImportError:
        if not SRC_DIR.is_dir():
            raise
        sys.path.insert(0, str(SRC_DIR))
        import pharmaml
    return pharmaml


# --------------------------------------------------------------------------- #
# small pure helpers (unit-tested in tests/test_playground.py)
# --------------------------------------------------------------------------- #
def parse_float(
    raw: str,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Parse a widget string into a finite float with a friendly error.

    Raises
    ------
    ValueError
        When the text is not a number, is NaN/inf, or violates the
        optional ``minimum`` / ``maximum`` bounds.
    """
    text = str(raw).strip()
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{field}: {raw!r} is not a number") from exc
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{field}: {raw!r} is not a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field}: {value} is below the minimum {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field}: {value} is above the maximum {maximum}")
    return value


def parse_int(
    raw: str,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Parse a widget string into an int with a friendly error."""
    text = str(raw).strip()
    try:
        value = int(float(text))
    except ValueError as exc:
        raise ValueError(f"{field}: {raw!r} is not a whole number") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{field}: {value} is below the minimum {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field}: {value} is above the maximum {maximum}")
    return value


def _split_triple(raw: str) -> list[str]:
    """Split an ``"x y z"`` / ``"x,y,z"`` widget string into three parts."""
    return str(raw).strip().replace(",", " ").split()


def parse_int_triple(
    raw: str,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> tuple[int, int, int]:
    """Parse an ``"n n n"`` grid-points string into a tuple of three ints.

    Accepts spaces or commas (``"60 60 60"`` and ``"60,60,60"`` are
    the same input); anything that is not exactly three whole numbers
    raises :class:`ValueError` with the field name in the message.
    """
    parts = _split_triple(raw)
    if len(parts) != 3:
        raise ValueError(
            f"{field}: {raw!r} must be three whole numbers, e.g. '60 60 60'"
        )
    first, second, third = (
        parse_int(part, field, minimum=minimum, maximum=maximum)
        for part in parts
    )
    return (first, second, third)


def parse_float_triple(raw: str, field: str) -> tuple[float, float, float]:
    """Parse an ``"x y z"`` centre/size string into three finite floats."""
    parts = _split_triple(raw)
    if len(parts) != 3:
        raise ValueError(f"{field}: {raw!r} must be three numbers, e.g. '20 20 20'")
    first, second, third = (parse_float(part, field) for part in parts)
    return (first, second, third)


#: The Docking tab's campaign-runner engines (combobox values).
_DOCK_ENGINES = ("AutoDock4 campaign", "Vina single ligand")


def docking_engine_binaries(engine: str) -> tuple[str, ...]:
    """The external binaries an engine needs on ``PATH``.

    The AutoDock4 campaign runner drives ``autogrid4`` + ``autodock4``
    over every ligand folder; the Vina engine needs the single ``vina``
    binary.  Used to pre-empt the workflow module's own RuntimeError
    with a friendly install hint before any job is started.
    """
    if engine == "AutoDock4 campaign":
        return ("autogrid4", "autodock4")
    return ("vina",)


def parse_smiles_lines(text: str) -> list[str]:
    """Extract non-empty SMILES lines from a corpus text widget.

    Blank lines and ``#`` comments are dropped; order is preserved and
    duplicates are kept (the optimisers decide how to handle them).
    """
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


# --------------------------------------------------------------------------- #
# the Chemistry tab's feature factory (the rar's 7-family ML pipeline)
# --------------------------------------------------------------------------- #
#: Feature-factory families in canonical (frame column) order:
#: ``(pharmaml family name, checkbox label)``.  The bit widths mirror
#: ``pharmaml.chem.features``; the descriptor family's width depends on
#: the installed RDKit and is resolved at build time instead.
_FEATURE_FAMILY_LABELS: tuple[tuple[str, str], ...] = (
    ("descriptors", "RDKit descriptors"),
    ("atompair_count", "Atom-pair counts (2048)"),
    ("ecfp", "ECFP bits (2048)"),
    ("avalon_count", "Avalon counts (512)"),
    ("rdkit_fingerprint", "RDKit fingerprint (2048)"),
    ("morgan_full", "Morgan full (2048)"),
    ("morgan_scaffold", "Morgan scaffold (2048)"),
)

#: Column count of each fingerprint family.
_FEATURE_FAMILY_BITS: dict[str, int] = {
    "atompair_count": 2048,
    "ecfp": 2048,
    "avalon_count": 512,
    "rdkit_fingerprint": 2048,
    "morgan_full": 2048,
    "morgan_scaffold": 2048,
}

#: Descriptor-family width when RDKit cannot be probed (estimate only).
_FEATURE_DESCRIPTORS_FALLBACK = 50

#: Column names auto-detected for the SMILES column (case-insensitive;
#: mirrors the docking module's convention, the rar hard-coded "SMILES").
_SMILES_COLUMN_CANDIDATES = (
    "smiles",
    "smile",
    "canonical_smiles",
    "smiles_nostereo",
    "smiles_no_stereo",
    "smiles_string",
    "canonical",
)

#: Target-combobox entry meaning "no target -- rank by variance instead".
_FEAT_NO_TARGET = "(none — variance ranking)"

#: The ranker combobox values, in display order.
_FEAT_RANKERS = (
    "Auto (RandomForest or LightGBM)",
    "Random forest",
    "LightGBM (if installed)",
    "Variance (no target needed)",
)

#: Ranker combobox label -> ``rank_features`` ``method`` argument.
_FEAT_RANKER_METHODS: dict[str, str] = {
    "Auto (RandomForest or LightGBM)": "auto",
    "Random forest": "random_forest",
    "LightGBM (if installed)": "lightgbm",
    "Variance (no target needed)": "variance",
}

#: The task combobox values, in display order.
_FEAT_TASKS = ("Auto detect", "Classification", "Regression")

#: Task combobox label -> ``rank_features`` ``task`` argument.
_FEAT_TASK_METHODS: dict[str, str] = {
    "Auto detect": "auto",
    "Classification": "classification",
    "Regression": "regression",
}

#: The card's default pasted corpus: three well-known drugs.
_FEAT_DEFAULT_SMILES = (
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",   # caffeine
    "CC(=O)Nc1ccc(O)cc1",             # paracetamol
)

#: Tooltip on the K entry: the rar pipeline's own top-K presets.
_FEAT_K_PRESETS = "200 / 500 / 1000 / 1500 are the pipeline's presets"


def feature_factory_width(families: Iterable[str], descriptor_count: int) -> int:
    """Estimated column count of a feature-factory run, before it happens.

    Sums the fingerprint bit widths of the requested families plus
    ``descriptor_count`` for the descriptor family, so the Chemistry tab
    can preview the matrix width as the family checkboxes toggle.

    Raises
    ------
    ValueError
        When a name is not a feature-factory family.
    """
    total = 0
    for family in families:
        if family == "descriptors":
            total += descriptor_count
        elif family in _FEATURE_FAMILY_BITS:
            total += _FEATURE_FAMILY_BITS[family]
        else:
            raise ValueError(
                f"unknown feature family {family!r}; valid families are: "
                + ", ".join(name for name, _label in _FEATURE_FAMILY_LABELS)
            )
    return total


def detect_smiles_column(frame: Any) -> str | None:
    """Find the SMILES column of a loaded table, or None when absent.

    Name-based and case-insensitive over the same candidate list the
    docking module uses (``smiles``, ``smile``, ``canonical_smiles``,
    ...); any object with a ``.columns`` attribute works.
    """
    lowered = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in _SMILES_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def format_number(value: float, digits: int = 4) -> str:
    """Human-friendly number rendering for result labels and tables."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    magnitude = abs(value)
    if value != 0 and (magnitude >= 1e5 or magnitude < 1e-3):
        return f"{value:.3e}"
    return f"{value:,.{digits}f}"


def build_objective(kind: str, lo: float, hi: float) -> Any:
    """Instantiate a generative objective selected by its GUI label."""
    from pharmaml.generative import (
        LogPBandObjective,
        MolecularWeightBandObjective,
        MultiObjective,
        QedObjective,
        TpsaBandObjective,
    )

    if kind == "QED drug-likeness":
        return QedObjective()
    if kind == "LogP window":
        return LogPBandObjective(lo, hi)
    if kind == "Molecular weight window":
        return MolecularWeightBandObjective(lo, hi)
    if kind == "TPSA window":
        return TpsaBandObjective(lo, hi)
    if kind == "Multi-objective (QED \u00d7 MW \u00d7 TPSA)":
        # the CLI's 'druglike' recipe: equal-weight triple whose blended
        # leaderboard hides the trade-offs the Pareto front card surfaces.
        return MultiObjective(
            [
                QedObjective(),
                MolecularWeightBandObjective(150, 450),
                TpsaBandObjective(30, 90),
            ]
        )
    return MultiObjective(
        [QedObjective(), LogPBandObjective(lo, hi)], weights=[0.7, 0.3]
    )


def _cell_text(value: Any, limit: int = 26) -> str:
    """Render one table cell for the preview/leaderboard trees."""
    if value is None:
        return ""
    if isinstance(value, float):
        return format_number(value)
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------- #
# widgets
# --------------------------------------------------------------------------- #
class Tooltip:
    """A small delayed hover tooltip attached to any widget."""

    def __init__(self, widget: Any, text: str) -> None:
        self._widget = widget
        self._text = text
        self._after_id: Any = None
        self._tip: Any = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)

    def _schedule(self, _event: Any = None) -> None:
        self._cancel()
        self._after_id = self._widget.after(550, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self._widget.after_cancel(self._after_id)
            except tk.TclError:  # pragma: no cover - widget already gone
                pass
            self._after_id = None

    def _show(self) -> None:
        if self._tip is not None or not self._text:
            return
        x = self._widget.winfo_rootx() + 16
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 6
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tip,
            text=self._text,
            justify="left",
            wraplength=340,
            background="#111827",
            foreground="#f9fafb",
            padx=9,
            pady=5,
        ).pack()

    def _hide(self, _event: Any = None) -> None:
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class ScrollFrame(ttk.Frame):  # type: ignore[misc] # ttk is None headless
    """A vertically scrollable tab page with mouse-wheel support.

    Cards live on ``self.body``; the inner frame always stays as wide as
    the viewport so cards stretch with the window.  ``self.footer`` is a
    NON-scrolling strip pinned below the canvas (packed first, so the
    scroll area shrinks around it) -- tab-level affordances that must
    stay visible live there (the per-tab reset button, 100x-55).
    """

    def __init__(self, master: Any) -> None:
        super().__init__(master, style="Tab.TFrame")
        self.footer = ttk.Frame(self, style="Tab.TFrame")
        self.footer.pack(side="bottom", fill="x")
        self._canvas = tk.Canvas(self, background=BG, highlightthickness=0, bd=0)
        self._bar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self.body = ttk.Frame(self._canvas, style="Tab.TFrame")
        self._window = self._canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind(
            "<Configure>",
            lambda _event: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.configure(yscrollcommand=self._bar.set)
        self._canvas.bind(
            "<Configure>",
            lambda _event: self._canvas.itemconfigure(
                self._window, width=self._canvas.winfo_width()
            ),
        )
        self._bar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        for host in (self, self._canvas, self.body):
            host.bind("<Enter>", lambda _event: self._bind_wheel())
            host.bind("<Leave>", lambda _event: self._unbind_wheel())

    def _bind_wheel(self) -> None:
        if self.tk.call("tk", "windowingsystem") == "x11":
            self.bind_all("<Button-4>", self._on_wheel)
            self.bind_all("<Button-5>", self._on_wheel)
        else:
            self.bind_all("<MouseWheel>", self._on_wheel)

    def _unbind_wheel(self) -> None:
        if self.tk.call("tk", "windowingsystem") == "x11":
            self.unbind_all("<Button-4>")
            self.unbind_all("<Button-5>")
        else:
            self.unbind_all("<MouseWheel>")

    def _on_wheel(self, event: Any) -> None:
        if getattr(event, "num", 0) == 4 or getattr(event, "delta", 0) > 0:
            self._canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", 0) == 5 or getattr(event, "delta", 0) < 0:
            self._canvas.yview_scroll(3, "units")


def make_card(
    parent: Any,
    title: str,
    subtitle: str = "",
    *,
    row: int = 0,
    column: int = 0,
    span: int = 1,
) -> Any:
    """Create a white card with a heading on a tab's card grid.

    Returns the content frame callers grid their controls into (the
    content frame is itself grid-managed with two weighted columns).
    """
    shell = tk.Frame(parent, background=CARD, highlightbackground=BORDER,
                     highlightthickness=1)
    shell.grid(row=row, column=column, columnspan=span, sticky="nsew",
               padx=6, pady=6)
    header = ttk.Frame(shell, style="Card.TFrame", padding=(14, 10, 14, 0))
    header.pack(fill="x")
    ttk.Label(header, text=title, style="CardTitle.TLabel").pack(anchor="w")
    if subtitle:
        ttk.Label(header, text=subtitle, style="CardMuted.TLabel",
                  wraplength=620, justify="left").pack(anchor="w", pady=(2, 0))
    content = ttk.Frame(shell, style="Card.TFrame", padding=(14, 8, 14, 12))
    content.pack(fill="both", expand=True)
    content.columnconfigure(1, weight=1)
    return content


def form_row(parent: Any, label: str, widget: Any) -> None:
    """Append a ``label | widget`` row onto a card's content grid."""
    row = parent.grid_size()[1]
    ttk.Label(parent, text=label, style="Field.TLabel").grid(
        row=row, column=0, sticky="w", padx=(0, 10), pady=3
    )
    widget.grid(row=row, column=1, sticky="ew", pady=3)


def grid_append(parent: Any, widget: Any, *, columnspan: int = 2,
                sticky: str = "w", pady: tuple[int, int] = (0, 0),
                weight: bool = False) -> int:
    """Append ``widget`` on the next free row of a card's content grid.

    Card content frames are grid-managed (``form_row`` siblings), so
    every child must use grid -- never pack -- inside them.
    """
    row = parent.grid_size()[1]
    widget.grid(row=row, column=0, columnspan=columnspan, sticky=sticky,
                pady=pady)
    if weight:
        parent.rowconfigure(row, weight=1)
    return row


def bind_enter(widget: Any, button: Any) -> None:
    """Let pressing Return in ``widget`` invoke ``button``."""
    widget.bind("<Return>", lambda _event: button.invoke())


def _grid_text(parent: Any, height: int = 6, mono: Any = None) -> Any:
    """A flat white multi-line text box (corpus editing, stats dumps)."""
    box = tk.Text(
        parent,
        height=height,
        background=CARD,
        foreground=INK,
        relief="solid",
        borderwidth=1,
        highlightthickness=0,
        padx=8,
        pady=6,
        font=mono or ("TkFixedFont", 10),
        wrap="none",
    )
    return box


# --------------------------------------------------------------------------- #
# the application
# --------------------------------------------------------------------------- #
if tk is not None:
    _TkBase: Any = tk.Tk
else:  # allow `import playground` on tkinter-less builds (probe path)
    class _TkBase:  # type: ignore[no-redef]
        """Placeholder superclass when tkinter is unavailable."""


class PlaygroundApp(_TkBase):  # type: ignore[misc]
    """The playground window: one notebook, seven tabs, one worker."""

    def __init__(self, dependencies: dict[str, bool] | None = None,
                 persist_session: bool = True) -> None:
        """Build the window.

        ``persist_session`` switches the whole session machinery on or
        off: the real app reads ``~/.pharmaml/playground-session.json``
        at startup (restoring inputs, geometry, active tab and theme)
        and writes it back on close, while ``--self-test`` passes
        ``False`` so CI-style runs stay deterministic and never touch
        (let alone overwrite) a developer's saved session.
        """
        if tk is None:  # pragma: no cover - guarded by main()
            raise RuntimeError(
                "tkinter is not available in this Python build -- install the "
                "python3-tk system package (or use an official python.org build)"
            )
        super().__init__()
        self.deps = dependencies if dependencies is not None else probe_dependencies()
        self.pharmaml: Any = None
        self.pharmaml_error = ""
        try:
            self.pharmaml = load_pharmaml()
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
            self.pharmaml_error = f"{type(exc).__name__}: {exc}"

        self._jobs: queue.Queue[tuple[str, Any, Any]] = queue.Queue()
        self._busy = False
        self._on_done: Callable[[Any], None] | None = None
        self._action_buttons: list[Any] = []
        self._last_figure: Any = None
        self._ml_frame: Any = None
        self._ml_path: Path | None = None
        self._desc_frame: Any = None
        self._desc_rows: list[tuple[str, str, str]] = []
        self._ga_result: Any = None
        # -- Generative Pareto card state (100x-58) ------------------ #
        self._pareto_front: Any = None      # ParetoFront of the last run
        self._pareto_result: Any = None     # the GenerationResult it came from
        # -- Cell Lines tab state ----------------------------------- #
        self._bio_screen: Any = None          # ResponseScreen (maybe collapsed)
        self._bio_screen_path: Path | None = None
        self._bio_collapsed = False
        self._bio_matrix: Any = None           # cell line x compound DataFrame
        self._bio_matrix_metric: str | None = None
        self._bio_expr: Any = None             # raw expression frame
        self._bio_expr_norm: Any = None        # normalize + impute cache
        # -- Docking tab state --------------------------------------- #
        self._dock_library: Any = None         # LigandLibrary from the CSV
        self._dock_lib_path: Path | None = None
        self._dock_out_dir: Path | None = None  # ligand campaign output root
        self._dock_report: Any = None          # LigandPreparationReport
        self._dock_receptor_source: Path | None = None  # PDB / CIF / PDBQT in
        self._dock_protein_report: Any = None  # ProteinReport
        self._dock_receptor_path: Path | None = None    # prepared receptor out
        self._dock_ligand_path: Path | None = None      # reference ligand PDBQT
        self._dock_box: Any = None             # GridBox
        self._dock_results_frame: Any = None   # current results table
        self._dock_results_source: str | None = None    # campaign / dlg / vina
        self._dock_campaign_root: Path | None = None    # AutoDock4 batch root
        self._dock_vina_receptor: Path | None = None
        self._dock_vina_ligand: Path | None = None
        # -- Feature factory state (Chemistry tab) ------------------- #
        self._feat_frame: Any = None            # full computed feature frame
        self._feat_report: Any = None           # FeatureReport of the last run
        self._feat_selection: Any = None        # FeatureSelection of the last run
        self._feat_smiles: list[str] = []       # valid SMILES, frame row order
        self._feat_target_series: list[float] | None = None  # aligned target
        self._feat_target_name: str | None = None  # target column name (or None)
        self._feat_k: int = 0                   # K requested by the last run
        self._feat_csv_frame: Any = None        # the loaded source CSV frame
        self._feat_csv_path: Path | None = None
        self._feat_smiles_column: str | None = None
        self._feat_descriptor_count: int = _FEATURE_DESCRIPTORS_FALLBACK

        # -- session + theme state ------------------------------------- #
        self._persist_session = persist_session
        self._session: dict[str, Any] | None = None
        if persist_session:
            self._session = load_session(self._read_session_file())
        self._theme = self._session["theme"] if self._session else "light"
        self._theme_var: Any = None          # built by _build_menu (View menu)
        self._menu_tints: list[Any] = []     # menus to recolour on theme flip
        apply_theme(self._resolve_auto_theme() if self._theme == "auto"
                    else self._theme)

        self.title("PharmaML Playground")
        self.configure(background=BG)
        self._setup_dpi()
        width, height = self._window_size(1180, 780)
        self.geometry(f"{width}x{height}")
        self.minsize(960, 620)

        self._setup_fonts()
        self._build_style()
        self._build_menu()
        self._build_statusbar()
        self._build_tabs()
        # Per-tab reset support (100x-54): snapshot the freshly built
        # (default) inputs and which tab owns which input BEFORE the
        # session restore overwrites either of them.
        self._tab_input_owners = self._tab_input_names()
        self._session_defaults = self._collect_session()
        restored_geometry = self._restore_session()
        if not restored_geometry:
            self._center_on_screen(width, height)
        self.lift()
        self.attributes("-topmost", True)
        self.after(250, lambda: self.attributes("-topmost", False))
        self.after(80, self._poll_jobs)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- plumbing -------------------------------------------------------- #
    def _setup_dpi(self) -> None:
        """Ask Windows for crisp text on high-DPI screens."""
        if sys.platform == "win32":
            import ctypes

            with contextlib.suppress(Exception):
                ctypes.windll.shcore.SetProcessDpiAwareness(1)

    def _window_size(self, width: int, height: int) -> tuple[int, int]:
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        return min(width, screen_w - 40), min(height, screen_h - 90)

    def _center_on_screen(self, width: int, height: int) -> None:
        x = max(0, (self.winfo_screenwidth() - width) // 2)
        y = max(0, (self.winfo_screenheight() - height) // 3)
        self.geometry(f"+{x}+{y}")

    def _setup_fonts(self) -> None:
        fixed = tkfont.nametofont("TkFixedFont")
        fixed.configure(size=10)
        self._mono = fixed
        result_font = fixed.copy()
        result_font.configure(size=12, weight="bold")
        self._mono_result = result_font

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:  # pragma: no cover - exotic Tk builds
            pass
        style.configure(".", background=BG, foreground=INK)
        style.configure("TFrame", background=BG)
        style.configure("Tab.TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("CardTitle.TLabel", background=CARD, foreground=INK,
                        font=("TkDefaultFont", 11, "bold"))
        style.configure("CardMuted.TLabel", background=CARD, foreground=MUTED,
                        font=("TkDefaultFont", 9))
        style.configure("Field.TLabel", background=CARD, foreground=INK)
        style.configure("Result.TLabel", background=CARD, foreground=ACCENT_DARK)
        style.configure("Title.TLabel", background=CARD, foreground=INK,
                        font=("TkDefaultFont", 15, "bold"))
        style.configure("Primary.TButton", background=ACCENT, foreground="#ffffff",
                        borderwidth=0, focusthickness=1, focuscolor="#ffffff",
                        padding=(16, 7), font=("TkDefaultFont", 10, "bold"))
        style.map("Primary.TButton",
                  background=[("active", ACCENT_DARK), ("disabled", BUTTON_DISABLED)],
                  foreground=[("disabled", BUTTON_DISABLED_INK)])
        style.configure("TButton", padding=(10, 5), background=BUTTON_FACE,
                        foreground=INK)
        style.map("TButton",
                  background=[("active", ACCENT_SOFT), ("disabled", BG)],
                  foreground=[("disabled", MUTED)])
        # the quiet tab-footer reset button (100x-55): flat on the page
        # background, muted until hovered -- an affordance, not an action
        style.configure("Ghost.TButton", background=BG, foreground=MUTED,
                        borderwidth=0, lightcolor=BG, darkcolor=BG,
                        focusthickness=1, focuscolor=MUTED, padding=(8, 2),
                        font=("TkDefaultFont", 9))
        style.map("Ghost.TButton",
                  background=[("active", BG), ("disabled", BG)],
                  foreground=[("active", ACCENT), ("disabled", BORDER)],
                  lightcolor=[("pressed", BG)],
                  darkcolor=[("pressed", BG)])
        style.configure("TNotebook", background=BG, borderwidth=0,
                        tabmargins=(10, 6, 12, 0))
        style.configure("TNotebook.Tab", background=TAB, foreground=MUTED,
                        padding=(16, 8), font=("TkDefaultFont", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", CARD)],
                  foreground=[("selected", ACCENT)])
        style.configure("Treeview", background=CARD, fieldbackground=CARD,
                        foreground=INK, rowheight=26, borderwidth=0)
        style.map("Treeview",
                  background=[("selected", ACCENT_SOFT)],
                  foreground=[("selected", INK)])
        style.configure("Treeview.Heading", background=HEADING, foreground=INK,
                        font=("TkDefaultFont", 9, "bold"), relief="flat")
        style.configure("TEntry", fieldbackground=CARD, bordercolor=BORDER,
                        lightcolor=BORDER, darkcolor=BORDER, padding=4)
        style.configure("TCombobox", fieldbackground=CARD, background=CARD,
                        arrowcolor=ACCENT, bordercolor=BORDER, padding=3)
        style.configure("TSpinbox", fieldbackground=CARD, bordercolor=BORDER,
                        arrowcolor=ACCENT, padding=3)
        style.configure("TCheckbutton", background=CARD, foreground=INK,
                        focuscolor=ACCENT)
        style.map("TCheckbutton", background=[("active", CARD)])
        style.configure("Horizontal.TProgressbar", background=ACCENT,
                        troughcolor=TAB, borderwidth=0, thickness=8)
        style.configure("TProgressbar", background=ACCENT, troughcolor=TAB)
        style.configure("Status.TFrame", background=STATUS)
        style.configure("Status.TLabel", background=STATUS, foreground=MUTED,
                        font=("TkDefaultFont", 9))
        # clam's state maps and trough colours ship full of light defaults
        # (readonly combobox fields, focus borders, pressed buttons,
        # scrollbar troughs).  ``style.map`` REPLACES the whole map for
        # each option, so this block pins every reachable state to the
        # active palette -- without it dark mode keeps bright stragglers.
        style.configure("TScrollbar", troughcolor=TAB, background=BUTTON_FACE,
                        arrowcolor=MUTED, lightcolor=BORDER, darkcolor=BORDER,
                        bordercolor=BORDER)
        style.map("TScrollbar",
                  background=[("active", ACCENT_SOFT),
                              ("pressed", ACCENT_SOFT)])
        style.configure("TEntry", insertcolor=INK)
        style.map("TEntry",
                  background=[("readonly", BG)],
                  fieldbackground=[("readonly", CARD)],
                  bordercolor=[("focus", ACCENT)])
        style.map("TCombobox",
                  fieldbackground=[("readonly", CARD)],
                  foreground=[("readonly", INK)],
                  bordercolor=[("focus", ACCENT)])
        style.map("TSpinbox",
                  background=[("readonly", BG)],
                  fieldbackground=[("readonly", CARD)],
                  bordercolor=[("focus", ACCENT)])
        style.map("TButton",
                  darkcolor=[("pressed", BORDER)],
                  lightcolor=[("pressed", BORDER)])
        style.map("Treeview", bordercolor=[("focus", BORDER)])
        for name in ("TEntry", "TCombobox", "TSpinbox"):
            style.configure(name, selectbackground=ACCENT_SOFT,
                            selectforeground=INK)
        for key, value in (
            ("*TCombobox*Listbox.background", CARD),
            ("*TCombobox*Listbox.foreground", INK),
            ("*TCombobox*Listbox.selectBackground", ACCENT_SOFT),
            ("*TCombobox*Listbox.selectForeground", INK),
        ):
            self.option_add(key, value)

    def _build_menu(self) -> None:
        bar = tk.Menu(self)
        file_menu = tk.Menu(bar, tearoff=0)
        file_menu.add_command(label="Open CSV…", accelerator="Ctrl+O",
                              command=self._menu_open_csv)
        file_menu.add_command(label="Save last figure…", accelerator="Ctrl+S",
                              command=self._menu_save_figure)
        file_menu.add_separator()
        file_menu.add_command(label="Save session now", accelerator="Ctrl+Shift+S",
                              command=self._save_session_now)
        file_menu.add_command(label="Clear saved session…",
                              command=self._clear_saved_session)
        file_menu.add_command(label="Export session…",
                              command=self._export_session)
        file_menu.add_command(label="Import session…",
                              command=self._import_session)
        file_menu.add_command(label="Reset current tab to defaults",
                              accelerator="Ctrl+Shift+R",
                              command=self._reset_current_tab)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", accelerator="Ctrl+Q",
                              command=self._on_close)
        bar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(bar, tearoff=0)
        self._theme_var = tk.StringVar(value=self._theme)
        for label, value in (("Follow system", "auto"), ("Light", "light"),
                             ("Dark", "dark")):
            view_menu.add_radiobutton(label=label, value=value,
                                       variable=self._theme_var,
                                       command=self._on_theme_change)
        bar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(bar, tearoff=0)
        help_menu.add_command(label="About PharmaML Playground",
                              command=self._show_about)
        help_menu.add_command(label="Dependency report",
                              command=self._show_dependency_report)
        bar.add_cascade(label="Help", menu=help_menu)
        self.configure(menu=bar)
        self._menu_tints = [bar, file_menu, view_menu, help_menu]
        self._tint_menus()
        self.bind_all("<Control-o>", lambda _e: self._menu_open_csv())
        self.bind_all("<Control-s>", lambda _e: self._menu_save_figure())
        self.bind_all("<Control-Shift-S>", lambda _e: self._save_session_now())
        self.bind_all("<Control-q>", lambda _e: self._on_close())

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self, style="Status.TFrame", padding=(12, 5))
        bar.pack(side="bottom", fill="x")
        if self.pharmaml is not None:
            left = f"pharmaml {self.pharmaml.__version__} · toolkit v{__version__}"
        else:
            left = ("pharmaml not loaded -- run from the toolkit root or "
                    f"pip install -e . · toolkit v{__version__}")
        ttk.Label(bar, text=left, style="Status.TLabel").pack(side="left")
        self._job_label = ttk.Label(bar, text="idle", style="Status.TLabel")
        self._job_label.pack(side="right", padx=(12, 0))
        self._progress = ttk.Progressbar(bar, mode="indeterminate", length=130)
        self._progress.pack(side="right")

    def _build_tabs(self) -> None:
        self._book = ttk.Notebook(self)
        self._book.pack(fill="both", expand=True, padx=10, pady=(10, 4))
        self._build_home_tab()
        self._build_chemistry_tab()
        self._build_pk_tab()
        self._build_cellline_tab()
        self._build_docking_tab()
        self._build_ml_tab()
        self._build_generative_tab()

    def _new_page(self, title: str, reset: bool = True) -> Any:
        page = ScrollFrame(self._book)
        grid = ttk.Frame(page.body, style="Tab.TFrame")
        grid.pack(fill="both", expand=True)
        grid.columnconfigure(0, weight=1, uniform="cards")
        grid.columnconfigure(1, weight=1, uniform="cards")
        page.grid = grid  # type: ignore[attr-defined]
        self._book.add(page, text=f"  {title}  ")
        if reset:
            # per-tab reset button (100x-55): the ownership map is keyed
            # by tab index, and add() just appended this page -- its index
            # is the last one.  Pinned in the non-scrolling footer so it
            # stays reachable without scrolling back up.
            index = len(self._book.tabs()) - 1
            button = ttk.Button(page.footer, text="Reset this tab to defaults",
                                style="Ghost.TButton",
                                command=lambda idx=index: self._reset_tab(idx))
            button.pack(side="right", padx=(0, 14), pady=(4, 5))
            Tooltip(button, "Restore only THIS tab's inputs to their launch "
                            "defaults -- every other tab keeps its live "
                            "values (same as File \u25b8 Reset current tab).")
            # per-tab copy button (100x-60): the same ownership map, read
            # LIVE -- a share-lite JSON snippet of this tab's inputs on
            # the clipboard (paste into an issue, a script, or a message).
            copy_button = ttk.Button(page.footer, text="Copy inputs as JSON",
                                     style="Ghost.TButton",
                                     command=lambda idx=index:
                                         self._copy_tab_inputs(idx))
            copy_button.pack(side="right", padx=(0, 6), pady=(4, 5))
            Tooltip(copy_button,
                    "Copy THIS tab's live inputs as a JSON snippet -- the "
                    "same fields the session file stores, for one tab. "
                    "Paste it into an issue report or a colleague's "
                    "import (File \u25b8 Import session takes full files).")
        return page

    def _action_button(self, parent: Any, text: str, command: Callable[[], None],
                       tooltip: str = "", primary: bool = False) -> Any:
        button = ttk.Button(parent, text=text, command=command,
                            style="Primary.TButton" if primary else "TButton")
        if tooltip:
            Tooltip(button, tooltip)
        self._action_buttons.append(button)
        return button

    def _show_error(self, title: str, exc: BaseException) -> None:
        detail = f"{type(exc).__name__}: {exc}"
        print(detail, file=sys.stderr)
        messagebox.showerror(title, detail)

    def _require(self, module: str, feature: str) -> bool:
        """Gate a feature on a probed dependency with an install hint."""
        if self.deps.get(module):
            return True
        pip_hint = next(
            hint for name, _l, _p, hint in _DEPENDENCIES if name == module
        )
        messagebox.showwarning(
            f"{feature} needs {module}",
            f"{feature} needs the {module} package, which is not installed.\n\n"
            f"    pip install {pip_hint}\n\n"
            "Install it, then restart the playground.",
        )
        return False

    def _require_pharmaml(self) -> bool:
        if self.pharmaml is not None:
            return True
        messagebox.showerror("pharmaml not loaded", _PHARMAML_HINT)
        return False

    def _start_job(self, label: str, work: Callable[[], Any],
                   on_done: Callable[[Any], None] | None = None) -> None:
        """Run ``work`` in a worker thread; render via ``on_done``."""
        if self._busy:
            messagebox.showinfo(
                "Busy",
                "Another calculation is still running -- please wait for it.",
            )
            return
        self._busy = True
        self._on_done = on_done
        for button in self._action_buttons:
            if button.winfo_exists():
                button.state(["disabled"])
        self._progress.start(12)
        self._job_label.configure(text=label)

        def runner() -> None:
            try:
                self._jobs.put((work(), None))
            except BaseException as exc:  # noqa: BLE001 - surfaced in the UI
                self._jobs.put((None, exc))

        threading.Thread(target=runner, name="playground-job", daemon=True).start()

    def _poll_jobs(self) -> None:
        try:
            result, error = self._jobs.get_nowait()
        except queue.Empty:
            self.after(80, self._poll_jobs)
            return
        self._busy = False
        self._progress.stop()
        self._job_label.configure(text="idle")
        for button in self._action_buttons:
            if button.winfo_exists():
                button.state(["!disabled"])
        if error is not None:
            traceback.print_exception(type(error), error, error.__traceback__)
            messagebox.showerror(
                "Calculation failed", f"{type(error).__name__}: {error}"
            )
        elif self._on_done is not None:
            try:
                self._on_done(result)
            except Exception as exc:  # noqa: BLE001 - render errors are ours
                self._show_error("Could not display the result", exc)
        self._on_done = None
        self.after(80, self._poll_jobs)

    def report_callback_exception(self, exc: Any, val: Any, tb: Any) -> None:
        """Route Tk callback crashes to the console and a dialog."""
        traceback.print_exception(exc, val, tb)
        messagebox.showerror("Unexpected error", f"{type(val).__name__}: {val}")

    def _copy(self, text: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)
        self._job_label.configure(text="copied to clipboard")
        self.after(1400, lambda: self._job_label.configure(
            text="busy…" if self._busy else "idle"))

    def _close_embedded(self, host: Any) -> None:
        figure = getattr(host, "_pg_figure", None)
        if figure is None:
            return
        with contextlib.suppress(Exception):
            from pharmaml.viz import close_figure

            close_figure(figure)
        host._pg_figure = None

    def _figure_canvas_class(self) -> Any:
        """Return ``FigureCanvasTkAgg`` once, or None when it cannot load."""
        global _FIGURE_CANVAS, _FIGURE_CANVAS_PROBED
        if not _FIGURE_CANVAS_PROBED:
            _FIGURE_CANVAS_PROBED = True
            try:
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

                _FIGURE_CANVAS = FigureCanvasTkAgg
            except ImportError:
                print(
                    "note: the TkAgg acceleration could not load on this Python "
                    "build -- figures render as PNG instead (fully functional)."
                )
                _FIGURE_CANVAS = None
        return _FIGURE_CANVAS

    def _render_figure(self, host: Any, figure: Any) -> None:
        """Embed a matplotlib figure into ``host`` (replacing any prior one).

        Prefers the live TkAgg canvas; when the TkAgg acceleration cannot
        load (some Tcl 9 builds), falls back to rendering the figure with
        the Agg backend and displaying the PNG directly -- same figures,
        every machine.
        """
        self._close_embedded(host)
        for child in host.winfo_children():
            child.destroy()
        canvas_cls = self._figure_canvas_class()
        if canvas_cls is not None:
            canvas = canvas_cls(figure, master=host)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.configure(background=CARD, highlightthickness=0)
            widget.pack(fill="both", expand=True)
            host._pg_figure = figure
            self._last_figure = figure
            return
        import io

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=110, facecolor=CARD)
        photo = tk.PhotoImage(data=buffer.getvalue())
        label = tk.Label(host, image=photo, background=CARD)
        label.pack(fill="both", expand=True)
        label._pg_photo = photo  # keep a reference -- Tk frees unreferenced images
        host._pg_figure = figure
        self._last_figure = figure

    def _figure_host(self, parent: Any, height: int = 330) -> Any:
        """A reserved white area (grid-managed) that figures embed into."""
        host = tk.Frame(parent, background=CARD, height=height)
        grid_append(parent, host, sticky="nsew", pady=(6, 0), weight=True)
        host.pack_propagate(False)
        return host

    # -- session + theme ------------------------------------------------- #
    def _read_session_file(self) -> Any:
        """Parse the session file, or ``None`` when absent/corrupt."""
        try:
            with session_path().open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return None

    def _resolve_auto_theme(self) -> str:
        """Map ``"auto"`` onto light/dark using the OS preference (Tk 8.6.13+).

        Anything the interpreter cannot answer (older Tk, Linux) falls
        back to light -- the palette the window was designed around.
        """
        with contextlib.suppress(Exception):
            if bool(self.tk.call("tk", "dark")):
                return "dark"
        return "light"

    def _on_theme_change(self) -> None:
        """View-menu callback: flip the palette across the live window."""
        choice = self._theme_var.get()
        self._theme = choice if choice in _THEMES else "light"
        self._set_theme(self._resolve_auto_theme() if self._theme == "auto"
                        else self._theme)

    def _set_theme(self, resolved: str) -> None:
        """Apply a resolved palette and recolour every live widget."""
        previous = _PALETTES[_ACTIVE_THEME]
        apply_theme(resolved)
        self.configure(background=BG)
        self._build_style()
        self._tint_menus()
        self._retint_plain_widgets(previous)

    def _tint_menus(self) -> None:
        """Recolour the menu bar for the active palette."""
        for menu in self._menu_tints:
            with contextlib.suppress(tk.TclError):
                menu.configure(
                    background=HEADING if _ACTIVE_THEME == "dark" else CARD,
                    foreground=INK,
                    activebackground=ACCENT_SOFT,
                    activeforeground=INK,
                )

    def _retint_plain_widgets(self, previous: dict[str, str]) -> None:
        """Walk the widget tree recolouring plain ``tk`` widgets.

        Only widgets still carrying a colour from the *previous* palette
        are touched, which is what keeps matplotlib's white figure
        canvases (also ``Canvas``/white in light mode) safely alone: the
        scroll canvases use ``BG``, never ``CARD``.
        """
        def walk(widget: Any) -> None:
            for child in widget.winfo_children():
                klass = child.winfo_class()
                try:
                    if klass == "Frame":
                        if str(child.cget("background")) == previous["CARD"]:
                            child.configure(background=CARD)
                        if str(child.cget("highlightbackground")) == previous["BORDER"]:
                            child.configure(highlightbackground=BORDER)
                    elif klass == "Canvas":
                        if str(child.cget("background")) == previous["BG"]:
                            child.configure(background=BG)
                    elif klass == "Text":
                        if str(child.cget("background")) == previous["CARD"]:
                            child.configure(background=CARD, foreground=INK)
                except tk.TclError:  # pragma: no cover - exotic widgets
                    pass
                walk(child)

        walk(self)

    def _collect_session(self) -> dict[str, Any]:
        """Snapshot every input on every tab into a session payload."""
        variables: dict[str, Any] = {}
        for name, value in sorted(vars(self).items()):
            if name.startswith("_book") or name == "_theme_var":
                continue  # transient bookkeeping, not user input
            if isinstance(value, tk.Variable):
                with contextlib.suppress(tk.TclError):
                    variables[name] = value.get()
            elif isinstance(value, ttk.Entry):  # covers Spinbox/Combobox
                variables[name] = value.get()
        families = {family: bool(var.get())
                    for family, var in self._feat_family_vars.items()}
        texts = {
            "pasted": self._feat_pasted_text.get("1.0", "end-1c"),
            "corpus": self._corpus_text.get("1.0", "end-1c"),
        }
        try:
            tab = self._book.index(self._book.select())
        except tk.TclError:  # pragma: no cover - no tab selected yet
            tab = 0
        return dump_session(theme=self._theme, geometry=self.geometry(),
                            tab=tab, variables=variables, families=families,
                            texts=texts)

    def _write_session_file(self) -> bool:
        """Persist the current session; ``True`` when the file landed."""
        try:
            path = session_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(self._collect_session(), handle, indent=2)
            return True
        except (OSError, TypeError, ValueError):
            return False

    def _save_session_now(self) -> None:
        """File ▸ Save session now -- write immediately and report."""
        if not self._persist_session:
            return
        if self._write_session_file():
            self._job_label.configure(
                text=f"session saved · {session_path()}")
        else:
            self._job_label.configure(text="session NOT saved (path unwritable)")

    def _clear_saved_session(self) -> None:
        """File ▸ Clear saved session -- forget the stored inputs."""
        if not self._persist_session:
            return
        removed = False
        with contextlib.suppress(OSError):
            session_path().unlink()
            removed = True
        self._session = None
        if messagebox is not None:
            messagebox.showinfo(
                "Session cleared",
                "The saved session was removed.\n"
                "Fresh defaults return on the next launch." if removed else
                "No saved session file existed.")

    def _export_session(self) -> None:
        """File ▸ Export session -- write a shareable session file."""
        if filedialog is None:
            return
        target = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile="pharmaml-playground-session.json",
            filetypes=[("Playground session", "*.json"), ("All files", "*.*")],
            title="Export playground session",
        )
        if not target:
            return
        if export_session(self._collect_session(), target):
            self._job_label.configure(text=f"session exported · {target}")
        else:
            self._job_label.configure(
                text="session NOT exported (path unwritable)")

    def _import_session(self) -> None:
        """File ▸ Import session -- apply a shareable session file."""
        if filedialog is None:
            return
        source = filedialog.askopenfilename(
            filetypes=[("Playground session", "*.json"), ("All files", "*.*")],
            title="Import playground session",
        )
        if not source:
            return
        session = load_session(read_json_file(source))
        if session is None:
            if messagebox is not None:
                messagebox.showerror(
                    "Invalid session file",
                    "That file is not a playground session (format marker "
                    "missing or unreadable).")
            return
        self._session = session
        self._restore_session()
        if session["theme"] in _THEMES and session["theme"] != self._theme:
            self._theme = session["theme"]
            if self._theme_var is not None:
                with contextlib.suppress(tk.TclError):
                    self._theme_var.set(self._theme)
            self._set_theme(self._resolve_auto_theme()
                            if self._theme == "auto" else self._theme)
        self._job_label.configure(text=f"session imported · {source}")

    def _reset_current_tab(self) -> None:
        """File ▸ Reset current tab to defaults (Ctrl+Shift+R, 100x-54)."""
        try:
            index = self._book.index(self._book.select())
        except tk.TclError:  # pragma: no cover - no tab selected
            return
        self._reset_tab(index)

    def _reset_tab(self, index: int) -> None:
        """Reset ONE tab's inputs to launch defaults (per-tab footer
        button, 100x-55; the File-menu command delegates here).

        Only the named tab's inputs are reset -- every other tab keeps
        its live values -- using the defaults snapshot taken right after
        the tabs were built (before any session restore).
        """
        names = self._tab_input_owners.get(index, set())
        defaults = self._session_defaults
        text_widgets = {"pasted": self._feat_pasted_text,
                        "corpus": self._corpus_text}
        applied = 0
        for name in sorted(names):
            if name.startswith("family:"):
                family = name[len("family:"):]
                var = self._feat_family_vars.get(family)
                if family in defaults["families"] and var is not None:
                    with contextlib.suppress(tk.TclError):
                        var.set(bool(defaults["families"][family]))
                        applied += 1
                continue
            if name.startswith("text:"):
                key = name[len("text:"):]
                widget = text_widgets.get(key)
                if key in defaults["texts"] and widget is not None:
                    widget.delete("1.0", "end")
                    content = str(defaults["texts"][key])
                    if content:
                        widget.insert("1.0", content)
                    applied += 1
                continue
            if name not in defaults["variables"]:
                continue
            widget = getattr(self, name, None)
            value = defaults["variables"][name]
            if isinstance(widget, tk.Variable):
                with contextlib.suppress(tk.TclError):
                    widget.set(value)
                    applied += 1
            elif isinstance(widget, ttk.Entry):
                setter = getattr(widget, "set", None)
                if callable(setter):  # Spinbox / Combobox
                    setter(str(value))
                else:
                    widget.delete(0, "end")
                    if str(value):
                        widget.insert(0, str(value))
                applied += 1
        label = self._book.tab(index, "text")
        self._job_label.configure(
            text=f"{label.lower()} tab reset · {applied} inputs back to defaults")

    def _copy_tab_inputs(self, index: int) -> None:
        """Copy ONE tab's live inputs as a share-lite JSON snippet.

        The same ownership map the per-tab reset uses decides what
        belongs to the tab; values are read LIVE (not the launch
        defaults) and coerced the session's way, so the snippet is a
        plain ``{tab, variables, families, texts}`` dict -- paste it
        into an issue report, a script, or a message.
        """
        names = self._tab_input_owners.get(index, set())
        text_widgets = {"pasted": self._feat_pasted_text,
                        "corpus": self._corpus_text}
        variables: dict[str, Any] = {}
        families: dict[str, bool] = {}
        texts: dict[str, str] = {}
        for name in sorted(names):
            if name.startswith("family:"):
                family = name[len("family:"):]
                var = self._feat_family_vars.get(family)
                if var is not None:
                    with contextlib.suppress(tk.TclError):
                        families[family] = bool(var.get())
                continue
            if name.startswith("text:"):
                key = name[len("text:"):]
                widget = text_widgets.get(key)
                if widget is not None:
                    texts[key] = widget.get("1.0", "end-1c")
                continue
            widget = getattr(self, name, None)
            if isinstance(widget, tk.Variable):
                with contextlib.suppress(tk.TclError):
                    variables[name] = widget.get()
            elif isinstance(widget, ttk.Entry):
                variables[name] = widget.get()
        label = self._book.tab(index, "text").strip()
        payload = {
            "tab": label,
            "variables": variables,
            "families": families,
            "texts": texts,
        }
        items = len(variables) + len(families) + len(texts)
        self._copy(json.dumps(payload, indent=2, sort_keys=True))
        self._job_label.configure(
            text=f"{label.lower()} inputs copied as JSON · {items} items")

    def _walk_widgets(self, widget: Any) -> Iterator[Any]:
        """Yield ``widget`` and every descendant (the GUI-tree walk)."""
        yield widget
        for child in widget.winfo_children():
            yield from self._walk_widgets(child)

    def _tab_input_names(self) -> dict[int, set[str]]:
        """Map each notebook tab to the session-input names it owns.

        Ownership is decided by the widget tree: every entry / spinbox /
        combobox / checkbutton / radiobutton living inside a tab's frame
        contributes its linked variable (Tcl name -> app attribute), an
        entry stored directly as an attribute contributes that name, the
        feature-family checkboxes contribute ``family:<name>`` keys and
        the two pasted corpora contribute ``text:<key>`` keys.  The theme
        radio and notebook bookkeeping are excluded (not user inputs).
        """
        var_by_tcl_name: dict[str, str] = {}
        entry_by_id: dict[int, str] = {}
        for name, value in vars(self).items():
            if name.startswith("_book") or name == "_theme_var":
                continue
            if isinstance(value, tk.Variable):
                var_by_tcl_name[str(value)] = name
            elif isinstance(value, ttk.Entry):
                entry_by_id[id(value)] = name
        family_by_tcl_name = {str(var): f"family:{family}"
                               for family, var in self._feat_family_vars.items()}
        text_by_id = {id(self._feat_pasted_text): "text:pasted",
                      id(self._corpus_text): "text:corpus"}
        owners: dict[int, set[str]] = {}
        with contextlib.suppress(tk.TclError):
            for index, tab_id in enumerate(self._book.tabs()):
                names: set[str] = set()
                for widget in self._walk_widgets(self._book.nametowidget(tab_id)):
                    if id(widget) in text_by_id:
                        names.add(text_by_id[id(widget)])
                        continue
                    if isinstance(widget, ttk.Entry):
                        linked = str(widget.cget("textvariable"))
                        if linked and linked in var_by_tcl_name:
                            names.add(var_by_tcl_name[linked])
                        elif id(widget) in entry_by_id:
                            names.add(entry_by_id[id(widget)])
                    elif isinstance(widget, (ttk.Checkbutton, ttk.Radiobutton,
                                             tk.Checkbutton, tk.Radiobutton)):
                        linked = str(widget.cget("variable"))
                        if linked in var_by_tcl_name:
                            names.add(var_by_tcl_name[linked])
                        elif linked in family_by_tcl_name:
                            names.add(family_by_tcl_name[linked])
                owners[index] = names
        return owners

    def _restore_session(self) -> bool:
        """Re-apply the saved session onto the freshly built tabs.

        Returns ``True`` when a geometry was restored (so ``__init__``
        can skip re-centering the window).  Restore never raises: a
        payload naming a widget that no longer exists (older session,
        renamed attribute) is simply skipped.
        """
        if not self._session:
            return False
        session = self._session
        for name, value in session["variables"].items():
            widget = getattr(self, name, None)
            if isinstance(widget, tk.Variable):
                with contextlib.suppress(tk.TclError):
                    widget.set(value)
            elif isinstance(widget, ttk.Entry):
                setter = getattr(widget, "set", None)
                if callable(setter):  # Spinbox / Combobox
                    setter(str(value))
                else:
                    widget.delete(0, "end")
                    widget.insert(0, str(value))
        for family, checked in session["families"].items():
            var = self._feat_family_vars.get(family)
            if var is not None:
                with contextlib.suppress(tk.TclError):
                    var.set(bool(checked))
        text_widgets = {"pasted": self._feat_pasted_text,
                        "corpus": self._corpus_text}
        for key, content in session["texts"].items():
            widget = text_widgets.get(key)
            if widget is not None:
                widget.delete("1.0", "end")
                widget.insert("1.0", content)
        tab = session["tab"]
        if 0 <= tab < self._book.index("end"):
            self._book.select(tab)
        restored_geometry = False
        parsed = parse_geometry(session["geometry"])
        if parsed is not None:
            width, height, x, y = parsed
            width = min(width, self.winfo_screenwidth())
            height = min(height, self.winfo_screenheight())
            on_screen = (0 <= x < self.winfo_screenwidth()
                         and 0 <= y < self.winfo_screenheight())
            spec = f"{width}x{height}" + (f"+{x}+{y}" if on_screen else "")
            with contextlib.suppress(tk.TclError):
                self.geometry(spec)
                restored_geometry = True
        if session["variables"] or session["texts"]:
            self._job_label.configure(
                text="session restored · File ▸ Clear saved session resets")
        return restored_geometry

    def _on_close(self) -> None:
        """Window close: save the session (unless disabled), then exit."""
        if self._persist_session:
            self._write_session_file()
        self.destroy()

    # -- menu actions ---------------------------------------------------- #
    def _menu_open_csv(self) -> None:
        self._book.select(self._ml_page)
        self._ml_open_csv()

    def _menu_save_figure(self) -> None:
        if self._last_figure is None:
            messagebox.showinfo("No figure yet", "Run a calculation that draws one first.")
            return
        if not self._require_pharmaml():
            return
        path = filedialog.asksaveasfilename(
            title="Save figure",
            defaultextension=".png",
            filetypes=[
                ("PNG figure", "*.png"),
                ("SVG figure", "*.svg"),
                ("PDF figure", "*.pdf"),
            ],
        )
        if not path:
            return
        try:
            from pharmaml.viz import save_figure

            saved = save_figure(self._last_figure, path)
            messagebox.showinfo("Figure saved", f"Written to\n{saved}")
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not save the figure", exc)

    def _show_about(self) -> None:
        pharmaml_line = (
            f"pharmaml {self.pharmaml.__version__}" if self.pharmaml is not None
            else "pharmaml: NOT LOADED (run from the toolkit root)"
        )
        messagebox.showinfo(
            "About PharmaML Playground",
            f"PharmaML Playground {__version__}\n{pharmaml_line}\n\n"
            "A single-file graphical front end for the pharmaml toolkit:\n"
            "chemistry · PK/PD · cell lines · docking · QSAR machine learning · "
            "generative design.\n\n"
            "Calculations run in pharmaml itself -- the playground only\n"
            "collects your inputs and renders the results.",
        )

    def _show_dependency_report(self) -> None:
        lines = [f"{'module':<14} status", "-" * 28]
        for module, label, _purpose, _pip in _DEPENDENCIES:
            lines.append(f"{label:<14} {'installed' if self.deps.get(module) else 'missing'}")
        lines.append(f"{'tkinter':<14} {'installed' if tk is not None else 'missing'}")
        report = "\n".join(lines)
        print(report)
        messagebox.showinfo("Dependency report", report)

    # ------------------------------------------------------------------ #
    # Home tab
    # ------------------------------------------------------------------ #
    def _build_home_tab(self) -> None:
        page = self._new_page("Home", reset=False)  # no inputs to reset
        hero = make_card(
            page.grid, "PharmaML Playground", span=2, row=0, column=0,
            subtitle="One window for the whole toolkit -- chemistry, PK/PD dosing, "
                     "cell-line biology, molecular docking, QSAR modelling and "
                     "the generative GA, without touching a terminal.",
        )
        version_line = (
            f"playground {__version__} · pharmaml {self.pharmaml.__version__}"
            if self.pharmaml is not None
            else f"playground {__version__} · pharmaml NOT LOADED"
        )
        ttk.Label(hero, text=version_line, style="CardMuted.TLabel").grid(
            row=0, column=0, sticky="w"
        )

        status = make_card(page.grid, "Dependency status", span=2, row=1, column=0,
                           subtitle="Each tab degrades politely and tells you the pip line it wants.")
        grid = ttk.Frame(status, style="Card.TFrame")
        grid_append(status, grid, sticky="ew")
        grid.columnconfigure(2, weight=1)
        rows = [(label, module, purpose, pip) for module, label, purpose, pip
                in _DEPENDENCIES]
        rows.append(("tkinter", "tkinter", "the GUI itself (stdlib)", "python3-tk"))
        for label, module, purpose, pip in rows:
            row = grid.grid_size()[1]
            ok = self.deps.get(module, tk is not None)
            dot = tk.Label(grid, text="●", background=CARD,
                           foreground=GOOD if ok else BAD,
                           font=("TkDefaultFont", 12))
            dot.grid(row=row, column=0, sticky="w", padx=(0, 8))
            ttk.Label(grid, text=label, style="Field.TLabel").grid(
                row=row, column=1, sticky="w", padx=(0, 10))
            ttk.Label(grid, text=purpose, style="CardMuted.TLabel").grid(
                row=row, column=2, sticky="w")
            state = "installed" if ok else f"missing → pip install {pip}"
            ttk.Label(grid, text=state, style="CardMuted.TLabel").grid(
                row=row, column=3, sticky="e")
        missing = [pip for module, _l, _p, pip in _DEPENDENCIES
                   if not self.deps.get(module)]
        if self.pharmaml is None:
            missing.append("-e .")
        if missing:
            grid_append(status, ttk.Label(status, style="CardMuted.TLabel",
                          justify="left", text=(
                "To light up every tab, from the toolkit root:\n"
                "    pip install -e . numpy pandas scikit-learn scipy matplotlib rdkit"
            )), pady=(8, 0))
        else:
            grid_append(status, ttk.Label(status, style="CardMuted.TLabel",
                          text="Everything is available -- every tab is fully armed."
                          ), pady=(8, 0))

        guide = make_card(page.grid, "What lives where", span=2, row=2, column=0)
        entries = [
            (
                "Chemistry",
                ("IC50/Ki → pIC50/pKi, molar mass, the 200+ descriptor table for "
                 "any SMILES, Tanimoto similarity, and the feature factory: "
                 "10,000+ descriptor/fingerprint columns ranked to a top-K."),
                "RDKit + pandas",
            ),
            (
                "PK / PD",
                ("Half-life, accumulation, IV bolus decay and Emax curves, "
                 "Cockcroft–Gault, loading/maintenance and pediatric dosing."),
                "numpy",
            ),
            (
                "Cell Lines",
                ("Drug-response screens: audit, the sensitivity matrix, sensitive "
                 "vs resistant calls and expression biomarker discovery."),
                "pandas + scipy",
            ),
            (
                "Docking",
                ("Ligand library and receptor preparation into PDBQT, the grid "
                 "box with GPF/DPF configs, DLG / Vina result parsing and "
                 "batch campaign runs."),
                "RDKit + pandas",
            ),
            (
                "Data & ML",
                ("Open a CSV (or a bundled example), train a model, read the metrics, "
                 "browse the cross-validation leaderboard."),
                "pandas + sklearn",
            ),
            (
                "Generative",
                ("Evolve molecules against QED / LogP objectives; adaptive runs show "
                 "the operator-mix learning curve."),
                "RDKit",
            ),
        ]
        for name, blurb, needs in entries:
            row = ttk.Frame(guide, style="Card.TFrame")
            grid_append(guide, row, sticky="ew", pady=2)
            ttk.Label(row, text=name, style="Field.TLabel",
                      font=("TkDefaultFont", 10, "bold")).pack(side="left")
            ttk.Label(row, text=f" — {blurb}", style="CardMuted.TLabel",
                      wraplength=560, justify="left").pack(side="left", padx=(8, 0))
            ttk.Label(row, text=f"[{needs}]", style="CardMuted.TLabel").pack(
                side="right")

        if self.pharmaml is None:
            broken = make_card(page.grid, "pharmaml could not be imported",
                               span=2, row=3, column=0)
            grid_append(broken, ttk.Label(broken, text=self.pharmaml_error,
                                           style="CardMuted.TLabel",
                                           wraplength=620))
            grid_append(broken, ttk.Label(broken, text=_PHARMAML_HINT,
                                          style="CardMuted.TLabel",
                                          justify="left"), pady=(6, 0))

    # ------------------------------------------------------------------ #
    # Chemistry tab
    # ------------------------------------------------------------------ #
    def _build_chemistry_tab(self) -> None:
        page = self._new_page("Chemistry")

        conv = make_card(
            page.grid, "Affinity conversion", row=0, column=0,
            subtitle="IC50/Ki → pIC50/pKi. Censored entries like '<100' are handled.",
        )
        self._aff_mode = ttk.Combobox(conv, state="readonly", width=22,
                                      values=("IC50 → pIC50", "Ki → pKi"))
        self._aff_mode.set("IC50 → pIC50")
        form_row(conv, "Endpoint", self._aff_mode)
        self._aff_value = ttk.Entry(conv, width=22)
        self._aff_value.insert(0, "100")
        form_row(conv, "Value", self._aff_value)
        self._aff_unit = ttk.Combobox(conv, state="readonly", width=22,
                                      values=("nM", "uM", "µM", "mM", "M", "pM"))
        self._aff_unit.set("nM")
        form_row(conv, "Unit", self._aff_unit)
        button = self._action_button(conv, "Convert", self._convert_affinity,
                                     tooltip="-log10 of the value converted to molar units.")
        button.grid(row=conv.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._aff_result = ttk.Label(conv, text="—", style="Result.TLabel",
                                     font=self._mono_result)
        self._aff_result.grid(row=button.grid_info()["row"], column=1,
                              sticky="w", padx=(10, 0))
        bind_enter(self._aff_value, button)
        copy_btn = self._action_button(conv, "Copy", lambda: self._copy(
            self._aff_result.cget("text")))
        copy_btn.grid(row=conv.grid_size()[1], column=0, sticky="w")

        mass = make_card(page.grid, "Molar mass", row=0, column=1,
                         subtitle="Formula → g/mol (works with no heavy dependencies installed).")
        self._formula = ttk.Entry(mass, width=22)
        self._formula.insert(0, "C8H10N4O2")
        form_row(mass, "Formula", self._formula)
        button = self._action_button(mass, "Weigh", self._molar_mass,
                                     tooltip="Supports hydrates (MgSO4.7H2O) and groups like Ca(OH)2.")
        button.grid(row=mass.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._mass_result = ttk.Label(mass, text="—", style="Result.TLabel",
                                      font=self._mono_result)
        self._mass_result.grid(row=button.grid_info()["row"], column=1,
                               sticky="w", padx=(10, 0))
        bind_enter(self._formula, button)

        desc = make_card(page.grid, "SMILES inspector", row=1, column=0,
                         subtitle="Every RDKit 2D descriptor for one molecule (type to filter).")
        self._smiles_entry = ttk.Entry(desc, width=34)
        self._smiles_entry.insert(0, "CC(=O)Oc1ccccc1C(=O)O")
        form_row(desc, "SMILES", self._smiles_entry)
        button = self._action_button(desc, "Analyse", self._analyse_smiles, primary=True,
                                     tooltip="Computes the ~200 RDKit descriptors in a worker thread.")
        button.grid(row=desc.grid_size()[1], column=0, sticky="w", pady=(4, 0))
        self._smiles_summary = ttk.Label(desc, text="—", style="CardMuted.TLabel",
                                         wraplength=420, justify="left")
        self._smiles_summary.grid(row=button.grid_info()["row"], column=1,
                                  sticky="w", padx=(10, 0))
        bind_enter(self._smiles_entry, button)
        self._desc_search = ttk.Entry(desc, width=24)
        self._desc_search.insert(0, "")
        self._desc_search.bind("<KeyRelease>", self._filter_descriptors)
        form_row(desc, "Filter", self._desc_search)
        tree_frame = ttk.Frame(desc, style="Card.TFrame")
        tree_frame.grid(row=desc.grid_size()[1], column=0, columnspan=2,
                        sticky="nsew", pady=(4, 0))
        desc.rowconfigure(tree_frame.grid_info()["row"], weight=1)
        self._desc_tree = ttk.Treeview(tree_frame, columns=("descriptor", "value"),
                                       show="headings", height=9)
        self._desc_tree.heading("descriptor", text="descriptor")
        self._desc_tree.heading("value", text="value")
        self._desc_tree.column("descriptor", width=200, stretch=True)
        self._desc_tree.column("value", width=130, stretch=False)
        desc_bar = ttk.Scrollbar(tree_frame, orient="vertical",
                                 command=self._desc_tree.yview)
        self._desc_tree.configure(yscrollcommand=desc_bar.set)
        self._desc_tree.pack(side="left", fill="both", expand=True)
        desc_bar.pack(side="right", fill="y")
        save_btn = self._action_button(desc, "Save descriptors as CSV…",
                                       self._save_descriptors)
        save_btn.grid(row=desc.grid_size()[1], column=0, columnspan=2, sticky="w",
                      pady=(6, 0))

        tani = make_card(page.grid, "Tanimoto similarity", row=1, column=1,
                         subtitle="Morgan fingerprint overlap between two molecules.")
        self._tani_a = ttk.Entry(tani, width=34)
        self._tani_a.insert(0, "CC(=O)Oc1ccccc1C(=O)O")
        form_row(tani, "SMILES A", self._tani_a)
        self._tani_b = ttk.Entry(tani, width=34)
        self._tani_b.insert(0, "CC(=O)Nc1ccc(O)cc1")
        form_row(tani, "SMILES B", self._tani_b)
        button = self._action_button(tani, "Compare", self._compare_similarity,
                                     tooltip="1.0 = identical fingerprint, 0.0 = nothing shared.")
        button.grid(row=tani.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._tani_result = ttk.Label(tani, text="—", style="Result.TLabel",
                                      font=self._mono_result)
        self._tani_result.grid(row=button.grid_info()["row"], column=1, sticky="w",
                               padx=(10, 0))
        self._tani_verdict = ttk.Label(tani, text="", style="CardMuted.TLabel",
                                       wraplength=300, justify="left")
        self._tani_verdict.grid(row=tani.grid_size()[1], column=0, columnspan=2,
                                sticky="w", pady=(6, 0))
        bind_enter(self._tani_b, button)

        # -- feature factory: the rar's own ML pipeline ------------------ #
        factory = make_card(
            page.grid, "Feature factory", row=2, column=0, span=2,
            subtitle="The rar's own ML pipeline as a card: 7 descriptor / "
                     "fingerprint families (~10,800 columns per molecule), "
                     "ranked and cut to the top K you keep.",
        )
        source_row = ttk.Frame(factory, style="Card.TFrame")
        grid_append(factory, source_row, sticky="w")
        ttk.Label(source_row, text="Source", style="Field.TLabel").pack(side="left")
        self._feat_source_var = tk.StringVar(value="Pasted SMILES")
        csv_radio = ttk.Radiobutton(
            source_row, text="CSV file", value="CSV file",
            variable=self._feat_source_var, command=self._feat_refresh_source,
        )
        csv_radio.pack(side="left", padx=(12, 0))
        pasted_radio = ttk.Radiobutton(
            source_row, text="Pasted SMILES", value="Pasted SMILES",
            variable=self._feat_source_var, command=self._feat_refresh_source,
        )
        pasted_radio.pack(side="left", padx=(8, 0))

        self._feat_csv_box = ttk.Frame(factory, style="Card.TFrame")
        grid_append(factory, self._feat_csv_box, sticky="ew", pady=(6, 0))
        csv_buttons = ttk.Frame(self._feat_csv_box, style="Card.TFrame")
        csv_buttons.grid(row=0, column=0, columnspan=2, sticky="w")
        csv_open_btn = self._action_button(csv_buttons, "Open CSV…",
                                           self._feat_open_csv)
        csv_open_btn.pack(side="left")
        csv_example_btn = self._action_button(
            csv_buttons, "Load example", self._feat_load_example,
            tooltip="examples/data/library_with_descriptors.csv -- 77 drugs "
                    "with SMILES, 8 numeric descriptor columns and an activity "
                    "label.",
        )
        csv_example_btn.pack(side="left", padx=(10, 0))
        self._feat_source_info = ttk.Label(
            self._feat_csv_box, text="no CSV loaded", style="CardMuted.TLabel",
            wraplength=520, justify="left",
        )
        self._feat_source_info.grid(row=1, column=0, columnspan=2, sticky="w",
                                    pady=(4, 0))
        self._feat_target_combo = ttk.Combobox(
            self._feat_csv_box, state="readonly", width=30,
            values=(_FEAT_NO_TARGET,),
        )
        self._feat_target_combo.set(_FEAT_NO_TARGET)
        form_row(self._feat_csv_box, "Target column", self._feat_target_combo)

        self._feat_pasted_box = ttk.Frame(factory, style="Card.TFrame")
        grid_append(factory, self._feat_pasted_box, sticky="ew", pady=(6, 0))
        self._feat_pasted_text = _grid_text(self._feat_pasted_box, height=4,
                                            mono=self._mono)
        self._feat_pasted_text.insert("1.0", "\n".join(_FEAT_DEFAULT_SMILES))
        grid_append(self._feat_pasted_box, self._feat_pasted_text, sticky="ew")
        grid_append(self._feat_pasted_box, ttk.Label(
            self._feat_pasted_box, style="CardMuted.TLabel",
            text="one SMILES per line; comments with # are ignored",
        ), pady=(2, 0))

        families_box = ttk.Frame(factory, style="Card.TFrame")
        grid_append(factory, families_box, sticky="ew", pady=(6, 0))
        self._feat_family_vars: dict[str, Any] = {}
        for index, (family, label) in enumerate(_FEATURE_FAMILY_LABELS):
            family_var = tk.BooleanVar(value=True)
            self._feat_family_vars[family] = family_var
            family_check = ttk.Checkbutton(
                families_box, text=label, variable=family_var,
                command=self._feat_update_width,
            )
            family_check.grid(row=index // 2, column=index % 2, sticky="w",
                              padx=(0, 24), pady=2)

        self._feat_k_entry = ttk.Entry(factory, width=10)
        self._feat_k_entry.insert(0, "1000")  # the rar pipeline's own default
        Tooltip(self._feat_k_entry, _FEAT_K_PRESETS)
        form_row(factory, "Features to keep (K)", self._feat_k_entry)
        self._feat_ranker_combo = ttk.Combobox(factory, state="readonly", width=28,
                                               values=_FEAT_RANKERS)
        self._feat_ranker_combo.set(_FEAT_RANKERS[0])
        form_row(factory, "Ranker", self._feat_ranker_combo)
        self._feat_task_combo = ttk.Combobox(factory, state="readonly", width=28,
                                             values=_FEAT_TASKS)
        self._feat_task_combo.set(_FEAT_TASKS[0])
        form_row(factory, "Task", self._feat_task_combo)
        self._feat_descriptor_count = _FEATURE_DESCRIPTORS_FALLBACK
        try:
            from rdkit.Chem import rdMolDescriptors

            self._feat_descriptor_count = len(
                rdMolDescriptors.Properties.GetAvailableProperties()
            )
        except Exception:  # noqa: BLE001 - estimate only; the run gates RDKit
            self._feat_descriptor_count = _FEATURE_DESCRIPTORS_FALLBACK
        self._feat_width_label = ttk.Label(factory, text="", style="CardMuted.TLabel")
        grid_append(factory, self._feat_width_label, sticky="w")

        feat_btn = self._action_button(
            factory, "Compute & select features", self._feat_compute, primary=True,
            tooltip="Computes the ticked families in a worker thread, ranks "
                    "every column, keeps the per-family median survivors, "
                    "then the top K.",
        )
        feat_btn.grid(row=factory.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._feat_summary = ttk.Label(
            factory, text="—", style="CardMuted.TLabel", wraplength=560,
            justify="left",
        )
        self._feat_summary.grid(row=feat_btn.grid_info()["row"], column=1,
                                sticky="w", padx=(10, 0))
        self._feat_family_label = ttk.Label(factory, text="",
                                            style="CardMuted.TLabel",
                                            wraplength=620, justify="left")
        grid_append(factory, self._feat_family_label, sticky="w")
        feat_tree_frame = ttk.Frame(factory, style="Card.TFrame")
        grid_append(factory, feat_tree_frame, sticky="nsew", pady=(6, 0),
                    weight=True)
        self._feat_tree = ttk.Treeview(
            feat_tree_frame, columns=("rank", "feature", "family", "importance"),
            show="headings", height=10,
        )
        for column, width in (
            ("rank", 56), ("feature", 210), ("family", 140), ("importance", 120),
        ):
            self._feat_tree.heading(column, text=column)
            self._feat_tree.column(column, width=width, minwidth=48,
                                   stretch=(column == "feature"))
        feat_bar = ttk.Scrollbar(feat_tree_frame, orient="vertical",
                                 command=self._feat_tree.yview)
        self._feat_tree.configure(yscrollcommand=feat_bar.set)
        self._feat_tree.pack(side="left", fill="both", expand=True)
        feat_bar.pack(side="right", fill="y")
        self._feat_more_label = ttk.Label(factory, text="", style="CardMuted.TLabel")
        grid_append(factory, self._feat_more_label, sticky="w")
        feat_export_buttons = ttk.Frame(factory, style="Card.TFrame")
        grid_append(factory, feat_export_buttons, sticky="w", pady=(2, 0))
        feat_csv_btn = self._action_button(
            feat_export_buttons, "Save selected features CSV…", self._feat_save_csv,
            tooltip="The rar-style output: SMILES, the target column (if any) "
                    "and the K selected features in rank order.",
        )
        feat_csv_btn.pack(side="left")
        feat_json_btn = self._action_button(
            feat_export_buttons, "Save metadata JSON…", self._feat_save_json,
            tooltip="selection_metadata: selected features, family counts, "
                    "fingerprint sizes, K and method.",
        )
        feat_json_btn.pack(side="left", padx=(10, 0))
        self._feat_update_width()
        self._feat_refresh_source()

    def _convert_affinity(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas", "conversion"):
            return
        from pharmaml.chem import ic50_to_pic50, ki_to_pki

        mode = self._aff_mode.get()
        func = ic50_to_pic50 if mode.startswith("IC50") else ki_to_pki
        try:
            value = func(self._aff_value.get().strip(), unit=self._aff_unit.get())
        except Exception as exc:  # noqa: BLE001
            self._show_error("Conversion failed", exc)
            return
        if value is None:
            self._aff_result.configure(text="n/a (unparseable entry)")
            return
        self._aff_result.configure(text=f"{value:.3f}")

    def _molar_mass(self) -> None:
        if not self._require_pharmaml():
            return
        from pharmaml.pharma.units import molar_mass

        try:
            value = molar_mass(self._formula.get().strip())
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not parse the formula", exc)
            return
        self._mass_result.configure(text=f"{value:,.3f} g/mol")

    def _analyse_smiles(self) -> None:
        if not self._require_pharmaml() or not self._require("rdkit", "descriptors"):
            return
        smiles = self._smiles_entry.get().strip()
        if not smiles:
            messagebox.showwarning("Empty SMILES", "Type a SMILES string first.")
            return

        def work() -> Any:
            from pharmaml.chem import compute_descriptors

            return compute_descriptors([smiles], progress=False)

        self._start_job(f"descriptors for {smiles[:28]}…", work,
                        self._render_descriptors)

    def _render_descriptors(self, frame: Any) -> None:
        self._desc_frame = frame
        self._desc_rows = []
        self._desc_tree.delete(*self._desc_tree.get_children())
        items = sorted(zip(frame.columns, frame.iloc[0].tolist()))
        for name, value in items:
            text = _cell_text(value, limit=40)
            item = self._desc_tree.insert("", "end", values=(name, text))
            self._desc_rows.append((item, str(name), text))
        highlights = {
            "MolWt": "MW", "MolLogP": "LogP", "TPSA": "TPSA",
            "NumHBD": "H-bond donors", "NumHBA": "H-bond acceptors",
            "NumRotatableBonds": "rotatable bonds", "RingCount": "rings",
        }
        found = [f"{short} {format_number(frame.iloc[0][key])}"
                 for key, short in highlights.items() if key in frame.columns]
        self._smiles_summary.configure(
            text=("  ·  ".join(found) if found else f"{len(items)} descriptors")
        )
        self._filter_descriptors()

    def _filter_descriptors(self, _event: Any = None) -> None:
        query = self._desc_search.get().strip().lower()
        for item, name, _value in self._desc_rows:
            if query and query not in name.lower():
                self._desc_tree.detach(item)
            else:
                self._desc_tree.move(item, "", "end")

    def _save_descriptors(self) -> None:
        if self._desc_frame is None:
            messagebox.showinfo("Nothing to save", "Analyse a SMILES string first.")
            return
        if not self._require_pharmaml():
            return
        path = filedialog.asksaveasfilename(
            title="Save descriptors", defaultextension=".csv",
            filetypes=[("CSV table", "*.csv")],
        )
        if not path:
            return
        try:
            from pharmaml.utils.io import write_table

            written = write_table(self._desc_frame, path)
            messagebox.showinfo("Saved", f"Descriptor table written to\n{written}")
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not save", exc)

    def _compare_similarity(self) -> None:
        if not self._require_pharmaml() or not self._require("rdkit", "fingerprints"):
            return
        from pharmaml.chem import compute_fingerprints, tanimoto_from_bitarrays

        first = self._tani_a.get().strip()
        second = self._tani_b.get().strip()
        if not first or not second:
            messagebox.showwarning("Missing SMILES", "Both molecules are needed.")
            return
        try:
            prints, valid = compute_fingerprints([first, second])
        except Exception as exc:  # noqa: BLE001
            self._show_error("Fingerprint failed", exc)
            return
        if len(valid) < 2:
            self._tani_result.configure(text="invalid SMILES")
            self._tani_verdict.configure(
                text="One of the two strings is not valid RDKit input."
            )
            return
        score = float(tanimoto_from_bitarrays(prints[0], prints[1]))
        if score >= 0.85:
            verdict = "near-identical fingerprints"
        elif score >= 0.6:
            verdict = "clearly related scaffolds"
        elif score >= 0.35:
            verdict = "loosely related"
        else:
            verdict = "essentially unrelated"
        self._tani_result.configure(text=f"{score:.3f}")
        self._tani_verdict.configure(text=f"{verdict} (1.0 = identical, 0.0 = nothing shared)")

    # -- feature factory card -------------------------------------------- #
    def _feat_refresh_source(self) -> None:
        """Show the CSV row or the pasted-SMILES box per the source radios."""
        if self._feat_source_var.get() == "CSV file":
            self._feat_csv_box.grid()
            self._feat_pasted_box.grid_remove()
        else:
            self._feat_csv_box.grid_remove()
            self._feat_pasted_box.grid()

    def _feat_update_width(self) -> None:
        """Re-pin the estimated column count from the ticked families."""
        selected = [
            family for family, _label in _FEATURE_FAMILY_LABELS
            if self._feat_family_vars[family].get()
        ]
        if not selected:
            self._feat_width_label.configure(
                text="no families selected — tick at least one"
            )
            return
        width = feature_factory_width(selected, self._feat_descriptor_count)
        self._feat_width_label.configure(
            text=(f"{len(selected)} families selected · ~{width:,} columns "
                  "will be computed")
        )

    def _feat_open_csv(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas",
                                                             "CSV loading"):
            return
        path = filedialog.askopenfilename(
            title="Open a SMILES CSV",
            filetypes=[("SMILES table", "*.csv *.tsv"), ("All files", "*.*")],
        )
        if path:
            self._feat_load_path(Path(path))

    def _feat_load_example(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas",
                                                             "CSV loading"):
            return
        path = EXAMPLES_DIR / "library_with_descriptors.csv"
        if not path.is_file():
            messagebox.showwarning(
                "Missing example",
                "examples/data/library_with_descriptors.csv was not found.",
            )
            return
        self._feat_load_path(path)

    def _feat_load_path(self, path: Path) -> None:
        try:
            from pharmaml.utils.io import read_table

            frame = read_table(path)
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not read the CSV", exc)
            return
        if frame.empty:
            messagebox.showwarning("Empty CSV", f"{path.name} has no rows.")
            return
        smiles_column = detect_smiles_column(frame)
        if smiles_column is None:
            messagebox.showwarning(
                "No SMILES column",
                f"{path.name} has no SMILES column (looked for smiles / smile / "
                "canonical_smiles / ...).",
            )
            return
        import pandas as pd

        candidates = [
            str(column) for column in frame.columns
            if str(column) != smiles_column
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        self._feat_csv_frame = frame
        self._feat_csv_path = path
        self._feat_smiles_column = smiles_column
        self._feat_target_combo["values"] = (_FEAT_NO_TARGET, *candidates)
        self._feat_target_combo.set(_FEAT_NO_TARGET)
        self._feat_source_info.configure(
            text=(f"{len(frame):,} rows · SMILES column '{smiles_column}' · "
                  f"{len(candidates)} numeric target candidates — {path.name}")
        )

    def _feat_compute(self) -> None:
        if not self._require_pharmaml():
            return
        if not self._require("rdkit", "the feature factory") or not self._require(
                "pandas", "the feature factory"):
            return
        source = self._feat_source_var.get()
        target_frame: Any = None
        target_name: str | None = None
        if source == "CSV file":
            if self._feat_csv_frame is None or self._feat_smiles_column is None:
                messagebox.showwarning(
                    "No CSV loaded",
                    "Open a SMILES CSV first (or switch the source to Pasted "
                    "SMILES).",
                )
                return
            smiles = [
                str(value)
                for value in self._feat_csv_frame[self._feat_smiles_column].tolist()
            ]
            picked = self._feat_target_combo.get()
            if picked and picked != _FEAT_NO_TARGET:
                target_name = picked
                target_frame = self._feat_csv_frame[picked]
        else:
            smiles = parse_smiles_lines(self._feat_pasted_text.get("1.0", "end"))
        if not smiles:
            messagebox.showwarning(
                "No SMILES", "The selected source holds no SMILES to compute."
            )
            return
        families = [
            family for family, _label in _FEATURE_FAMILY_LABELS
            if self._feat_family_vars[family].get()
        ]
        if not families:
            messagebox.showwarning(
                "No families", "Tick at least one feature family."
            )
            return
        try:
            k = parse_int(self._feat_k_entry.get(), "features to keep (K)",
                          minimum=1, maximum=100000)
        except ValueError as exc:
            messagebox.showerror("Input", str(exc))
            return
        method = _FEAT_RANKER_METHODS.get(self._feat_ranker_combo.get(), "auto")
        task = _FEAT_TASK_METHODS.get(self._feat_task_combo.get(), "auto")
        # variance ranking needs numpy only; every model ranker needs sklearn
        if method != "variance" and not self._require("sklearn", "feature ranking"):
            return
        if method in ("random_forest", "lightgbm") and target_frame is None:
            messagebox.showwarning(
                "Ranker needs a target",
                f"{self._feat_ranker_combo.get()} ranking needs a target column "
                "-- pick one from the loaded CSV, or switch the ranker to "
                "Variance (no target needed).",
            )
            return

        def work() -> Any:
            from pharmaml.chem.features import (
                compute_feature_frame,
                rank_features,
                select_top_features,
            )

            frame, report = compute_feature_frame(smiles, families=families)
            target_values: list[float] | None = None
            if target_frame is not None:
                # align against the frame's original-row index (dropped
                # molecules take their target row with them)
                target_values = [
                    float(value) for value in target_frame.iloc[list(frame.index)]
                ]
            ranked = rank_features(frame, target_values, method=method, task=task)
            # select_top_features demands the scores in frame column order
            scores = ranked.reindex(frame.columns)
            selection = select_top_features(frame, scores, k,
                                            median_prefilter=True, method=method)
            # k >= the candidate count returns every median survivor, so the
            # survivor count for the summary comes from the same public call
            survivors = len(
                select_top_features(frame, scores, len(frame.columns)).selected
            )
            return frame, report, selection, survivors, target_values

        def render(payload: Any) -> None:
            self._feat_render(payload, k, target_name)

        label = f"computing {len(families)} families over {len(smiles)} molecules"
        self._start_job(label, work, render)

    def _feat_render(self, payload: Any, k: int,
                     target_name: str | None) -> None:
        frame, report, selection, survivors, target_values = payload
        self._feat_frame = frame
        self._feat_report = report
        self._feat_selection = selection
        self._feat_smiles = list(report.smiles)
        self._feat_target_series = target_values
        self._feat_target_name = target_name
        self._feat_k = k
        tree = self._feat_tree
        tree.delete(*tree.get_children())
        if report.n_valid == 0:
            self._feat_summary.configure(
                text=(f"{report.total_columns:,} features · 0 valid molecules "
                      f"({len(report.dropped)} dropped) — check the SMILES source"),
                style="Field.TLabel",
            )
            self._feat_family_label.configure(text="selected per family: (none)")
            self._feat_more_label.configure(text="")
            return
        from pharmaml.chem.features import family_of_column

        dropped_note = f" ({len(report.dropped)} dropped)" if report.dropped else ""
        method_label = str(selection.method).replace("_", " ")
        self._feat_summary.configure(
            text=(f"{report.total_columns:,} features · {report.n_valid} "
                  f"molecules{dropped_note} · kept top "
                  f"{len(selection.selected):,} of {survivors:,} "
                  f"median-survivors · {method_label}"),
            style="Field.TLabel",
        )
        family_line = " · ".join(
            f"{family} {count}"
            for family, count in selection.family_counts.items()
        )
        self._feat_family_label.configure(
            text=(f"selected per family: {family_line}" if family_line
                  else "selected per family: (none)")
        )
        shown = list(selection.selected[:300])
        for rank, column in enumerate(shown, start=1):
            tree.insert(
                "", "end",
                values=(rank, column, family_of_column(column),
                        format_number(float(selection.scores[column]))),
            )
        hidden = len(selection.selected) - len(shown)
        self._feat_more_label.configure(
            text=(f"+{hidden:,} more rows — save the CSV for the full list"
                  if hidden > 0 else "")
        )

    def _feat_save_csv(self) -> None:
        if self._feat_selection is None:
            messagebox.showinfo("Nothing to save",
                                "Compute a feature selection first.")
            return
        if not self._require_pharmaml():
            return
        path = filedialog.asksaveasfilename(
            title="Save the selected features", defaultextension=".csv",
            initialfile="selected_features.csv",
            filetypes=[("CSV table", "*.csv")],
        )
        if not path:
            return
        try:
            import pandas as pd

            from pharmaml.utils.io import write_table

            out = pd.DataFrame({"SMILES": list(self._feat_smiles)})
            if (self._feat_target_name is not None
                    and self._feat_target_series is not None):
                out[self._feat_target_name] = list(self._feat_target_series)
            for column in self._feat_selection.selected:
                out[column] = self._feat_frame[column].tolist()
            written = write_table(out, path)
            messagebox.showinfo(
                "Saved", f"Selected-feature table written to\n{written}"
            )
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not save", exc)

    def _feat_save_json(self) -> None:
        if self._feat_selection is None:
            messagebox.showinfo("Nothing to save",
                                "Compute a feature selection first.")
            return
        if not self._require_pharmaml():
            return
        path = filedialog.asksaveasfilename(
            title="Save the selection metadata", defaultextension=".json",
            initialfile="selection_metadata.json",
            filetypes=[("JSON metadata", "*.json")],
        )
        if not path:
            return
        try:
            from pharmaml.chem.features import selection_metadata

            meta = selection_metadata(self._feat_selection, self._feat_report,
                                      k=self._feat_k)
            if self._feat_target_name is not None:
                meta["target_column"] = self._feat_target_name
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2, ensure_ascii=False)
            messagebox.showinfo("Saved", f"Metadata written to\n{path}")
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not save", exc)

    # ------------------------------------------------------------------ #
    # PK / PD tab
    # ------------------------------------------------------------------ #
    def _build_pk_tab(self) -> None:
        page = self._new_page("PK / PD")

        half = make_card(page.grid, "Half-life ↔ elimination", row=0, column=0,
                         subtitle="k(e) = ln2 / t½  ·  accumulation over a dosing interval.")
        self._thalf_a = ttk.Entry(half, width=14); self._thalf_a.insert(0, "4")
        form_row(half, "t½ (h)", self._thalf_a)
        self._ke_a = ttk.Entry(half, width=14)
        form_row(half, "k(e) (1/h)", self._ke_a)
        self._tau_a = ttk.Entry(half, width=14); self._tau_a.insert(0, "6")
        form_row(half, "dosing interval τ (h)", self._tau_a)
        button = self._action_button(half, "Compute", self._halflife_math,
                                     tooltip="Derives k(e), accumulation ratio and time to steady state.")
        button.grid(row=half.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._half_result = ttk.Label(half, text="—", style="Result.TLabel",
                                      font=self._mono_result, wraplength=330, justify="left")
        self._half_result.grid(row=button.grid_info()["row"], column=1, sticky="w",
                               padx=(10, 0))
        bind_enter(self._tau_a, button)

        ivp = make_card(page.grid, "IV bolus concentration profile", row=0, column=1,
                        subtitle="C(t) = C0·e^(−k(e)·t) with half-life reference lines.")
        self._iv_c0 = ttk.Entry(ivp, width=14); self._iv_c0.insert(0, "10")
        form_row(ivp, "C0 (mg/L)", self._iv_c0)
        self._iv_thalf = ttk.Entry(ivp, width=14); self._iv_thalf.insert(0, "4")
        form_row(ivp, "t½ (h)", self._iv_thalf)
        self._iv_hours = ttk.Entry(ivp, width=14); self._iv_hours.insert(0, "24")
        form_row(ivp, "horizon (h)", self._iv_hours)
        button = self._action_button(ivp, "Plot profile", self._plot_iv_bolus, primary=True,
                                     tooltip="Draws the decay curve with 1·t½, 2·t½, 3·t½ markers.")
        button.grid(row=ivp.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._iv_host = self._figure_host(ivp)

        oral = make_card(page.grid, "Oral absorption PK", row=1, column=0,
                         subtitle="One-compartment first-order absorption.")
        self._po_dose = ttk.Entry(oral, width=14); self._po_dose.insert(0, "250")
        form_row(oral, "dose (mg)", self._po_dose)
        self._po_ka = ttk.Entry(oral, width=14); self._po_ka.insert(0, "1.2")
        form_row(oral, "k(a) (1/h)", self._po_ka)
        self._po_ke = ttk.Entry(oral, width=14); self._po_ke.insert(0, "0.173")
        form_row(oral, "k(e) (1/h)", self._po_ke)
        self._po_vd = ttk.Entry(oral, width=14); self._po_vd.insert(0, "40")
        form_row(oral, "Vd (L)", self._po_vd)
        self._po_f = ttk.Entry(oral, width=14); self._po_f.insert(0, "90")
        form_row(oral, "bioavailability F (%)", self._po_f)
        button = self._action_button(oral, "Compute", self._oral_pk)
        button.grid(row=oral.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._oral_result = ttk.Label(oral, text="—", style="Result.TLabel",
                                      font=self._mono_result, wraplength=320, justify="left")
        self._oral_result.grid(row=button.grid_info()["row"], column=1, sticky="w",
                               padx=(10, 0))
        bind_enter(self._po_f, button)

        crcl = make_card(page.grid, "Cockcroft–Gault creatinine clearance", row=1,
                         column=1, subtitle="Plus the renal dose adjustment for a normal regimen.")
        self._cg_age = ttk.Entry(crcl, width=14); self._cg_age.insert(0, "55")
        form_row(crcl, "age (years)", self._cg_age)
        self._cg_weight = ttk.Entry(crcl, width=14); self._cg_weight.insert(0, "70")
        form_row(crcl, "weight (kg)", self._cg_weight)
        self._cg_scr = ttk.Entry(crcl, width=14); self._cg_scr.insert(0, "1.0")
        form_row(crcl, "serum creatinine (mg/dL)", self._cg_scr)
        self._cg_sex = ttk.Combobox(crcl, state="readonly", width=12,
                                    values=("male", "female"))
        self._cg_sex.set("male")
        form_row(crcl, "sex", self._cg_sex)
        self._cg_dose = ttk.Entry(crcl, width=14); self._cg_dose.insert(0, "500")
        form_row(crcl, "normal dose (mg)", self._cg_dose)
        button = self._action_button(crcl, "Compute", self._creatinine_math)
        button.grid(row=crcl.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._cg_result = ttk.Label(crcl, text="—", style="Result.TLabel",
                                    font=self._mono_result, wraplength=300, justify="left")
        self._cg_result.grid(row=button.grid_info()["row"], column=1, sticky="w",
                             padx=(10, 0))
        bind_enter(self._cg_dose, button)

        doses = make_card(page.grid, "Loading & maintenance dosing", row=2, column=0)
        self._md_cl = ttk.Entry(doses, width=14); self._md_cl.insert(0, "5")
        form_row(doses, "clearance CL (L/h)", self._md_cl)
        self._md_css = ttk.Entry(doses, width=14); self._md_css.insert(0, "8")
        form_row(doses, "target Css (mg/L)", self._md_css)
        self._md_tau = ttk.Entry(doses, width=14); self._md_tau.insert(0, "8")
        form_row(doses, "interval τ (h)", self._md_tau)
        self._md_f = ttk.Entry(doses, width=14); self._md_f.insert(0, "90")
        form_row(doses, "bioavailability F (%)", self._md_f)
        self._ld_vd = ttk.Entry(doses, width=14); self._ld_vd.insert(0, "40")
        form_row(doses, "Vd (L) — loading", self._ld_vd)
        self._ld_target = ttk.Entry(doses, width=14); self._ld_target.insert(0, "2")
        form_row(doses, "target C (mg/L)", self._ld_target)
        button = self._action_button(doses, "Compute both", self._dose_design)
        button.grid(row=doses.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._dose_result = ttk.Label(doses, text="—", style="Result.TLabel",
                                      font=self._mono_result, wraplength=320, justify="left")
        self._dose_result.grid(row=button.grid_info()["row"], column=1, sticky="w",
                               padx=(10, 0))
        bind_enter(self._ld_target, button)

        bsa = make_card(page.grid, "BSA & weight-based dosing", row=2, column=1,
                        subtitle="Mosteller BSA; optional maximum doses cap the result.")
        self._bsa_height = ttk.Entry(bsa, width=14); self._bsa_height.insert(0, "175")
        form_row(bsa, "height (cm)", self._bsa_height)
        self._bsa_weight = ttk.Entry(bsa, width=14); self._bsa_weight.insert(0, "70")
        form_row(bsa, "weight (kg)", self._bsa_weight)
        self._bsa_per_m2 = ttk.Entry(bsa, width=14); self._bsa_per_m2.insert(0, "100")
        form_row(bsa, "dose per m²", self._bsa_per_m2)
        self._bsa_max = ttk.Entry(bsa, width=14)
        form_row(bsa, "max dose (optional)", self._bsa_max)
        self._wt_per_kg = ttk.Entry(bsa, width=14); self._wt_per_kg.insert(0, "15")
        form_row(bsa, "dose per kg", self._wt_per_kg)
        button = self._action_button(bsa, "Compute", self._bsa_math)
        button.grid(row=bsa.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._bsa_result = ttk.Label(bsa, text="—", style="Result.TLabel",
                                     font=self._mono_result, wraplength=300, justify="left")
        self._bsa_result.grid(row=button.grid_info()["row"], column=1, sticky="w",
                              padx=(10, 0))
        bind_enter(self._wt_per_kg, button)

        peds = make_card(page.grid, "Pediatric dose rules", row=3, column=0,
                         subtitle="Young's rule (age) and Clark's rule (weight in lb).")
        self._young_age = ttk.Entry(peds, width=14); self._young_age.insert(0, "6")
        form_row(peds, "child age (years)", self._young_age)
        self._clark_lb = ttk.Entry(peds, width=14); self._clark_lb.insert(0, "40")
        form_row(peds, "child weight (lb)", self._clark_lb)
        self._adult_dose = ttk.Entry(peds, width=14); self._adult_dose.insert(0, "500")
        form_row(peds, "adult dose (mg)", self._adult_dose)
        button = self._action_button(peds, "Compute both", self._pediatric_math)
        button.grid(row=peds.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._ped_result = ttk.Label(peds, text="—", style="Result.TLabel",
                                     font=self._mono_result, wraplength=320, justify="left")
        self._ped_result.grid(row=button.grid_info()["row"], column=1, sticky="w",
                              padx=(10, 0))
        bind_enter(self._adult_dose, button)

        emax = make_card(page.grid, "Emax pharmacodynamics", row=3, column=1,
                         subtitle="E = E0 + Emax·C/(EC50 + C) — table plus the full curve.")
        self._pd_e0 = ttk.Entry(emax, width=14); self._pd_e0.insert(0, "5")
        form_row(emax, "baseline E0", self._pd_e0)
        self._pd_emax = ttk.Entry(emax, width=14); self._pd_emax.insert(0, "60")
        form_row(emax, "Emax", self._pd_emax)
        self._pd_ec50 = ttk.Entry(emax, width=14); self._pd_ec50.insert(0, "2.5")
        form_row(emax, "EC50", self._pd_ec50)
        self._pd_concs = ttk.Entry(emax, width=24)
        self._pd_concs.insert(0, "0.5, 1, 2.5, 5, 10, 25")
        form_row(emax, "concentrations (comma list)", self._pd_concs)
        button = self._action_button(emax, "Compute & plot", self._emax_math, primary=True)
        button.grid(row=emax.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._pd_result = ttk.Label(emax, text="—", style="CardMuted.TLabel",
                                    wraplength=300, justify="left")
        self._pd_result.grid(row=button.grid_info()["row"], column=1, sticky="w",
                             padx=(10, 0))
        self._pd_host = self._figure_host(emax, height=300)
        bind_enter(self._pd_concs, button)

    def _halflife_math(self) -> None:
        if not self._require_pharmaml() or not self._require("numpy", "PK math"):
            return
        from pharmaml.pharma.pk import (
            accumulation_ratio,
            elimination_rate_constant,
            half_life,
        )

        try:
            t_half = parse_float(self._thalf_a.get(), "t½", minimum=1e-9)
            tau = parse_float(self._tau_a.get(), "τ", minimum=1e-9)
            ke = elimination_rate_constant(t_half)
            acc = accumulation_ratio(tau, t_half)
            t95 = math.log(20.0) / ke
            lines = [
                f"k(e) = {ke:.4f} 1/h",
                f"accumulation over τ = {acc:.2f}×",
                f"95% of steady state after {t95:.1f} h",
            ]
            raw_ke = self._ke_a.get().strip()
            if raw_ke:
                lines.append(f"t½ from k(e) = {half_life(parse_float(raw_ke, 'k(e)', minimum=1e-12)):.2f} h")
            self._half_result.configure(text="\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            self._show_error("Calculation failed", exc)

    def _plot_iv_bolus(self) -> None:
        if not self._require_pharmaml() or not self._require("numpy", "PK math"):
            return
        if not self._require("matplotlib", "the decay curve"):
            return
        try:
            c0 = parse_float(self._iv_c0.get(), "C0", minimum=0.0)
            t_half = parse_float(self._iv_thalf.get(), "t½", minimum=1e-9)
            hours = parse_float(self._iv_hours.get(), "horizon", minimum=0.1)
        except ValueError as exc:
            messagebox.showerror("Input", str(exc))
            return

        def work() -> Any:
            import numpy as np

            from pharmaml.pharma.pk import (
                elimination_rate_constant,
                iv_bolus_concentration,
            )

            ke = elimination_rate_constant(t_half)
            times = np.linspace(0.0, hours, 260)
            conc = np.asarray(iv_bolus_concentration(c0, ke, times), dtype=float)
            return ke, times, conc

        def render(payload: Any) -> None:
            ke, times, conc = payload
            from pharmaml.viz import PALETTE, new_figure

            fig, ax = new_figure(figsize=(5.2, 3.0))
            ax.plot(times, conc, color=PALETTE[0], lw=2.2, label="C(t)")
            for multiple in (1, 2, 3):
                mark = multiple * t_half
                if mark <= hours:
                    ax.axvline(mark, color=PALETTE[1], ls="--", lw=0.9, alpha=0.8)
                    ax.annotate(
                        f"{multiple}·t½",
                        xy=(mark, float(c0) * math.exp(-ke * mark)),
                        xytext=(4, 6),
                        textcoords="offset points",
                        fontsize=8,
                        color=MUTED,
                    )
            ax.set_xlabel("time (h)")
            ax.set_ylabel("concentration (mg/L)")
            ax.set_title(f"IV bolus decay — t½ {t_half:g} h, k(e) {ke:.3f} 1/h",
                         loc="left")
            ax.legend(frameon=False, fontsize=8)
            self._render_figure(self._iv_host, fig)

        self._start_job("IV bolus profile", work, render)

    def _oral_pk(self) -> None:
        if not self._require_pharmaml() or not self._require("numpy", "PK math"):
            return
        from pharmaml.pharma.pk import oral_cmax, oral_tmax

        try:
            dose = parse_float(self._po_dose.get(), "dose", minimum=0.0)
            ka = parse_float(self._po_ka.get(), "k(a)", minimum=1e-9)
            ke = parse_float(self._po_ke.get(), "k(e)", minimum=1e-12)
            vd = parse_float(self._po_vd.get(), "Vd", minimum=1e-9)
            bio = parse_float(self._po_f.get(), "F %", minimum=0.0, maximum=100.0) / 100.0
            if ka <= ke:
                raise ValueError("k(a) must exceed k(e) for a peak to exist")
            t_max = oral_tmax(ka, ke)
            c_max = oral_cmax(dose, ka, ke, vd, f=bio)
            self._oral_result.configure(
                text=f"t(max) = {t_max:.2f} h\nC(max)  = {c_max:.3f} mg/L"
            )
        except Exception as exc:  # noqa: BLE001
            self._show_error("Calculation failed", exc)

    def _creatinine_math(self) -> None:
        if not self._require_pharmaml() or not self._require("numpy", "PK math"):
            return
        from pharmaml.pharma.dosage import renal_dose_adjustment
        from pharmaml.pharma.pk import creatinine_clearance

        try:
            age = parse_float(self._cg_age.get(), "age", minimum=1.0, maximum=120.0)
            weight = parse_float(self._cg_weight.get(), "weight", minimum=0.1)
            scr = parse_float(self._cg_scr.get(), "serum creatinine", minimum=1e-6)
            sex = self._cg_sex.get()
            dose = parse_float(self._cg_dose.get(), "normal dose", minimum=0.0)
            clearance = creatinine_clearance(age, weight, scr, sex)
            adjusted = renal_dose_adjustment(dose, clearance)
            self._cg_result.configure(
                text=(f"CrCl = {clearance:.1f} mL/min\n"
                      f"adjusted dose = {adjusted:.1f} mg")
            )
        except Exception as exc:  # noqa: BLE001
            self._show_error("Calculation failed", exc)

    def _dose_design(self) -> None:
        if not self._require_pharmaml() or not self._require("numpy", "PK math"):
            return
        from pharmaml.pharma.dosage import loading_dose, maintenance_dose

        try:
            clearance = parse_float(self._md_cl.get(), "CL", minimum=1e-9)
            css = parse_float(self._md_css.get(), "target Css", minimum=1e-9)
            tau = parse_float(self._md_tau.get(), "τ", minimum=1e-9)
            bio = parse_float(self._md_f.get(), "F %", minimum=0.0, maximum=100.0) / 100.0
            vd = parse_float(self._ld_vd.get(), "Vd", minimum=1e-9)
            target = parse_float(self._ld_target.get(), "target C", minimum=1e-9)
            maintenance = maintenance_dose(clearance, css, tau, f=bio)
            loading = loading_dose(vd, target, f=bio)
            self._dose_result.configure(
                text=(f"maintenance = {maintenance:.1f} mg per τ\n"
                      f"loading      = {loading:.1f} mg")
            )
        except Exception as exc:  # noqa: BLE001
            self._show_error("Calculation failed", exc)

    def _bsa_math(self) -> None:
        if not self._require_pharmaml() or not self._require("numpy", "PK math"):
            return
        from pharmaml.pharma.dosage import bsa_mosteller, dose_by_bsa, dose_by_weight

        try:
            height = parse_float(self._bsa_height.get(), "height", minimum=1.0)
            weight = parse_float(self._bsa_weight.get(), "weight", minimum=0.1)
            per_m2 = parse_float(self._bsa_per_m2.get(), "dose per m²", minimum=0.0)
            per_kg = parse_float(self._wt_per_kg.get(), "dose per kg", minimum=0.0)
            raw_max = self._bsa_max.get().strip()
            cap = parse_float(raw_max, "max dose", minimum=0.0) if raw_max else None
            area = bsa_mosteller(height, weight)
            by_bsa = dose_by_bsa(per_m2, height, weight, max_dose=cap)
            by_weight = dose_by_weight(per_kg, weight, max_dose=cap)
            capped = " (capped)" if cap is not None and (
                by_bsa == cap or by_weight == cap) else ""
            self._bsa_result.configure(
                text=(f"BSA = {area:.2f} m²\n"
                      f"BSA dose = {by_bsa:.1f} mg\n"
                      f"weight dose = {by_weight:.1f} mg{capped}")
            )
        except Exception as exc:  # noqa: BLE001
            self._show_error("Calculation failed", exc)

    def _pediatric_math(self) -> None:
        if not self._require_pharmaml() or not self._require("numpy", "PK math"):
            return
        from pharmaml.pharma.dosage import clarks_rule, youngs_rule

        try:
            age = parse_float(self._young_age.get(), "child age", minimum=0.0, maximum=17.0)
            pounds = parse_float(self._clark_lb.get(), "child weight (lb)", minimum=0.1)
            adult = parse_float(self._adult_dose.get(), "adult dose", minimum=0.0)
            young = youngs_rule(age, adult)
            clark = clarks_rule(pounds, adult)
            self._ped_result.configure(
                text=f"Young (age)   = {young:.1f} mg\nClark (weight) = {clark:.1f} mg"
            )
        except Exception as exc:  # noqa: BLE001
            self._show_error("Calculation failed", exc)

    def _emax_math(self) -> None:
        if not self._require_pharmaml() or not self._require("numpy", "PD math"):
            return
        try:
            e0 = parse_float(self._pd_e0.get(), "E0")
            emax = parse_float(self._pd_emax.get(), "Emax")
            ec50 = parse_float(self._pd_ec50.get(), "EC50", minimum=1e-12)
            concs = [parse_float(part, "concentration", minimum=0.0)
                     for part in self._pd_concs.get().split(",") if part.strip()]
        except ValueError as exc:
            messagebox.showerror("Input", str(exc))
            return
        if not concs:
            messagebox.showwarning("Input", "Give at least one concentration.")
            return
        want_plot = self.deps.get("matplotlib")

        def work() -> Any:
            import numpy as np

            from pharmaml.pharma.pd import emax_effect

            effects = np.asarray(emax_effect(np.asarray(concs), e0, emax, ec50),
                                 dtype=float)
            grid = np.linspace(0.0, max(concs) * 1.15 or ec50 * 20, 240)
            curve = np.asarray(emax_effect(grid, e0, emax, ec50), dtype=float)
            return effects, grid, curve

        def render(payload: Any) -> None:
            effects, grid, curve = payload
            table = "  ·  ".join(
                f"C {format_number(c)} → E {format_number(e, 2)}"
                for c, e in zip(concs, effects)
            )
            self._pd_result.configure(text=table)
            if not want_plot:
                return
            try:
                from pharmaml.viz import PALETTE, new_figure

                fig, ax = new_figure(figsize=(5.2, 2.9))
                ax.plot(grid, curve, color=PALETTE[0], lw=2.2, label="E(C)")
                ax.scatter(concs, effects, color=PALETTE[1], zorder=3,
                           label="your points")
                ax.axvline(ec50, color=MUTED, ls=":", lw=1.0)
                ax.annotate("EC50", xy=(ec50, e0 + emax * 0.5), xytext=(5, -2),
                            textcoords="offset points", fontsize=8, color=MUTED)
                ax.set_xlabel("concentration")
                ax.set_ylabel("effect")
                ax.set_title("Emax model", loc="left")
                ax.legend(frameon=False, fontsize=8)
                self._render_figure(self._pd_host, fig)
            except Exception as exc:  # noqa: BLE001
                self._show_error("Plot failed", exc)

        self._start_job("Emax model", work, render)

    # ------------------------------------------------------------------ #
    # Cell Lines tab
    # ------------------------------------------------------------------ #
    def _build_cellline_tab(self) -> None:
        page = self._new_page("Cell Lines")
        self._bio_compound_var = tk.StringVar()

        screen = make_card(
            page.grid, "Response screen", span=2, row=0, column=0,
            subtitle="A GDSC/CCLE-style long table: one row per cell line × "
                     "compound, one metric column (LN_IC50, AUC, ...).",
        )
        screen_buttons = ttk.Frame(screen, style="Card.TFrame")
        screen_buttons.grid(row=0, column=0, columnspan=2, sticky="w")
        open_btn = self._action_button(screen_buttons, "Open CSV…", self._bio_open_csv)
        open_btn.pack(side="left")
        example_btn = self._action_button(
            screen_buttons, "Load example", self._bio_load_example,
            tooltip="The bundled 23-line × 8-compound screen "
                    "(examples/data/cellline_screen.csv).",
        )
        example_btn.pack(side="left", padx=(10, 0))
        self._bio_collapse_var = tk.BooleanVar(value=True)
        collapse_check = ttk.Checkbutton(
            screen, text="collapse replicates (median per duplicated pair)",
            variable=self._bio_collapse_var,
        )
        collapse_check.grid(row=0, column=1, sticky="e")
        self._bio_info = ttk.Label(screen, text="no screen loaded",
                                   style="CardMuted.TLabel")
        self._bio_info.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._bio_audit_text = _grid_text(screen, height=6, mono=self._mono)
        self._bio_audit_text.insert(
            "1.0", "the screen audit appears here (missing values, duplicate pairs, ...)"
        )
        grid_append(screen, self._bio_audit_text, sticky="ew", pady=(6, 0))

        matrix = make_card(
            page.grid, "Sensitivity matrix", span=2, row=1, column=0,
            subtitle="Pivot one metric into the classic cell line × compound table.",
        )
        self._bio_metric_combo = ttk.Combobox(matrix, state="readonly", width=22)
        form_row(matrix, "metric", self._bio_metric_combo)
        build_btn = self._action_button(
            matrix, "Build matrix", self._bio_build_matrix, primary=True,
            tooltip="Rows are cell lines, columns compounds; untested pairs stay "
                    "blank.",
        )
        build_btn.grid(row=matrix.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._bio_matrix_info = ttk.Label(
            matrix, text="no matrix yet — load a screen and press Build matrix",
            style="CardMuted.TLabel",
        )
        self._bio_matrix_info.grid(row=build_btn.grid_info()["row"], column=1,
                                   sticky="w", padx=(10, 0))
        matrix_frame = ttk.Frame(matrix, style="Card.TFrame")
        grid_append(matrix, matrix_frame, sticky="nsew", weight=True)
        self._bio_matrix_tree = ttk.Treeview(matrix_frame, show="tree headings",
                                             height=10)
        matrix_bar_x = ttk.Scrollbar(matrix_frame, orient="horizontal",
                                     command=self._bio_matrix_tree.xview)
        matrix_bar_y = ttk.Scrollbar(matrix_frame, orient="vertical",
                                     command=self._bio_matrix_tree.yview)
        self._bio_matrix_tree.configure(xscrollcommand=matrix_bar_x.set,
                                        yscrollcommand=matrix_bar_y.set)
        self._bio_matrix_tree.heading("#0", text="cell line")
        self._bio_matrix_tree.column("#0", width=110, stretch=True)
        self._bio_matrix_tree.pack(side="left", fill="both", expand=True)
        matrix_bar_y.pack(side="right", fill="y")
        matrix_bar_x.pack(side="bottom", fill="x")

        calls = make_card(
            page.grid, "Sensitive vs resistant", row=2, column=0,
            subtitle="Threshold one compound's panel into sensitivity calls.",
        )
        self._bio_compound_combo = ttk.Combobox(
            calls, state="readonly", width=22, textvariable=self._bio_compound_var
        )
        form_row(calls, "compound", self._bio_compound_combo)
        self._bio_q1 = ttk.Entry(calls, width=10)
        self._bio_q1.insert(0, "0.25")
        form_row(calls, "sensitive quantile", self._bio_q1)
        self._bio_q2 = ttk.Entry(calls, width=10)
        self._bio_q2.insert(0, "0.75")
        form_row(calls, "resistant quantile", self._bio_q2)
        self._bio_lower_var = tk.BooleanVar(value=True)
        lower_check = ttk.Checkbutton(
            calls, text="lower value = more sensitive (LN_IC50)",
            variable=self._bio_lower_var,
        )
        form_row(calls, "direction", lower_check)
        call_btn = self._action_button(calls, "Call", self._bio_call_sensitive)
        call_btn.grid(row=calls.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._bio_calls_summary = ttk.Label(calls, text="—", style="CardMuted.TLabel",
                                            wraplength=300, justify="left")
        self._bio_calls_summary.grid(row=call_btn.grid_info()["row"], column=1,
                                     sticky="w", padx=(10, 0))
        calls_frame = ttk.Frame(calls, style="Card.TFrame")
        grid_append(calls, calls_frame, sticky="nsew", weight=True)
        self._bio_calls_tree = ttk.Treeview(calls_frame, columns=("call",),
                                            show="tree headings", height=10)
        self._bio_calls_tree.heading("#0", text="cell line")
        self._bio_calls_tree.heading("call", text="call")
        self._bio_calls_tree.column("#0", width=130, stretch=True)
        self._bio_calls_tree.column("call", width=110, stretch=False)
        calls_bar = ttk.Scrollbar(calls_frame, orient="vertical",
                                  command=self._bio_calls_tree.yview)
        self._bio_calls_tree.configure(yscrollcommand=calls_bar.set)
        self._bio_calls_tree.pack(side="left", fill="both", expand=True)
        calls_bar.pack(side="right", fill="y")

        convert = make_card(
            page.grid, "LN_IC50 ↔ pIC50", row=2, column=1,
            subtitle="The GDSC and the chemist's scale, one number at a time.",
        )
        self._bio_ic50_entry = ttk.Entry(convert, width=22)
        self._bio_ic50_entry.insert(0, "0")
        form_row(convert, "value", self._bio_ic50_entry)
        convert_buttons = ttk.Frame(convert, style="Card.TFrame")
        convert_buttons.grid(row=convert.grid_size()[1], column=0, columnspan=2,
                             sticky="w", pady=(6, 0))
        ln_btn = self._action_button(convert_buttons, "LN_IC50 → pIC50",
                                     self._bio_ln_to_pic50)
        ln_btn.pack(side="left")
        pi_btn = self._action_button(convert_buttons, "pIC50 → LN_IC50",
                                     self._bio_pic50_to_ln)
        pi_btn.pack(side="left", padx=(10, 0))
        bind_enter(self._bio_ic50_entry, ln_btn)
        result_row = ttk.Frame(convert, style="Card.TFrame")
        result_row.grid(row=convert.grid_size()[1], column=0, columnspan=2,
                        sticky="w", pady=(6, 0))
        self._bio_ic50_result = ttk.Label(result_row, text="—",
                                          style="Result.TLabel",
                                          font=self._mono_result)
        self._bio_ic50_result.pack(side="left")
        copy_btn = self._action_button(
            result_row, "Copy", lambda: self._copy(self._bio_ic50_result.cget("text"))
        )
        copy_btn.pack(side="left", padx=(10, 0))
        grid_append(convert, ttk.Label(
            convert, style="CardMuted.TLabel", wraplength=300, justify="left",
            text="GDSC convention: LN_IC50 = ln(IC50 in µM), so 0 → pIC50 6.",
        ), pady=(4, 0))

        expression = make_card(
            page.grid, "Expression table", span=2, row=3, column=0,
            subtitle="Rows are cell lines, columns genes (log2 counts or "
                     "intensities — normalisation happens inside the analysis).",
        )
        expression_buttons = ttk.Frame(expression, style="Card.TFrame")
        expression_buttons.grid(row=0, column=0, columnspan=2, sticky="w")
        expr_open_btn = self._action_button(
            expression_buttons, "Open expression CSV…", self._bio_open_expr_csv
        )
        expr_open_btn.pack(side="left")
        expr_example_btn = self._action_button(
            expression_buttons, "Load example", self._bio_load_expr_example,
            tooltip="The bundled 23-line × 300-gene table "
                    "(examples/data/cellline_expression.csv).",
        )
        expr_example_btn.pack(side="left", padx=(10, 0))
        self._bio_expr_info = ttk.Label(expression, text="no expression table loaded",
                                        style="CardMuted.TLabel")
        self._bio_expr_info.grid(row=1, column=0, columnspan=2, sticky="w",
                                 pady=(6, 0))

        biomarkers = make_card(
            page.grid, "Biomarker discovery", span=2, row=4, column=0,
            subtitle="Correlate every gene's expression with one drug's response "
                     "(Benjamini–Hochberg FDR control) — the EGFR / erlotinib "
                     "story on the example data.",
        )
        self._bio_marker_compound_combo = ttk.Combobox(
            biomarkers, state="readonly", width=22,
            textvariable=self._bio_compound_var,
        )
        form_row(biomarkers, "compound", self._bio_marker_compound_combo)
        self._bio_method_combo = ttk.Combobox(
            biomarkers, state="readonly", width=22,
            values=("spearman", "pearson"),
        )
        self._bio_method_combo.set("spearman")
        form_row(biomarkers, "correlation method", self._bio_method_combo)
        self._bio_topn_entry = ttk.Entry(biomarkers, width=10)
        self._bio_topn_entry.insert(0, "10")
        form_row(biomarkers, "top N genes", self._bio_topn_entry)
        self._bio_marker_lower_var = tk.BooleanVar(value=True)
        marker_lower_check = ttk.Checkbutton(
            biomarkers, text="lower response = more sensitive (LN_IC50)",
            variable=self._bio_marker_lower_var,
        )
        form_row(biomarkers, "direction", marker_lower_check)
        marker_btn = self._action_button(
            biomarkers, "Discover biomarkers", self._bio_discover_biomarkers,
            primary=True,
            tooltip="Normalises + imputes the expression table, correlates every "
                    "gene against the compound's response column, ranks by |rho|.",
        )
        marker_btn.grid(row=biomarkers.grid_size()[1], column=0, sticky="w",
                        pady=(6, 0))
        self._bio_marker_summary = ttk.Label(
            biomarkers,
            text="no discovery yet — load a screen and an expression table",
            style="CardMuted.TLabel", wraplength=620, justify="left",
        )
        self._bio_marker_summary.grid(row=marker_btn.grid_info()["row"], column=1,
                                      sticky="w", padx=(10, 0))
        marker_frame = ttk.Frame(biomarkers, style="Card.TFrame")
        grid_append(biomarkers, marker_frame, sticky="nsew", weight=True)
        self._bio_marker_tree = ttk.Treeview(
            marker_frame,
            columns=("gene", "rho", "p_value", "fdr", "significant", "direction"),
            show="headings", height=10,
        )
        for column, width in (
            ("gene", 110), ("rho", 90), ("p_value", 110), ("fdr", 110),
            ("significant", 90), ("direction", 240),
        ):
            self._bio_marker_tree.heading(column, text=column)
            self._bio_marker_tree.column(column, width=width, minwidth=64,
                                         stretch=(column == "direction"))
        marker_bar_y = ttk.Scrollbar(marker_frame, orient="vertical",
                                     command=self._bio_marker_tree.yview)
        self._bio_marker_tree.configure(yscrollcommand=marker_bar_y.set)
        self._bio_marker_tree.pack(side="left", fill="both", expand=True)
        marker_bar_y.pack(side="right", fill="y")

    def _bio_open_csv(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas", "screen loading"):
            return
        path = filedialog.askopenfilename(
            title="Open a response screen",
            filetypes=[("Data files", "*.csv *.tsv *.txt"), ("All files", "*.*")],
        )
        if path:
            self._bio_load_path(Path(path))

    def _bio_load_example(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas", "screen loading"):
            return
        path = EXAMPLES_DIR / "cellline_screen.csv"
        if not path.is_file():
            messagebox.showwarning("Missing example",
                                   "examples/data/cellline_screen.csv was not found.")
            return
        self._bio_load_path(path)

    def _bio_load_path(self, path: Path) -> None:
        try:
            from pharmaml.biology import (
                audit_response_table,
                collapse_replicates,
                load_response_table,
            )

            raw = load_response_table(path)
            audit_frame = audit_response_table(raw)
            screen = raw
            if bool(self._bio_collapse_var.get()):
                screen = collapse_replicates(raw)
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not read the response screen", exc)
            return
        self._bio_screen = screen
        self._bio_screen_path = path
        self._bio_collapsed = screen is not raw
        self._bio_matrix = None
        self._bio_matrix_metric = None

        # the audit always describes the ORIGINAL table, so replicate
        # issues stay visible even when they were just collapsed away
        self._bio_audit_text.delete("1.0", "end")
        if len(audit_frame):
            self._bio_audit_text.insert("1.0", audit_frame.to_string())
        else:
            self._bio_audit_text.insert(
                "1.0", "audit: no issues — identifiers, metric values and pairs all clean"
            )

        raw_rows = len(raw.frame)
        rows_note = f"{raw_rows:,} rows"
        if len(screen.frame) != raw_rows:
            rows_note += f" ({len(screen.frame):,} after collapsing replicates)"
        metrics = ", ".join(screen.metrics)
        self._bio_info.configure(
            text=(f"{path.name} — {rows_note} · {len(screen.cell_lines)} cell lines · "
                  f"{len(screen.compounds)} compounds · metric(s): {metrics}")
        )
        self._bio_metric_combo["values"] = list(screen.metrics)
        if screen.metrics:
            self._bio_metric_combo.set(screen.metrics[0])
        compounds = list(screen.compounds)
        self._bio_compound_combo["values"] = compounds
        self._bio_marker_compound_combo["values"] = compounds
        if compounds:
            self._bio_compound_var.set(compounds[0])

    def _bio_build_matrix(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas", "the matrix"):
            return
        if self._bio_screen is None:
            messagebox.showwarning("No screen", "Load a response screen first.")
            return
        metric = self._bio_metric_combo.get()
        if not metric:
            messagebox.showwarning("No metric", "Pick the response metric.")
            return

        def work() -> Any:
            from pharmaml.biology import response_matrix

            return response_matrix(self._bio_screen, metric), metric

        def render(payload: Any) -> None:
            matrix, used_metric = payload
            self._bio_render_matrix(matrix, used_metric)

        self._start_job(f"sensitivity matrix ({metric})", work, render)

    def _bio_render_matrix(self, matrix: Any, metric: str) -> None:
        self._bio_matrix = matrix
        self._bio_matrix_metric = metric
        tree = self._bio_matrix_tree
        tree.delete(*tree.get_children())
        compounds = [str(column) for column in matrix.columns]
        tree["columns"] = compounds
        for column in compounds:
            tree.heading(column, text=column)
            tree.column(column, width=96, minwidth=72, stretch=True)
        for cell_line, row in matrix.iterrows():
            values = [
                "" if isinstance(value, float) and math.isnan(value)
                else _cell_text(value)
                for value in row.tolist()
            ]
            tree.insert("", "end", text=str(cell_line), values=values)
        self._bio_matrix_info.configure(
            text=(f"{len(matrix.index)} cell lines × {len(matrix.columns)} compounds · "
                  f"untested pairs blank · metric {metric}")
        )

    def _bio_call_sensitive(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas", "sensitivity calls"):
            return
        if self._bio_screen is None:
            messagebox.showwarning("No screen", "Load a response screen first.")
            return
        compound = self._bio_compound_var.get()
        if not compound:
            messagebox.showwarning("No compound", "Pick the compound to call.")
            return
        try:
            sensitive_q = parse_float(self._bio_q1.get(), "sensitive quantile",
                                      minimum=0.0, maximum=1.0)
            resistant_q = parse_float(self._bio_q2.get(), "resistant quantile",
                                      minimum=0.0, maximum=1.0)
            if sensitive_q >= resistant_q:
                raise ValueError(
                    "the sensitive quantile must be below the resistant quantile"
                )
        except ValueError as exc:
            messagebox.showerror("Input", str(exc))
            return
        lower = bool(self._bio_lower_var.get())
        metric = self._bio_metric_combo.get()

        def work() -> Any:
            from pharmaml.biology import call_sensitive_resistant, response_matrix

            matrix = self._bio_matrix
            built = matrix is None or self._bio_matrix_metric != metric
            if built:
                matrix = response_matrix(self._bio_screen, metric)
            calls = call_sensitive_resistant(
                matrix[compound], sensitive_q, resistant_q, lower_is_sensitive=lower
            )
            return matrix, metric, built, calls, compound

        def render(payload: Any) -> None:
            matrix, used_metric, built, calls, used_compound = payload
            if built:
                self._bio_render_matrix(matrix, used_metric)
            counts = calls.value_counts()
            labels = ("sensitive", "intermediate", "resistant", "not tested")
            summary = " · ".join(f"{int(counts.get(label, 0))} {label}"
                                 for label in labels)
            self._bio_calls_summary.configure(
                text=f"{used_compound}: {summary}", style="Field.TLabel"
            )
            self._bio_calls_tree.delete(*self._bio_calls_tree.get_children())
            rank = {label: index for index, label in enumerate(labels)}
            for cell_line, call in sorted(
                calls.items(), key=lambda item: (rank.get(item[1], 99), str(item[0]))
            ):
                self._bio_calls_tree.insert("", "end", text=str(cell_line),
                                            values=(call,))

        self._start_job(f"sensitivity calls ({compound})", work, render)

    def _bio_ln_to_pic50(self) -> None:
        self._bio_convert_ic50("ln")

    def _bio_pic50_to_ln(self) -> None:
        self._bio_convert_ic50("pic50")

    def _bio_convert_ic50(self, direction: str) -> None:
        if not self._require_pharmaml() or not self._require("numpy", "conversion"):
            return
        from pharmaml.biology import ln_ic50_to_pic50, pic50_to_ln_ic50

        try:
            value = parse_float(self._bio_ic50_entry.get(), "value")
            result = (
                ln_ic50_to_pic50(value) if direction == "ln" else pic50_to_ln_ic50(value)
            )
        except Exception as exc:  # noqa: BLE001
            self._show_error("Conversion failed", exc)
            return
        self._bio_ic50_result.configure(text=f"{result:.4f}")

    def _bio_open_expr_csv(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas", "expression loading"):
            return
        path = filedialog.askopenfilename(
            title="Open an expression table",
            filetypes=[("CSV table", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._bio_load_expr_path(Path(path))

    def _bio_load_expr_example(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas", "expression loading"):
            return
        path = EXAMPLES_DIR / "cellline_expression.csv"
        if not path.is_file():
            messagebox.showwarning(
                "Missing example", "examples/data/cellline_expression.csv was not found."
            )
            return
        self._bio_load_expr_path(path)

    def _bio_load_expr_path(self, path: Path) -> None:
        try:
            import pandas as pd

            try:
                frame = pd.read_csv(path, index_col="cell_line")
            except ValueError:
                # user tables name their identifier column differently --
                # fall back to the first column as the cell-line index
                frame = pd.read_csv(path, index_col=0)
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not read the expression table", exc)
            return
        if frame.empty:
            messagebox.showwarning("Empty table", f"{path.name} has no rows.")
            return
        self._bio_expr = frame
        self._bio_expr_norm = None
        self._bio_expr_info.configure(
            text=f"{path.name} — {len(frame):,} cell lines × {len(frame.columns):,} genes"
        )

    def _bio_discover_biomarkers(self) -> None:
        if not self._require_pharmaml():
            return
        for module in ("numpy", "pandas", "scipy"):
            if not self._require(module, "biomarker discovery"):
                return
        if self._bio_screen is None or self._bio_expr is None:
            messagebox.showwarning(
                "Missing data",
                "Biomarker discovery needs BOTH a response screen and an "
                "expression table — load one of each above, then try again.",
            )
            return
        compound = self._bio_compound_var.get()
        if not compound:
            messagebox.showwarning("No compound", "Pick the compound to analyse.")
            return
        method = self._bio_method_combo.get() or "spearman"
        try:
            top_n = parse_int(self._bio_topn_entry.get(), "top N genes",
                              minimum=1, maximum=500)
        except ValueError as exc:
            messagebox.showerror("Input", str(exc))
            return
        lower = bool(self._bio_marker_lower_var.get())
        metric = self._bio_metric_combo.get()

        def work() -> Any:
            from pharmaml.biology import (
                impute_expression,
                normalize_expression,
                response_matrix,
                sensitivity_associations,
                top_biomarkers,
            )

            norm = self._bio_expr_norm
            if norm is None:
                norm = impute_expression(normalize_expression(self._bio_expr))
            matrix = self._bio_matrix
            built = matrix is None or self._bio_matrix_metric != metric
            if built:
                matrix = response_matrix(self._bio_screen, metric)
            associations = sensitivity_associations(
                norm, matrix[compound], method=method
            )
            top = top_biomarkers(associations, n=top_n)
            return top, associations, norm, matrix, metric, built, compound

        def render(payload: Any) -> None:
            from pharmaml.biology import direction_of_association

            top, associations, norm, matrix, used_metric, built, used_compound = payload
            self._bio_expr_norm = norm
            if built:
                self._bio_render_matrix(matrix, used_metric)
            n_significant = int(associations["significant"].sum())
            strongest = float(associations["rho"].abs().max())
            self._bio_marker_summary.configure(
                text=(f"{len(associations):,} genes correlated · "
                      f"{n_significant} significant at FDR < 0.05 · "
                      f"strongest |rho| {strongest:.3f} — {used_compound}"),
                style="Field.TLabel",
            )
            self._bio_marker_tree.delete(*self._bio_marker_tree.get_children())
            for _, row in top.iterrows():
                try:
                    rho = float(row["rho"])
                except (TypeError, ValueError):
                    rho = float("nan")
                self._bio_marker_tree.insert(
                    "", "end",
                    values=(
                        _cell_text(row["gene"], limit=18),
                        format_number(rho),
                        _cell_text(row["p_value"]),
                        _cell_text(row["fdr"]),
                        "Yes" if bool(row["significant"]) else "No",
                        direction_of_association(
                            rho, lower_is_sensitive=lower
                        ),
                    ),
                )

        self._start_job(f"biomarkers for {compound}", work, render)

    # ------------------------------------------------------------------ #
    # Docking tab
    # ------------------------------------------------------------------ #
    def _build_docking_tab(self) -> None:
        page = self._new_page("Docking")

        # -- card 1: ligand library -> PDBQT campaign -------------------- #
        ligands = make_card(
            page.grid, "Ligand library → PDBQT", row=0, column=0,
            subtitle="A SMILES CSV becomes one folder per ligand, each with a "
                     "3D l.pdbqt (Meeko + RDKit).",
        )
        lib_buttons = ttk.Frame(ligands, style="Card.TFrame")
        lib_buttons.grid(row=0, column=0, columnspan=2, sticky="w")
        lib_open_btn = self._action_button(lib_buttons, "Open CSV…",
                                           self._dock_open_csv)
        lib_open_btn.pack(side="left")
        lib_example_btn = self._action_button(
            lib_buttons, "Load example", self._dock_load_example,
            tooltip="The bundled 10-row library (8 valid SMILES, 2 broken "
                    "rows on purpose) -- examples/data/ligand_library.csv.",
        )
        lib_example_btn.pack(side="left", padx=(10, 0))
        self._dock_lib_info = ttk.Label(ligands, text="no library loaded",
                                        style="CardMuted.TLabel")
        self._dock_lib_info.grid(row=1, column=0, columnspan=2, sticky="w",
                                 pady=(6, 0))
        out_buttons = ttk.Frame(ligands, style="Card.TFrame")
        out_buttons.grid(row=2, column=0, columnspan=2, sticky="w")
        out_btn = self._action_button(out_buttons, "Choose output folder…",
                                      self._dock_choose_out_dir)
        out_btn.pack(side="left")
        self._dock_out_info = ttk.Label(out_buttons, text="no output folder chosen",
                                        style="CardMuted.TLabel", wraplength=280,
                                        justify="left")
        self._dock_out_info.pack(side="left", padx=(10, 0))
        prep_btn = self._action_button(
            ligands, "Prepare ligands", self._dock_prepare_ligands, primary=True,
            tooltip="Embeds and atom-types every valid SMILES in a worker "
                    "thread; one sub-folder per ligand under the output root.",
        )
        prep_btn.grid(row=ligands.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._dock_prep_summary = ttk.Label(
            ligands, text="—", style="CardMuted.TLabel", wraplength=300,
            justify="left",
        )
        self._dock_prep_summary.grid(row=prep_btn.grid_info()["row"], column=1,
                                     sticky="w", padx=(10, 0))
        lib_tree_frame = ttk.Frame(ligands, style="Card.TFrame")
        grid_append(ligands, lib_tree_frame, sticky="nsew", weight=True)
        self._dock_lib_tree = ttk.Treeview(
            lib_tree_frame,
            columns=("folder", "name", "ok", "atoms", "torsdof", "error"),
            show="headings", height=8,
        )
        for column, width in (
            ("folder", 120), ("name", 100), ("ok", 44), ("atoms", 60),
            ("torsdof", 66), ("error", 170),
        ):
            self._dock_lib_tree.heading(column, text=column)
            self._dock_lib_tree.column(column, width=width, minwidth=44,
                                       stretch=(column == "error"))
        lib_bar = ttk.Scrollbar(lib_tree_frame, orient="vertical",
                                command=self._dock_lib_tree.yview)
        self._dock_lib_tree.configure(yscrollcommand=lib_bar.set)
        self._dock_lib_tree.pack(side="left", fill="both", expand=True)
        lib_bar.pack(side="right", fill="y")
        lib_save_btn = self._action_button(ligands, "Save report CSV…",
                                           self._dock_save_ligand_report)
        lib_save_btn.grid(row=ligands.grid_size()[1], column=0, columnspan=2,
                          sticky="w", pady=(6, 0))

        # -- card 2: receptor preparation -------------------------------- #
        receptor = make_card(
            page.grid, "Receptor preparation", row=0, column=1,
            subtitle="PDB / CIF / PDBQT → a docking-ready receptor PDBQT "
                     "(waters out, atom types in).",
        )
        rec_buttons = ttk.Frame(receptor, style="Card.TFrame")
        rec_buttons.grid(row=0, column=0, columnspan=2, sticky="w")
        rec_open_btn = self._action_button(rec_buttons,
                                           "Open PDB / CIF / PDBQT…",
                                           self._dock_open_receptor)
        rec_open_btn.pack(side="left")
        rec_example_btn = self._action_button(
            rec_buttons, "Load example", self._dock_load_receptor_example,
            tooltip="The bundled hen egg-white lysozyme "
                    "(examples/data/lysozyme_1aki.pdb).",
        )
        rec_example_btn.pack(side="left", padx=(10, 0))
        self._dock_receptor_info = ttk.Label(
            receptor, text="no receptor source selected",
            style="CardMuted.TLabel", wraplength=300, justify="left",
        )
        self._dock_receptor_info.grid(row=1, column=0, columnspan=2, sticky="w",
                                      pady=(6, 0))
        self._dock_remove_waters_var = tk.BooleanVar(value=True)
        waters_check = ttk.Checkbutton(receptor, text="remove waters",
                                       variable=self._dock_remove_waters_var)
        form_row(receptor, "waters", waters_check)
        self._dock_keep_hetatm_var = tk.BooleanVar(value=False)
        hetatm_check = ttk.Checkbutton(receptor, text="keep HETATM ligands/ions",
                                       variable=self._dock_keep_hetatm_var)
        form_row(receptor, "HETATM", hetatm_check)
        self._dock_strip_h_var = tk.BooleanVar(value=True)
        strip_check = ttk.Checkbutton(receptor, text="strip hydrogens",
                                      variable=self._dock_strip_h_var)
        form_row(receptor, "hydrogens", strip_check)
        self._dock_chain_entry = ttk.Entry(receptor, width=10)
        form_row(receptor, "chain (blank = all)", self._dock_chain_entry)
        rec_btn = self._action_button(
            receptor, "Prepare receptor", self._dock_prepare_receptor, primary=True,
            tooltip="Asks where to save the PDBQT, then prepares it in a "
                    "worker thread.",
        )
        rec_btn.grid(row=receptor.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._dock_protein_text = _grid_text(receptor, height=10, mono=self._mono)
        self._dock_protein_text.insert(
            "1.0", "the receptor report appears here (mode, atoms, chains, ...)"
        )
        grid_append(receptor, self._dock_protein_text, sticky="ew", pady=(6, 0))
        dist_buttons = ttk.Frame(receptor, style="Card.TFrame")
        grid_append(receptor, dist_buttons, sticky="w", pady=(6, 0))
        dist_btn = self._action_button(
            dist_buttons, "Distribute into campaign…",
            self._dock_distribute_receptor,
            tooltip="Copies the prepared receptor into every ligand folder "
                    "of a campaign root.",
        )
        dist_btn.pack(side="left")
        self._dock_distribute_info = ttk.Label(
            dist_buttons, text="", style="CardMuted.TLabel", wraplength=260,
            justify="left",
        )
        self._dock_distribute_info.pack(side="left", padx=(10, 0))

        # -- card 3: grid box & docking configs -------------------------- #
        grid_card = make_card(
            page.grid, "Grid box & docking configs", row=1, column=0,
            subtitle="Enclose a reference ligand, then write the AutoDock4 "
                     "r.gpf / r.dpf pair next to it.",
        )
        lig_buttons = ttk.Frame(grid_card, style="Card.TFrame")
        lig_buttons.grid(row=0, column=0, columnspan=2, sticky="w")
        lig_open_btn = self._action_button(lig_buttons, "Open reference ligand PDBQT…",
                                           self._dock_open_ligand)
        lig_open_btn.pack(side="left")
        self._dock_ligand_info = ttk.Label(
            lig_buttons, text="no ligand selected", style="CardMuted.TLabel",
            wraplength=220, justify="left",
        )
        self._dock_ligand_info.pack(side="left", padx=(10, 0))
        self._dock_padding_entry = ttk.Entry(grid_card, width=10)
        self._dock_padding_entry.insert(0, "4.0")
        form_row(grid_card, "padding (Å)", self._dock_padding_entry)
        self._dock_npts_entry = ttk.Entry(grid_card, width=12)
        self._dock_npts_entry.insert(0, "60 60 60")
        form_row(grid_card, "npts (x y z)", self._dock_npts_entry)
        self._dock_spacing_entry = ttk.Entry(grid_card, width=10)
        self._dock_spacing_entry.insert(0, "0.375")
        form_row(grid_card, "spacing (Å)", self._dock_spacing_entry)
        self._dock_receptor_name_entry = ttk.Entry(grid_card, width=14)
        self._dock_receptor_name_entry.insert(0, "r.pdbqt")
        form_row(grid_card, "receptor file", self._dock_receptor_name_entry)
        self._dock_ga_run_entry = ttk.Entry(grid_card, width=10)
        self._dock_ga_run_entry.insert(0, "100")
        form_row(grid_card, "ga_run", self._dock_ga_run_entry)
        self._dock_ga_pop_entry = ttk.Entry(grid_card, width=10)
        self._dock_ga_pop_entry.insert(0, "150")
        form_row(grid_card, "ga population", self._dock_ga_pop_entry)
        grid_btn = self._action_button(
            grid_card, "Compute box & write GPF/DPF", self._dock_compute_grid,
            primary=True,
            tooltip="Grid box from the ligand plus padding; GPF and DPF land "
                    "in the ligand's own folder.",
        )
        grid_btn.grid(row=grid_card.grid_size()[1], column=0, columnspan=2,
                      sticky="w", pady=(6, 0))
        self._dock_grid_text = _grid_text(grid_card, height=7, mono=self._mono)
        self._dock_grid_text.insert(
            "1.0", "box centre / size, ligand atom types and the written "
                   "paths appear here"
        )
        grid_append(grid_card, self._dock_grid_text, sticky="ew", pady=(6, 0))

        # -- card 4: docking results (one tree, three sources) ----------- #
        results = make_card(
            page.grid, "Docking results", row=1, column=1,
            subtitle="One tree, three sources: a campaign scan, an AutoDock4 "
                     "DLG cluster table, or a Vina mode table.",
        )
        scan_buttons = ttk.Frame(results, style="Card.TFrame")
        scan_buttons.grid(row=0, column=0, columnspan=2, sticky="w")
        scan_btn = self._action_button(scan_buttons, "Scan campaign folder…",
                                       self._dock_scan_campaign)
        scan_btn.pack(side="left")
        scan_example_btn = self._action_button(
            scan_buttons, "Load example campaign", self._dock_load_example_campaign,
            tooltip="The bundled 14-folder campaign "
                    "(examples/data/docking_campaign).",
        )
        scan_example_btn.pack(side="left", padx=(10, 0))
        dlg_buttons = ttk.Frame(results, style="Card.TFrame")
        dlg_buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        dlg_btn = self._action_button(dlg_buttons, "Open DLG…", self._dock_open_dlg)
        dlg_btn.pack(side="left")
        dlg_example_btn = self._action_button(
            dlg_buttons, "Open example DLG", self._dock_open_example_dlg,
            tooltip="The aspirin run of the example campaign "
                    "(01_aspirin/r.dlg): five clusters.",
        )
        dlg_example_btn.pack(side="left", padx=(10, 0))
        self._dock_heavyatoms_entry = ttk.Entry(dlg_buttons, width=6)
        self._dock_heavyatoms_entry.pack(side="left", padx=(10, 0))
        ttk.Label(dlg_buttons, text="heavy atoms (adds ligand efficiency)",
                  style="CardMuted.TLabel").pack(side="left", padx=(4, 0))
        vina_row = ttk.Frame(results, style="Card.TFrame")
        vina_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        vina_btn = self._action_button(vina_row, "Open Vina output…",
                                       self._dock_open_vina)
        vina_btn.pack(side="left")
        self._dock_results_info = ttk.Label(
            results, text="no results yet — scan a campaign or open a log",
            style="CardMuted.TLabel", wraplength=320, justify="left",
        )
        self._dock_results_info.grid(row=3, column=0, columnspan=2, sticky="w",
                                     pady=(6, 0))
        results_tree_frame = ttk.Frame(results, style="Card.TFrame")
        grid_append(results, results_tree_frame, sticky="nsew", weight=True)
        self._dock_results_tree = ttk.Treeview(results_tree_frame, show="headings",
                                               height=8)
        results_bar = ttk.Scrollbar(results_tree_frame, orient="vertical",
                                    command=self._dock_results_tree.yview)
        self._dock_results_tree.configure(yscrollcommand=results_bar.set)
        self._dock_results_tree.pack(side="left", fill="both", expand=True)
        results_bar.pack(side="right", fill="y")
        results_save_btn = self._action_button(results, "Save results CSV…",
                                               self._dock_save_results)
        results_save_btn.grid(row=results.grid_size()[1], column=0, columnspan=2,
                              sticky="w", pady=(6, 0))

        # -- card 5: campaign runner ------------------------------------- #
        runner = make_card(
            page.grid, "Campaign runner", row=2, column=0,
            subtitle="autogrid4 + autodock4 over every prepared ligand folder, "
                     "or one AutoDock Vina run.",
        )
        self._dock_engine_combo = ttk.Combobox(runner, state="readonly", width=22,
                                               values=_DOCK_ENGINES)
        self._dock_engine_combo.set(_DOCK_ENGINES[0])
        form_row(runner, "engine", self._dock_engine_combo)
        root_buttons = ttk.Frame(runner, style="Card.TFrame")
        root_buttons.grid(row=runner.grid_size()[1], column=0, columnspan=2,
                          sticky="w")
        root_btn = self._action_button(root_buttons, "Choose campaign root…",
                                       self._dock_choose_campaign_root)
        root_btn.pack(side="left")
        self._dock_campaign_info = ttk.Label(
            root_buttons, text="no campaign root chosen", style="CardMuted.TLabel",
            wraplength=240, justify="left",
        )
        self._dock_campaign_info.pack(side="left", padx=(10, 0))
        vina_in_buttons = ttk.Frame(runner, style="Card.TFrame")
        vina_in_buttons.grid(row=runner.grid_size()[1], column=0, columnspan=2,
                             sticky="w", pady=(4, 0))
        vina_rec_btn = self._action_button(vina_in_buttons, "Receptor PDBQT…",
                                           self._dock_open_vina_receptor)
        vina_rec_btn.pack(side="left")
        vina_lig_btn = self._action_button(vina_in_buttons, "Ligand PDBQT…",
                                           self._dock_open_vina_ligand)
        vina_lig_btn.pack(side="left", padx=(10, 0))
        self._dock_vina_info = ttk.Label(
            vina_in_buttons, text="no receptor / ligand chosen",
            style="CardMuted.TLabel", wraplength=240, justify="left",
        )
        self._dock_vina_info.pack(side="left", padx=(10, 0))
        self._dock_center_entry = ttk.Entry(runner, width=12)
        self._dock_center_entry.insert(0, "0 0 0")
        form_row(runner, "centre (x y z, Vina)", self._dock_center_entry)
        self._dock_size_entry = ttk.Entry(runner, width=12)
        self._dock_size_entry.insert(0, "20 20 20")
        form_row(runner, "size (x y z, Vina)", self._dock_size_entry)
        self._dock_exhaust_entry = ttk.Entry(runner, width=10)
        self._dock_exhaust_entry.insert(0, "8")
        form_row(runner, "exhaustiveness (Vina)", self._dock_exhaust_entry)
        self._dock_workers_entry = ttk.Entry(runner, width=10)
        self._dock_workers_entry.insert(0, "4")
        form_row(runner, "workers (AutoDock4)", self._dock_workers_entry)
        run_btn = self._action_button(
            runner, "Run docking", self._dock_run, primary=True,
            tooltip="Real docking takes minutes to hours per ligand -- the "
                    "window stays responsive and the run lands in the logs.",
        )
        run_btn.grid(row=runner.grid_size()[1], column=0, sticky="w", pady=(6, 0))
        self._dock_run_summary = ttk.Label(
            runner, text="—", style="CardMuted.TLabel", wraplength=320,
            justify="left",
        )
        self._dock_run_summary.grid(row=run_btn.grid_info()["row"], column=1,
                                    sticky="w", padx=(10, 0))

    # -- ligand library card ------------------------------------------- #
    def _dock_open_csv(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas",
                                                            "library loading"):
            return
        path = filedialog.askopenfilename(
            title="Open a ligand library",
            filetypes=[("SMILES library", "*.csv *.tsv"), ("All files", "*.*")],
        )
        if path:
            self._dock_load_lib_path(Path(path))

    def _dock_load_example(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas",
                                                            "library loading"):
            return
        path = EXAMPLES_DIR / "ligand_library.csv"
        if not path.is_file():
            messagebox.showwarning(
                "Missing example", "examples/data/ligand_library.csv was not found."
            )
            return
        self._dock_load_lib_path(path)

    def _dock_load_lib_path(self, path: Path) -> None:
        try:
            from pharmaml.docking import load_ligand_csv

            library = load_ligand_csv(path)
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not read the ligand library", exc)
            return
        if library.frame.empty:
            messagebox.showwarning("Empty library", f"{path.name} has no rows.")
            return
        self._dock_library = library
        self._dock_lib_path = path
        valid = int((library.frame["status"] == "valid").sum())
        self._dock_lib_info.configure(
            text=(f"{path.name} — {len(library.frame)} rows · {valid} valid "
                  f"SMILES · {len(library.frame) - valid} invalid")
        )

    def _dock_choose_out_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose the ligand output folder")
        if path:
            self._dock_out_dir = Path(path)
            self._dock_out_info.configure(text=f"campaign output: {path}")

    def _dock_prepare_ligands(self) -> None:
        if not self._require_pharmaml():
            return
        if not self._require("pandas", "ligand preparation") or not self._require(
                "rdkit", "ligand preparation"):
            return
        if self._dock_library is None:
            messagebox.showwarning("No library", "Load a ligand library first.")
            return
        if self._dock_out_dir is None:
            messagebox.showwarning(
                "No output folder",
                "Choose the campaign output folder first (the tab creates one "
                "sub-folder per ligand inside it).",
            )
            return
        library, output_root = self._dock_library, self._dock_out_dir

        def work() -> Any:
            from pharmaml.docking import prepare_ligand_library

            return prepare_ligand_library(library, output_root)

        def render(report: Any) -> None:
            self._dock_render_ligand_report(report)

        self._start_job("preparing ligands", work, render)

    def _dock_render_ligand_report(self, report: Any) -> None:
        self._dock_report = report
        frame = report.frame
        ok_count = int(frame["ok"].sum())
        failed = len(frame) - ok_count
        self._dock_prep_summary.configure(
            text=(f"{ok_count}/{len(frame)} prepared · {failed} failed · "
                  f"campaign at {report.root}"),
            style="Field.TLabel",
        )
        tree = self._dock_lib_tree
        tree.delete(*tree.get_children())
        for row in frame.to_dict("records"):
            tree.insert(
                "", "end",
                values=(
                    _cell_text(row["folder"], limit=22),
                    _cell_text(row["name"], limit=16),
                    "Yes" if bool(row["ok"]) else "No",
                    _cell_text(row["atoms"]),
                    _cell_text(row["torsdof"]),
                    _cell_text(row["error"], limit=60),
                ),
            )

    def _dock_save_ligand_report(self) -> None:
        if self._dock_report is None:
            messagebox.showinfo("Nothing to save", "Prepare a ligand library first.")
            return
        if not self._require_pharmaml():
            return
        path = filedialog.asksaveasfilename(
            title="Save the ligand preparation report", defaultextension=".csv",
            filetypes=[("CSV table", "*.csv")],
        )
        if not path:
            return
        try:
            from pharmaml.utils.io import write_table

            written = write_table(self._dock_report.frame, path)
            messagebox.showinfo("Saved", f"Preparation report written to\n{written}")
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not save", exc)

    # -- receptor card -------------------------------------------------- #
    def _dock_open_receptor(self) -> None:
        if not self._require_pharmaml() or not self._require("rdkit",
                                                            "receptor preparation"):
            return
        path = filedialog.askopenfilename(
            title="Open a receptor structure",
            filetypes=[("Structure files", "*.pdb *.cif *.pdbqt"),
                       ("All files", "*.*")],
        )
        if path:
            self._dock_set_receptor_source(Path(path))

    def _dock_load_receptor_example(self) -> None:
        if not self._require_pharmaml() or not self._require("rdkit",
                                                            "receptor preparation"):
            return
        path = EXAMPLES_DIR / "lysozyme_1aki.pdb"
        if not path.is_file():
            messagebox.showwarning(
                "Missing example", "examples/data/lysozyme_1aki.pdb was not found."
            )
            return
        self._dock_set_receptor_source(path)

    def _dock_set_receptor_source(self, path: Path) -> None:
        self._dock_receptor_source = path
        self._dock_receptor_info.configure(text=f"{path.name}")

    def _dock_prepare_receptor(self) -> None:
        if not self._require_pharmaml() or not self._require("rdkit",
                                                            "receptor preparation"):
            return
        if self._dock_receptor_source is None:
            messagebox.showwarning(
                "No receptor", "Open a PDB / CIF / PDBQT structure first."
            )
            return
        source = self._dock_receptor_source
        output_pdbqt = filedialog.asksaveasfilename(
            title="Save the prepared receptor", defaultextension=".pdbqt",
            initialfile="receptor.pdbqt",
            filetypes=[("PDBQT receptor", "*.pdbqt")],
        )
        if not output_pdbqt:
            return
        remove_waters = bool(self._dock_remove_waters_var.get())
        keep_hetatm = bool(self._dock_keep_hetatm_var.get())
        strip_hydrogens = bool(self._dock_strip_h_var.get())
        chain = self._dock_chain_entry.get().strip() or None
        target = Path(output_pdbqt)

        def work() -> Any:
            from pharmaml.docking import prepare_protein

            return prepare_protein(
                source, output_pdbqt=target,
                remove_waters=remove_waters, keep_hetatm=keep_hetatm,
                chain=chain, strip_hydrogens=strip_hydrogens,
            )

        def render(report: Any) -> None:
            self._dock_render_protein_report(report)

        self._start_job(f"preparing {source.name}", work, render)

    def _dock_render_protein_report(self, report: Any) -> None:
        self._dock_protein_report = report
        self._dock_receptor_path = Path(report.output_pdbqt)
        chains = ", ".join(report.chains) if report.chains else "—"
        types = " · ".join(
            f"{name} {count:,}" for name, count in
            sorted(report.atom_types.items(), key=lambda item: -item[1])
        )
        lines = [
            f"mode            {report.mode} ({report.source.suffix or 'input'})",
            (f"input           {report.input_atoms:,} atoms · "
             f"{report.input_hetatms:,} HETATM"),
            (f"removed         {report.waters_removed:,} waters · "
             f"{report.hetatm_removed:,} HETATM · "
             f"{report.altloc_removed:,} altloc · "
             f"{report.hydrogens_removed:,} hydrogens"),
            f"chains          {chains}",
            f"residues        {report.residues:,}",
            f"output          {report.output_atoms:,} atoms ({types})",
            (f"charges         {report.charge_min:+.3f} … {report.charge_max:+.3f} "
             f"(mean {report.charge_mean:+.4f})"),
            f"written to      {report.output_pdbqt}",
        ]
        if report.notes:
            lines.append("notes           " + " · ".join(str(note) for note in
                                                         report.notes))
        else:
            lines.append("notes           (none)")
        self._dock_protein_text.delete("1.0", "end")
        self._dock_protein_text.insert("1.0", "\n".join(lines))

    def _dock_distribute_receptor(self) -> None:
        if not self._require_pharmaml():
            return
        receptor = self._dock_receptor_path
        if receptor is None:
            messagebox.showinfo(
                "No receptor yet",
                "Prepare a receptor above first -- or pick an already-prepared "
                "PDBQT in the next dialog.",
            )
            path = filedialog.askopenfilename(
                title="Open the prepared receptor PDBQT",
                filetypes=[("PDBQT receptor", "*.pdbqt"), ("All files", "*.*")],
            )
            if not path:
                return
            receptor = self._dock_receptor_path = Path(path)
        root = filedialog.askdirectory(title="Choose the campaign root")
        if not root:
            return
        campaign_root = Path(root)

        def work() -> Any:
            from pharmaml.docking import distribute_receptor

            return distribute_receptor(receptor, campaign_root)

        def render(copies: Any) -> None:
            self._dock_distribute_info.configure(
                text=f"receptor copied into {len(copies)} folders "
                     f"({receptor.name} → {campaign_root})",
                style="Field.TLabel",
            )

        self._start_job("distributing receptor", work, render)

    # -- grid box card --------------------------------------------------- #
    def _dock_open_ligand(self) -> None:
        if not self._require_pharmaml():
            return
        path = filedialog.askopenfilename(
            title="Open a reference ligand PDBQT",
            filetypes=[("PDBQT ligand", "*.pdbqt"), ("All files", "*.*")],
        )
        if path:
            self._dock_set_ligand_path(Path(path))

    def _dock_set_ligand_path(self, path: Path) -> None:
        self._dock_ligand_path = path
        self._dock_ligand_info.configure(text=path.name)

    def _dock_compute_grid(self) -> None:
        if not self._require_pharmaml():
            return
        if self._dock_ligand_path is None:
            messagebox.showwarning(
                "No ligand", "Open a reference ligand PDBQT first."
            )
            return
        ligand = self._dock_ligand_path
        try:
            padding = parse_float(self._dock_padding_entry.get(), "padding",
                                  minimum=0.0)
            npts = parse_int_triple(self._dock_npts_entry.get(), "npts", minimum=1)
            spacing = parse_float(self._dock_spacing_entry.get(), "spacing",
                                  minimum=0.01)
            receptor_file = self._dock_receptor_name_entry.get().strip()
            if not receptor_file:
                raise ValueError("receptor file: pick a file name (e.g. r.pdbqt)")
            ga_run = parse_int(self._dock_ga_run_entry.get(), "ga_run", minimum=1)
            ga_pop = parse_int(self._dock_ga_pop_entry.get(), "ga population",
                               minimum=1)
        except ValueError as exc:
            messagebox.showerror("Input", str(exc))
            return

        def work() -> Any:
            from pharmaml.docking import (
                compute_grid_box,
                extract_pdbqt_atom_types,
                write_dpf,
                write_gpf,
            )

            box = compute_grid_box(ligand, padding=padding, npts=npts,
                                   spacing=spacing)
            atom_types = extract_pdbqt_atom_types(ligand)
            gpf = write_gpf(ligand, receptor_file=receptor_file, box=box,
                            output_path=ligand.parent / "r.gpf")
            dpf = write_dpf(ligand, receptor_file=receptor_file,
                            output_path=ligand.parent / "r.dpf",
                            ga_run=ga_run, ga_pop_size=ga_pop)
            return box, atom_types, gpf, dpf, ga_run, ga_pop

        def render(payload: Any) -> None:
            box, atom_types, gpf, dpf, used_run, used_pop = payload
            self._dock_box = box
            centre = (box.center_x, box.center_y, box.center_z)
            size = box.size
            self._dock_grid_text.delete("1.0", "end")
            self._dock_grid_text.insert("1.0", "\n".join([
                (f"box centre     ({centre[0]:.3f}, {centre[1]:.3f}, "
                 f"{centre[2]:.3f}) Å"),
                (f"box size       ({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}) Å "
                 f"· npts {box.npts} at {box.spacing:g} Å"),
                f"ligand types   {' '.join(atom_types)}",
                f"r.gpf          {gpf}",
                f"r.dpf          {dpf}   (ga_run {used_run} · pop {used_pop})",
            ]))

        self._start_job(f"grid box for {ligand.name}", work, render)

    # -- docking results card -------------------------------------------- #
    def _dock_scan_campaign(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas",
                                                            "the campaign scan"):
            return
        path = filedialog.askdirectory(title="Choose the campaign root to scan")
        if path:
            self._dock_scan_campaign_path(Path(path))

    def _dock_load_example_campaign(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas",
                                                            "the campaign scan"):
            return
        path = EXAMPLES_DIR / "docking_campaign"
        if not path.is_dir():
            messagebox.showwarning(
                "Missing example", "examples/data/docking_campaign was not found."
            )
            return
        self._dock_scan_campaign_path(path)

    def _dock_scan_campaign_path(self, root: Path) -> None:
        def work() -> Any:
            from pharmaml.docking import scan_docking_directory

            return scan_docking_directory(root)

        def render(frame: Any) -> None:
            self._dock_results_frame = frame
            self._dock_results_source = "campaign"
            tree = self._dock_results_tree
            tree.delete(*tree.get_children())
            columns = ["folder", "n_clusters", "top_lowest_energy",
                       "best_lowest_energy", "status"]
            tree["columns"] = columns
            for column in columns:
                tree.heading(column, text=column)
                tree.column(column, width=150 if column == "folder" else 130,
                            minwidth=80, stretch=(column == "folder"))
            for _, row in frame.iterrows():
                values = [
                    _cell_text(None if (isinstance(row[column], float)
                                        and math.isnan(row[column]))
                               else row[column], limit=48)
                    for column in columns
                ]
                tree.insert("", "end", values=values)
            ok = int((frame["status"] == "OK").sum())
            self._dock_results_info.configure(
                text=(f"{len(frame)} folders scanned · {ok} OK · "
                      f"{len(frame) - ok} incomplete — {root.name}"),
                style="Field.TLabel",
            )

        self._start_job(f"scanning {root.name}", work, render)

    def _dock_open_dlg(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas",
                                                            "DLG parsing"):
            return
        path = filedialog.askopenfilename(
            title="Open an AutoDock4 DLG log",
            filetypes=[("AutoDock DLG log", "*.dlg"), ("All files", "*.*")],
        )
        if path:
            self._dock_parse_dlg_path(Path(path))

    def _dock_open_example_dlg(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas",
                                                            "DLG parsing"):
            return
        path = EXAMPLES_DIR / "docking_campaign" / "01_aspirin" / "r.dlg"
        if not path.is_file():
            messagebox.showwarning(
                "Missing example",
                "examples/data/docking_campaign/01_aspirin/r.dlg was not found.",
            )
            return
        self._dock_parse_dlg_path(path)

    def _dock_parse_dlg_path(self, path: Path) -> None:
        heavy_raw = self._dock_heavyatoms_entry.get().strip()
        heavy_atoms: int | None = None
        if heavy_raw:
            try:
                heavy_atoms = parse_int(heavy_raw, "heavy atoms", minimum=1,
                                        maximum=500)
            except ValueError as exc:
                messagebox.showerror("Input", str(exc))
                return

        def work() -> Any:
            import pandas as pd

            from pharmaml.docking import (
                ligand_efficiency_series,
                parse_dlg_cluster_table,
                parse_dlg_summary,
            )

            clusters = parse_dlg_cluster_table(path)
            summary = parse_dlg_summary(path)
            frame = pd.DataFrame([
                {"rank": cluster.rank, "lowest_energy": cluster.lowest_energy,
                 "mean_energy": cluster.mean_energy,
                 "num_in_cluster": cluster.num_in_cluster}
                for cluster in clusters
            ])
            if heavy_atoms is not None:
                frame["n_heavy"] = heavy_atoms
                frame["LE (kcal/mol/atom)"] = ligand_efficiency_series(
                    frame, affinity_col="lowest_energy", atoms_col="n_heavy"
                )
            return frame, summary, heavy_atoms

        def render(payload: Any) -> None:
            frame, summary, used_atoms = payload
            self._dock_results_frame = frame
            self._dock_results_source = "dlg"
            tree = self._dock_results_tree
            tree.delete(*tree.get_children())
            columns = [str(column) for column in frame.columns
                       if column != "n_heavy"]
            tree["columns"] = columns
            for column in columns:
                tree.heading(column, text=column)
                tree.column(column, width=170 if column.startswith("LE") else 120,
                            minwidth=80, stretch=False)
            for row in frame.to_dict("records"):
                values = [
                    _cell_text(None if (isinstance(row[column], float)
                                        and math.isnan(row[column]))
                               else row[column], limit=40)
                    for column in columns
                ]
                tree.insert("", "end", values=values)
            info = (f"{summary['n_clusters']} clusters · top cluster rank "
                    f"{summary['top_rank']} at "
                    f"{summary['top_lowest_energy']:.2f} kcal/mol "
                    f"({summary['top_num_in_cluster']} poses) · largest cluster "
                    f"rank {summary['best_rank']} "
                    f"({summary['best_num_in_cluster']} poses)")
            if used_atoms is not None:
                info += f" · LE over {used_atoms} heavy atoms"
            self._dock_results_info.configure(text=info, style="Field.TLabel")

        self._start_job(f"parsing {path.name}", work, render)

    def _dock_open_vina(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas",
                                                            "Vina parsing"):
            return
        path = filedialog.askopenfilename(
            title="Open a Vina output",
            filetypes=[("Vina output", "*.pdbqt *.log *.txt"),
                       ("All files", "*.*")],
        )
        if path:
            self._dock_parse_vina_path(Path(path))

    def _dock_parse_vina_path(self, path: Path) -> None:
        def work() -> Any:
            import pandas as pd

            from pharmaml.docking import parse_vina_output

            poses = parse_vina_output(path)
            frame = pd.DataFrame([
                {"mode": pose.mode, "affinity": pose.affinity,
                 "rmsd_ub": pose.rmsd_ub, "rmsd_lb": pose.rmsd_lb}
                for pose in poses
            ])
            return frame

        def render(frame: Any) -> None:
            self._dock_results_frame = frame
            self._dock_results_source = "vina"
            tree = self._dock_results_tree
            tree.delete(*tree.get_children())
            columns = [str(column) for column in frame.columns]
            tree["columns"] = columns
            for column in columns:
                tree.heading(column, text=column)
                tree.column(column, width=120, minwidth=80, stretch=False)
            for row in frame.to_dict("records"):
                tree.insert(
                    "", "end",
                    values=[_cell_text(row[column]) for column in columns],
                )
            best = float(frame["affinity"].min()) if len(frame) else float("nan")
            self._dock_results_info.configure(
                text=f"{len(frame)} modes · best affinity {best:.2f} kcal/mol "
                     f"— {path.name}",
                style="Field.TLabel",
            )

        self._start_job(f"parsing {path.name}", work, render)

    def _dock_save_results(self) -> None:
        if self._dock_results_frame is None:
            messagebox.showinfo("Nothing to save",
                                "Scan a campaign or parse a log first.")
            return
        if not self._require_pharmaml():
            return
        path = filedialog.asksaveasfilename(
            title="Save the docking results", defaultextension=".csv",
            filetypes=[("CSV table", "*.csv")],
        )
        if not path:
            return
        try:
            from pharmaml.utils.io import write_table

            written = write_table(self._dock_results_frame, path)
            messagebox.showinfo("Saved", f"Results written to\n{written}")
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not save", exc)

    # -- campaign runner card -------------------------------------------- #
    def _dock_choose_campaign_root(self) -> None:
        path = filedialog.askdirectory(title="Choose the AutoDock4 campaign root")
        if path:
            self._dock_campaign_root = Path(path)
            self._dock_campaign_info.configure(text=f"campaign root: {path}")

    def _dock_open_vina_receptor(self) -> None:
        path = filedialog.askopenfilename(
            title="Open the receptor PDBQT",
            filetypes=[("PDBQT receptor", "*.pdbqt"), ("All files", "*.*")],
        )
        if path:
            self._dock_vina_receptor = Path(path)
            self._dock_refresh_vina_info()

    def _dock_open_vina_ligand(self) -> None:
        path = filedialog.askopenfilename(
            title="Open the ligand PDBQT",
            filetypes=[("PDBQT ligand", "*.pdbqt"), ("All files", "*.*")],
        )
        if path:
            self._dock_vina_ligand = Path(path)
            self._dock_refresh_vina_info()

    def _dock_refresh_vina_info(self) -> None:
        receptor = (self._dock_vina_receptor.name
                    if self._dock_vina_receptor else "—")
        ligand = (self._dock_vina_ligand.name
                  if self._dock_vina_ligand else "—")
        self._dock_vina_info.configure(
            text=f"receptor {receptor} · ligand {ligand}"
        )

    def _dock_missing_engine_dialog(self, engine: str) -> None:
        binaries = ", ".join(docking_engine_binaries(engine))
        if engine == "AutoDock4 campaign":
            hint = (
                f"AutoDock4 campaign docking needs the {binaries} binaries on "
                "PATH (Scripps AutoDock Tools, or conda-forge):\n\n"
                "    conda install -c conda-forge autodock\n\n"
                "Install them, reopen this terminal, then run again."
            )
        else:
            hint = (
                f"Vina docking needs the {binaries} binary on PATH (Scripps "
                "AutoDock Vina, or conda-forge):\n\n"
                "    conda install -c conda-forge vina\n\n"
                "Install it, reopen this terminal, then run again."
            )
        messagebox.showinfo("Docking engine not installed", hint)

    def _dock_run(self) -> None:
        if not self._require_pharmaml():
            return
        engine = self._dock_engine_combo.get()
        binaries = docking_engine_binaries(engine)
        missing = [name for name in binaries if shutil.which(name) is None]
        if missing:
            # pre-empt pharmaml's own RuntimeError with a friendly hint
            self._dock_missing_engine_dialog(engine)
            return

        if engine == "AutoDock4 campaign":
            if self._dock_campaign_root is None:
                messagebox.showwarning(
                    "No campaign root",
                    "Choose the campaign root (the folder of prepared ligand "
                    "sub-directories) first.",
                )
                return
            try:
                workers = parse_int(self._dock_workers_entry.get(), "workers",
                                    minimum=1, maximum=64)
            except ValueError as exc:
                messagebox.showerror("Input", str(exc))
                return
            root = self._dock_campaign_root

            def work() -> Any:
                from pharmaml.docking import batch_run_directories

                return batch_run_directories(root, workers=workers)

            def render(results: Any) -> None:
                ok_runs = sum(1 for result in results if result.ok)
                failed_runs = len(results) - ok_runs
                total_seconds = sum(result.seconds for result in results)
                self._dock_run_summary.configure(
                    text=(f"{len(results)} folders docked · {ok_runs} ok · "
                          f"{failed_runs} failed · {total_seconds:.0f} s total "
                          f"— {root.name}"),
                    style="Field.TLabel",
                )

            self._start_job(f"AutoDock4 campaign ({root.name})", work, render)
            return

        # -- Vina single ligand ------------------------------------------ #
        if self._dock_vina_receptor is None or self._dock_vina_ligand is None:
            messagebox.showwarning(
                "Missing inputs", "Open both the receptor and ligand PDBQT first."
            )
            return
        try:
            center = parse_float_triple(self._dock_center_entry.get(), "centre")
            size = parse_float_triple(self._dock_size_entry.get(), "size")
            exhaustiveness = parse_int(self._dock_exhaust_entry.get(),
                                       "exhaustiveness", minimum=1, maximum=256)
        except ValueError as exc:
            messagebox.showerror("Input", str(exc))
            return
        receptor, ligand = self._dock_vina_receptor, self._dock_vina_ligand
        out = ligand.parent / "out.pdbqt"

        def work() -> Any:
            from pharmaml.docking import run_vina

            return run_vina(receptor, ligand, out=out, center=center, size=size,
                            exhaustiveness=exhaustiveness)

        def render(result: Any) -> None:
            state = "ok" if result.ok else f"FAILED — {result.error}"
            self._dock_run_summary.configure(
                text=(f"Vina run {state} · {result.seconds:.1f} s · "
                      f"output {out}"),
                style="Field.TLabel",
            )

        self._start_job(f"Vina run ({ligand.name})", work, render)

    # ------------------------------------------------------------------ #
    # Data & ML tab
    # ------------------------------------------------------------------ #
    def _build_ml_tab(self) -> None:
        page = self._new_page("Data & ML")
        self._ml_page = page

        data = make_card(page.grid, "Dataset", span=2, row=0, column=0,
                         subtitle="Any CSV the toolkit's read_table can sniff, or a bundled example.")
        open_btn = self._action_button(data, "Open CSV…", self._ml_open_csv)
        open_btn.grid(row=0, column=0, sticky="w")
        example_files = [
            name for name in (
                "qsar_regression.csv", "qsar_classification.csv",
                "qsar_multiclass.csv", "cellline_screen.csv", "pk_profile.csv",
            ) if (EXAMPLES_DIR / name).is_file()
        ]
        self._example_combo = ttk.Combobox(data, state="readonly", width=26,
                                           values=example_files)
        if example_files:
            self._example_combo.set(example_files[0])
        self._example_combo.grid(row=0, column=1, sticky="w", padx=(12, 0))
        load_btn = self._action_button(data, "Load example", self._ml_load_example)
        load_btn.grid(row=0, column=2, sticky="w", padx=(12, 0))
        self._ml_info = ttk.Label(data, text="no dataset loaded", style="CardMuted.TLabel")
        self._ml_info.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        preview = make_card(page.grid, "Preview (first 12 rows)", span=2, row=1,
                            column=0)
        tree_frame = ttk.Frame(preview, style="Card.TFrame")
        grid_append(preview, tree_frame, sticky="nsew", weight=True)
        self._preview_tree = ttk.Treeview(tree_frame, show="headings", height=8)
        preview_bar_x = ttk.Scrollbar(tree_frame, orient="horizontal",
                                      command=self._preview_tree.xview)
        preview_bar_y = ttk.Scrollbar(tree_frame, orient="vertical",
                                      command=self._preview_tree.yview)
        self._preview_tree.configure(xscrollcommand=preview_bar_x.set,
                                     yscrollcommand=preview_bar_y.set)
        self._preview_tree.pack(side="left", fill="both", expand=True)
        preview_bar_y.pack(side="right", fill="y")
        preview_bar_x.pack(side="bottom", fill="x")

        config = make_card(page.grid, "Train a model", span=2, row=2, column=0,
                           subtitle="Features = the numeric columns except the target "
                                    "(helpers like SMILES ids are dropped automatically).")
        self._target_combo = ttk.Combobox(config, state="readonly", width=26)
        form_row(config, "target column", self._target_combo)
        self._task_combo = ttk.Combobox(config, state="readonly", width=26,
                                        values=("auto", "regression", "classification"))
        self._task_combo.set("auto")
        form_row(config, "task", self._task_combo)
        self._model_combo = ttk.Combobox(config, state="readonly", width=26,
                                         values=("auto",))
        self._model_combo.set("auto")
        form_row(config, "model", self._model_combo)
        self._tune_var = tk.BooleanVar(value=False)
        tune_check = ttk.Checkbutton(config, text="tune hyperparameters (grid search, slower)",
                                     variable=self._tune_var)
        form_row(config, "options", tune_check)
        buttons = ttk.Frame(config, style="Card.TFrame")
        buttons.grid(row=config.grid_size()[1], column=0, columnspan=2, sticky="w",
                     pady=(8, 0))
        train_btn = self._action_button(buttons, "Train & evaluate", self._ml_train,
                                        primary=True,
                                        tooltip="80/20 split, the toolkit's own preprocessing "
                                                "pipeline, then held-out metrics + figure.")
        train_btn.pack(side="left")
        compare_btn = self._action_button(buttons, "Compare models (3-fold CV)",
                                          self._ml_compare,
                                          tooltip="Cross-validated leaderboard over the "
                                                  "available registry.")
        compare_btn.pack(side="left", padx=(10, 0))

        results = make_card(page.grid, "Results", span=2, row=3, column=0)
        self._ml_metrics = ttk.Frame(results, style="Card.TFrame")
        grid_append(results, self._ml_metrics, sticky="ew")
        ttk.Label(self._ml_metrics, text="train a model to see metrics",
                  style="CardMuted.TLabel").pack(anchor="w")
        self._ml_host = self._figure_host(results, height=330)

        board = make_card(page.grid, "Model leaderboard (cross-validation)", span=2,
                          row=4, column=0)
        board_frame = ttk.Frame(board, style="Card.TFrame")
        grid_append(board, board_frame, sticky="nsew", weight=True)
        self._board_tree = ttk.Treeview(board_frame, show="headings", height=6)
        board_bar = ttk.Scrollbar(board_frame, orient="vertical",
                                  command=self._board_tree.yview)
        self._board_tree.configure(yscrollcommand=board_bar.set)
        self._board_tree.pack(side="left", fill="both", expand=True)
        board_bar.pack(side="right", fill="y")

    def _ml_open_csv(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas", "CSV loading"):
            return
        path = filedialog.askopenfilename(
            title="Open a dataset",
            filetypes=[("Data files", "*.csv *.tsv *.txt"), ("All files", "*.*")],
        )
        if path:
            self._ml_load_path(Path(path))

    def _ml_load_example(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas", "CSV loading"):
            return
        name = self._example_combo.get()
        path = EXAMPLES_DIR / name
        if not path.is_file():
            messagebox.showwarning("Missing example", f"{name} was not found.")
            return
        self._ml_load_path(path)

    def _ml_load_path(self, path: Path) -> None:
        try:
            from pharmaml.utils.io import read_table

            frame = read_table(path)
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not read the dataset", exc)
            return
        if frame.empty:
            messagebox.showwarning("Empty dataset", f"{path.name} has no rows.")
            return
        self._ml_frame = frame
        self._ml_path = path
        columns = [str(column) for column in frame.columns]
        self._preview_tree.delete(*self._preview_tree.get_children())
        self._preview_tree["columns"] = columns
        for column in columns:
            self._preview_tree.heading(column, text=column)
            self._preview_tree.column(column, width=96, minwidth=64, stretch=True)
        for _, row in frame.head(12).iterrows():
            self._preview_tree.insert(
                "", "end", values=[_cell_text(value) for value in row.tolist()]
            )
        self._target_combo["values"] = columns
        self._target_combo.set(columns[-1])
        self._ml_info.configure(
            text=f"{path.name} — {len(frame):,} rows × {len(columns)} columns"
        )
        self._refresh_model_choices(self._infer_task(columns[-1]))

    def _infer_task(self, target: str) -> str:
        """Classify the target column as a regression or a classification label."""
        if self._ml_frame is None:
            return "regression"
        column = self._ml_frame[target]
        if str(column.dtype) == "object":
            return "classification"
        if column.nunique() <= 6:
            return "classification"
        return "regression"

    def _refresh_model_choices(self, task: str) -> None:
        try:
            from pharmaml.ml.models import list_models

            names = [entry["name"] for entry in list_models(task, only_available=True)]
        except Exception:  # noqa: BLE001 - fall back to a safe fast pair
            names = ["ridge"] if task == "regression" else ["logistic_regression"]
        if not names:
            names = ["ridge"] if task == "regression" else ["logistic_regression"]
        self._model_combo["values"] = ["auto"] + names
        self._model_combo.set("auto")

    def _ml_train(self) -> None:
        if not self._require_pharmaml():
            return
        if not self._require("pandas", "model training") or not self._require(
                "sklearn", "model training"
        ):
            return
        if self._ml_frame is None:
            messagebox.showwarning("No dataset", "Open a CSV or load an example first.")
            return
        target = self._target_combo.get()
        if not target:
            messagebox.showwarning("No target", "Pick the column to predict.")
            return
        task = self._task_combo.get()
        if task == "auto":
            task = self._infer_task(target)
        model_name = self._model_combo.get()
        if model_name == "auto":
            model_name = "ridge" if task == "regression" else "logistic_regression"
        tune = bool(self._tune_var.get())

        def work() -> Any:
            import pandas as pd

            from pharmaml.ml import evaluate_model, train_model
            from pharmaml.ml.preprocessing import make_train_test_split

            frame = self._ml_frame
            y = frame[target]
            numeric = [
                column for column in frame.columns
                if column != target and pd.api.types.is_numeric_dtype(frame[column])
            ]
            dropped = [column for column in frame.columns
                       if column != target and column not in numeric]
            x_train, x_test, y_train, y_test = make_train_test_split(
                frame[numeric], y, random_state=42
            )
            trained = train_model(
                model_name, task, x_train, y_train, tune=tune, cv=5, n_jobs=1
            )
            report = evaluate_model(
                trained.model, x_test, y_test, task,
                keep_predictions=(task == "regression"),
            )
            return trained, report, task, dropped, y_test

        def render(payload: Any) -> None:
            trained, report, task, dropped, y_test = payload
            self._fill_ml_metrics(trained, report, task, dropped)
            if not self.deps.get("matplotlib"):
                return
            try:
                from pharmaml.viz import plot_confusion_matrix, plot_predicted_vs_actual

                if task == "regression":
                    figure = plot_predicted_vs_actual(
                        y_test,
                        report.predictions,
                        metrics={
                            "R²": report.r2,
                            "RMSE": report.rmse,
                            "MAE": report.mae,
                        },
                    )
                else:
                    figure = plot_confusion_matrix(
                        report.confusion, labels=report.labels
                    )
                self._render_figure(self._ml_host, figure)
            except Exception as exc:  # noqa: BLE001
                self._show_error("Plot failed", exc)

        self._start_job(f"training {model_name} ({task})", work, render)

    def _fill_ml_metrics(self, trained: Any, report: Any, task: str,
                         dropped: list[str]) -> None:
        for child in self._ml_metrics.winfo_children():
            child.destroy()
        cv_line = "no CV"
        if trained.cv_mean is not None:
            spread = trained.cv_std if trained.cv_std is not None else 0.0
            cv_line = f"{trained.cv_mean:.3f} ± {spread:.3f}"
        if task == "regression":
            pairs = [
                ("model", f"{trained.name} ({task})"),
                ("R²", format_number(getattr(report, "r2", float("nan")))),
                ("RMSE", format_number(report.rmse)),
                ("MAE", format_number(report.mae)),
                ("CV score", cv_line),
                ("fit time", f"{trained.fit_seconds:.2f} s"),
            ]
        else:
            auc = getattr(report, "roc_auc", None)
            pairs = [
                ("model", f"{trained.name} ({task})"),
                ("accuracy", format_number(report.accuracy)),
                ("balanced accuracy", format_number(report.balanced_accuracy)),
                ("precision (macro)", format_number(report.precision_macro)),
                ("recall (macro)", format_number(report.recall_macro)),
                ("MCC", format_number(report.mcc)),
                ("ROC AUC", format_number(auc) if auc is not None else "n/a"),
                ("CV score", cv_line),
                ("fit time", f"{trained.fit_seconds:.2f} s"),
            ]
        if dropped:
            pairs.append(("dropped non-numeric", ", ".join(dropped)))
        for index, (name, value) in enumerate(pairs):
            ttk.Label(self._ml_metrics, text=name, style="CardMuted.TLabel").grid(
                row=index // 4, column=(index % 4) * 2, sticky="w", padx=(0, 6)
            )
            ttk.Label(self._ml_metrics, text=value, style="Field.TLabel",
                      font=self._mono).grid(
                row=index // 4, column=(index % 4) * 2 + 1, sticky="w", padx=(0, 18)
            )

    def _ml_compare(self) -> None:
        if not self._require_pharmaml():
            return
        if not self._require("pandas", "cross-validation") or not self._require(
                "sklearn", "cross-validation"
        ):
            return
        if self._ml_frame is None:
            messagebox.showwarning("No dataset", "Open a CSV or load an example first.")
            return
        target = self._target_combo.get()
        task = self._task_combo.get()
        if task == "auto":
            task = self._infer_task(target)

        def work() -> Any:
            import pandas as pd

            from pharmaml.ml import compare_models

            frame = self._ml_frame
            y = frame[target]
            numeric = [
                column for column in frame.columns
                if column != target and pd.api.types.is_numeric_dtype(frame[column])
            ]
            return compare_models(frame[numeric], y, task, cv=3, n_jobs=1)

        def render(board: Any) -> None:
            self._board_tree.delete(*self._board_tree.get_children())
            columns = [str(column) for column in board.columns]
            self._board_tree["columns"] = columns
            for column in columns:
                self._board_tree.heading(column, text=column)
                self._board_tree.column(column, width=140, minwidth=80, stretch=True)
            for _, row in board.iterrows():
                self._board_tree.insert(
                    "", "end", values=[_cell_text(value, limit=30) for value in row.tolist()]
                )

        self._start_job(f"comparing models ({task})", work, render)

    # ------------------------------------------------------------------ #
    # Generative tab
    # ------------------------------------------------------------------ #
    def _build_generative_tab(self) -> None:
        page = self._new_page("Generative")

        corpus = make_card(page.grid, "Corpus / seed molecules", row=0, column=0,
                           subtitle="One SMILES per line. Empty text = the bundled "
                                    "244-molecule drug-like corpus.")
        self._corpus_text = _grid_text(corpus, height=9, mono=self._mono)
        grid_append(corpus, self._corpus_text, sticky="nsew", weight=True)
        corpus_buttons = ttk.Frame(corpus, style="Card.TFrame")
        grid_append(corpus, corpus_buttons, sticky="w", pady=(6, 0))
        example_btn = self._action_button(corpus_buttons, "Load example corpus",
                                          self._load_example_corpus,
                                          tooltip="The 12 seed molecules shipped in examples/data.")
        example_btn.pack(side="left")
        clear_btn = self._action_button(corpus_buttons, "Clear (built-in corpus)",
                                        lambda: self._corpus_text.delete("1.0", "end"))
        clear_btn.pack(side="left", padx=(10, 0))
        self._seed_mode_var = tk.BooleanVar(value=False)
        seed_check = ttk.Checkbutton(
            corpus,
            text="treat the lines as seed molecules (seeded mode; otherwise they are "
                 "the sampling pool)",
            variable=self._seed_mode_var,
        )
        grid_append(corpus, seed_check, pady=(4, 0))

        controls = make_card(page.grid, "Objective & evolution", row=0, column=1)
        self._objective_combo = ttk.Combobox(controls, state="readonly", width=26,
                                             values=_OBJECTIVE_KINDS)
        self._objective_combo.set(_OBJECTIVE_KINDS[0])
        self._objective_combo.bind("<<ComboboxSelected>>", self._objective_hint)
        form_row(controls, "objective", self._objective_combo)
        self._obj_lo = ttk.Entry(controls, width=12)
        self._obj_lo.insert(0, "1.0")
        form_row(controls, "window low", self._obj_lo)
        self._obj_hi = ttk.Entry(controls, width=12)
        self._obj_hi.insert(0, "3.0")
        form_row(controls, "window high", self._obj_hi)
        self._objective_hint_label = ttk.Label(controls, text="", style="CardMuted.TLabel",
                                               wraplength=280, justify="left")
        self._objective_hint_label.grid(row=controls.grid_size()[1], column=0,
                                        columnspan=2, sticky="w", pady=(2, 4))
        self._objective_hint()
        self._pop_size = ttk.Spinbox(controls, from_=8, to=96, increment=4, width=10)
        self._pop_size.set("24")
        form_row(controls, "population", self._pop_size)
        self._generations = ttk.Spinbox(controls, from_=4, to=48, increment=4, width=10)
        self._generations.set("16")
        form_row(controls, "generations", self._generations)
        self._ga_seed = ttk.Entry(controls, width=12)
        self._ga_seed.insert(0, "42")
        form_row(controls, "random seed", self._ga_seed)
        self._mutation_combo = ttk.Combobox(controls, state="readonly", width=18,
                                            values=("uniform", "scaffold", "topology"))
        self._mutation_combo.set("topology")
        form_row(controls, "mutation", self._mutation_combo)
        self._crossover_combo = ttk.Combobox(controls, state="readonly", width=18,
                                             values=("splice", "scaffold_hop"))
        self._crossover_combo.set("splice")
        form_row(controls, "crossover", self._crossover_combo)
        self._operator_combo = ttk.Combobox(controls, state="readonly", width=18,
                                            values=("uniform", "adaptive"))
        self._operator_combo.set("uniform")
        form_row(controls, "operator selection", self._operator_combo)
        run_btn = self._action_button(controls, "Run evolution", self._run_evolution,
                                      primary=True,
                                      tooltip="Runs in a worker thread; results appear below "
                                              "with the leaderboard (and the learning curve "
                                              "when adaptive).")
        run_btn.grid(row=controls.grid_size()[1], column=0, columnspan=2, sticky="w",
                     pady=(8, 0))

        results = make_card(page.grid, "Results", span=2, row=1, column=0)
        self._ga_summary = ttk.Label(results, text="no run yet — pick an objective and press "
                                                   "Run evolution", style="CardMuted.TLabel",
                                    wraplength=640)
        grid_append(results, self._ga_summary)
        tree_frame = ttk.Frame(results, style="Card.TFrame")
        grid_append(results, tree_frame, sticky="nsew", weight=True)
        self._ga_tree = ttk.Treeview(tree_frame, show="headings", height=8)
        ga_bar = ttk.Scrollbar(tree_frame, orient="vertical", command=self._ga_tree.yview)
        self._ga_tree.configure(yscrollcommand=ga_bar.set)
        self._ga_tree.pack(side="left", fill="both", expand=True)
        ga_bar.pack(side="right", fill="y")
        self._ga_stats_text = _grid_text(results, height=7, mono=self._mono)
        self._ga_stats_text.insert("1.0", "operator statistics appear here on adaptive runs")
        grid_append(results, self._ga_stats_text, sticky="ew", pady=(4, 0))
        stats_buttons = ttk.Frame(results, style="Card.TFrame")
        grid_append(results, stats_buttons, sticky="w", pady=(4, 0))
        save_btn = self._action_button(stats_buttons, "Save leaderboard as CSV…",
                                       self._save_ga_leaderboard)
        save_btn.pack(side="left")
        self._ga_host = self._figure_host(results, height=340)

        # -- Pareto front card (100x-58) ------------------------------ #
        pareto = make_card(
            page.grid, "Pareto front", span=2, row=2, column=0,
            subtitle="Molecules no other candidate dominates — the trade-offs "
                     "a blended leaderboard hides. Filled after every "
                     "multi-objective run."
        )
        self._pareto_summary = ttk.Label(
            pareto, text="no multi-objective run yet — pick 'QED + LogP blend' or "
                         "'Multi-objective (QED × MW × TPSA)' and press Run evolution",
            style="CardMuted.TLabel", wraplength=640, justify="left",
        )
        grid_append(pareto, self._pareto_summary)
        pareto_tree_frame = ttk.Frame(pareto, style="Card.TFrame")
        grid_append(pareto, pareto_tree_frame, sticky="nsew", weight=True)
        self._pareto_tree = ttk.Treeview(pareto_tree_frame, show="headings", height=6)
        pareto_bar = ttk.Scrollbar(pareto_tree_frame, orient="vertical",
                                   command=self._pareto_tree.yview)
        self._pareto_tree.configure(yscrollcommand=pareto_bar.set)
        self._pareto_tree.pack(side="left", fill="both", expand=True)
        pareto_bar.pack(side="right", fill="y")
        pareto_buttons = ttk.Frame(pareto, style="Card.TFrame")
        grid_append(pareto, pareto_buttons, sticky="w", pady=(6, 0))
        ttk.Label(pareto_buttons, text="plot", style="Field.TLabel").pack(side="left")
        self._pareto_x_combo = ttk.Combobox(pareto_buttons, state="readonly", width=14,
                                            values=())
        self._pareto_x_combo.pack(side="left", padx=(6, 3))
        ttk.Label(pareto_buttons, text="×", style="CardMuted.TLabel").pack(side="left")
        self._pareto_y_combo = ttk.Combobox(pareto_buttons, state="readonly", width=14,
                                            values=())
        self._pareto_y_combo.pack(side="left", padx=(3, 6))
        plot_front_btn = self._action_button(
            pareto_buttons, "Plot front", self._plot_pareto_front,
            tooltip="Scatter of the two selected objectives: dominated molecules "
                    "muted, the front highlighted and connected.",
        )
        plot_front_btn.pack(side="left")
        save_front_btn = self._action_button(pareto_buttons, "Save front as CSV…",
                                             self._save_pareto_front)
        save_front_btn.pack(side="left", padx=(10, 0))
        self._pareto_host = self._figure_host(pareto, height=300)

    def _objective_hint(self, _event: Any = None) -> None:
        kind = self._objective_combo.get()
        hints = {
            "QED drug-likeness": "QED weights eight drug-likeness properties; the window "
                                 "values are ignored.",
            "LogP window": "Scores 1 inside the [low, high] LogP window, falling off "
                           "outside.",
            "Molecular weight window": "Target a molecular-weight band (e.g. 250–450).",
            "TPSA window": "Target a polar-surface-area band (e.g. 40–90 Å²).",
            "QED + LogP blend": "0.7 × QED + 0.3 × the LogP window.",
            "Multi-objective (QED \u00d7 MW \u00d7 TPSA)":
                "Equal-weight triple (MW 150–450, TPSA 30–90); the Pareto "
                "front card below shows the trade-offs the blend hides.",
        }
        self._objective_hint_label.configure(text=hints.get(kind, ""))

    def _load_example_corpus(self) -> None:
        path = EXAMPLES_DIR / "seed_molecules.csv"
        if not path.is_file():
            messagebox.showinfo("Example missing",
                                "examples/data/seed_molecules.csv was not found.")
            return
        with path.open(newline="", encoding="utf-8") as handle:
            smiles = [
                (row.get("smiles") or "").strip() for row in csv.DictReader(handle)
            ]
        smiles = [entry for entry in smiles if entry]
        self._corpus_text.delete("1.0", "end")
        self._corpus_text.insert("1.0", "\n".join(smiles))

    def _run_evolution(self) -> None:
        if not self._require_pharmaml() or not self._require("rdkit", "the GA"):
            return
        try:
            kind = self._objective_combo.get()
            lo = parse_float(self._obj_lo.get(), "window low")
            hi = parse_float(self._obj_hi.get(), "window high")
            if lo > hi:
                raise ValueError("window low exceeds window high")
            population = parse_int(self._pop_size.get(), "population", minimum=4)
            generations = parse_int(self._generations.get(), "generations", minimum=1)
            seed = parse_int(self._ga_seed.get(), "seed", minimum=0)
            mutation = self._mutation_combo.get()
            crossover = self._crossover_combo.get()
            operator_selection = self._operator_combo.get()
        except ValueError as exc:
            messagebox.showerror("Input", str(exc))
            return
        lines = parse_smiles_lines(self._corpus_text.get("1.0", "end"))
        seed_mode = bool(self._seed_mode_var.get())
        want_plot = operator_selection == "adaptive" and self.deps.get("matplotlib")

        def work() -> Any:
            from pharmaml.generative import GeneticOptimizer

            objective = build_objective(kind, lo, hi)
            optimizer = GeneticOptimizer(
                objective,
                population_size=population,
                generations=generations,
                mutation=mutation,
                crossover=crossover,
                operator_selection=operator_selection,
                seed=seed,
            )
            if not lines:
                return optimizer.optimize()
            if seed_mode:
                return optimizer.optimize(seed_smiles=lines)
            return optimizer.optimize(corpus=lines)

        def render(result: Any) -> None:
            self._ga_result = result
            elapsed = result.metadata.get("elapsed_s")
            timing = f" · {elapsed:.1f} s" if isinstance(elapsed, (int, float)) else ""
            mode = "seeded" if seed_mode and lines else (
                "custom corpus" if lines else "built-in corpus"
            )
            self._ga_summary.configure(
                text=(f"best {result.best_score:.3f} — {result.best}   "
                      f"({len(result.smiles)} unique molecules, {mode}{timing})"),
                style="Field.TLabel",
            )
            frame = result.to_dataframe()
            self._ga_tree.delete(*self._ga_tree.get_children())
            columns = [str(column) for column in frame.columns]
            self._ga_tree["columns"] = columns
            for column in columns:
                self._ga_tree.heading(column, text=column)
                width = 320 if column == "smiles" else 96
                self._ga_tree.column(column, width=width, stretch=(column == "smiles"))
            for _, row in frame.head(20).iterrows():
                self._ga_tree.insert(
                    "", "end",
                    values=[_cell_text(value, limit=60) for value in row.tolist()],
                )
            stats = _format_operator_stats(result.metadata)
            self._ga_stats_text.delete("1.0", "end")
            self._ga_stats_text.insert("1.0", stats)
            if want_plot:
                try:
                    from pharmaml.viz import plot_operator_trajectory

                    figure = plot_operator_trajectory(result)
                    self._render_figure(self._ga_host, figure)
                except Exception as exc:  # noqa: BLE001
                    self._show_error("Trajectory plot failed", exc)
            else:
                self._close_embedded(self._ga_host)
                for child in self._ga_host.winfo_children():
                    child.destroy()
            self._render_pareto(result)

        label = f"evolving {generations} generations ({operator_selection})"
        self._start_job(label, work, render)

    def _save_ga_leaderboard(self) -> None:
        if self._ga_result is None:
            messagebox.showinfo("Nothing to save", "Run an evolution first.")
            return
        if not self._require_pharmaml():
            return
        path = filedialog.asksaveasfilename(
            title="Save leaderboard", defaultextension=".csv",
            filetypes=[("CSV table", "*.csv")],
        )
        if not path:
            return
        try:
            from pharmaml.utils.io import write_table

            written = write_table(self._ga_result.to_dataframe(), path)
            messagebox.showinfo("Saved", f"Leaderboard written to\n{written}")
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not save", exc)

    # -- the Pareto front card (100x-58) ------------------------------- #
    def _render_pareto(self, result: Any) -> None:
        """Fill the Pareto-front card from a finished run.

        A single-objective run carries no per-objective breakdown, so the
        card explains itself instead; a MultiObjective run gets the audit
        line, the front table, the plottable objective pair and — when
        matplotlib is available — the auto-drawn trade-off scatter.
        """
        self._pareto_front = None
        self._pareto_result = None
        self._pareto_tree.delete(*self._pareto_tree.get_children())
        self._pareto_x_combo.set("")
        self._pareto_y_combo.set("")
        self._pareto_x_combo["values"] = ()
        self._pareto_y_combo["values"] = ()
        self._close_embedded(self._pareto_host)
        for child in self._pareto_host.winfo_children():
            child.destroy()
        if not any(result.breakdown):
            self._pareto_summary.configure(
                text="this run optimises a single objective — the Pareto front "
                     "needs a multi-objective run ('QED + LogP blend' or "
                     "'Multi-objective (QED × MW × TPSA)')",
                style="CardMuted.TLabel",
            )
            return
        try:
            from pharmaml.generative import pareto_front

            front = pareto_front(result)
        except Exception as exc:  # noqa: BLE001 - surfaced inline, not modal
            self._pareto_summary.configure(
                text=f"Pareto front unavailable: {exc}", style="CardMuted.TLabel")
            return
        self._pareto_front = front
        self._pareto_result = result
        summary = str(front)
        deepest = max(
            (result.smiles.index(smiles) for smiles in front.smiles), default=0,
        ) + 1
        if deepest > 1:
            summary += (
                "\ndeepest front member sits at leaderboard rank "
                f"{deepest} of {len(result)} — a trade-off the blended ranking hides"
            )
        self._pareto_summary.configure(text=summary, style="Field.TLabel")
        frame = front.to_dataframe()
        self._pareto_tree.delete(*self._pareto_tree.get_children())
        columns = [str(column) for column in frame.columns]
        self._pareto_tree["columns"] = columns
        for column in columns:
            self._pareto_tree.heading(column, text=column)
            width = 320 if column == "smiles" else 96
            self._pareto_tree.column(column, width=width, stretch=(column == "smiles"))
        for _, row in frame.head(24).iterrows():
            self._pareto_tree.insert(
                "", "end",
                values=[_cell_text(value, limit=60) for value in row.tolist()],
            )
        self._pareto_x_combo["values"] = list(front.objectives)
        self._pareto_y_combo["values"] = list(front.objectives)
        self._pareto_x_combo.set(front.objectives[0])
        self._pareto_y_combo.set(front.objectives[1])
        if self.deps.get("matplotlib"):
            self._plot_pareto_front()

    def _plot_pareto_front(self) -> None:
        """Draw the trade-off scatter for the selected objective pair."""
        if self._pareto_result is None or self._pareto_front is None:
            messagebox.showinfo("Nothing to plot",
                                "Run a multi-objective evolution first.")
            return
        if not self._require("matplotlib", "the Pareto plot"):
            return
        x = self._pareto_x_combo.get()
        y = self._pareto_y_combo.get()
        try:
            from pharmaml.viz import plot_pareto_front

            figure = plot_pareto_front(self._pareto_result, x=x, y=y)
            self._render_figure(self._pareto_host, figure)
        except Exception as exc:  # noqa: BLE001
            self._show_error("Pareto plot failed", exc)

    def _save_pareto_front(self) -> None:
        """Write the front table of the last multi-objective run as CSV."""
        if self._pareto_front is None:
            messagebox.showinfo("Nothing to save",
                                "Run a multi-objective evolution first.")
            return
        if not self._require_pharmaml():
            return
        path = filedialog.asksaveasfilename(
            title="Save Pareto front", defaultextension=".csv",
            filetypes=[("CSV table", "*.csv")],
        )
        if not path:
            return
        try:
            from pharmaml.utils.io import write_table

            written = write_table(self._pareto_front.to_dataframe(), path)
            messagebox.showinfo("Saved", f"Pareto front written to\n{written}")
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not save", exc)


def _format_operator_stats(metadata: dict[str, Any]) -> str:
    """Render ``metadata['operator_stats']`` as a compact text table."""
    stats = metadata.get("operator_stats") or {}
    lines: list[str] = []
    for level in ("level1", "geometry", "mutation"):
        operators = stats.get(level)
        if not operators:
            continue
        lines.append(f"[{level}]")
        for name, info in operators.items():
            draws = info.get("draws", 0)
            successes = info.get("successes", 0)
            probability = info.get("final_probability", 0.0)
            lines.append(
                f"  {name:<18} draws {draws:>4}   successes {successes:>4}   "
                f"p {probability:.3f}"
            )
    if not lines:
        return "operator statistics appear here on adaptive runs"
    lines.insert(0, "operator statistics (adaptive run)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# command line entry points
# --------------------------------------------------------------------------- #
def print_banner(app: PlaygroundApp) -> None:
    """Explain the launch state on stdout (the VSCode terminal)."""
    print("=" * 66)
    print(f"  PharmaML Playground {__version__} — window is open")
    print("=" * 66)
    if app.pharmaml is not None:
        print(f"  pharmaml {app.pharmaml.__version__} loaded")
    else:
        print("  pharmaml NOT loaded:", app.pharmaml_error)
        print("  fix:", _PHARMAML_HINT.replace("\n", " "))
    missing = [
        label for module, label, _purpose, _pip in _DEPENDENCIES
        if not app.deps.get(module)
    ]
    if missing:
        print(f"  optional dependencies missing: {', '.join(missing)}")
        print("  (the tabs that need them explain their pip line when you use them)")
    else:
        print("  all optional dependencies installed — every tab fully armed")
    print("  close the window or press Ctrl+Q to exit.")
    print("=" * 66)


def self_test() -> int:
    """Build the whole UI, pump the event loop, visit every tab, exit.

    Runs headlessly under ``xvfb-run``; used by CI-style checks::

        xvfb-run -a python playground.py --self-test

    The build is fully deterministic: ``persist_session=False`` keeps
    the run from reading -- and, worse, re-writing -- the developer's
    real ``~/.pharmaml/playground-session.json``.
    """
    app = PlaygroundApp(persist_session=False)
    for _ in range(4):
        app.update_idletasks()
        app.update()
    for index in range(app._book.index("end")):
        app._book.select(index)
        app.update()
    app.update()
    app.destroy()
    print("self-test: UI built, every tab visited, no errors")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and start the playground (or its helpers)."""
    parser = argparse.ArgumentParser(
        prog="playground.py",
        description="PharmaML Playground — the pharmaml toolkit behind one window.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="build the UI, pump the event loop and exit (headless check)",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="print a dependency report and exit",
    )
    parser.add_argument(
        "--version", action="store_true", help="print versions and exit"
    )
    args = parser.parse_args(argv)

    if args.version:
        print(f"playground {__version__}")
        try:
            module = load_pharmaml()
            print(f"pharmaml {module.__version__}")
        except Exception as exc:  # noqa: BLE001
            print(f"pharmaml unavailable ({exc})")
        print(f"toolkit v{__version__}")
        return 0

    if args.probe:
        deps = probe_dependencies()
        print(f"playground {__version__} · python {sys.version.split()[0]}")
        print(f"toolkit root: {TOOLKIT_ROOT}")
        for module, label, purpose, pip in _DEPENDENCIES:
            state = "ok" if deps.get(module) else f"MISSING (pip install {pip})"
            print(f"  {label:<14} {state:<34} {purpose}")
        print(f"  {'tkinter':<14} {'ok' if tk is not None else 'MISSING'}")
        try:
            module = load_pharmaml()
            print(f"  pharmaml      ok ({module.__version__})")
        except Exception as exc:  # noqa: BLE001
            print(f"  pharmaml      FAILED — {exc}")
            return 1
        return 0

    if tk is None:
        print(
            "tkinter is not available in this Python build.\n"
            "Install the python3-tk system package (Debian/Ubuntu) or use an\n"
            "official python.org build, then run playground.py again.",
            file=sys.stderr,
        )
        return 2

    if args.self_test:
        return self_test()

    app = PlaygroundApp()
    print_banner(app)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
