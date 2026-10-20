"""PharmaML Playground -- the whole toolkit behind one window.

Run this file from the toolkit root (what you get when you unzip the
downloaded archive)::

    python playground.py

A tabbed graphical interface opens with every calculation domain of the
:mod:`pharmaml` package, no command-line knowledge required:

* **Chemistry**     -- IC50/Ki -> pIC50/pKi conversion, molar mass, the
  full RDKit descriptor table for any SMILES, Tanimoto similarity, the
  RISM-KH solvation card (ten 3D-RISM-KH-inspired descriptors per
  molecule -- solvation free energy split into Lennard-Jones and
  Coulomb parts, water-site distribution peaks, first-shell
  coordination numbers, excluded volume, dipole, charge asymmetry,
  over an embedded 3D geometry with the SPC / TIP3P site models),
  and the feature factory: the rar's own pipeline grown to an
  8-family, 10,000+-column descriptor + fingerprint matrix (the
  RISM-KH family included), ranked and cut to a top-K selection.
* **PK / PD**       -- half-life <-> k(e), accumulation, IV bolus decay
  and Emax curves (plotted), oral Cmax/tmax, Cockcroft-Gault creatinine
  clearance with renal dose adjustment, loading/maintenance dosing,
  BSA/weight-based and pediatric dose rules.
* **Cell Lines**    -- GDSC-style drug-response screens: load and audit
  a screen, pivot the sensitivity matrix, call sensitive vs resistant
  lines, convert ``LN_IC50`` <-> pIC50, and discover expression
  biomarkers for any compound (the EGFR / erlotinib story).
* **Docking**       -- molecular docking end to end: prepare a ligand
  library and a receptor into PDBQT files (with an auto-remove bad
  residues fallback for the broken-density PDBs Meeko's templates
  reject -- the library twin of ``--delete_bad_res``), compute the
  grid box and write the GPF/DPF configs, parse DLG / Vina results or
  scan a whole campaign, launch batch AutoDock4 or single Vina runs
  -- and dock a whole pasted SMILES ARRAY in one call: every SMILES
  prepared into its own campaign folder, the Vina or AutoDock4 engine
  run SIMULTANEOUSLY across a worker pool, and the best affinities
  ranked into the results table.
* **Data & ML**     -- open any CSV (or one of the bundled example
  datasets), pick a target column, train a registry model with the
  toolkit's own preprocessing pipeline, inspect the metrics and the
  parity/confusion figure, cross-validate a model leaderboard, build
  VOTING and STACKING ensembles by hand (tick the base models, pick
  soft/hard voting or the stacking meta-learner, train), write the
  whole run as a report bundle -- ``report.html`` + figures +
  tables (+ the multi-page ``report.pdf`` when ticked), the CLI
  ``--plots``/``--pdf-report`` story, mouse edition -- edit the selected
  model's registry knobs by hand in the Manual hyperparameters card
  (blank keeps the estimator default, pinned keys are excluded from the
  grid search, values persist per model with the session, Apply best
  prefills the winner, the cv-folds spinner sets the fold count), watch
  every long job as a determinate progress bar with a live
  remaining-time estimate (a search's trials, a network's epochs, a
  leaderboard's models), run a
  keras-tuner-style hyperparameter search (``ml tune``'s RandomSearch
  over the ann / dnn / hsvr families, or Hyperband successive halving
  over the resource budget -- ann's ``max_iter``, dnn's ``epochs`` --
  with the rung ladder sized from the family's own declared grid via
  ``resource_bounds``, the epoch-economics budget line, the defaults
  baseline, the held-out verdict, an apply-best bridge to the Deep
  Learning tab and a trials CSV export) -- save the trained model as a
  provenance-carrying BUNDLE (joblib + metadata sidecar: feature
  order, hyperparameters, metrics, library versions, timestamp), load
  any bundle back, audit new data against its feature contract and
  predict (the ``ml save`` / ``ml check`` story, mouse edition) -- and
  audit any new CSV against the last
  trained model before predicting (the ``ml check`` story, mouse
  edition: every difference named).
* **Deep Learning** -- the toolkit's PyTorch estimators behind one card
  of inputs: pick the architecture (feed-forward or residual MLP), the
  hidden widths, dropout, activation, Adam's learning rate and weight
  decay, the early-stopping patience and validation fraction, toggle
  batch normalisation (``batchnorm=True``, 1.2.36 -- a BatchNorm1d
  between every hidden linear map and its activation), then watch the
  held-out metrics, the parity/confusion figure and the epoch-by-epoch
  loss curves (``plot_training_history`` -- the early-stopping verdict
  rides on the title, the per-epoch records export as CSV) -- and audit
  any new CSV against the last trained network before predicting (the
  ``ml check`` story, deep edition: every difference named, the
  training target dropped automatically).
* **Generative**    -- run the genetic algorithm OR the reinforcement
  learning optimiser on a SMILES corpus with any bundled objective (QED,
  LogP window, ...); adaptive GA runs render the operator-mix learning
  curve (``plot_operator_trajectory``), RL runs render the four-panel
  prior-drift introspection (``plot_rl_introspection``: score trajectory,
  agent-vs-prior log-probs, the anchored drift, the exploration entropy)
  plus a first→last history table; and multi-objective runs fill the
  Pareto front card: the audit line, the non-dominated table (with the
  crowding-distance column and the deepest hidden trade-off rank), the
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
* A quiet keyboard-shortcut hint row sits above the status bar,
  listing every global accelerator (Ctrl+O open · Ctrl+S save figure ·
  Ctrl+Shift+S save session · Ctrl+Shift+R reset this tab · Ctrl+Q
  quit -- all five bound and pinned by tests).  ``View ▸ Shortcut
  hints`` or the row's own ``Hide`` button toggles it, and the
  preference rides the session file like any other input.

Command line helpers (useful for scripting and CI)::

    python playground.py --probe        # dependency report, then exit
    python playground.py --self-test    # build the UI headlessly, then exit
    python playground.py --version      # print versions, then exit
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import inspect
import json
import math
import queue
import re
import shutil
import sys
import threading
import time
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
    ("torch", "PyTorch", "deep-learning estimators + the RL optimiser", "torch"),
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


def fpdf_available() -> bool:
    """Report whether the PDF-report dependency is importable.

    Companion to :func:`probe_dependencies` for the one optional dep the
    heavy-packages table does not carry: ``fpdf`` (the ``fpdf2`` wheel,
    part of the ``reports`` extra) writes the multi-page ``report.pdf``
    behind the Data & ML tab's *include report.pdf* checkbox.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("fpdf") is not None
    except (ImportError, ValueError):  # pragma: no cover - exotic setups
        return False


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


def parse_typed_value(raw: str, default: Any) -> Any:
    """Parse one manual-hyperparameter field against its default's type.

    The manual-hyperparameter card (100x-106) edits registry grid knobs
    whose defaults come from the estimator itself, so the string in the
    entry box is converted by the *type of the default*: bool fields
    accept ``true/false/yes/no/1/0`` (case-insensitive), int and float
    fields go through :class:`int`/:class:`float` (``ValueError``
    propagates with the raw text in it), string fields pass through
    unchanged, tuple fields (a forest's ``hidden_layer_sizes``) parse
    ``"128, 64"`` / ``"(128, 64)``"`` into the tuple itself, and
    ``None`` defaults (e.g. ``max_depth``) accept ``none``/``null``
    verbatim or any literal int/float, falling back to the raw string.
    A blank raw value never reaches this helper — the caller keeps
    the default instead.
    """
    text = str(raw).strip()
    if not text:
        return default
    if isinstance(default, bool):  # before int: bool IS an int subclass
        lowered = text.lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
        raise ValueError(f"{text!r} is not a true/false value")
    if isinstance(default, int):
        return int(text)
    if isinstance(default, float):
        return float(text)
    if isinstance(default, tuple):
        cleaned = text.strip("()[] ")
        parts = [part for part in cleaned.replace(",", " ").split() if part]
        if not parts:
            raise ValueError(f"{text!r} is not a comma-separated width list")
        members = []
        for part in parts:
            try:
                members.append(int(part))
            except ValueError as exc:
                raise ValueError(
                    f"{text!r} mixes non-integer widths ({part!r})"
                ) from exc
        return tuple(members)
    if default is None:
        lowered = text.lower()
        if lowered in ("none", "null"):
            return None
        # comma-separated integer lists parse as tuples -- the ensemble
        # ``weights`` knob (a None-default parameter) takes them
        if "," in text:
            cleaned = text.strip("()[] ")
            parts = [part for part in cleaned.split(",") if part.strip()]
            try:
                return tuple(int(part) for part in parts)
            except ValueError:
                pass  # not an integer list -- fall through to the casts
        for cast in (int, float):
            try:
                return cast(text)
            except ValueError:
                continue
        return text
    return text


def _fmt_duration(seconds: float) -> str:
    """Compact human duration for the progress bar's ETA readout.

    ``42 s`` → ``3 min 05 s`` → ``1 h 04 min``: one consistent format
    per magnitude, always from a clamped non-negative input (a fresh
    job's sub-second elapsed time must never render as ``-0 s``).
    """
    total = max(0, round(float(seconds)))
    if total < 60:
        return f"{total} s"
    minutes, seconds_left = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} min {seconds_left:02d} s"
    hours, minutes_left = divmod(minutes, 60)
    return f"{hours} h {minutes_left:02d} min"


def _accepts_progress_argument(work: Callable[..., Any]) -> bool:
    """True when ``work`` wants the progress reporter injected.

    A zero-argument ``work`` (every historical job) is called as
    ``work()``; a one-argument ``work`` receives the reporter callable
    — the signature decides, so no call site has to change.
    """
    try:
        parameters = list(inspect.signature(work).parameters.values())
    except (TypeError, ValueError):  # builtins / C callables -- no report
        return False
    return any(
        parameter.kind in (inspect.Parameter.POSITIONAL_ONLY,
                           inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           inspect.Parameter.VAR_POSITIONAL)
        for parameter in parameters
    )


def _call_work(work: Callable[..., Any], report: Callable[..., None]) -> Any:
    """Invoke ``work`` with the reporter only when it accepts one."""
    if _accepts_progress_argument(work):
        return work(report)
    return work()


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

#: The SMILES-array card's engines (dock_smiles_array tokens).
_SMILES_DOCK_ENGINES = ("vina", "autodock4")

#: The SMILES-array card's default pasted array: the trivial-name drugs
#: the example campaign ships (matched to folder names by the library's
#: own trivial-name table).
_DOCK_SMILES_DEFAULTS = (
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",   # caffeine
    "CC(=O)Nc1ccc(O)cc1",             # paracetamol
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # ibuprofen
    "COc1cc2nc(CC(C)C)sc2c(O)cc1",    # naproxen
)

# Canonical evidence columns for the consensus rescore card, in
# preference order -- the names consensus_rescore's own examples and the
# §9 notebook demo use.  The prefill intersects these with the loaded
# table's columns; anything else is typed by hand.
_RESCORE_PREFERRED_COLUMNS = (
    "ligand_efficiency",
    "LE (kcal/mol/atom)",
    "best_lowest_energy",
    "lowest_energy",
    "affinity",
    "best_num_in_cluster",
    "num_in_cluster",
    "cluster_size",
)


def docking_engine_binaries(engine: str) -> tuple[str, ...]:
    """The external binaries an engine needs on ``PATH``.

    The AutoDock4 campaign runner (and the SMILES array's
    ``autodock4`` engine) drives ``autogrid4`` + ``autodock4`` over
    every ligand folder; the Vina engines need the single ``vina``
    binary.  Used to pre-empt the workflow module's own RuntimeError
    with a friendly install hint before any job is started.
    """
    if engine in ("AutoDock4 campaign", "autodock4"):
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
    ("rism", "RISM-KH solvation (10)"),
)

#: Column count of each fingerprint family.
_FEATURE_FAMILY_BITS: dict[str, int] = {
    "atompair_count": 2048,
    "ecfp": 2048,
    "avalon_count": 512,
    "rdkit_fingerprint": 2048,
    "morgan_full": 2048,
    "morgan_scaffold": 2048,
    "rism": 10,
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

#: Plain-language meaning + unit of every RISM-KH descriptor column
#: (the Chemistry tab's RISM card renders them next to the numbers).
_RISM_MEANINGS: dict[str, tuple[str, str]] = {
    "RISM_Gsolv_kcalmol": ("total solvation free energy", "kcal/mol"),
    "RISM_Glj_kcalmol": ("Lennard-Jones (dispersion) part", "kcal/mol"),
    "RISM_Gcoul_kcalmol": ("Coulomb charging part", "kcal/mol"),
    "RISM_Ow_gmax": ("water-O distribution peak", "g(d) max"),
    "RISM_Hw_gmax": ("water-H distribution peak", "g(d) max"),
    "RISM_Ow_cn1": ("first-shell water-O coordination", "molecules"),
    "RISM_Hw_cn1": ("first-shell water-H coordination", "molecules"),
    "RISM_Ow_exclvol_A3": ("water-O excluded volume", "Å³"),
    "RISM_dipole_D": ("molecular dipole moment", "Debye"),
    "RISM_q_neg_frac": ("negative-charge fraction", "fraction"),
}

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


def default_target_column(frame: Any) -> str | None:
    """Pick the most likely target column of a loaded dataset.

    Walks from the RIGHT, skipping the SMILES column and string columns
    whose every value is unique -- an ID column (``compound_id`` with
    420 distinct values), never a prediction target.  The bundled
    examples all cooperate: ``qsar_regression.csv`` ends in ``pIC50``
    and ``qsar_multiclass.csv`` in ``Target``, but
    ``qsar_classification.csv`` ends in ``compound_id`` with the real
    ``activity`` label one step left -- the naive last-column default
    trained a 420-class classifier on row IDs.
    """
    columns = [str(column) for column in frame.columns]
    if not columns:
        return None
    smiles = detect_smiles_column(frame)
    rows = len(frame)
    for column in reversed(columns):
        if smiles is not None and column == smiles:
            continue
        series = frame[column]
        if (
            rows > 6
            and str(series.dtype) == "object"
            and series.nunique() == rows
        ):
            continue  # unique-per-row strings: an ID, not a label
        return column
    return columns[-1]


def numeric_feature_columns(frame: Any, target: str) -> tuple[list[str], list[str]]:
    """Split a dataset's non-target columns into ``(numeric, dropped)``.

    The train flows' shared rule (Data & ML, Deep Learning): numeric
    columns become features; everything else -- SMILES, IDs, free text
    -- is reported as dropped so the user sees WHY those columns never
    reached the model.
    """
    import pandas as pd

    numeric = [
        column for column in frame.columns
        if column != target and pd.api.types.is_numeric_dtype(frame[column])
    ]
    dropped = [
        str(column) for column in frame.columns
        if column != target and column not in numeric
    ]
    return numeric, dropped


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


#: Deep-learning tab (100x-61): architecture labels the combobox offers.
_DL_ARCHITECTURES: tuple[str, ...] = (
    "Deep MLP (feed-forward)",
    "Residual MLP (skip connections)",
)

#: Hidden-activation tokens -- mirrors ``pharmaml.ml.neural._ACTIVATIONS``
#: (the test suite pins the two together, so they cannot drift).
_DL_ACTIVATIONS: tuple[str, ...] = ("relu", "tanh", "gelu", "sigmoid")

#: The keyboard-shortcut hint row (100x-65): one quiet line above the
#: status bar listing every global accelerator.  The five entries mirror
#: the ``bind_all`` sequences in ``_build_menu`` (the test suite pins the
#: set together, so a new binding without a hint -- or vice versa --
#: fails loudly).
SHORTCUT_HINTS: tuple[tuple[str, str], ...] = (
    ("Ctrl+O", "open"),
    ("Ctrl+S", "save figure"),
    ("Ctrl+Shift+S", "save session"),
    ("Ctrl+Shift+R", "reset this tab"),
    ("Ctrl+Q", "quit"),
)


def shortcut_hints_text() -> str:
    """Render the hint row's one-line label from :data:`SHORTCUT_HINTS`."""
    return " \u00b7 ".join(f"{keys} {what}" for keys, what in SHORTCUT_HINTS)


def parse_hidden_layers(raw: str) -> tuple[int, ...]:
    """Parse the hidden-layers entry (``"64, 32"``) into a width tuple.

    An empty entry is the empty tuple -- a single linear layer, the
    regularised multinomial/linear baseline the estimators themselves
    build for ``hidden_layers=()``.

    Raises
    ------
    ValueError
        When a token is not a number or a width below 1.
    """
    widths: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            width = int(float(token))  # "64" and "64.0" both parse
        except ValueError as exc:
            raise ValueError(
                f"hidden layer widths must be numbers, got {token!r}"
            ) from exc
        if width < 1:
            raise ValueError(f"hidden layer widths must be >= 1, got {token!r}")
        widths.append(width)
    return tuple(widths)


def deep_estimator_name(architecture: str, task: str) -> str:
    """Map the GUI's architecture/task pair onto an estimator class name.

    Pure on purpose: the playground never imports ``pharmaml.ml.neural``
    at module scope -- the worker thread resolves the returned name at
    fit time, so the mapping stays testable (and typo-checkable) without
    pharmaml or torch installed.

    Raises
    ------
    ValueError
        On an unknown architecture label or task token.
    """
    if architecture not in _DL_ARCHITECTURES:
        raise ValueError(f"unknown architecture {architecture!r}")
    if task not in ("classification", "regression"):
        raise ValueError(f"unknown task {task!r}")
    family = "ResidualMLP" if architecture.startswith("Residual") else "DeepMLP"
    return family + ("Classifier" if task == "classification" else "Regressor")


#: Data & ML tab (100x-90): hypermodel families the Tune card offers --
#: the same tokens the CLI's ``ml tune --estimator`` accepts.
_TUNE_ESTIMATORS: tuple[str, ...] = ("ann", "hsvr", "dnn")

#: Ensemble-builder card: registry members offered as checkboxes --
#: ``(registry token, label)``.  Dual-task, dependency-free families
#: only, so every combination builds for both tasks.
_ENSEMBLE_MEMBER_LABELS: tuple[tuple[str, str], ...] = (
    ("random_forest", "Random forest"),
    ("extra_trees", "Extra trees"),
    ("svm", "SVM"),
    ("knn", "KNN"),
    ("decision_tree", "Decision tree"),
    ("logistic_regression", "Logistic (cls)"),
    ("ridge", "Ridge (reg)"),
    ("gradient_boosting", "Gradient boosting"),
    ("ann", "ANN (MLP)"),
    ("gaussian_process", "Gaussian process"),
)

#: The default ticked trio -- the rar's own Voting.py / Stacking.py
#: members (RF + SVM + KNN), matching the registry ensembles' defaults.
_ENSEMBLE_DEFAULT_MEMBERS: tuple[str, ...] = ("random_forest", "svm", "knn")

#: Searcher tokens for the Tune card -- the same two the CLI's
#: ``ml tune --searcher`` accepts (random | hyperband).
_TUNE_SEARCHERS: tuple[str, ...] = ("random", "hyperband")

#: The resource slot each Hyperband-able family halves over -- the
#: CLI's own rule (``max_iter`` for the sklearn ann, ``epochs`` for
#: the torch dnn); hsvr declares neither, so the card guards it.
_TUNE_RESOURCES: dict[str, str] = {"ann": "max_iter", "dnn": "epochs"}

#: Hypermodel builder names for the Tune card, keyed by
#: ``(estimator, task)`` -- resolved on the worker thread (the playground
#: never imports ``pharmaml`` at module scope), and pinned to the CLI's
#: own ``ml tune`` registry by the test suite so the two surfaces cannot
#: drift apart (the ``_DL_ACTIVATIONS`` pin pattern).
_TUNE_HYPERMODELS: dict[tuple[str, str], str] = {
    ("ann", "regression"): "build_ann_regressor",
    ("ann", "classification"): "build_ann_classifier",
    ("dnn", "regression"): "build_deep_regressor",
    ("dnn", "classification"): "build_deep_classifier",
    ("hsvr", "regression"): "build_hsvr",
}


def tune_config_text(params: dict[str, Any]) -> str:
    """Render one trial's hyperparameters as ``k=v`` tokens.

    Floats go through ``%g`` (``alpha=0.001``, not ``alpha=0.0010000``),
    booleans render as ``on`` / ``off`` and everything else as ``str``.
    Keys are sorted so the leaderboard's configuration column is stable
    across trials (and across runs -- determinism is the tuner's own
    contract).

    >>> tune_config_text({"units": 64, "alpha": 0.001, "batchnorm": True})
    'alpha=0.001 batchnorm=on units=64'
    >>> tune_config_text({"gamma": "scale"})
    'gamma=scale'
    """
    tokens: list[str] = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, bool):
            text = "on" if value else "off"
        elif isinstance(value, float):
            text = f"{value:g}"
        else:
            text = str(value)
        tokens.append(f"{key}={text}")
    return " ".join(tokens)


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
    """The playground window: one notebook, eight tabs, one worker."""

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
        # -- progress + ETA plumbing (100x-106) ----------------------- #
        # Worker threads push (done, total, note) tuples here; the
        # ``_poll_progress`` after-loop drains them onto the status bar
        # (determinate progressbar + "~ 42 s left" in the job label).
        # Updates arriving while no job runs are dropped.
        self._progress_updates: queue.Queue[tuple[int, int | None, str]] = queue.Queue()
        self._job_started = 0.0
        self._job_label_base = ""
        self._action_buttons: list[Any] = []
        self._last_figure: Any = None
        self._ml_frame: Any = None
        self._ml_path: Path | None = None
        # -- Data & ML report state (100x-76, extended 100x-79) ----- #
        # (trained, report, task, X_test, y_test, train_target) of the
        # last successful train -- the Save-report button rebuilds the
        # CLI's bundle from it; the Check-CSV button audits new data
        # against it (the training target rides along as the automatic
        # --target drop).
        self._ml_last_run: tuple[Any, Any, str, Any, Any, str] | None = None
        # -- Hyperparameter-search card state (100x-90) ------------- #
        # (TunerResults, estimator token, task, target column) of the
        # last search -- the Apply-best and Save-trials payloads.
        self._tune_last: tuple[Any, str, str, str] | None = None
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
        self._dock_rescore_last: tuple[Any, list[str], list[float] | None,
                                       dict[str, bool], str] | None = None
        # (rescored frame, evidence columns, weights, higher overrides,
        #  source label) — the Save report… payload
        self._dock_results_source: str | None = None    # campaign / dlg / vina / smiles
        self._dock_campaign_root: Path | None = None    # AutoDock4 batch root
        self._dock_vina_receptor: Path | None = None
        self._dock_vina_ligand: Path | None = None
        # -- SMILES array docking state (the one-call batch card) ---- #
        self._dock_smiles_receptor: Path | None = None  # picked receptor PDBQT
        self._dock_smiles_out: Path | None = None       # campaign output root
        self._dock_smiles_last: Any = None              # ranked DataFrame
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
        # -- RISM-KH solvation card state ----------------------------- #
        self._rism_frame: Any = None            # last computed descriptor frame
        # -- Ensemble-builder + model-persistence state (ML tab) ------ #
        self._ml_bundle: tuple[Any, dict[str, Any], Path] | None = None
        self._ml_predictions: Any = None        # last bundle prediction frame

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
        self._build_hintbar()
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
        self.after(150, self._poll_progress)
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
        # Shortcut-hint row visibility (100x-65): a plain tk.Variable, so
        # the generic session machinery persists it for free.
        self._hints_var = tk.BooleanVar(value=True)
        view_menu.add_checkbutton(
            label="Shortcut hints", variable=self._hints_var,
            command=self._apply_hint_visibility)
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
        # 100x-65: the menu (and the docstring) had advertised Ctrl+Shift+R
        # since 100x-54, but no binding ever existed -- pressing it did
        # nothing.  Bound now, and the test suite pins all five sequences.
        self.bind_all("<Control-Shift-R>", lambda _e: self._reset_current_tab())
        self.bind_all("<Control-q>", lambda _e: self._on_close())

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self, style="Status.TFrame", padding=(12, 5))
        bar.pack(side="bottom", fill="x")
        self._status_bar = bar  # pinned so tests can order hint row vs bar
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

    def _build_hintbar(self) -> None:
        """The keyboard-shortcut hint row (100x-65), above the status bar.

        One quiet line listing every global accelerator, so the bindings
        are discoverable without opening a menu.  ``View \u25b8 Shortcut
        hints`` (or the row's own ``Hide`` button) toggles it; the
        preference is a plain ``tk.Variable`` on ``self``, so the session
        machinery saves and restores it like any other input.  Packed
        ``side="bottom"`` AFTER the status bar claimed the bottom edge,
        which pins it directly above the status line.
        """
        bar = ttk.Frame(self, style="Status.TFrame", padding=(12, 2))
        label = ttk.Label(bar, text=shortcut_hints_text(),
                          style="Status.TLabel")
        label.pack(side="left")
        Tooltip(label,
                "The global keyboard shortcuts, also listed in the File "
                "menu.  View \u25b8 Shortcut hints re-enables this row "
                "after hiding it (the preference is saved with the "
                "session).")
        hide = ttk.Button(bar, text="Hide", style="Ghost.TButton",
                          command=self._hide_hints)
        hide.pack(side="right", padx=(12, 0))
        Tooltip(hide, "Hide the shortcut hint row (View \u25b8 Shortcut "
                      "hints brings it back).")
        self._hint_bar = bar
        bar.pack(side="bottom", fill="x")
        self._apply_hint_visibility()

    def _apply_hint_visibility(self) -> None:
        """Show/hide the hint row to match :attr:`_hints_var`."""
        bar = getattr(self, "_hint_bar", None)
        if bar is None:
            return
        if self._hints_var.get():
            bar.pack(side="bottom", fill="x")
        else:
            bar.pack_forget()

    def _hide_hints(self) -> None:
        """The hint row's own Hide button -- flip the preference off."""
        self._hints_var.set(False)
        self._apply_hint_visibility()

    def _build_tabs(self) -> None:
        self._book = ttk.Notebook(self)
        self._book.pack(fill="both", expand=True, padx=10, pady=(10, 4))
        self._build_home_tab()
        self._build_chemistry_tab()
        self._build_pk_tab()
        self._build_cellline_tab()
        self._build_docking_tab()
        self._build_ml_tab()
        self._build_dl_tab()
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

    def _start_job(self, label: str, work: Callable[..., Any],
                   on_done: Callable[[Any], None] | None = None) -> None:
        """Run ``work`` in a worker thread; render via ``on_done``.

        Progress (100x-106): a ``work`` that *accepts an argument*
        receives a reporter callable to push ``(done, total, note)``
        updates -- the status bar then switches from the marquee to a
        determinate bar carrying the fraction and a live "~ x left"
        ETA (see :meth:`_poll_progress`).  Zero-argument ``work``
        functions keep the historical indeterminate marquee.
        """
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
        self._job_label_base = label
        self._job_started = time.monotonic()
        with contextlib.suppress(queue.Empty):
            while True:  # stale updates from a previous job never leak
                self._progress_updates.get_nowait()
        self._progress.start(12)
        self._job_label.configure(text=label)

        def runner() -> None:
            try:
                self._jobs.put(
                    (_call_work(work, self._report_progress), None)
                )
            except BaseException as exc:  # noqa: BLE001 - surfaced in the UI
                self._jobs.put((None, exc))

        threading.Thread(target=runner, name="playground-job", daemon=True).start()

    def _report_progress(self, done: int, total: int | None = None,
                         note: str = "") -> None:
        """Worker-thread progress feed (thread-safe: queue → poller).

        Called from inside ``work`` for every completed unit -- a tuner
        trial, a training epoch, a leaderboard model -- with the unit
        count, the total (``None`` for open-ended work) and an optional
        note rendered after the fraction.  Only the latest update is
        shown; the poller coalesces bursts.
        """
        self._progress_updates.put((int(done), total, note))

    def _poll_progress(self) -> None:
        """Drain progress updates into the status bar (bar + ETA).

        Coalesces everything queued into the newest update, then
        renders it: a determinate bar at ``done / total`` and a job
        label like ``tuning ann (regression) — 3/8 · ~45 s left · best
        0.712``.  The ETA extrapolates the elapsed/fraction ratio;
        total-less updates degrade to a plain counter.  Updates that
        arrive while no job is running (late stragglers) are dropped.
        """
        latest: tuple[int, int | None, str] | None = None
        with contextlib.suppress(queue.Empty):
            while True:
                latest = self._progress_updates.get_nowait()
        if latest is not None and self._busy:
            done, total, note = latest
            elapsed = max(time.monotonic() - self._job_started, 1e-9)
            text = self._job_label_base
            if total is not None and total > 0:
                fraction = min(done / total, 1.0)
                with contextlib.suppress(tk.TclError):
                    self._progress.stop()
                    self._progress.configure(mode="determinate", maximum=total)
                    self._progress["value"] = done
                text += f" — {done}/{total}"
                if fraction > 0.0:
                    remaining = elapsed * (1.0 - fraction) / fraction
                    text += f" · ~{_fmt_duration(remaining)} left"
            else:
                text += f" — {done}"
            if note:
                text += f" · {note}"
            with contextlib.suppress(tk.TclError):
                self._job_label.configure(text=text)
        self.after(150, self._poll_progress)

    def _poll_jobs(self) -> None:
        try:
            result, error = self._jobs.get_nowait()
        except queue.Empty:
            self.after(80, self._poll_jobs)
            return
        self._busy = False
        self._progress.stop()
        with contextlib.suppress(tk.TclError):
            # back to the launch-time marquee for the next job: the
            # determinate mode a progress-reporting job switched to is
            # per-job state, never a permanent UI change.
            self._progress.configure(mode="indeterminate")
            self._progress["value"] = 0
        with contextlib.suppress(queue.Empty):
            while True:  # late progress stragglers never leak either
                self._progress_updates.get_nowait()
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
            "smiles": self._dock_smiles_text.get("1.0", "end-1c"),
            "rism": self._rism_text.get("1.0", "end-1c"),
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
                        "corpus": self._corpus_text,
                        "smiles": getattr(self, "_dock_smiles_text", None),
                        "rism": getattr(self, "_rism_text", None)}
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
        # the ML tab owns one DYNAMIC input -- the manual-hyperparameter
        # card.  Its JSON variable was reset with the others above; the
        # panel itself must be rebuilt to show it (100x-106).
        if label.strip() == "Data & ML":
            self._ml_params_var.set("{}")
            with contextlib.suppress(Exception):  # best effort
                self._ml_refresh_hyperparam_panel()
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
                        "corpus": self._corpus_text,
                        "smiles": getattr(self, "_dock_smiles_text", None),
                        "rism": getattr(self, "_rism_text", None)}
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
                      id(self._corpus_text): "text:corpus",
                      id(getattr(self, "_dock_smiles_text", None)): "text:smiles",
                      id(getattr(self, "_rism_text", None)): "text:rism"}
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
        # the hint row's visibility rides a tk.Variable, so a restored
        # session may have flipped it -- re-apply after the variables.
        self._apply_hint_visibility()
        text_widgets = {"pasted": self._feat_pasted_text,
                        "corpus": self._corpus_text,
                        "smiles": getattr(self, "_dock_smiles_text", None),
                        "rism": getattr(self, "_rism_text", None)}
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
        # the ML tab's manual-hyperparameter card is DYNAMIC (rebuilt
        # per model selection, values carried by the JSON variable the
        # loop above already restored) -- rebuild it once, after the
        # comboboxes hold their restored selections (100x-106).
        with contextlib.suppress(Exception):  # never block restore
            self._ml_refresh_hyperparam_panel()
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
            "deep learning · generative design.\n\n"
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
                     "cell-line biology, molecular docking, QSAR modelling, "
                     "deep learning and the generative GA, without touching "
                     "a terminal.",
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
                 "any SMILES, Tanimoto similarity, the RISM-KH solvation "
                 "card (ten solvation-theory descriptors), and the feature "
                 "factory: 10,000+ descriptor/fingerprint columns ranked to "
                 "a top-K."),
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
                ("Ligand library and receptor preparation into PDBQT (bad "
                 "residues auto-removed), the grid box with GPF/DPF "
                 "configs, DLG / Vina result parsing, batch campaign runs "
                 "-- and the SMILES array card: paste a list, dock every "
                 "molecule simultaneously with Vina or AutoDock4."),
                "RDKit + pandas",
            ),
            (
                "Data & ML",
                ("Open a CSV (or a bundled example), train a model, read the metrics, "
                 "browse the cross-validation leaderboard, build voting / "
                 "stacking ensembles by hand, and save the trained model as "
                 "a bundle you can load back and predict with."),
                "pandas + sklearn",
            ),
            (
                "Generative",
                ("Evolve molecules against QED / LogP objectives with the GA "
                 "or reinforcement learning; adaptive runs show the "
                 "operator-mix learning curve, RL runs the prior-drift "
                 "introspection."),
                "RDKit (+ torch for RL)",
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
            subtitle="The rar's own ML pipeline as a card: 8 descriptor / "
                     "fingerprint families (~10,810 columns per molecule), "
                     "ranked and cut to the top K you keep -- now including "
                     "the RISM-KH solvation family.",
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

        # -- RISM-KH solvation descriptors (the 3D surrogate card) ------ #
        rism = make_card(
            page.grid, "RISM-KH solvation", row=3, column=0, span=2,
            subtitle="Ten 3D-RISM-KH-inspired solvation descriptors per "
                     "molecule (KH-closure pair distributions over an "
                     "embedded 3D geometry): solvation free energy split "
                     "into Lennard-Jones and Coulomb parts, water-site "
                     "distribution peaks, first-shell coordination numbers, "
                     "excluded volume, dipole and charge asymmetry.",
        )
        rism_left = ttk.Frame(rism, style="Card.TFrame")
        rism_left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self._rism_text = _grid_text(rism_left, height=5, mono=self._mono)
        self._rism_text.insert("1.0", "\n".join(_FEAT_DEFAULT_SMILES))
        self._rism_text.pack(fill="both", expand=True)
        rism_run_row = ttk.Frame(rism_left, style="Card.TFrame")
        rism_run_row.pack(fill="x", pady=(6, 0))
        rism_btn = self._action_button(
            rism_run_row, "Compute RISM-KH", self._rism_run, primary=True,
            tooltip="Embeds every SMILES in 3D (ETKDGv3 + MMFF/UFF, fixed "
                    "seed), applies the KH closure pair distributions "
                    "against the chosen water model and integrates the "
                    "same quantities a full 3D-RISM-KH solve reports -- "
                    "milliseconds per molecule instead of grid hours.",
        )
        rism_btn.pack(side="left")
        self._rism_summary = ttk.Label(
            rism_run_row, text="—", style="CardMuted.TLabel",
            wraplength=430, justify="left")
        self._rism_summary.pack(side="left", padx=(10, 0))

        rism_right = ttk.Frame(rism, style="Card.TFrame")
        rism_right.grid(row=0, column=1, sticky="n")
        rism_right.columnconfigure(1, weight=1)
        self._rism_solvent_combo = ttk.Combobox(
            rism_right, state="readonly", width=16, values=("spc", "tip3p"))
        self._rism_solvent_combo.set("spc")
        Tooltip(self._rism_solvent_combo,
                "The water model's site charges / LJ parameters: SPC or "
                "TIP3P (both keep the surrogate solvent charge-neutral).")
        form_row(rism_right, "water model", self._rism_solvent_combo)
        self._rism_temp_entry = ttk.Entry(rism_right, width=16)
        self._rism_temp_entry.insert(0, "298.15")
        form_row(rism_right, "temperature (K)", self._rism_temp_entry)
        rism_save_btn = self._action_button(
            rism_right, "Save descriptors CSV…", self._rism_save,
            tooltip="SMILES + the ten RISM-KH descriptor columns, ready "
                    "for the feature factory or a QSAR table.",
        )
        rism_save_btn.grid(row=rism_right.grid_size()[1], column=0,
                           columnspan=2, sticky="w", pady=(6, 0))
        rism_tree_frame = ttk.Frame(rism, style="Card.TFrame")
        grid_append(rism, rism_tree_frame, sticky="nsew", pady=(6, 0),
                    weight=True)
        self._rism_tree = ttk.Treeview(
            rism_tree_frame, columns=("molecule", "descriptor", "value", "meaning"),
            show="headings", height=9,
        )
        for column, width in (
            ("molecule", 150), ("descriptor", 170), ("value", 90),
            ("meaning", 230),
        ):
            self._rism_tree.heading(column, text=column)
            self._rism_tree.column(column, width=width, minwidth=56,
                                   stretch=(column == "meaning"))
        rism_bar = ttk.Scrollbar(rism_tree_frame, orient="vertical",
                                 command=self._rism_tree.yview)
        self._rism_tree.configure(yscrollcommand=rism_bar.set)
        self._rism_tree.pack(side="left", fill="both", expand=True)
        rism_bar.pack(side="right", fill="y")

    def _rism_run(self) -> None:
        """Compute the RISM-KH solvation descriptors for the pasted array."""
        if not self._require_pharmaml() or not self._require(
                "pandas", "RISM-KH descriptors") or not self._require(
                "rdkit", "RISM-KH descriptors"):
            return
        smiles = parse_smiles_lines(self._rism_text.get("1.0", "end"))
        if not smiles:
            messagebox.showwarning(
                "Empty array", "Paste at least one SMILES (one per line).")
            return
        solvent = self._rism_solvent_combo.get()
        try:
            temperature = parse_float(self._rism_temp_entry.get(),
                                      "temperature (K)")
            if temperature <= 0:
                raise ValueError("temperature (K) must be positive")
        except ValueError as exc:
            messagebox.showerror("Input", str(exc))
            return

        def work() -> Any:
            from pharmaml.chem.rism import compute_rism_descriptors

            return compute_rism_descriptors(
                smiles, solvent=solvent, temperature=temperature,
                progress=False,
            )

        def render(frame: Any) -> None:
            self._rism_frame = frame
            tree = self._rism_tree
            tree.delete(*tree.get_children())
            for _, row in frame.iterrows():
                molecule = str(row["SMILES"])
                for descriptor, (meaning, unit) in _RISM_MEANINGS.items():
                    value = row[descriptor]
                    tree.insert("", "end", values=(
                        _cell_text(molecule, limit=24), descriptor,
                        _cell_text(value, limit=14),
                        f"{meaning} ({unit})",
                    ))
            gsolv = frame["RISM_Gsolv_kcalmol"]
            self._rism_summary.configure(
                text=(f"{len(frame)} molecule(s) · {solvent.upper()} water · "
                      f"{temperature:.2f} K · Gsolv "
                      f"{gsolv.min():+.2f} … {gsolv.max():+.2f} kcal/mol"),
                style="Field.TLabel",
            )

        self._start_job(f"computing RISM-KH ({solvent})", work, render)

    def _rism_save(self) -> None:
        frame = getattr(self, "_rism_frame", None)
        if frame is None:
            messagebox.showinfo("Nothing to save",
                                "Compute the RISM-KH descriptors first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save the RISM-KH descriptors", defaultextension=".csv",
            filetypes=[("CSV table", "*.csv")],
        )
        if not path:
            return
        try:
            frame.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"RISM-KH table written to\n{path}")
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not save", exc)

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
        self._dock_bad_res_var = tk.BooleanVar(value=True)
        bad_res_check = ttk.Checkbutton(
            receptor, text="auto-remove bad residues", variable=self._dock_bad_res_var)
        Tooltip(bad_res_check,
                "When Meeko's templates cannot match a residue (broken "
                "density, non-standard chemistry), delete that residue and "
                "retry instead of failing -- Meeko CLI's --delete_bad_res, "
                "checkbox edition. Removed residues are named in the report.")
        form_row(receptor, "bad residues", bad_res_check)
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

        # -- card 6: consensus rescore (the explainable ranking) -------- #
        rescore = make_card(
            page.grid, "Consensus rescore", row=2, column=1,
            subtitle="Combine the loaded table's evidence into one 0–1 "
                     "consensus, with the per-column units that explain "
                     "the winner (the docking rescore CLI, mouse edition).",
        )
        rescore_buttons = ttk.Frame(rescore, style="Card.TFrame")
        rescore_buttons.grid(row=0, column=0, columnspan=2, sticky="w")
        rescore_csv_btn = self._action_button(
            rescore_buttons, "Open summary CSV…", self._dock_open_rescore_csv,
            tooltip="Any summary table (a saved scan, a rescored CSV, an "
                    "enriched export) -- the --data twin.",
        )
        rescore_csv_btn.pack(side="left")
        self._dock_rescore_info = ttk.Label(
            rescore_buttons, text="rescores the results table above",
            style="CardMuted.TLabel", wraplength=240, justify="left",
        )
        self._dock_rescore_info.pack(side="left", padx=(10, 0))
        self._dock_rescore_columns_entry = ttk.Entry(rescore, width=32)
        self._dock_rescore_columns_entry.insert(
            0, "best_lowest_energy, best_num_in_cluster")
        form_row(rescore, "evidence columns", self._dock_rescore_columns_entry)
        self._dock_rescore_weights_entry = ttk.Entry(rescore, width=32)
        form_row(rescore, "weights (blank = uniform)",
                 self._dock_rescore_weights_entry)
        self._dock_rescore_higher_entry = ttk.Entry(rescore, width=32)
        form_row(rescore, "higher-is-better (blank = auto)",
                 self._dock_rescore_higher_entry)
        rescore_btn = self._action_button(
            rescore, "Rescore table", self._dock_rescore, primary=True,
            tooltip="consensus_rescore's explainable twin: one 0–1 _norm "
                    "unit per evidence column plus the consensus. The "
                    "leaderboard replaces the results tree; Save results "
                    "CSV… writes the rescored table.",
        )
        rescore_btn.grid(row=rescore.grid_size()[1], column=0, sticky="w",
                         pady=(6, 0))
        report_btn = self._action_button(
            rescore, "Save report…", self._dock_save_report,
            tooltip="Write the last rescore as the one-click visual "
                    "bundle -- report.html + the consensus-ranking figure "
                    "+ the breakdown table (the CLI --plots/--pdf-report "
                    "story, mouse edition).",
        )
        report_btn.grid(row=rescore_btn.grid_info()["row"], column=1,
                        sticky="w", padx=(10, 0))
        self._dock_pdf_var = tk.BooleanVar(value=False)
        pdf_check = ttk.Checkbutton(
            rescore, text="include report.pdf", variable=self._dock_pdf_var)
        Tooltip(pdf_check, "Also write the multi-page PDF document -- "
                           "needs fpdf (pharmaml[reports])")
        grid_append(rescore, pdf_check, sticky="w", pady=(2, 0))
        self._dock_rescore_text = _grid_text(rescore, height=7, mono=self._mono)
        self._dock_rescore_text.insert(
            "1.0", "the evidence recipe and the driving-evidence verdict "
                   "appear here",
        )
        grid_append(rescore, self._dock_rescore_text, sticky="ew", pady=(6, 0))

        # -- card 7: SMILES array docking (the one-call batch) ---------- #
        array = make_card(
            page.grid, "SMILES array docking", span=2, row=3, column=0,
            subtitle="Paste a whole SMILES array and dock every molecule "
                     "against one receptor in a single run — each ligand is "
                     "prepared (Meeko + RDKit), the engine runs "
                     "simultaneously across workers, and the best affinities "
                     "land in the results table above.",
        )
        array.columnconfigure(0, weight=3)
        array.columnconfigure(1, weight=2)
        array_left = ttk.Frame(array, style="Card.TFrame")
        array_left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self._dock_smiles_text = _grid_text(array_left, height=6, mono=self._mono)
        self._dock_smiles_text.insert("1.0", "\n".join(_DOCK_SMILES_DEFAULTS))
        self._dock_smiles_text.pack(fill="both", expand=True)
        smiles_example_btn = self._action_button(
            array_left, "Fill example array", self._dock_smiles_example,
            tooltip="The bundled trivial-name drugs (aspirin, caffeine, "
                    "paracetamol, ibuprofen, ...).",
        )
        smiles_example_btn.pack(anchor="w", pady=(6, 0))
        ttk.Label(array_left, text="one SMILES per line · # comments ignored",
                  style="CardMuted.TLabel").pack(anchor="w", pady=(4, 0))

        array_right = ttk.Frame(array, style="Card.TFrame")
        array_right.grid(row=0, column=1, sticky="n")
        array_right.columnconfigure(1, weight=1)
        self._dock_smiles_engine_combo = ttk.Combobox(
            array_right, state="readonly", width=20, values=_SMILES_DOCK_ENGINES)
        self._dock_smiles_engine_combo.set(_SMILES_DOCK_ENGINES[0])
        Tooltip(self._dock_smiles_engine_combo,
                "vina = one Vina process per ligand (fast, the default); "
                "autodock4 = the full classic stack per ligand (receptor "
                "distributed, GPF + DPF written, autogrid4 + autodock4). "
                "Both run simultaneously across the workers below.")
        form_row(array_right, "engine", self._dock_smiles_engine_combo)
        smiles_rec_buttons = ttk.Frame(array_right, style="Card.TFrame")
        smiles_rec_buttons.grid(row=array_right.grid_size()[1],
                                column=0, columnspan=2, sticky="w")
        smiles_rec_btn = self._action_button(smiles_rec_buttons,
                                             "Receptor PDBQT…",
                                             self._dock_smiles_open_receptor)
        smiles_rec_btn.pack(side="left")
        self._dock_smiles_receptor_info = ttk.Label(
            smiles_rec_buttons, text="none — pick one (or prepare it above)",
            style="CardMuted.TLabel", wraplength=240, justify="left",
        )
        self._dock_smiles_receptor_info.pack(side="left", padx=(10, 0))
        smiles_out_buttons = ttk.Frame(array_right, style="Card.TFrame")
        smiles_out_buttons.grid(row=array_right.grid_size()[1],
                                column=0, columnspan=2, sticky="w", pady=(4, 0))
        smiles_out_btn = self._action_button(smiles_out_buttons,
                                             "Output folder…",
                                             self._dock_smiles_choose_out)
        smiles_out_btn.pack(side="left")
        self._dock_smiles_out_info = ttk.Label(
            smiles_out_buttons, text="no output folder chosen",
            style="CardMuted.TLabel", wraplength=240, justify="left",
        )
        self._dock_smiles_out_info.pack(side="left", padx=(10, 0))
        self._dock_smiles_center_entry = ttk.Entry(array_right, width=12)
        self._dock_smiles_center_entry.insert(0, "0 0 0")
        form_row(array_right, "centre (x y z)", self._dock_smiles_center_entry)
        self._dock_smiles_size_entry = ttk.Entry(array_right, width=12)
        self._dock_smiles_size_entry.insert(0, "22.5 22.5 22.5")
        form_row(array_right, "size (x y z)", self._dock_smiles_size_entry)
        self._dock_smiles_exhaust_entry = ttk.Entry(array_right, width=10)
        self._dock_smiles_exhaust_entry.insert(0, "8")
        form_row(array_right, "exhaustiveness (Vina)",
                 self._dock_smiles_exhaust_entry)
        self._dock_smiles_workers_entry = ttk.Entry(array_right, width=10)
        self._dock_smiles_workers_entry.insert(0, "4")
        Tooltip(self._dock_smiles_workers_entry,
                "Engine subprocesses to run simultaneously — real docking "
                "takes minutes per ligand, so 4 workers ≈ 4× throughput.")
        form_row(array_right, "workers", self._dock_smiles_workers_entry)
        smiles_run_row = ttk.Frame(array_right, style="Card.TFrame")
        smiles_run_row.grid(row=array_right.grid_size()[1], column=0,
                            columnspan=2, sticky="w", pady=(6, 0))
        smiles_run_btn = self._action_button(
            smiles_run_row, "Dock the array", self._dock_smiles_run, primary=True,
            tooltip="Prepares every SMILES into its own campaign folder, then "
                    "docks them all with the chosen engine (simultaneously "
                    "across workers); the ranked best-affinity table lands in "
                    "the results card above and Save results CSV… writes it.",
        )
        smiles_run_btn.pack(side="left")
        self._dock_smiles_summary = ttk.Label(
            smiles_run_row, text="—", style="CardMuted.TLabel",
            wraplength=280, justify="left",
        )
        self._dock_smiles_summary.pack(side="left", padx=(10, 0))

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
        allow_bad_res = bool(getattr(self, "_dock_bad_res_var", tk.BooleanVar(value=True)).get())
        chain = self._dock_chain_entry.get().strip() or None
        target = Path(output_pdbqt)

        def work() -> Any:
            from pharmaml.docking import prepare_protein

            return prepare_protein(
                source, output_pdbqt=target,
                remove_waters=remove_waters, keep_hetatm=keep_hetatm,
                chain=chain, strip_hydrogens=strip_hydrogens,
                allow_bad_res=allow_bad_res,
            )

        def render(report: Any) -> None:
            self._dock_render_protein_report(report)

        self._start_job(f"preparing {source.name}", work, render)

    def _dock_render_protein_report(self, report: Any) -> None:
        self._dock_protein_report = report
        self._dock_receptor_path = Path(report.output_pdbqt)
        # a freshly prepared receptor becomes the SMILES array card's
        # default too (its own explicit pick always wins)
        if self._dock_smiles_receptor is None:
            with contextlib.suppress(Exception):
                self._dock_smiles_receptor_info.configure(
                    text=f"prepared: {report.output_pdbqt}")
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
             f"{report.hydrogens_removed:,} hydrogens"
             + (f" · {report.bad_residues_removed} bad residues "
                f"({', '.join(report.bad_residue_ids)})"
                if getattr(report, "bad_residue_ids", None) else "")),
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
            self._dock_rescore_prefill()

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
            self._dock_rescore_prefill()

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
            self._dock_rescore_prefill()

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

    # -- consensus rescore card ------------------------------------------ #
    def _dock_open_rescore_csv(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas",
                                                            "summary loading"):
            return
        path = filedialog.askopenfilename(
            title="Open a docking summary table",
            filetypes=[("Summary table", "*.csv *.tsv"), ("All files", "*.*")],
        )
        if path:
            self._dock_load_rescore_csv(Path(path))

    def _dock_load_rescore_csv(self, path: Path) -> None:
        def work() -> Any:
            from pharmaml.utils.io import read_table

            return read_table(path)

        def render(frame: Any) -> None:
            if frame.empty:
                messagebox.showwarning("Empty table", f"{path.name} has no rows.")
                return
            self._dock_results_frame = frame
            self._dock_results_source = "csv"
            tree = self._dock_results_tree
            tree.delete(*tree.get_children())
            columns = [str(column) for column in frame.columns]
            tree["columns"] = columns
            for column in columns:
                tree.heading(column, text=column)
                tree.column(column, width=150 if column == columns[0] else 120,
                            minwidth=80, stretch=(column == columns[0]))
            for row in frame.to_dict("records"):
                values = [
                    _cell_text(None if (isinstance(row[column], float)
                                        and math.isnan(row[column]))
                               else row[column], limit=40)
                    for column in columns
                ]
                tree.insert("", "end", values=values)
            self._dock_results_info.configure(
                text=f"{len(frame)} row(s) · {path.name} — rescore ready",
                style="Field.TLabel",
            )
            self._dock_rescore_prefill()

        self._start_job(f"reading {path.name}", work, render)

    def _dock_rescore_prefill(self) -> None:
        """Seed the evidence-columns entry from the loaded table.

        Preference-ordered: the toolkit's canonical evidence names win;
        otherwise the numeric columns (minus the identity) are listed.
        """
        frame = self._dock_results_frame
        if frame is None:
            return
        available = {str(column) for column in frame.columns}
        preferred = [name for name in _RESCORE_PREFERRED_COLUMNS
                     if name in available]
        if not preferred:
            preferred = [str(column) for column in frame.columns[1:8]]
        if preferred:
            entry = self._dock_rescore_columns_entry
            entry.delete(0, "end")
            entry.insert(0, ", ".join(preferred[:6]))

    def _dock_rescore(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas",
                                                            "consensus rescoring"):
            return
        frame = self._dock_results_frame
        if frame is None:
            messagebox.showwarning(
                "No table loaded",
                "Scan a campaign, open a DLG / Vina log or a summary CSV "
                "first -- the rescore ranks the loaded results table.",
            )
            return
        columns = [name.strip() for name in
                   self._dock_rescore_columns_entry.get().split(",")
                   if name.strip()]
        if not columns:
            messagebox.showerror("Input", "name at least one evidence column")
            return
        weights: list[float] | None = None
        raw_weights = self._dock_rescore_weights_entry.get().strip()
        if raw_weights:
            try:
                weights = [parse_float(raw, "weight", minimum=0.0, maximum=100.0)
                           for raw in raw_weights.split(",")]
            except ValueError as exc:
                messagebox.showerror("Input", str(exc))
                return
        higher_is_better: dict[str, bool] = {}
        raw_higher = self._dock_rescore_higher_entry.get().strip()
        if raw_higher:
            higher_is_better = {name.strip(): True for name in
                                raw_higher.split(",") if name.strip()}

        def work() -> Any:
            from pharmaml.docking import consensus_breakdown

            breakdown = consensus_breakdown(
                frame, columns, weights=weights,
                higher_is_better=higher_is_better or None,
            )
            rescored = frame.copy()
            for column in breakdown.columns:
                rescored[column] = breakdown[column].to_numpy()
            return rescored.sort_values(
                "consensus", ascending=False, na_position="last"
            ).reset_index(drop=True)

        def render(rescored: Any) -> None:
            self._dock_results_frame = rescored
            self._dock_results_source = "rescore"
            source_label = (self._dock_results_info.cget("text")
                            if self._dock_results_info.winfo_exists()
                            else "loaded table")
            self._dock_rescore_last = (
                rescored, columns, weights, dict(higher_is_better), source_label,
            )
            tree = self._dock_results_tree
            tree.delete(*tree.get_children())
            identity = str(rescored.columns[0])
            norm_columns = [f"{column}_norm" for column in columns]
            show = [identity] + [column for column in norm_columns + ["consensus"]
                                 if column != identity]
            tree["columns"] = show
            for column in show:
                tree.heading(column, text=column)
                tree.column(column, width=150 if column == identity else 130,
                            minwidth=80, stretch=(column == identity))
            for row in rescored.to_dict("records"):
                values = [
                    _cell_text(None if (isinstance(row[column], float)
                                        and math.isnan(row[column]))
                               else row[column], limit=40)
                    for column in show
                ]
                tree.insert("", "end", values=values)
            # The introspection lines, the CLI's recipe + verdict.
            directions = higher_is_better or {}
            recipe = ", ".join(
                f"{column} ("
                + ("higher" if directions.get(
                    column, not column.endswith(("energy", "affinity")))
                   else "lower")
                + "=better"
                + (f", weight {weight:g}" if weights else "")
                + ")"
                for column, weight in zip(columns, weights or [1.0] * len(columns))
            )
            lines = [f"evidence: {recipe}"]
            lines.append(
                f"{len(rescored)} row(s) rescored -- consensus = weighted "
                "mean of the _norm columns, higher is better"
            )
            valid = rescored["consensus"].dropna()
            if len(valid):
                # Column-wise access keeps each dtype: .loc[scalar] would
                # upcast the whole row Series to float64 and render the
                # integer identity ("rank 4") as "4.0".
                top_index = int(valid.idxmax())
                identity_value = rescored[identity].iloc[top_index]
                top_consensus = float(rescored["consensus"].iloc[top_index])
                top_norms = {
                    column: float(rescored[column].iloc[top_index])
                    for column in norm_columns
                }
                driver = max(top_norms, key=lambda name: top_norms[name])
                lines.append(
                    f"{identity_value} tops the table (consensus "
                    f"{top_consensus:.3f}): {driver} = "
                    f"{top_norms[driver]:.3f} is the driving evidence"
                )
            self._dock_rescore_text.delete("1.0", "end")
            self._dock_rescore_text.insert("1.0", "\n".join(lines))
            self._dock_rescore_info.configure(
                text=f"rescored · {len(rescored)} row(s) · {len(columns)} "
                     f"evidence column(s) -- Save results CSV… writes the table",
                style="Field.TLabel",
            )
            self._dock_results_info.configure(
                text=f"{len(rescored)} row(s) · consensus ranking "
                     f"({len(columns)} evidence columns)",
                style="Field.TLabel",
            )

        self._start_job("rescoring the loaded table", work, render)

    def _dock_save_report(self) -> None:
        """Write the last rescore as the one-click visual bundle.

        The same ``report.html`` + figures + tables bundle the CLI's
        ``docking rescore --plots`` writes (plus the multi-page
        ``report.pdf`` when the checkbox is ticked -- the
        ``--pdf-report`` twin), rebuilt from the stored rescore and
        saved into a folder the user picks.  Figures are created under
        the package's forced-Agg backend inside the worker thread; they
        never touch Tk.
        """
        if not self._require_pharmaml() or not self._require(
                "matplotlib", "the report figures"):
            return
        if self._dock_rescore_last is None:
            messagebox.showwarning(
                "No rescore yet",
                "Rescore a table first -- the report needs the explainable "
                "leaderboard of the last rescore.",
            )
            return
        want_pdf = bool(self._dock_pdf_var.get())
        if want_pdf and not fpdf_available():
            messagebox.showwarning(
                "PDF needs fpdf",
                "The multi-page PDF document needs fpdf -- install it "
                "with:  pip install 'pharmaml[reports]'\n"
                "(the HTML report works without it -- untick the box to "
                "continue).",
            )
            return
        if filedialog is None:  # pragma: no cover - headless fallback
            return
        target = filedialog.askdirectory(title="Choose the report folder")
        if not target:
            return
        rescored, columns, weights, higher_is_better, source_label = (
            self._dock_rescore_last
        )
        directions = higher_is_better or {}

        def work() -> Any:
            from pharmaml.exceptions import DataValidationError
            from pharmaml.viz import (
                plot_consensus_ranking,
                plot_energy_distribution,
                save_visual_bundle,
            )

            figures: dict[str, Any] = {
                "consensus_ranking": plot_consensus_ranking(rescored)
            }
            try:
                # scan-shaped tables also carry the energy landscape; a
                # consensus-only table simply skips it (the CLI's skip
                # pattern, byte for byte).
                figures["energy_distribution"] = plot_energy_distribution(
                    rescored
                )
            except DataValidationError:
                pass
            recipe = ", ".join(
                f"{column} ("
                + ("higher" if directions.get(
                    column, not column.endswith(("energy", "affinity")))
                   else "lower")
                + "=better"
                + (f", weight {weight:g}" if weights else "")
                + ")"
                for column, weight in zip(columns, weights or [1.0] * len(columns))
            )
            return save_visual_bundle(
                figures,
                tables={"consensus_breakdown": rescored},
                out_dir=target,
                title=(f"playground docking rescore — "
                       f"{len(columns)} evidence column(s)"),
                subtitle=f"{recipe} — {source_label}",
                pdf_report=want_pdf,
            )

        def render(written: Any) -> None:
            names = [path.name for path in written]
            extra = " + report.pdf" if want_pdf else ""
            self._job_label.configure(
                text=f"report saved · {target} ({len(written)} files{extra})"
            )
            messagebox.showinfo(
                "Report saved",
                f"{len(written)} files written to {target}:\n"
                + "\n".join(names[:8])
                + ("\n…" if len(names) > 8 else ""),
            )

        self._start_job("writing the report bundle", work, render)

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
        if engine in ("AutoDock4 campaign", "autodock4"):
            hint = (
                f"AutoDock4 docking needs the {binaries} binaries on "
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

    # -- SMILES array docking card ------------------------------------- #
    def _dock_smiles_example(self) -> None:
        """Refill the array box with the bundled trivial-name drugs."""
        self._dock_smiles_text.delete("1.0", "end")
        self._dock_smiles_text.insert("1.0", "\n".join(_DOCK_SMILES_DEFAULTS))
        self._dock_smiles_summary.configure(
            text=f"{len(_DOCK_SMILES_DEFAULTS)} example SMILES loaded")

    def _dock_smiles_open_receptor(self) -> None:
        path = filedialog.askopenfilename(
            title="Open the receptor PDBQT",
            filetypes=[("PDBQT receptor", "*.pdbqt"), ("All files", "*.*")],
        )
        if not path:
            return
        self._dock_smiles_receptor = Path(path)
        self._dock_smiles_receptor_info.configure(text=path)

    def _dock_smiles_choose_out(self) -> None:
        path = filedialog.askdirectory(
            title="Choose the SMILES-array campaign output folder")
        if not path:
            return
        self._dock_smiles_out = Path(path)
        self._dock_smiles_out_info.configure(text=f"campaign output: {path}")

    def _dock_smiles_receptor_or_none(self) -> Path | None:
        """The array card's receptor: its own pick, or the prepared one."""
        if self._dock_smiles_receptor is not None:
            return self._dock_smiles_receptor
        return self._dock_receptor_path or self._dock_vina_receptor

    def _dock_smiles_run(self) -> None:
        """Dock the whole pasted SMILES array with the chosen engine."""
        if not self._require_pharmaml() or not self._require(
                "pandas", "SMILES docking") or not self._require(
                "rdkit", "SMILES docking"):
            return
        smiles = parse_smiles_lines(self._dock_smiles_text.get("1.0", "end"))
        if not smiles:
            messagebox.showwarning(
                "Empty array", "Paste at least one SMILES (one per line)."
            )
            return
        receptor = self._dock_smiles_receptor_or_none()
        if receptor is None:
            messagebox.showwarning(
                "No receptor",
                "Pick a prepared receptor PDBQT first (or prepare one in "
                "the receptor card above).",
            )
            return
        if self._dock_smiles_out is None:
            messagebox.showwarning(
                "No output folder",
                "Choose the campaign output folder first -- the prepared "
                "ligands and engine outputs land there.",
            )
            return
        engine = self._dock_smiles_engine_combo.get()
        binaries = docking_engine_binaries(engine)
        missing = [name for name in binaries if shutil.which(name) is None]
        if missing:
            self._dock_missing_engine_dialog(engine)
            return
        try:
            center = parse_float_triple(self._dock_smiles_center_entry.get(),
                                        "centre")
            size = parse_float_triple(self._dock_smiles_size_entry.get(), "size")
            exhaustiveness = parse_int(self._dock_smiles_exhaust_entry.get(),
                                       "exhaustiveness", minimum=1, maximum=256)
            workers = parse_int(self._dock_smiles_workers_entry.get(),
                                "workers", minimum=1, maximum=64)
        except ValueError as exc:
            messagebox.showerror("Input", str(exc))
            return
        out_root = self._dock_smiles_out
        receptor_path = receptor

        def work(report: Any) -> Any:
            from pharmaml.docking import dock_smiles_array

            return dock_smiles_array(
                smiles,
                receptor_path,
                out_root,
                engine=engine,
                center=center,
                size=size,
                exhaustiveness=exhaustiveness,
                workers=workers,
                progress=report,
            )

        def render(frame: Any) -> None:
            self._dock_smiles_last = frame
            self._dock_results_frame = frame
            self._dock_results_source = "smiles"
            tree = self._dock_results_tree
            tree.delete(*tree.get_children())
            columns = [str(column) for column in frame.columns]
            tree["columns"] = columns
            for column in columns:
                width = 210 if column in ("smiles", "error") else 96
                tree.heading(column, text=column)
                tree.column(column, width=width, minwidth=56,
                            stretch=(column in ("smiles", "error")))
            for row in frame.to_dict("records"):
                values = [
                    _cell_text(None if (isinstance(row[column], float)
                                        and math.isnan(row[column]))
                               else row[column], limit=52)
                    for column in columns
                ]
                tree.insert("", "end", values=values)
            ok = int(frame["ok"].sum())
            best = frame.loc[frame["ok"], "best_score"]
            best_text = f" · best {best.min():.2f} kcal/mol" if ok else ""
            self._dock_results_info.configure(
                text=(f"{len(frame)} SMILES docked ({engine}) · {ok} ok · "
                      f"{len(frame) - ok} failed{best_text} — {out_root.name}"),
                style="Field.TLabel",
            )
            self._dock_smiles_summary.configure(
                text=(f"{ok}/{len(frame)} docked{best_text} · folders under "
                      f"{out_root.name}"),
                style="Field.TLabel",
            )
            self._dock_rescore_prefill()

        self._start_job(
            f"docking {len(smiles)} SMILES ({engine}, {workers} worker(s))",
            work, render,
        )

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
        # manual model settings (100x-106): the CV fold count the
        # Train button threads into train_model (tune on) and the
        # leaderboard button's own 3-fold convention stays untouched.
        self._ml_cv_spin = ttk.Spinbox(config, from_=2, to=10, increment=1,
                                       width=10)
        self._ml_cv_spin.set("5")
        Tooltip(self._ml_cv_spin,
                "Cross-validation folds for Train & evaluate with the tune "
                "tick on (and the CV columns of the saved report). The "
                "leaderboard button keeps its own 3-fold convention.")
        form_row(config, "cv folds", self._ml_cv_spin)
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
        self._ml_pdf_var = tk.BooleanVar(value=False)
        pdf_check = ttk.Checkbutton(
            buttons, text="include report.pdf", variable=self._ml_pdf_var)
        Tooltip(pdf_check, "Also write the multi-page PDF document -- "
                           "needs fpdf (pharmaml[reports])")
        pdf_check.pack(side="left", padx=(10, 0))
        report_btn = self._action_button(
            buttons, "Save report…", self._ml_save_report,
            tooltip="Write the last training run as the one-click visual "
                    "bundle -- report.html + figures + tables (+ report.pdf "
                    "when ticked), the CLI --plots/--pdf-report story.")
        report_btn.pack(side="left", padx=(10, 0))
        check_btn = self._action_button(
            buttons, "Check CSV…", self._ml_check_csv,
            tooltip="Audit any CSV against the last trained model before "
                    "predicting -- missing features, extra columns, "
                    "reordering and version drift, all named (the ml check "
                    "story, mouse edition; the training target is dropped "
                    "automatically when present).")
        check_btn.pack(side="left", padx=(10, 0))

        # picking a concrete task re-lists the models for it, and any
        # model selection rebuilds the manual-hyperparameter card
        # (100x-106): programmatic set() never fires <<ComboboxSelected>>,
        # so every code path that moves these combos calls the refresh
        # itself (load, apply-best, restore, reset).
        self._task_combo.bind("<<ComboboxSelected>>", self._ml_task_changed)
        self._model_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._ml_refresh_hyperparam_panel(),
        )

        manual = make_card(
            page.grid, "Manual hyperparameters", span=2, row=3, column=0,
            subtitle="The selected model's registry knobs, editable by hand "
                     "— each field starts at the estimator's own default; "
                     "blank keeps that default. Pinned keys are excluded "
                     "from the grid search, so tick 'tune' to search only "
                     "the slots you leave blank. Per-model values persist "
                     "with the session.",
        )
        self._ml_params_var = tk.StringVar(value="{}")
        self._ml_params_host = ttk.Frame(manual, style="Card.TFrame")
        grid_append(manual, self._ml_params_host, sticky="ew")
        self._ml_param_entries: dict[str, ttk.Entry] = {}
        self._ml_params_status = ttk.Label(
            manual, text="pick a model to edit its hyperparameters",
            style="CardMuted.TLabel", wraplength=620, justify="left")
        grid_append(manual, self._ml_params_status, sticky="w", pady=(6, 0))
        manual_buttons = ttk.Frame(manual, style="Card.TFrame")
        grid_append(manual, manual_buttons, sticky="w", pady=(8, 0))
        reset_params_btn = self._action_button(
            manual_buttons, "Reset to defaults", self._ml_params_reset,
            tooltip="Clear every field back to the estimator's own "
                    "defaults (the blank state — nothing is pinned, the "
                    "grid search gets all its slots back).")
        reset_params_btn.pack(side="left")

        tuner = make_card(
            page.grid, "Hyperparameter search", span=2, row=4, column=0,
            subtitle="The ml tune story, mouse edition: a keras-tuner-style "
                     "search over a whole estimator family (not the "
                     "per-model grid search above). RandomSearch samples "
                     "full-budget trials; Hyperband starts brackets of "
                     "configs at a fraction of the resource budget (ann's "
                     "max_iter, dnn's epochs -- the ladder is the family's "
                     "own declared grid) and keeps only the top 1/η per "
                     "rung. Every trial is scored by mean CV on the SAME "
                     "shuffled folds; the family's defaults are scored too "
                     "-- the number the search has to beat; the winner is "
                     "refit and scored on the held-out 20%."
        )
        self._tune_estimator_combo = ttk.Combobox(tuner, state="readonly", width=26,
                                                  values=_TUNE_ESTIMATORS)
        self._tune_estimator_combo.set("ann")
        Tooltip(self._tune_estimator_combo,
                "ann = the sklearn MLP hypermodel; dnn = the torch deep MLP "
                "(batchnorm searchable, needs torch); hsvr = the hierarchical "
                "SVR (regression only). The same families the CLI's "
                "ml tune --estimator accepts.")
        form_row(tuner, "estimator", self._tune_estimator_combo)
        self._tune_searcher_combo = ttk.Combobox(
            tuner, state="readonly", width=26, values=_TUNE_SEARCHERS)
        self._tune_searcher_combo.set("random")
        Tooltip(self._tune_searcher_combo,
                "random = RandomSearch, every trial at the full budget "
                "(the trials spinbox decides how many); hyperband = "
                "successive halving over the family's resource slot -- "
                "brackets start many configs at a small budget and keep "
                "only the top 1/η per rung, so the trials spinbox is "
                "ignored (the brackets decide; ann and dnn only -- hsvr "
                "declares no resource slot). The same tokens the CLI's "
                "ml tune --searcher accepts.")
        form_row(tuner, "searcher", self._tune_searcher_combo)
        self._tune_trials = ttk.Spinbox(tuner, from_=4, to=40, increment=1, width=10)
        self._tune_trials.set("8")
        Tooltip(self._tune_trials,
                "RandomSearch only: how many full-budget trials to draw. "
                "Hyperband ignores this -- its brackets fix the trial "
                "count from η and the resource ladder.")
        form_row(tuner, "trials", self._tune_trials)
        self._tune_eta = ttk.Spinbox(tuner, from_=2, to=4, increment=1, width=10)
        self._tune_eta.set("3")
        Tooltip(self._tune_eta,
                "Hyperband only: the halving factor η -- every rung keeps "
                "the top ceil(n/η) configs and multiplies their budget by "
                "η. Bigger η = more brackets, more trials at tiny budgets; "
                "smaller η = fewer, deeper brackets. RandomSearch ignores "
                "this.")
        form_row(tuner, "η (hyperband)", self._tune_eta)
        self._tune_cv = ttk.Spinbox(tuner, from_=2, to=5, increment=1, width=10)
        self._tune_cv.set("3")
        form_row(tuner, "cv folds", self._tune_cv)
        tune_buttons = ttk.Frame(tuner, style="Card.TFrame")
        grid_append(tuner, tune_buttons, sticky="w", pady=(8, 0))
        search_btn = self._action_button(
            tune_buttons, "Run search", self._ml_tune,
            tooltip="Search the chosen family's space -- RandomSearch "
                    "samples full-budget trials, Hyperband halves the "
                    "resource rung by rung; trials ranked by mean CV score "
                    "on the leaderboard card below; the summary line "
                    "carries best vs defaults (the gain) and the held-out "
                    "verdict. Deterministic end-to-end (the same seed "
                    "folds the same way twice).")
        search_btn.pack(side="left")
        apply_btn = self._action_button(
            tune_buttons, "Apply best", self._ml_tune_apply,
            tooltip="Carry the winning configuration forward: ann/hsvr point "
                    "the model combobox at the family; dnn prefills the whole "
                    "Deep Learning tab (layers, dropout, learning rate, "
                    "activation, batchnorm, epochs) and switches to it.")
        apply_btn.pack(side="left", padx=(10, 0))
        save_trials_btn = self._action_button(
            tune_buttons, "Save trials CSV…", self._ml_tune_save_trials,
            tooltip="Write the full trials table (one row per trial: score, "
                    "std, every hyperparameter) -- the CLI --out story.")
        save_trials_btn.pack(side="left", padx=(10, 0))
        self._tune_summary = ttk.Label(
            tuner, text="no search yet — pick an estimator family and run it",
            style="CardMuted.TLabel", wraplength=620, justify="left")
        grid_append(tuner, self._tune_summary, sticky="w", pady=(6, 0))

        results = make_card(page.grid, "Results", span=2, row=5, column=0)
        self._ml_metrics = ttk.Frame(results, style="Card.TFrame")
        grid_append(results, self._ml_metrics, sticky="ew")
        ttk.Label(self._ml_metrics, text="train a model to see metrics",
                  style="CardMuted.TLabel").pack(anchor="w")
        self._ml_host = self._figure_host(results, height=330)

        board = make_card(
            page.grid, "Leaderboard", span=2, row=6, column=0,
            subtitle="Compare models fills it with the cross-validated "
                     "registry ranking; a hyperparameter search fills it "
                     "with the trial ranking, best first."
        )
        self._board_status = ttk.Label(
            board, text="empty — run Compare models or a hyperparameter search",
            style="CardMuted.TLabel", wraplength=620, justify="left")
        grid_append(board, self._board_status, sticky="w")
        board_frame = ttk.Frame(board, style="Card.TFrame")
        grid_append(board, board_frame, sticky="nsew", weight=True)
        self._board_tree = ttk.Treeview(board_frame, show="headings", height=6)
        board_bar = ttk.Scrollbar(board_frame, orient="vertical",
                                  command=self._board_tree.yview)
        self._board_tree.configure(yscrollcommand=board_bar.set)
        self._board_tree.pack(side="left", fill="both", expand=True)
        board_bar.pack(side="right", fill="y")

        # -- ensemble builder (voting + stacking, members by hand) ---- #
        ensemble = make_card(
            page.grid, "Ensemble builder (voting & stacking)", span=2, row=7,
            column=0,
            subtitle="Combine several registry models into one ensemble: "
                     "voting averages the members' predictions (soft = "
                     "probability averaging, hard = majority vote; "
                     "regression always averages), stacking trains a "
                     "meta-learner on the members' cross-validated "
                     "predictions. The same ensembles are registry models "
                     "too (pick 'voting' / 'stacking' in the model "
                     "combobox above for the default RF + SVM + KNN trio).",
        )
        ens_members = ttk.Frame(ensemble, style="Card.TFrame")
        grid_append(ensemble, ens_members, sticky="ew")
        self._ens_member_vars: dict[str, Any] = {}
        for index, (token, label) in enumerate(_ENSEMBLE_MEMBER_LABELS):
            member_var = tk.BooleanVar(value=token in _ENSEMBLE_DEFAULT_MEMBERS)
            self._ens_member_vars[token] = member_var
            member_check = ttk.Checkbutton(
                ens_members, text=label, variable=member_var)
            member_check.grid(row=index // 6, column=index % 6, sticky="w",
                              padx=(0, 16), pady=2)
        ens_controls = ttk.Frame(ensemble, style="Card.TFrame")
        grid_append(ensemble, ens_controls, sticky="ew", pady=(6, 0))
        ens_controls.columnconfigure(1, weight=1)
        self._ens_kind_combo = ttk.Combobox(
            ens_controls, state="readonly", width=18, values=("voting", "stacking"))
        self._ens_kind_combo.set("voting")
        Tooltip(self._ens_kind_combo,
                "voting = averaged predictions (the Voting.py family); "
                "stacking = a meta-learner over the members' "
                "cross-validated predictions (the Stacking.py family).")
        form_row(ens_controls, "kind", self._ens_kind_combo)
        self._ens_voting_combo = ttk.Combobox(
            ens_controls, state="readonly", width=18, values=("soft", "hard"))
        self._ens_voting_combo.set("soft")
        Tooltip(self._ens_voting_combo,
                "Classification only: soft averages class probabilities, "
                "hard takes the majority vote. Regression ignores this.")
        form_row(ens_controls, "voting rule", self._ens_voting_combo)
        self._ens_meta_combo = ttk.Combobox(
            ens_controls, state="readonly", width=18)
        self._ens_meta_combo.set("auto")
        Tooltip(self._ens_meta_combo,
                "Stacking only: the level-1 blender -- 'auto' picks "
                "logistic_regression (classification) or ridge "
                "(regression).")
        form_row(ens_controls, "meta model", self._ens_meta_combo)
        self._ens_meta_combo["values"] = ["auto"] + [
            token for token, _label in _ENSEMBLE_MEMBER_LABELS
        ]
        ens_buttons = ttk.Frame(ens_controls, style="Card.TFrame")
        ens_buttons.grid(row=ens_controls.grid_size()[1], column=0,
                         columnspan=2, sticky="w", pady=(8, 0))
        ens_train_btn = self._action_button(
            ens_buttons, "Train ensemble", self._ens_train, primary=True,
            tooltip="Builds the ensemble from the ticked members with the "
                    "chosen rule / meta-learner, fits it on the same 80/20 "
                    "split as Train & evaluate, and renders the held-out "
                    "metrics + figure in the Results card (Save report… "
                    "and Save model bundle… then work on it too).",
        )
        ens_train_btn.pack(side="left")
        self._ens_summary = ttk.Label(
            ens_buttons, text="—", style="CardMuted.TLabel",
            wraplength=520, justify="left")
        self._ens_summary.pack(side="left", padx=(10, 0))

        # -- model persistence (save trained bundles, reuse them) ----- #
        persist = make_card(
            page.grid, "Model persistence (save & reuse)", span=2, row=8,
            column=0,
            subtitle="One half saves the last trained model as a "
                     "provenance-carrying bundle (joblib + metadata: "
                     "feature order, hyperparameters, metrics, library "
                     "versions, timestamp); the other half loads any "
                     "bundle back, audits new data against its feature "
                     "contract and predicts.",
        )
        persist.columnconfigure(0, weight=1)
        persist.columnconfigure(1, weight=1)
        save_side = ttk.Frame(persist, style="Card.TFrame")
        save_side.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        ttk.Label(save_side, text="Save the last trained model",
                  style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(save_side,
                  text="writes <name>.joblib + a human-readable "
                       "<name>.json sidecar",
                  style="CardMuted.TLabel", wraplength=430,
                  justify="left").pack(anchor="w", pady=(2, 0))
        save_bundle_btn = self._action_button(
            save_side, "Save model bundle…", self._ml_save_model_bundle,
            primary=True,
            tooltip="The ml save CLI story, mouse edition: the fitted "
                    "estimator plus its provenance metadata (name, task, "
                    "feature order, best hyperparameters, held-out "
                    "metrics, pharmaml/sklearn/numpy versions, UTC "
                    "timestamp) -- everything a model loaded two years "
                    "later needs to know about itself.",
        )
        save_bundle_btn.pack(anchor="w", pady=(8, 0))
        self._ml_save_info = ttk.Label(
            save_side, text="no run yet — train a model (or an ensemble) first",
            style="CardMuted.TLabel", wraplength=430, justify="left")
        self._ml_save_info.pack(anchor="w", pady=(4, 0))

        load_side = ttk.Frame(persist, style="Card.TFrame")
        load_side.grid(row=0, column=1, sticky="nsew")
        ttk.Label(load_side, text="Load a bundle & predict",
                  style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(load_side,
                  text="any bundle this toolkit (or the CLI) ever wrote",
                  style="CardMuted.TLabel", wraplength=430,
                  justify="left").pack(anchor="w", pady=(2, 0))
        load_buttons = ttk.Frame(load_side, style="Card.TFrame")
        load_buttons.pack(anchor="w", pady=(8, 0))
        load_bundle_btn = self._action_button(
            load_buttons, "Open model bundle…", self._ml_load_model_bundle,
            tooltip="Loads the estimator and its metadata sidecar; the "
                    "card shows the provenance summary (name, task, "
                    "trained-at, versions, feature count).")
        load_bundle_btn.pack(side="left")
        predict_csv_btn = self._action_button(
            load_buttons, "Predict on CSV…", self._ml_predict_with_bundle,
            tooltip="Opens a CSV, audits every column against the "
                    "bundle's recorded feature contract (missing / extra "
                    "/ reordered features are named up front), predicts "
                    "with the loaded estimator and shows the result "
                    "table; Save predictions CSV… writes it.")
        predict_csv_btn.pack(side="left", padx=(10, 0))
        self._ml_bundle_info = ttk.Label(
            load_side, text="no bundle loaded", style="CardMuted.TLabel",
            wraplength=430, justify="left")
        self._ml_bundle_info.pack(anchor="w", pady=(4, 0))
        predict_buttons = ttk.Frame(load_side, style="Card.TFrame")
        predict_buttons.pack(anchor="w", pady=(4, 0))
        save_pred_btn = self._action_button(
            predict_buttons, "Save predictions CSV…", self._ml_save_predictions,
            tooltip="The last bundle prediction table -- input columns "
                    "plus the prediction column (probabilities too for "
                    "classifiers).")
        save_pred_btn.pack(side="left")
        self._predict_tree_frame = ttk.Frame(load_side, style="Card.TFrame")
        self._predict_tree_frame.pack(fill="both", expand=True, pady=(6, 0))
        self._predict_tree = ttk.Treeview(self._predict_tree_frame,
                                          show="headings", height=7)
        predict_bar = ttk.Scrollbar(self._predict_tree_frame, orient="vertical",
                                    command=self._predict_tree.yview)
        self._predict_tree.configure(yscrollcommand=predict_bar.set)
        self._predict_tree.pack(side="left", fill="both", expand=True)
        predict_bar.pack(side="right", fill="y")

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
        chosen = default_target_column(frame) or columns[-1]
        self._target_combo.set(chosen)
        self._ml_info.configure(
            text=f"{path.name} — {len(frame):,} rows × {len(columns)} columns"
        )
        self._refresh_model_choices(self._infer_task(chosen, frame))

    def _infer_task(self, target: str, frame: Any = None) -> str:
        """Classify a target column as a regression or a classification label.

        ``frame`` defaults to the Data & ML tab's dataset; the Deep
        Learning tab passes its own so both loaders share one rule.
        """
        if frame is None:
            frame = self._ml_frame
        if frame is None:
            return "regression"
        column = frame[target]
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
        self._ml_refresh_hyperparam_panel()

    def _ml_task_changed(self, _event: Any = None) -> None:
        """A concrete task pick re-lists its models and the manual knobs."""
        task = self._task_combo.get()
        if task != "auto":
            self._refresh_model_choices(task)
        self._ml_refresh_hyperparam_panel()

    def _ml_params_saved(self) -> dict[str, dict[str, str]]:
        """The persisted per-model manual values (``{model: {param: raw}}``).
        One JSON string variable (``_ml_params_var``) carries every
        model's manual values, so the session machinery saves and
        restores them with zero extra plumbing — and switching models
        back and forth never loses an edit.
        """
        try:
            payload = json.loads(self._ml_params_var.get() or "{}")
        except ValueError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(model): {str(key): str(value) for key, value in values.items()}
            for model, values in payload.items()
            if isinstance(values, dict)
        }

    def _ml_sync_params_var(self) -> None:
        """Persist the CURRENT model's fields into the JSON variable."""
        payload = self._ml_params_saved()
        model = self._model_combo.get()
        values = {
            key: entry.get().strip()
            for key, entry in getattr(self, "_ml_param_entries", {}).items()
            if entry.get().strip()
        }
        if values:
            payload[model] = values
        else:
            payload.pop(model, None)
        self._ml_params_var.set(json.dumps(payload, sort_keys=True))

    def _ml_refresh_hyperparam_panel(self) -> None:
        """Rebuild the manual-hyperparameter rows for the selected model.

        The registry's own ``param_grid`` decides WHICH knobs are
        editable (the toolkit's curated tunable set), and the freshly
        built estimator supplies each default — so the card can never
        drift from the models it configures.  A model with an empty
        grid says so instead of pretending.
        """
        host = getattr(self, "_ml_params_host", None)
        if host is None:  # built before the ML tab exists (safety)
            return
        for child in host.winfo_children():
            child.destroy()
        self._ml_param_entries = {}
        selection = self._model_combo.get()
        task = self._task_combo.get()
        if task == "auto" and self._ml_frame is not None:
            target = self._target_combo.get()
            if target in [str(column) for column in self._ml_frame.columns]:
                task = self._infer_task(target)
        if selection == "auto" or task not in ("regression", "classification"):
            self._ml_params_status.configure(
                text="pick a model to edit its hyperparameters"
                     + ("" if selection != "auto"
                        else " (the model combobox above)"))
            return
        try:
            from pharmaml.ml.models import (
                MODEL_REGISTRY,
                build_model,
                resolve_model_name,
            )

            spec = MODEL_REGISTRY[resolve_model_name(selection)]
            defaults = build_model(selection, task).get_params()
            grid = dict(spec.param_grid)
        except Exception as exc:  # noqa: BLE001 - unusable model, say so
            self._ml_params_status.configure(
                text=f"{selection} could not be built here ({exc}) — "
                     "its defaults are used as-is")
            return
        if not grid:
            self._ml_params_status.configure(
                text=f"{spec.name} exposes no registry search grid — its "
                     "estimator defaults are used as-is")
            return
        saved = self._ml_params_saved().get(selection, {})
        for index, key in enumerate(grid):
            row = ttk.Frame(host, style="Card.TFrame")
            row.grid(row=index, column=0, sticky="ew", pady=(0, 4))
            ttk.Label(row, text=key, style="Field.TLabel").grid(
                row=0, column=0, sticky="w")
            entry = ttk.Entry(row, width=16)
            entry.grid(row=0, column=1, sticky="w", padx=(10, 0))
            # BLANK means "keep the estimator's default" — the hint
            # label names that default, so the field itself stays empty
            # unless the user pins something (only edited keys pin, and
            # only unpinned keys keep their grid-search slots).
            prefill = saved.get(key, "")
            if prefill:
                entry.insert(0, prefill)
            entry.bind("<KeyRelease>", lambda _e: self._ml_sync_params_var())
            self._ml_param_entries[key] = entry
            grid_hint = " · ".join(_cell_text(value, limit=14) for value in grid[key])
            default_text = _cell_text(defaults.get(key, ""), limit=14)
            ttk.Label(
                row,
                text=f"default {default_text} · grid {grid_hint}",
                style="CardMuted.TLabel", wraplength=430, justify="left",
            ).grid(row=0, column=2, sticky="w", padx=(10, 0))
        pinned = sum(1 for key, entry in self._ml_param_entries.items()
                     if entry.get().strip())
        self._ml_params_status.configure(
            text=f"{spec.name} · {len(grid)} knob(s) · blank = default · "
                 f"{pinned} pinned · pinned keys are excluded from the "
                 "grid search")

    def _ml_params_reset(self) -> None:
        """The manual card's own reset — every field back to default.

        Drops the current model's saved values from the JSON variable
        and rebuilds the panel, which prefills each field with the
        estimator's own default — the exact state a fresh selection
        shows (never blank-with-hidden-values).
        """
        payload = self._ml_params_saved()
        payload.pop(self._model_combo.get(), None)
        self._ml_params_var.set(json.dumps(payload, sort_keys=True))
        self._ml_refresh_hyperparam_panel()
        if getattr(self, "_ml_param_entries", {}):
            self._ml_params_status.configure(
                text="manual values cleared — estimator defaults in effect")

    def _ml_manual_params(self, model_name: str, task: str) -> dict[str, Any] | None:
        """Parse the manual fields into ``train_model``'s pinned values.

        Blank fields keep their default and are simply not pinned; a
        field that fails its type parse raises :class:`ValueError`
        naming the parameter — the pre-flight convention every other
        input on the tab follows.
        """
        entries = getattr(self, "_ml_param_entries", {})
        if not entries or model_name == "auto":
            return None
        try:
            from pharmaml.ml.models import build_model

            defaults = build_model(model_name, task).get_params()
        except Exception:  # noqa: BLE001 - degrade to defaults, never block
            return None
        params: dict[str, Any] = {}
        for key, entry in entries.items():
            raw = entry.get().strip()
            if not raw:
                continue
            try:
                params[key] = parse_typed_value(raw, defaults.get(key))
            except ValueError as exc:
                raise ValueError(f"{key}: {exc}") from None
        return params or None

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
        # pre-flight (100x-62): same rule as the Deep Learning tab -- a
        # zero-numeric-feature dataset fails BEFORE the job starts.
        numeric, dropped = numeric_feature_columns(self._ml_frame, target)
        if not numeric:
            messagebox.showwarning(
                "No numeric features",
                "Every non-target column is non-numeric"
                + (f" ({', '.join(dropped)})" if dropped else "")
                + ".  The models need numeric feature columns -- compute "
                "descriptors first (the Chemistry tab's feature factory).",
            )
            return
        task = self._task_combo.get()
        if task == "auto":
            task = self._infer_task(target)
        model_name = self._model_combo.get()
        if model_name == "auto":
            model_name = "ridge" if task == "regression" else "logistic_regression"
        tune = bool(self._tune_var.get())
        # manual hyperparameters + cv folds (100x-106): parsed BEFORE
        # the job starts, naming the offending field -- the tab's
        # pre-flight convention.
        try:
            model_params = self._ml_manual_params(model_name, task)
            cv = parse_int(self._ml_cv_spin.get(), "cv folds",
                           minimum=2, maximum=10)
        except ValueError as exc:
            messagebox.showwarning("Check the manual settings", str(exc))
            return

        def work() -> Any:
            from pharmaml.ml import evaluate_model, train_model
            from pharmaml.ml.preprocessing import make_train_test_split

            frame = self._ml_frame
            y = frame[target]
            numeric, dropped = numeric_feature_columns(frame, target)
            x_train, x_test, y_train, y_test = make_train_test_split(
                frame[numeric], y, random_state=42
            )
            trained = train_model(
                model_name, task, x_train, y_train, tune=tune, cv=cv, n_jobs=1,
                model_params=model_params,
            )
            report = evaluate_model(
                trained.model, x_test, y_test, task,
                keep_predictions=(task == "regression"),
            )
            return trained, report, task, dropped, y_test, x_test

        def render(payload: Any) -> None:
            trained, report, task, dropped, y_test, x_test = payload
            # the Save-report button rebuilds the CLI bundle from this run;
            # the Check-CSV button audits new data against it (the training
            # target rides along as the automatic --target drop).
            self._ml_last_run = (trained, report, task, x_test, y_test, target)
            gp = self._ml_gp_sigma(trained, x_test)
            self._fill_ml_metrics(trained, report, task, dropped, gp=gp)
            if not self.deps.get("matplotlib"):
                return
            try:
                from pharmaml.viz import (
                    plot_confusion_matrix,
                    plot_gp_uncertainty,
                    plot_predicted_vs_actual,
                )

                if task == "regression":
                    if gp is not None:
                        # GP runs get their own honest readout: the parity
                        # scatter with ±1.96·σ error bars and the coverage
                        # annotation (plot_gp_uncertainty draws both).
                        figure = plot_gp_uncertainty(
                            y_test,
                            gp[0],
                            gp[1],
                            metrics={
                                "R²": report.r2,
                                "RMSE": report.rmse,
                                "MAE": report.mae,
                            },
                        )
                    else:
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

    # -- ensemble builder card ------------------------------------------ #
    def _ens_train(self) -> None:
        """Train a voting / stacking ensemble of the ticked members."""
        if not self._require_pharmaml():
            return
        if not self._require("pandas", "ensemble training") or not self._require(
                "sklearn", "ensemble training"
        ):
            return
        if self._ml_frame is None:
            messagebox.showwarning("No dataset", "Open a CSV or load an example first.")
            return
        target = self._target_combo.get()
        if not target:
            messagebox.showwarning("No target", "Pick the column to predict.")
            return
        numeric, dropped = numeric_feature_columns(self._ml_frame, target)
        if not numeric:
            messagebox.showwarning(
                "No numeric features",
                "Every non-target column is non-numeric.  Compute descriptors "
                "first (the Chemistry tab's feature factory).",
            )
            return
        members = [token for token, var in self._ens_member_vars.items()
                   if var.get()]
        if not members:
            messagebox.showwarning(
                "No members", "Tick at least one base model for the ensemble."
            )
            return
        kind = self._ens_kind_combo.get()
        voting_rule = self._ens_voting_combo.get()
        meta_token = self._ens_meta_combo.get()
        task = self._task_combo.get()
        if task == "auto":
            task = self._infer_task(target)
        if kind == "voting" and task == "regression":
            voting_rule = "soft"  # ignored by VotingRegressor; keep valid

        def work() -> Any:
            from pharmaml.ml.evaluation import evaluate_model
            from pharmaml.ml.models import build_stacking, build_voting
            from pharmaml.ml.preprocessing import make_train_test_split
            from pharmaml.ml.training import TrainingResult

            frame = self._ml_frame
            y = frame[target]
            x_train, x_test, y_train, y_test = make_train_test_split(
                frame[numeric], y, random_state=42
            )
            if kind == "voting":
                estimator = build_voting(task, members, voting=voting_rule)
                name = f"voting ({'+'.join(members)})"
            else:
                meta = meta_token if meta_token != "auto" else None
                estimator = build_stacking(task, members, meta_model=meta)
                name = (f"stacking ({'+'.join(members)}"
                        f" → {meta_token})")
            import time as _time

            started = _time.perf_counter()
            estimator.fit(x_train, y_train)
            fit_seconds = _time.perf_counter() - started
            trained = TrainingResult(
                model=estimator,
                name=name,
                task=task,
                best_params={},
                cv_scores=None,
                cv_mean=None,
                cv_std=None,
                fit_seconds=fit_seconds,
            )
            report = evaluate_model(
                estimator, x_test, y_test, task,
                keep_predictions=(task == "regression"),
            )
            return trained, report, task, dropped, y_test, x_test

        def render(payload: Any) -> None:
            trained, report, task, dropped, y_test, x_test = payload
            self._ml_last_run = (trained, report, task, x_test, y_test, target)
            self._fill_ml_metrics(trained, report, task, dropped)
            members_named = "+".join(members)
            self._ens_summary.configure(
                text=(f"{kind} ({members_named}) trained — metrics in the "
                      "Results card; Save report… / Save model bundle… "
                      "carry it forward"),
                style="Field.TLabel",
            )
            self._ml_save_info.configure(
                text=(f"last run: {trained.name} — ready to save as a "
                      "bundle"))
            if not self.deps.get("matplotlib"):
                return
            try:
                from pharmaml.viz import (
                    plot_confusion_matrix,
                    plot_predicted_vs_actual,
                )

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

        self._start_job(f"training {kind} ensemble ({len(members)} members)",
                        work, render)

    # -- model persistence card ----------------------------------------- #
    def _ml_save_model_bundle(self) -> None:
        """Save the last trained model (or ensemble) as a provenance bundle."""
        if not self._require_pharmaml():
            return
        if self._ml_last_run is None:
            messagebox.showwarning(
                "No run yet",
                "Train a model (or an ensemble) first -- the bundle saves "
                "the last run's fitted estimator and its metrics.",
            )
            return
        trained, report, task, _x_test, _y_test, target = self._ml_last_run
        path = filedialog.asksaveasfilename(
            title="Save the model bundle",
            defaultextension=".joblib",
            initialfile=f"{trained.name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib",
            filetypes=[("Model bundle", "*.joblib *.pkl"), ("All files", "*.*")],
        )
        if not path:
            return
        frame = self._ml_frame
        numeric, _dropped = numeric_feature_columns(frame, target) if (
            frame is not None) else ([], [])
        metrics: dict[str, Any] = {}
        for key in ("r2", "rmse", "mae", "accuracy", "balanced_accuracy",
                    "precision_macro", "recall_macro", "mcc", "roc_auc"):
            value = getattr(report, key, None)
            if isinstance(value, (int, float)):
                metrics[key] = float(value)

        def work() -> Any:
            from pharmaml.ml.persistence import save_model, write_metadata_sidecar

            written = save_model(
                trained.model,
                path,
                name=trained.name,
                task=task,
                features=list(numeric) if numeric else None,
                best_params=dict(getattr(trained, "best_params", {}) or {}),
                metrics=metrics,
                target=target,
            )
            sidecar = write_metadata_sidecar(written)
            return written, sidecar

        def render(payload: Any) -> None:
            written, sidecar = payload
            self._ml_save_info.configure(
                text=(f"bundle written: {written.name} (+ {sidecar.name}) — "
                      "load it back below or with the CLI"),
                style="Field.TLabel",
            )
            messagebox.showinfo(
                "Saved",
                f"Model bundle written to\n{written}\n\n"
                f"Provenance sidecar: {sidecar}",
            )

        self._start_job("saving model bundle", work, render)

    def _ml_load_model_bundle(self) -> None:
        """Load a saved bundle and show its provenance summary."""
        if not self._require_pharmaml():
            return
        path = filedialog.askopenfilename(
            title="Open a model bundle",
            filetypes=[("Model bundle", "*.joblib *.pkl"), ("All files", "*.*")],
        )
        if not path:
            return
        bundle_path = Path(path)

        def work() -> Any:
            from pharmaml.ml.persistence import load_model

            model, metadata = load_model(bundle_path)
            return model, metadata

        def render(payload: Any) -> None:
            model, metadata = payload
            self._ml_bundle = (model, metadata, bundle_path)
            features = metadata.get("features")
            feature_text = (f"{len(features)} features"
                            if features else "feature count contract")
            self._ml_bundle_info.configure(
                text=(f"{metadata.get('name', '?')} · {metadata.get('task', '?')}"
                      f" · trained {str(metadata.get('trained_at', '?'))[:19]}"
                      f" · {feature_text} · pharmaml "
                      f"{metadata.get('pharmaml_version', '?')} · sklearn "
                      f"{metadata.get('sklearn_version', '?')}"),
                style="Field.TLabel",
            )
            self._ml_predictions = None
            self._predict_tree.delete(*self._predict_tree.get_children())

        self._start_job(f"loading {bundle_path.name}", work, render)

    def _ml_predict_with_bundle(self) -> None:
        """Audit a CSV against the loaded bundle's contract and predict."""
        if not self._require_pharmaml() or not self._require(
                "pandas", "bundle prediction"):
            return
        if self._ml_bundle is None:
            messagebox.showwarning(
                "No bundle", "Open a model bundle first (the button above).")
            return
        path = filedialog.askopenfilename(
            title="Open the data to predict on",
            filetypes=[("Data files", "*.csv *.tsv *.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        model, metadata, bundle_path = self._ml_bundle
        task = str(metadata.get("task", "regression"))

        def work() -> Any:
            import pandas as pd

            from pharmaml.ml.persistence import validate_bundle_input

            frame = pd.read_csv(path)
            # the training target rides along when the scored CSV is the
            # training table itself -- the bundle records its name, and
            # the audit drops it automatically (the ml check convention)
            target_name = metadata.get("target")
            if target_name and target_name in frame.columns:
                frame = frame.drop(columns=[target_name])
            audit = validate_bundle_input(bundle_path, frame)
            if not audit.ok:
                raise ValueError(
                    "the CSV does not satisfy the bundle's feature "
                    f"contract:\n{audit}"
                )
            features = metadata.get("features")
            if features:
                x = frame[list(features)]
            else:
                x = frame.select_dtypes("number")
            prediction = model.predict(x)
            result = frame.copy()
            result["prediction"] = prediction
            if task == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(x)
                classes = list(getattr(model, "classes_", []))
                for index, cls in enumerate(classes):
                    result[f"p_{cls}"] = proba[:, index]
            return result, str(audit)

        def render(payload: Any) -> None:
            result, _audit_text = payload
            self._ml_predictions = result
            tree = self._predict_tree
            tree.delete(*tree.get_children())
            columns = [str(column) for column in result.columns]
            tree["columns"] = columns
            for column in columns:
                width = 190 if column.lower() in ("smiles", "prediction") else 110
                tree.heading(column, text=column)
                tree.column(column, width=width, minwidth=56,
                            stretch=(column.lower() in ("smiles", "prediction")))
            for row in result.head(30).to_dict("records"):
                tree.insert("", "end", values=[
                    _cell_text(row[column], limit=40) for column in columns
                ])
            self._ml_bundle_info.configure(
                text=(f"predicted {len(result)} row(s) — audit OK "
                      f"({bundle_path.name}); Save predictions CSV… writes "
                      "the full table"),
                style="Field.TLabel",
            )

        self._start_job(f"predicting with {bundle_path.name}", work, render)

    def _ml_save_predictions(self) -> None:
        frame = getattr(self, "_ml_predictions", None)
        if frame is None:
            messagebox.showinfo(
                "Nothing to save", "Predict with a loaded bundle first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save the predictions", defaultextension=".csv",
            filetypes=[("CSV table", "*.csv")],
        )
        if not path:
            return
        try:
            frame.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Predictions written to\n{path}")
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not save", exc)

    def _ml_gp_sigma(self, trained: Any, x_test: Any) -> tuple[Any, Any] | None:
        """Predictive ``(mean, sigma)`` when the run's model is a GP.

        ``predict_with_uncertainty`` raises its typed ModelError for
        every non-GP family -- that error IS the probe, caught locally
        so ordinary runs never pay anything for the check.  Returns
        ``None`` when pharmaml (or the GP itself) is unavailable.
        """
        try:
            from pharmaml.exceptions import ModelError
            from pharmaml.ml.evaluation import predict_with_uncertainty
        except Exception:  # noqa: BLE001 - pharmaml missing or older
            return None
        try:
            return predict_with_uncertainty(trained.model, x_test)
        except ModelError:
            return None

    def _ml_save_report(self) -> None:
        """Write the last training run as the one-click visual bundle.

        The same ``report.html`` + figures + tables bundle the CLI's
        ``ml train --plots`` writes (plus the multi-page ``report.pdf``
        when the checkbox is ticked -- the ``--pdf-report`` twin),
        rebuilt from the stored run and saved into a folder the user
        picks.  Figures are created under the package's forced-Agg
        backend inside the worker thread; they never touch Tk.
        """
        if not self._require_pharmaml() or not self._require(
                "matplotlib", "the report figures"):
            return
        if self._ml_last_run is None:
            messagebox.showwarning(
                "No run yet",
                "Train a model first -- the report needs the held-out "
                "metrics and figures of the last run.",
            )
            return
        want_pdf = bool(self._ml_pdf_var.get())
        if want_pdf and not fpdf_available():
            messagebox.showwarning(
                "PDF needs fpdf",
                "The multi-page PDF document needs fpdf -- install it "
                "with:  pip install 'pharmaml[reports]'\n"
                "(the HTML report works without it -- untick the box to "
                "continue).",
            )
            return
        if filedialog is None:  # pragma: no cover - headless fallback
            return
        target = filedialog.askdirectory(title="Choose the report folder")
        if not target:
            return
        trained, report, task, x_test, y_test, _train_target = self._ml_last_run
        model_name = trained.name

        def work() -> Any:
            import pandas as pd

            from pharmaml.exceptions import DataValidationError
            from pharmaml.viz import (
                metrics_to_dataframe,
                ml_plots,
                save_visual_bundle,
            )

            figures: dict[str, Any] = {}
            tables: dict[str, Any] = {}
            if task == "classification":
                figures["confusion_matrix"] = ml_plots.plot_confusion_matrix(
                    report
                )
                if hasattr(trained.model, "predict_proba"):
                    figures["roc_curves"] = ml_plots.plot_roc_curves(
                        trained.model, x_test, y_test
                    )
                counts = pd.Series(y_test).value_counts(sort=False).to_dict()
                figures["class_distribution"] = ml_plots.plot_class_distribution(
                    counts
                )
                tables["class_counts"] = pd.DataFrame(
                    {"class": list(counts), "count": list(counts.values())}
                )
            else:
                y_pred = trained.model.predict(x_test)
                figures["predicted_vs_actual"] = ml_plots.plot_predicted_vs_actual(
                    y_test, y_pred, metrics=report.as_dict()
                )
                figures["residuals"] = ml_plots.plot_residuals(y_test, y_pred)
                gp = self._ml_gp_sigma(trained, x_test)
                if gp is not None:
                    # the GP twin: parity with ±1.96·σ error bars plus
                    # the coverage annotation (the figure the GUI shows).
                    figures["gp_uncertainty"] = ml_plots.plot_gp_uncertainty(
                        y_test, gp[0], gp[1], metrics=report.as_dict()
                    )
            if trained.cv_scores is not None:
                figures["cv_scores"] = ml_plots.plot_cv_scores(trained)
            try:
                figures["feature_importance"] = ml_plots.plot_feature_importance(
                    trained.model
                )
            except DataValidationError:  # kernel SVM / KNN: no importances
                pass
            tables["held_out_metrics"] = metrics_to_dataframe(report)
            return save_visual_bundle(
                figures,
                tables,
                metrics=report,
                out_dir=target,
                title=f"playground ml train — {model_name} ({task})",
                subtitle=self._ml_path.name if self._ml_path else "playground run",
                pdf_report=want_pdf,
            )

        def render(written: Any) -> None:
            names = [path.name for path in written]
            extra = " + report.pdf" if want_pdf else ""
            self._job_label.configure(
                text=f"report saved · {target} ({len(written)} files{extra})"
            )
            messagebox.showinfo(
                "Report saved",
                f"{len(written)} files written to {target}:\n"
                + "\n".join(names[:8])
                + ("\n…" if len(names) > 8 else ""),
            )

        self._start_job("writing the report bundle", work, render)

    def _ml_check_csv(self) -> None:
        """Audit any CSV against the last trained model (``ml check``, GUI).

        The production failure mode this closes, mouse edition: a model
        trained months ago, a regenerated CSV, and a crash deep inside
        sklearn.  The handler reads the chosen file, drops the training
        target column when the CSV still carries it (the GUI's automatic
        ``--target`` -- it knows the name, the CLI asks), writes a
        throwaway bundle from the stored run inside a temp directory and
        renders the verdict of :func:`validate_bundle_input`: missing
        features, extra columns, reordering and version drift, every
        difference named.  The temp bundle is deleted with its directory;
        nothing touches the user's tree.
        """
        if not self._require_pharmaml() or not self._require(
                "pandas", "CSV auditing"):
            return
        if self._ml_last_run is None:
            messagebox.showwarning(
                "No run yet",
                "Train a model first -- the audit checks a new dataset "
                "against the model bundle of the last run.",
            )
            return
        if filedialog is None:  # pragma: no cover - headless fallback
            return
        chosen = filedialog.askopenfilename(
            title="Choose the dataset to audit",
            filetypes=[("Data files", "*.csv *.tsv *.txt"), ("All files", "*.*")],
        )
        if not chosen:
            return
        path = Path(chosen)
        trained, _report, task, x_test, _y_test, train_target = self._ml_last_run
        model_name = trained.name

        def work() -> Any:
            import tempfile

            from pharmaml.ml import save_model, validate_bundle_input
            from pharmaml.utils.io import read_table

            frame = read_table(path)
            if train_target and train_target in frame.columns:
                # the automatic --target: re-auditing the very CSV the
                # model was trained on works out of the box.
                frame = frame.drop(columns=[train_target])
            with tempfile.TemporaryDirectory() as tmp:
                bundle = Path(tmp) / "playground_model.joblib"
                save_model(
                    trained.model, bundle,
                    name=trained.name, task=task,
                    features=list(x_test.columns),
                )
                return validate_bundle_input(bundle, frame)

        def render(audit: Any) -> None:
            verdict = "OK" if audit.ok else "FAIL"
            self._job_label.configure(text=f"audit: {verdict} · {path.name}")
            message = (
                f"{path.name} vs the {model_name} bundle:\n\n{audit}\n\n"
                "The CLI twin:\n"
                "  pharmaml ml check --model <bundle> --data "
                f"{path.name}"
            )
            if train_target:
                message += f" --target {train_target}"
            messagebox.showinfo(f"Dataset audit: {verdict}", message)

        self._start_job(f"auditing {path.name}", work, render)

    def _fill_ml_metrics(self, trained: Any, report: Any, task: str,
                         dropped: list[str],
                         gp: tuple[Any, Any] | None = None) -> None:
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
            if gp is not None:
                # the GP's honest uncertainty line: the median predictive
                # sigma across the held-out set (the error-bar scale the
                # figure draws at ±1.96σ).
                import numpy as _np

                sigma = _np.asarray(gp[1], dtype=float)
                pairs.append(
                    ("predictive σ (median)", format_number(float(_np.median(sigma))))
                )
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

    def _ml_tune(self) -> None:
        """Run a hyperparameter search over the chosen family.

        The ``ml tune`` story, mouse edition: RandomSearch samples
        full-budget trials; Hyperband (``--searcher hyperband``)
        successive-halves the family's resource slot -- ann's
        ``max_iter``, dnn's ``epochs`` -- with the rung ladder sized
        from the family's own declared grid via the public
        ``resource_bounds`` (so every rung stays inside the declared
        space and the final rung is exactly the full budget).  Both
        searchers score every trial AND the hypermodel's defaults
        under the same public ``make_cv_splitter`` folds (the
        apples-to-apples baseline the search has to beat), and the
        winner is refit on the train split and scored on the held-out
        20% -- all inside the worker thread, deterministically
        (seed 42, the tab's own convention).
        """
        if not self._require_pharmaml():
            return
        if not self._require("pandas", "the search") or not self._require(
                "sklearn", "the search"
        ):
            return
        if self._ml_frame is None:
            messagebox.showwarning("No dataset", "Open a CSV or load an example first.")
            return
        target = self._target_combo.get()
        if not target:
            messagebox.showwarning("No target", "Pick the column to predict.")
            return
        # pre-flight (100x-62): the search shares the tab's rule -- a
        # zero-numeric-feature dataset fails before any job starts.
        numeric, dropped = numeric_feature_columns(self._ml_frame, target)
        if not numeric:
            messagebox.showwarning(
                "No numeric features",
                "Every non-target column is non-numeric"
                + (f" ({', '.join(dropped)})" if dropped else "")
                + ".  The hypermodels need numeric feature columns -- compute "
                "descriptors first (the Chemistry tab's feature factory).",
            )
            return
        task = self._task_combo.get()
        if task == "auto":
            task = self._infer_task(target)
        estimator = self._tune_estimator_combo.get()
        hypermodel_name = _TUNE_HYPERMODELS.get((estimator, task))
        if hypermodel_name is None:
            messagebox.showwarning(
                "hsvr is regression-only",
                "The HierarchicalSVR hypermodel has no classification "
                "variant -- pick ann or dnn for a classification target.",
            )
            return
        if estimator == "dnn" and not self._require("torch", "the dnn search"):
            return
        searcher = self._tune_searcher_combo.get()
        resource = _TUNE_RESOURCES.get(estimator)
        if searcher == "hyperband" and resource is None:
            messagebox.showwarning(
                "hsvr has no resource slot",
                "Hyperband halves a resource budget rung by rung -- ann's "
                "max_iter, dnn's epochs. The HierarchicalSVR hypermodel "
                "declares no resource slot, so there is nothing to halve. "
                "Pick ann or dnn, or switch the searcher back to random.",
            )
            return
        try:
            max_trials = parse_int(self._tune_trials.get(), "trials",
                                   minimum=4, maximum=40)
            cv = parse_int(self._tune_cv.get(), "cv folds",
                           minimum=2, maximum=5)
            eta = parse_int(self._tune_eta.get(), "eta",
                            minimum=2, maximum=4)
        except ValueError as exc:
            messagebox.showwarning("Check the search settings", str(exc))
            return

        def work(report: Any) -> Any:
            from sklearn.model_selection import cross_val_score

            import pharmaml.ml.tuner as tuner_module
            from pharmaml.ml import evaluate_model
            from pharmaml.ml.preprocessing import make_train_test_split
            from pharmaml.ml.tuner import (
                Hyperband,
                HyperParameters,
                RandomSearch,
                make_cv_splitter,
                resource_bounds,
            )

            def on_trial(info: dict[str, Any]) -> None:
                # the status bar's heartbeat (100x-106): trial/total
                # drive the determinate bar + ETA, best-so-far rides
                # along as the note.
                note = ""
                if info.get("best_score") is not None:
                    note = f"best {info['best_score']:.3f}"
                if info.get("error") is not None:
                    note = (note + " · " if note else "") + "trial failed"
                report(info["trial"], info.get("total"), note)

            hypermodel = getattr(tuner_module, hypermodel_name)
            objective = "r2" if task == "regression" else "accuracy"
            frame = self._ml_frame
            y = frame[target]
            numeric, _dropped = numeric_feature_columns(frame, target)
            x_train, x_test, y_train, y_test = make_train_test_split(
                frame[numeric], y, random_state=42
            )
            ladder: tuple[int, int] | None = None
            if searcher == "hyperband":
                # the ladder IS the family's declared resource grid --
                # r_min its smallest value, R its largest (the full
                # budget a plain Train runs at).
                r_min, r_max = resource_bounds(hypermodel, resource or "epochs")
                ladder = (r_min, r_max)
                tuner: Any = Hyperband(
                    hypermodel, objective=objective, resource=resource or "epochs",
                    max_resource=r_max, min_resource=r_min, eta=eta,
                    cv=cv, task=task, seed=42, on_trial=on_trial,
                )
            else:
                tuner = RandomSearch(
                    hypermodel, objective=objective, max_trials=max_trials,
                    cv=cv, task=task, seed=42, on_trial=on_trial,
                )
            results = tuner.search(x_train, y_train)
            # the baseline: the hypermodel's defaults under the tuner's
            # own fold protocol (the public splitter -- same folds, only
            # the configuration differs).
            defaults = hypermodel(HyperParameters())
            base = cross_val_score(
                defaults, x_train, y_train, scoring=objective,
                cv=make_cv_splitter(task, y_train, cv),
            )
            # the held-out verdict: the winner was refit on the train
            # split inside search(); score it like any trained model.
            report = evaluate_model(
                results.best_estimator, x_test, y_test, task,
                keep_predictions=False,
            )
            return results, float(base.mean()), float(base.std()), report, objective, ladder

        def render(payload: Any) -> None:
            results, base_mean, base_std, report, objective, ladder = payload
            self._tune_last = (results, estimator, task, target)
            held_out = report.r2 if task == "regression" else report.accuracy
            held_label = "held-out R²" if task == "regression" else "held-out accuracy"
            gain = results.best_score - base_mean
            self._tune_summary.configure(
                text=f"best {results.best_score:.4f} ± "
                     f"{results.best_trial.score_std:.4f} CV {objective} · "
                     f"defaults {base_mean:.4f} ± {base_std:.4f} → gain "
                     f"{gain:+.4f} · {held_label} {held_out:.4f} · winner: "
                     + tune_config_text(results.best_params)
            )
            ranked = sorted(
                (t for t in results.trials if not t.failed),
                key=lambda t: t.score, reverse=True,
            )
            failed = [t for t in results.trials if t.failed]
            self._board_tree.delete(*self._board_tree.get_children())
            if ladder is None:
                columns = ["rank", "trial", "configuration", "score", "± std"]
            else:
                columns = ["rank", "trial", "configuration", "resource", "score", "± std"]
            self._board_tree["columns"] = columns
            for column in columns:
                self._board_tree.heading(column, text=column)
                self._board_tree.column(column, width=140, minwidth=72, stretch=True)

            def row_values(rank: int | str, trial: Any) -> list[str]:
                cells = [str(rank), str(trial.trial_id),
                         _cell_text(tune_config_text(trial.hyperparameters), limit=52)]
                if ladder is not None:
                    cells.append("" if trial.resource is None
                                 else f"{trial.resource:g}")
                cells.append(f"{trial.score:.4f}")
                cells.append(f"{trial.score_std:.4f}")
                return cells

            for rank, trial in enumerate(ranked, start=1):
                self._board_tree.insert(
                    "", "end", values=row_values(rank, trial))
            for trial in failed:
                self._board_tree.insert(
                    "", "end", values=row_values("—", trial))
            if ladder is None:
                self._board_status.configure(
                    text=f"RandomSearch over {estimator} ({task}) — "
                         f"{len(results.trials)} trial(s), objective {objective}; "
                         f"ranked by mean CV score, failed trials last"
                )
            else:
                r_min, r_max = ladder
                spent = sum(t.resource or 0.0 for t in results.trials)
                naive = len(results.trials) * r_max
                saved = 100.0 * (1.0 - spent / naive) if naive > 0 else 0.0
                self._board_status.configure(
                    text=f"Hyperband over {estimator} ({task}) — "
                         f"{len(results.trials)} trial(s), objective {objective} · "
                         f"η {eta}, resource {resource} {r_min}–{r_max} · "
                         f"budget {spent:g} of a naive {naive:g} {resource} "
                         f"({saved:.0f}% saved) · ranked by mean CV score, "
                         "failed trials last (promotions keep the top ⌈n/η⌉)"
                )

        self._start_job(f"tuning {estimator} ({task})", work, render)

    def _ml_tune_apply(self) -> None:
        """Carry the winning configuration forward (Apply best).

        ``ann`` / ``hsvr``: point the model combobox at the family so
        Train & evaluate picks it up (the registry's own defaults --
        the winning hyperparameters live in the leaderboard and the
        trials CSV).  ``dnn``: prefill the whole Deep Learning tab from
        the winning parameters (layers, dropout, learning rate,
        activation, batchnorm, epochs -- the hypermodel's own knobs) and
        switch to it, dataset included, so the loss curves and the
        report pipeline run on the exact winner.
        """
        if self._tune_last is None:
            messagebox.showwarning(
                "No search yet",
                "Run a hyperparameter search first -- Apply best carries "
                "the winning configuration of the last search forward.",
            )
            return
        results, estimator, task, target = self._tune_last
        params = results.best_params
        if estimator == "dnn":
            if not self._require("torch", "the Deep Learning tab"):
                return
            if self._ml_path is not None:
                self._dl_load_path(self._ml_path)
            if target in list(self._dl_target_combo["values"]):
                self._dl_target_combo.set(target)
            self._dl_task_combo.set(task)
            units = int(params.get("units", 64))
            n_layers = int(params.get("n_layers", 2))
            self._dl_hidden_entry.delete(0, "end")
            self._dl_hidden_entry.insert(
                0, ", ".join(str(units) for _ in range(n_layers)))
            self._dl_dropout.set(f"{float(params.get('dropout', 0.1)):g}")
            self._dl_lr_entry.delete(0, "end")
            self._dl_lr_entry.insert(0, f"{float(params.get('learning_rate', 1e-3)):g}")
            activation = str(params.get("activation", "relu"))
            if activation in _DL_ACTIVATIONS:
                self._dl_activation_combo.set(activation)
            self._dl_batchnorm_var.set(bool(params.get("batchnorm", False)))
            self._dl_epochs.set(str(int(params.get("epochs", 80))))
            self._dl_patience.set("10")  # the hypermodel's own fixed value
            self._tune_summary.configure(
                text="applied to the Deep Learning tab — hit Train & evaluate "
                     "there for the loss curves and the report bundle")
            with contextlib.suppress(tk.TclError):
                self._book.select(self._dl_page)
            return
        self._task_combo.set(task)
        self._refresh_model_choices(task)
        self._model_combo.set(estimator)
        self._ml_refresh_hyperparam_panel()
        # carry the winning configuration into the manual card too
        # (100x-106): every winner key that is a registry knob of the
        # family gets pinned, so the very next Train runs the winner
        # instead of the defaults.
        entries = getattr(self, "_ml_param_entries", {})
        filled = 0
        for key, entry in entries.items():
            if key in params:
                entry.delete(0, "end")
                entry.insert(0, str(params[key]))
                filled += 1
        self._ml_sync_params_var()
        self._ml_params_status.configure(
            text=f"{len(entries)} knob(s) · {filled} pinned from the winning "
                 "configuration · hit Train & evaluate to refit at these values"
        )
        self._tune_summary.configure(
            text=f"applied — model set to {estimator}; hit Train & evaluate "
                 "for the full report pipeline (the winning config: "
                 + tune_config_text(params)
                 + "; the manual hyperparameters card carries it)"
        )

    def _ml_tune_save_trials(self) -> None:
        """Write the last search's trials table (the CLI ``--out`` story)."""
        if not self._require_pharmaml() or not self._require(
                "pandas", "the trials table"):
            return
        if self._tune_last is None:
            messagebox.showwarning(
                "No search yet",
                "Run a hyperparameter search first -- the trials table is "
                "the leaderboard's full record (score, std, every "
                "hyperparameter, failed flag).",
            )
            return
        if filedialog is None:  # pragma: no cover - headless fallback
            return
        path = filedialog.asksaveasfilename(
            title="Save the trials table",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        results, _estimator, _task, _target = self._tune_last
        try:
            from pharmaml.utils.io import write_table

            written = write_table(results.trials_frame(), Path(path))
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not write the trials table", exc)
            return
        self._job_label.configure(
            text=f"trials saved · {Path(written).name} "
                 f"({len(results.trials)} rows)"
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
        if not target:
            messagebox.showwarning("No target", "Pick the column to predict.")
            return
        # pre-flight (100x-62): the compare flow shares the rule -- a
        # zero-numeric-feature dataset fails before any CV job starts.
        numeric, dropped = numeric_feature_columns(self._ml_frame, target)
        if not numeric:
            messagebox.showwarning(
                "No numeric features",
                "Every non-target column is non-numeric"
                + (f" ({', '.join(dropped)})" if dropped else "")
                + ".  The models need numeric feature columns -- compute "
                "descriptors first (the Chemistry tab's feature factory).",
            )
            return
        task = self._task_combo.get()
        if task == "auto":
            task = self._infer_task(target)

        def work(report: Any) -> Any:
            from pharmaml.ml import compare_models

            frame = self._ml_frame
            y = frame[target]
            numeric, _dropped = numeric_feature_columns(frame, target)

            def on_model(info: dict[str, Any]) -> None:
                # one heartbeat per leaderboard row (100x-106): the
                # model name is the note, index/total the fraction.
                report(info["index"], info["total"], info["model"])

            return compare_models(frame[numeric], y, task, cv=3, n_jobs=1,
                                  on_model=on_model)

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
            self._board_status.configure(
                text="3-fold CV over the available registry — the score column "
                     "is the task's objective (higher is better)"
            )

        self._start_job(f"comparing models ({task})", work, render)

    # ------------------------------------------------------------------ #
    # Deep Learning tab (100x-61) -- the pharmaml.ml.neural estimators
    # ------------------------------------------------------------------ #
    def _build_dl_tab(self) -> None:
        page = self._new_page("Deep Learning")
        self._dl_page = page
        self._dl_frame = None
        self._dl_path = None
        self._dl_history: list[dict[str, Any]] = []
        self._dl_last_run: tuple[Any, str, str, Any, str] | None = None

        data = make_card(page.grid, "Dataset", span=2, row=0, column=0,
                         subtitle="Any CSV the toolkit's read_table can sniff, or a "
                                  "bundled example. Features = the numeric columns "
                                  "except the target (same rule as the Data & ML tab).")
        open_btn = self._action_button(data, "Open CSV…", self._dl_open_csv)
        open_btn.grid(row=0, column=0, sticky="w")
        example_files = [
            name for name in (
                "qsar_regression.csv", "qsar_classification.csv", "qsar_multiclass.csv",
            ) if (EXAMPLES_DIR / name).is_file()
        ]
        self._dl_example_combo = ttk.Combobox(data, state="readonly", width=26,
                                              values=example_files)
        if example_files:
            self._dl_example_combo.set(example_files[0])
        self._dl_example_combo.grid(row=0, column=1, sticky="w", padx=(12, 0))
        load_btn = self._action_button(data, "Load example", self._dl_load_example)
        load_btn.grid(row=0, column=2, sticky="w", padx=(12, 0))
        self._dl_target_combo = ttk.Combobox(data, state="readonly", width=26)
        form_row(data, "target column", self._dl_target_combo)
        self._dl_task_combo = ttk.Combobox(data, state="readonly", width=26,
                                           values=("auto", "regression", "classification"))
        self._dl_task_combo.set("auto")
        form_row(data, "task", self._dl_task_combo)
        self._dl_info = ttk.Label(data, text="no dataset loaded", style="CardMuted.TLabel")
        self._dl_info.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))

        preview = make_card(page.grid, "Preview (first 12 rows)", span=2, row=1,
                            column=0)
        tree_frame = ttk.Frame(preview, style="Card.TFrame")
        grid_append(preview, tree_frame, sticky="nsew", weight=True)
        self._dl_preview_tree = ttk.Treeview(tree_frame, show="headings", height=8)
        preview_bar_x = ttk.Scrollbar(tree_frame, orient="horizontal",
                                      command=self._dl_preview_tree.xview)
        preview_bar_y = ttk.Scrollbar(tree_frame, orient="vertical",
                                      command=self._dl_preview_tree.yview)
        self._dl_preview_tree.configure(xscrollcommand=preview_bar_x.set,
                                        yscrollcommand=preview_bar_y.set)
        self._dl_preview_tree.pack(side="left", fill="both", expand=True)
        preview_bar_y.pack(side="right", fill="y")
        preview_bar_x.pack(side="bottom", fill="x")

        net = make_card(page.grid, "Network & training", span=2, row=2, column=0,
                        subtitle="The pharmaml.ml.neural estimators: Adam with weight "
                                 "decay, per-feature standardisation and early stopping "
                                 "on a held-out validation fraction. Hidden layers are "
                                 "comma-separated widths -- empty = a single linear "
                                 "layer, the regularised baseline. Batch norm (1.2.36) "
                                 "inserts a BatchNorm1d between every hidden linear "
                                 "map and its activation.")
        self._dl_arch_combo = ttk.Combobox(net, state="readonly", width=26,
                                           values=_DL_ARCHITECTURES)
        self._dl_arch_combo.set(_DL_ARCHITECTURES[0])
        form_row(net, "architecture", self._dl_arch_combo)
        self._dl_hidden_entry = ttk.Entry(net, width=16)
        self._dl_hidden_entry.insert(0, "64, 32")
        form_row(net, "hidden layers", self._dl_hidden_entry)
        self._dl_activation_combo = ttk.Combobox(net, state="readonly", width=16,
                                                 values=_DL_ACTIVATIONS)
        self._dl_activation_combo.set("relu")
        form_row(net, "activation", self._dl_activation_combo)
        self._dl_dropout = ttk.Spinbox(net, from_=0.0, to=0.9, increment=0.05, width=10)
        self._dl_dropout.set("0.1")
        form_row(net, "dropout", self._dl_dropout)
        self._dl_epochs = ttk.Spinbox(net, from_=10, to=2000, increment=10, width=10)
        self._dl_epochs.set("150")
        form_row(net, "epochs (max)", self._dl_epochs)
        self._dl_batch = ttk.Spinbox(net, from_=1, to=512, increment=8, width=10)
        self._dl_batch.set("32")
        form_row(net, "batch size", self._dl_batch)
        self._dl_lr_entry = ttk.Entry(net, width=12)
        self._dl_lr_entry.insert(0, "0.001")
        form_row(net, "learning rate", self._dl_lr_entry)
        self._dl_wd_entry = ttk.Entry(net, width=12)
        self._dl_wd_entry.insert(0, "0.0")
        form_row(net, "weight decay", self._dl_wd_entry)
        self._dl_patience = ttk.Spinbox(net, from_=1, to=100, increment=1, width=10)
        self._dl_patience.set("15")
        form_row(net, "patience", self._dl_patience)
        self._dl_valsplit = ttk.Spinbox(net, from_=0.0, to=0.5, increment=0.05, width=10)
        self._dl_valsplit.set("0.15")
        form_row(net, "val split", self._dl_valsplit)
        self._dl_seed_entry = ttk.Entry(net, width=12)
        self._dl_seed_entry.insert(0, "42")
        form_row(net, "random seed", self._dl_seed_entry)
        self._dl_batchnorm_var = tk.BooleanVar(value=False)
        batchnorm_check = ttk.Checkbutton(
            net, text="batch normalisation between the hidden layers",
            variable=self._dl_batchnorm_var)
        Tooltip(batchnorm_check,
                "batchnorm=True (1.2.36): a BatchNorm1d after every hidden "
                "linear map, before its activation -- residual blocks "
                "normalise BEFORE the skip add. Running statistics ride in "
                "the portable weight dict, so predict/pickle work unchanged. "
                "Needs batch size >= 2 (singleton tail batches are skipped "
                "with a debug log).")
        form_row(net, "batch norm", batchnorm_check)
        buttons = ttk.Frame(net, style="Card.TFrame")
        buttons.grid(row=net.grid_size()[1], column=0, columnspan=2, sticky="w",
                     pady=(8, 0))
        train_btn = self._action_button(
            buttons, "Train & evaluate", self._dl_train, primary=True,
            tooltip="80/20 split, the estimator's own standardisation and "
                    "stratified early stopping, then held-out metrics, the "
                    "parity/confusion figure and the loss curves.")
        train_btn.pack(side="left")
        check_btn = self._action_button(
            buttons, "Check CSV…", self._dl_check_csv,
            tooltip="Audit any CSV against the last trained network before "
                    "predicting -- missing features, extra columns, "
                    "reordering and version drift, all named (the ml check "
                    "story, deep edition; the training target is dropped "
                    "automatically when present).")
        check_btn.pack(side="left", padx=(10, 0))

        results = make_card(page.grid, "Results", span=2, row=3, column=0)
        self._dl_metrics = ttk.Frame(results, style="Card.TFrame")
        grid_append(results, self._dl_metrics, sticky="ew")
        ttk.Label(self._dl_metrics, text="train a network to see metrics",
                  style="CardMuted.TLabel").pack(anchor="w")
        self._dl_host = self._figure_host(results, height=330)

        hist = make_card(page.grid, "Training history", span=2, row=4, column=0,
                         subtitle="Per-epoch training/validation loss -- the "
                                  "early-stopping verdict rides on the figure title.")
        self._dl_hist_summary = ttk.Label(
            hist, text="train a network to see the loss curves",
            style="CardMuted.TLabel")
        grid_append(hist, self._dl_hist_summary, sticky="w")
        self._dl_hist_host = self._figure_host(hist, height=300)
        hist_buttons = ttk.Frame(hist, style="Card.TFrame")
        grid_append(hist, hist_buttons, sticky="w", pady=(6, 0))
        save_btn = self._action_button(
            hist_buttons, "Save history as CSV…", self._dl_save_history,
            tooltip="The exact per-epoch records the estimator kept -- "
                    "epoch, train_loss, val_loss.")
        save_btn.pack(side="left")

    def _dl_open_csv(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas", "CSV loading"):
            return
        path = filedialog.askopenfilename(
            title="Open a dataset",
            filetypes=[("Data files", "*.csv *.tsv *.txt"), ("All files", "*.*")],
        )
        if path:
            self._dl_load_path(Path(path))

    def _dl_load_example(self) -> None:
        if not self._require_pharmaml() or not self._require("pandas", "CSV loading"):
            return
        name = self._dl_example_combo.get()
        path = EXAMPLES_DIR / name
        if not path.is_file():
            messagebox.showwarning("Missing example", f"{name} was not found.")
            return
        self._dl_load_path(path)

    def _dl_load_path(self, path: Path) -> None:
        try:
            from pharmaml.utils.io import read_table

            frame = read_table(path)
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not read the dataset", exc)
            return
        if frame.empty:
            messagebox.showwarning("Empty dataset", f"{path.name} has no rows.")
            return
        self._dl_frame = frame
        self._dl_path = path
        columns = [str(column) for column in frame.columns]
        self._dl_preview_tree.delete(*self._dl_preview_tree.get_children())
        self._dl_preview_tree["columns"] = columns
        for column in columns:
            self._dl_preview_tree.heading(column, text=column)
            self._dl_preview_tree.column(column, width=96, minwidth=64, stretch=True)
        for _, row in frame.head(12).iterrows():
            self._dl_preview_tree.insert(
                "", "end", values=[_cell_text(value) for value in row.tolist()]
            )
        self._dl_target_combo["values"] = columns
        self._dl_target_combo.set(default_target_column(frame) or columns[-1])
        self._dl_info.configure(
            text=f"{path.name} — {len(frame):,} rows × {len(columns)} columns"
        )

    def _dl_train(self) -> None:
        if not self._require_pharmaml():
            return
        for module in ("pandas", "sklearn", "torch"):
            if not self._require(module, "network training"):
                return
        if self._dl_frame is None:
            messagebox.showwarning("No dataset", "Open a CSV or load an example first.")
            return
        target = self._dl_target_combo.get()
        if not target:
            messagebox.showwarning("No target", "Pick the column to predict.")
            return
        # pre-flight (100x-62): a dataset with zero numeric feature
        # columns used to surface as a mid-job DataValidationError --
        # catch it BEFORE the job starts, naming the dropped columns and
        # the remedy (the feature factory computes descriptors).
        numeric, dropped = numeric_feature_columns(self._dl_frame, target)
        if not numeric:
            messagebox.showwarning(
                "No numeric features",
                "Every non-target column is non-numeric"
                + (f" ({', '.join(dropped)})" if dropped else "")
                + ".  The estimators need numeric feature columns -- compute "
                "descriptors first (the Chemistry tab's feature factory).",
            )
            return
        try:
            hidden = parse_hidden_layers(self._dl_hidden_entry.get())
            learning_rate = parse_float(self._dl_lr_entry.get(), "learning rate",
                                        minimum=1e-8)
            weight_decay = parse_float(self._dl_wd_entry.get(), "weight decay",
                                       minimum=0.0)
            dropout = parse_float(self._dl_dropout.get(), "dropout",
                                  minimum=0.0, maximum=0.9)
            val_split = parse_float(self._dl_valsplit.get(), "val split",
                                    minimum=0.0, maximum=0.5)
            epochs = parse_int(self._dl_epochs.get(), "epochs", minimum=1)
            batch_size = parse_int(self._dl_batch.get(), "batch size", minimum=1)
            patience = parse_int(self._dl_patience.get(), "patience", minimum=1)
            seed = parse_int(self._dl_seed_entry.get(), "random seed", minimum=0)
        except ValueError as exc:
            messagebox.showwarning("Check the network settings", str(exc))
            return
        batchnorm = bool(self._dl_batchnorm_var.get())
        if batchnorm and batch_size < 2:
            # the estimator's own guard, surfaced before the job starts
            # (the GUI's pre-flight convention): BatchNorm1d cannot run on
            # single-row batches.
            messagebox.showwarning(
                "Batch norm needs batches",
                "Batch normalisation needs a batch size of at least 2 -- "
                "BatchNorm1d cannot estimate statistics from single rows. "
                "Raise the batch size or untick batch norm.",
            )
            return
        task = self._dl_task_combo.get()
        if task == "auto":
            task = self._infer_task(target, self._dl_frame)
        architecture = self._dl_arch_combo.get()
        activation = self._dl_activation_combo.get()
        estimator_name = deep_estimator_name(architecture, task)

        def work(notify: Any) -> Any:
            import time as _time

            from pharmaml.ml import evaluate_model, neural
            from pharmaml.ml.preprocessing import make_train_test_split

            def on_epoch(info: dict[str, Any]) -> None:
                # epoch/epochs drive the bar + ETA (100x-106); the
                # validation loss rides along once it exists.
                # (``notify`` -- not ``report`` -- because this work
                # function's own evaluation payload is named report.)
                if info.get("val_loss") is not None:
                    note = f"val_loss {info['val_loss']:.3g}"
                else:
                    note = f"train_loss {info['train_loss']:.3g}"
                notify(info["epoch"], info["epochs"], note)

            frame = self._dl_frame
            y = frame[target]
            numeric, dropped = numeric_feature_columns(frame, target)
            x_train, x_test, y_train, y_test = make_train_test_split(
                frame[numeric], y, random_state=seed
            )
            estimator_cls = getattr(neural, estimator_name)
            network = estimator_cls(
                hidden_layers=hidden, dropout=dropout, activation=activation,
                epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                weight_decay=weight_decay, patience=patience, val_split=val_split,
                batchnorm=batchnorm, random_state=seed,
            )
            started = _time.perf_counter()
            network.fit(x_train, y_train, on_epoch=on_epoch)
            seconds = _time.perf_counter() - started
            report = evaluate_model(
                network, x_test, y_test, task,
                keep_predictions=(task == "regression"),
            )
            return network, report, task, dropped, y_test, seconds, x_test

        def render(payload: Any) -> None:
            network, report, task, dropped, y_test, seconds, x_test = payload
            self._dl_history = list(network.history_)
            # the audit state (100x-81): the fitted network plus the run's
            # feature contract -- set before the matplotlib guard so Check
            # CSV also works when plotting is unavailable.
            self._dl_last_run = (network, estimator_name, task, x_test, target)
            self._fill_dl_metrics(network, report, task, dropped, seconds)
            self._fill_dl_history_summary(network)
            if not self.deps.get("matplotlib"):
                return
            try:
                from pharmaml.viz import (
                    plot_confusion_matrix,
                    plot_predicted_vs_actual,
                    plot_training_history,
                )

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
                self._render_figure(self._dl_host, figure)
                history_figure = plot_training_history(
                    network.history_,
                    title=f"{type(network).__name__} · {target}",
                )
                self._render_figure(self._dl_hist_host, history_figure)
            except Exception as exc:  # noqa: BLE001
                self._show_error("Plot failed", exc)

        self._start_job(f"training {estimator_name} ({task})", work, render)

    def _fill_dl_metrics(self, network: Any, report: Any, task: str,
                         dropped: list[str], seconds: float) -> None:
        """Fill the Results card's metric grid for a finished deep run."""
        for child in self._dl_metrics.winfo_children():
            child.destroy()
        hidden = network.get_params()["hidden_layers"]  # the sklearn contract
        pairs = [
            ("network", f"{type(network).__name__} ({task})"),
            ("hidden layers", " × ".join(str(width) for width in hidden)
             or "linear (none)"),
            ("features", str(network.n_features_in_)),
        ]
        if task == "regression":
            pairs += [
                ("R²", format_number(getattr(report, "r2", float("nan")))),
                ("RMSE", format_number(report.rmse)),
                ("MAE", format_number(report.mae)),
            ]
        else:
            pairs += [
                ("accuracy", format_number(report.accuracy)),
                ("balanced accuracy", format_number(report.balanced_accuracy)),
                ("MCC", format_number(report.mcc)),
            ]
        best_val: float | None = None
        for entry in network.history_:
            value = entry.get("val_loss")
            if value is not None and (best_val is None or value < best_val):
                best_val = float(value)
        pairs += [
            ("epochs run", str(network.n_iter_)),
            ("best val loss", "n/a" if best_val is None
             else format_number(best_val)),
            ("fit time", f"{seconds:.2f} s"),
        ]
        if dropped:
            pairs.append(("dropped non-numeric", ", ".join(dropped)))
        for index, (name, value) in enumerate(pairs):
            ttk.Label(self._dl_metrics, text=name, style="CardMuted.TLabel").grid(
                row=index // 4, column=(index % 4) * 2, sticky="w", padx=(0, 6)
            )
            ttk.Label(self._dl_metrics, text=str(value), style="Field.TLabel",
                      font=self._mono).grid(
                row=index // 4, column=(index % 4) * 2 + 1, sticky="w", padx=(0, 18)
            )

    def _fill_dl_history_summary(self, network: Any) -> None:
        """One muted line above the loss curves: what actually happened."""
        val_losses = [entry.get("val_loss") for entry in network.history_
                      if entry.get("val_loss") is not None]
        if val_losses:
            self._dl_hist_summary.configure(
                text=f"{type(network).__name__} · {network.n_iter_} epochs run · "
                     f"best val loss {format_number(min(val_losses))}"
            )
        else:
            self._dl_hist_summary.configure(
                text=f"{type(network).__name__} · {network.n_iter_} epochs run · "
                     "no validation split (val split = 0)"
            )

    def _dl_save_history(self) -> None:
        if not self._dl_history:
            messagebox.showinfo("Nothing to save", "Train a network first.")
            return
        if not self._require_pharmaml():
            return
        path = filedialog.asksaveasfilename(
            title="Save training history", defaultextension=".csv",
            filetypes=[("CSV table", "*.csv")],
        )
        if not path:
            return
        try:
            import pandas as pd

            from pharmaml.utils.io import write_table

            frame = pd.DataFrame(self._dl_history)
            written = write_table(frame, Path(path))
            messagebox.showinfo("Saved", f"Training history written to\n{written}")
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not save", exc)

    def _dl_check_csv(self) -> None:
        """Audit any CSV against the last trained network (``ml check``).

        The deep twin of ``_ml_check_csv`` (100x-79): the neural
        estimators satisfy the same sklearn contract -- ``n_features_in_``
        and ``feature_names_in_`` recorded at fit, joblib-serialisable --
        so :func:`validate_bundle_input` audits them exactly like the
        registry models.  The handler reads the chosen file, drops the
        training target column when the CSV still carries it (the GUI's
        automatic ``--target``), writes a throwaway bundle from the
        stored run inside a temp directory and renders the verdict:
        missing features, extra columns, reordering and version drift,
        every difference named.  The temp bundle is deleted with its
        directory; nothing touches the user's tree.
        """
        if not self._require_pharmaml() or not self._require(
                "pandas", "CSV auditing"):
            return
        if self._dl_last_run is None:
            messagebox.showwarning(
                "No run yet",
                "Train a network first -- the audit checks a new dataset "
                "against the model bundle of the last run.",
            )
            return
        if filedialog is None:  # pragma: no cover - headless fallback
            return
        chosen = filedialog.askopenfilename(
            title="Choose the dataset to audit",
            filetypes=[("Data files", "*.csv *.tsv *.txt"), ("All files", "*.*")],
        )
        if not chosen:
            return
        path = Path(chosen)
        network, estimator_name, task, x_test, train_target = self._dl_last_run

        def work() -> Any:
            import tempfile

            from pharmaml.ml import save_model, validate_bundle_input
            from pharmaml.utils.io import read_table

            frame = read_table(path)
            if train_target and train_target in frame.columns:
                # the automatic --target: re-auditing the very CSV the
                # network was trained on works out of the box.
                frame = frame.drop(columns=[train_target])
            with tempfile.TemporaryDirectory() as tmp:
                bundle = Path(tmp) / "playground_model.joblib"
                save_model(
                    network, bundle,
                    name=estimator_name, task=task,
                    features=list(x_test.columns),
                )
                return validate_bundle_input(bundle, frame)

        def render(audit: Any) -> None:
            verdict = "OK" if audit.ok else "FAIL"
            self._job_label.configure(text=f"audit: {verdict} · {path.name}")
            message = (
                f"{path.name} vs the {estimator_name} bundle:\n\n{audit}\n\n"
                "The CLI twin:\n"
                "  pharmaml ml check --model <bundle> --data "
                f"{path.name}"
            )
            if train_target:
                message += f" --target {train_target}"
            messagebox.showinfo(f"Dataset audit: {verdict}", message)

        self._start_job(f"auditing {path.name}", work, render)

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
        # the optimiser itself: the GA (structure-space search) or the RL
        # policy-gradient fine-tuning of an on-the-fly SMILES RNN prior
        self._method_combo = ttk.Combobox(
            controls, state="readonly", width=26,
            values=("genetic algorithm", "reinforcement learning (RL)"),
        )
        self._method_combo.set("genetic algorithm")
        self._method_combo.bind("<<ComboboxSelected>>", self._on_method_change)
        form_row(controls, "optimiser", self._method_combo)
        # GA-specific rows live in their own frame so the RL selection can
        # swap the whole block out (grid_remove collapses the row; re-grid
        # restores the remembered position -- the card never reshuffles)
        ga_frame = ttk.Frame(controls, style="Card.TFrame")
        grid_append(controls, ga_frame, sticky="ew")
        ga_frame.columnconfigure(1, weight=1)
        self._pop_size = ttk.Spinbox(ga_frame, from_=8, to=96, increment=4, width=10)
        self._pop_size.set("24")
        form_row(ga_frame, "population", self._pop_size)
        self._generations = ttk.Spinbox(ga_frame, from_=4, to=48, increment=4, width=10)
        self._generations.set("16")
        form_row(ga_frame, "generations", self._generations)
        self._mutation_combo = ttk.Combobox(ga_frame, state="readonly", width=18,
                                            values=("uniform", "scaffold", "topology"))
        self._mutation_combo.set("topology")
        form_row(ga_frame, "mutation", self._mutation_combo)
        self._crossover_combo = ttk.Combobox(ga_frame, state="readonly", width=18,
                                             values=("splice", "scaffold_hop"))
        self._crossover_combo.set("splice")
        form_row(ga_frame, "crossover", self._crossover_combo)
        self._operator_combo = ttk.Combobox(ga_frame, state="readonly", width=18,
                                            values=("uniform", "adaptive"))
        self._operator_combo.set("uniform")
        form_row(ga_frame, "operator selection", self._operator_combo)
        # RL-specific rows: swapped in when the optimiser combo says RL
        rl_frame = ttk.Frame(controls, style="Card.TFrame")
        grid_append(controls, rl_frame, sticky="ew")
        rl_frame.columnconfigure(1, weight=1)
        self._rl_iterations = ttk.Spinbox(rl_frame, from_=2, to=60, increment=2, width=10)
        self._rl_iterations.set("12")
        form_row(rl_frame, "iterations", self._rl_iterations)
        self._rl_batch = ttk.Spinbox(rl_frame, from_=8, to=64, increment=8, width=10)
        self._rl_batch.set("16")
        form_row(rl_frame, "batch size", self._rl_batch)
        self._rl_prior_epochs = ttk.Spinbox(rl_frame, from_=1, to=16, increment=1, width=10)
        self._rl_prior_epochs.set("8")
        Tooltip(self._rl_prior_epochs,
                "Epochs for the on-the-fly SMILES prior -- 8 is the library "
                "default; fewer than ~6 undertrains it and the agent mostly "
                "samples invalid strings (the run reports no valid molecules).")
        form_row(rl_frame, "prior epochs", self._rl_prior_epochs)
        self._rl_anchor = ttk.Entry(rl_frame, width=12)
        self._rl_anchor.insert(0, "0.1")
        form_row(rl_frame, "anchor weight", self._rl_anchor)
        rl_frame.grid_remove()  # the GA rows are the default
        self._rl_frame = rl_frame
        self._ga_frame = ga_frame
        self._ga_seed = ttk.Entry(controls, width=12)
        self._ga_seed.insert(0, "42")
        form_row(controls, "random seed", self._ga_seed)
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

    def _on_method_change(self, _event: Any = None) -> None:
        """Swap the GA / RL parameter blocks (grid_remove collapses the row)."""
        if self._method_combo.get() == "reinforcement learning (RL)":
            self._ga_frame.grid_remove()
            self._rl_frame.grid()
        else:
            self._rl_frame.grid_remove()
            self._ga_frame.grid()

    def _run_evolution(self) -> None:
        method = self._method_combo.get()
        if method == "reinforcement learning (RL)":
            if not self._require_pharmaml() or not self._require(
                "torch", "the RL optimiser"
            ):
                return
        elif not self._require_pharmaml() or not self._require("rdkit", "the GA"):
            return
        try:
            kind = self._objective_combo.get()
            lo = parse_float(self._obj_lo.get(), "window low")
            hi = parse_float(self._obj_hi.get(), "window high")
            if lo > hi:
                raise ValueError("window low exceeds window high")
            seed = parse_int(self._ga_seed.get(), "seed", minimum=0)
            if method == "reinforcement learning (RL)":
                iterations = parse_int(
                    self._rl_iterations.get(), "iterations", minimum=1
                )
                batch_size = parse_int(
                    self._rl_batch.get(), "batch size", minimum=1
                )
                prior_epochs = parse_int(
                    self._rl_prior_epochs.get(), "prior epochs", minimum=1
                )
                anchor_weight = parse_float(
                    self._rl_anchor.get(), "anchor weight"
                )
                if anchor_weight < 0:
                    raise ValueError("anchor weight cannot be negative")
                population = generations = 0
                mutation = crossover = operator_selection = ""
            else:
                iterations = batch_size = prior_epochs = 0
                anchor_weight = 0.0
                population = parse_int(self._pop_size.get(), "population", minimum=4)
                generations = parse_int(self._generations.get(), "generations", minimum=1)
                mutation = self._mutation_combo.get()
                crossover = self._crossover_combo.get()
                operator_selection = self._operator_combo.get()
        except ValueError as exc:
            messagebox.showerror("Input", str(exc))
            return
        lines = parse_smiles_lines(self._corpus_text.get("1.0", "end"))
        seed_mode = bool(self._seed_mode_var.get())
        if method == "reinforcement learning (RL)":
            want_plot = bool(self.deps.get("matplotlib"))
        else:
            want_plot = operator_selection == "adaptive" and self.deps.get("matplotlib")

        def work() -> Any:
            objective = build_objective(kind, lo, hi)
            if method == "reinforcement learning (RL)":
                from pharmaml.generative import DEFAULT_CORPUS, ReinventOptimizer

                # the CLI idiom: seeded mode anchors the on-the-fly prior's
                # corpus with the user's molecules ahead of the bundle
                if not lines:
                    corpus_arg: Any = None
                elif seed_mode:
                    corpus_arg = list(dict.fromkeys([*lines, *DEFAULT_CORPUS]))
                else:
                    corpus_arg = lines
                optimizer: Any = ReinventOptimizer(
                    objective,
                    corpus=corpus_arg,
                    iterations=iterations,
                    batch_size=batch_size,
                    prior_epochs=prior_epochs,
                    anchor_weight=anchor_weight,
                    seed=seed,
                )
            else:
                from pharmaml.generative import GeneticOptimizer

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
            return optimizer.optimize()

        def render(result: Any) -> None:
            self._ga_result = result
            elapsed = result.metadata.get("elapsed_s")
            timing = f" · {elapsed:.1f} s" if isinstance(elapsed, (int, float)) else ""
            mode = "seeded" if seed_mode and lines else (
                "custom corpus" if lines else "built-in corpus"
            )
            if method == "reinforcement learning (RL)":
                mode = f"RL · {mode}"
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
            if method == "reinforcement learning (RL)":
                stats = _format_rl_history(result)
            else:
                stats = _format_operator_stats(result.metadata)
            self._ga_stats_text.delete("1.0", "end")
            self._ga_stats_text.insert("1.0", stats)
            if want_plot:
                try:
                    if method == "reinforcement learning (RL)":
                        from pharmaml.viz import plot_rl_introspection

                        figure = plot_rl_introspection(result)
                    else:
                        from pharmaml.viz import plot_operator_trajectory

                        figure = plot_operator_trajectory(result)
                    self._render_figure(self._ga_host, figure)
                except Exception as exc:  # noqa: BLE001
                    self._show_error("Introspection plot failed", exc)
            else:
                self._close_embedded(self._ga_host)
                for child in self._ga_host.winfo_children():
                    child.destroy()
            self._render_pareto(result)
            if method == "reinforcement learning (RL)" and not result.smiles:
                # an undertrained prior samples mostly invalid strings:
                # the honest advisory instead of an empty leaderboard
                messagebox.showinfo(
                    "No valid molecules yet",
                    "The RL run finished but sampled no valid SMILES -- the "
                    "prior is still undertrained.\n\n"
                    "Raise 'prior epochs' (8+ is the CLI default), keep "
                    "iterations at 10+, and re-run: the anchor needs a "
                    "prior that already writes chemistry.",
                )

        if method == "reinforcement learning (RL)":
            label = f"RL optimising {iterations} iterations (anchor {anchor_weight:g})"
        else:
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


def _format_rl_history(result: Any) -> str:
    """Render an RL run's ``history`` first→last as a compact learning table.

    The 100x-66 introspection made the drift measurable; this is its GUI
    twin -- the numbers the figure plots, in text form.
    """
    history = list(getattr(result, "history", []) or [])
    if not history:
        return "RL history appears here after a run"
    first, last = history[0], history[-1]

    def _fmt(key: str) -> str:
        a, b = first.get(key), last.get(key)
        if a is None or b is None:
            return f"  {key:<16} (missing)"
        return f"  {key:<16} {a:>9.3f} → {b:>9.3f}"

    lines = ["RL run — history (first → last iteration)"]
    lines.append(_fmt("best_score"))
    lines.append(_fmt("mean_score"))
    lines.append(_fmt("valid_fraction"))
    # the prior-drift diagnostics (the anchor penalty's own quantities)
    drift_a, drift_b = first.get("drift"), last.get("drift")
    if drift_a is not None and drift_b is not None:
        lines.append(
            f"  {'drift':<16} {drift_a:>9.3f} → {drift_b:>9.3f}   "
            "(the squared agent-vs-prior gap the anchor penalises; "
            "iteration 0 is exactly 0.0)"
        )
    lines.append(_fmt("entropy"))
    lines.append(_fmt("agent_logprob"))
    lines.append(_fmt("prior_logprob"))
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
    real ``~/.pharmaml/playground-session.json``.  Since 100x-106 the
    self-test also exercises the manual-hyperparameter card end to end
    (dataset load → model pick → panel build → typed parse → JSON
    persistence) and the progress plumbing (report → poll → determinate
    bar + ETA label), asserting loudly instead of trusting the build.
    """
    app = PlaygroundApp(persist_session=False)
    for _ in range(4):
        app.update_idletasks()
        app.update()
    for index in range(app._book.index("end")):
        app._book.select(index)
        app.update()
    app.update()

    # -- manual hyperparameters (100x-106) ---------------------------- #
    app._ml_load_example()                       # qsar_regression.csv
    assert app._model_combo.get() == "auto", "a fresh load resets the model"
    app._model_combo.set("ridge")
    app._ml_refresh_hyperparam_panel()
    entries = getattr(app, "_ml_param_entries", {})
    assert list(entries) == ["alpha"], f"ridge's grid knobs: {list(entries)}"
    assert entries["alpha"].get().strip() == "", "blank field = estimator default"
    entries["alpha"].delete(0, "end")
    entries["alpha"].insert(0, "0.5")
    app._ml_sync_params_var()
    assert json.loads(app._ml_params_var.get()) == {"ridge": {"alpha": "0.5"}}, (
        "per-model JSON persistence"
    )
    parsed = app._ml_manual_params("ridge", "regression")
    assert parsed == {"alpha": 0.5}, f"typed parse: {parsed!r}"
    entries["alpha"].delete(0, "end")
    entries["alpha"].insert(0, "not-a-number")
    try:
        app._ml_manual_params("ridge", "regression")
        raise AssertionError("a bad float must raise with the field named")
    except ValueError as exc:
        assert "alpha" in str(exc)
    entries["alpha"].delete(0, "end")
    entries["alpha"].insert(0, "0.5")            # leave a pinned value
    app._ml_params_reset()
    assert app._ml_params_var.get() == "{}", "reset clears the JSON"
    assert app._ml_manual_params("ridge", "regression") is None

    # -- progress + ETA plumbing (100x-106) --------------------------- #
    assert _fmt_duration(0) == "0 s" and _fmt_duration(42.4) == "42 s"
    assert _fmt_duration(192) == "3 min 12 s" and _fmt_duration(3720) == "1 h 02 min"
    assert _fmt_duration(-5) == "0 s", "negative ETAs clamp to zero"
    assert parse_typed_value("true", False) is True
    assert parse_typed_value("0", True) is False
    assert parse_typed_value("7", None) == 7 and parse_typed_value("none", None) is None
    assert parse_typed_value("0.25", 1.0) == 0.25 and parse_typed_value("l1", "l2") == "l1"
    assert parse_typed_value("(128, 64)", (100,)) == (128, 64)
    assert _accepts_progress_argument(lambda report: None) is True
    assert _accepts_progress_argument(lambda: None) is False
    app._busy = True
    app._job_label_base = "probe"
    app._job_started = time.monotonic() - 4.0    # 4 s "elapsed"
    app._report_progress(3, 10, "best 0.71")
    app._poll_progress()
    assert str(app._progress.cget("mode")) == "determinate"
    assert int(app._progress["value"]) == 3 and int(app._progress.cget("maximum")) == 10
    label = app._job_label.cget("text")
    assert label.startswith("probe — 3/10 · ~") and label.endswith("left · best 0.71"), (
        f"progress label: {label!r}"
    )
    app._report_progress(2, None, "")            # total-less degradation
    app._poll_progress()
    assert app._job_label.cget("text") == "probe — 2"
    app._busy = False
    app._report_progress(9, 10, "straggler")     # dropped when idle
    app._poll_progress()
    assert app._job_label.cget("text") == "probe — 2"

    app.destroy()
    print("self-test: UI built, every tab visited, manual hyperparameters "
          "+ progress/ETA verified, no errors")
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
