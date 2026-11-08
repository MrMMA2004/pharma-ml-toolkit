"""Matplotlib backend helpers — lazy, headless-safe, format-aware.

PharmaML keeps matplotlib an *optional* dependency (the ``reports``
extra), exactly like RDKit or the boosting libraries.  Every plotting
function in :mod:`pharmaml.viz` routes through :func:`require_matplotlib`
so the failure mode is one clear :class:`~pharmaml.exceptions.DependencyError`
instead of an ``ImportError`` deep inside a run.

The non-interactive ``Agg`` backend is enforced before ``pyplot`` is
touched, which makes the module safe on headless servers and CI
containers (the legacy scripts crashed here with
``_tkinter.TclError: no display name``).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from pharmaml.exceptions import DataValidationError, DependencyError

__all__ = [
    "PALETTE",
    "PLOT_FORMATS",
    "close_figure",
    "figure_to_base64",
    "get_pyplot",
    "new_figure",
    "require_matplotlib",
    "save_figure",
    "style_axis",
]

#: File formats :func:`save_figure` can write (matplotlib backends).
PLOT_FORMATS: tuple[str, ...] = ("png", "svg", "pdf")

#: Colour-blind-friendly accents used consistently across all figures.
PALETTE: tuple[str, ...] = (
    "#0F766E",  # teal
    "#B45309",  # amber-700
    "#525252",  # neutral-600
    "#7F1D1D",  # red-900
    "#4D7C0F",  # lime-700
    "#6D28D9",  # violet-700
)

_MIME = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}

#: Fixed salt for SVG element ids — replaces matplotlib's random-per-save
#: uuid4 so two saves of the same figure produce byte-identical SVG files.
_SVG_HASHSALT = "pharmaml"


def _reproducible_kwargs(fmt: str) -> dict[str, Any]:
    """Extra ``savefig`` kwargs that make svg/pdf output byte-reproducible.

    PNG files are already timestamp-free; svg otherwise embeds a save-time
    ``<dc:date>`` and random element ids, pdf a second-resolution
    ``/CreationDate`` — both stripped here so hashing/pinning flows can
    compare artifacts across processes. PNG takes no kwargs and therefore
    keeps its exact historical bytes.
    """
    if fmt == "svg":
        return {"metadata": {"Date": None}}
    if fmt == "pdf":
        return {"metadata": {"CreationDate": None}}
    return {}


def _ipython_shell_running() -> bool:
    """True only when an IPython shell is actually RUNNING.

    The notebook detection upgrade (100x-160): the old guard —
    ``"IPython" not in sys.modules`` — was fooled by any library that
    merely IMPORTS IPython.  ``import shap`` does exactly that (its
    plotting stack pulls IPython in), so a desktop GUI that had just
    explained a tree model silently skipped the headless ``Agg`` pin,
    pyplot resolved a GUI backend under ``$DISPLAY``, and every figure
    the GUI's worker threads drew became an off-main-thread Tk window
    creation — the intermittent ``Illegal instruction`` crash the
    round's new SHAP surfaces exposed (the old beeswarm tests only
    survived full-suite runs because an earlier test had forced Agg
    directly).  The honest test is the classic one: IPython's own
    ``get_ipython()`` returns the running interactive shell or ``None``
    — a bare import without a shell is NOT a notebook.
    """
    import sys

    module = sys.modules.get("IPython")
    if module is None:
        return False
    try:
        get_ipython = module.get_ipython
    except AttributeError:  # pragma: no cover - a stub without the API
        return False
    return get_ipython() is not None


def require_matplotlib():
    """Import matplotlib lazily and pick the right backend.

    In a plain Python process (scripts, CI, servers — including any
    process that merely imported IPython without a running shell, see
    :func:`_ipython_shell_running`) the headless ``Agg`` backend is
    forced — no window, no ``$DISPLAY`` needed, and figure creation
    becomes thread-safe (``Agg`` managers touch no GUI toolkit).  Inside
    a RUNNING Jupyter/IPython session the *existing inline backend is
    kept* so figures produced by this package render directly in
    notebook cells.

    Raises
    ------
    DependencyError
        If matplotlib is not installed, with the exact extra to install.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on env
        raise DependencyError(
            "matplotlib is required for plotting and HTML reports. "
            "Install it with:  pip install 'pharmaml[reports]'  "
            "(or pip install matplotlib)"
        ) from exc
    # Headless-safe: never open a GUI window, never require $DISPLAY —
    # but don't hijack the inline backend of a running notebook.
    if not _ipython_shell_running():
        matplotlib.use("Agg", force=True)
    return matplotlib


def get_pyplot():
    """Return ``matplotlib.pyplot`` with a safe backend already selected."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    return plt


def new_figure(nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] = (8.0, 6.0), **kwargs):
    """Create a styled ``(figure, axes)`` pair (grid + soft spines).

    All plot functions in this package build on this helper so every
    figure shares the same visual language.
    """
    import numpy as np

    plt = get_pyplot()
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    for ax in np.atleast_1d(axes).ravel():
        style_axis(ax)
    return fig, axes


def style_axis(ax) -> None:
    """Apply the house style to one axis (grid behind data, subtle frame)."""
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save_figure(fig, path: str | Path) -> Path:
    """Save a figure to ``png``/``svg``/``pdf`` (chosen by extension).

    Parent directories are created automatically; returns the resolved
    path so CLI handlers can echo the artifact back to the user. Output
    is byte-reproducible: identical figures hash identically across
    processes (svg/pdf timestamps and random ids are stripped).
    """
    file_path = Path(path)
    fmt = file_path.suffix.lower().lstrip(".")
    if fmt not in PLOT_FORMATS:
        raise DataValidationError(
            f"Unsupported figure format {file_path.suffix!r}; "
            f"supported: {', '.join(PLOT_FORMATS)} (picked from the file extension)"
        )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    mpl = require_matplotlib()
    with mpl.rc_context({"svg.hashsalt": _SVG_HASHSALT}):
        fig.savefig(
            file_path, dpi=150, bbox_inches="tight", **_reproducible_kwargs(fmt)
        )
    return file_path


def figure_to_base64(fig, fmt: str = "png") -> str:
    """Render a figure to a ``data:`` URI for embedding in HTML reports."""
    if fmt not in PLOT_FORMATS:
        raise DataValidationError(
            f"Unsupported figure format {fmt!r}; supported: {', '.join(PLOT_FORMATS)}"
        )
    buffer = io.BytesIO()
    mpl = require_matplotlib()
    with mpl.rc_context({"svg.hashsalt": _SVG_HASHSALT}):
        fig.savefig(
            buffer, format=fmt, dpi=110, bbox_inches="tight",
            **_reproducible_kwargs(fmt),
        )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{_MIME[fmt]};base64,{encoded}"


def close_figure(fig) -> None:
    """Close a figure defensively (no-op when matplotlib is absent)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - depends on env
        return
    plt.close(fig)
