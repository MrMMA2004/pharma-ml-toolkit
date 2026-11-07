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
    "DEFAULT_FIGSIZE",
    "EXPORT_DPI",
    "HTML_DPI",
    "MAX_RASTER_PIXELS",
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

#: Readable defaults shared by notebooks, exported files and HTML reports.
#: Vector formats remain resolution-independent; the DPI still improves any
#: rasterised artists they contain.
DEFAULT_FIGSIZE: tuple[float, float] = (9.0, 6.0)
EXPORT_DPI: int = 220
HTML_DPI: int = 160
#: Avoid multi-hundred-megabyte RGBA buffers for exceptionally large
#: physical figures (for example an 80-label similarity heat-map).
MAX_RASTER_PIXELS: int = 40_000_000


def _bounded_raster_dpi(fig: Any, dpi: float) -> float:
    """Keep normal exports crisp without allocating unbounded rasters.

    SVG/PDF text and paths stay vector regardless of this value; their DPI
    controls rasterized artists such as heat-maps.  A five-percent margin
    covers tight-bound label/padding growth beyond the base figure size.
    """
    width, height = (float(value) for value in fig.get_size_inches())
    pixels = width * height * dpi * dpi
    if pixels <= MAX_RASTER_PIXELS:
        return dpi
    import math

    return max(1.0, dpi * math.sqrt(MAX_RASTER_PIXELS / pixels) * 0.95)


def _positive_number(value: Any, name: str) -> float:
    """Validate a finite, positive display/export scale."""
    if isinstance(value, bool):
        raise DataValidationError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be a finite positive number") from exc
    import math

    if not math.isfinite(number) or number <= 0.0:
        raise DataValidationError(f"{name} must be a finite positive number")
    return number


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
    # set_title() reapplies these rcParams after style_axis() has run.  Keep
    # the typography here as well so every plotting module gets the same
    # treatment; explicit per-plot font settings still take precedence.
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.labelcolor": "#263238",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
        }
    )
    return matplotlib


def get_pyplot():
    """Return ``matplotlib.pyplot`` with a safe backend already selected."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    return plt


def new_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    *,
    scale: float = 1.0,
    layout: str | None = "tight",
    **kwargs: Any,
):
    """Create a readable, consistently styled ``(figure, axes)`` pair.

    ``scale`` multiplies both physical dimensions without changing font
    sizes: it is useful when a dense chart needs more room, rather than a
    blurry upscale of the same cramped layout.  ``layout='tight'`` reserves
    space for labels as artists are added; pass ``'constrained'`` or ``None``
    for matplotlib's other layout policies.  Explicit ``figsize`` and
    ``dpi`` keyword arguments retain their normal matplotlib meaning.
    """
    import numpy as np

    factor = _positive_number(scale, "scale")
    if len(figsize) != 2:
        raise DataValidationError("figsize must contain exactly (width, height)")
    width, height = (
        _positive_number(value, "figsize dimensions") for value in figsize
    )
    kwargs.setdefault("dpi", 120)
    plt = get_pyplot()
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width * factor, height * factor),
        layout=layout,
        **kwargs,
    )
    fig.patch.set_facecolor("white")
    for ax in np.atleast_1d(axes).ravel():
        style_axis(ax)
    return fig, axes


def style_axis(ax) -> None:
    """Apply the house style: quiet grid, legible type and subtle frame."""
    ax.set_facecolor("#FFFFFF")
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        color="#D7DEE5",
        alpha=0.68,
        linewidth=0.7,
    )
    ax.grid(False, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#A8B2BC")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=9,
        colors="#4B5563",
        length=3.5,
        width=0.7,
        pad=3,
    )
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_fontsize(10)
        label.set_color("#263238")
    ax.xaxis.labelpad = 6
    ax.yaxis.labelpad = 7
    ax.title.set_fontsize(12)
    ax.title.set_fontweight("semibold")
    ax.title.set_color("#17242B")


def save_figure(fig, path: str | Path, *, dpi: int = EXPORT_DPI) -> Path:
    """Save a crisp, uncropped ``png``/``svg``/``pdf`` figure.

    The 220-DPI default keeps small labels and thin lines readable in
    documents and zoomed presentations.  ``bbox_inches='tight'`` plus a
    small padding guard prevents title/tick-label clipping.  Parent
    directories are created automatically.  SVG/PDF metadata is stripped
    so identical figures remain byte-reproducible across processes.
    """
    file_path = Path(path)
    fmt = file_path.suffix.lower().lstrip(".")
    if fmt not in PLOT_FORMATS:
        raise DataValidationError(
            f"Unsupported figure format {file_path.suffix!r}; "
            f"supported: {', '.join(PLOT_FORMATS)} (picked from the file extension)"
        )
    resolution = _bounded_raster_dpi(fig, _positive_number(dpi, "dpi"))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    mpl = require_matplotlib()
    with mpl.rc_context({"svg.hashsalt": _SVG_HASHSALT}):
        fig.savefig(
            file_path,
            dpi=resolution,
            bbox_inches="tight",
            pad_inches=0.16,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            **_reproducible_kwargs(fmt),
        )
    return file_path


def figure_to_base64(fig, fmt: str = "png", *, dpi: int = HTML_DPI) -> str:
    """Render a crisp ``data:`` URI for zoomable HTML report figures."""
    if fmt not in PLOT_FORMATS:
        raise DataValidationError(
            f"Unsupported figure format {fmt!r}; supported: {', '.join(PLOT_FORMATS)}"
        )
    resolution = _bounded_raster_dpi(fig, _positive_number(dpi, "dpi"))
    buffer = io.BytesIO()
    mpl = require_matplotlib()
    with mpl.rc_context({"svg.hashsalt": _SVG_HASHSALT}):
        fig.savefig(
            buffer,
            format=fmt,
            dpi=resolution,
            bbox_inches="tight",
            pad_inches=0.16,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
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
