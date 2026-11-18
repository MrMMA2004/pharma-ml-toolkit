"""The plot-coverage registry (part8 §4.2).

Records **every public visualization function** in the eight backend
plotting modules and its integration status, as the directive demands:

    Build a plot coverage registry that records every public
    visualization function and its integration status.

The registry is introspected (never hand-typed): it reads the modules'
``__all__`` and their callables, so a new backend plot can never be
forgotten here — adding it to a module's ``__all__`` makes it appear.

Each entry carries:

* ``function`` — the qualified name (``ml_plots.plot_roc_curves``);
* ``workspace`` — the §12 workspace whose workflow it serves;
* ``title`` — the human title (from the docstring's first line);
* ``status`` — one of ``wired`` (a GUI surface renders it through the
  shared FigureCanvasHost), ``native-equivalent`` (a Qt-native chart
  shows the same information — the documented mapping §4.1 allows),
  or ``backend-only`` (no GUI surface yet — honest, not advertised).

The registry powers the Image Lab's "Coverage" browser: high-value
plots are exposed in their correct workspace; internal diagnostics
stay reachable without being dumped as buttons.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

#: The eight §4 modules, in the directive's own order.
PLOT_MODULES: tuple[str, ...] = (
    "chem_plots",
    "docking_plots",
    "generative_plots",
    "md_plots",
    "ml_plots",
    "multiobjective_plots",
    "pharma_plots",
    "statplots",
)

#: Which §12 workspace each module serves (its natural home).
MODULE_WORKSPACE: dict[str, str] = {
    "chem_plots": "datasets",            # descriptor/chemistry diagnostics
    "docking_plots": "molecular-docking",
    "generative_plots": "generative-ai",
    "md_plots": "molecular-dynamics-analysis",
    "ml_plots": "machine-learning",
    "multiobjective_plots": "generative-ai",
    "pharma_plots": "pkpd",
    "statplots": "machine-learning",
}

WIRED = "wired"
NATIVE_EQUIVALENT = "native-equivalent"
BACKEND_ONLY = "backend-only"

#: Functions already rendered through the shared FigureCanvasHost in a
#: GUI surface (kept in sync with the pages that wire them; the test
#: suite cross-checks this table against the modules' real __all__).
_WIRED: dict[str, tuple[str, ...]] = {
    # ML Workbench / Model Analysis render these through the host
    "ml_plots": (
        "plot_confusion_matrix",
        "plot_roc_curves",
        "plot_pr_curves",
        "plot_calibration_curve",
        "plot_learning_curve",
        "plot_training_history",
        "plot_predicted_vs_actual",
        "plot_residuals",
        "plot_model_comparison",
        "plot_class_distribution",
        "plot_feature_importance",
        "plot_permutation_importance",
        "plot_cv_scores",
        "plot_applicability_domain",
        "plot_correlation_heatmap",
    ),
    # statplots: the Statistics tab's figure surfaces
    "statplots": (
        "plot_histogram",
        "plot_box",
        "plot_ecdf",
        "plot_regression",
    ),
    # MD page renders these two through the host already (round 250)
    "md_plots": ("plot_md_series", "plot_md_correlation"),
    # PK/PD exposure figures
    "pharma_plots": ("plot_concentration_profile", "plot_dose_accumulation"),
    # Generative: SA distribution + pareto front surfaces
    "generative_plots": ("plot_sa_distribution", "plot_pareto_front"),
}

#: Qt-native charts that show the same information (the §4.1 documented
#: mapping — "existing Qt-native charts may remain where they are more
#: appropriate").  The entry records WHICH surface carries it.
_NATIVE: dict[str, tuple[str, ...]] = {
    "ml_plots": (
        "plot_shap_beeswarm",   # Model Analysis explains tab (pyqtgraph)
    ),
    "chem_plots": (
        "plot_descriptor_distributions",  # Chemistry Lab profile tab
    ),
}


@dataclasses.dataclass(frozen=True)
class PlotCoverageEntry:
    """One public visualization function and its GUI status."""

    function: str          # "ml_plots.plot_roc_curves"
    module: str            # "ml_plots"
    workspace: str         # §12 workspace key
    title: str             # docstring first line
    status: str            # wired | native-equivalent | backend-only

    def as_row(self) -> tuple[str, str, str, str]:
        return (self.function, self.workspace, self.title, self.status)


def _first_doc_line(func: Any) -> str:
    doc = (func.__doc__ or "").strip()
    if not doc:
        return "(undocumented)"
    # drop the trailing period; the tables read cleaner
    line = doc.splitlines()[0].strip()
    return line.removesuffix(".")


def plot_coverage_table() -> list[PlotCoverageEntry]:
    """Introspect the eight modules and build the coverage table.

    Import failures surface as zero-entry modules (the registry stays
    importable for the GUI even on a broken install).
    """
    entries: list[PlotCoverageEntry] = []
    for module_name in PLOT_MODULES:
        try:
            module = importlib.import_module(f"pharmaml.viz.{module_name}")
        except Exception:  # noqa: BLE001,S112 - registry never blocks the GUI
            continue
        exported = list(getattr(module, "__all__", ()))
        wired = set(_WIRED.get(module_name, ()))
        native = set(_NATIVE.get(module_name, ()))
        for name in exported:
            func = getattr(module, name, None)
            if func is None or not callable(func):
                continue
            status = (
                WIRED if name in wired
                else NATIVE_EQUIVALENT if name in native
                else BACKEND_ONLY
            )
            entries.append(
                PlotCoverageEntry(
                    function=f"{module_name}.{name}",
                    module=module_name,
                    workspace=MODULE_WORKSPACE[module_name],
                    title=_first_doc_line(func),
                    status=status,
                )
            )
    return entries


def coverage_counts() -> dict[str, int]:
    """Total / wired / native-equivalent / backend-only counts."""
    table = plot_coverage_table()
    return {
        "total": len(table),
        WIRED: sum(1 for e in table if e.status == WIRED),
        NATIVE_EQUIVALENT: sum(1 for e in table if e.status == NATIVE_EQUIVALENT),
        BACKEND_ONLY: sum(1 for e in table if e.status == BACKEND_ONLY),
    }
