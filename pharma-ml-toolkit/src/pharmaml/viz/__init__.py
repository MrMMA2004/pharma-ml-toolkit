"""Visual output for every PharmaML domain — plots, tables, HTML reports.

One import gives you the full "results in various visual formats"
layer for **all** toolkit codes:

* ML            — :func:`plot_confusion_matrix` (legacy PNG),
                  :func:`plot_roc_curves` (legacy per-class ROC),
                  :func:`plot_predicted_vs_actual` (legacy parity
                  scatter), plus new :func:`plot_residuals`,
                  :func:`plot_model_comparison`,
                  :func:`plot_class_distribution`,
                  :func:`plot_feature_importance`,
                  :func:`plot_cv_scores`;
* pharmaceutics — :func:`plot_concentration_profile` (linear + semi-log
                  with shaded AUC), :func:`plot_dose_accumulation`
                  (steady-state envelope);
* chemistry     — :func:`plot_descriptor_distributions`,
                  :func:`plot_similarity_heatmap`;
* docking       — :func:`plot_energy_distribution`,
                  :func:`plot_top_ligands`;
* everywhere    — :class:`HTMLReport` (self-contained report with
                  embedded figures), :func:`export_table`
                  (csv/tsv/markdown/json/html), :func:`save_visual_bundle`.

matplotlib stays optional: importing this package never needs it —
only *calling* a plotting function does (clear
:class:`~pharmaml.exceptions.DependencyError` with the
``pip install 'pharmaml[reports]'`` hint).  Figures render through the
headless ``Agg`` backend, so servers and CI containers are safe.

Example
-------
>>> from pharmaml.viz import plot_confusion_matrix, save_figure   # doctest: +SKIP
>>> fig = plot_confusion_matrix(report)        # ClassificationReport  # doctest: +SKIP
>>> save_figure(fig, "confusion.png")          # png/svg/pdf            # doctest: +SKIP
"""

from pharmaml.viz._compat import (
    PALETTE,
    PLOT_FORMATS,
    close_figure,
    figure_to_base64,
    get_pyplot,
    new_figure,
    require_matplotlib,
    save_figure,
)
from pharmaml.viz.chem_plots import (
    plot_descriptor_distributions,
    plot_similarity_heatmap,
)
from pharmaml.viz.docking_plots import (
    plot_energy_distribution,
    plot_top_ligands,
)
from pharmaml.viz.ml_plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_cv_scores,
    plot_feature_importance,
    plot_model_comparison,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_roc_curves,
)
from pharmaml.viz.pharma_plots import (
    plot_concentration_profile,
    plot_dose_accumulation,
)
from pharmaml.viz.report import (
    HTMLReport,
    TABLE_FORMATS,
    export_table,
    metrics_to_dataframe,
    save_visual_bundle,
)

__all__ = [
    # backend
    "PALETTE",
    "PLOT_FORMATS",
    "close_figure",
    "figure_to_base64",
    "get_pyplot",
    "new_figure",
    "require_matplotlib",
    "save_figure",
    # ML figures
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_cv_scores",
    "plot_feature_importance",
    "plot_model_comparison",
    "plot_predicted_vs_actual",
    "plot_residuals",
    "plot_roc_curves",
    # pharma figures
    "plot_concentration_profile",
    "plot_dose_accumulation",
    # chem figures
    "plot_descriptor_distributions",
    "plot_similarity_heatmap",
    # docking figures
    "plot_energy_distribution",
    "plot_top_ligands",
    # reports & tables
    "HTMLReport",
    "TABLE_FORMATS",
    "export_table",
    "metrics_to_dataframe",
    "save_visual_bundle",
]
