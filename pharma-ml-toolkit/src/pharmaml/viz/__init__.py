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
                  :func:`plot_cv_scores`,
                  :func:`plot_calibration_curve` (reliability diagram
                  + Brier score), :func:`plot_learning_curve`
                  (data-size vs CV score);
* pharmaceutics — :func:`plot_concentration_profile` (linear + semi-log
                  with shaded AUC), :func:`plot_dose_accumulation`
                  (steady-state envelope);
* chemistry     — :func:`plot_descriptor_distributions`,
                  :func:`plot_similarity_heatmap`,
                  :func:`plot_table_comparison` (the figure twin of
                  the ligand-set reconciliation utilities);
* docking       — :func:`plot_energy_distribution`,
                  :func:`plot_top_ligands`,
                  :func:`plot_consensus_ranking` (the figure twin of
                  the explainable consensus leaderboard),
                  :func:`plot_receptor_atom_types`;
* generative    — :func:`plot_operator_trajectory` (the adaptive GA's
                  operator mix per generation — the "GA that learns"
                  learning curve, synced with the score trajectory every
                  run already carries in ``history``),
                  :func:`plot_pareto_front` (a MultiObjective run's
                  trade-off scatter: dominated points muted, the
                  non-dominated front highlighted) and
                  :func:`plot_rl_introspection` (an RL run's prior-drift
                  story: agent-vs-prior log-probabilities, the squared
                  drift the anchor penalises, and the exploration
                  entropy — four panels over ``history``);
* everywhere    — :class:`HTMLReport` (self-contained report with
                  embedded figures), :func:`export_table`
                  (csv/tsv/markdown/json/html), :func:`save_pdf_report`
                  (multi-page PDF document — the legacy scripts' only
                  report format), :func:`export_docx` (Word tables),
                  :func:`save_visual_bundle`.

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
    plot_table_comparison,
)
from pharmaml.viz.docking_plots import (
    plot_consensus_ranking,
    plot_energy_distribution,
    plot_receptor_atom_types,
    plot_top_ligands,
)
from pharmaml.viz.generative_plots import (
    plot_operator_trajectory,
    plot_pareto_front,
    plot_rl_introspection,
)
from pharmaml.viz.ml_plots import (
    plot_calibration_curve,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_cv_scores,
    plot_feature_importance,
    plot_learning_curve,
    plot_model_comparison,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_roc_curves,
    plot_training_history,
)
from pharmaml.viz.pharma_plots import (
    plot_concentration_profile,
    plot_dose_accumulation,
)
from pharmaml.viz.report import (
    TABLE_FORMATS,
    HTMLReport,
    export_docx,
    export_table,
    metrics_to_dataframe,
    save_pdf_report,
    save_visual_bundle,
)

__all__ = [
    # backend
    "PALETTE",
    "PLOT_FORMATS",
    "TABLE_FORMATS",
    # reports & tables
    "HTMLReport",
    "close_figure",
    "export_docx",
    "export_table",
    "figure_to_base64",
    "get_pyplot",
    "metrics_to_dataframe",
    "new_figure",
    # ML figures
    "plot_calibration_curve",
    "plot_class_distribution",
    # pharma figures
    "plot_concentration_profile",
    "plot_confusion_matrix",
    "plot_consensus_ranking",
    "plot_cv_scores",
    # chem figures
    "plot_descriptor_distributions",
    "plot_dose_accumulation",
    "plot_energy_distribution",
    "plot_feature_importance",
    "plot_learning_curve",
    "plot_model_comparison",
    # generative figures
    "plot_operator_trajectory",
    "plot_pareto_front",
    "plot_predicted_vs_actual",
    "plot_receptor_atom_types",
    "plot_residuals",
    "plot_rl_introspection",
    "plot_roc_curves",
    "plot_similarity_heatmap",
    "plot_table_comparison",
    "plot_top_ligands",
    "plot_training_history",
    "require_matplotlib",
    "save_figure",
    "save_pdf_report",
    "save_visual_bundle",
]
