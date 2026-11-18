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
                  (data-size vs CV score),
                  :func:`plot_applicability_domain` (the uploaded AD
                  workflow's combined reliability figure — training
                  distance density + distance-vs-prediction scatter);
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
* multi-objective ML — :func:`plot_study_pareto` (full-objective front);
* molecular dynamics — :func:`plot_md_series` (metric traces),
                  :func:`plot_md_correlation` (aligned metric heatmap);
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
                  entropy — four panels over ``history``) and
                  :func:`plot_sa_distribution` (a generation run's raw
                  Ertl SA-score histogram with the drug-like band
                  shaded — part3 section 24.3);
* molecular
  dynamics      — :func:`plot_md_series` (a parsed GROMACS metric as a
                  line chart with a mean reference line and the uploaded
                  pipeline's centered moving average as an explicit
                  overlay, styled by the ten-metric classifier) and
                  :func:`plot_md_correlation` (the cross-metric Pearson
                  heatmap over a ligand's ``{name: XvgSeries}`` mapping,
                  computed by the pipeline's own min-length-aligned
                  :func:`pharmaml.md.study.metric_correlations` — the
                  Python twins of the web Molecular-Dynamics tab);
* everywhere    — :class:`HTMLReport` (self-contained report with
                  embedded figures), :func:`export_table`
                  (csv/tsv/markdown/json/html), :func:`save_pdf_report`
                  (multi-page PDF document — the legacy scripts' only
                  report format), :func:`export_docx` (Word tables),
                  :func:`save_visual_bundle`, and the Part-3 §17
                  first-class image-export surface:
                  :func:`export_image` / :func:`export_matrix` /
                  :func:`export_figure` (PNG/TIFF/JPEG — 16-bit
                  TIFFs, multi-page stacks, JPEG quality, validated
                  writes) plus :class:`ExportManager` and
                  :func:`sniff_format`/ :func:`validate_export`
                  (magic-byte honesty checks).

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
    plot_selfies_robustness,
    plot_similarity_heatmap,
    plot_table_comparison,
)
from pharmaml.viz.docking_plots import (
    plot_consensus_ranking,
    plot_contact_map,
    plot_energy_distribution,
    plot_plif_heatmap,
    plot_receptor_atom_types,
    plot_top_ligands,
)
from pharmaml.viz.export import (
    EXPORT_EXTENSIONS,
    EXPORT_FILTERS,
    TIFF_COMPRESSIONS,
    ExportFormat,
    ExportManager,
    ExportResult,
    ImageExportOptions,
    export_figure,
    export_image,
    export_matrix,
    sniff_format,
    validate_export,
)
from pharmaml.viz.generative_plots import (
    plot_mcts_introspection,
    plot_operator_trajectory,
    plot_pareto_front,
    plot_rl_introspection,
    plot_sa_distribution,
)
from pharmaml.viz.md_plots import (
    plot_md_correlation,
    plot_md_series,
)
from pharmaml.viz.ml_plots import (
    plot_applicability_domain,
    plot_calibration_curve,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_cv_scores,
    plot_edf,
    plot_feature_importance,
    plot_gmm_selection,
    plot_gp_uncertainty,
    plot_hexbin_density,
    plot_intermediate_values,
    plot_kde_density,
    plot_kmeans_diagnostics,
    plot_learning_curve,
    plot_manifold_embedding,
    plot_model_comparison,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_pca_variance,
    plot_permutation_heatmap,
    plot_permutation_importance,
    plot_pr_curves,
    plot_predicted_vs_actual,
    plot_prediction_intervals,
    plot_residuals,
    plot_roc_curves,
    plot_shap_beeswarm,
    plot_shap_dependence,
    plot_shap_force,
    plot_shap_group_difference,
    plot_shap_heatmap,
    plot_shap_importance,
    plot_shap_waterfall,
    plot_signed_correlations,
    plot_slice,
    plot_timeline,
    plot_training_history,
)
from pharmaml.viz.multiobjective_plots import plot_study_pareto
from pharmaml.viz.pharma_plots import (
    plot_concentration_profile,
    plot_dose_accumulation,
)
from pharmaml.viz.report import (
    PDF_THEMES,
    REPORT_THEMES,
    TABLE_FORMATS,
    HTMLReport,
    export_docx,
    export_table,
    metrics_to_dataframe,
    save_pdf_report,
    save_visual_bundle,
)

__all__ = [
    # image export (Part-3 §17)
    "EXPORT_EXTENSIONS",
    "EXPORT_FILTERS",
    # backend
    "PALETTE",
    "PDF_THEMES",
    "PLOT_FORMATS",
    "REPORT_THEMES",
    "TABLE_FORMATS",
    "TIFF_COMPRESSIONS",
    "ExportFormat",
    "ExportManager",
    "ExportResult",
    # reports & tables
    "HTMLReport",
    "ImageExportOptions",
    "close_figure",
    "export_docx",
    "export_figure",
    "export_image",
    "export_matrix",
    "export_table",
    "figure_to_base64",
    "get_pyplot",
    "metrics_to_dataframe",
    "new_figure",
    # ML figures
    "plot_applicability_domain",
    "plot_calibration_curve",
    "plot_class_distribution",
    # pharma figures
    "plot_concentration_profile",
    "plot_confusion_matrix",
    "plot_consensus_ranking",
    "plot_contact_map",
    "plot_correlation_heatmap",
    "plot_cv_scores",
    # chem figures
    "plot_descriptor_distributions",
    "plot_dose_accumulation",
    "plot_edf",
    "plot_energy_distribution",
    "plot_feature_importance",
    "plot_gmm_selection",
    "plot_gp_uncertainty",
    "plot_hexbin_density",
    "plot_intermediate_values",
    "plot_kde_density",
    "plot_kmeans_diagnostics",
    "plot_learning_curve",
    "plot_manifold_embedding",
    "plot_mcts_introspection",
    "plot_md_correlation",
    "plot_md_series",
    "plot_model_comparison",
    "plot_operator_trajectory",
    "plot_optimization_history",
    "plot_parallel_coordinate",
    "plot_param_importances",
    "plot_pareto_front",
    "plot_pca_variance",
    "plot_permutation_heatmap",
    "plot_permutation_importance",
    "plot_plif_heatmap",
    "plot_pr_curves",
    "plot_predicted_vs_actual",
    "plot_prediction_intervals",
    "plot_receptor_atom_types",
    "plot_residuals",
    "plot_rl_introspection",
    "plot_roc_curves",
    "plot_sa_distribution",
    "plot_selfies_robustness",
    "plot_shap_beeswarm",
    "plot_shap_dependence",
    "plot_shap_force",
    "plot_shap_group_difference",
    "plot_shap_heatmap",
    "plot_shap_importance",
    "plot_shap_waterfall",
    "plot_signed_correlations",
    "plot_similarity_heatmap",
    "plot_slice",
    "plot_study_pareto",
    "plot_table_comparison",
    "plot_timeline",
    "plot_top_ligands",
    "plot_training_history",
    "require_matplotlib",
    "save_figure",
    "save_pdf_report",
    "save_visual_bundle",
    "sniff_format",
    "validate_export",
]
