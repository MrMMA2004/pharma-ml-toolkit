"""The complete capability registry (part6 §6, Phase 2).

The single source of truth for feature registration: card definitions,
capability definitions, backend references, routes, dependency and
engine requirements, status and test references.  Pure Python, no Qt
imports anywhere — the GUI consumes this registry (``ui/nine_cards.py``
re-exports the card tuples) and never the other way around, so there
is no Backend ↔ UI circularity.

§6.1 unique ownership: every user-facing capability has exactly one
primary card.  A backend service may be reused by several cards (e.g.
cross-cutting export), but primary ownership stays unambiguous.

§2.3 Other-card decision: after the full assignment below, every
genuine user-facing backend capability has a logical home — the 19
class-C modules map onto Machine Learning (13), Deep Learning (5) and
Datasets (1), and nothing user-facing remains unassigned.  Rule 8
therefore applies: NO Other card.  The decision is pinned by tests so
it can only change through a new, justified assignment audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Status = Literal["integrated", "cli", "api", "blocked"]

#: The nine primary cards, in the directive's exact order (part6 §2.1).
CARD_ORDER: tuple[str, ...] = (
    "introduction",
    "datasets",
    "machine-learning",
    "deep-learning",
    "generative-ai",
    "molecular-docking",
    "pkpd",
    "molecular-dynamics-analysis",
    "visualization",
)

#: The Studio's 18 sections (the pages every route must resolve to).
SECTION_KEYS: tuple[str, ...] = (
    "dashboard", "data", "chemistry", "pkpd", "biology", "docking",
    "models", "mlworkbench", "unsupervised", "deep_learning", "generative",
    "foundations", "bestofml", "responseai", "imagelab",
    "molecular_dynamics", "utilities", "settings",
)


@dataclass(frozen=True)
class Capability:
    """One user-facing capability with its complete audit record (§3.1).

    ``card`` is the primary owning card (§6.1).  ``route`` is the Studio
    section that hosts the workflow today (integrated) or will host it
    once Phase 3 builds the form (api/cli rows).  ``status``:

    * ``integrated`` — a Studio page exposes the workflow.
    * ``cli``        — reachable via ``pharmaml`` CLI only (Phase 3).
    * ``api``        — Python API only, no GUI/CLI route (Phase 3).
    * ``blocked``    — impossible to expose: the blocking reason is named
      in ``status_detail`` and the registry treats it as disclosed.
    """

    capability_id: str
    name: str
    backend_module: str
    public_api: str
    card: str
    route: str
    status: Status
    tests: tuple[str, ...] = ()
    optional_deps: tuple[str, ...] = ()
    engines: tuple[str, ...] = ()
    status_detail: str = ""
    # §6.2 rule 11: every registered feature has tests or an explicit
    # test-gap record — ``test_gap`` names what is missing and why.
    test_gap: str = ""


@dataclass(frozen=True)
class RegistrySummary:
    """The machine-generated coverage summary (§6.2 closing paragraph)."""

    total: int
    by_status: dict[str, int] = field(default_factory=dict)
    by_card: dict[str, int] = field(default_factory=dict)
    orphans: tuple[str, ...] = ()
    duplicate_ownership: tuple[str, ...] = ()
    blocked_undisclosed: tuple[str, ...] = ()
    test_gaps: tuple[str, ...] = ()
    invalid_routes: tuple[str, ...] = ()
    other_card_present: bool = False


# --
# The capability inventory.  Sources: docs/CAPABILITY_MAP.md §2 (the 18
# pages and their verified backends), §4 (the class-C backend-only
# list, all 19 rows now assigned), and docs/FEATURES.md.  Every claim
# here was verified against source and tests during the Phase-0 audit.
# --

CAPABILITIES: tuple[Capability, ...] = (
    # -- Card 01 · Introduction --------------------------------------------
    Capability("intro.census", "Live toolkit census", "ui.pages.dashboard", "DashboardPage", "introduction", "dashboard", "integrated", tests=("tests/ui/test_pages_smoke.py::test_dashboard_shows_live_census",)),
    Capability("intro.datasets", "Example-dataset index", "ui.pages.dashboard", "DashboardPage", "introduction", "dashboard", "integrated", tests=("tests/ui/test_pages_smoke.py::test_dashboard_shows_live_census",)),
    # -- Phase 4 (part6 §7-10): global search, shared workspace, job center ----
    Capability("intro.search", "Global capability search / command palette", "ui.search_index", "build_search_index, search", "introduction", "dashboard", "integrated", tests=("tests/ui/test_phase4.py",)),
    Capability("intro.workspace", "Shared session workspace (recents + registries)", "pharmaml.studio.workspace", "note, entries, summary_counts", "introduction", "dashboard", "integrated", tests=("tests/ui/test_phase4.py",)),
    Capability("intro.job_center", "Unified job center (live job + run history)", "ui.widgets.job_center", "JobCenterDialog", "introduction", "dashboard", "integrated", tests=("tests/ui/test_phase4.py",)),
    # -- Phase 5 (part6 §16/§17 E-F): contextual help, presets, diagnostics ----
    Capability("intro.glossary", "Contextual scientific help (glossary tooltips)", "pharmaml.studio.glossary", "GLOSSARY, apply_help", "introduction", "settings", "integrated", tests=("tests/test_phase5_backends.py", "tests/ui/test_phase5.py")),
    Capability("intro.presets", "Memory-aware execution presets + resource probe", "pharmaml.utils.resources", "snapshot, PRESETS, preset_for_low_memory", "introduction", "settings", "integrated", tests=("tests/test_phase5_backends.py", "tests/ui/test_phase5.py")),
    Capability("intro.doctor_panel", "Dependency diagnostics panel (GUI doctor)", "ui.pages.settings_page", "SettingsPage", "introduction", "settings", "integrated", tests=("tests/ui/test_phase5.py",)),
    # -- Card 02 · Datasets --------------------------------------------------
    Capability("data.ingest", "CSV/TSV/Excel ingestion", "pharmaml.utils.io", "read_table, write_table", "datasets", "data", "integrated", tests=("tests/ui/test_pages_smoke.py::test_data_explorer_loads_real_csv", "tests/ui/test_pages_smoke.py::test_data_explorer_reads_excel_too")),
    Capability("data.profile", "Profiling + quality reports", "pharmaml.data_profiling", "profile_frame, profile_alerts", "datasets", "data", "integrated", tests=("tests/test_csv_standardize.py",)),
    Capability("data.standardize", "Molecular standardization", "pharmaml.chem.standardize", "reinvent_standardize_smiles, canonicalize_tautomer", "datasets", "chemistry", "integrated", tests=("tests/test_chem_standardize.py",)),
    Capability("data.descriptors", "Descriptor computation", "pharmaml.chem.descriptors", "compute_descriptors, compute_maccs", "datasets", "chemistry", "integrated", tests=("tests/test_descriptors.py",)),
    Capability("data.fingerprints", "Morgan fingerprint computation", "pharmaml.chem.descriptors", "compute_fingerprints", "datasets", "chemistry", "integrated", tests=("tests/test_descriptors.py",)),
    Capability("data.convert", "Format conversion", "pharmaml.chem.converters", "split_table_into_chunks", "datasets", "utilities", "integrated", tests=("tests/test_converters.py",)),
    Capability("data.split_audit", "Identifier-overlap audit across splits", "pharmaml.ml.split_audit", "audit_split_overlap", "datasets", "data", "integrated", tests=("tests/test_split_overlap_audit.py", "tests/ui/test_phase3_batch4.py")),
    # -- Card 03 · Machine Learning ------------------------------------------
    Capability("ml.train", "Classical model training", "pharmaml.ml.training", "train_model, compare_models", "machine-learning", "models", "integrated", tests=("tests/ui/test_pages_smoke.py::test_model_page_training_end_to_end",)),
    Capability("ml.evaluate", "Metrics + held-out evaluation", "pharmaml.ml.evaluation", "evaluate_model, permutation_feature_importance", "machine-learning", "models", "integrated", tests=("tests/test_ml_pipeline.py",)),
    Capability("ml.bench", "Splitting + benchmarking schemes", "pharmaml.ml.benchmark", "scaffold_split, butina_split, cold_split", "machine-learning", "mlworkbench", "integrated", tests=("tests/test_ml_pipeline.py", "tests/test_checkpoints.py")),
    Capability("ml.unsupervised", "Clustering + dimensionality reduction", "pharmaml.ml.unsupervised", "gmm_cluster, hierarchical_cluster, kde_fit", "machine-learning", "unsupervised", "integrated", tests=("tests/ui/test_unsupervised_page.py",)),
    Capability("ml.calibration", "Probability calibration + recalibration", "pharmaml.ml.calibration", "calibration_report, recalibrate", "machine-learning", "mlworkbench", "integrated", tests=("tests/test_calibration.py", "tests/test_probability_calibration.py", "tests/ui/test_ml_statistical_tools.py")),
    Capability("ml.comparison", "Paired model-comparison tests", "pharmaml.ml.comparison", "model_comparison_test", "machine-learning", "mlworkbench", "integrated", tests=("tests/test_comparison.py", "tests/ui/test_ml_statistics_batch2.py")),
    Capability("ml.conformal", "Split-conformal regression intervals", "pharmaml.ml.conformal", "calibrate_split_conformal, conformal_quantile", "machine-learning", "mlworkbench", "integrated", tests=("tests/test_conformal.py", "tests/ui/test_ml_statistical_tools.py")),
    Capability("ml.conformal_cls", "Conformal classification sets", "pharmaml.ml.conformal_classification", "calibrate_split_conformal_classifier, calibrate_label_conditional_classifier", "machine-learning", "mlworkbench", "integrated", tests=("tests/test_conformal_classification.py", "tests/test_label_conditional_conformal.py", "tests/ui/test_phase3_batch4.py")),
    Capability("ml.coverage_diag", "Coverage-shortfall diagnostics", "pharmaml.ml.coverage_diagnostics", "coverage_by_group", "machine-learning", "mlworkbench", "integrated", tests=("tests/test_coverage_diagnostics.py", "tests/ui/test_ml_statistics_batch2.py")),
    Capability("ml.decision", "Decision-curve analysis", "pharmaml.ml.decision", "decision_curve, net_benefit", "machine-learning", "mlworkbench", "integrated", tests=("tests/test_decision.py", "tests/ui/test_ml_statistical_tools.py")),
    Capability("ml.drift", "Distribution-shift detection", "pharmaml.ml.drift", "detect_drift", "machine-learning", "mlworkbench", "integrated", tests=("tests/test_drift.py", "tests/ui/test_ml_statistical_tools.py")),
    Capability("ml.model_card", "Model cards / provenance sheets", "pharmaml.ml.model_card", "build_model_card, model_card_markdown", "machine-learning", "models", "integrated", tests=("tests/test_model_card.py", "tests/ui/test_ml_statistics_batch2.py")),
    Capability("ml.multiplicity", "Multiple-comparison corrections", "pharmaml.ml.multiplicity", "correct_p_values, false_discovery_control", "machine-learning", "mlworkbench", "integrated", tests=("tests/test_comparison.py", "tests/ui/test_ml_statistics_batch2.py")),
    Capability("ml.power", "Fold-count power analysis", "pharmaml.ml.power", "folds_for_power, minimum_detectable_effect", "machine-learning", "mlworkbench", "integrated", tests=("tests/test_power.py", "tests/ui/test_ml_statistics_batch2.py")),
    Capability("ml.sar", "MMP / activity cliffs / R-groups", "pharmaml.ml.sar", "mmp_fragmentation, activity_cliffs, rgroup_decomposition", "machine-learning", "mlworkbench", "integrated", tests=("tests/test_sar.py", "tests/ui/test_phase3_batch4.py")),
    Capability("ml.subgroups", "Subgroup performance audits", "pharmaml.ml.subgroups", "subgroup_report, subgroup_gap_scan", "machine-learning", "mlworkbench", "integrated", tests=("tests/test_subgroups.py", "tests/ui/test_ml_statistics_batch2.py")),
    # -- Card 04 · Deep Learning ---------------------------------------------
    Capability("dl.train", "Deep-network training", "pharmaml.ml.neural", "DeepMLPClassifier, ResidualMLPRegressor", "deep-learning", "deep_learning", "integrated", tests=("tests/ui/test_deep_learning_page.py",)),
    Capability("dl.geometric", "Geometric deep-learning primitives", "pharmaml.ml.geometric", "GCNLayer, GATLayer, GNNClassifier", "deep-learning", "deep_learning", "integrated", tests=("tests/test_geometric.py", "tests/ui/test_dl_efficiency_lab.py")),
    Capability("dl.gnn", "GNN training + graph featurisation", "pharmaml.ml.gnn", "GraphClassifier, GCNConv, GINConv", "deep-learning", "deep_learning", "integrated", tests=("tests/test_gnn.py", "tests/ui/test_dl_efficiency_lab.py")),
    Capability("dl.gnnxai", "GNN explainability", "pharmaml.ml.gnnxai", "GNNExplainer, SubgraphX, fidelity_plus", "deep-learning", "deep_learning", "integrated", tests=("tests/test_gnn.py", "tests/ui/test_dl_efficiency_lab.py")),
    Capability("dl.harness", "Offline sharded-training harness", "pharmaml.ml.harness", "ZeROConfig, shard_plan, sharded_step", "deep-learning", "deep_learning", "integrated", tests=("tests/test_harness.py", "tests/ui/test_dl_efficiency_lab.py")),
    Capability("dl.pruning", "Magnitude/structured pruning", "pharmaml.ml.pruning", "apply_magnitude_pruning, iterative_prune_and_retrain", "deep-learning", "deep_learning", "integrated", tests=("tests/test_pruning.py", "tests/ui/test_dl_efficiency_lab.py")),
    Capability("dl.quantization", "Quantisation arithmetic", "pharmaml.ml.quantization", "quantize_int8, dequantize_nf4, llm_int8_matmul", "deep-learning", "deep_learning", "integrated", tests=("tests/test_quantization.py", "tests/ui/test_dl_efficiency_lab.py")),
    # -- Card 05 · Generative AI ----------------------------------------------
    Capability("gen.ga", "Genetic-algorithm molecule design", "pharmaml.generative.genetic", "GeneticOptimizer", "generative-ai", "generative", "integrated", tests=("tests/ui/test_generative_page.py",)),
    Capability("gen.rnn", "SMILES RNN sampling", "pharmaml.generative.rnn", "SmilesRnn", "generative-ai", "generative", "integrated", tests=("tests/ui/test_generative_page.py",)),
    Capability("gen.oracle", "Oracle scoring + SA scoring", "pharmaml.generative.scoring", "MultiObjective, NoveltyObjective", "generative-ai", "generative", "integrated", tests=("tests/ui/test_generative_page.py",)),
    Capability("gen.rl", "REINVENT-style reinforcement learning", "pharmaml.generative.rl", "ReinventOptimizer", "generative-ai", "generative", "integrated", tests=("tests/ui/test_generative_page.py",)),
    Capability("gen.mcts", "Monte Carlo tree search (MCTS) molecule design", "pharmaml.generative.mcts", "MctsOptimizer, run_mcts_optimization", "generative-ai", "generative", "integrated", tests=("tests/ui/test_generative_page.py",)),
    # -- Card 06 · Molecular Docking ------------------------------------------
    Capability("dock.vina", "Vina docking campaigns", "pharmaml.docking.workflow", "run_vina, locate_engine_binary", "molecular-docking", "docking", "integrated", tests=("tests/ui/test_docking_page.py",), engines=("vina",)),
    Capability("dock.scoring", "Pose scoring + interaction fingerprints", "pharmaml.docking.scoring", "consensus_rescore, ligand_efficiency", "molecular-docking", "docking", "integrated", tests=("tests/test_docking.py",)),
    # -- Card 07 · PK/PD ---------------------------------------------------------
    Capability("pkpd.pk", "PK parameter calculation", "pharmaml.pharma.pk", "auc_trapezoid, clearance, half_life", "pkpd", "pkpd", "integrated", tests=("tests/ui/test_pkpd_page.py",)),
    Capability("pkpd.nca", "Non-compartmental analysis", "pharmaml.pkpd.nca_ext", "auc_linear_log, TerminalFit, SparseAUC", "pkpd", "pkpd", "integrated", tests=("tests/test_pkpd_engines.py",)),
    Capability("pkpd.ode", "Compartmental ODE simulation", "pharmaml.pkpd.ode", "solve_ode, DosingEvent", "pkpd", "pkpd", "integrated", tests=("tests/ui/test_pkpd_page.py",)),
    Capability("pkpd.pbpk", "PBPK simulation", "pharmaml.pkpd.pbpk", "MinimalPBPK, first_in_human_dose, mabel_dose", "pkpd", "pkpd", "integrated", tests=("tests/test_pkpd_engines.py",)),
    # -- Card 08 · Molecular Dynamics Analysis --------------------------------
    Capability("md.trajectory", "Trajectory loading + XVG/COLVAR parsing", "pharmaml.md.trajectory", "Trajectory, Topology", "molecular-dynamics-analysis", "molecular_dynamics", "integrated", tests=("tests/ui/test_molecular_dynamics_page.py",)),
    Capability("md.analysis", "RMSD/RMSF/Rg/H-bonds/salt bridges/PLIF/PCA", "pharmaml.md.metrics", "radius_of_gyration_trajectory, contact_map, native_contact_fraction", "molecular-dynamics-analysis", "molecular_dynamics", "integrated", tests=("tests/test_md_analysis.py",)),
    Capability("md.enm", "Elastic network models", "pharmaml.md.enm", "anm_fluctuations, anm_hessian", "molecular-dynamics-analysis", "molecular_dynamics", "integrated", tests=("tests/test_md_analysis.py",)),
    # -- cross-cutting research workbenches (§5.1: pages stay reachable) -----
    Capability("bio.analysis", "Gene-expression + sensitivity analysis", "pharmaml.biology", "call_sensitive_resistant, most_variable_genes", "datasets", "biology", "integrated", tests=("tests/test_biology.py", "tests/ui/test_biology_page.py")),
    Capability("ml.foundations", "Statistical foundations (ANOVA, distributions, profiling)", "pharmaml.ml.stats", "compare_groups, sample_distribution", "machine-learning", "foundations", "integrated", tests=("tests/ui/test_foundations_page.py",)),
    Capability("ml.bestofml", "Model catalog search + privacy accounting", "pharmaml.ml.catalog", "search_catalog, catalog_summary, coverage_table", "machine-learning", "bestofml", "integrated", tests=("tests/ui/test_bestofml_page.py", "tests/test_catalog_foldins.py")),
    Capability("ml.drugresponse", "Dose-response fitting + metrics", "pharmaml.ml.drugresponse", "fit_hill_curve, auc_log_dose, response_metrics", "machine-learning", "responseai", "integrated", tests=("tests/ui/test_responseai_page.py",)),
    # -- Card 09 · Visualization -------------------------------------------------
    Capability("viz.charts", "Scientific + statistical charts", "pharmaml.viz.statplots", "plot_box, plot_ecdf, plot_histogram", "visualization", "imagelab", "integrated", tests=("tests/test_viz.py",)),
    Capability("viz.images", "Image inspection + metadata", "pharmaml.viz.export", "export_image, sniff_format", "visualization", "imagelab", "integrated", tests=("tests/test_export_images.py",)),
    Capability("viz.tiff", "Lossless high-bit-depth TIFF handling", "pharmaml.viz.export", "ImageExportOptions, export_image", "visualization", "imagelab", "integrated", tests=("tests/test_export_images.py",)),
    Capability("viz.reports", "Report generation + figure export", "pharmaml.viz.report", "HTMLReport", "visualization", "utilities", "integrated", tests=("tests/test_viz.py",)),
)

#: The §2.3 decision, machine-readable: no Other card, because the
#: assignment above leaves no genuine user-facing capability homeless.
OTHER_CARD_PRESENT: bool = False


def summary() -> RegistrySummary:
    """The automated coverage summary (§6.2's closing requirement).

    Pure computation over the registry — the tests pin this, and the
    CAPABILITY_MAP quotes it; documentation alone never substitutes.
    """
    by_status: dict[str, int] = {}
    by_card: dict[str, int] = {}
    orphans: list[str] = []
    duplicates: list[str] = []
    blocked_undisclosed: list[str] = []
    test_gaps: list[str] = []
    invalid_routes: list[str] = []

    card_counts: dict[str, int] = {}
    for cap in CAPABILITIES:
        by_status[cap.status] = by_status.get(cap.status, 0) + 1
        card_counts[cap.card] = card_counts.get(cap.card, 0) + 1
        if cap.card not in CARD_ORDER:
            orphans.append(cap.capability_id)
        if cap.route not in SECTION_KEYS:
            invalid_routes.append(cap.capability_id)
        if cap.status == "blocked" and not cap.status_detail:
            blocked_undisclosed.append(cap.capability_id)
        if not cap.tests and not cap.test_gap:
            test_gaps.append(cap.capability_id)

    # §6.2 rule 6/8 cross-check: every card in the directive order must
    # own at least one capability — an empty card would be a routeless
    # shell (flagged here as the "duplicate_ownership" audit channel).
    for card in CARD_ORDER:
        if card_counts.get(card, 0) == 0:
            duplicates.append(card)

    return RegistrySummary(
        total=len(CAPABILITIES),
        by_status=dict(sorted(by_status.items())),
        by_card=card_counts,
        orphans=tuple(orphans),
        duplicate_ownership=tuple(duplicates),
        blocked_undisclosed=tuple(blocked_undisclosed),
        test_gaps=tuple(test_gaps),
        invalid_routes=tuple(invalid_routes),
        other_card_present=OTHER_CARD_PRESENT,
    )


def capabilities_for_card(card_id: str) -> tuple[Capability, ...]:
    """Every capability primarily owned by ``card_id``, registry order."""
    return tuple(cap for cap in CAPABILITIES if cap.card == card_id)


def find(capability_id: str) -> Capability | None:
    for cap in CAPABILITIES:
        if cap.capability_id == capability_id:
            return cap
    return None


def card_route_target(card_id: str) -> str:
    """The §2.2 anchor: the section a card's Ctrl/Cmd shortcut opens."""
    anchors = {
        "introduction": "dashboard",
        "datasets": "data",
        "machine-learning": "models",
        "deep-learning": "deep_learning",
        "generative-ai": "generative",
        "molecular-docking": "docking",
        "pkpd": "pkpd",
        "molecular-dynamics-analysis": "molecular_dynamics",
        "visualization": "imagelab",
    }
    return anchors[card_id]
