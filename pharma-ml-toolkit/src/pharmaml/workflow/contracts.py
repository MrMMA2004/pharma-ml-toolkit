"""The typed step contracts of the lightweight Scientific Workflow
Engine (part13 Phase 6, §8 — the round-322 API lane).

The §8 directive's own constraint governs this module: *do not
introduce a large workflow engine automatically; first assess whether
the existing JobManager and capability registry can support a
lightweight implementation; keep the application fully local and
offline-capable.*  The assessment lives in
``docs/ARCHITECTURE_REVIEW.md`` (Prefect / Knime / JupyterLab /
marimo reviewed; the decision: the in-tree layer over JobManager +
the capability vocabulary).  This package IS that layer — pure Python,
zero new dependencies, every step bound to a REAL backend callable
that already exists in this tree (no algorithm is reimplemented
here, the JobManager's own rule).

The contract vocabulary mirrors the capability registry's discipline
(:mod:`pharmaml.studio.capability_registry`): one frozen record per
user-facing step, typed input/output ports with a closed kind
vocabulary, and an explicit audit note — so a workflow node can be
validated against its step's contract BEFORE any execution begins
(§8's "typed input/output contracts" + "dependency validation").
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PORT_KINDS",
    "STEP_REGISTRY",
    "ParamSpec",
    "PortSpec",
    "StepSpec",
]

#: The closed port-kind vocabulary (§8's typed contracts).  Two ports
#: are compatible only when their kinds are EQUAL — the engine never
#: guesses an implicit conversion between artifact families.
PORT_KINDS: tuple[str, ...] = (
    "records",     # list[dict] — raw tabular rows (a dataset's import form)
    "dataframe",   # pandas DataFrame — the pipeline's working frame
    "split",       # the train/test quartet bundle from the split step
    "model",       # a FITTED sklearn-compatible estimator
    "metrics",     # dict[str, float] — an evaluation/summary answer
    "ad_report",   # dict — an applicability-domain verdict block
    "shap_summary",  # dict — the SHAP feature-attribution answer
    "report",      # dict — a generated report artifact record
)


@dataclass(frozen=True)
class PortSpec:
    """One typed input or output port of a step (§8's contracts).

    ``kind`` must be one of :data:`PORT_KINDS`; ``required`` only
    applies to inputs (an optional input port may be left unwired —
    the step then runs without that artifact and must tolerate it).
    """

    name: str
    kind: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class ParamSpec:
    """One tunable parameter of a step — surfaced by the registry
    routes so a caller knows the dial exists without reading the
    backend's source."""

    name: str
    default: object
    description: str = ""


@dataclass(frozen=True)
class StepSpec:
    """One user-facing workflow step: the typed contract + the REAL
    backend it delegates to (the runner lives in
    :mod:`pharmaml.workflow.runners` and imports that backend lazily
    — heavy stacks stay out of module import time)."""

    step_id: str
    title: str
    description: str
    backend: str
    workspace: str
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = ()
    parameters: tuple[ParamSpec, ...] = ()
    notes: str = ""


def _steps() -> dict[str, StepSpec]:
    """The step inventory (§8's example chain, one step per stage).

    Every ``backend`` names a module that EXISTS in this tree and is
    covered by its own test suite — the capability registry's rule 11
    (tests or an explicit test-gap record) holds here too: the
    engine-level lane is ``tests/test_workflow_engine.py`` + the
    route family's ``tests/api/test_workflow.py``.
    """
    registry: dict[str, StepSpec] = {}

    def add(spec: StepSpec) -> None:
        registry[spec.step_id] = spec

    add(StepSpec(
        step_id="dataset.import",
        title="Dataset Import",
        description=(
            "Build the working frame from inline records (list of "
            "dict rows, the bounded API form).  The local/offline "
            "CSV-path import is the CLI's own lane — the REST lane "
            "never reads server-side paths."
        ),
        backend="pandas.DataFrame (the toolkit's own frame layer)",
        workspace="datasets",
        inputs=(PortSpec("records", "records",
                         description="the raw tabular rows to import"),),
        outputs=(PortSpec("dataset", "dataframe",
                          description="the imported working frame"),),
        parameters=(ParamSpec("smiles_column", "smiles",
                              "the column holding the molecules"),
                    ParamSpec("target_column", "pIC50",
                              "the numeric target column (regression)")),
        notes=(
            "Honest scope: a request-borne records list, never a "
            "server-side file read; the honest local CSV lane is "
            "the pharmaml CLI."
        ),
    ))
    add(StepSpec(
        step_id="chem.standardize",
        title="Standardization",
        description=(
            "Standardize every SMILES through the backend's own "
            "per-row standardizer, carrying the per-row verdicts "
            "(valid/invalid counts) forward as an artifact."
        ),
        backend="pharmaml.chem.csv_standardize.standardize_smiles",
        workspace="datasets",
        inputs=(PortSpec("dataset", "dataframe",
                         description="the frame whose smiles column is "
                                     "standardized"),),
        outputs=(
            PortSpec("dataset", "dataframe",
                     description="the frame with standardized SMILES "
                                 "(invalid rows dropped)"),
            PortSpec("standardization", "metrics",
                     description="the per-row verdict counts"),
        ),
        parameters=(ParamSpec("smiles_column", "smiles",
                              "the column to standardize"),),
        notes=(
            "Rows that fail standardization are reported in the "
            "verdict artifact and dropped — never silently repaired."
        ),
    ))
    add(StepSpec(
        step_id="chem.descriptors",
        title="Descriptor Generation",
        description=(
            "Compute molecular descriptors over the standardized "
            "SMILES with the backend's own descriptor engine."
        ),
        backend="pharmaml.chem.descriptors.compute_descriptors",
        workspace="datasets",
        inputs=(PortSpec("dataset", "dataframe",
                         description="a frame with a SMILES column"),),
        outputs=(PortSpec("dataset", "dataframe",
                          description="the descriptor matrix (frame)"),),
        parameters=(ParamSpec("descriptor_set", "rdkit_all",
                               "rdkit_all | morgan | mordred"),
                    ParamSpec("smiles_column", "smiles",
                              "the column to featurize")),
        notes=(
            "Mordred's per-cell Missing/Error verdicts are NaN — "
            "the backend's own documented contract, untouched here."
        ),
    ))
    add(StepSpec(
        step_id="ml.split",
        title="Data Split",
        description=(
            "Split the descriptor frame into the train/test quartet "
            "with the strategy-aware splitter."
        ),
        backend="pharmaml.ml.preprocessing.make_train_test_split",
        workspace="machine-learning",
        inputs=(PortSpec("dataset", "dataframe",
                         description="the feature frame with the target "
                                     "column"),),
        outputs=(PortSpec("split", "split",
                          description="the train/test quartet bundle"),),
        parameters=(ParamSpec("target_column", "pIC50",
                               "the numeric target column"),
                    ParamSpec("test_size", 0.2,
                              "the held-out fraction"),
                    ParamSpec("random_state", 42,
                              "the reproducibility seed")),
        notes=(
            "The split runs BEFORE feature selection in the built-in "
            "template — the toolkit's own leakage rule (selection on "
            "training data only, the look.ipynb fix) — even though "
            "§8's illustrative order lists selection first."
        ),
    ))
    add(StepSpec(
        step_id="ml.feature_selection",
        title="Feature Selection",
        description=(
            "Select the union of per-method top-k columns, fitted on "
            "the TRAINING split only and applied to the test frame as "
            "a pure column subset."
        ),
        backend="pharmaml.ml.preprocessing.select_kbest_union",
        workspace="machine-learning",
        inputs=(PortSpec("split", "split",
                         description="the train/test quartet bundle"),),
        outputs=(PortSpec("split", "split",
                          description="the column-subset quartet "
                                      "bundle"),),
        parameters=(ParamSpec("k", 8,
                              "per-method top-k columns"),
                    ParamSpec("methods", ("f_regression",
                                          "mutual_info_regression"),
                              "the scoring function pair")),
        notes=(
            "The leakage-free contract is the backend's own: nothing "
            "here ever refits on the test frame."
        ),
    ))
    add(StepSpec(
        step_id="ml.train",
        title="Model Training",
        description=(
            "Instantiate a registry model and fit it on the training "
            "split."
        ),
        backend="pharmaml.ml.models.build_model",
        workspace="machine-learning",
        inputs=(PortSpec("split", "split",
                         description="the (selected) quartet bundle"),),
        outputs=(PortSpec("model", "model",
                          description="the fitted estimator"),),
        parameters=(ParamSpec("model", "random_forest",
                              "a MODEL_REGISTRY key"),
                    ParamSpec("task", "regression",
                              "classification | regression"),
                    ParamSpec("random_state", 42,
                              "the estimator's reproducibility seed")),
        notes=(
            "Unknown model names raise the backend's own typed "
            "ModelError — a partial-failure event in a workflow run, "
            "never a silent fallback."
        ),
    ))
    add(StepSpec(
        step_id="ml.evaluate",
        title="Model Evaluation",
        description=(
            "Evaluate the fitted model on the held-out split with the "
            "train-side mirror metrics."
        ),
        backend="pharmaml.ml.evaluation.evaluate_model",
        workspace="machine-learning",
        inputs=(
            PortSpec("split", "split",
                     description="the quartet bundle"),
            PortSpec("model", "model",
                     description="the fitted estimator"),
        ),
        outputs=(PortSpec("metrics", "metrics",
                          description="the held-out metric dict"),),
        parameters=(ParamSpec("task", "regression",
                              "the evaluation task"),),
        notes=(
            "TRAIN-side metrics are an overfit DIAGNOSTIC, never a "
            "quality claim — the backend's own caveat, carried "
            "verbatim."
        ),
    ))
    add(StepSpec(
        step_id="ml.applicability",
        title="Applicability Domain",
        description=(
            "The leverage/Williams applicability-domain readout over "
            "the fitted model's design matrix."
        ),
        backend="pharmaml.ml.applicability.leverage_williams",
        workspace="machine-learning",
        inputs=(
            PortSpec("split", "split",
                     description="the quartet bundle"),
            PortSpec("model", "model",
                     description="the fitted estimator"),
        ),
        outputs=(PortSpec("ad_report", "ad_report",
                          description="the Williams verdict block"),),
        parameters=(ParamSpec("residual_limit", 3.0,
                              "the standardized-residual band"),),
        notes=(
            "Regression-only geometry: the residual axis needs a "
            "numeric target — the backend's own guard, honoured here."
        ),
    ))
    add(StepSpec(
        step_id="ml.shap",
        title="SHAP",
        description=(
            "The tree-explainer SHAP summary over the fitted model's "
            "training rows."
        ),
        backend="pharmaml.ml.explain.shap_summary",
        workspace="machine-learning",
        inputs=(
            PortSpec("split", "split",
                     description="the quartet bundle"),
            PortSpec("model", "model",
                     description="the FITTED tree estimator"),
        ),
        outputs=(PortSpec("shap_summary", "shap_summary",
                          description="the per-feature mean |SHAP| "
                                      "answer"),),
        notes=(
            "The tree explainer refuses non-tree estimators with a "
            "typed error — an honest partial-failure event, never a "
            "silent kernel fallback."
        ),
    ))
    add(StepSpec(
        step_id="report.generate",
        title="Report Generation",
        description=(
            "Assemble the run's answers into a multi-page PDF report."
        ),
        backend="pharmaml.viz.report.save_pdf_report",
        workspace="visualization",
        inputs=(
            PortSpec("metrics", "metrics",
                     description="the evaluation metric dict"),
            PortSpec("ad_report", "ad_report",
                     description="the applicability-domain block",
                     required=False),
            PortSpec("shap_summary", "shap_summary",
                     description="the SHAP answer", required=False),
        ),
        outputs=(PortSpec("report", "report",
                          description="the report artifact record"),),
        parameters=(ParamSpec("title", "PharmaML workflow report",
                              "the report's title page"),),
        notes=(
            "The artifact records the generated file's path + "
            "sha-256 — the bytes stay on the local disk (offline "
            "guarantee), the digest rides the lineage."
        ),
    ))
    return registry


#: The single step registry (frozen specs, keyed by step_id).
STEP_REGISTRY: dict[str, StepSpec] = _steps()
