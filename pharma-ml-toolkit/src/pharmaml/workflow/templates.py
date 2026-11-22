"""The built-in workflow templates (part13 Phase 6, §8 — the
round-322 API lane; §8's "saved workflow templates").

The guided QSAR template transcribes §8's own example chain —
Dataset Import → Standardization → Descriptor Generation →
Feature Selection → Data Split → Model Training → Model Evaluation →
Applicability Domain → SHAP → Report Generation — with ONE honest
reorder: the split runs BEFORE feature selection, because the
toolkit's own leakage rule (selection fitted on training data only,
the look.ipynb fix) is a contract, not a preference.  The template's
note says so explicitly; nothing is silently rearranged.

Every template is a plain :class:`WorkflowSpec` — serialisable,
validatable, and executable by the same engine the GUI will drive
(the GUI lane rides the next continuation).
"""

from __future__ import annotations

from .engine import NodeSpec, WorkflowSpec

__all__ = ["TEMPLATE_REGISTRY", "list_templates"]


def _guided_qsar() -> WorkflowSpec:
    nodes: dict[str, NodeSpec] = {}
    nodes["import"] = NodeSpec(
        node_id="import",
        step_id="dataset.import",
        params={"smiles_column": "smiles", "target_column": "pIC50"},
    )
    nodes["standardize"] = NodeSpec(
        node_id="standardize",
        step_id="chem.standardize",
        params={"smiles_column": "smiles"},
        wiring={"dataset": {"node": "import", "port": "dataset"}},
    )
    nodes["descriptors"] = NodeSpec(
        node_id="descriptors",
        step_id="chem.descriptors",
        params={"descriptor_set": "rdkit_all", "smiles_column": "smiles"},
        wiring={"dataset": {"node": "standardize", "port": "dataset"}},
    )
    nodes["split"] = NodeSpec(
        node_id="split",
        step_id="ml.split",
        params={"target_column": "pIC50", "test_size": 0.25},
        wiring={"dataset": {"node": "descriptors", "port": "dataset"}},
    )
    nodes["selection"] = NodeSpec(
        node_id="selection",
        step_id="ml.feature_selection",
        params={"k": 8},
        wiring={"split": {"node": "split", "port": "split"}},
    )
    nodes["train"] = NodeSpec(
        node_id="train",
        step_id="ml.train",
        params={"model": "random_forest", "task": "regression"},
        wiring={"split": {"node": "selection", "port": "split"}},
    )
    nodes["evaluate"] = NodeSpec(
        node_id="evaluate",
        step_id="ml.evaluate",
        params={"task": "regression"},
        wiring={
            "split": {"node": "selection", "port": "split"},
            "model": {"node": "train", "port": "model"},
        },
    )
    nodes["applicability"] = NodeSpec(
        node_id="applicability",
        step_id="ml.applicability",
        params={"residual_limit": 3.0},
        wiring={
            "split": {"node": "selection", "port": "split"},
            "model": {"node": "train", "port": "model"},
        },
    )
    nodes["shap"] = NodeSpec(
        node_id="shap",
        step_id="ml.shap",
        wiring={
            "split": {"node": "selection", "port": "split"},
            "model": {"node": "train", "port": "model"},
        },
    )
    nodes["report"] = NodeSpec(
        node_id="report",
        step_id="report.generate",
        params={"title": "Guided QSAR workflow report"},
        wiring={
            "metrics": {"node": "evaluate", "port": "metrics"},
            "ad_report": {"node": "applicability", "port": "ad_report"},
            "shap_summary": {"node": "shap", "port": "shap_summary"},
        },
    )
    return WorkflowSpec(
        name="guided-qsar",
        description=(
            "part13 §8's own example chain, wired over the REAL "
            "backends: import, standardization, rdkit descriptors, "
            "the leakage-safe split-then-select arrangement, a "
            "random-forest regression fit with its held-out "
            "evaluation, the leverage/Williams applicability domain, "
            "the tree-explainer SHAP summary, and the multi-page PDF "
            "report."
        ),
        nodes=nodes,
        metadata={
            "lineage": "part13 §8 example chain (Dataset Import -> "
                        "Standardization -> Descriptor Generation -> "
                        "[Data Split before Feature Selection: the "
                        "toolkit's leakage rule] -> Model Training -> "
                        "Model Evaluation -> Applicability Domain -> "
                        "SHAP -> Report Generation)",
            "leakage_note": (
                "feature selection is fitted on the training split "
                "only and applied to the held-out frame as a pure "
                "column subset -- the look.ipynb fix, kept as the "
                "contract even though §8's illustrative order lists "
                "selection first"
            ),
        },
    )


def _minimal_regression() -> WorkflowSpec:
    nodes: dict[str, NodeSpec] = {}
    nodes["import"] = NodeSpec(
        node_id="import",
        step_id="dataset.import",
        params={"smiles_column": "smiles", "target_column": "pIC50"},
    )
    nodes["descriptors"] = NodeSpec(
        node_id="descriptors",
        step_id="chem.descriptors",
        params={"descriptor_set": "rdkit_all", "smiles_column": "smiles"},
        wiring={"dataset": {"node": "import", "port": "dataset"}},
    )
    nodes["split"] = NodeSpec(
        node_id="split",
        step_id="ml.split",
        params={"target_column": "pIC50", "test_size": 0.25},
        wiring={"dataset": {"node": "descriptors", "port": "dataset"}},
    )
    nodes["train"] = NodeSpec(
        node_id="train",
        step_id="ml.train",
        params={"model": "random_forest", "task": "regression"},
        wiring={"split": {"node": "split", "port": "split"}},
    )
    nodes["evaluate"] = NodeSpec(
        node_id="evaluate",
        step_id="ml.evaluate",
        params={"task": "regression"},
        wiring={
            "split": {"node": "split", "port": "split"},
            "model": {"node": "train", "port": "model"},
        },
    )
    nodes["report"] = NodeSpec(
        node_id="report",
        step_id="report.generate",
        params={"title": "Minimal regression workflow report"},
        wiring={"metrics": {"node": "evaluate", "port": "metrics"}},
    )
    return WorkflowSpec(
        name="minimal-regression",
        description=(
            "The six-node core: import, rdkit descriptors, split, "
            "random-forest regression fit, held-out evaluation, PDF "
            "report — the smallest honest chain a caller can rewire "
            "from (a split needs numeric features: the SMILES frame "
            "alone carries none)."
        ),
        nodes=nodes,
        metadata={
            "lineage": "the minimal core chain (import -> descriptors "
                        "-> split -> train -> evaluate -> report)",
        },
    )


#: The built-in template registry (§8's "saved workflow templates").
TEMPLATE_REGISTRY: dict[str, WorkflowSpec] = {
    "guided-qsar": _guided_qsar(),
    "minimal-regression": _minimal_regression(),
}


def list_templates() -> list[dict]:
    """The template index the registry route serves (each entry: the
    spec dict + the node/step counts + the honest notes)."""
    index: list[dict] = []
    for template_id, spec in TEMPLATE_REGISTRY.items():
        index.append({
            "template_id": template_id,
            "name": spec.name,
            "description": spec.description,
            "node_count": len(spec.nodes),
            "nodes": [
                {"node_id": n.node_id, "step_id": n.step_id}
                for n in spec.nodes.values()
            ],
            "metadata": spec.metadata,
            "spec": spec.to_dict(),
        })
    return index
