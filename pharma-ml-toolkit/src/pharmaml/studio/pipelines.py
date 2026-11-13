"""Sequential workflow pipelines + the §17A template library.

part6 §17B asks for a *reliable sequential workflow engine* first —
users chain stages such as::

    Dataset → Preprocessing → Descriptors → Training → Evaluation → Export

and the explicit instruction is NOT to build an excessively complex
drag-and-drop system before the core feature registry is stable.  This
module is that engine: a :class:`Pipeline` is an ordered list of
:class:`Stage` objects, each a thin, typed wrapper around ONE existing
public backend function (no scientific logic is re-implemented here —
§18 forbids rebuilding working backend modules).

§17A workflow templates are then nothing more than factories that
pre-declare a stage chain for a named scientific workflow (QSAR model
development, molecular descriptor generation, drug-response prediction,
molecular generation and screening, docking preparation and analysis,
PK/PD modeling, molecular-dynamics analysis).  Every stage of every
template calls an existing backend implementation.

The engine's honesty rules (§12/§13 spirit):

* A stage may declare :attr:`Stage.requires` — backend dependencies
  that must import before the stage runs.  A missing dependency fails
  the stage with the toolkit's actionable :class:`DependencyError`
  (message carries the exact pip command), never a silent empty.
* Artifacts (tables, figures, model summaries) are accumulated under
  caller-chosen keys; nothing is silently dropped and nothing is
  silently overwritten — a duplicate artifact key is an error.
* The run record (:class:`PipelineResult`) carries per-stage timing,
  inputs and outputs so the §17C comparison and §17D report surfaces
  can consume it later without re-running anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol
import json
import time

from pharmaml.exceptions import DependencyError

__all__ = [
    "Stage",
    "Pipeline",
    "PipelineResult",
    "StageOutcome",
    "run_pipeline",
    "TEMPLATES",
    "template_names",
    "build_template",
    "template_documentation",
]


# ---------------------------------------------------------------------------
# The engine (§17B)
# ---------------------------------------------------------------------------

class StageFn(Protocol):
    """A stage body: receives the artifact dict, returns new artifacts."""

    def __call__(self, artifacts: dict[str, Any]) -> dict[str, Any]: ...  # pragma: no cover


@dataclass(frozen=True)
class Stage:
    """One named step of a pipeline.

    ``fn`` must be a pure-with-respect-to-``artifacts`` callable: it
    READS the artifacts it needs by key and RETURNS a dict of the new
    artifacts it produces (which the engine merges in).  Returning a
    non-dict is a stage-author error and fails the run loudly.
    """

    name: str
    fn: StageFn
    description: str = ""
    requires: tuple[str, ...] = ()  # import names that must exist

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"stage name must be an identifier, got {self.name!r}")


@dataclass
class StageOutcome:
    """What one stage did (§20: exact evidence per step)."""

    stage: str
    ok: bool
    seconds: float
    produced: tuple[str, ...]
    error: str | None = None


@dataclass
class PipelineResult:
    """Everything a finished pipeline run knows about itself."""

    pipeline: str
    stages: list[StageOutcome] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    started: float = 0.0
    finished: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.stages) and all(s.ok for s in self.stages)

    @property
    def seconds(self) -> float:
        return self.finished - self.started

    def to_json(self) -> str:
        """A JSON-safe run record (artifacts are summarized, not dumped)."""
        return json.dumps(
            {
                "pipeline": self.pipeline,
                "ok": self.ok,
                "seconds": round(self.seconds, 3),
                "stages": [
                    {
                        "stage": s.stage,
                        "ok": s.ok,
                        "seconds": round(s.seconds, 3),
                        "produced": list(s.produced),
                        "error": s.error,
                    }
                    for s in self.stages
                ],
                "artifact_keys": sorted(self.artifacts),
            },
            indent=2,
        )


def _check_dependencies(stage: Stage) -> None:
    """Fail with the actionable message when a stage's import is absent."""
    import importlib.util

    for name in stage.requires:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            raise DependencyError(
                f"stage '{stage.name}' requires the optional dependency "
                f"'{name}' — pip install {name} "
                f"(or the matching pharmaml extra)"
            )


def run_pipeline(
    name: str,
    stages: list[Stage],
    artifacts: dict[str, Any] | None = None,
    on_stage: Callable[[StageOutcome], None] | None = None,
) -> PipelineResult:
    """Execute ``stages`` in order, threading artifacts between them.

    The first failing stage stops the run (sequential = fail-fast);
    earlier stage outcomes are preserved in the result so partial work
    is inspectable, and the exception surfaces unchanged to the caller.
    """
    result = PipelineResult(pipeline=name, artifacts=dict(artifacts or {}))
    result.started = time.perf_counter()
    for stage in stages:
        _check_dependencies(stage)
        t0 = time.perf_counter()
        try:
            produced = stage.fn(result.artifacts)
            if not isinstance(produced, dict):
                raise TypeError(
                    f"stage '{stage.name}' returned "
                    f"{type(produced).__name__}, expected dict of artifacts"
                )
            overlap = set(produced) & set(result.artifacts)
            if overlap:
                raise KeyError(
                    f"stage '{stage.name}' would overwrite artifact keys "
                    f"{sorted(overlap)} — pipelines never clobber"
                )
            outcome = StageOutcome(
                stage=stage.name,
                ok=True,
                seconds=time.perf_counter() - t0,
                produced=tuple(sorted(produced)),
            )
            result.artifacts.update(produced)
        except Exception as exc:  # noqa: BLE001 - record, re-raise
            outcome = StageOutcome(
                stage=stage.name,
                ok=False,
                seconds=time.perf_counter() - t0,
                produced=(),
                error=f"{type(exc).__name__}: {exc}",
            )
            result.stages.append(outcome)
            if on_stage is not None:
                on_stage(outcome)
            result.finished = time.perf_counter()
            raise
        result.stages.append(outcome)
        if on_stage is not None:
            on_stage(outcome)
    result.finished = time.perf_counter()
    return result


@dataclass(frozen=True)
class Pipeline:
    """A named, reusable chain of stages (what a §17A template builds)."""

    name: str
    stages: tuple[Stage, ...]
    description: str = ""
    card: str = ""  # owning card per the §4 mapping

    def run(
        self,
        artifacts: dict[str, Any] | None = None,
        on_stage: Callable[[StageOutcome], None] | None = None,
    ) -> PipelineResult:
        return run_pipeline(self.name, list(self.stages), artifacts, on_stage)


# ---------------------------------------------------------------------------
# The template library (§17A) — every stage calls an existing backend
# ---------------------------------------------------------------------------

def _stage_dataset(path: str | Path, target: str | None = None) -> Stage:
    """Dataset stage: load via the existing pharmaml.data.load_dataset.

    Stores BOTH the raw frame (descriptor stages need the SMILES
    column) and the numeric ``X`` / ``y`` the dataset object already
    derives (non-numeric columns excluded, visibly).
    """
    from pharmaml.data import load_dataset

    def fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        dataset = load_dataset(str(path), target=target)
        return {"dataset": dataset, "frame": dataset.frame,
                "target": dataset.target, "task": dataset.task,
                "X": dataset.X, "y": dataset.y}

    return Stage(
        name="dataset",
        fn=fn,
        description=f"load {Path(path).name} via pharmaml.data.load_dataset",
    )


def _stage_split(test_size: float = 0.25, seed: int = 42) -> Stage:
    """Preprocessing stage: the existing pharmaml.data.split_table."""
    from pharmaml.data import split_table

    def fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        frame = artifacts["frame"]
        target = artifacts["target"]
        split = split_table(frame=frame, target=target,
                            test_size=test_size, random_state=seed)

        def xy(which: str) -> tuple[Any, Any]:
            part = getattr(split, which)
            return part.drop(columns=[target]), part[target]

        X_train, y_train = xy("train")
        X_test, y_test = xy("test")
        # keep only the columns the dataset stage derived as numeric
        # features — a raw SMILES column must never reach an estimator
        keep = list(artifacts["X"].columns)
        X_train = X_train[keep]
        X_test = X_test[keep]
        return {"split": split, "X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test}

    return Stage(
        name="preprocessing",
        fn=fn,
        description=f"train/test split (test_size={test_size}, seed={seed}) "
                     "via pharmaml.data.split_table",
    )


def _stage_descriptors(smiles_col: str = "smiles", kind: str = "morgan") -> Stage:
    """Descriptors stage: the existing chem.descriptors.compute_* APIs.

    Reads the table as a plain SMILES column (not :func:`load_dataset`
    — a descriptor workflow has no numeric target to infer).
    """

    def fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        import pandas as pd
        from pharmaml.chem.descriptors import compute_fingerprints

        frame = artifacts["frame"]
        smiles = frame[smiles_col]
        X, _set_bits = compute_fingerprints(smiles, kind=kind)
        columns = [f"{kind}_{i:04d}" for i in range(X.shape[1])]
        return {"X": pd.DataFrame(X, columns=columns, index=frame.index)}

    return Stage(
        name="descriptors",
        fn=fn,
        description=f"{kind} fingerprints via "
                     "pharmaml.chem.descriptors.compute_fingerprints",
        requires=("rdkit",),
    )


def _stage_training(models: list[str] | None = None, cv: int = 5) -> Stage:
    """Training stage: the existing ml.training.compare_models leaderboard."""

    def fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        from pharmaml.ml.training import compare_models

        board = compare_models(artifacts["X_train"], artifacts["y_train"],
                               task=artifacts["task"], models=models, cv=cv)
        return {"leaderboard": board}

    return Stage(
        name="training",
        fn=fn,
        description=f"cross-validated leaderboard over "
                     f"{models or 'all available models'} (cv={cv}) via "
                     "pharmaml.ml.training.compare_models",
    )


def _stage_evaluation() -> Stage:
    """Evaluation stage: score the leaderboard winner on the held-out split."""

    def fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        from pharmaml.ml.models import build_model

        board = artifacts["leaderboard"]
        best = str(board.iloc[0]["model"])
        model = build_model(best, task=artifacts["task"])
        model.fit(artifacts["X_train"], artifacts["y_train"])
        pred = model.predict(artifacts["X_test"])
        if artifacts["task"] == "classification":
            from sklearn.metrics import balanced_accuracy_score

            metric_name, value = "balanced_accuracy", balanced_accuracy_score(
                artifacts["y_test"], pred)
        else:
            from sklearn.metrics import r2_score

            metric_name, value = "r2", float(np.asarray(
                r2_score(artifacts["y_test"], pred)))
        return {"best_model": best, "metric_name": metric_name,
                "metric_value": float(value),
                "heldout_n": int(len(artifacts["y_test"]))}

    return Stage(
        name="evaluation",
        fn=fn,
        description="fit the leaderboard winner, score on the held-out "
                    "split (balanced accuracy / R²)",
    )


def _stage_export(out_dir: str | Path) -> Stage:
    """Export stage: the existing viz.report.export_table."""

    def fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        from pharmaml.viz.report import export_table

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for key in ("leaderboard",):
            if key in artifacts:
                p = export_table(artifacts[key], out / f"{key}.csv")
                paths[key] = str(p)
        return {"exports": paths}

    return Stage(
        name="export",
        fn=fn,
        description="write the leaderboard CSV via "
                     "pharmaml.viz.report.export_table",
    )


def _qsar_template(out_dir: str | Path, data_path: str | Path,
                   target: str | None, models: list[str] | None,
                   cv: int) -> Pipeline:
    return Pipeline(
        name="qsar_model_development",
        description="QSAR model development: dataset → split → leaderboard "
                    "→ held-out evaluation → export",
        card="Machine Learning",
        stages=(
            _stage_dataset(data_path, target),
            _stage_split(),
            _stage_training(models, cv),
            _stage_evaluation(),
            _stage_export(out_dir),
        ),
    )


def _descriptors_template(out_dir: str | Path, data_path: str | Path,
                          smiles_col: str, kind: str) -> Pipeline:
    def dataset_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        import pandas as pd

        frame = pd.read_csv(data_path)
        if smiles_col not in frame.columns:
            raise ValueError(
                f"SMILES column {smiles_col!r} not in "
                f"{Path(data_path).name} — columns: "
                f"{', '.join(frame.columns)}"
            )
        return {"frame": frame}

    def eval_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        X = artifacts["X"]
        return {"n_molecules": int(X.shape[0]),
                "n_features": int(X.shape[1])}

    return Pipeline(
        name="molecular_descriptor_generation",
        description="Molecular descriptor generation: dataset → "
                    f"{kind} descriptors → inventory",
        card="Chemistry",
        stages=(
            Stage(name="dataset", fn=dataset_fn,
                  description=f"read the SMILES table {Path(data_path).name}"),
            Stage(name="descriptors",
                  fn=_stage_descriptors(smiles_col, kind).fn,
                  description=f"{kind} descriptors via "
                              "pharmaml.chem.descriptors.compute_fingerprints",
                  requires=("rdkit",)),
            Stage(name="inventory", fn=eval_fn,
                  description="count molecules and features produced"),
            _stage_export(out_dir),
        ),
    )


def _response_template(out_dir: str | Path, screen_path: str | Path,
                       expression_path: str | Path, compound: str | None,
                       models: list[str] | None, cv: int,
                       n_top_genes: int = 40) -> Pipeline:
    """Drug-response prediction — the notebook's proven chain:

    response screen → normalized expression features (top-variable
    genes) → per-cell-line join → cross-validated leaderboard →
    held-out evaluation → export.  When ``compound`` is ``None`` the
    first compound of the screen is used (stated in the run record).
    """

    def screen_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        from pharmaml.biology.response import load_response_table

        screen = load_response_table(screen_path)
        chosen = compound or screen.compounds[0]
        return {"screen": screen, "compound": chosen}

    def features_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        import pandas as pd
        from pharmaml.biology.omics import (
            most_variable_genes,
            normalize_expression,
        )
        from pharmaml.biology.response import (
            collapse_replicates,
            response_matrix,
        )

        screen = collapse_replicates(artifacts["screen"])
        expression = pd.read_csv(expression_path, index_col="cell_line")
        expr_norm = normalize_expression(expression)
        top_genes = most_variable_genes(expr_norm, n=n_top_genes)
        metric = screen.metrics[0]
        matrix = response_matrix(screen, metric)
        y = matrix[artifacts["compound"]].dropna()
        X = expr_norm.loc[y.index, top_genes]
        return {"X": X, "y": y}

    def split_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        from pharmaml.ml import make_train_test_split

        Xtr, Xte, ytr, yte = make_train_test_split(
            artifacts["X"], artifacts["y"], test_size=0.25, random_state=42)
        return {"X_train": Xtr, "X_test": Xte,
                "y_train": ytr, "y_test": yte}

    def train_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        from pharmaml.ml.training import compare_models

        board = compare_models(artifacts["X_train"], artifacts["y_train"],
                               task="regression", models=models, cv=cv)
        return {"leaderboard": board}

    def eval_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        from sklearn.metrics import r2_score

        from pharmaml.ml.models import build_model

        board = artifacts["leaderboard"]
        best = str(board.iloc[0]["model"])
        model = build_model(best, task="regression")
        model.fit(artifacts["X_train"], artifacts["y_train"])
        pred = model.predict(artifacts["X_test"])
        return {"best_model": best, "metric_name": "r2",
                "metric_value": float(np.asarray(
                    r2_score(artifacts["y_test"], pred))),
                "heldout_n": int(len(artifacts["y_test"]))}

    return Pipeline(
        name="drug_response_prediction",
        description="Drug-response prediction: response screen + expression "
                    "features → cross-validated leaderboard → held-out "
                    "evaluation → export",
        card="Biology",
        stages=(
            Stage(name="dataset", fn=screen_fn,
                  description="load the response screen via "
                              "pharmaml.biology.response.load_response_table"),
            Stage(name="features", fn=features_fn,
                  description="normalize expression, pick top-variable "
                              "genes, join per cell line"),
            Stage(name="preprocessing", fn=split_fn,
                  description="train/test split via pharmaml.ml."
                              "make_train_test_split"),
            Stage(name="training", fn=train_fn,
                  description=f"cross-validated leaderboard over "
                              f"{models or 'all available models'} "
                              f"(cv={cv})"),
            Stage(name="evaluation", fn=eval_fn,
                  description="fit the winner, score R² on the held-out "
                              "split"),
            _stage_export(out_dir),
        ),
    )


def _generative_template(out_dir: str | Path, objective: str,
                         generations: int, seed: int) -> Pipeline:
    """Molecular generation and screening: GA optimize → screen export.

    ``objective`` names one of the bundled scoring objectives (the
    recipe registry the CLI already exposes): ``'qed'``, ``'sa'``,
    ``'logp_band'``, ``'mw_band'``, ``'tpsa_band'``.
    """
    _OBJECTIVES = {
        "qed": ("pharmaml.generative.scoring", "QedObjective"),
        "sa": ("pharmaml.generative.scoring", "SAScoreObjective"),
        "logp_band": ("pharmaml.generative.scoring", "LogPBandObjective"),
        "mw_band": ("pharmaml.generative.scoring", "MolecularWeightBandObjective"),
        "tpsa_band": ("pharmaml.generative.scoring", "TpsaBandObjective"),
    }

    def gen_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        import importlib

        from pharmaml.generative.genetic import run_genetic_optimization

        if objective not in _OBJECTIVES:
            raise ValueError(
                f"unknown objective {objective!r} — known: "
                f"{', '.join(sorted(_OBJECTIVES))}"
            )
        module_name, class_name = _OBJECTIVES[objective]
        obj_cls = getattr(importlib.import_module(module_name), class_name)
        result = run_genetic_optimization(
            obj_cls(), generations=generations, seed=seed)
        return {"generation_result": result,
                "smiles": list(result.smiles),
                "scores": list(result.scores)}

    def screen_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        import pandas as pd

        board = pd.DataFrame({"smiles": artifacts["smiles"],
                              "score": artifacts["scores"]})
        return {"leaderboard": board.sort_values(
            "score", ascending=False).reset_index(drop=True)}

    return Pipeline(
        name="molecular_generation_screening",
        description=f"Molecular generation and screening: GA optimization "
                    f"({objective}) → scored leaderboard",
        card="Molecule Generation",
        stages=(
            Stage(name="generation", fn=gen_fn,
                  description=f"genetic optimization for the {objective} "
                              "objective via "
                              "pharmaml.generative.genetic.run_genetic_optimization",
                  requires=("rdkit",)),
            Stage(name="screening", fn=screen_fn,
                  description="rank the generated molecules by score"),
            _stage_export(out_dir),
        ),
    )


def _docking_template(out_dir: str | Path, ligands_path: str | Path,
                      receptor_pdbqt: str | Path,
                      smiles_col: str | None = None) -> Pipeline:
    """Docking preparation and analysis: CSV ligands → PDBQT campaign
    tree → receptor grid box."""

    def lig_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        from pharmaml.docking.ligands import prepare_ligand_library

        out = Path(out_dir) / "ligands"
        report = prepare_ligand_library(str(ligands_path), out,
                                        smiles_col=smiles_col)
        return {"preparation_report": report,
                "n_prepared": report.n_prepared,
                "ligand_root": str(out)}

    def grid_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        from pharmaml.docking.preparation import compute_grid_box

        box = compute_grid_box(str(receptor_pdbqt))
        return {"grid_box": box}

    return Pipeline(
        name="docking_preparation_analysis",
        description="Docking preparation and analysis: SMILES CSV → PDBQT "
                    "campaign tree → receptor grid box",
        card="Docking",
        stages=(
            Stage(name="ligand_preparation", fn=lig_fn,
                  description="prepare the ligand library via "
                              "pharmaml.docking.ligands.prepare_ligand_library",
                  requires=("rdkit", "meeko")),
            Stage(name="receptor_preparation", fn=grid_fn,
                  description="compute the receptor grid box via "
                              "pharmaml.docking.preparation.compute_grid_box"),
            _stage_export(out_dir),
        ),
    )


def _pkpd_template(out_dir: str | Path, ka: float, ke: float, vd: float,
                   dose: float, hours: float = 24.0) -> Pipeline:
    """PK/PD modeling: one-compartment oral PK via the RK4 simulator."""
    import numpy as np

    def model_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        from pharmaml.pharma.compartment import simulate_compartment

        t = np.linspace(0, hours, int(4 * hours) + 1)
        result = simulate_compartment(t, dose, model="one", route="oral",
                                      ka=ka, ke=ke, vd=vd)
        return {"pk_result": result,
                "time_hours": result.times,
                "concentration": result.concentrations}

    def param_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        from pharmaml.pharma.pk import auc_trapezoid, half_life, oral_cmax, oral_tmax

        result = artifacts["pk_result"]
        return {"tmax_h": float(oral_tmax(ka, ke)),
                "cmax": float(oral_cmax(dose, ka, ke, vd)),
                "half_life_h": float(half_life(ke)),
                "auc": float(auc_trapezoid(result.times,
                                           result.concentrations))}

    def export_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        import pandas as pd

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        curve = pd.DataFrame({"time_h": artifacts["time_hours"],
                              "concentration": artifacts["concentration"]})
        path = out / "pk_curve.csv"
        curve.to_csv(path, index=False)
        return {"exports": {"pk_curve": str(path)}}

    return Pipeline(
        name="pk_pd_modeling",
        description=f"PK/PD modeling: one-compartment oral PK "
                    f"(ka={ka}, ke={ke}, Vd={vd}, dose={dose})",
        card="PK/PD",
        stages=(
            Stage(name="modeling", fn=model_fn,
                  description="concentration-time curve via "
                              "pharmaml.pharma.compartment.simulate_compartment"),
            Stage(name="parameters", fn=param_fn,
                  description="Tmax / Cmax / half-life / AUC via "
                              "pharmaml.pharma.pk"),
            Stage(name="export", fn=export_fn,
                  description="write the curve CSV"),
        ),
    )


def _md_template(out_dir: str | Path, n_frames: int = 80, seed: int = 0,
                 flexible_residues: tuple[int, ...] = (4, 5, 6),
                 noise: float = 0.25, drift_radians: float = 0.05) -> Pipeline:
    """MD analysis: trajectory → RMSD series → export.

    The toolkit's MD layer analyzes :class:`~pharmaml.md.trajectory
    .Trajectory` objects (array-backed; the synthetic factory is the
    bundled demo path, exactly what the MD page's own examples use) —
    there is no file loader to pretend to call, so the template states
    plainly that it builds the trajectory via ``synthetic_trajectory``
    unless the caller passes one in through the initial artifacts.
    """

    def load_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        from pharmaml.md.trajectory import Trajectory, synthetic_trajectory

        traj = artifacts.get("trajectory") or synthetic_trajectory(
            n_frames=n_frames, seed=seed, flexible_residues=flexible_residues,
            noise=noise, drift_radians=drift_radians)
        if not isinstance(traj, Trajectory):
            raise TypeError(
                "md_analysis expects a pharmaml.md.trajectory.Trajectory "
                "in the 'trajectory' artifact (or none, to synthesize one)"
            )
        return {"trajectory": traj}

    def metrics_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        from pharmaml.md.metrics import rmsd_trajectory

        series = rmsd_trajectory(artifacts["trajectory"])
        return {"rmsd": series}

    def export_fn(artifacts: dict[str, Any]) -> dict[str, Any]:
        import pandas as pd

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({"rmsd_angstrom": artifacts["rmsd"]})
        path = out / "rmsd.csv"
        frame.to_csv(path, index=False)
        return {"exports": {"rmsd": str(path)}}

    return Pipeline(
        name="md_analysis",
        description="Molecular-dynamics analysis: trajectory → per-frame "
                    "RMSD series → export",
        card="Molecular Dynamics",
        stages=(
            Stage(name="loading", fn=load_fn,
                  description="bind or synthesize the trajectory via "
                              "pharmaml.md.trajectory.synthetic_trajectory"),
            Stage(name="analysis", fn=metrics_fn,
                  description="per-frame RMSD via "
                              "pharmaml.md.metrics.rmsd_trajectory"),
            Stage(name="export", fn=export_fn,
                  description="write the RMSD CSV"),
        ),
    )


#: The §17A library: template name -> (factory, owning card).
TEMPLATES: dict[str, dict[str, Any]] = {
    "qsar_model_development": {
        "factory": _qsar_template,
        "card": "Machine Learning",
        "summary": "Dataset → split → cross-validated model leaderboard → "
                   "held-out evaluation → CSV export",
    },
    "molecular_descriptor_generation": {
        "factory": _descriptors_template,
        "card": "Chemistry",
        "summary": "Dataset → fingerprints/descriptors → inventory → export",
    },
    "drug_response_prediction": {
        "factory": _response_template,
        "card": "Biology",
        "summary": "Response dataset → split → leaderboard → evaluation → "
                   "export",
    },
    "molecular_generation_screening": {
        "factory": _generative_template,
        "card": "Molecule Generation",
        "summary": "GA optimization against a scoring objective → scored "
                   "leaderboard → export",
    },
    "docking_preparation_analysis": {
        "factory": _docking_template,
        "card": "Docking",
        "summary": "SMILES ligands → PDBQT preparation → receptor grid",
    },
    "pk_pd_modeling": {
        "factory": _pkpd_template,
        "card": "PK/PD",
        "summary": "One-compartment oral PK curve, Tmax/Cmax/half-life, "
                   "curve export",
    },
    "md_analysis": {
        "factory": _md_template,
        "card": "Molecular Dynamics",
        "summary": "Trajectory load → RMSD series → export",
    },
}


def template_names() -> list[str]:
    """The seven §17A template names, in declaration order."""
    return list(TEMPLATES)


def build_template(name: str, **kwargs: Any) -> Pipeline:
    """Instantiate one of the §17A workflow templates.

    Unknown names raise ``ModelError``-style loud errors; every factory
    argument is a keyword so the UI can map form fields directly.
    """
    try:
        entry = TEMPLATES[name]
    except KeyError:
        known = ", ".join(sorted(TEMPLATES))
        raise ValueError(
            f"unknown workflow template {name!r} — known templates: {known}"
        ) from None
    return entry["factory"](**kwargs)


def template_documentation() -> str:
    """Human-readable template index (the UI renders this verbatim)."""
    lines = ["§17A workflow templates (each stage calls an existing "
             "backend implementation):", ""]
    for name, entry in TEMPLATES.items():
        lines.append(f"- {name} [{entry['card']}]: {entry['summary']}")
    return "\n".join(lines)
