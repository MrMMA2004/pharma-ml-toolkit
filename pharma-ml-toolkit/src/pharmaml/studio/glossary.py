"""Contextual scientific help (part6 §17 F, Phase 5).

A pure, Qt-free glossary of the scientific terms the Studio's controls
actually use — parameters, metrics, units, algorithms and assumptions
— with short, honest one-to-three-sentence explanations.  The pages
apply these via ``apply_help(widget, key)`` (tooltips), so a parameter
keeps ONE authoritative explanation wherever it appears, instead of
each page improvising its own.

§F's rule: "Use tooltips and contextual help rather than overwhelming
every page with long text."  Every entry here is therefore SHORT, and
says what the thing is, its unit when it has one, and the assumption
a user is implicitly making when they touch it.  No entry claims a
capability the backend does not have.

Terms are keyed by stable slugs; the GUI never shows these slugs, only
the explanations.  ``missing_keys()`` exists so tests can pin that a
page's requested keys all exist (a typo'd key would otherwise silently
produce a tooltip-less widget).
"""

from __future__ import annotations

from typing import Callable, Iterable

#: slug -> (short title, explanation with unit/assumption)
GLOSSARY: dict[str, tuple[str, str]] = {
    # -- general modeling -----------------------------------------------------
    "random_seed": (
        "Random seed",
        "The integer that fixes every random draw (splits, init, "
        "sampling). Same seed + same data + same versions ⇒ same "
        "result; leave it unset to let the run vary.",
    ),
    "n_jobs": (
        "Worker processes",
        "How many CPU workers the job may use. 1 = sequential; -1 = all "
        "cores. More workers trade RAM for wall-clock time.",
    ),
    "cv_folds": (
        "Cross-validation folds",
        "K in K-fold CV: the data is split into K parts, each part is "
        "held out once. More folds cost more time but a less noisy "
        "estimate; K must leave enough rows per fold.",
    ),
    "test_fraction": (
        "Test fraction",
        "Share of rows reserved for the final held-out evaluation, "
        "between 0 and 1 (0.2 = 20%). The test set is touched once, "
        "after model selection.",
    ),
    "metric_r2": (
        "R² (coefficient of determination)",
        "The share of target variance explained on the evaluated split; "
        "1.0 is perfect, 0.0 equals predicting the mean, negative is "
        "worse than the mean. Unit-free.",
    ),
    "metric_rmse": (
        "RMSE",
        "Root-mean-square error, in the TARGET's units (e.g. pIC50). "
        "Penalises large misses more than MAE does.",
    ),
    "metric_mae": (
        "MAE",
        "Mean absolute error, in the TARGET's units. Robust to a few "
        "large errors; read next to RMSE to gauge error spread.",
    ),
    "metric_auc": (
        "ROC AUC",
        "Probability a random active ranks above a random inactive; "
        "0.5 = chance, 1.0 = perfect ordering. Unit-free, "
        "threshold-independent.",
    ),
    "metric_mcc": (
        "MCC (Matthews correlation)",
        "Balanced classification score in [-1, 1] that uses the whole "
        "confusion matrix — more honest than accuracy on imbalanced "
        "sets.",
    ),
    # -- chemistry --------------------------------------------------------------
    "smiles": (
        "SMILES",
        "A molecule as a line notation string (e.g. ``CCO`` ethanol). "
        "Invalid strings are refused before any compute; ring closures "
        "and stereochemistry carry meaning.",
    ),
    "descriptors": (
        "Molecular descriptors",
        "Numeric features computed from the structure (MW, logP, TPSA, "
        "counts…). Units are per-descriptor; the computation reports "
        "them with the table.",
    ),
    "morgan_fingerprint": (
        "Morgan fingerprint (ECFP)",
        "A binary circular fingerprint: each bit records the presence "
        "of a local atom environment within 'radius' bonds. Radius 2 "
        "(2048 bits) is the community default.",
    ),
    "standardize": (
        "Molecular standardization",
        "Normalises salts, charges, tautomers and stereo so that the "
        "same molecule always yields the same canonical form BEFORE "
        "any descriptor or model touches it.",
    ),
    "qed": (
        "QED",
        "Quantitative Estimate of Drug-likeness in [0, 1] — an "
        "empirical desirability over eight property rules (Bickerton "
        "et al., 2012). Higher is more drug-like; it is a heuristic, "
        "not a verdict.",
    ),
    "sa_score": (
        "SA score",
        "Synthetic-accessibility score, roughly 1 (easy) to 10 (very "
        "hard), from the Ertl fragment-frequency model. A heuristic "
        "for chemists, not a synthesis plan.",
    ),
    "logp": (
        "logP",
        "Octanol/water partition coefficient, log10 units. The classic "
        "lipophilicity axis: ~1-3 is the usual oral-drug comfort zone.",
    ),
    "tpsa": (
        "TPSA",
        "Topological polar surface area in Å². Sum of polar-atom "
        "contributions; >140 Å² usually hurts passive permeability.",
    ),
    # -- PK/PD -----------------------------------------------------------------
    "auc_pk": (
        "AUC (exposure)",
        "Area under the concentration-time curve; amount × time / "
        "volume units (e.g. ng·h/mL). Total systemic exposure — the "
        "PK meaning, distinct from ROC AUC.",
    ),
    "clearance": (
        "Clearance (CL)",
        "Volume of plasma fully cleared per time, e.g. mL/min. Relates "
        "dose rate to steady-state exposure: rate = CL × concentration.",
    ),
    "half_life": (
        "Half-life (t½)",
        "Time for the concentration to halve, time units. Terminal "
        "t½ is fitted to the log-linear tail after the absorption "
        "phase.",
    ),
    "pbpk": (
        "PBPK",
        "Physiologically based PK: compartmental flow models over "
        "organs with real blood flows and partitioning. Predictions "
        "are only as good as the compound parameters fed in.",
    ),
    "mabel_dose": (
        "MABEL dose",
        "Minimum anticipated biological effect level — the lowest "
        "dose expected to show activity, used as a first-in-human "
        "safety anchor (10% of MABEL is a common FIH start).",
    ),
    # -- docking ------------------------------------------------------------------
    "vina_exhaustiveness": (
        "Exhaustiveness",
        "Vina's search effort (default 8). Higher explores more of "
        "the pose space at near-linear CPU cost; it does not change "
        "the scoring function itself.",
    ),
    "binding_mode": (
        "Binding pose",
        "A 3D ligand placement inside the receptor. Docking SCORES "
        "poses; it does not measure true affinity — treat ranks as "
        "hypotheses to test.",
    ),
    "ligand_efficiency": (
        "Ligand efficiency (LE)",
        "Binding score per heavy atom, ~kcal/mol per atom. Lets you "
        "compare ligands of very different size on one scale.",
    ),
    # -- MD --------------------------------------------------------------------------
    "rmsd": (
        "RMSD",
        "Root-mean-square deviation between structures, in Å after "
        "superposition (alignment matters: values are not comparable "
        "across different alignments).",
    ),
    "rmsf": (
        "RMSF",
        "Per-atom fluctuation around the average structure, in Å. "
        "High RMSF marks flexible regions (loops, termini); low RMSF "
        "marks stable core.",
    ),
    "radius_gyration": (
        "Radius of gyration (Rg)",
        "The system's mass spread around its centre, in Å. A growing "
        "Rg trend signals expansion/unfolding.",
    ),
    "plif": (
        "PLIF",
        "Protein-ligand interaction fingerprint: per-frame contacts "
        "(H-bonds, hydrophobic, ionic…) as a bit vector over "
        "interactions, not over atoms.",
    ),
    # -- deep learning -------------------------------------------------------------
    "epochs": (
        "Epochs",
        "Full passes over the training set. More epochs fit tighter "
        "until validation loss stops improving — that plateau, not "
        "the epoch count, is the stop signal.",
    ),
    "batch_size": (
        "Batch size",
        "Rows per gradient step. Bigger batches are gentler on the "
        "gradient noise but hungrier for RAM; the learning rate often "
        "needs rescaling with it.",
    ),
    "learning_rate": (
        "Learning rate",
        "Step size along the gradient. Too high diverges; too low "
        "crawls. If training is unstable, tune this FIRST.",
    ),
    "dropout": (
        "Dropout",
        "Fraction of activations zeroed at random each step "
        "(0-0.5 typical) to curb overfitting; disabled at inference "
        "automatically.",
    ),
    "weight_decay": (
        "Weight decay",
        "L2 pull on the weights per step. A small value (1e-5-1e-3) "
        "regularises without dominating the loss.",
    ),
    "pruning_sparsity": (
        "Pruning target sparsity",
        "Share of weights removed (0-1). Magnitude pruning zeroes the "
        "smallest ones; accuracy typically degrades gradually, then "
        "falls off a cliff.",
    ),
    "quantization_bits": (
        "Quantisation width",
        "Numeric precision of the stored weights (int8, nf4…). "
        "Narrower widths shrink memory and speed compute at some "
        "accuracy cost; the arithmetic here is exact, not emulated.",
    ),
    # -- generative -------------------------------------------------------------------
    "mcts_iterations": (
        "MCTS iterations",
        "Monte-Carlo tree search rollout budget. More iterations "
        "explore deeper SMILES space per candidate; cost grows "
        "roughly linearly.",
    ),
    "population_size": (
        "Population size",
        "Candidates kept per generation in the genetic search. Larger "
        "populations explore wider but cost more oracle calls per "
        "generation.",
    ),
    "novelty": (
        "Novelty",
        "Share of generated molecules NOT in the training corpus, "
        "matched by canonical form. It says nothing about "
        "plausibility — pair it with validity and SA.",
    ),
    "qed_band": (
        "Property band",
        "An accept window [low, high] a property must land in. Band "
        "objectives score 1 inside the window and fall off outside — "
        "softly, not a hard filter.",
    ),
    # -- data -------------------------------------------------------------------------
    "scaffold_split": (
        "Scaffold split",
        "Train/test split by Murcko scaffold so no scaffold (hence "
        "near-no chemotype) is seen in both. The honest choice for "
        "prospective-like evaluation.",
    ),
    "split_audit": (
        "Split-overlap audit",
        "Checks identifier overlap across splits (same compound "
        "leaking between train and test) — the quiet killer of "
        "over-optimistic scores.",
    ),
    "drift": (
        "Distribution drift",
        "A statistical comparison of a new batch against training "
        "distributions (PSI, KS…). A drift FLAG is a warning to "
        "re-validate, not a verdict that predictions are wrong.",
    ),
    "conformal": (
        "Conformal interval",
        "A prediction plus a calibrated range covering the truth with "
        "a chosen probability (e.g. 90%), valid under exchangeability "
        "of calibration and test data — the assumption to check.",
    ),
    "chunk_rows": (
        "Chunk rows",
        "Rows read per chunk in the streaming reader. Smaller chunks "
        "peak lower in RAM; larger chunks parse faster. Only affects "
        "resource use, never the parsed result.",
    ),
}


def explanation(key: str) -> tuple[str, str] | None:
    """The ``(title, text)`` for ``key``, or None when unknown."""
    return GLOSSARY.get(key)


def tooltip_text(key: str) -> str:
    """The ready tooltip: "<b>title</b> — text" (empty when unknown)."""
    entry = GLOSSARY.get(key)
    if entry is None:
        return ""
    title, text = entry
    return f"{title} — {text}"


def missing_keys(keys: Iterable[str]) -> tuple[str, ...]:
    """Which requested keys have no glossary entry (test surface)."""
    return tuple(k for k in keys if k not in GLOSSARY)


def apply_help(widget: object, key: str) -> bool:
    """Set ``widget``'s tooltip from the glossary (Qt duck-typed).

    Returns whether the key existed — pages can assert on this in
    tests, and a missing key is a bug to fix in the glossary, never a
    silent no-op in the UI.
    """
    text = tooltip_text(key)
    if not text:
        return False
    setter: Callable[[str], None] | None = getattr(widget, "setToolTip", None)
    if setter is None:
        return False
    setter(text)
    return True
