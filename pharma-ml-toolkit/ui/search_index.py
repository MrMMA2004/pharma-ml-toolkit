"""The global capability search / command palette (part6 §7, Phase 4).

A searchable feature directory over the ONE source of truth — the
pure-Python capability registry under ``pharmaml.studio`` — plus the
nine cards and the 18 Studio sections.  The user can find a capability
without knowing which card owns it (§7's eight example queries are
pinned by tests), see the parent card and destination section, and
navigate directly.

§7 honesty rules this module follows:

* aliases are scientific terminology that the backend really answers to
  (taken from the modules' own docstrings and the GUI's method lists),
  never invented keywords;
* unavailable optional dependencies are DISTINGUISHED from missing
  functionality — a capability whose optional dependency is not
  importable in THIS environment is still listed, with a
  "needs <dep>" tag instead of being hidden or dropped;
* no private implementation details or arbitrary internal class names
  are exposed: the palette shows the registry's public ``name`` and
  ``public_api`` strings, its card, and its destination section.

The search index construction is pure (no Qt) so it can be tested
without a QApplication; only :class:`SearchPalette` is a widget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import importlib.util

from pharmaml.studio.capability_registry import (
    CAPABILITIES,
    CARD_ORDER,
    SECTION_KEYS,
    capabilities_for_card,
    find,
)

#: The ``ui`` package is Qt-side, but THIS module imports nothing from
#: it — only the pure registry — so the index builds under a plain
#: ``python`` with no QApplication (the tests rely on that).#: Scientific-terminology aliases per capability id (§7 "search
#: scientific terminology / backend feature aliases").  Every entry is
#: terminology the backend's own documentation or the GUI already uses;
#: the probing test refuses rows that alias a capability to a term its
#: backend module never mentions.
ALIASES: dict[str, tuple[str, ...]] = {
    "data.fingerprints": ("morgan", "ecfp", "circular fingerprint"),
    "data.descriptors": ("mordred", "maccs", "physicochemical properties"),
    "data.standardize": ("canonicalize", "tautomer", "reinvent standardization"),
    "ml.conformal": ("conformal prediction", "prediction intervals",
                     "coverage guarantee", "uncertainty intervals"),
    "ml.conformal_cls": ("conformal prediction", "prediction sets",
                         "label-conditional", "coverage guarantee"),
    "ml.calibration": ("reliability", "brier", "ece"),
    "ml.drift": ("psi", "kolmogorov-smirnov", "distribution shift"),
    "ml.sar": ("mmp", "matched pairs", "activity cliffs", "rgroup"),
    "ml.subgroups": ("fairness", "stratified performance"),
    "dl.train": ("mlp", "neural network", "residual network"),
    "dl.geometric": ("gcn", "gat", "graph attention"),
    "dl.gnn": ("graph neural network", "gin", "graph conv"),
    "dl.gnnxai": ("subgraphx", "explainability", "fidelity"),
    "dl.pruning": ("sparsity", "prune"),
    "dl.quantization": ("int8", "nf4", "llm.int8"),
    "dl.harness": ("zero", "sharded", "offloading"),
    "gen.ga": ("genetic algorithm", "evolution", "crossover", "mutation"),
    "gen.rl": ("reinforcement learning", "reinvent", "policy gradient", "rl"),
    "gen.mcts": ("monte carlo tree search", "mcts", "tree search"),
    "gen.rnn": ("smiles rnn", "character-level", "sampling"),
    "gen.oracle": ("qed", "sa score", "synthetic accessibility", "logp"),
    "dock.vina": ("autodock vina", "docking", "pose", "binding mode"),
    "dock.scoring": ("rescoring", "ligand efficiency", "interaction fingerprint"),
    "pkpd.pk": ("auc", "clearance", "half-life", "pharmacokinetics"),
    "pkpd.nca": ("non-compartmental", "terminal fit", "sparse auc"),
    "pkpd.ode": ("compartmental", "ode", "dosing"),
    "pkpd.pbpk": ("physiologically based", "first in human", "mabel"),
    "md.analysis": ("rmsd", "rmsf", "radius of gyration", "hydrogen bonds",
                    "plif", "pca"),
    "viz.tiff": ("tiff", "16-bit", "high bit depth", "lossless"),
    "data.split_audit": ("leakage", "overlap", "split audit"),
    "ml.model_card": ("provenance", "model card", "documentation"),
    "ml.bench": ("scaffold split", "butina", "cold split", "benchmark"),
    "ml.tune": ("hyperparameter", "grid search", "random search"),
    "ml.ensemble": ("stacking", "voting", "bagging"),
}

#: import names for optional dependencies (registry ``optional_deps``
#: values).  A capability is NOT hidden when its dependency is absent —
#: §7 requires distinguishing the two cases in the result line.
_OPTIONAL_IMPORT_NAMES: dict[str, str] = {
    "rdkit": "rdkit",
    "torch": "torch",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "shap": "shap",
    "matplotlib": "matplotlib",
}


def dependency_available(name: str) -> bool:
    """Whether an optional dependency imports in THIS environment."""
    import_name = _OPTIONAL_IMPORT_NAMES.get(name, name)
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):  # a broken/ambiguous parent package
        return False


@dataclass(frozen=True)
class SearchRow:
    """One searchable destination (§7: label, parent card, route, honesty)."""

    kind: str            # "capability" | "card" | "section"
    label: str           # what the user sees first
    detail: str          # the parent card + destination (§7 requirement)
    route: str           # the Studio section key to navigate to
    card: str            # the owning card id ("" for sections)
    capability_id: str   # "" for card/section rows
    terms: tuple[str, ...] = field(default=())  # everything searchable
    needs_dep: str = ""  # non-empty => the optional dep is NOT installed

    def status_line(self) -> str:
        """The one-line result text (with the honest dependency tag)."""
        if self.needs_dep:
            return f"{self.label} — needs {self.needs_dep} (not installed)"
        return self.label


def _capability_terms(cap: Any) -> tuple[str, ...]:
    """Everything searchable for one registry row."""
    terms = [
        cap.capability_id,
        cap.name,
        cap.public_api.replace(", ", " "),
        cap.card,
        cap.route,
    ]
    backend_tail = cap.backend_module.rsplit(".", 1)[-1]
    terms.append(backend_tail)
    terms.extend(ALIASES.get(cap.capability_id, ()))
    return tuple(terms)


def _unavailable_dependency(cap: Any) -> str:
    """The first optional dependency of ``cap`` that is not importable."""
    for dep in cap.optional_deps:
        if not dependency_available(dep):
            return dep
    return ""


def build_search_index() -> tuple[SearchRow, ...]:
    """The full §7 directory: capabilities, cards, sections.

    Pure function — no Qt, no widget state; the tests call it directly.
    """
    rows: list[SearchRow] = []

    # the 18 Studio sections — direct navigation destinations
    for key in SECTION_KEYS:
        rows.append(
            SearchRow(
                kind="section",
                label=f"Go to {key.replace('_', ' ').title()}",
                detail="Studio section (direct navigation)",
                route=key,
                card="",
                capability_id="",
                terms=(key, key.replace("_", " ")),
            )
        )

    # the nine cards — the §2.1 home destinations
    for card_id in CARD_ORDER:
        caps = capabilities_for_card(card_id)
        rows.append(
            SearchRow(
                kind="card",
                label=f"Open the {card_id.replace('-', ' ').title()} card",
                detail=f"card workspace · {len(caps)} capabilities",
                route=_card_route(card_id),
                card=card_id,
                capability_id="",
                terms=(card_id, card_id.replace("-", " ")),
            )
        )

    # every capability — the §6 registry is the single source of truth
    for cap in CAPABILITIES:
        rows.append(
            SearchRow(
                kind="capability",
                label=cap.name,
                detail=(
                    f"{cap.card.replace('-', ' ').title()} card → "
                    f"{cap.route.replace('_', ' ').title()}"
                ),
                route=cap.route,
                card=cap.card,
                capability_id=cap.capability_id,
                terms=_capability_terms(cap),
                needs_dep=_unavailable_dependency(cap),
            )
        )
    return tuple(rows)


def _card_route(card_id: str) -> str:
    """The §2.2 anchor section for a card."""
    from pharmaml.studio.capability_registry import card_route_target

    return card_route_target(card_id)


def search(index: tuple[SearchRow, ...], query: str, limit: int = 12
           ) -> tuple[SearchRow, ...]:
    """Rank the index rows against ``query`` (§7's substring semantics).

    Scoring: exact label match > label startswith > term substring >
    detail substring.  Never returns more than ``limit`` rows; the
    palette shows the parent card and destination for every hit.
    """
    needle = query.strip().lower()
    if not needle:
        # the empty query surfaces the nine cards first, then sections —
        # a browsable launch pad rather than an empty list
        cards = [r for r in index if r.kind == "card"]
        sections = [r for r in index if r.kind == "section"]
        return tuple((cards + sections)[:limit])
    scored: list[tuple[int, SearchRow]] = []
    for row in index:
        label = row.label.lower()
        if label == needle:
            score = 0
        elif label.startswith(needle):
            score = 1
        elif any(needle in term.lower() for term in row.terms):
            score = 2
        elif needle in row.detail.lower():
            score = 3
        else:
            continue
        scored.append((score, row))
    scored.sort(key=lambda pair: (pair[0], pair[1].label))
    return tuple(row for _score, row in scored[:limit])
