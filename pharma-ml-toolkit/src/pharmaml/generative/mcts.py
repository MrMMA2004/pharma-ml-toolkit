"""Monte Carlo tree search molecule generation -- the ChemTSv2 model.

ChemTSv2 (Ishida et al., *WIREs Comput. Mol. Sci.* 2023,
``molecule-generator-collection/ChemTSv2``, MIT) frames de novo generation
as a one-player game: the *moves* are SMILES characters, the *policy* is a
language model's next-character distribution, and the *reward* is whatever
"good molecule" means to you.  A Monte Carlo tree search then spends its
budget where the policy and the reward disagree -- exploring prefixes the
prior finds unlikely but the objective rewards, and exploiting the
prefixes both agree on.  This module is that programming model ported to
the toolkit's stack: the policy is :class:`~pharmaml.generative.SmilesRnn`
(or any callable you inject), the reward is any
:class:`~pharmaml.generative.ScoringObjective`, and the search runs in
pure Python on a laptop CPU.

The port, mechanism by mechanism (ChemTSv2 → here):

* **Tree policy** -- UCB1 exactly as ChemTSv2 v2 computes it,
  ``Q/N + c_val * sqrt(2 * ln(N_parent) / N)``, ties broken randomly; or
  ``"puct"``, ChemTS **v1**'s variant ``Q/N + puct_c * prior *
  sqrt(N_parent) / (1 + N)`` where ``prior`` is the policy's probability
  of the child's leading character (v2 replaced it with the expansion
  threshold; both are selectable here because the policy surface makes
  the difference measurable);
* **Expansion** -- ChemTSv2's cumulative-probability rule: children are
  the policy's top characters until their probabilities sum to
  ``expansion_threshold`` (default ``0.995``), never fewer than two --
  an adaptive branching factor (confident policy → narrow, uncertain
  policy → wide) instead of a fixed top-k;
* **Simulation** -- ``simulation_num`` rollouts per expanded child per
  cycle, each sampling characters from the policy until the end token or
  ``max_len`` (the truncation-evaluates-anyway rule included);
* **Evaluation** -- the completed string is canonicalised through RDKit;
  unparseable rollouts score ``0.0`` (the toolkit's [0, 1] objective
  convention -- ChemTSv2's ``-1`` invalid penalty mapped onto the same
  scale), and a **dedup cache** scores every distinct canonical molecule
  once no matter how often the tree revisits it (ChemTSv2's
  ``generated_dict``);
* **Budget** -- ``iterations`` search cycles (a deterministic count, so
  seeded runs are byte-reproducible, unlike ChemTSv2's wall-clock
  ``hours``), each cycle = select → expand → simulate → backpropagate,
  with one ``history`` record per cycle for the introspection plot;
* **Extension mode** -- ``optimize(prefix=...)`` roots the tree at a
  chosen partial SMILES (ChemTSv2's ``--input_smiles``): grow
  ''something like aspirin's ester head'' by searching from
  ``"CC(=O)O"``.

Example
-------
>>> from pharmaml.generative import MctsOptimizer, QedObjective
>>> def flat_policy(prefix):        # a stand-in for the RNN prior
...     return [("C", 0.4), ("N", 0.2), ("O", 0.2), ("<end>", 0.2)]
>>> result = MctsOptimizer(
...     QedObjective(), policy=flat_policy, iterations=6, seed=42).optimize()
>>> result.metadata["method"]
'mcts'
>>> len(result.smiles) > 0 and all(0.0 <= s <= 1.0 for s in result.scores)
True

Dependencies: RDKit for the objective (as everywhere in the generative
subpackage); torch only when the default RNN policy is used -- an
injected ``policy`` callable needs neither.  Both are imported lazily.

References
----------
.. [Ishida2023] Ishida, Kayano, Yamauchi, "De novo molecule generation
   combining Monte Carlo tree search with recurrent neural networks",
   ChemTSv2, WIREs Comput. Mol. Sci. 13:e1680 (2023).
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from typing import Any

from pharmaml.exceptions import DataValidationError
from pharmaml.generative._utils import canonical_smiles
from pharmaml.generative.result import GenerationResult
from pharmaml.generative.rnn import END as _RNN_END
from pharmaml.generative.rnn import SmilesRnn
from pharmaml.logger import get_logger

__all__ = ["MctsOptimizer", "run_mcts_optimization"]

_LOG = get_logger(__name__)

#: A policy maps a SMILES prefix to the next-character distribution:
#: ``(character, probability)`` pairs sorted by probability descending,
#: the end token included as an ordinary move.  The RNN default is
#: :meth:`SmilesRnn.next_token_distribution`.
_Policy = Callable[[str], list[tuple[str, float]]]


class _Node:
    """One MCTS node: a SMILES prefix plus its search statistics.

    Package-private by design -- the optimiser's callers see only
    :class:`GenerationResult`.
    """

    __slots__ = (
        "children",
        "is_terminal",
        "parent",
        "position",
        "prior",
        "total_value",
        "visits",
    )

    def __init__(
        self,
        position: str,
        parent: _Node | None = None,
        prior: float = 1.0,
        is_terminal: bool = False,
    ) -> None:
        self.position = position
        self.parent = parent
        self.children: list[_Node] = []
        self.visits = 0
        self.total_value = 0.0
        self.prior = prior
        self.is_terminal = is_terminal


class MctsOptimizer:
    """ChemTSv2-style MCTS over SMILES prefixes, guided by a language model.

    Parameters
    ----------
    objective:
        What a good molecule means -- any
        :class:`~pharmaml.generative.ScoringObjective` (single or
        :class:`~pharmaml.generative.MultiObjective`; scores in ``[0, 1]``,
        unparseable molecules ``0.0``).
    policy:
        The move generator: a callable mapping a SMILES prefix to the
        next-character distribution (``(character, probability)`` pairs,
        probability descending, the end token included).  ``None`` builds
        the default -- a :class:`~pharmaml.generative.SmilesRnn` prior
        trained on ``corpus`` for ``prior_epochs`` epochs (requires
        torch; an injected callable needs neither torch nor training
        time, which is what the tests exploit).
    corpus:
        Training corpus for the default policy (default:
        :data:`~pharmaml.generative.DEFAULT_CORPUS`); ignored when
        ``policy`` is injected.  Must be non-empty when given.
    iterations:
        Search cycles (each = select → expand → simulate →
        backpropagate).  A deterministic budget: two identically-seeded
        runs explore identical trees.
    c_val:
        UCB1 exploration constant (ChemTSv2's ``c_val``; 1.0 →
        exploration, 0.1 → exploitation, per their README).
    tree_policy:
        ``"uct"`` (ChemTSv2 v2's UCB1) or ``"puct"`` (v1's
        prior-weighted variant -- the policy's probability enters the
        tree policy itself).
    puct_c:
        The PUCT exploration strength (v1's ``RC`` analogue; only used
        when ``tree_policy="puct"``).
    expansion_threshold:
        Cumulative-probability cut for child expansion (ChemTSv2's
        ``expansion_threshold``, default 0.995) -- children are the
        policy's top characters until their probabilities sum this high,
        never fewer than two.
    simulation_num:
        Rollouts per expanded child per cycle (ChemTSv2's
        ``simulation_num``, default 3).
    max_len:
        Hard character cap per molecule (ChemTSv2 reads this from the
        RNN's training length; here it is explicit, default 80).
    seed:
        Seed for the local ``random.Random`` (tie-breaking and rollout
        sampling); the process-global RNG is never touched.
    prior_epochs / prior_hidden_dim:
        Default-policy training knobs (ignored when ``policy`` is
        injected).

    Raises
    ------
    DataValidationError
        On an unknown ``tree_policy``, a non-callable ``policy``, or
        out-of-range numeric parameters.

    Example
    -------
    >>> from pharmaml.generative import MctsOptimizer, QedObjective
    >>> def flat_policy(prefix):
    ...     return [("C", 0.4), ("N", 0.2), ("O", 0.2), ("<end>", 0.2)]
    >>> mcts = MctsOptimizer(QedObjective(), policy=flat_policy,
    ...                      iterations=4, seed=0)
    >>> result = mcts.optimize(prefix="CC")     # extension mode
    >>> result.metadata["prefix"]
    'CC'
    >>> MctsOptimizer(QedObjective(), iterations=0)
    Traceback (most recent call last):
        ...
    pharmaml.exceptions.DataValidationError: iterations must be >= 1, got 0
    """

    def __init__(
        self,
        objective: Any,
        *,
        policy: _Policy | None = None,
        corpus: Sequence[str] | None = None,
        iterations: int = 16,
        c_val: float = 1.0,
        tree_policy: str = "uct",
        puct_c: float = 2.5,
        expansion_threshold: float = 0.995,
        simulation_num: int = 3,
        max_len: int = 80,
        seed: int = 42,
        prior_epochs: int = 8,
        prior_hidden_dim: int = 96,
    ) -> None:
        if not hasattr(objective, "score") or not callable(objective.score):
            raise DataValidationError(
                "objective must expose a callable score() method "
                f"(got {objective!r})"
            )
        if policy is not None and not callable(policy):
            raise DataValidationError(
                f"policy must be a callable prefix -> distribution, got {policy!r}"
            )
        if tree_policy not in ("uct", "puct"):
            raise DataValidationError(
                f"tree_policy must be 'uct' or 'puct', got {tree_policy!r}"
            )
        if iterations < 1:
            raise DataValidationError(f"iterations must be >= 1, got {iterations}")
        if simulation_num < 1:
            raise DataValidationError(
                f"simulation_num must be >= 1, got {simulation_num}"
            )
        if max_len < 4:
            raise DataValidationError(f"max_len must be >= 4, got {max_len}")
        if not 0.0 < expansion_threshold <= 1.0:
            raise DataValidationError(
                "expansion_threshold must be in (0, 1], got "
                f"{expansion_threshold}"
            )
        if not isinstance(c_val, (int, float)) or c_val < 0:
            raise DataValidationError(f"c_val must be >= 0, got {c_val!r}")
        if not isinstance(puct_c, (int, float)) or puct_c < 0:
            raise DataValidationError(f"puct_c must be >= 0, got {puct_c!r}")
        if prior_epochs < 1:
            raise DataValidationError(f"prior_epochs must be >= 1, got {prior_epochs}")
        if corpus is not None and isinstance(corpus, str):
            raise DataValidationError(
                "corpus must be a sequence of SMILES strings, not a single "
                f"string (got {corpus!r} of length {len(corpus)})"
            )
        if corpus is not None and len(list(corpus)) == 0:
            raise DataValidationError(
                "corpus is empty -- pass a non-empty list of SMILES strings "
                "(or None for the bundled default corpus)"
            )
        self.objective = objective
        self.policy = policy
        self.corpus = corpus
        self.iterations = int(iterations)
        self.c_val = float(c_val)
        self.tree_policy = tree_policy
        self.puct_c = float(puct_c)
        self.expansion_threshold = float(expansion_threshold)
        self.simulation_num = int(simulation_num)
        self.max_len = int(max_len)
        self.seed = seed
        self.prior_epochs = int(prior_epochs)
        self.prior_hidden_dim = int(prior_hidden_dim)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def optimize(self, *, prefix: str = "") -> GenerationResult:
        """Run the search and return the ranked :class:`GenerationResult`.

        Parameters
        ----------
        prefix:
            Extension mode (ChemTSv2's ``--input_smiles``): root the tree
            at this partial SMILES and search only its continuations.
            The characters of the prefix must exist in the policy's
            vocabulary (the RNN maps unknowns to ``<unk>``, which rarely
            leads anywhere productive -- audit the prefix first).

        Returns
        -------
        GenerationResult
            The ranked unique molecules found, one ``history`` record
            per cycle, and ``metadata`` carrying the search statistics
            (``n_nodes``, ``n_rollouts``, ``n_cache_hits``, ``max_depth``,
            the configuration echo and ``elapsed_s``).

        Raises
        ------
        DataValidationError
            If ``prefix`` is not a string or exceeds ``max_len``.
        DependencyError
            When the default RNN policy is used and torch is missing, or
            the objective needs RDKit and it is missing.
        """
        if not isinstance(prefix, str):
            raise DataValidationError(f"prefix must be a str, got {prefix!r}")
        if len(prefix) >= self.max_len:
            raise DataValidationError(
                f"prefix ({len(prefix)} chars) must be shorter than max_len "
                f"({self.max_len}) -- there is no room to grow"
            )
        started = time.perf_counter()
        rng = random.Random(self.seed)
        from pharmaml.generative import DEFAULT_CORPUS

        corpus = (
            list(self.corpus)
            if self.corpus is not None
            else list(DEFAULT_CORPUS)
        )
        resolve_policy = self._build_policy(corpus)

        root = _Node(position=prefix)
        n_nodes = 1
        n_rollouts = 0
        n_cache_hits = 0
        max_depth = 0
        # the dedup cache (ChemTSv2's generated_dict): canonical molecule
        # -> (score, breakdown) computed exactly once
        cache: dict[str, tuple[float, dict[str, float]]] = {}
        # the distribution cache: prefix -> policy distribution (the policy
        # is deterministic, so expansion and rollouts share one call)
        dist_cache: dict[str, list[tuple[str, float]]] = {}
        history: list[dict[str, Any]] = []

        for cycle in range(self.iterations):
            # -- 1. selection ------------------------------------------------
            node = root
            while node.children and not node.is_terminal:
                node = self._select_child(node, rng)
            # -- 2. expansion ------------------------------------------------
            if (
                not node.is_terminal
                and len(node.position) < self.max_len
                and not node.children
            ):
                distribution = self._policy_at(node.position, resolve_policy, dist_cache)
                if distribution:
                    for token, prob in self._expansion_set(distribution):
                        child = _Node(
                            # the end token terminates the molecule -- it is
                            # a move, never a character of the SMILES itself
                            position=(
                                node.position if token == _RNN_END
                                else node.position + token
                            ),
                            parent=node,
                            prior=prob,
                            is_terminal=token == _RNN_END,
                        )
                        node.children.append(child)
                        n_nodes += 1
                        max_depth = max(max_depth, len(child.position))
            # -- 3. simulation -----------------------------------------------
            cycle_rewards: list[float] = []
            leaves = node.children if node.children else [node]
            for leaf in leaves:
                if leaf.is_terminal or len(leaf.position) >= self.max_len:
                    molecule = leaf.position
                else:
                    molecule = self._rollout(
                        leaf.position, resolve_policy, dist_cache, rng
                    )
                for _ in range(self.simulation_num):
                    n_rollouts += 1
                    score, _, from_cache = self._evaluate(molecule, cache)
                    if from_cache:
                        n_cache_hits += 1
                    cycle_rewards.append(score)
                    # -- 4. backpropagation (once per rollout occurrence) --
                    walker: _Node | None = leaf
                    while walker is not None:
                        walker.visits += 1
                        walker.total_value += score
                        walker = walker.parent
            best_now = max(cache.items(), key=lambda item: item[1][0], default=None)
            history.append(
                {
                    "cycle": cycle,
                    "best_score": best_now[1][0] if best_now else 0.0,
                    "best_smiles": best_now[0] if best_now else "",
                    "mean_score": (
                        sum(cycle_rewards) / len(cycle_rewards)
                        if cycle_rewards
                        else 0.0
                    ),
                    "n_unique": len(cache),
                    "n_nodes": n_nodes,
                    "cache_hits": n_cache_hits,
                }
            )

        elapsed = time.perf_counter() - started
        ranked = sorted(cache.items(), key=lambda item: (-item[1][0], item[0]))
        smiles = [canonical for canonical, _ in ranked]
        scores = [values[0] for _, values in ranked]
        breakdown = [values[1] for _, values in ranked]
        _LOG.info(
            "MCTS finished: %d cycles, %d nodes, %d rollouts, %d unique "
            "molecules (%d cache hits) in %.1fs",
            self.iterations,
            n_nodes,
            n_rollouts,
            len(cache),
            n_cache_hits,
            elapsed,
        )
        return GenerationResult(
            smiles=smiles,
            scores=scores,
            breakdown=breakdown,
            best=smiles[0] if smiles else "",
            best_score=scores[0] if scores else 0.0,
            history=history,
            metadata={
                "method": "mcts",
                "tree_policy": self.tree_policy,
                "c_val": self.c_val,
                "puct_c": self.puct_c,
                "expansion_threshold": self.expansion_threshold,
                "simulation_num": self.simulation_num,
                "iterations": self.iterations,
                "max_len": self.max_len,
                "seed": self.seed,
                "prefix": prefix,
                "policy": "custom" if self.policy is not None else "smiles-rnn",
                "corpus_size": None if self.policy is not None else len(corpus),
                "n_nodes": n_nodes,
                "n_rollouts": n_rollouts,
                "n_valid": len(cache),
                "n_unique": len(cache),
                "n_cache_hits": n_cache_hits,
                "max_depth": max_depth,
                "elapsed_s": elapsed,
            },
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _build_policy(self, corpus: list[str]) -> _Policy:
        """Return the injected policy, or the trained RNN default."""
        if self.policy is not None:
            return self.policy
        prior = SmilesRnn(hidden_dim=self.prior_hidden_dim, seed=self.seed)
        _LOG.info(
            "MCTS: training the default RNN prior (%d molecules, %d epochs)",
            len(corpus),
            self.prior_epochs,
        )
        prior.train_prior(corpus, epochs=self.prior_epochs)
        return prior.next_token_distribution

    @staticmethod
    def _policy_at(
        prefix: str,
        resolve_policy: _Policy,
        dist_cache: dict[str, list[tuple[str, float]]],
    ) -> list[tuple[str, float]]:
        """The policy distribution at ``prefix`` (memoised)."""
        cached = dist_cache.get(prefix)
        if cached is None:
            cached = resolve_policy(prefix)
            if not cached:
                return []
            total = sum(prob for _, prob in cached)
            if total <= 0.0:
                return []
            if abs(total - 1.0) > 1e-9:  # renormalise sloppy custom policies
                cached = [(token, prob / total) for token, prob in cached]
            dist_cache[prefix] = cached
        return cached

    def _select_child(self, node: _Node, rng: random.Random) -> _Node:
        """The tree policy: UCB1 (v2) or PUCT (v1), ties broken randomly."""
        best_value = -1.0
        best: list[_Node] = []
        log_parent = _safe_log(node.visits)
        for child in node.children:
            if child.visits == 0:
                value = float("inf")
            elif self.tree_policy == "puct":
                value = child.total_value / child.visits + self.puct_c * child.prior * (
                    (node.visits) ** 0.5 / (1.0 + child.visits)
                )
            else:
                value = child.total_value / child.visits + self.c_val * (
                    2.0 * log_parent / child.visits
                ) ** 0.5
            if value > best_value:
                best_value = value
                best = [child]
            elif value == best_value:
                best.append(child)
        return best[rng.randrange(len(best))] if best else node.children[0]

    def _expansion_set(
        self, distribution: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """ChemTSv2's cumulative-probability expansion (min two children)."""
        chosen: list[tuple[str, float]] = []
        cumulative = 0.0
        for token, prob in distribution:
            chosen.append((token, prob))
            cumulative += prob
            if cumulative >= self.expansion_threshold and len(chosen) >= 2:
                break
        return chosen

    def _rollout(
        self,
        prefix: str,
        resolve_policy: _Policy,
        dist_cache: dict[str, list[tuple[str, float]]],
        rng: random.Random,
    ) -> str:
        """Sample characters to completion (end token or ``max_len``)."""
        position = prefix
        while len(position) < self.max_len:
            distribution = self._policy_at(position, resolve_policy, dist_cache)
            if not distribution:
                break
            draw = rng.random()
            cumulative = 0.0
            token = distribution[-1][0]
            for candidate, prob in distribution:
                cumulative += prob
                if draw <= cumulative:
                    token = candidate
                    break
            if token == _RNN_END:
                break
            position += token
        return position

    def _evaluate(
        self, molecule: str, cache: dict[str, tuple[float, dict[str, float]]]
    ) -> tuple[float, str, bool]:
        """Score one rollout result through the dedup cache.

        Returns ``(score, canonical, cache_hit)``; unparseable or blank
        rollouts score ``0.0`` (ChemTSv2's ``-1`` penalty mapped onto the
        toolkit's ``[0, 1]`` objective scale) and are never cached
        (``canonical == ""``).
        """
        canonical = canonical_smiles(molecule)
        if not canonical:
            return 0.0, "", False
        cached = cache.get(canonical)
        if cached is not None:
            return cached[0], canonical, True
        breakdown: dict[str, float] = {}
        score_with_breakdown = getattr(self.objective, "score_with_breakdown", None)
        if callable(score_with_breakdown):
            score, breakdown = score_with_breakdown(canonical)
        else:
            score = self.objective.score(canonical)
        score = float(score)
        cache[canonical] = (score, breakdown)
        return score, canonical, False


def _safe_log(visits: int) -> float:
    """``log`` that stays finite for unvisited parents (== 0)."""
    return math.log(visits) if visits > 0 else 0.0


def run_mcts_optimization(objective: Any, **kwargs: Any) -> GenerationResult:
    """One-call MCTS convenience (mirrors ``run_genetic_optimization``).

    Example
    -------
    >>> from pharmaml.generative import run_mcts_optimization, QedObjective
    >>> def flat_policy(prefix):
    ...     return [("C", 0.4), ("N", 0.2), ("O", 0.2), ("<end>", 0.2)]
    >>> result = run_mcts_optimization(
    ...     QedObjective(), policy=flat_policy, iterations=4, seed=42)
    >>> result.metadata["method"]
    'mcts'
    """
    return MctsOptimizer(objective, **kwargs).optimize()
