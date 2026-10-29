"""DrugCell's visible neural network: the ontology as architecture.

``idekerlab/DrugCell`` was reviewed file by file for this module:
``code/drugcell_NN.py`` (the model — 185 lines), ``code/train_drugcell.py``
(the trainer — 201 lines), ``code/util.py`` (the ontology loader), the
two prediction scripts, ``data/drugcell_ont.txt`` (62,920 rows of the
GO hierarchy in the three-column format), the feature files'
contracts (``cell2ind``/``cell2mutation``/``drug2fingerprint``) and the
README's data story (509,294 (cell, drug) pairs, 1,235 cell lines,
684 drugs; genotype = binary mutation vector of the 3,008 most
frequently mutated genes; drug = 2,048-bit Morgan fingerprint).

What makes DrugCell *DrugCell* is not the prediction (a plain MLP does
that) but the **visibility**: every neuron sits in a subsystem, the
connectivity mirrors a Gene-Ontology DAG, and every term carries an
*aux head* — a single output trained toward the drug response itself —
so the model can be read out subsystem by subsystem.  That is exactly
what lands here, in plain NumPy with the training included:

* :func:`parse_ontology` -- the ``drugcell_ont.txt`` format (three
  columns: parent term, child term, ``default`` -- or parent term,
  gene, anything else), the gene filter, the single-root and
  empty-term checks the original ``sys.exit``s on, turned into
  actionable ``DataValidationError``s.
* :func:`hierarchy_report` -- the peeling audit: leaf-first layers
  (``construct_NN_graph``'s algorithm), depth, term sizes, the
  enrichment of directly-annotated genes.
* :class:`VisibleNetwork` -- the network itself: per-term linear+tanh
  subnetworks over (children states ⊕ directly-annotated genes), the
  aux head per term (h → 1 → 1, the paper's contribution readout),
  the drug branch MLP, and the final concat of root state + drug
  state through its own aux head -- the prediction.  Trained by exact
  manual backpropagation through the DAG with the paper's loss
  (``MSE(final, y) + 0.2 · Σ_terms MSE(aux_t, y)``), verified against
  finite differences by :meth:`VisibleNetwork.gradient_check`.
* :meth:`VisibleNetwork.contribution_report` -- the interpretability
  the architecture exists for: each subsystem's aux contribution,
  ranked; which terms move the prediction.

The honest scope, stated once: BatchNorm is omitted (an optimisation
for deep stacks at 500k-pair scale; the concept and the gradients are
unaffected), the GDSC-scale training runs, GPU kernels and the
pretrained ``.pth`` checkpoints stay with the PyTorch engine the
toolkit deliberately does not vendor -- what is absorbed is the
architecture, its training mathematics and its interpretation, exact
and runnable at teaching scale.

Examples
--------
>>> from pharmaml.ml.visiblenet import (
...     VisibleNetwork, hierarchy_report, parse_ontology,
... )
>>> ont_text = '''
... GO:ROOT GO:MITOSIS default
... GO:ROOT GO:DNA_REPAIR default
... GO:MITOSIS GENE_A
... GO:MITOSIS GENE_B
... GO:DNA_REPAIR GENE_B
... GO:DNA_REPAIR GENE_C
... '''
>>> graph = parse_ontology(ont_text)
>>> graph.root
'GO:ROOT'
>>> sorted(graph.term_direct_genes["GO:MITOSIS"])
['GENE_A', 'GENE_B']
>>> report = hierarchy_report(graph)
>>> report.n_terms
3
>>> report.layers[0]                      # the leaves peel first
('GO:DNA_REPAIR', 'GO:MITOSIS')
>>> net = VisibleNetwork.build(graph, gene_dim=3, drug_dim=4, seed=42)
>>> out = net.forward(
...     genes=[[1, 0, 1], [0, 1, 0]],
...     fingerprints=[[1, 0, 0, 1], [0, 1, 1, 0]],
... )
>>> out.predictions.shape
(2, 1)
>>> sorted(out.contributions)             # every subsystem speaks
['GO:DNA_REPAIR', 'GO:MITOSIS', 'GO:ROOT', 'final']
>>> import numpy as np
>>> genes = np.random.default_rng(42).integers(0, 2, size=(60, 3))
>>> fps = np.random.default_rng(43).integers(0, 2, size=(60, 4))
>>> y = (genes[:, 0] * 2.0 - fps[:, 1] * 1.0).reshape(-1, 1)
>>> history = net.fit(genes, fps, y, epochs=60, lr=0.05, seed=42)
>>> history.train_losses[-1] < history.train_losses[0]
True
>>> float(net.pearson(genes, fps, y)[0]) > 0.5
True
>>> ranking = net.contribution_report(genes, fps)
>>> {item.term for item in ranking} >= {"GO:MITOSIS", "GO:DNA_REPAIR", "GO:ROOT"}
True
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: Random seed threaded through every weight init (the pin).
RANDOM_STATE = 42

#: The paper's aux-loss weight (train_drugcell.py: 0.2 per term).
AUX_LOSS_WEIGHT = 0.2


# ---------------------------------------------------------------------------
# The ontology (drugcell_ont.txt, parsed and audited)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OntologyGraph:
    """A parsed DrugCell ontology: the DAG plus its gene annotations.

    ``children`` maps every term to its child terms (edges run parent
    → child, exactly the ``add_edge(line[0], line[1])`` direction of
    ``util.load_ontology``); ``term_direct_genes`` holds only the genes
    *directly* annotated to each term; ``layers`` is the leaf-first
    peeling order (``construct_NN_graph``'s algorithm -- a term appears
    once every one of its children has appeared before it).
    """

    root: str
    children: dict[str, tuple[str, ...]]
    term_direct_genes: dict[str, tuple[str, ...]]
    term_sizes: dict[str, int]
    layers: tuple[tuple[str, ...], ...]
    n_genes: int
    genes: tuple[str, ...]

    @property
    def n_terms(self) -> int:
        return len(self.children)

    @property
    def depth(self) -> int:
        """The number of peeling layers (root-only graph -> 1)."""
        return len(self.layers)


def parse_ontology(
    text: str | list[str] | tuple[str, ...],
    known_genes: list[str] | tuple[str, ...] | None = None,
    root: str | None = None,
) -> OntologyGraph:
    """Parse the ``drugcell_ont.txt`` three-column format.

    A row ``TERM CHILD default`` is a parent→child edge; any other
    third column (or no third column) annotates ``GENE`` directly to
    ``TERM`` -- exactly ``util.load_ontology``'s reading, with the
    ``known_genes`` filter keeping the original's silent skip of
    genes outside the profiled set.  Validation the original
    ``sys.exit``s on is raised as actionable errors instead: unknown
    genes are *counted* (the loader prints how many survived), a
    cyclic graph fails the peeling, and an ontology with zero or
    several roots is rejected by name.

    Raises
    ------
    DataValidationError
        On blank rows with content, a cyclic graph, or a root count
        other than one.
    """
    rows: list[list[str]] = []
    if isinstance(text, str):
        lines = text.splitlines()
    else:
        lines = list(text)
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            raise DataValidationError(
                f"ontology row {stripped!r} needs at least term and child/gene columns"
            )
        rows.append(parts)
    if not rows:
        raise DataValidationError("the ontology text is empty")

    edges: list[tuple[str, str]] = []
    annotations: list[tuple[str, str]] = []
    for parts in rows:
        if len(parts) >= 3 and parts[2] == "default":
            edges.append((parts[0], parts[1]))
        else:
            annotations.append((parts[0], parts[1]))

    known = set(known_genes) if known_genes is not None else None
    skipped = 0
    term_direct_genes: dict[str, set[str]] = {}
    gene_order: list[str] = []
    seen_genes: set[str] = set()
    for term, gene in annotations:
        if known is not None and gene not in known:
            skipped += 1
            continue
        term_direct_genes.setdefault(term, set()).add(gene)
        if gene not in seen_genes:
            seen_genes.add(gene)
            gene_order.append(gene)
    if skipped:
        _LOG.debug("parse_ontology: %d gene rows skipped (not profiled)", skipped)

    nodes: set[str] = {term for term, _child in edges}
    nodes.update(child for _term, child in edges)
    nodes.update(term_direct_genes)
    children_map: dict[str, list[str]] = {term: [] for term in nodes}
    indegree: dict[str, int] = {term: 0 for term in nodes}
    for parent, child in edges:
        children_map[parent].append(child)
        indegree[child] += 1
    # deterministic order: the peeling and the reports are stable
    for term, child_list in children_map.items():
        children_map[term] = sorted(set(child_list))

    roots = sorted(term for term, degree in indegree.items() if degree == 0)
    if not roots:
        raise DataValidationError("the ontology has no root (every term has a parent)")
    if len(roots) > 1:
        raise DataValidationError(
            f"the ontology has {len(roots)} roots ({', '.join(roots)}); "
            "DrugCell requires exactly one -- pass root= to pick it after merging"
        )
    chosen_root = root if root is not None else roots[0]
    if chosen_root not in nodes:
        raise DataValidationError(f"root {chosen_root!r} is not a term of the ontology")

    # term_sizes: direct genes ∪ all descendants' direct genes (the
    # util.load_ontology computation via networkx descendants)
    children: dict[str, tuple[str, ...]] = {t: tuple(v) for t, v in children_map.items()}
    term_sizes: dict[str, int] = {}
    for term in nodes:
        total = set(term_direct_genes.get(term, ()))
        stack = list(children[term])
        seen: set[str] = set()
        while stack:
            descendant = stack.pop()
            if descendant in seen:
                continue
            seen.add(descendant)
            total.update(term_direct_genes.get(descendant, ()))
            stack.extend(children[descendant])
        term_sizes[term] = len(total)

    # the peeling (construct_NN_graph): leaves = no children, remove, repeat
    remaining_children = {t: list(v) for t, v in children.items()}
    layers: list[list[str]] = []
    remaining = set(nodes)
    while remaining:
        leaves = sorted(t for t in remaining if not remaining_children[t])
        if not leaves:
            raise DataValidationError("the ontology is cyclic -- the leaf peeling cannot terminate")
        layers.append(leaves)
        for leaf in leaves:
            remaining.discard(leaf)
            for parent in remaining:
                if leaf in remaining_children[parent]:
                    remaining_children[parent].remove(leaf)

    graph = OntologyGraph(
        root=chosen_root,
        children=children,
        term_direct_genes={term: tuple(sorted(genes)) for term, genes in term_direct_genes.items()},
        term_sizes=term_sizes,
        layers=tuple(tuple(layer) for layer in layers),
        n_genes=len(gene_order),
        genes=tuple(gene_order),
    )
    _LOG.debug(
        "parse_ontology: %d terms, %d genes, %d layers, root=%s",
        graph.n_terms,
        graph.n_genes,
        graph.depth,
        chosen_root,
    )
    return graph


@dataclass(frozen=True)
class OntologyRow:
    """One term's audit row: size, depth, layer, gene enrichment."""

    term: str
    n_children: int
    n_direct_genes: int
    total_genes: int
    depth_from_root: int
    layer_index: int


@dataclass(frozen=True)
class OntologyReport:
    """The hierarchy audit: rows per term plus the headline stats."""

    root: str
    n_terms: int
    n_genes: int
    depth: int
    n_root_direct_genes: int
    rows: tuple[OntologyRow, ...] = field(default=())

    def layer(self, index: int) -> tuple[str, ...]:
        """The terms of peeling layer ``index`` (0 = the leaves)."""
        terms = tuple(row.term for row in self.rows if row.layer_index == index)
        return terms

    @property
    def layers(self) -> tuple[tuple[str, ...], ...]:
        """The full leaf-first peeling order, layer by layer."""
        return tuple(self.layer(index) for index in range(self.depth))


def hierarchy_report(graph: OntologyGraph) -> OntologyReport:
    """The peeling audit: per-term rows and the hierarchy's shape.

    ``depth_from_root`` is the longest path from the root (the root
    itself is 0); ``layer_index`` is the leaf-first peeling position
    (the order :class:`VisibleNetwork` evaluates in).
    """
    depth_from_root: dict[str, int] = {graph.root: 0}
    # BFS over the DAG edges (parent -> child); longest-path via
    # topological order = reverse peeling
    for layer in reversed(graph.layers):
        for term in layer:
            for child in graph.children[term]:
                current = depth_from_root.get(child, 0)
                depth_from_root[child] = max(current, depth_from_root[term] + 1)
    layer_index: dict[str, int] = {
        term: index for index, layer in enumerate(graph.layers) for term in layer
    }
    rows = tuple(
        OntologyRow(
            term=term,
            n_children=len(graph.children[term]),
            n_direct_genes=len(graph.term_direct_genes.get(term, ())),
            total_genes=graph.term_sizes[term],
            depth_from_root=depth_from_root[term],
            layer_index=layer_index[term],
        )
        for term in sorted(graph.children)
    )
    report = OntologyReport(
        root=graph.root,
        n_terms=graph.n_terms,
        n_genes=graph.n_genes,
        depth=graph.depth,
        n_root_direct_genes=len(graph.term_direct_genes.get(graph.root, ())),
        rows=rows,
    )
    _LOG.debug(
        "hierarchy_report: %d terms, depth %d, %d root-direct genes",
        report.n_terms,
        report.depth,
        report.n_root_direct_genes,
    )
    return report


# ---------------------------------------------------------------------------
# The visible network
# ---------------------------------------------------------------------------


def _linear_init(rng: np.random.Generator, n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Xavier-style init for one linear layer (seeds pinned)."""
    limit = math_sqrt(6.0 / (n_in + n_out)) if (n_in + n_out) > 0 else 0.5
    weight = rng.uniform(-limit, limit, size=(n_in, n_out))
    bias = np.zeros(n_out)
    return weight, bias


def math_sqrt(value: float) -> float:
    """float sqrt without importing math wholesale (kept tiny on purpose)."""
    return float(np.sqrt(value))


@dataclass(frozen=True)
class TermContribution:
    """One subsystem's readout: the aux head's contribution, ranked."""

    term: str
    mean_contribution: float
    mean_abs_contribution: float
    n_samples: int


@dataclass(frozen=True)
class VisibleForward:
    """One forward pass through the hierarchy.

    ``predictions`` is ``(n, 1)``; ``contributions`` maps every term
    (and ``"final"``) to its ``(n,)`` aux-head output -- the paper's
    subsystem readout; ``states`` maps every term to its ``(n, h)``
    hidden state.
    """

    predictions: np.ndarray
    contributions: dict[str, np.ndarray]
    states: dict[str, np.ndarray]


class VisibleNetwork:
    """DrugCell's architecture in plain NumPy, training included.

    Built from an :class:`OntologyGraph` (a GO-style DAG): every term
    gets a subnetwork whose input is the concat of its children's
    states and its directly-annotated genes (the original's
    ``direct_gene_layer`` is a masked Linear that functions as a
    gather -- implemented here as the gather it is), followed by the
    aux head (h → 1 → 1) whose output *is* the term's contribution.
    The drug branch is a plain tanh MLP; the final layer concatenates
    the root state and the drug state and predicts through its own
    aux head.  Training minimises the paper's loss
    (``MSE(final, y) + 0.2 · Σ MSE(aux_t, y)``) by exact backprop
    through the DAG -- no engine, hand-derived, finite-difference
    verified (:meth:`gradient_check`).
    """

    def __init__(
        self,
        ontology: OntologyGraph,
        weights: dict[str, dict[str, np.ndarray]],
        gene_order: list[str],
        drug_hiddens: tuple[int, ...] = (32,),
        final_hidden: int = 16,
    ) -> None:
        self.ontology = ontology
        self.weights = weights
        self.gene_order = list(gene_order)
        self.gene_index = {gene: index for index, gene in enumerate(gene_order)}
        self.drug_hiddens = tuple(drug_hiddens)
        self.final_hidden = final_hidden

    # -- construction ---------------------------------------------------- #

    @classmethod
    def build(
        cls,
        ontology: OntologyGraph,
        gene_dim: int,
        drug_dim: int,
        hidden_per_term: int = 6,
        drug_hiddens: tuple[int, ...] = (16,),
        final_hidden: int = 8,
        seed: int = RANDOM_STATE,
    ) -> VisibleNetwork:
        """Seeded construction of every subnetwork in the hierarchy.

        Raises
        ------
        DataValidationError
            On non-positive dims or an ontology whose terms carry no
            genes at all.
        """
        if gene_dim < 1 or drug_dim < 1:
            raise DataValidationError(
                f"gene_dim and drug_dim must be >= 1, got {(gene_dim, drug_dim)!r}"
            )
        if hidden_per_term < 1 or final_hidden < 1 or not drug_hiddens or min(drug_hiddens) < 1:
            raise DataValidationError("hidden sizes must be >= 1")
        for term in ontology.children:
            if ontology.term_sizes.get(term, 0) == 0:
                raise DataValidationError(
                    f"term {term!r} has no annotated genes anywhere in its "
                    "subtree -- DrugCell deletes such terms (util.load_ontology)"
                )
        rng = np.random.default_rng(seed)
        weights: dict[str, dict[str, np.ndarray]] = {}
        gene_order = (
            list(ontology.genes)
            + [f"gene_{i}" for i in range(max(0, gene_dim - len(ontology.genes)))][
                : max(gene_dim, len(ontology.genes))
            ]
        )
        if gene_dim < len(ontology.genes):
            raise DataValidationError(
                f"gene_dim={gene_dim} is smaller than the ontology's "
                f"{len(ontology.genes)} annotated genes"
            )
        for layer in ontology.layers:
            for term in layer:
                child_sizes = [hidden_per_term for _child in ontology.children[term]]
                n_genes = len(ontology.term_direct_genes.get(term, ()))
                n_in = sum(child_sizes) + n_genes
                weight, bias = _linear_init(rng, max(n_in, 1), hidden_per_term)
                aux_w1, aux_b1 = _linear_init(rng, hidden_per_term, 1)
                aux_w2, aux_b2 = _linear_init(rng, 1, 1)
                weights[term] = {
                    "W": weight,
                    "b": bias,
                    "aux_w1": aux_w1,
                    "aux_b1": aux_b1,
                    "aux_w2": aux_w2,
                    "aux_b2": aux_b2,
                }
        # the drug branch
        input_size = drug_dim
        for index, width in enumerate(drug_hiddens):
            weight, bias = _linear_init(rng, input_size, width)
            weights[f"drug_{index}"] = {"W": weight, "b": bias}
            input_size = width
        # the final layer over (root state ⊕ last drug hidden)
        final_in = hidden_per_term + drug_hiddens[-1]
        weight, bias = _linear_init(rng, final_in, final_hidden)
        aux_w1, aux_b1 = _linear_init(rng, final_hidden, 1)
        aux_w2, aux_b2 = _linear_init(rng, 1, 1)
        weights["final"] = {
            "W": weight,
            "b": bias,
            "aux_w1": aux_w1,
            "aux_b1": aux_b1,
            "aux_w2": aux_w2,
            "aux_b2": aux_b2,
        }
        _LOG.debug(
            "visible network built: %d term subnets, drug %s, seed %d",
            ontology.n_terms,
            drug_hiddens,
            seed,
        )
        return cls(
            ontology=ontology,
            weights=weights,
            gene_order=gene_order,
            drug_hiddens=tuple(drug_hiddens),
            final_hidden=final_hidden,
        )

    # -- forward ---------------------------------------------------------- #

    def _term_gene_columns(self, term: str) -> list[int]:
        return [
            self.gene_index[gene]
            for gene in self.ontology.term_direct_genes.get(term, ())
            if gene in self.gene_index
        ]

    def forward(
        self,
        genes: np.ndarray | list[list[float]],
        fingerprints: np.ndarray | list[list[float]],
    ) -> VisibleForward:
        """The leaf-first evaluation of the whole hierarchy.

        Raises
        ------
        DataValidationError
        On wrong shapes or non-finite inputs.
        """
        gene_matrix = _as_matrix(genes, "genes")
        drug_matrix = _as_matrix(fingerprints, "fingerprints")
        if gene_matrix.shape[1] != len(self.gene_order):
            raise DataValidationError(
                f"genes has {gene_matrix.shape[1]} columns, the network "
                f"was built on {len(self.gene_order)}"
            )
        states: dict[str, np.ndarray] = {}
        contributions: dict[str, np.ndarray] = {}
        n = gene_matrix.shape[0]
        for layer in self.ontology.layers:
            for term in layer:
                inputs: list[np.ndarray] = []
                for child in self.ontology.children[term]:
                    if child not in states:
                        raise DataValidationError(
                            f"term {child!r} was not evaluated before its parent "
                            f"{term!r} -- the peeling order is broken"
                        )
                    inputs.append(states[child])
                columns = self._term_gene_columns(term)
                if columns:
                    inputs.append(gene_matrix[:, columns])
                if not inputs:
                    inputs.append(np.zeros((n, 1)))
                concatenated = np.concatenate(inputs, axis=1)
                block = self.weights[term]
                hidden = np.tanh(concatenated @ block["W"] + block["b"])
                aux_hidden = np.tanh(hidden @ block["aux_w1"] + block["aux_b1"])
                contribution = aux_hidden @ block["aux_w2"] + block["aux_b2"]
                states[term] = hidden
                contributions[term] = contribution.reshape(-1)
        # the drug branch
        drug_out = drug_matrix
        for index in range(len(self.drug_hiddens)):
            block = self.weights[f"drug_{index}"]
            drug_out = np.tanh(drug_out @ block["W"] + block["b"])
        # the final layer: root state ⊕ drug state
        final_input = np.concatenate([states[self.ontology.root], drug_out], axis=1)
        block = self.weights["final"]
        final_hidden_state = np.tanh(final_input @ block["W"] + block["b"])
        aux_hidden = np.tanh(final_hidden_state @ block["aux_w1"] + block["aux_b1"])
        prediction = aux_hidden @ block["aux_w2"] + block["aux_b2"]
        contributions["final"] = prediction.reshape(-1)
        return VisibleForward(
            predictions=prediction,
            contributions=contributions,
            states=states,
        )

    # -- the paper's loss and its exact gradient ------------------------- #

    def loss(self, genes: np.ndarray, fingerprints: np.ndarray, y: np.ndarray) -> float:
        """MSE(final) + 0.2 · Σ_terms MSE(aux_t) -- train_drugcell's total_loss."""
        result = self.forward(genes, fingerprints)
        target = _as_matrix(y, "y")
        total = float(np.mean((result.predictions - target) ** 2))
        for term in self.ontology.children:
            total += AUX_LOSS_WEIGHT * float(
                np.mean((result.contributions[term].reshape(-1, 1) - target) ** 2)
            )
        return total

    def _forward_backward(
        self,
        genes: np.ndarray,
        fingerprints: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, dict[str, dict[str, np.ndarray]]]:
        """One loss evaluation plus its exact gradient w.r.t. every weight.

        The backward pass threads the chain rule through the DAG:
        each term's hidden state receives gradient from its own aux
        head *and* from every parent's linear layer (via the parent's
        input-concat slicing), tanh's derivative ``(1 - h²)`` applied,
        and the standard ``dW = xᵀδ``, ``db = Σδ``, ``dx = δWᵀ``
        per linear layer -- the same hand-derived pattern the
        toolkit's from-scratch LSTM (ml/sequences) carries, verified
        by :meth:`gradient_check` to match finite differences.
        """
        gene_matrix = genes
        drug_matrix = fingerprints
        target = y
        n = gene_matrix.shape[0]

        # ---- forward, caching every concat and activation ----
        concats: dict[str, np.ndarray] = {}
        child_slices: dict[str, list[tuple[str, tuple[int, int]]]] = {}
        states: dict[str, np.ndarray] = {}
        contributions: dict[str, np.ndarray] = {}
        aux_hiddens: dict[str, np.ndarray] = {}
        for layer in self.ontology.layers:
            for term in layer:
                inputs: list[np.ndarray] = []
                slices: list[tuple[str, tuple[int, int]]] = []
                offset = 0
                for child in self.ontology.children[term]:
                    inputs.append(states[child])
                    slices.append((child, (offset, offset + states[child].shape[1])))
                    offset += states[child].shape[1]
                columns = self._term_gene_columns(term)
                if columns:
                    inputs.append(gene_matrix[:, columns])
                    slices.append(("__genes__", (offset, offset + len(columns))))
                if not inputs:
                    inputs.append(np.zeros((n, 1)))
                    slices.append(("__zero__", (0, 1)))
                concatenated = np.concatenate(inputs, axis=1)
                block = self.weights[term]
                pre = concatenated @ block["W"] + block["b"]
                hidden = np.tanh(pre)
                aux_hidden = np.tanh(hidden @ block["aux_w1"] + block["aux_b1"])
                contribution = aux_hidden @ block["aux_w2"] + block["aux_b2"]
                concats[term] = concatenated
                child_slices[term] = slices
                states[term] = hidden
                aux_hiddens[term] = aux_hidden
                contributions[term] = contribution

        drug_out = drug_matrix
        drug_concats: dict[int, np.ndarray] = {}
        for index in range(len(self.drug_hiddens)):
            block = self.weights[f"drug_{index}"]
            drug_concats[index] = drug_out
            drug_out = np.tanh(drug_out @ block["W"] + block["b"])

        final_input = np.concatenate([states[self.ontology.root], drug_out], axis=1)
        block = self.weights["final"]
        final_hidden = np.tanh(final_input @ block["W"] + block["b"])
        final_aux_hidden = np.tanh(final_hidden @ block["aux_w1"] + block["aux_b1"])
        prediction = final_aux_hidden @ block["aux_w2"] + block["aux_b2"]

        # ---- the loss ----
        error = prediction - target
        total = float(np.mean(error**2))
        for term in self.ontology.children:
            aux_error = contributions[term].reshape(-1, 1) - target
            total += AUX_LOSS_WEIGHT * float(np.mean(aux_error**2))

        # ---- backward ----
        grads: dict[str, dict[str, np.ndarray]] = {
            name: {key: np.zeros_like(value) for key, value in blk.items()}
            for name, blk in self.weights.items()
        }
        # final head
        d_pred = (2.0 / n) * error  # (n,1)
        grads["final"]["aux_w2"] += final_aux_hidden.T @ d_pred
        grads["final"]["aux_b2"] += d_pred.sum(axis=0)
        d_aux_hidden = d_pred @ self.weights["final"]["aux_w2"].T
        d_final_hidden = d_aux_hidden * (1.0 - final_aux_hidden**2)
        grads["final"]["aux_w1"] += final_hidden.T @ d_final_hidden
        grads["final"]["aux_b1"] += d_final_hidden.sum(axis=0)
        d_pre_final = d_final_hidden @ self.weights["final"]["aux_w1"].T * (1.0 - final_hidden**2)
        grads["final"]["W"] += final_input.T @ d_pre_final
        grads["final"]["b"] += d_pre_final.sum(axis=0)
        d_final_input = d_pre_final @ self.weights["final"]["W"].T
        root_width = states[self.ontology.root].shape[1]
        d_root_state = d_final_input[:, :root_width]
        d_drug_out = d_final_input[:, root_width:]

        # drug branch backward
        for index in reversed(range(len(self.drug_hiddens))):
            blk = self.weights[f"drug_{index}"]
            d_pre = (
                d_drug_out * (1.0 - drug_out**2)
                if index == len(self.drug_hiddens) - 1
                else d_drug_out
            )
            # recompute drug_out per layer index for the tanh derivative:
            # drug_out at index i is the OUTPUT of layer i
            grads[f"drug_{index}"]["W"] += drug_concats[index].T @ d_pre
            grads[f"drug_{index}"]["b"] += d_pre.sum(axis=0)
            d_input = d_pre @ blk["W"].T
            if index > 0:
                previous = np.tanh(
                    drug_concats[index] @ self.weights[f"drug_{index - 1}"]["W"]
                    + self.weights[f"drug_{index - 1}"]["b"]
                )
                d_drug_out = d_input * (1.0 - previous**2)
            else:
                d_drug_out = d_input

        # the term DAG backward: gradients flow from parents to children
        d_states: dict[str, np.ndarray] = {self.ontology.root: d_root_state.copy()}
        for layer in reversed(self.ontology.layers):
            for term in layer:
                # aux-loss gradient onto this term's hidden state
                aux_error = contributions[term].reshape(-1, 1) - target
                d_contribution = (2.0 * AUX_LOSS_WEIGHT / n) * aux_error
                blk = self.weights[term]
                aux_hidden = aux_hiddens[term]
                d_aux_hidden = d_contribution @ blk["aux_w2"].T
                d_hidden_aux = d_aux_hidden * (1.0 - aux_hidden**2)
                grads[term]["aux_w2"] += aux_hidden.T @ d_contribution
                grads[term]["aux_b2"] += d_contribution.sum(axis=0)
                grads[term]["aux_w1"] += states[term].T @ d_hidden_aux
                grads[term]["aux_b1"] += d_hidden_aux.sum(axis=0)
                d_hidden_from_aux = d_hidden_aux @ blk["aux_w1"].T

                d_hidden = d_hidden_from_aux + d_states.get(term, np.zeros_like(states[term]))
                d_pre = d_hidden * (1.0 - states[term] ** 2)
                grads[term]["W"] += concats[term].T @ d_pre
                grads[term]["b"] += d_pre.sum(axis=0)
                d_concat = d_pre @ blk["W"].T
                for child, (start, stop) in child_slices[term]:
                    if child == "__genes__" or child == "__zero__":
                        continue  # gene inputs and the zero pad have no weights
                    d_states[child] = (
                        d_states.get(child, np.zeros_like(states[child])) + d_concat[:, start:stop]
                    )

        return total, grads

    def fit(
        self,
        genes: np.ndarray | list[list[float]],
        fingerprints: np.ndarray | list[list[float]],
        y: np.ndarray | list[float],
        epochs: int = 100,
        lr: float = 0.05,
        seed: int = RANDOM_STATE,
    ) -> VisibleHistory:
        """Full-batch gradient-descent training on the paper's loss.

        Deterministic (seeded shuffling is *not* used: full-batch GD
        is exactly reproducible, the toolkit's pin).

        Raises
        ------
        DataValidationError
        On non-positive epochs/lr or mismatched shapes.
        """
        gene_matrix = _as_matrix(genes, "genes")
        drug_matrix = _as_matrix(fingerprints, "fingerprints")
        target = _as_matrix(y, "y")
        if target.shape[1] != 1:
            raise DataValidationError(f"y must have exactly one column, got {target.shape}")
        if target.shape[0] != gene_matrix.shape[0] or drug_matrix.shape[0] != gene_matrix.shape[0]:
            raise DataValidationError(
                f"row mismatch: genes {gene_matrix.shape[0]}, fingerprints "
                f"{drug_matrix.shape[0]}, y {target.shape[0]}"
            )
        if epochs < 1:
            raise DataValidationError(f"epochs must be >= 1, got {epochs}")
        if lr <= 0:
            raise DataValidationError(f"lr must be > 0, got {lr}")
        losses: list[float] = []
        for epoch in range(epochs):
            loss, grads = self._forward_backward(gene_matrix, drug_matrix, target)
            for name, block_grads in grads.items():
                for key, gradient in block_grads.items():
                    self.weights[name][key] -= lr * gradient
            losses.append(loss)
            if epoch % max(1, epochs // 10) == 0:
                _LOG.debug("visible net epoch %d/%d loss %.6f", epoch, epochs, loss)
        pearson, _spear = self.pearson(gene_matrix, drug_matrix, target)
        history = VisibleHistory(
            train_losses=tuple(losses),
            final_loss=losses[-1],
            pearson_r=pearson,
            n_epochs=int(epochs),
        )
        return history

    def pearson(
        self,
        genes: np.ndarray | list[list[float]],
        fingerprints: np.ndarray | list[list[float]],
        y: np.ndarray | list[float],
    ) -> tuple[float, float]:
        """(Pearson r, Spearman ρ) of the predictions -- DrugCell's eval.

        Spearman is the rank correlation (average ranks for ties),
        computed without scipy: the toolkit's own ranked pairs.
        """
        result = self.forward(genes, fingerprints)
        predicted = result.predictions.reshape(-1)
        actual = np.asarray(y, dtype=float).reshape(-1)
        pearson = _pearson(predicted, actual)
        spearman = _pearson(_rankdata(predicted), _rankdata(actual))
        return pearson, spearman

    def gradient_check(
        self,
        genes: np.ndarray | list[list[float]],
        fingerprints: np.ndarray | list[list[float]],
        y: np.ndarray | list[float],
        epsilon: float = 1e-6,
    ) -> float:
        """Max |analytic − finite-difference| over every weight.

        The house precedent (ml/sequences' LSTM) verifies to ~1e-8;
        this network's check is the same contract for the DAG.
        """
        gene_matrix = _as_matrix(genes, "genes")
        drug_matrix = _as_matrix(fingerprints, "fingerprints")
        target = _as_matrix(y, "y")
        _loss, analytic = self._forward_backward(gene_matrix, drug_matrix, target)
        worst = 0.0
        for name, block in self.weights.items():
            for key in block:
                flat = block[key].reshape(-1)
                for index in range(flat.size):
                    original = flat[index]
                    flat[index] = original + epsilon
                    plus = self.loss(gene_matrix, drug_matrix, target)
                    flat[index] = original - epsilon
                    minus = self.loss(gene_matrix, drug_matrix, target)
                    flat[index] = original
                    numeric = (plus - minus) / (2.0 * epsilon)
                    worst = max(worst, abs(numeric - analytic[name][key].reshape(-1)[index]))
        _LOG.debug("gradient_check: max deviation %.3e", worst)
        return float(worst)

    def contribution_report(
        self,
        genes: np.ndarray | list[list[float]],
        fingerprints: np.ndarray | list[list[float]],
    ) -> list[TermContribution]:
        """The subsystem ranking: mean aux contribution per term.

        This is the paper's interpretability readout -- which pathways
        move the prediction -- as a sorted list (largest mean absolute
        contribution first).
        """
        result = self.forward(genes, fingerprints)
        ranking = [
            TermContribution(
                term=term,
                mean_contribution=float(np.mean(values)),
                mean_abs_contribution=float(np.mean(np.abs(values))),
                n_samples=int(values.size),
            )
            for term, values in result.contributions.items()
        ]
        ranking.sort(key=lambda item: (-item.mean_abs_contribution, item.term))
        return ranking


@dataclass(frozen=True)
class VisibleHistory:
    """The training curve plus the final agreement numbers."""

    train_losses: tuple[float, ...]
    final_loss: float
    pearson_r: float
    n_epochs: int


# ---------------------------------------------------------------------------
# Small NumPy helpers (no scipy: the toolkit's own)
# ---------------------------------------------------------------------------


def _as_matrix(values: np.ndarray | list[list[float]] | list[float], name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise DataValidationError(f"{name} must be 2-D, got {matrix.ndim}-D")
    if matrix.shape[0] < 1:
        raise DataValidationError(f"{name} must have at least one row")
    if not np.all(np.isfinite(matrix)):
        raise DataValidationError(f"{name} must be finite (no NaN/inf)")
    return matrix


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    centered_a = a - a.mean()
    centered_b = b - b.mean()
    denominator = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
    if denominator < 1e-24:
        return float("nan")
    return float(np.dot(centered_a, centered_b) / denominator)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean of their span."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.size, dtype=float)
    index = 0
    while index < array.size:
        start = index
        value = array[order[index]]
        while index < array.size and array[order[index]] == value:
            index += 1
        average_rank = 0.5 * (start + 1 + index)  # 1-based mean of [start+1, index]
        ranks[order[start:index]] = average_rank
    return ranks
