"""SMILES enumeration: the classic atom-order augmentation, as a toolkit surface.

A SMILES string is not a name for a molecule, it is *one of ~n! equivalent
descriptions* of it (one per atom traversal order).  A character-level model
trained only on canonical strings memorises the string, not the molecule;
the classic remedy -- Bjerrum, *SMILES enumeration as data augmentation for
network modeling of molecules* (arXiv:1703.07076, 2017), the ``SmilesEnumerator``
whose Keras-1.2.2-era ``Iterator`` / ``SmilesIterator`` plumbing the vendored
file itself credits -- randomises the atom order with ``Chem.RenumberAtoms``
and re-serialises non-canonically, so every epoch shows the model a
different-but-equivalent string.  That code was carried into molgpt/LigGPT
(https://github.com/devalab/molgpt) and from there into
https://github.com/AspirinCode/drug-likeness-space (``crt/utils.py`` --
``SmilesEnumerator.randomize_smiles`` plus the Keras iterator classes;
``crt/dataset.py`` flips an ``aug_prob`` coin per sample and tokenises with
the regex below) -- the copy this toolkit absorbed.

What did NOT land, honestly: the one-hot tensor machinery.
``SmilesEnumerator.fit`` / ``transform`` / ``reverse_transform`` -- the
charset-to-padding-width fit and the left-padded ``(batch, pad, charset)``
int8 matrices a character-level torch model trains on, with the argmax
decode back -- did NOT land: this toolkit surface ships no character-level
torch trainer for them to feed (and the CRT GPT model itself --
``crt/main_gpt.py`` with the torch sampling helpers around it -- did not
land either).  What lands is the *string* layer an augmentation workflow
actually needs:

1. **Tokenisation** -- :data:`SMILES_TOKEN_PATTERN` + :func:`tokenize_smiles`:
   the canonical character-level SMILES regex (bracket atoms as single
   tokens) that ``crt/dataset.py`` compiles -- the same tokeniser the
   molgpt/LigGPT and DeepChem lineage uses;
2. **Randomisation** -- :func:`randomize_smiles`: the enumerator's own
   recipe, one call = one equivalent-but-different SMILES string;
3. **Enumeration** -- :func:`enumerate_smiles`: ``k`` seeded attempts,
   deduplicated, with the canonical form, its tokens and the per-molecule
   charset alongside -- the string half of ``fit()`` without the tensors,
   returned as a :class:`SmilesAugmentationReport` (``as_frame`` /
   ``__str__``).

Determinism: the vendored ``randomize_smiles`` drew its shuffle from the
process-global ``np.random``; every entry point here threads a *local*
``random.Random`` seeded from ``random_state`` (default 42, the toolkit
convention) instead -- the global RNG is never touched, and the same seed
reproduces the same augmentation list.  Dependencies: RDKit (lazily
imported, the ``[chem]`` extra) and pandas for the report frame; numpy
appears in the lineage only as the substrate of the one-hot matrices and
the global shuffle -- the tensors did not land and the shuffle is
re-seeded through ``random.Random`` -- so this surface imports neither
numpy nor torch.  Tokenisation is pure regex and runs without RDKit.

Example
-------
>>> from pharmaml.chem.smiles_enumeration import (
...     SMILES_TOKEN_PATTERN, enumerate_smiles, randomize_smiles, tokenize_smiles)

>>> tokenize_smiles("CC(=O)Oc1ccccc1C(=O)O")  # doctest: +NORMALIZE_WHITESPACE
['C', 'C', '(', '=', 'O', ')', 'O', 'c', '1', 'c', 'c', 'c', 'c', 'c', '1',
 'C', '(', '=', 'O', ')', 'O']
>>> tokenize_smiles("C[C@H](O)C(=O)O")        # bracket atoms stay ONE token
['C', '[C@H]', '(', 'O', ')', 'C', '(', '=', 'O', ')', 'O']
>>> import re
>>> re.findall(SMILES_TOKEN_PATTERN, "CC(=O)N") == tokenize_smiles("CC(=O)N")
True

>>> report = enumerate_smiles("CC(=O)Oc1ccccc1C(=O)O", k=8, random_state=42)
>>> report.n_unique_augmentations >= 3
True
>>> from rdkit import Chem
>>> all(Chem.MolToSmiles(Chem.MolFromSmiles(s)) == report.canonical
...     for s in report.augmentations)             # the equivalence invariant
True
>>> report.random_state
42
>>> again = enumerate_smiles("CC(=O)Oc1ccccc1C(=O)O", k=8, random_state=42)
>>> again.augmentations == report.augmentations   # same seed, same strings
True

>>> tiny = enumerate_smiles("CCO", k=8, random_state=42)  # the small-molecule
>>> tiny.n_unique_augmentations <= 8                      # ceiling: < k exist
True
>>> len(tiny.as_frame()) == len(tiny.augmentations)
True

>>> import random
>>> randomize_smiles("not_a_smiles", random.Random(42)) is None
True
>>> enumerate_smiles("not_a_smiles") is None
True
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger

__all__ = [
    "SMILES_TOKEN_PATTERN",
    "SmilesAugmentationReport",
    "enumerate_smiles",
    "randomize_smiles",
    "tokenize_smiles",
]

_LOG = get_logger(__name__)

#: RDKit import hint for the randomisation surfaces (tokenisation needs none).
_RDKIT_HINT = (
    "This surface requires RDKit. From the toolkit package root run: "
    "python -m pip install -e '.[chem]'"
)

#: The canonical character-level SMILES tokenisation regex of the
#: molgpt/LigGPT/DeepChem lineage (compiled verbatim in ``crt/dataset.py``):
#: a bracket expression (``[C@H]``, ``[N+]``, ``[13C]`` ...) as ONE token,
#: then the organic-subset atoms, the two-letter halogens, every bond /
#: branch / ring punctuation mark, ``%nn`` ring closures, digits -- and the
#: ``<`` / ``>`` marks, so molgpt's ``<`` pad token tokenises like any atom.
#: Split across two raw literals only to respect the 100-column limit; the
#: concatenated value is the exact upstream pattern.
SMILES_TOKEN_PATTERN: str = (
    r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|"
    r"\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

_SMILES_TOKEN_REGEX = re.compile(SMILES_TOKEN_PATTERN)


def _require_rdkit() -> Any:
    """Import and return ``rdkit.Chem``, or raise the typed error."""
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - env-dependent branch
        raise DependencyError(_RDKIT_HINT) from exc
    return Chem


# --------------------------------------------------------------------------- #
# tokenisation (pure regex -- no RDKit involved)
# --------------------------------------------------------------------------- #
def tokenize_smiles(smiles: str) -> list[str]:
    """Split one SMILES string into molgpt/LigGPT tokens.

    ``re.findall`` with :data:`SMILES_TOKEN_PATTERN`: atoms, bracket
    expressions, bond symbols, branch / ring punctuation and charges come
    out as separate tokens -- bracket atoms (``[C@H]``, ``[N+]``,
    ``[O-]``) never split into their characters.  This is the tokenisation
    ``crt/dataset.py`` feeds its character-level vocabulary (it pads with
    ``<`` tokens, which the pattern matches like any atom).  Pure regex:
    no RDKit, no validation -- garbage in, garbage tokenised out.

    Example
    -------
    >>> tokenize_smiles("CCO")
    ['C', 'C', 'O']
    >>> "[C@H]" in tokenize_smiles("C[C@H](O)C(=O)O")   # bracket atom: ONE token
    True
    >>> tokenize_smiles("<CCO>")                  # '<' is the molgpt pad token
    ['<', 'C', 'C', 'O', '>']
    """
    return [str(token) for token in _SMILES_TOKEN_REGEX.findall(smiles)]


# --------------------------------------------------------------------------- #
# randomisation (the SmilesEnumerator recipe)
# --------------------------------------------------------------------------- #
def randomize_smiles(smiles: str, rng: random.Random) -> str | None:
    """One randomised-but-equivalent SMILES string (the enumerator's recipe).

    Exactly ``SmilesEnumerator.randomize_smiles`` from the vendored
    ``crt/utils.py``: parse, shuffle the atom order with the caller's
    generator, ``Chem.RenumberAtoms``, then re-serialise with
    ``MolToSmiles(canonical=False, isomericSmiles=True)`` -- stereochemistry
    kept, canonical ordering dropped, so the result is a *different string
    for the same molecule*.  ``None`` when ``smiles`` is empty or does not
    parse (the lineage's ``get_mol`` contract).

    Parameters
    ----------
    smiles:
        The SMILES string to randomise.
    rng:
        The caller's ``random.Random`` instance -- the shuffle draws from
        THIS generator (the vendored original mutated the process-global
        ``np.random``), so the caller owns reproducibility.

    Example
    -------
    >>> from rdkit import Chem
    >>> text = randomize_smiles("CCO", random.Random(42))
    >>> Chem.MolToSmiles(Chem.MolFromSmiles(text))    # same molecule back
    'CCO'
    >>> randomize_smiles("not_a_smiles", random.Random(42)) is None
    True
    """
    Chem = _require_rdkit()
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    order = list(range(mol.GetNumAtoms()))
    rng.shuffle(order)
    return Chem.MolToSmiles(
        Chem.RenumberAtoms(mol, order), canonical=False, isomericSmiles=True
    )


# --------------------------------------------------------------------------- #
# the report record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SmilesAugmentationReport:
    """One molecule's augmentation report: tokens, charset, k randomisations.

    The string half of the vendored ``SmilesEnumerator`` in one immutable
    record -- what an augmentation workflow (or the GUI preview table)
    needs per molecule, without the one-hot tensors that did not land.

    Attributes
    ----------
    source:
        The input SMILES exactly as given (before canonicalisation).
    canonical:
        RDKit's canonical SMILES of the molecule -- every augmentation
        re-canonicalises to this (the equivalence invariant).
    tokens:
        ``tokenize_smiles(canonical)`` -- the molgpt/LigGPT token sequence.
    n_tokens:
        ``len(tokens)`` -- the padding-relevant length of this molecule.
    charset:
        Sorted unique tokens of THIS molecule (the per-molecule slice of
        the vocabulary; ``SmilesEnumerator.fit`` derived the same thing
        per dataset).
    augmentations:
        The distinct randomised SMILES the k attempts produced, in first
        occurrence order -- INCLUDING the canonical form whenever a random
        atom order happens to re-serialise to it (nothing is excluded; for
        ``"CCO"`` one of the strings IS ``"CCO"``).
    n_unique_augmentations:
        ``len(augmentations)`` -- can be shorter than ``k``: tiny
        molecules have only a handful of atom orderings, so k attempts
        simply exhaust them (the small-molecule ceiling).
    random_state:
        The seed the augmentations were drawn with (kept for provenance:
        the same seed and k reproduce the list exactly).

    Example
    -------
    >>> report = enumerate_smiles("CCO", k=8, random_state=42)
    >>> report.canonical, report.n_tokens, report.random_state
    ('CCO', 3, 42)
    >>> report.charset
    ['C', 'O']
    >>> str(report).startswith("3 tokens")
    True
    >>> frame = report.as_frame()
    >>> len(frame) == len(report.augmentations)
    True
    >>> list(frame.columns)
    ['index', 'smiles', 'length']
    """

    source: str
    canonical: str
    tokens: list[str]
    n_tokens: int
    charset: list[str]
    augmentations: list[str]
    n_unique_augmentations: int
    random_state: int

    def as_frame(self) -> pd.DataFrame:
        """The augmentations as a tidy DataFrame -- one row per string.

        Columns ``index`` (position in ``augmentations``), ``smiles`` (the
        randomised string) and ``length`` (its character count): the rows a
        preview table or a character-level trainer's string feed wants.
        """
        return pd.DataFrame(
            {
                "index": list(range(len(self.augmentations))),
                "smiles": list(self.augmentations),
                "length": [len(text) for text in self.augmentations],
            }
        )

    def __str__(self) -> str:
        """One line: token count, unique-augmentation count, one example."""
        example = self.augmentations[0] if self.augmentations else self.canonical
        return (
            f"{self.n_tokens} tokens · {self.n_unique_augmentations} unique "
            f"randomised SMILES · e.g. {example}"
        )


# --------------------------------------------------------------------------- #
# enumeration (k seeded attempts, deduplicated)
# --------------------------------------------------------------------------- #
def enumerate_smiles(
    smiles: str,
    k: int = 8,
    random_state: int = 42,
) -> SmilesAugmentationReport | None:
    """k seeded attempts at a randomised SMILES, deduplicated, with tokens.

    Builds a local ``random.Random(random_state)`` and calls
    :func:`randomize_smiles` ``k`` times through it, keeping every DISTINCT
    string produced -- the canonical form included whenever a random atom
    order re-serialises to it, nothing excluded.  The report carries the
    canonical form, its molgpt tokens and the per-molecule charset
    alongside, so one call yields both the augmented strings and the
    vocabulary view of the molecule.

    The small-molecule ceiling: a molecule admits only so many distinct
    serialisations under the renumber-and-serialise recipe, so ``k``
    attempts can exhaust them -- ``"CCO"`` yields only a handful of
    distinct strings no matter how large ``k`` is, and the report says so
    honestly via ``n_unique_augmentations < k``.  Returns ``None`` when
    ``smiles`` is empty or does not parse.

    Parameters
    ----------
    smiles:
        The molecule to augment.
    k:
        Randomisation attempts (>= 0; the deduplicated result can be
        shorter -- see the ceiling above).
    random_state:
        Seed for the local ``random.Random`` generator (the process-global
        RNG is never touched; 42 is the toolkit convention).

    Raises
    ------
    DataValidationError
        If ``k`` is negative.

    Example
    -------
    >>> report = enumerate_smiles("CCO", k=8, random_state=42)
    >>> 1 <= report.n_unique_augmentations <= 8   # the ceiling in action
    True
    >>> report.canonical in report.augmentations  # canonical is reachable
    True
    >>> enumerate_smiles("not_a_smiles") is None
    True
    """
    Chem = _require_rdkit()
    if k < 0:
        raise DataValidationError(f"k must be >= 0, got {k}")
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        _LOG.debug("SMILES enumeration skipped — %r does not parse", smiles)
        return None
    canonical = Chem.MolToSmiles(mol)
    tokens = tokenize_smiles(canonical)
    rng = random.Random(random_state)
    seen: dict[str, None] = {}
    for _ in range(k):
        randomized = randomize_smiles(smiles, rng)
        if randomized is not None:
            seen.setdefault(randomized, None)
    augmentations = list(seen)
    _LOG.debug(
        "SMILES enumeration: %d distinct augmentations of %r (k=%d, seed=%d)",
        len(augmentations),
        smiles,
        k,
        random_state,
    )
    return SmilesAugmentationReport(
        source=smiles,
        canonical=canonical,
        tokens=tokens,
        n_tokens=len(tokens),
        charset=sorted(set(tokens)),
        augmentations=augmentations,
        n_unique_augmentations=len(augmentations),
        random_state=random_state,
    )
