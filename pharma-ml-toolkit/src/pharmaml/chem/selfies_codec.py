"""SELFIES: the robust molecular string representation, as a toolkit surface.

SELFIES (SELF-referencIng Embedded Strings, Krenn et al. *Mach. Learn.: Sci.
Technol.* **1** 045024, 2020; ``the-matter-lab/selfies`` v2.x, Apache-2.0)
answers one question SMILES cannot: **is there a string representation of a
molecule in which EVERY string is a valid molecule?**  The answer is the
SELFIES grammar -- a state machine whose rules clamp every bond, branch and
ring to the remaining bonding capacity, so any sequence of SELFIES symbols
(a *semantic robust alphabet*, 69 symbols under the default constraints)
decodes to a chemically valid structure.  Random corruption never breaks
the molecule; it only changes it.

This module is the toolkit's typed surface over the ``selfies`` package
(optional dependency -- ``pip install 'pharmaml[selfies]'``; the package
itself is pure-Python stdlib, zero transitive dependencies):

1. **Codec** -- :func:`smiles_to_selfies` / :func:`selfies_to_smiles`
   (single molecules, typed errors), :func:`batch_smiles_to_selfies`
   (a whole column, per-row error log), :func:`selfies_len` /
   :func:`selfies_tokens` (the symbol-level view of a string);
2. **Audit** -- :func:`selfies_roundtrip_report` runs the repo's own QA
   methodology (encode → decode → RDKit canonical-SMILES equality, the
   pattern of ``tests/test_on_datasets.py``) over a whole dataset;
3. **Robustness** -- :func:`selfies_robustness_experiment` reproduces the
   paper's Figure 3: mutate k random symbols (SELFIES) vs k random
   characters (SMILES) and measure the fraction that still parses;
4. **Vocabulary** -- :func:`selfies_alphabet` /
   :func:`selfies_symbol_frequencies` (dataset vocabularies for language
   models), :func:`random_selfies_molecules` (sample uniformly over the
   robust alphabet -- every sample is valid by construction);
5. **ML features** -- :func:`selfies_feature_matrix` turns a SMILES column
   into a label-encoded or flattened one-hot numpy matrix via the
   package's own ``selfies_to_encoding`` / ``batch_selfies_to_flat_hot``
   helpers (the featurisation of the SELFIES VAE literature);
6. **Generation** -- :func:`selfies_mutants` enumerates a molecule's
   SELFIES-symbol neighbourhood (STONED-style point mutants): every
   offspring is valid by construction, the property the SMILES character
   muter of the genetic optimiser can only approximate;
7. **Constraints** -- :func:`semantic_constraints` is a save-and-restore
   *context manager* over the package's **process-global** constraint
   table (``set_semantic_constraints`` mutates interpreter-wide state; a
   bare call would silently change every later decode in the process --
   the context manager puts the previous table back, even on exceptions).

Dependencies: the ``selfies`` package (lazily imported; importing this
module needs nothing).  RDKit is required only by the audit / robustness /
feature halves -- the codec, vocabulary and constraint surfaces run
without it.

Example
-------
>>> from pharmaml.chem.selfies_codec import (
...     smiles_to_selfies, selfies_to_smiles, selfies_tokens)
>>> smiles_to_selfies("CCO")            # ethanol as three symbols
'[C][C][O]'
>>> selfies_to_smiles("[C][C][O]")
'CCO'
>>> selfies_tokens(smiles_to_selfies("CC(=O)Oc1ccccc1C(=O)O"))[:4]
['[C]', '[C]', '[=Branch1]', '[C]']

>>> from pharmaml.chem.selfies_codec import batch_smiles_to_selfies
>>> report = batch_smiles_to_selfies(["CCO", "CCN", "C1CC"])  # C1CC: dangling ring
>>> report.n_input, report.n_encoded
(3, 2)
>>> report.errors[0][0], report.errors[0][1]                  # (index, reason)
(2, 'C1CC')

>>> from pharmaml.chem.selfies_codec import selfies_mutants
>>> mutants = selfies_mutants("CCO", n=8, seed=42)   # every one valid
>>> from rdkit import Chem
>>> all(Chem.MolFromSmiles(s) is not None for s in mutants)
True
"""

from __future__ import annotations

import contextlib
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger

__all__ = [
    "SelfiesBatchConversion",
    "SelfiesRobustnessReport",
    "SelfiesRoundtripReport",
    "batch_smiles_to_selfies",
    "random_selfies_molecules",
    "selfies_alphabet",
    "selfies_available",
    "selfies_feature_matrix",
    "selfies_len",
    "selfies_mutants",
    "selfies_robustness_experiment",
    "selfies_roundtrip_report",
    "selfies_symbol_frequencies",
    "selfies_to_smiles",
    "selfies_tokens",
    "semantic_constraint_presets",
    "semantic_constraints",
    "smiles_to_selfies",
]

_LOG = get_logger(__name__)

#: pip command embedded in the DependencyError raised when the optional
#: ``selfies`` package is missing (the package is pure-Python, so the
#: install is always fast and dependency-free).
_SELFIES_HINT = (
    "SELFIES support requires the selfies package — "
    "pip install 'pharmaml[selfies]' (or pip install selfies)"
)

#: RDKit import hint for the audit / robustness halves.
_RDKIT_HINT = (
    "this surface requires RDKit — pip install 'pharmaml[chem]' "
    "(or pip install rdkit)"
)

#: Maximum number of error / mismatch examples kept in the typed reports
#: (the full lists are reconstructed from the input on demand; the report
#: carries a bounded, displayable sample).
_MAX_EXAMPLES = 5


def selfies_available() -> bool:
    """Report whether the optional ``selfies`` package is importable.

    The check uses :func:`importlib.util.find_spec` -- no actual import,
    so probing never pays the package's import cost (and never fails on
    a broken installation).
    """
    import importlib.util

    return importlib.util.find_spec("selfies") is not None


def _require_selfies() -> ModuleType:
    """Import and return the ``selfies`` module, or raise the typed error."""
    try:
        import selfies
    except ImportError as exc:  # pragma: no cover - env-dependent branch
        raise DependencyError(_SELFIES_HINT) from exc
    return selfies


def _require_rdkit() -> Any:
    """Import and return ``rdkit.Chem``, or raise the typed error."""
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - env-dependent branch
        raise DependencyError(_RDKIT_HINT) from exc
    return Chem


# --------------------------------------------------------------------------- #
# codec: single molecules
# --------------------------------------------------------------------------- #
def smiles_to_selfies(smiles: str) -> str:
    """Translate one SMILES string to its SELFIES encoding.

    Raises
    ------
    DataValidationError
        On empty input, or when the SMILES cannot be encoded (wildcard
        ``*`` atoms, exotic chirality, unkekulizable aromatics, valence
        beyond the active semantic constraints) -- the package's
        ``EncoderError`` text is preserved verbatim.

    Example
    -------
    >>> smiles_to_selfies("CCO")
    '[C][C][O]'
    >>> smiles_to_selfies("")           # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    pharmaml.exceptions.DataValidationError: ...
    """
    selfies = _require_selfies()
    if not isinstance(smiles, str) or not smiles.strip():
        raise DataValidationError(
            "smiles_to_selfies needs a non-empty SMILES string "
            f"(got {smiles!r})"
        )
    try:
        return str(selfies.encoder(smiles))
    except selfies.EncoderError as exc:
        raise DataValidationError(f"SELFIES encoder rejected {smiles!r}: {exc}") from exc


def selfies_to_smiles(selfies_str: str) -> str:
    """Translate one SELFIES string back to SMILES.

    The output is guaranteed chemically valid under the *active* semantic
    constraints (see :func:`semantic_constraints`).

    Raises
    ------
    DataValidationError
        On empty input or a malformed SELFIES string (hanging bracket,
        unrecognised symbol) -- the package's ``DecoderError`` text is
        preserved verbatim.

    Example
    -------
    >>> selfies_to_smiles("[C][C][O]")
    'CCO'
    """
    selfies = _require_selfies()
    if not isinstance(selfies_str, str) or not selfies_str.strip():
        raise DataValidationError(
            "selfies_to_smiles needs a non-empty SELFIES string "
            f"(got {selfies_str!r})"
        )
    try:
        return str(selfies.decoder(selfies_str))
    except selfies.DecoderError as exc:
        raise DataValidationError(
            f"SELFIES decoder rejected {selfies_str!r}: {exc}"
        ) from exc


def selfies_len(selfies_str: str) -> int:
    """Number of SELFIES symbols in a string (no full parse).

    Example
    -------
    >>> selfies_len("[C][C][O]")
    3
    """
    return int(_require_selfies().len_selfies(selfies_str))


def selfies_tokens(selfies_str: str) -> list[str]:
    """Split one SELFIES string into its symbols (the tokenizer).

    Example
    -------
    >>> selfies_tokens("[C][C][O]")
    ['[C]', '[C]', '[O]']
    """
    selfies = _require_selfies()
    return [str(token) for token in selfies.split_selfies(selfies_str)]


# --------------------------------------------------------------------------- #
# codec: batch conversion
# --------------------------------------------------------------------------- #
class SelfiesBatchConversion:
    """Result of :func:`batch_smiles_to_selfies` -- per-row, never row-destructive.

    Attributes
    ----------
    selfies:
        One entry per input row: the SELFIES string, or ``None`` where the
        row failed to encode (the row is preserved so indexes stay aligned
        with the input).
    errors:
        ``(index, smiles, message)`` for every failed row, input order.

    Example
    -------
    >>> report = batch_smiles_to_selfies(["CCO", "C1CC"])
    >>> report.n_input, report.n_encoded, report.n_failed
    (2, 1, 1)
    >>> report.errors[0][0]
    1
    """

    __slots__ = ("_errors", "_selfies")

    def __init__(
        self,
        selfies_list: Sequence[str | None],
        errors: Sequence[tuple[int, str, str]],
    ) -> None:
        self._selfies = list(selfies_list)
        self._errors = [(int(i), str(s), str(m)) for i, s, m in errors]

    @property
    def selfies(self) -> list[str | None]:
        """The encoded column (``None`` where the row failed)."""
        return list(self._selfies)

    @property
    def errors(self) -> list[tuple[int, str, str]]:
        """``(index, smiles, message)`` per failed row."""
        return list(self._errors)

    @property
    def n_input(self) -> int:
        """Number of input rows."""
        return len(self._selfies)

    @property
    def n_encoded(self) -> int:
        """Rows successfully encoded."""
        return sum(1 for value in self._selfies if value is not None)

    @property
    def n_failed(self) -> int:
        """Rows the encoder rejected."""
        return len(self._errors)

    def summary(self) -> str:
        """One-line human summary for report cards."""
        return (
            f"{self.n_encoded}/{self.n_input} rows encoded "
            f"({self.n_failed} failed)"
        )

    def __repr__(self) -> str:  # pragma: no cover - debug nicety
        return (
            f"SelfiesBatchConversion(n_input={self.n_input}, "
            f"n_encoded={self.n_encoded})"
        )


def batch_smiles_to_selfies(smiles_list: Sequence[str]) -> SelfiesBatchConversion:
    """Encode a whole SMILES column, logging per-row failures.

    The SELFIES repo's own QA (``tests/test_on_datasets.py``) converts
    molecule datasets exactly this way; this surface keeps every row --
    a failed row becomes ``None`` plus an error record, never a dropped
    row, so the output stays index-aligned with the input for DataFrame
    assembly.

    Example
    -------
    >>> report = batch_smiles_to_selfies(["CCO", "CCN"])
    >>> report.selfies
    ['[C][C][O]', '[C][C][N]']
    >>> report.n_failed
    0
    """
    selfies = _require_selfies()
    encoded: list[str | None] = []
    errors: list[tuple[int, str, str]] = []
    for index, smiles in enumerate(smiles_list):
        text = smiles if isinstance(smiles, str) else str(smiles)
        try:
            encoded.append(str(selfies.encoder(text)))
        except selfies.EncoderError as exc:
            encoded.append(None)
            errors.append((index, text, str(exc)))
    if errors:
        _LOG.info(
            "SELFIES batch encode: %d/%d rows failed (first: %r)",
            len(errors),
            len(smiles_list),
            errors[0][1],
        )
    return SelfiesBatchConversion(encoded, errors)


# --------------------------------------------------------------------------- #
# vocabulary surfaces
# --------------------------------------------------------------------------- #
def selfies_alphabet(selfies_iter: Iterable[str]) -> list[str]:
    """Sorted set of SELFIES symbols appearing in an iterable of strings.

    Example
    -------
    >>> selfies_alphabet(["[C][C][O]", "[C][C][N]"])
    ['[C]', '[N]', '[O]']
    """
    selfies = _require_selfies()
    alphabet = selfies.get_alphabet_from_selfies(
        s for s in selfies_iter if isinstance(s, str) and s
    )
    return sorted(str(symbol) for symbol in alphabet)


def selfies_symbol_frequencies(selfies_iter: Iterable[str]) -> dict[str, int]:
    """Symbol → occurrence count over an iterable of SELFIES strings.

    The histogram behind language-model vocabulary design: which atoms,
    branches and ring closures actually carry the dataset's chemistry.

    Example
    -------
    >>> selfies_symbol_frequencies(["[C][C][O]", "[C][N]"])
    {'[C]': 3, '[N]': 1, '[O]': 1}
    """
    selfies = _require_selfies()
    counter: Counter[str] = Counter()
    for text in selfies_iter:
        if not isinstance(text, str) or not text:
            continue
        counter.update(str(token) for token in selfies.split_selfies(text))
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def random_selfies_molecules(
    n: int,
    length: int,
    *,
    alphabet: Sequence[str] | None = None,
    seed: int = 42,
) -> list[str]:
    """Sample ``n`` molecules uniformly from the robust SELFIES alphabet.

    Every sample decodes to a chemically valid molecule *by construction*
    (the grammar clamps bonds, branches and rings to the remaining
    capacities) -- the property the README calls the recipe for a random
    molecule generator.  Chemistry-free but structure-valid: the samples
    are uniform over *strings*, not over chemical space.

    Parameters
    ----------
    n:
        Number of strings to sample (must be ≥ 0).
    length:
        Symbols per string (must be ≥ 1).
    alphabet:
        Symbols to draw from; ``None`` = the full semantic robust
        alphabet of the active constraints (69 symbols by default).
    seed:
        Seed for the local ``random.Random`` generator (the process-global
        RNG is never touched).

    Example
    -------
    >>> mols = random_selfies_molecules(4, 6, seed=42)
    >>> len(mols), len(set(mols)) >= 1
    (4, True)
    >>> from rdkit import Chem
    >>> all(Chem.MolFromSmiles(s) is not None for s in mols)
    True
    """
    selfies = _require_selfies()
    if n < 0:
        raise DataValidationError(f"n must be >= 0, got {n}")
    if length < 1:
        raise DataValidationError(f"length must be >= 1, got {length}")
    # sorted() on BOTH branches: get_semantic_robust_alphabet() returns a
    # set, and iterating a raw set rides PYTHONHASHSEED -- the same seed
    # would sample different symbols in every process (the flaky-gate
    # bug this line fixes; a seeded sampler owes cross-process determinism).
    symbols = sorted(
        str(symbol) for symbol in alphabet
    ) if alphabet else sorted(
        str(symbol) for symbol in selfies.get_semantic_robust_alphabet()
    )
    if not symbols:
        raise DataValidationError("the alphabet is empty — nothing to draw from")
    rng = random.Random(seed)
    molecules: list[str] = []
    for _ in range(n):
        text = "".join(rng.choice(symbols) for _ in range(length))
        molecules.append(str(selfies.decoder(text)))
    return molecules


def selfies_mutants(
    smiles: str,
    n: int,
    *,
    mutations: int = 1,
    seed: int = 42,
) -> list[str]:
    """Enumerate a molecule's SELFIES-symbol point mutants (STONED-style).

    Encode the molecule, replace ``mutations`` random symbols with random
    symbols from the semantic robust alphabet, decode -- every offspring
    is a valid molecule by construction (the robustness property the
    SMILES character muter of :class:`~pharmaml.generative.GeneticOptimizer`
    can only approximate statistically).  Duplicates are removed; the
    original molecule is excluded.

    Parameters
    ----------
    smiles:
        The parent molecule.
    n:
        Number of mutant attempts (the returned list can be shorter after
        deduplication).
    mutations:
        Symbols replaced per mutant (default 1).
    seed:
        Seed for the local ``random.Random`` generator.

    Example
    -------
    >>> mutants = selfies_mutants("CCO", n=8, mutations=1, seed=42)
    >>> mutants and mutants != ["CCO"]
    True
    >>> from rdkit import Chem
    >>> all(Chem.MolFromSmiles(s) is not None for s in mutants)
    True
    """
    selfies = _require_selfies()
    if n < 0:
        raise DataValidationError(f"n must be >= 0, got {n}")
    if mutations < 1:
        raise DataValidationError(f"mutations must be >= 1, got {mutations}")
    encoded = smiles_to_selfies(smiles)
    tokens = selfies_tokens(encoded)
    if not tokens:
        raise DataValidationError(f"{smiles!r} encoded to an empty SELFIES string")
    alphabet = sorted(str(symbol) for symbol in selfies.get_semantic_robust_alphabet())
    rng = random.Random(seed)
    seen: dict[str, None] = {}
    for _ in range(n):
        mutant = list(tokens)
        for position in rng.sample(range(len(mutant)), min(mutations, len(mutant))):
            mutant[position] = rng.choice(alphabet)
        try:
            decoded = str(selfies.decoder("".join(mutant)))
        except selfies.DecoderError:  # pragma: no cover - grammar is robust
            continue
        if decoded != smiles:
            seen.setdefault(decoded, None)
    return list(seen)


# --------------------------------------------------------------------------- #
# audit: round-trip fidelity
# --------------------------------------------------------------------------- #
class SelfiesRoundtripReport:
    """Round-trip fidelity audit of a SMILES column, the repo's own QA pattern.

    Attributes
    ----------
    n_input / n_parsed / n_encoded / n_decoded / n_canonical_match:
        The funnel: input rows → RDKit-parseable → SELFIES-encodable →
        decodable back to parseable SMILES → canonical-SMILES-equal.
    fidelity:
        ``n_canonical_match / n_input`` (1.0 = perfect round-trip).
    alphabet_size / max_selfies_len / mean_selfies_len:
        Vocabulary and length statistics of the encoded column.
    encode_errors:
        ``(index, smiles, message)`` for encoder rejections (bounded).
    mismatch_examples:
        ``(original, decoded)`` pairs whose canonical SMILES differ
        (bounded; usually stereo or kekulization-order differences).

    Example
    -------
    >>> report = selfies_roundtrip_report(["CCO", "CCN", "c1ccccc1"])
    >>> report.n_input, report.n_canonical_match, round(report.fidelity, 2)
    (3, 3, 1.0)
    >>> report.alphabet_size > 5
    True
    """

    __slots__ = (
        "_alphabet_size",
        "_encode_errors",
        "_max_len",
        "_mean_len",
        "_mismatches",
        "_n_decoded",
        "_n_encoded",
        "_n_input",
        "_n_match",
        "_n_parsed",
    )

    def __init__(
        self,
        *,
        n_input: int,
        n_parsed: int,
        n_encoded: int,
        n_decoded: int,
        n_match: int,
        alphabet_size: int,
        max_len: int,
        mean_len: float,
        encode_errors: Sequence[tuple[int, str, str]],
        mismatches: Sequence[tuple[str, str]],
    ) -> None:
        self._n_input = int(n_input)
        self._n_parsed = int(n_parsed)
        self._n_encoded = int(n_encoded)
        self._n_decoded = int(n_decoded)
        self._n_match = int(n_match)
        self._alphabet_size = int(alphabet_size)
        self._max_len = int(max_len)
        self._mean_len = float(mean_len)
        self._encode_errors = [
            (int(i), str(s), str(m)) for i, s, m in encode_errors[:_MAX_EXAMPLES]
        ]
        self._mismatches = [
            (str(a), str(b)) for a, b in mismatches[:_MAX_EXAMPLES]
        ]

    @property
    def n_input(self) -> int:
        return self._n_input

    @property
    def n_parsed(self) -> int:
        return self._n_parsed

    @property
    def n_encoded(self) -> int:
        return self._n_encoded

    @property
    def n_decoded(self) -> int:
        return self._n_decoded

    @property
    def n_canonical_match(self) -> int:
        return self._n_match

    @property
    def fidelity(self) -> float:
        """``n_canonical_match / n_input`` (0.0 on empty input)."""
        return self._n_match / self._n_input if self._n_input else 0.0

    @property
    def alphabet_size(self) -> int:
        return self._alphabet_size

    @property
    def max_selfies_len(self) -> int:
        return self._max_len

    @property
    def mean_selfies_len(self) -> float:
        return self._mean_len

    @property
    def encode_errors(self) -> list[tuple[int, str, str]]:
        return list(self._encode_errors)

    @property
    def mismatch_examples(self) -> list[tuple[str, str]]:
        return list(self._mismatches)

    def summary(self) -> str:
        """Multi-line summary for report cards."""
        lines = [
            (f"round-trip fidelity: {self._n_match}/{self._n_input} "
            f"({self.fidelity:.1%})"),
            (f"funnel: {self._n_parsed} parsed → {self._n_encoded} encoded "
            f"→ {self._n_decoded} decoded → {self._n_match} canonical-equal"),
            (f"vocabulary: {self._alphabet_size} symbols · length "
            f"max {self._max_len}, mean {self._mean_len:.1f}"),
        ]
        if self._encode_errors:
            index, smiles, message = self._encode_errors[0]
            lines.append(
                f"encode errors: {len(self._encode_errors)} kept "
                f"(first: row {index} {smiles!r}: {message})"
            )
        if self._mismatches:
            original, decoded = self._mismatches[0]
            lines.append(
                f"mismatches: {len(self._mismatches)} kept "
                f"(first: {original!r} → {decoded!r})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debug nicety
        return (
            f"SelfiesRoundtripReport(n_input={self._n_input}, "
            f"fidelity={self.fidelity:.3f})"
        )


def selfies_roundtrip_report(
    smiles_list: Sequence[str],
    *,
    sample_limit: int | None = None,
    seed: int = 42,
) -> SelfiesRoundtripReport:
    """Audit a SMILES column: encode → decode → canonical-SMILES equality.

    The methodology is the SELFIES repo's own dataset QA
    (``tests/test_on_datasets.py``): every molecule is encoded to
    SELFIES, decoded back, and the canonical RDKit SMILES of both sides
    compared.  Rows are never dropped -- each failure mode lands in its
    own funnel stage (unparseable → unencodable → undecodable →
    mismatch).

    Parameters
    ----------
    smiles_list:
        The SMILES column to audit.
    sample_limit:
        Audit at most this many rows, drawn without replacement from the
        input by a seeded ``random.Random`` (``None`` = audit everything).

    Example
    -------
    >>> report = selfies_roundtrip_report(["CCO", "CCN"])
    >>> report.fidelity
    1.0
    """
    selfies = _require_selfies()
    Chem = _require_rdkit()
    rows = [s if isinstance(s, str) else str(s) for s in smiles_list]
    if sample_limit is not None and 0 <= sample_limit < len(rows):
        rows = random.Random(seed).sample(rows, sample_limit)

    n_parsed = n_encoded = n_decoded = n_match = 0
    encoded_ok: list[str] = []
    encode_errors: list[tuple[int, str, str]] = []
    mismatches: list[tuple[str, str]] = []
    for index, smiles in enumerate(rows):
        if not smiles.strip():
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        n_parsed += 1
        try:
            encoded = str(selfies.encoder(smiles))
        except selfies.EncoderError as exc:
            encode_errors.append((index, smiles, str(exc)))
            continue
        n_encoded += 1
        encoded_ok.append(encoded)
        decoded = str(selfies.decoder(encoded))
        decoded_mol = Chem.MolFromSmiles(decoded)
        if decoded_mol is None:  # pragma: no cover - grammar guarantees validity
            continue
        n_decoded += 1
        if Chem.MolToSmiles(mol) == Chem.MolToSmiles(decoded_mol):
            n_match += 1
        elif len(mismatches) < _MAX_EXAMPLES:
            mismatches.append((smiles, decoded))

    lengths = [selfies.len_selfies(text) for text in encoded_ok]
    alphabet = selfies_alphabet(encoded_ok) if encoded_ok else []
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    _LOG.info(
        "SELFIES round-trip audit: %d/%d canonical-equal", n_match, len(rows)
    )
    return SelfiesRoundtripReport(
        n_input=len(rows),
        n_parsed=n_parsed,
        n_encoded=n_encoded,
        n_decoded=n_decoded,
        n_match=n_match,
        alphabet_size=len(alphabet),
        max_len=max(lengths) if lengths else 0,
        mean_len=mean_len,
        encode_errors=encode_errors,
        mismatches=mismatches,
    )


# --------------------------------------------------------------------------- #
# robustness experiment (the paper's Figure 3)
# --------------------------------------------------------------------------- #
class SelfiesRobustnessReport:
    """Validity under random corruption: SELFIES symbols vs SMILES characters.

    Attributes
    ----------
    mutations:
        The corruption levels probed (k symbols / characters replaced).
    n_molecules / n_trials:
        Experiment size: molecules sampled from the input, trials per
        molecule per level.
    selfies_validity / smiles_validity:
        Fraction of corrupted strings that still parse as molecules, one
        entry per level, aligned with ``mutations``.

    Example
    -------
    >>> report = selfies_robustness_experiment(
    ...     ["c1ccccc1", "CCO"], mutations=(1, 4), trials=4, seed=42)
    >>> len(report.selfies_validity), len(report.smiles_validity)
    (2, 2)
    >>> report.selfies_validity[0] >= report.smiles_validity[0]
    True
    """

    __slots__ = ("_mutations", "_n_molecules", "_n_trials", "_selfies", "_smiles")

    def __init__(
        self,
        *,
        mutations: Sequence[int],
        n_molecules: int,
        n_trials: int,
        selfies_validity: Sequence[float],
        smiles_validity: Sequence[float],
    ) -> None:
        self._mutations = [int(m) for m in mutations]
        self._n_molecules = int(n_molecules)
        self._n_trials = int(n_trials)
        self._selfies = [float(v) for v in selfies_validity]
        self._smiles = [float(v) for v in smiles_validity]

    @property
    def mutations(self) -> list[int]:
        return list(self._mutations)

    @property
    def n_molecules(self) -> int:
        return self._n_molecules

    @property
    def n_trials(self) -> int:
        return self._n_trials

    @property
    def selfies_validity(self) -> list[float]:
        return list(self._selfies)

    @property
    def smiles_validity(self) -> list[float]:
        return list(self._smiles)

    def as_frame(self) -> pd.DataFrame:
        """The experiment as a tidy DataFrame (level × representation)."""
        return pd.DataFrame(
            {
                "mutations": self._mutations * 2,
                "representation": ["SELFIES"] * len(self._mutations)
                + ["SMILES"] * len(self._mutations),
                "validity": self._selfies + self._smiles,
            }
        )

    def summary(self) -> str:
        """Multi-line summary for report cards."""
        lines = [
            (f"validity under corruption ({self._n_molecules} molecules × "
            f"{self._n_trials} trials per level):")
        ]
        for level, sf_valid, sm_valid in zip(
            self._mutations, self._selfies, self._smiles
        ):
            lines.append(
                f"  k={level}: SELFIES {sf_valid:.0%} · SMILES {sm_valid:.0%}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debug nicety
        return (
            f"SelfiesRobustnessReport(levels={self._mutations}, "
            f"selfies={self._selfies})"
        )


def selfies_robustness_experiment(
    smiles_list: Sequence[str],
    *,
    mutations: Sequence[int] = (1, 2, 4, 8),
    trials: int = 8,
    sample_limit: int = 64,
    seed: int = 42,
) -> SelfiesRobustnessReport:
    """Reproduce the SELFIES paper's Figure 3 on your own molecules.

    For every encodable input molecule and every corruption level ``k``:
    replace ``k`` random SELFIES *symbols* (side) / ``k`` random SMILES
    *characters* (side) with uniformly drawn replacements, then measure
    the fraction that still parses as a molecule.  The SELFIES side keeps
    its ~100% validity at every level (the grammar clamps every mutation
    to a legal structure); the SMILES side collapses -- the empirical
    demonstration of why generative models over SELFIES need no validity
    filter.

    Parameters
    ----------
    smiles_list:
        The molecule pool (invalid rows are skipped, never crash the run).
    mutations:
        Corruption levels probed (each ≥ 1).
    trials:
        Trials per molecule per level.
    sample_limit:
        Cap on molecules sampled from the pool (seeded, without
        replacement).
    seed:
        Seed for the local ``random.Random`` generator.

    Example
    -------
    >>> report = selfies_robustness_experiment(
    ...     ["c1ccccc1", "CCO"], mutations=(1,), trials=6, seed=42)
    >>> report.selfies_validity[0] > report.smiles_validity[0]   # the point
    True
    """
    selfies = _require_selfies()
    Chem = _require_rdkit()
    levels = [int(m) for m in mutations]
    if not levels:
        raise DataValidationError("mutations must be a non-empty sequence")
    if any(level < 1 for level in levels):
        raise DataValidationError(f"mutation levels must be >= 1, got {levels}")
    if trials < 1:
        raise DataValidationError(f"trials must be >= 1, got {trials}")
    if sample_limit < 1:
        raise DataValidationError(f"sample_limit must be >= 1, got {sample_limit}")

    # the molecule pool: encodable rows only (both sides start from the
    # same molecules so the comparison is apples-to-apples)
    pool: list[str] = []
    for value in smiles_list:
        text = value if isinstance(value, str) else str(value)
        if not text.strip():
            continue
        try:
            selfies.encoder(text)
        except selfies.EncoderError:
            continue
        pool.append(text)
    if not pool:
        raise DataValidationError(
            "no encodable molecules in the input — the robustness "
            "experiment needs at least one SELFIES-encodable SMILES"
        )
    rng = random.Random(seed)
    if len(pool) > sample_limit:
        pool = rng.sample(pool, sample_limit)

    # the mutation alphabets: the full semantic robust alphabet on the
    # SELFIES side (the paper's setting — arbitrary symbols); the pool's
    # own observed charset on the SMILES side (its organic-subset sibling)
    alphabet = sorted(str(symbol) for symbol in selfies.get_semantic_robust_alphabet())
    charset = sorted(set("".join(pool))) or list("CNO")
    encoded = {smiles: selfies_tokens(str(selfies.encoder(smiles))) for smiles in pool}

    selfies_validity: list[float] = []
    smiles_validity: list[float] = []
    for level in levels:
        sf_ok = sm_ok = total = 0
        for smiles, tokens in encoded.items():
            for _ in range(trials):
                total += 1
                # SELFIES side: k random symbol substitutions
                mutant = list(tokens)
                for position in rng.sample(
                    range(len(mutant)), min(level, len(mutant))
                ):
                    mutant[position] = rng.choice(alphabet)
                decoded = str(selfies.decoder("".join(mutant)))
                if Chem.MolFromSmiles(decoded) is not None:
                    sf_ok += 1
                # SMILES side: k random character substitutions
                chars = list(smiles)
                for position in rng.sample(
                    range(len(chars)), min(level, len(chars))
                ):
                    replacement = rng.choice(charset)
                    chars[position] = replacement
                if Chem.MolFromSmiles("".join(chars)) is not None:
                    sm_ok += 1
        selfies_validity.append(sf_ok / total if total else 0.0)
        smiles_validity.append(sm_ok / total if total else 0.0)
    _LOG.info(
        "SELFIES robustness experiment: %d molecules, %d levels",
        len(pool),
        len(levels),
    )
    return SelfiesRobustnessReport(
        mutations=levels,
        n_molecules=len(pool),
        n_trials=trials,
        selfies_validity=selfies_validity,
        smiles_validity=smiles_validity,
    )


# --------------------------------------------------------------------------- #
# ML features
# --------------------------------------------------------------------------- #
def selfies_feature_matrix(
    smiles_list: Sequence[str],
    *,
    encoding: str = "label",
    pad_to_len: int | None = None,
) -> tuple[Any, list[str]]:
    """Featurise a SMILES column as a SELFIES label / one-hot matrix.

    The encoding of the SELFIES VAE literature (``selfies_to_encoding`` /
    ``batch_selfies_to_flat_hot``): the column is encoded to SELFIES, a
    vocabulary is built from the data (plus the ``[nop]`` padding
    symbol), and every molecule becomes a fixed-width row.

    Parameters
    ----------
    smiles_list:
        The SMILES column (every row must encode — audit and clean first
        with :func:`selfies_roundtrip_report`).
    encoding:
        ``"label"`` → ``(n, L)`` int matrix of vocabulary indexes;
        ``"one_hot"`` → ``(n, L * V)`` float matrix (flattened one-hots).
    pad_to_len:
        Row width in symbols (``None`` = the column's maximum).
    Returns
    -------
    ``(matrix, vocabulary)`` -- the numpy matrix and the ordered symbol
    list (row i of the label matrix indexes into it).

    Raises
    ------
    DataValidationError
        On an unknown ``encoding``, an unencodable row (the first
        offender is named), or an empty column.

    Example
    -------
    >>> matrix, vocab = selfies_feature_matrix(["CCO", "CCN"])
    >>> matrix.shape, len(vocab)
    ((2, 3), 4)
    >>> matrix[0].tolist()
    [0, 0, 2]
    """
    selfies = _require_selfies()
    if encoding not in ("label", "one_hot"):
        raise DataValidationError(
            f"encoding must be 'label' or 'one_hot', got {encoding!r}"
        )
    rows = [s if isinstance(s, str) else str(s) for s in smiles_list]
    if not rows:
        raise DataValidationError("the SMILES column is empty — nothing to featurise")
    encoded: list[str] = []
    for index, smiles in enumerate(rows):
        if not smiles.strip():
            raise DataValidationError(
                f"row {index} is empty — clean the column first "
                "(selfies_roundtrip_report audits it)"
            )
        try:
            encoded.append(str(selfies.encoder(smiles)))
        except selfies.EncoderError as exc:
            raise DataValidationError(
                f"row {index} ({smiles!r}) is not SELFIES-encodable: {exc} "
                "-- clean the column first (selfies_roundtrip_report audits it)"
            ) from exc

    vocab = sorted(selfies_alphabet(encoded) + ["[nop]"])
    stoi = {symbol: i for i, symbol in enumerate(vocab)}
    width = pad_to_len if pad_to_len is not None else max(
        selfies.len_selfies(text) for text in encoded
    )
    if width < 1:
        raise DataValidationError(f"pad_to_len must be >= 1, got {width}")

    if encoding == "label":
        matrix = np.array(
            [
                selfies.selfies_to_encoding(
                    text, stoi, pad_to_len=width, enc_type="label"
                )
                for text in encoded
            ],
            dtype=np.int64,
        )
        return matrix, vocab
    flat = np.array(
        selfies.batch_selfies_to_flat_hot(encoded, stoi, pad_to_len=width),
        dtype=np.float64,
    )
    return flat.reshape(len(encoded), width * len(vocab)), vocab


# --------------------------------------------------------------------------- #
# semantic constraints (process-global state, handled safely)
# --------------------------------------------------------------------------- #
def semantic_constraint_presets() -> list[str]:
    """The constraint presets the package ships.

    Example
    -------
    >>> semantic_constraint_presets()
    ['default', 'hypervalent', 'octet_rule']
    """
    return ["default", "hypervalent", "octet_rule"]


@contextlib.contextmanager
def semantic_constraints(
    constraints: str | dict[str, int],
) -> Iterator[None]:
    """Temporarily set the global SELFIES semantic constraints.

    ``set_semantic_constraints`` mutates **process-global** state -- every
    later ``decoder`` call in the interpreter uses the new table.  This
    context manager snapshots the active table, applies yours, and puts
    the snapshot back on exit (including on exceptions), so an
    experiment with ``"hypervalent"`` constraints or a restricted atom
    set can never silently leak into unrelated code.

    Parameters
    ----------
    constraints:
        A preset name (``"default"`` / ``"octet_rule"`` /
        ``"hypervalent"``) or a full ``{symbol: max_bonds}`` table (which
        must contain the ``"?"`` catch-all key -- the package raises
        otherwise).

    Example
    -------
    >>> from pharmaml.chem.selfies_codec import (
    ...     semantic_constraints, selfies_to_smiles)
    >>> selfies_to_smiles("[C][#C]")          # default: acetylene
    'C#C'
    >>> with semantic_constraints({"C": 1, "O": 1, "N": 1, "?": 1}):
    ...     selfies_to_smiles("[C][#C]")      # carbon capped at one bond
    'CC'
    >>> selfies_to_smiles("[C][#C]")          # restored -- nothing leaked
    'C#C'
    """
    selfies = _require_selfies()
    saved = selfies.get_semantic_constraints()
    try:
        selfies.set_semantic_constraints(constraints)
        yield
    finally:
        selfies.set_semantic_constraints(saved)
