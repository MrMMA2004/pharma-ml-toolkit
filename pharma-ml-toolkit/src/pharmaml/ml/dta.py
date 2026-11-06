"""Protein--ligand affinity features: DeepDTA encodings plus paired ECFP/AAC.

The catalog's drug--target interaction repositories (``hkmztrk/DeepDTA``,
``kexinhuang12345/DeepPurpose``, ``awslabs/dgl-lifesci`` and
``DeepGraphLearning/torchdrug``) train paired drug/target encoders.  Their
large torch/DGL checkpoints are deliberately not vendored.  This module
supplies the reusable data boundary in plain NumPy/Pandas:

* :func:`label_sequence` and :func:`one_hot_sequence` reproduce DeepDTA's
  ``source/datahelper.py`` exactly: its 25-symbol, 1-based protein alphabet,
  left-aligned truncation and zero padding;
* :func:`protein_feature_frame` adds 20-standard-amino-acid composition and
  optional 400-dipeptide frequencies, with non-standard residues counted
  explicitly instead of silently relabelled;
* :func:`paired_dta_features` aligns those target features with the toolkit's
  existing 2,048-bit ECFP ligand features.  Invalid molecules remain visible
  in ``dropped`` and cannot silently change row/affinity alignment;
* :func:`deepdta_rm2` ports DeepDTA's ``source/emetrics.py`` external-
  validation metric (squared correlation penalised by deviation from the
  zero-intercept fit).  Concordance and RMSE stay in ``ml.benchmetrics``.

This is a feature/metric fold-in, not a claim to reproduce DeepDTA's CNN,
DeepPurpose's transformer/CNN_RNN stack, DGL-LifeSci's graph models or
TorchDrug's protein language models.  The resulting ordinary feature matrix
can be trained by any regressor already in the toolkit.

Examples
--------
DeepDTA reserves zero for padding and truncates on the left side of the
padding (never left-pads the sequence itself):

>>> from pharmaml.ml.dta import label_sequence, one_hot_sequence
>>> label_sequence("ACX", max_length=5).tolist()
[1.0, 2.0, 24.0, 0.0, 0.0]
>>> one_hot_sequence("AC", max_length=3).shape
(3, 25)
>>> from pharmaml.ml.dta import label_smiles, one_hot_smiles
>>> label_smiles("C(C)O", max_length=6).tolist()
[42.0, 1.0, 42.0, 31.0, 48.0, 0.0]
>>> one_hot_smiles("CO", max_length=3, isomeric=False).shape
(3, 62)

Paired features preserve the surviving source-row positions:

>>> import pandas as pd
>>> from pharmaml.ml.dta import paired_dta_features
>>> pairs = pd.DataFrame({"smiles": ["CCO", "not-a-smiles", "CCN"],
...                       "protein": ["ACDE", "ACDE", "ACDF"],
...                       "affinity": [5.0, 6.0, 7.0]})
>>> features = paired_dta_features(pairs)
>>> features.kept_indices, len(features.dropped)
((0, 2), 1)
>>> features.X.shape
(2, 2071)
>>> features.y.tolist()
[5.0, 7.0]
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "AMINO_ACIDS",
    "DEEPDTA_CHARCANSMISET",
    "DEEPDTA_CHARISOSMISET",
    "DEEPDTA_CHARPROTSET",
    "DTAFeatureSet",
    "deepdta_rm2",
    "demo_dta_frame",
    "label_sequence",
    "label_smiles",
    "one_hot_sequence",
    "one_hot_smiles",
    "paired_dta_features",
    "protein_feature_frame",
]

#: Verbatim DeepDTA ``CHARPROTSET`` (1-based; zero is the padding code).
DEEPDTA_CHARPROTSET: dict[str, int] = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,
    "Z": 25,
}

#: Verbatim DeepDTA canonical-SMILES alphabet (zero is padding).
DEEPDTA_CHARCANSMISET: dict[str, int] = {
    "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, ".": 7,
    "1": 8, "0": 9, "3": 10, "2": 11, "5": 12, "4": 13, "7": 14,
    "6": 15, "9": 16, "8": 17, "=": 18, "A": 19, "C": 20, "B": 21,
    "E": 22, "D": 23, "G": 24, "F": 25, "I": 26, "H": 27, "K": 28,
    "M": 29, "L": 30, "O": 31, "N": 32, "P": 33, "S": 34, "R": 35,
    "U": 36, "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
    "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, "d": 49,
    "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, "l": 55, "o": 56,
    "n": 57, "s": 58, "r": 59, "u": 60, "t": 61, "y": 62,
}

#: Verbatim DeepDTA isomeric-SMILES alphabet (the upstream default).
DEEPDTA_CHARISOSMISET: dict[str, int] = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34,
    ".": 2, "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5,
    "7": 38, "6": 6, "9": 39, "8": 7, "=": 40, "A": 41, "@": 8,
    "C": 42, "B": 9, "E": 43, "D": 10, "G": 44, "F": 11, "I": 45,
    "H": 12, "K": 46, "M": 47, "L": 13, "O": 48, "N": 14, "P": 15,
    "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, "V": 18, "Y": 52,
    "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, "b": 21,
    "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28,
    "y": 64,
}

AMINO_ACIDS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")


def _clean_sequence(sequence: Any) -> str:
    if not isinstance(sequence, str):
        raise DataValidationError("protein sequence must be a string")
    cleaned = "".join(str(sequence).split()).upper()
    if not cleaned:
        raise DataValidationError("protein sequence is empty")
    unknown = sorted(set(cleaned) - set(DEEPDTA_CHARPROTSET))
    if unknown:
        raise DataValidationError(
            "protein sequence contains symbols outside DeepDTA's alphabet: "
            + ", ".join(unknown)
            + "; accepted symbols are "
            + "".join(DEEPDTA_CHARPROTSET)
        )
    return cleaned


def _max_length(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise DataValidationError("max_length must be a positive integer")
    return int(value)


def label_sequence(sequence: str, max_length: int = 1000) -> np.ndarray:
    """Encode a protein with DeepDTA's exact labels, truncation and padding."""
    cleaned = _clean_sequence(sequence)
    size = _max_length(max_length)
    encoded = np.zeros(size, dtype=float)
    encoded[: min(size, len(cleaned))] = [
        DEEPDTA_CHARPROTSET[symbol] for symbol in cleaned[:size]
    ]
    return encoded


def one_hot_sequence(sequence: str, max_length: int = 1000) -> np.ndarray:
    """DeepDTA's dense ``(max_length, 25)`` one-hot protein encoding."""
    labels = label_sequence(sequence, max_length=max_length).astype(int)
    encoded = np.zeros((len(labels), len(DEEPDTA_CHARPROTSET)), dtype=float)
    present = labels > 0
    rows = np.flatnonzero(present)
    encoded[rows, labels[present] - 1] = 1.0
    return encoded


def _smiles_alphabet(smiles: str, *, isomeric: bool) -> tuple[str, dict[str, int]]:
    if not isinstance(smiles, str) or not smiles:
        raise DataValidationError("SMILES must be a non-empty string")
    alphabet = DEEPDTA_CHARISOSMISET if isomeric else DEEPDTA_CHARCANSMISET
    unknown = sorted(set(smiles) - set(alphabet))
    if unknown:
        raise DataValidationError(
            "SMILES contains symbols outside the selected DeepDTA alphabet: "
            + ", ".join(repr(symbol) for symbol in unknown)
        )
    return smiles, alphabet


def label_smiles(
    smiles: str, max_length: int = 100, *, isomeric: bool = True
) -> np.ndarray:
    """Encode SMILES with DeepDTA's exact labels, truncation and zero padding.

    ``isomeric=True`` selects the upstream default 64-symbol alphabet;
    ``False`` selects its 62-symbol canonical-SMILES alphabet.  This function
    encodes the supplied string; it does not silently canonicalize it first.
    """
    cleaned, alphabet = _smiles_alphabet(smiles, isomeric=isomeric)
    size = _max_length(max_length)
    encoded = np.zeros(size, dtype=float)
    encoded[: min(size, len(cleaned))] = [
        alphabet[symbol] for symbol in cleaned[:size]
    ]
    return encoded


def one_hot_smiles(
    smiles: str, max_length: int = 100, *, isomeric: bool = True
) -> np.ndarray:
    """DeepDTA's dense ``(max_length, alphabet_size)`` SMILES one-hot."""
    labels = label_smiles(smiles, max_length=max_length, isomeric=isomeric).astype(int)
    alphabet_size = 64 if isomeric else 62
    encoded = np.zeros((len(labels), alphabet_size), dtype=float)
    present = labels > 0
    rows = np.flatnonzero(present)
    encoded[rows, labels[present] - 1] = 1.0
    return encoded


def _protein_row(sequence: str, dipeptides: bool) -> dict[str, float]:
    cleaned = _clean_sequence(sequence)
    length = len(cleaned)
    standard = [symbol for symbol in cleaned if symbol in AMINO_ACIDS]
    row = {
        "prot_length": float(length),
        "prot_standard_fraction": len(standard) / length,
        "prot_nonstandard_fraction": 1.0 - len(standard) / length,
    }
    for symbol in AMINO_ACIDS:
        row[f"prot_aac_{symbol}"] = cleaned.count(symbol) / length
    if dipeptides:
        pairs = [cleaned[i : i + 2] for i in range(max(0, length - 1))]
        denominator = max(1, len(pairs))
        valid = sum(pair[0] in AMINO_ACIDS and pair[1] in AMINO_ACIDS for pair in pairs)
        row["prot_dipeptide_standard_fraction"] = valid / denominator
        for first in AMINO_ACIDS:
            for second in AMINO_ACIDS:
                pair = first + second
                row[f"prot_dipep_{pair}"] = pairs.count(pair) / denominator
    return row


def protein_feature_frame(
    sequences: Iterable[str], *, dipeptides: bool = False
) -> pd.DataFrame:
    """Return aligned AAC (and optionally dipeptide) features per sequence."""
    if isinstance(sequences, (str, bytes)):
        raise DataValidationError("sequences must be an iterable, not one bare string")
    values = list(sequences)
    if not values:
        raise DataValidationError("at least one protein sequence is required")
    return pd.DataFrame([_protein_row(value, dipeptides) for value in values])


@dataclass(slots=True)
class DTAFeatureSet:
    """Aligned paired features, optional target and explicit row accounting."""

    X: pd.DataFrame
    y: pd.Series | None
    kept_indices: tuple[int, ...]
    dropped: tuple[tuple[int, str], ...]
    n_protein_features: int

    @property
    def n_ligand_features(self) -> int:
        return self.X.shape[1] - self.n_protein_features

    def training_frame(self) -> pd.DataFrame:
        """Features plus ``affinity`` when a target was supplied."""
        frame = self.X.copy()
        if self.y is not None:
            frame["affinity"] = self.y.to_numpy()
        return frame


def paired_dta_features(
    pairs: pd.DataFrame,
    *,
    smiles_column: str = "smiles",
    protein_column: str = "protein",
    affinity_column: str | None = "affinity",
    dipeptides: bool = False,
) -> DTAFeatureSet:
    """Build an aligned ECFP + protein matrix from a paired DTA table.

    Source-row positions are zero-based positional indices, not DataFrame
    labels.  RDKit-invalid ligands are dropped together with their protein
    and affinity values and are returned in ``dropped``.  Sequence errors
    and non-numeric affinities are hard errors: dropping those silently
    would make a benchmark look cleaner than its source data.
    """
    if not isinstance(pairs, pd.DataFrame) or pairs.empty:
        raise DataValidationError("pairs must be a non-empty pandas DataFrame")
    required = [smiles_column, protein_column]
    if affinity_column is not None:
        required.append(affinity_column)
    missing = [name for name in required if name not in pairs.columns]
    if missing:
        raise DataValidationError(
            "paired DTA table is missing columns: " + ", ".join(missing)
        )
    proteins = protein_feature_frame(
        pairs[protein_column].tolist(), dipeptides=dipeptides
    )
    target: pd.Series | None = None
    if affinity_column is not None:
        target = pd.to_numeric(pairs[affinity_column], errors="coerce")
        bad = np.flatnonzero(~np.isfinite(target.to_numpy(dtype=float))).tolist()
        if bad:
            raise DataValidationError(
                "affinity must be finite numeric at source rows: "
                + ", ".join(map(str, bad[:10]))
            )

    # Local import keeps the lightweight sequence helpers independent of RDKit.
    from pharmaml.chem.features import compute_feature_frame

    ligand, report = compute_feature_frame(
        pairs[smiles_column].astype(str).tolist(), families=("ecfp",)
    )
    kept = tuple(int(index) for index in ligand.index.tolist())
    if not kept:
        raise DataValidationError("no valid ligand/protein pairs remain after ECFP featurization")
    proteins = proteins.iloc[list(kept)].reset_index(drop=True)
    ligand = ligand.reset_index(drop=True)
    X = pd.concat([ligand, proteins], axis=1)
    if target is not None:
        target = target.iloc[list(kept)].reset_index(drop=True)
        target.name = affinity_column
    dropped = tuple((int(index), str(smiles)) for index, smiles in report.dropped)
    return DTAFeatureSet(
        X=X,
        y=target,
        kept_indices=kept,
        dropped=dropped,
        n_protein_features=proteins.shape[1],
    )


def deepdta_rm2(y_true: Any, y_pred: Any) -> float:
    """DeepDTA's ``get_rm2`` external-prediction metric.

    The port includes its zero-intercept slope ``k = Σ(yŷ)/Σ(ŷ²)`` and
    penalty ``r² * (1 - sqrt(|(r²)² - (r0²)²|))``.  Undefined constant/degenerate
    inputs raise instead of emitting a misleading finite score.
    """
    observed = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if observed.size != predicted.size or observed.size < 2:
        raise DataValidationError("rm2 needs equally sized observed/predicted vectors (n >= 2)")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        raise DataValidationError("rm2 observed/predicted values must be finite")
    obs_centered = observed - observed.mean()
    pred_centered = predicted - predicted.mean()
    denominator = float(np.dot(obs_centered, obs_centered) * np.dot(pred_centered, pred_centered))
    pred_square = float(np.dot(predicted, predicted))
    obs_square = float(np.dot(obs_centered, obs_centered))
    if denominator <= 0.0 or pred_square <= 0.0 or obs_square <= 0.0:
        raise DataValidationError("rm2 is undefined for a constant or all-zero vector")
    r2 = float(np.dot(pred_centered, obs_centered) ** 2 / denominator)
    slope = float(np.dot(observed, predicted) / pred_square)
    r0_squared = 1.0 - float(
        np.dot(observed - slope * predicted, observed - slope * predicted) / obs_square
    )
    return r2 * (1.0 - np.sqrt(abs(r2 * r2 - r0_squared * r0_squared)))


def demo_dta_frame() -> pd.DataFrame:
    """Twelve synthetic, clearly non-benchmark pairs for the offline GUI."""
    proteins = [
        "MKTAYIAKQRQISFVKSHFSRQ",
        "GAVLILKKGGSSHHTRPFSL",
        "ACDEFGHIKLMNPQRSTVWY",
    ]
    smiles = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CC(=O)O"]
    rows = []
    for index in range(12):
        rows.append(
            {
                "smiles": smiles[index % len(smiles)],
                "protein": proteins[index % len(proteins)],
                "affinity": 4.5 + 0.31 * index + 0.17 * (index % 3),
            }
        )
    return pd.DataFrame(rows)
