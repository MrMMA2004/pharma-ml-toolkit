"""Morgan-bit interpretability — decode fingerprint bits back to chemistry.

The ``Ml/D2.py`` upload's interpretability block (its best idea the
toolkit did not yet carry), ported to the toolkit's contracts.  After
training its stacking ensemble the original did three things:

1. correlated every feature with pEC50, ranked by ``|R|`` and drew the
   top-10 as a SIGNED bar chart (green positive / red negative) —
   ``Top10_Correlation_Direction.svg``;
2. drew the top-10 + target correlation heatmap with seaborn —
   ``Feature_Correlation_Heatmap_with_pEC50.svg``;
3. decoded the top Morgan bits back to atom environments
   (``decode_morgan_bit``) and wrote ``morgan_atom_environments.txt``,
   so a fingerprint column finally *meant* something ("this bit is the
   carbonyl environment").

This module ships the computation halves of (1) and (3):
:func:`signed_feature_correlations` and :func:`decode_morgan_bit`.  The
figures live beside the other interpretability plots as
:func:`pharmaml.viz.ml_plots.plot_signed_correlations` and
:func:`pharmaml.viz.ml_plots.plot_correlation_heatmap`.

Honest deviations from the upload (every one verified against the
original's source):

* **one molecule, all its hits.**  ``decode_morgan_bit`` took a list of
  SMILES and scanned up to 100 of them for the first 3 *example
  molecules* carrying the bit; this port decodes ONE molecule and
  reports EVERY ``(atom, radius)`` hit the bit fires on inside it —
  ``bitInfo`` already enumerates them, so nothing is left to a scan's
  luck.  The multi-molecule example hunt is a caller-side loop.
* **the hit's own radius.**  The original passed the FINGERPRINT radius
  to ``FindAtomEnvironmentOfRadiusN`` while discarding the hit's own
  radius (``atom_idx, _ = bit_info[bit_index][0]``) — for a bit that
  fires at radius 1 it drew the radius-2 environment around the center
  atom instead.  Fixed here: each hit is decoded with the radius
  recorded for THAT hit, pinned by test.
* **radius-0 hits are kept.**  The original's ``if
  submol.GetNumAtoms() > 0`` guard silently dropped every radius-0 hit
  (a radius-0 environment has no bonds, so the sub-molecule is empty).
  Here the environment of a radius-0 hit is the center atom alone,
  rendered as a single-atom SMILES.
* **the modern generator API.**  ``GetMorganFingerprintAsBitVect`` (the
  original's route) emits a deprecation warning on RDKit 2026.09.1;
  this port uses ``rdFingerprintGenerator.GetMorganGenerator`` with the
  ``AdditionalOutput`` bit-info map — verified bit-identical to the
  legacy route on the pinned probe molecule (same on-bit set, same
  ``(atom, radius)`` hits).
* **2048 default bits.**  The toolkit's :data:`ECFP_BITS` convention
  (the original folded its R2 fingerprint into 4096 bits; the
  ``ecfp`` family this module interprets is the 2048-bit twin).
* **typed errors, not silent skips.**  The original ``continue``d past
  unparseable SMILES and appended a fake ``0.0`` correlation whenever
  ``pearsonr`` raised — a failed column masqueraded as "no
  correlation".  This port raises :class:`DataValidationError` on a
  bad SMILES / unset bit, and an undefined correlation (constant
  column) is *dropped*, never zeroed.
* **no 1000-column cap.**  The original correlated only the first 1000
  features "for speed"; every numeric column is correlated here.
* **feature-invariant flavour.**  ``use_features=True`` decodes the
  FCFP (feature-invariant) twin the original never exposed.

Examples
--------
>>> info = decode_morgan_bit("CN1C=NC2=C1C(=O)N(C(=O)O2)C", 314)
>>> (info.bit, info.radius, info.n_bits, info.use_features)
(314, 2, 2048, False)
>>> [(h.atom_index, h.radius) for h in info.hits]   # both carbonyls
[(7, 1), (10, 1)]
>>> {h.environment_smiles for h in info.hits}
{'c=O'}
>>> sorted(info.hits[0].atom_indices)               # first hit's atom pair
[6, 7]
>>> info.drawing.startswith("<?xml")                # the RDKit bit legend
True
>>> str(info)
'Morgan bit 314 (radius 2, 2048 bits, connectivity invariants): 2 hit(s) on CN1C=NC2=C1C(=O)N(C(=O)O2)C — atom 7 @r1, atom 10 @r1'
>>> sorted(info.as_dict())
['bit', 'drawing', 'hits', 'n_bits', 'radius', 'smiles', 'use_features']

The feature-invariant (FCFP) flavour decodes the same way:

>>> feat = decode_morgan_bit("CN1C=NC2=C1C(=O)N(C(=O)O2)C", 418, use_features=True)
>>> (feat.use_features, [(h.atom_index, h.radius) for h in feat.hits])
(True, [(4, 2)])
>>> feat.hits[0].environment_smiles
'cnc(oc)c(c)n'
>>> quiet = decode_morgan_bit("CN1C=NC2=C1C(=O)N(C(=O)O2)C", 314, with_drawing=False)
>>> quiet.drawing is None
True

Error contracts (the house style):

>>> decode_morgan_bit("CN1C=NC2=C1C(=O)N(C(=O)O2)C", 1)   # caffeine never sets bit 1
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: Morgan bit 1 is not set by 'CN1C=NC2=C1C(=O)N(C(=O)O2)C' (radius 2, 2048 bits, connectivity invariants); only the 27 bit(s) this molecule sets are decodable
>>> decode_morgan_bit("not_a_smiles", 314)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: SMILES 'not_a_smiles' does not parse to a valid RDKit molecule
>>> decode_morgan_bit("CCO", 2048)                     # out of fingerprint range
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: bit index 2048 is out of range for a 2048-bit fingerprint; pass 0 <= bit < 2048

>>> import pandas as pd
>>> y = [1.0, 2.0, 3.0, 4.0]
>>> frame = pd.DataFrame({"perfect": y, "inverse": [-v for v in y], "flat": [7.0] * 4})
>>> corr = signed_feature_correlations(frame, y)
>>> corr.index.tolist()              # |r| descending; the constant column dropped
['perfect', 'inverse']
>>> (float(corr["perfect"]), float(corr["inverse"]))
(1.0, -1.0)
>>> cube = pd.DataFrame({"cube": [v ** 3 for v in y]})
>>> float(signed_feature_correlations(cube, y, method="spearman").iloc[0])
1.0
>>> float(signed_feature_correlations(cube, y).iloc[0]) < 1.0   # Pearson sees curvature
True
>>> signed_feature_correlations(frame, [1.0, 2.0])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: frame has 4 row(s) but target carries 2 value(s) -- pass one target value per row
>>> signed_feature_correlations(pd.DataFrame({"smiles": ["CCO", "CCC"]}), [1.0, 2.0])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: no numeric columns to correlate -- the frame carries only non-numeric column(s): ['smiles']
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.chem.descriptors import require_rdkit
from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

__all__ = [
    "AtomEnvironment",
    "MorganBitInfo",
    "decode_morgan_bit",
    "signed_feature_correlations",
]

_LOG = get_logger(__name__)

_CORRELATION_METHODS: tuple[str, ...] = ("pearson", "spearman")


@dataclass(frozen=True, slots=True)
class AtomEnvironment:
    """One ``(atom, radius)`` hit a Morgan bit fires on.

    Attributes
    ----------
    atom_index:
        Index of the environment's center atom in the parsed molecule.
    radius:
        The radius of THIS hit's environment (``0..fingerprint
        radius``) — a bit can fire at several radii within one
        molecule, and the environment is decoded with the hit's own
        radius, not the fingerprint's.
    atom_indices:
        Original-molecule indices of every atom in the environment
        (the center plus its neighbours out to ``radius``); for a
        radius-0 hit this is the center atom alone.
    environment_smiles:
        Canonical SMILES of the environment sub-molecule (the
        readable form of the upload's ``environment_smarts`` twin
        below).
    environment_smarts:
        SMARTS of the environment sub-molecule — the form the original
        wrote to ``morgan_atom_environments.txt``.
    center_atom:
        Element symbol of the center atom (the original's
        ``center_atom``).
    """

    atom_index: int
    radius: int
    atom_indices: tuple[int, ...]
    environment_smiles: str
    environment_smarts: str
    center_atom: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready view of the hit."""
        return {
            "atom_index": self.atom_index,
            "radius": self.radius,
            "atom_indices": list(self.atom_indices),
            "environment_smiles": self.environment_smiles,
            "environment_smarts": self.environment_smarts,
            "center_atom": self.center_atom,
        }


@dataclass(frozen=True, slots=True)
class MorganBitInfo:
    """Decode record for one Morgan fingerprint bit on one molecule.

    Attributes
    ----------
    bit:
        The decoded bit index (``0 <= bit < n_bits``).
    radius / n_bits / use_features:
        Echoes of the fingerprint configuration the decode ran with
        (so a stale record can never be mistaken for another
        flavour's decode — the same index means different chemistry
        under different ``n_bits``/invariants).
    smiles:
        The molecule as passed in.
    hits:
        Every :class:`AtomEnvironment` the bit fires on in this
        molecule, in generator order (ascending center-atom index).
    drawing:
        The RDKit bit legend — an SVG string drawing the FIRST hit's
        atom environment with the fingerprint's highlight colours
        (:func:`rdkit.Chem.Draw.DrawMorganBit`) — or ``None`` when
        decoded with ``with_drawing=False``.
    """

    bit: int
    radius: int
    n_bits: int
    use_features: bool
    smiles: str
    hits: list[AtomEnvironment] = field(default_factory=list)
    drawing: str | None = None

    @property
    def n_hits(self) -> int:
        """Number of atom environments the bit fires on."""
        return len(self.hits)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready view of the record (hits nested as dicts)."""
        return {
            "bit": self.bit,
            "radius": self.radius,
            "n_bits": self.n_bits,
            "use_features": self.use_features,
            "smiles": self.smiles,
            "hits": [hit.as_dict() for hit in self.hits],
            "drawing": self.drawing,
        }

    def __str__(self) -> str:  # pragma: no cover - presentation only
        flavour = "feature" if self.use_features else "connectivity"
        hits = ", ".join(
            f"atom {hit.atom_index} @r{hit.radius}" for hit in self.hits
        ) or "no hits"
        return (
            f"Morgan bit {self.bit} (radius {self.radius}, {self.n_bits} bits, "
            f"{flavour} invariants): {self.n_hits} hit(s) on {self.smiles} — {hits}"
        )


def _decode_hit(mol: Any, atom_index: int, hit_radius: int) -> AtomEnvironment:
    """Decode one ``(atom, radius)`` hit into an :class:`AtomEnvironment`."""
    Chem = require_rdkit()
    center = mol.GetAtomWithIdx(atom_index)
    if hit_radius == 0:
        # A radius-0 environment has no bonds (the upload's
        # GetNumAtoms() > 0 guard silently dropped these hits); the
        # honest environment is the center atom alone.
        single = Chem.RWMol()
        single.AddAtom(copy.copy(center))
        atom_only = single.GetMol()
        return AtomEnvironment(
            atom_index=atom_index,
            radius=hit_radius,
            atom_indices=(atom_index,),
            environment_smiles=Chem.MolToSmiles(atom_only),
            environment_smarts=Chem.MolToSmarts(atom_only),
            center_atom=center.GetSymbol(),
        )
    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, hit_radius, atom_index))
    atom_map: dict[int, int] = {}
    submol = Chem.PathToSubmol(mol, bonds, atomMap=atom_map)
    return AtomEnvironment(
        atom_index=atom_index,
        radius=hit_radius,
        atom_indices=tuple(sorted(int(idx) for idx in atom_map)),
        environment_smiles=Chem.MolToSmiles(submol),
        environment_smarts=Chem.MolToSmarts(submol),
        center_atom=center.GetSymbol(),
    )


def decode_morgan_bit(
    smiles: str,
    bit: int,
    *,
    radius: int = 2,
    n_bits: int = 2048,
    use_features: bool = False,
    with_drawing: bool = True,
) -> MorganBitInfo:
    """Decode one Morgan fingerprint bit into its atom environments.

    The upload's ``decode_morgan_bit``, ported: the bit-info map of the
    molecule's folded Morgan fingerprint tells which ``(atom, radius)``
    environments hashed onto ``bit``; every hit is decoded to its atom
    indices, an environment SMILES/SMARTS and the center element, and
    the first hit's environment is drawn as the canonical RDKit bit
    legend (an SVG string).

    Parameters
    ----------
    smiles:
        One SMILES string.
    bit:
        The bit index to decode (``0 <= bit < n_bits``).
    radius:
        Morgan fingerprint radius (default 2, the ``ecfp`` family).
    n_bits:
        Fold length of the fingerprint (default 2048,
        :data:`pharmaml.chem.features.ECFP_BITS`; the upload folded its
        R2 block into 4096 — pick the length your feature frame was
        built with, the same index means different chemistry under a
        different fold).
    use_features:
        Decode the FCFP (feature-invariant) flavour instead of the
        connectivity flavour — the same bit index under different
        invariants decodes different environments.
    with_drawing:
        Draw the first hit's bit legend (SVG string, ~3-8 KB) when
        true; ``False`` keeps the record light for batch sweeps.

    Returns
    -------
    MorganBitInfo
        The decode record (echoes the configuration, every hit, the
        optional drawing).

    Raises
    ------
    DependencyError
        If RDKit is not installed.
    DataValidationError
        If ``smiles`` does not parse, the guards fail (``bit`` out of
        range, non-integer ``radius``/``bit``/``n_bits``), or the bit
        is simply not set by this molecule — the message names the
        bit, the SMILES and the configuration so a sweep can catch
        and skip.

    Examples
    --------
    Caffeine's bit 314 (radius 2, 2048 bits, connectivity invariants)
    is the carbonyl environment — verified live on RDKit 2026.09.1:

    >>> info = decode_morgan_bit("CN1C=NC2=C1C(=O)N(C(=O)O2)C", 314)
    >>> [(h.atom_index, h.radius) for h in info.hits]
    [(7, 1), (10, 1)]
    >>> {h.environment_smiles for h in info.hits}
    {'c=O'}
    """
    # Int-ness before range checks (the descriptors.py convention): a
    # str/None would leak a raw TypeError from the comparisons, and a
    # float would blow up inside the RDKit Boost bindings.
    if not isinstance(radius, (int, np.integer)) or isinstance(radius, bool):
        raise DataValidationError(f"radius must be an integer >= 0, got {radius!r}")
    radius = int(radius)
    if radius < 0:
        raise DataValidationError(f"radius must be >= 0, got {radius}")
    if not isinstance(n_bits, (int, np.integer)) or isinstance(n_bits, bool):
        raise DataValidationError(f"n_bits must be an integer >= 1, got {n_bits!r}")
    n_bits = int(n_bits)
    if n_bits < 1:
        raise DataValidationError(f"n_bits must be >= 1, got {n_bits}")
    if not isinstance(bit, (int, np.integer)) or isinstance(bit, bool):
        raise DataValidationError(f"bit must be an integer, got {bit!r}")
    bit = int(bit)
    if not 0 <= bit < n_bits:
        raise DataValidationError(
            f"bit index {bit} is out of range for a {n_bits}-bit fingerprint; "
            f"pass 0 <= bit < {n_bits}"
        )
    if not isinstance(smiles, str) or not smiles.strip():
        raise DataValidationError(
            f"smiles must be a non-empty SMILES string, got {smiles!r}"
        )

    Chem = require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DataValidationError(
            f"SMILES {smiles!r} does not parse to a valid RDKit molecule"
        )

    # The modern generator route (GetMorganFingerprintAsBitVect is
    # deprecated on RDKit 2026.09.1); verified bit-identical to the
    # legacy call on the probe molecule.
    from rdkit.Chem import rdFingerprintGenerator

    invariants = (
        rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        if use_features
        else None
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, atomInvariantsGenerator=invariants
    )
    additional = rdFingerprintGenerator.AdditionalOutput()
    additional.AllocateBitInfoMap()
    generator.GetFingerprint(mol, additionalOutput=additional)
    bit_map = additional.GetBitInfoMap()

    if bit not in bit_map:
        flavour = "feature" if use_features else "connectivity"
        raise DataValidationError(
            f"Morgan bit {bit} is not set by {smiles!r} (radius {radius}, "
            f"{n_bits} bits, {flavour} invariants); only the {len(bit_map)} "
            "bit(s) this molecule sets are decodable"
        )

    hits = [
        _decode_hit(mol, int(atom_index), int(hit_radius))
        for atom_index, hit_radius in bit_map[bit]
    ]

    drawing: str | None = None
    if with_drawing:
        from rdkit.Chem.Draw import DrawMorganBit

        # whichExample=0 -> the first hit's environment is the legend.
        drawing = DrawMorganBit(mol, bit, bit_map)

    _LOG.debug(
        "decoded Morgan bit %d on %r: %d hit(s)", bit, smiles, len(hits)
    )
    return MorganBitInfo(
        bit=bit,
        radius=radius,
        n_bits=n_bits,
        use_features=bool(use_features),
        smiles=smiles,
        hits=hits,
        drawing=drawing,
    )


def _robust_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r via the offset-then-scale recipe (100x-154).

    The web twin (``src/lib/ml-math.ts`` ``pearsonCorrelation``) uses
    the SAME recipe, so the two twins agree everywhere either of them
    is honest.  ``NaN`` -- never a fake 0 -- for a constant column:
    re-centring on the first element is exact for same-magnitude
    values (Sterbenz), so a constant column collapses to exact zeros
    at ANY scale.  Re-scaling by the max deviation keeps every
    product finite: the naive ``numpy.corrcoef`` two-pass this
    replaced lost defined correlations at extreme scales (1e300-scale
    deviation products overflow to inf, 1e-300-scale squares
    underflow to 0) and reported a FAKE ``r = 0`` for a 1e300-scale
    constant column (the mean's ULP-scale rounding error makes the
    "zero" deviations ~1e284; their squares overflow to inf; 0/inf =
    0 -- while scipy honestly returns NaN).
    """
    if x.size < 2 or x.size != y.size:
        return float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    aa = x - x[0]
    bb = y - y[0]
    max_a = float(np.abs(aa).max())
    max_b = float(np.abs(bb).max())
    if max_a == 0.0 or max_b == 0.0:
        return float("nan")  # constant: undefined, never zero
    # scale, THEN subtract the scaled mean: the offset alone leaves
    # the n*(mean_a - a0)*(mean_b - b0) cross term in the covariance --
    # the centering is not optional, and at this scale it is exact.
    xa = aa / max_a
    yb = bb / max_b
    xa = xa - xa.mean()
    yb = yb - yb.mean()
    num = float((xa * yb).sum())
    den = float(np.sqrt(float((xa * xa).sum()) * float((yb * yb).sum())))
    if den == 0.0:
        return float("nan")
    return num / den


def signed_feature_correlations(
    frame: pd.DataFrame,
    target: Sequence[float],
    *,
    method: str = "pearson",
) -> pd.Series:
    """Signed correlation of every numeric column with ``target``.

    The computation half of the upload's ``analyze_correlations``: the
    original correlated only the first 1000 features "for speed" and
    appended a fake ``0.0`` whenever ``pearsonr`` raised (a failed
    column masquerading as "no correlation"); this port correlates
    EVERY numeric column and never invents a zero.

    The result is sorted by **absolute** value descending with the
    **sign preserved** — the ranking the upload's top-10 bar chart
    drew — so ``.head(10)`` feeds
    :func:`pharmaml.viz.ml_plots.plot_signed_correlations` directly.

    Undefined correlations are dropped, never zeroed: a constant
    column (zero variance — scipy's ``ConstantInputWarning`` case, at
    ANY scale: the recipe's first-element re-centring is exact, so a
    1e300-scale constant column collapses to exact zeros and drops
    honestly where the naive two-pass reported a fake ``r = 0``), a
    column numerically undefined at extreme scales, or a column with
    fewer than two pairwise-complete rows yields NaN and is removed
    from the result, with the drop logged.  NaN cells inside a column
    are handled pairwise (a missing value removes its row from that
    column's correlation only, the ``pandas.corr`` convention).

    Parameters
    ----------
    frame:
        Feature frame (any shape; non-numeric columns are skipped with
        a warning naming them, the ``clean_numeric_frame`` precedent —
        complex-dtype columns ride the same skip path (100x-153):
        pandas counts them "numeric", but a correlation that
        silently discards the imaginary part is the same species of
        partial truth this function refuses everywhere else).
    target:
        One finite numeric value per frame row.
    method:
        ``"pearson"`` (linear, the offset-then-scale :func:`_robust_pearson`
        recipe the web twin shares) or ``"spearman"``
        (rank, ``scipy.stats.spearmanr`` — monotone relationships
        score ±1 however curved).

    Returns
    -------
    pandas.Series
        Correlations indexed by column name, ``|r|`` descending, sign
        preserved (ties broken by column order, first-declared wins —
        the stable-sort convention of the permutation-importance
        plotter).  The Series name is the method.

    Raises
    ------
    DataValidationError
        On a row/target length mismatch, a non-numeric or non-finite
        target, an unknown ``method``, fewer than two rows, no numeric
        columns at all, or an empty result after the undefined-drop.

    Examples
    --------
    >>> import pandas as pd
    >>> y = [1.0, 2.0, 3.0, 4.0]
    >>> frame = pd.DataFrame({"perfect": y, "inverse": [-v for v in y]})
    >>> corr = signed_feature_correlations(frame, y)
    >>> corr.index.tolist(), (float(corr["perfect"]), float(corr["inverse"]))
    (['perfect', 'inverse'], (1.0, -1.0))
    """
    if method not in _CORRELATION_METHODS:
        raise DataValidationError(
            f"method must be one of {_CORRELATION_METHODS}, got {method!r}"
        )
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError(
            f"frame must be a pandas.DataFrame, got {type(frame).__name__}"
        )
    try:
        target_values = np.asarray(list(target), dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"target must be numeric ({exc})"
        ) from exc
    target_values = target_values.reshape(-1)
    if len(target_values) != len(frame):
        raise DataValidationError(
            f"frame has {len(frame)} row(s) but target carries "
            f"{len(target_values)} value(s) -- pass one target value per row"
        )
    if len(target_values) < 2:
        raise DataValidationError(
            f"need at least 2 rows to correlate, got {len(target_values)}"
        )
    if not np.isfinite(target_values).all():
        bad = int((~np.isfinite(target_values)).sum())
        raise DataValidationError(
            f"target carries {bad} non-finite value(s) (NaN/inf); a missing "
            "target has no honest correlation with anything"
        )

    numeric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        # (100x-153) complex dtype IS "numeric" to pandas, and the
        # to_numpy(dtype=float) cast below would silently discard the
        # imaginary part (an unread ComplexWarning).  Complex columns
        # ride the non-numeric skip path with the naming warning
        # instead — never a half-honest correlation.
        and not pd.api.types.is_complex_dtype(frame[column])
    ]
    skipped = [column for column in frame.columns if column not in numeric_columns]
    if skipped:
        _LOG.warning(
            "Skipping %d non-numeric column(s): %s", len(skipped), skipped[:10]
        )
    if not numeric_columns:
        raise DataValidationError(
            "no numeric columns to correlate -- the frame carries only "
            f"non-numeric column(s): {skipped[:10]}"
        )

    values: list[float] = []
    names: list[str] = []
    dropped_undefined: list[str] = []
    if method == "spearman":
        from scipy.stats import ConstantInputWarning, spearmanr

        for column in numeric_columns:
            column_values = frame[column].to_numpy(dtype=float)
            mask = np.isfinite(column_values)
            pairs_x = column_values[mask]
            pairs_y = target_values[mask]
            if pairs_x.size < 2:
                dropped_undefined.append(column)
                continue
            with warnings.catch_warnings():
                # A constant column is the documented NaN-drop case,
                # not a warning to surface to the caller.
                warnings.simplefilter("ignore", ConstantInputWarning)
                correlation = float(spearmanr(pairs_x, pairs_y).statistic)
            if np.isnan(correlation):
                dropped_undefined.append(column)
                continue
            values.append(correlation)
            names.append(column)
    else:
        for column in numeric_columns:
            column_values = frame[column].to_numpy(dtype=float)
            mask = np.isfinite(column_values)
            pairs_x = column_values[mask]
            pairs_y = target_values[mask]
            if pairs_x.size < 2:
                dropped_undefined.append(column)
                continue
            correlation = _robust_pearson(pairs_x, pairs_y)
            if np.isnan(correlation):
                dropped_undefined.append(column)
                continue
            values.append(correlation)
            names.append(column)

    if dropped_undefined:
        _LOG.warning(
            "Dropping %d column(s) with an undefined correlation "
            "(constant, numerically undefined at extreme scales, or "
            "fewer than 2 pairwise-complete rows): %s",
            len(dropped_undefined),
            dropped_undefined[:10],
        )
    if not names:
        raise DataValidationError(
            "no correlation could be computed -- every numeric column was "
            f"undefined (constant or incomplete): {dropped_undefined[:10]}"
        )

    # |r| descending, sign preserved, stable so ties keep column order.
    order = np.argsort(-np.abs(values), kind="stable")
    return pd.Series(
        [values[int(index)] for index in order],
        index=[names[int(index)] for index in order],
        name=method,
    )
