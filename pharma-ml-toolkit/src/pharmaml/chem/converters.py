"""Bioactivity endpoint and file-format conversions.

Upgrades the original flat scripts ``Convert IC50(nM) to pIC50.py``,
``Split CSV file.py`` and friends into typed, reusable functions that:

* handle censored values (``<`` / ``>`` qualifiers) explicitly instead
  of silently stripping them,
* never mutate the caller's DataFrame in place,
* validate units and ranges with precise error messages.

The molecular file-format converters pick up the remaining scripts of
the same family: ``Convert SMI to CSV.py`` (memory-flat streaming, BOM
CSV output), ``Convert CSV to SDF.py`` (ETKDG-embedded ligand libraries
with per-row failure accounting instead of a bare ``print``) and the
``Sdf reading.ipynb`` notebook (SDF -> table round-trip without
notebook-only ``PandasTools`` helpers).  RDKit stays optional: only the
three molecule-aware functions import it, lazily, with an actionable
:class:`~pharmaml.exceptions.DependencyError` when it is missing.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger
from pharmaml.utils.io import read_table, write_table
from pharmaml.utils.validation import require_columns

_LOG = get_logger(__name__)

__all__ = [
    "EndpointDatasetReport",
    "SdfWriteReport",
    "add_ic50_column",
    "add_pic50_column",
    "ic50_to_pic50",
    "ki_to_pki",
    "pic50_from_ic50_series",
    "pic50_to_ic50",
    "prepare_pic50_dataset",
    "sdf_to_table",
    "smi_to_csv",
    "split_table_into_chunks",
    "table_to_sdf",
    "table_to_smi",
]


@dataclass(frozen=True, slots=True)
class EndpointValue:
    """A parsed bioactivity measurement, censored or exact.

    Attributes
    ----------
    value:
        Numeric magnitude in the input unit (e.g. nM).
    qualifier:
        ``''`` for exact measurements, ``'<'`` or ``'>'`` for censored
        ones (reporter-style entries such as ``<0.5`` or ``>10000``).
    """

    value: float
    qualifier: str = ""

    @property
    def is_censored(self) -> bool:
        return self.qualifier != ""


def parse_endpoint(raw: object) -> EndpointValue | None:
    """Parse one endpoint cell into an :class:`EndpointValue`.

    Accepts numbers or strings, tolerating ``'<'``/``'>'`` prefixes,
    thousands separators, surrounding units (``nM``, ``uM``) and stray
    whitespace.  Returns ``None`` for genuinely unusable entries
    (blank, text, or non-positive magnitudes).
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return None

    qualifier = ""
    if text[0] in "<>":
        qualifier = text[0]
        text = text[1:].strip()

    # Strip a trailing unit token such as 'nM' / 'uM' / 'M'.  Matched
    # case-insensitively (27-A): spreadsheets write 'NM'/'um'/'μM' as
    # often as the canonical 'nM'/'uM'/'µM', and a pure case variance
    # must not silently make the row unparseable -- the caller's
    # ``unit`` parameter stays the authoritative scale, so no
    # interpretation is at risk here, only lexing.  casefold() also
    # unifies the micro sign (U+00B5) and Greek mu (U+03BC) spellings
    # of 'µM'.  'pM' completes the concentration set of
    # ``ic50_to_pic50``'s scale dict (2-char tokens first: 'pM'/'mM'
    # must match before the bare 'M').
    lowered = text.casefold()
    for unit in ("nM", "uM", "µM", "pM", "mM", "M"):
        if lowered.endswith(unit.casefold()):
            text = text[: -len(unit)].strip()
            break
    text = text.replace(",", "")

    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return EndpointValue(value=value, qualifier=qualifier)


def ic50_to_pic50(ic50: object, unit: str = "nM") -> float | None:
    """Convert an IC50 measurement to pIC50 = -log10(IC50 [M]).

    Parameters
    ----------
    ic50:
        Numeric value or string (censored entries like ``"<100"`` are
        accepted; the qualifier is stripped and only the magnitude is
        converted).
    unit:
        Input concentration unit: one of ``'M', 'mM', 'uM', 'nM', 'pM'``.

    Returns
    -------
    float or None
        pIC50 value, or ``None`` when the input cannot be parsed.

    Examples
    --------
    >>> ic50_to_pic50(1000)            # 1000 nM == 1 uM
    6.0
    >>> ic50_to_pic50("<10")           # censored: 10 nM
    8.0
    """
    scale = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "nM": 1e-9, "pM": 1e-12}
    unit_key = unit.strip()
    if unit_key not in scale:
        raise DataValidationError(
            f"Unknown concentration unit {unit!r}; expected one of {sorted(scale)}"
        )
    parsed = parse_endpoint(ic50)
    if parsed is None:
        return None
    molar = parsed.value * scale[unit_key]
    result = -math.log10(molar)
    # -log10(1.0) is IEEE negative zero; normalise it so a 1 M endpoint
    # never surfaces as "-0.0" in tables or reports (27-A).
    return 0.0 if result == 0.0 else result


def ki_to_pki(ki: object, unit: str = "nM") -> float | None:
    """Convert an inhibition constant to pKi = -log10(Ki [M])."""
    return ic50_to_pic50(ki, unit=unit)  # same math, different label


def pic50_from_ic50_series(
    series: pd.Series, unit: str = "nM"
) -> pd.Series:
    """Vectorised IC50 -> pIC50 conversion preserving NaN for bad rows."""
    return series.map(lambda value: ic50_to_pic50(value, unit=unit))


def add_pic50_column(
    frame: pd.DataFrame,
    source_col: str = "IC50 (nM)",
    out_col: str = "pIC50",
    unit: str = "nM",
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """Append a pIC50 column to a table of measurements.

    This is the upgraded equivalent of ``Convert IC50(nM) to pIC50.py``:
    the original mutated a global ``df`` read from a hardcoded Excel
    path and wrote to another hardcoded path; this pure function
    validates the column, converts, and returns a new frame.

    Parameters
    ----------
    frame:
        Table containing the raw activity column.
    source_col:
        Column holding IC50 values (numeric or censored strings).
    out_col:
        Name of the appended column.
    unit:
        Unit of the source values.
    drop_invalid:
        Drop rows whose activity could not be parsed.

    Returns
    -------
    pandas.DataFrame
        A copy with the new column; rows with unparseable activity are
        removed when ``drop_invalid`` is True, otherwise they receive
        ``NaN``.
    """
    require_columns(frame, [source_col])
    result = frame.copy()
    result[out_col] = pic50_from_ic50_series(result[source_col], unit=unit)
    n_invalid = int(result[out_col].isna().sum())
    if n_invalid:
        _LOG.warning("%d rows had unparseable %s values", n_invalid, source_col)
    if drop_invalid:
        result = result[result[out_col].notna()].reset_index(drop=True)
    return result


def pic50_to_ic50(pic50: object, unit: str = "nM") -> float | None:
    """Convert a pIC50 value back to an IC50 concentration.

    The inverse of :func:`ic50_to_pic50` and the library port of
    ``fak_ml.ipynb`` cells 102/150 (``dm.molar.log_to_molar`` -- the
    notebook converts its predicted pIC50 back to nM to triage hits by
    potency thresholds).  ``IC50 = 10 ** -pIC50`` molar, rescaled to
    the requested unit.

    Parameters
    ----------
    pic50:
        Numeric pIC50 value; non-numeric input returns ``None`` (the
        series twin's NaN-preservation path).  NaN and +/-inf pass
        through as NaN (a non-finite pIC50 has no honest concentration).
    unit:
        Output concentration unit: one of ``'M', 'mM', 'uM', 'nM', 'pM'``.

    Returns
    -------
    float or None
        IC50 in ``unit``, or ``None`` when the input cannot be parsed.
        A NaN input yields NaN (not None) so a series round-trip keeps
        its NaN cells distinguishable from unparseable ones.

    Raises
    ------
    DataValidationError
        If ``unit`` is not a supported concentration unit.

    Examples
    --------
    >>> pic50_to_ic50(6.0)              # pIC50 6 -> 1000 nM
    1000.0
    >>> pic50_to_ic50(9.0)              # pIC50 9 -> 1 nM
    1.0
    >>> ic50_to_pic50(pic50_to_ic50(7.5))  # exact round-trip
    7.5

    Notes
    -----
    The arithmetic is exponent-based (``10 ** (exponent - pIC50)``
    with ``exponent`` the unit's scale exponent, e.g. 9 for nM) rather
    than the notebook's ``10 ** -p * 1e9`` product: integer pIC50s land
    on EXACT decimal powers instead of datamol's float noise
    (``dm.molar.log_to_molar(6.0, 'nM')`` returns 999.9999999999999;
    this function returns 1000.0 -- probed, and the round-trip stays
    bit-exact both ways).
    """
    exponents = {"M": 0.0, "mM": 3.0, "uM": 6.0, "µM": 6.0, "nM": 9.0, "pM": 12.0}
    unit_key = unit.strip()
    if unit_key not in exponents:
        raise DataValidationError(
            f"Unknown concentration unit {unit!r}; expected one of {sorted(exponents)}"
        )
    if pic50 is None:
        return None
    try:
        value = float(pic50)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return float("nan")
    if math.isinf(value):
        # -inf pIC50 is an infinitely potent compound (0 concentration);
        # +inf is unbound.  Neither has a finite nM answer -- NaN keeps
        # the cell honest instead of a silent 0.
        return float("nan")
    return 10.0 ** (exponents[unit_key] - value)


def ic50_from_pic50_series(
    series: pd.Series, unit: str = "nM"
) -> pd.Series:
    """Vectorised pIC50 -> IC50 conversion preserving NaN for bad rows."""
    return series.map(lambda value: pic50_to_ic50(value, unit=unit))


def add_ic50_column(
    frame: pd.DataFrame,
    pic50_col: str = "pIC50",
    out_col: str = "IC50_nM",
    unit: str = "nM",
    drop_invalid: bool = False,
) -> pd.DataFrame:
    """Append an IC50 column converted from a pIC50 column.

    The table twin of :func:`pic50_to_ic50` -- mirrors
    :func:`add_pic50_column`'s contract (the notebook's predicted-pIC50
    sheets gain their nM column before the potency-threshold triage).
    ``drop_invalid`` defaults to False here, unlike the IC50->pIC50
    direction: a prediction sheet keeps every molecule, an unconvertible
    pIC50 simply lands as NaN.

    Parameters
    ----------
    frame:
        Table containing the pIC50 column.
    pic50_col:
        Column holding pIC50 values.
    out_col:
        Name of the appended column.
    unit:
        Unit of the appended IC50 values.
    drop_invalid:
        Drop rows whose pIC50 could not be parsed (default False -- see
        above).

    Returns
    -------
    pandas.DataFrame
        A copy with the new column; rows with unparseable pIC50 receive
        ``NaN`` (or are removed when ``drop_invalid`` is True).

    Examples
    --------
    >>> frame = pd.DataFrame({"pIC50": [6.0, 7.0]})
    >>> add_ic50_column(frame)[["pIC50", "IC50_nM"]]
       pIC50  IC50_nM
    0    6.0   1000.0
    1    7.0    100.0
    """
    require_columns(frame, [pic50_col])
    result = frame.copy()
    result[out_col] = ic50_from_pic50_series(result[pic50_col], unit=unit)
    n_invalid = int(result[out_col].isna().sum())
    if n_invalid:
        _LOG.warning("%d rows had unparseable %s values", n_invalid, pic50_col)
    if drop_invalid:
        result = result[result[out_col].notna()].reset_index(drop=True)
    return result


@dataclass(frozen=True, slots=True)
class EndpointDatasetReport:
    """Audit record for one :func:`prepare_pic50_dataset` run.

    The typed upgrade of ``D2.py``'s silent ``process_data`` counters:
    every row that did not survive to the output frame is accounted
    for by exactly one bucket, so ``rows_valid + rows_missing_smiles +
    rows_unparsable + rows_out_of_range == rows_in`` and ``rows_out ==
    rows_valid - duplicates_collapsed``.

    Attributes
    ----------
    smiles_col:
        Resolved SMILES column name.
    endpoint_col:
        Resolved endpoint column name.
    unit:
        Input concentration unit the values were read in.
    aggregate:
        Duplicate-resolution rule that was applied.
    upper_limit:
        Exclusive magnitude cap applied in the input unit.
    rows_in:
        Rows the input frame carried.
    rows_missing_smiles:
        Rows dropped because the SMILES cell was blank or NaN.
    rows_unparsable:
        Rows whose endpoint cell could not be parsed at all (blank,
        text, or a non-positive magnitude).
    rows_out_of_range:
        Rows whose magnitude exceeded ``upper_limit``.
    rows_censored:
        Rows carried a ``<``/``>`` qualifier; the magnitude was used
        and the row kept (counted inside ``rows_valid`` too).
    rows_valid:
        Rows that survived parsing and the range filter.
    duplicates_collapsed:
        Valid rows removed by the aggregate rule (0 when
        ``aggregate="keep"``).
    rows_out:
        Rows in the returned frame.
    """

    smiles_col: str
    endpoint_col: str
    unit: str
    aggregate: str
    upper_limit: float
    rows_in: int
    rows_missing_smiles: int
    rows_unparsable: int
    rows_out_of_range: int
    rows_censored: int
    rows_valid: int
    duplicates_collapsed: int
    rows_out: int

    def as_dict(self) -> dict[str, Any]:
        """Flat dict for DataFrame/JSON consumption."""
        return {
            "smiles_col": self.smiles_col,
            "endpoint_col": self.endpoint_col,
            "unit": self.unit,
            "aggregate": self.aggregate,
            "upper_limit": self.upper_limit,
            "rows_in": self.rows_in,
            "rows_missing_smiles": self.rows_missing_smiles,
            "rows_unparsable": self.rows_unparsable,
            "rows_out_of_range": self.rows_out_of_range,
            "rows_censored": self.rows_censored,
            "rows_valid": self.rows_valid,
            "duplicates_collapsed": self.duplicates_collapsed,
            "rows_out": self.rows_out,
        }

    def __str__(self) -> str:
        return (
            f"smiles={self.smiles_col!r} endpoint={self.endpoint_col!r} "
            f"unit={self.unit} aggregate={self.aggregate}\n"
            f"rows: {self.rows_in} in -> {self.rows_out} out "
            f"(valid {self.rows_valid}, duplicates collapsed "
            f"{self.duplicates_collapsed})\n"
            f"dropped: {self.rows_missing_smiles} missing SMILES, "
            f"{self.rows_unparsable} unparsable, "
            f"{self.rows_out_of_range} above {self.upper_limit:g} "
            f"{self.unit}\n"
            f"censored magnitudes used: {self.rows_censored}"
        )


#: Column-name candidates for the SMILES side of
#: :func:`prepare_pic50_dataset` (D2.py's ``Ligand SMILES`` first --
#: the upload's own spelling -- then the common alternatives).
_SMILES_COLUMN_CANDIDATES: tuple[str, ...] = (
    "Ligand SMILES",
    "SMILES",
    "smiles",
    "ligand_smiles",
)

#: Column-name candidates for the endpoint side (D2.py's ``EC50 (nM)``
#: first, then the IC50 sibling and the bare spellings).
_ENDPOINT_COLUMN_CANDIDATES: tuple[str, ...] = (
    "EC50 (nM)",
    "IC50 (nM)",
    "EC50_nM",
    "IC50_nM",
    "EC50",
    "IC50",
)


def _resolve_endpoint_column(
    frame: pd.DataFrame,
    explicit: str | None,
    candidates: tuple[str, ...],
    role: str,
) -> str:
    """Pick an endpoint-prep column by explicit name or candidate list."""
    if explicit is not None:
        if explicit not in frame.columns:
            raise DataValidationError(
                f"{role} column {explicit!r} not found; available columns: "
                + ", ".join(repr(str(name)) for name in frame.columns)
            )
        return explicit
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise DataValidationError(
        f"no {role} column found; tried "
        + ", ".join(repr(str(name)) for name in candidates)
        + ". Available columns: "
        + ", ".join(repr(str(name)) for name in frame.columns)
    )


def prepare_pic50_dataset(
    frame: pd.DataFrame,
    smiles_col: str | None = None,
    endpoint_col: str | None = None,
    unit: str = "nM",
    out_col: str = "pEC50",
    upper_limit: float = 1e9,
    aggregate: str = "median",
) -> tuple[pd.DataFrame, EndpointDatasetReport]:
    """Turn a raw SMILES + EC50 table into a modelling-ready pEC50 set.

    The upgraded equivalent of ``D2.py``'s ``process_data``: the
    original hard-coded ``Ligand SMILES`` / ``EC50 (nM)`` column pair,
    silently stripped ``<``/``>`` qualifiers, a silent ``0 < EC50 <=
    1e9`` range filter and a silent per-SMILES median collapse.  Here
    every one of those steps is a named, counted, reversible choice:

    * the SMILES and endpoint columns are auto-detected from the
      upload's own spellings (or given explicitly),
    * censored ``<``/``>`` entries keep their magnitude and are counted
      in the report instead of being stripped in silence,
    * the range filter (positive values at or below ``upper_limit``)
    * duplicate SMILES collapse to one row under ``aggregate``
      (``"median"`` -- D2.py's own rule -- ``"mean"``, ``"first"`` or
      ``"keep"`` for no collapse),
    * the conversion rides :func:`ic50_to_pic50`, so ``pEC50 =
      -log10(endpoint [M])`` with the input ``unit`` respected.

    Parameters
    ----------
    frame:
        Table with a SMILES column and an endpoint column.
    smiles_col:
        Explicit SMILES column (default: auto-detect).
    endpoint_col:
        Explicit endpoint column (default: auto-detect).
    unit:
        Unit of the endpoint values (see :func:`ic50_to_pic50`).
    out_col:
        Name of the converted column in the result.
    upper_limit:
        Magnitude cap in the input unit (D2.py's 1e9 nM default; a
        value ABOVE the cap drops the row as out-of-range).
    aggregate:
        ``"median"``, ``"mean"``, ``"first"`` or ``"keep"``.

    Returns
    -------
    tuple
        ``(result, report)``.  ``result`` carries the SMILES column
        (renamed ``SMILES``), ``n_measurements``, the aggregated raw
        magnitude in the input unit (``endpoint_median`` -- the name is
        literal even for mean/first, see :attr:`aggregate`) and the
        converted ``out_col``; one row per unique SMILES in input
        order.  ``report`` is the :class:`EndpointDatasetReport` audit.

    Raises
    ------
    DataValidationError
        If a requested column is missing, ``aggregate`` or
        ``upper_limit`` is invalid, ``out_col`` collides with one of
        the reserved result columns (``SMILES`` / ``n_measurements`` /
        ``endpoint_median`` -- the collision would silently clobber
        that column, so it is refused up front), or no row survives
        the pipeline.

    Examples
    --------
    >>> frame = pd.DataFrame(
    ...     {"Ligand SMILES": ["CCO", "CCO", "C", ""],
    ...      "EC50 (nM)": ["<50", 200.0, "junk", 10.0]})
    >>> out, report = prepare_pic50_dataset(frame)
    >>> out["SMILES"].tolist(), out["n_measurements"].tolist()
    (['CCO'], [2])
    >>> round(float(out["pEC50"].iloc[0]), 3)   # median(50, 200) nM
    6.903
    >>> (report.rows_in, report.rows_valid, report.rows_out)
    (4, 2, 1)
    >>> (report.rows_censored, report.rows_unparsable,
    ...  report.rows_missing_smiles, report.duplicates_collapsed)
    (1, 1, 1, 1)
    >>> prepare_pic50_dataset(frame, out_col="SMILES")   # reserved name
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: out_col 'SMILES' collides with a reserved result column (SMILES, n_measurements, endpoint_median) -- pick a different name for the converted endpoint
    """
    if aggregate not in ("median", "mean", "first", "keep"):
        raise DataValidationError(
            "aggregate must be 'median', 'mean', 'first' or 'keep', got "
            f"{aggregate!r}"
        )
    if not (math.isfinite(upper_limit) and upper_limit > 0):
        raise DataValidationError(
            f"upper_limit must be a positive finite number, got {upper_limit!r}"
        )
    # (100x-146) a colliding out_col would silently clobber its twin in
    # the result dict (the round-146 audit's duplicate-key finding) --
    # refuse it up front instead.
    _reserved_result_columns = ("SMILES", "n_measurements", "endpoint_median")
    if out_col in _reserved_result_columns:
        raise DataValidationError(
            f"out_col {out_col!r} collides with a reserved result column "
            f"({', '.join(_reserved_result_columns)}) -- pick a different "
            "name for the converted endpoint"
        )
    # Validate the unit early (a bad unit must not surface as NaN rows
    # deep inside the conversion -- ic50_to_pic50 raises on unknown units).
    ic50_to_pic50(1.0, unit=unit)

    smiles_name = _resolve_endpoint_column(frame, smiles_col, _SMILES_COLUMN_CANDIDATES, "SMILES")
    endpoint_name = _resolve_endpoint_column(
        frame, endpoint_col, _ENDPOINT_COLUMN_CANDIDATES, "endpoint"
    )

    rows_in = len(frame)
    rows_missing_smiles = 0
    rows_unparsable = 0
    rows_out_of_range = 0
    rows_censored = 0

    kept: list[tuple[str, float]] = []
    for smiles_value, endpoint_value in zip(
        frame[smiles_name], frame[endpoint_name], strict=False
    ):
        smiles_text = "" if smiles_value is None else str(smiles_value).strip()
        if not smiles_text or smiles_text.lower() in {"nan", "none", "null"}:
            rows_missing_smiles += 1
            continue
        parsed = parse_endpoint(endpoint_value)
        if parsed is None:
            rows_unparsable += 1
            continue
        if parsed.value > upper_limit:
            rows_out_of_range += 1
            continue
        if parsed.is_censored:
            rows_censored += 1
        kept.append((smiles_text, parsed.value))

    rows_valid = len(kept)
    if not kept:
        raise DataValidationError(
            "no usable rows survived the endpoint preparation "
            f"({rows_in} in: {rows_missing_smiles} missing SMILES, "
            f"{rows_unparsable} unparsable, {rows_out_of_range} out of range)"
        )

    magnitudes = pd.Series([value for _smiles, value in kept], dtype=float)
    if aggregate == "keep":
        smiles_out = [smiles for smiles, _value in kept]
        values_out = magnitudes
        n_measurements = pd.Series(1, index=range(len(kept)), dtype=int)
        duplicates_collapsed = 0
    else:
        work = pd.DataFrame({"SMILES": [smiles for smiles, _value in kept],
                             "value": magnitudes})
        how = {"median": "median", "mean": "mean", "first": "first"}[aggregate]
        grouped = work.groupby("SMILES", sort=False)["value"]
        values_out = grouped.agg(how).reset_index(drop=True)
        smiles_out = work["SMILES"].drop_duplicates().tolist()
        n_measurements = grouped.count().reset_index(drop=True).astype(int)
        duplicates_collapsed = rows_valid - len(smiles_out)

    pic50 = [ic50_to_pic50(value, unit=unit) for value in values_out]
    result = pd.DataFrame(
        {
            "SMILES": smiles_out,
            "n_measurements": n_measurements.to_numpy(),
            "endpoint_median": values_out.to_numpy(),
            out_col: pic50,
        }
    )
    report = EndpointDatasetReport(
        smiles_col=smiles_name,
        endpoint_col=endpoint_name,
        unit=unit,
        aggregate=aggregate,
        upper_limit=float(upper_limit),
        rows_in=rows_in,
        rows_missing_smiles=rows_missing_smiles,
        rows_unparsable=rows_unparsable,
        rows_out_of_range=rows_out_of_range,
        rows_censored=rows_censored,
        rows_valid=rows_valid,
        duplicates_collapsed=duplicates_collapsed,
        rows_out=len(result),
    )
    _LOG.info(
        "prepared pEC50 dataset: %d rows in -> %d out (%d censored magnitudes "
        "used, %d duplicates collapsed by %s)",
        rows_in,
        len(result),
        rows_censored,
        duplicates_collapsed,
        aggregate,
    )
    return result, report


def split_table_into_chunks(
    input_path: str | Path,
    output_dir: str | Path,
    chunk_rows: int = 1_500_000,
    prefix: str | None = None,
) -> list[Path]:
    """Stream a large CSV into fixed-size part files.

    Upgrades ``Split CSV file.py``: instead of writing parts next to a
    hardcoded input, chunks land in an explicit output directory with a
    generated manifest-friendly naming scheme, and memory stays flat
    because the file is read lazily.

    Parameters
    ----------
    input_path:
        CSV (or Excel) file to split.
    output_dir:
        Destination directory (created if missing).
    chunk_rows:
        Maximum rows per part.
    prefix:
        Base name for parts; defaults to the input stem.

    Returns
    -------
    list of Path
        Paths of the written part files, in order.
    """
    if chunk_rows < 1:
        raise DataValidationError(f"chunk_rows must be >= 1, got {chunk_rows}")
    input_file = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = prefix or input_file.stem

    written: list[Path] = []
    if input_file.suffix.lower() == ".csv":
        iterator: Iterable[pd.DataFrame] = pd.read_csv(
            input_file, chunksize=chunk_rows
        )
    else:
        frame = read_table(input_file)
        iterator = (
            frame.iloc[start : start + chunk_rows]
            for start in range(0, len(frame), chunk_rows)
        )

    for index, chunk in enumerate(iterator, start=1):
        part_path = out_dir / f"{base}_part{index:03d}.csv"
        chunk.to_csv(part_path, index=False)
        written.append(part_path)
        _LOG.info("Wrote %s (%d rows)", part_path, len(chunk))
    return written


# ---------------------------------------------------------------------------
# Molecular file-format converters (.smi <-> table <-> .sdf)
# ---------------------------------------------------------------------------


def _require_rdkit():
    """Import the RDKit ``Chem`` module or raise an actionable error.

    Mirrors :func:`pharmaml.chem.descriptors.require_rdkit` but with a
    message scoped to file-format conversion.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise DependencyError(
            "RDKit is required for molecular file-format conversion. "
            "Install it with:  pip install 'pharmaml[chem]'  "
            "(or pip install rdkit)"
        ) from exc
    return Chem, AllChem


def _iter_smi_lines(smi_path: Path) -> Iterable[tuple[str, str]]:
    """Stream ``(smiles, identifier)`` pairs from a ``.smi`` file.

    Blank lines are skipped and the identifier is the whitespace-joined
    remainder of the line (possibly empty) -- exactly the parsing of
    ``Convert SMI to CSV.py``, minus its in-memory list.
    """
    with open(smi_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            yield parts[0], " ".join(parts[1:])


def smi_to_csv(smi_path: str | Path, out_csv: str | Path | None = None) -> Path:
    """Stream a ``.smi`` file to CSV one line at a time.

    Upgrades ``Convert SMI to CSV.py``: rows are written as they are
    read (no full in-memory list, so million-ligand files stay flat on
    memory), multi-word identifiers survive as a single ID cell, blank
    lines are skipped, and the output carries a ``utf-8-sig`` BOM so
    Excel opens it without mangling the header.

    Parameters
    ----------
    smi_path:
        Input ``.smi`` file -- one ``<SMILES> <ID>`` pair per line.
    out_csv:
        Output CSV path; defaults to ``<stem>.csv`` next to the input.

    Returns
    -------
    Path
        The written CSV file (columns ``SMILES`` and ``ID``).

    Raises
    ------
    DataValidationError
        If the input file does not exist.
    """
    src = Path(smi_path)
    if not src.is_file():
        raise DataValidationError(f"SMILES file not found: {src}")
    out = Path(out_csv) if out_csv is not None else src.with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with open(out, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES", "ID"])
        for smiles, identifier in _iter_smi_lines(src):
            writer.writerow([smiles, identifier])
            n_rows += 1
    _LOG.info("Converted %d SMILES lines: %s -> %s", n_rows, src, out)
    return out


def _cell_text(raw: object) -> str | None:
    """Stringify a name/tag cell; NaN/blank cells return ``None``.

    Numeric identifiers lose their trailing ``.0`` (``1.0`` -> ``"1"``),
    mirroring :func:`pharmaml.docking.ligands._row_name`.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def _load_conversion_table(data: pd.DataFrame | str | Path) -> tuple[pd.DataFrame, Path]:
    """Accept a DataFrame or a CSV/TSV/Excel path via :func:`read_table`."""
    if isinstance(data, pd.DataFrame):
        return data, Path("<dataframe>")
    if not isinstance(data, (str, Path)):
        raise DataValidationError(
            f"data must be a DataFrame or a table path, got {type(data).__name__}"
        )
    source = Path(data)
    if not source.is_file():
        raise DataValidationError(f"input table not found: {source}")
    return read_table(source), source


@dataclass(slots=True)
class SdfWriteReport:
    """Audit of one :func:`table_to_sdf` conversion.

    Attributes
    ----------
    source:
        Table file (or the ``"<dataframe>"`` marker) the molecules came
        from.
    output_sdf:
        SDF file that was written.
    molecules:
        Number of molecules written.
    failures:
        Rows whose SMILES failed to parse (counted, not written).
    smiles_column:
        Column the SMILES were read from.
    embedded_3d:
        Molecules that received ETKDG 3D coordinates.
    notes:
        Non-fatal remarks: embedding fallbacks, UFF convergence.
    failed_smiles:
        The raw SMILES strings that failed -- the same accounting the
        :class:`~pharmaml.docking.ligands.LigandLibrary` keeps for its
        ``failed_smiles.csv`` exports.
    """

    source: Path
    output_sdf: Path
    molecules: int
    failures: int
    smiles_column: str
    embedded_3d: int
    notes: list[str]
    failed_smiles: list[str]

    def __str__(self) -> str:
        lines = [
            "Table -> SDF conversion",
            f"  source        : {self.source}",
            f"  output sdf    : {self.output_sdf}",
            f"  molecules     : {self.molecules}",
            f"  3D embedded   : {self.embedded_3d}",
            f"  failures      : {self.failures}"
            + ("  (invalid SMILES, skipped)" if self.failures else ""),
            f"  smiles column : {self.smiles_column}",
        ]
        lines.extend(f"  note          : {note}" for note in self.notes)
        return "\n".join(lines)


def table_to_sdf(
    data: pd.DataFrame | str | Path,
    out_sdf: str | Path,
    *,
    smiles_col: str = "SMILES",
    name_col: str | None = None,
    embed3d: bool = True,
    uff: bool = True,
    seed: int = 0xF10E,
) -> SdfWriteReport:
    """Write a SMILES table to an SD file with 3D structures.

    Upgrades ``Convert CSV to SDF.py``: the original hard-coded
    ``yyy.csv``/``ligands.sdf``, printed ``Invalid SMILES at index i``
    and silently lost the row.  Here every non-SMILES column becomes an
    SD tag, invalid SMILES are counted in the returned
    :class:`SdfWriteReport` (and never abort the batch), and each
    molecule is AddHs -> ETKDG-embedded with a fixed ``seed`` ->
    UFF-optimised.  Embedding or UFF failures are non-fatal: the
    molecule is kept without coordinates and a note is recorded.

    Parameters
    ----------
    data:
        CSV/TSV/Excel path or a DataFrame (read via
        :func:`pharmaml.utils.io.read_table`).
    out_sdf:
        Output ``.sdf`` path.
    smiles_col:
        Column holding the SMILES.
    name_col:
        Column used for molecule titles; empty/NaN cells fall back to
        the 1-based row number.  ``None`` reproduces the rar's
        ``ligand_{i}`` numbering.
    embed3d:
        Add hydrogens and embed 3D coordinates (ETKDG, fixed seed).
    uff:
        UFF energy minimisation after embedding (with a convergence
        check that only *notes* non-convergence).
    seed:
        Random seed for the embedding (reproducible geometries).

    Returns
    -------
    SdfWriteReport

    Raises
    ------
    DataValidationError
        Missing columns, an empty table, or a table where no SMILES
        parses (nothing would be written).
    DependencyError
        If RDKit is not installed.
    """
    frame, source = _load_conversion_table(data)
    require_columns(frame, [smiles_col])
    if name_col is not None and name_col not in frame.columns:
        raise DataValidationError(
            f"name column {name_col!r} not found -- available columns: "
            f"{list(frame.columns)}"
        )
    if frame.empty:
        raise DataValidationError(f"table has no rows: {source}")

    Chem, AllChem = _require_rdkit()
# rdkit-stubs does not re-export DisableLog/EnableLog on rdkit.RDLogger,
    # but rdBase (their defining module) is fully typed.
    from rdkit.Chem import SDWriter
    from rdkit.rdBase import DisableLog, EnableLog

    embed_params = None
    if embed3d:
        embed_params = AllChem.ETKDG()
        embed_params.randomSeed = seed

    tag_columns = [col for col in frame.columns if col != smiles_col]
    prepared: list = []
    failures: list[str] = []
    notes: list[str] = []
    embedded = 0
    columns = list(frame.columns)

    # RDKit logs parse errors to stderr before returning None; the
    # report below is friendlier than the raw parser trace.
    DisableLog("rdApp.error")
    try:
        for i, record in enumerate(frame.itertuples(index=False, name=None), start=1):
            values = dict(zip(columns, record))
            raw_smiles = values[smiles_col]
            # Blank cells must count as parse failures: RDKit parses "" as
            # a *valid* 0-atom molecule, which then crashes the 3D embedder
            # ("molecule has no atoms") or silently writes an empty record.
            # Same guard family as chem/descriptors._smiles_to_molecules.
            mol = (
                Chem.MolFromSmiles(raw_smiles)
                if isinstance(raw_smiles, str) and raw_smiles.strip()
                else None
            )
            if mol is None:
                failures.append(str(raw_smiles))
                _LOG.warning("Invalid SMILES at row %d: %r", i, raw_smiles)
                continue

            if name_col is not None:
                name = _cell_text(values.get(name_col)) or str(i)
            else:
                name = f"ligand_{i}"  # rar parity: ligand_1, ligand_2, ...
            mol.SetProp("_Name", name)
            for col in tag_columns:
                cell = values[col]
                if cell is None or (isinstance(cell, float) and pd.isna(cell)):
                    continue  # missing tag values are simply not written
                mol.SetProp(col, str(cell))

            mol = Chem.AddHs(mol)
            if embed_params is not None:
                if AllChem.EmbedMolecule(mol, embed_params) != 0:
                    notes.append(
                        f"3D embedding failed for row {i} ({raw_smiles}) "
                        "-- kept without coordinates"
                    )
                else:
                    embedded += 1
                    if uff and AllChem.UFFOptimizeMolecule(mol) != 0:
                        notes.append(f"UFF optimisation did not converge for row {i}")
            prepared.append(mol)
    finally:
        EnableLog("rdApp.error")

    if not prepared:
        raise DataValidationError(
            f"no valid molecules to write -- {len(failures)} of {len(frame)} "
            "SMILES failed to parse"
        )

    out_path = Path(out_sdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = SDWriter(str(out_path))
    try:
        for mol in prepared:
            writer.write(mol)
    finally:
        writer.close()

    report = SdfWriteReport(
        source=source,
        output_sdf=out_path,
        molecules=len(prepared),
        failures=len(failures),
        smiles_column=smiles_col,
        embedded_3d=embedded,
        notes=notes,
        failed_smiles=failures,
    )
    _LOG.info(
        "Wrote %d molecules to %s (%d embedded, %d failures)",
        len(prepared),
        out_path,
        embedded,
        len(failures),
    )
    return report


def sdf_to_table(
    sdf_path: str | Path,
    out: str | Path | None = None,
    smiles_col: str = "SMILES",
) -> pd.DataFrame:
    """Read an SD file back into a table (SMILES, tags and titles).

    Upgrades the ``Sdf reading.ipynb`` notebook: instead of
    notebook-only ``PandasTools.LoadSDF`` plus ad-hoc
    ``df['SMILES'] = df.ROMol.apply(Chem.MolToSmiles)``, every molecule
    becomes one row with its canonical SMILES, its title in a ``name``
    column and every SD property as a typed column (numbers come back
    as numbers via ``GetPropsAsDict``).

    Records that fail to parse are never dropped *silently*: the count
    is reported with a warning (``"N of M SDF records unparseable and
    dropped"``) and recorded on ``frame.attrs`` as
    ``sdf_dropped_records`` / ``sdf_total_records`` so CLI callers can
    exit non-zero -- the mirror of :func:`table_to_sdf`'s failure
    accounting (``data convert`` exits 1 on any drop).  A file whose
    records are *all* unparseable still raises.

    Parameters
    ----------
    sdf_path:
        Input ``.sdf`` file.
    out:
        Optional output table path (CSV/TSV/Excel via
        :func:`pharmaml.utils.io.write_table`).
    smiles_col:
        Name of the SMILES column in the returned frame.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    DataValidationError
        Missing file, or an empty/unparseable SDF.
    DependencyError
        If RDKit is not installed.
    """
    src = Path(sdf_path)
    if not src.is_file():
        raise DataValidationError(f"SDF file not found: {src}")
    Chem, _AllChem = _require_rdkit()
    # rdkit-stubs does not re-export DisableLog/EnableLog on rdkit.RDLogger,
    # but rdBase (their defining module) is fully typed (cf. table_to_sdf).
    from rdkit.rdBase import DisableLog, EnableLog

    rows: list[dict] = []
    total = 0
    # RDKit logs parse errors to stderr before returning None; the
    # warning below is the single typed report (same guard as
    # table_to_sdf).
    DisableLog("rdApp.error")
    try:
        supplier = Chem.SDMolSupplier(str(src))
        total = len(supplier)  # record count, dropped records included
        for mol in supplier:
            if mol is None:
                continue  # unparseable record: dropped, counted below
            row: dict[str, object] = dict(
                mol.GetPropsAsDict(includePrivate=False, includeComputed=False)
            )
            row[smiles_col] = Chem.MolToSmiles(mol)
            row["name"] = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            rows.append(row)
    except OSError as exc:
        raise DataValidationError(f"empty or unreadable SDF file: {src}") from exc
    finally:
        EnableLog("rdApp.error")

    if not rows:
        raise DataValidationError(f"no parseable molecules found in {src}")

    dropped = total - len(rows)
    if dropped:
        _LOG.warning("%d of %d SDF records unparseable and dropped", dropped, total)

    frame = pd.DataFrame(rows)
    ordered = [smiles_col, "name"] + [
        col for col in frame.columns if col not in (smiles_col, "name")
    ]
    frame = frame[ordered]
    # Drop accounting for downstream callers (the CLI exits 1 on any drop).
    frame.attrs["sdf_dropped_records"] = dropped
    frame.attrs["sdf_total_records"] = total
    _LOG.info("Read %d molecules from %s", len(frame), src)
    if out is not None:
        written = write_table(frame, out)
        _LOG.info("Wrote %d rows to %s", len(frame), written)
    return frame


def table_to_smi(
    data: pd.DataFrame | str | Path,
    out_smi: str | Path,
    *,
    smiles_col: str = "SMILES",
    name_col: str | None = None,
) -> Path:
    """Write a SMILES table to a ``.smi`` list file.

    The inverse of :func:`smi_to_csv`: each row becomes one
    ``<smiles> <name>`` line (the name is omitted when ``name_col`` is
    not given or the cell is empty), blank SMILES cells are skipped the
    way blank ``.smi`` lines are, and validation mirrors
    :func:`smi_to_csv` (missing file -> :class:`DataValidationError`).

    Parameters
    ----------
    data:
        CSV/TSV/Excel path or a DataFrame.
    out_smi:
        Output ``.smi`` path.
    smiles_col:
        Column holding the SMILES.
    name_col:
        Column written after the SMILES; ``None`` writes bare SMILES.

    Returns
    -------
    Path
        The written ``.smi`` file.

    Raises
    ------
    DataValidationError
        Missing file, missing columns, or an empty table.
    """
    frame, source = _load_conversion_table(data)
    require_columns(frame, [smiles_col])
    if name_col is not None and name_col not in frame.columns:
        raise DataValidationError(
            f"name column {name_col!r} not found -- available columns: "
            f"{list(frame.columns)}"
        )
    if frame.empty:
        raise DataValidationError(f"table has no rows: {source}")

    out_path = Path(out_smi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(frame.columns)
    n_rows = 0
    with open(out_path, "w", encoding="utf-8") as handle:
        for record in frame.itertuples(index=False, name=None):
            values = dict(zip(columns, record))
            smiles = _cell_text(values[smiles_col])
            if smiles is None:
                continue  # blank SMILES cell: skipped like blank .smi lines
            name = _cell_text(values.get(name_col)) if name_col is not None else None
            handle.write(smiles + (f" {name}" if name else "") + "\n")
            n_rows += 1
    _LOG.info("Wrote %d SMILES lines to %s", n_rows, out_path)
    return out_path
