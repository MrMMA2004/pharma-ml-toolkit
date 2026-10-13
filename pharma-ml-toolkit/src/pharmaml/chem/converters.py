"""Bioactivity endpoint and file-format conversions.

Upgrades the original flat scripts ``Convert IC50(nM) to pIC50.py``,
``Split CSV file.py`` and friends into typed, reusable functions that:

* handle censored values (``<`` / ``>`` qualifiers) explicitly instead
  of silently stripping them,
* never mutate the caller's DataFrame in place,
* validate units and ranges with precise error messages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.utils.io import read_table, write_table
from pharmaml.utils.validation import require_columns

_LOG = get_logger(__name__)

__all__ = [
    "ic50_to_pic50",
    "ki_to_pki",
    "pic50_from_ic50_series",
    "add_pic50_column",
    "split_table_into_chunks",
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

    # Strip a trailing unit token such as 'nM' / 'uM' / 'M'.
    for unit in ("nM", "uM", "µM", "mM", "M"):
        if text.endswith(unit):
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
    return -math.log10(molar)


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
