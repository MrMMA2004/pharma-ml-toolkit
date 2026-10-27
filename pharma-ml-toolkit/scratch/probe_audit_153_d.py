"""Probe D: filter_low_variance edges + rank_features twin agreement.

Round-36 audit (100x-153). Deterministic, self-contained, no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pharmaml.chem.features import filter_low_variance, rank_features

# ---------------------------------------------------------------- D1: inf column
# An inf-bearing column has an UNDEFINED (NaN) sample variance.  What do
# the twins do with it?
finf = pd.DataFrame({
    "inf": [np.inf, 1.0, 2.0, 3.0],
    "moves": [1.0, 2.0, 3.0, 4.0],
    "flat": [7.0, 7.0, 7.0, 7.0],
})
result = filter_low_variance(finf)
print("D1 kept:", result.kept)
print("D1 dropped ledger:", result.dropped)
scores = rank_features(finf, method="variance")
print("D1 ranker scores:", scores.to_dict())
print("D1 inf kept by filter:", "inf" in result.kept,
      "| ranker score for inf:", float(scores["inf"]))

# ---------------------------------------------------------------- D2: bool threshold / str threshold
for bad in (True, "0.01", np.float64(0.01)):
    try:
        r = filter_low_variance(pd.DataFrame({"a": [1.0, 2.0]}), threshold=bad)  # type: ignore[arg-type]
        print("D2 threshold", repr(bad), "ACCEPTED ->", r.threshold)
    except ValueError as exc:
        print("D2 threshold", repr(bad), "typed:", str(exc)[:60])

# ---------------------------------------------------------------- D3: bool column
fbool = pd.DataFrame({"flag": [True, False, True, True], "flat": [1.0, 1.0, 1.0, 1.0]})
r = filter_low_variance(fbool)
print("D3 bool column kept (var>0):", "flag" in r.kept, "| dropped:", r.dropped)

# ---------------------------------------------------------------- D4: integer columns + large frame
fint = pd.DataFrame({"i": [1, 2, 3, 4000000000], "f": [1, 1, 1, 1]})
r = filter_low_variance(fint)
print("D4 int frame kept:", r.kept, "dropped:", r.dropped)

# ---------------------------------------------------------------- D5: NaN-heavy column
fnan = pd.DataFrame({"hole": [np.nan, np.nan, np.nan, 1.0], "moves": [1.0, 2.0, 3.0, 4.0]})
r = filter_low_variance(fnan)
print("D5 all-NaN-but-one column: kept ->", r.kept, "dropped ->", r.dropped)
print("D5 ledger variance for hole:", r.dropped.get("hole"))

# ---------------------------------------------------------------- D6: dropped ledger completeness
frame = pd.DataFrame({
    "a": [1.0, 1.0, 1.0, 1.0],
    "b": [1.0, 1.0, 1.0, 1.01],
    "c": [1.0, 1.0, 1.0, 1.001],
    "keep": [1.0, 5.0, 3.0, 4.0],
})
r = filter_low_variance(frame)
print("D6 every cut in ledger:", sorted(r.dropped), "kept:", r.kept)
print("D6 all cuts strictly below threshold:",
      all(v < r.threshold for v in r.dropped.values()))

# ---------------------------------------------------------------- D7: pass-through + frame identity
fpass = pd.DataFrame({
    "SMILES": ["CCO", "CCC", "CC", "C"],
    "flat": [7.0, 7.0, 7.0, 7.0],
    "moves": [1.0, 2.0, 3.0, 4.0],
})
r = filter_low_variance(fpass)
print("D7 pass-through kept:", r.kept, "frame cols:", list(r.frame.columns))
print("D7 rows untouched:", len(r.frame) == 4, "| strings intact:", r.frame["SMILES"].tolist())

# ---------------------------------------------------------------- D8: mutation of input?
before = fpass.copy()
_ = filter_low_variance(fpass)
print("D8 input frame untouched:", fpass.equals(before))

# ---------------------------------------------------------------- D9: ranker verdicts agree (finite frame)
frame2 = pd.DataFrame({
    "a": [1.0, 2.0, 3.0, 4.0],
    "b": [2.0, 2.0, 2.0, 2.0],
    "c": [1.0, 1.0, 1.0, 1.1],
})
sc = rank_features(frame2, method="variance")
rf = filter_low_variance(frame2, threshold=0.05)
agree = all(
    (sc[col] >= 0.05) == (col in rf.kept) for col in frame2.columns
)
print("D9 twin agreement on finite frame:", agree)

# ---------------------------------------------------------------- D10: duplicate column labels
fdup = pd.DataFrame(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), columns=["x", "y", "x"])
try:
    r = filter_low_variance(fdup)
    print("D10 duplicate labels accepted:", r.kept, r.dropped)
except Exception as exc:  # noqa: BLE001
    print("D10 duplicate labels raise:", type(exc).__name__, str(exc)[:80])

print("PROBE_D_DONE")
