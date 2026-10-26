"""Task 2-a (100x-146) probe B -- round-145 library functions + io.py edges.

Verifies (pure library, no Tk):
  P. prepare_pic50_dataset: the accounting identity on a mixed table; the
     out_col COLLISION edge (out_col naming a built-in column silently
     drops that column -- last dict key wins); a censored row ABOVE the
     cap lands in rows_out_of_range, NOT rows_censored; 'first' and
     'keep' aggregate rules;
  Q. permutation_feature_importance: bool n_repeats rejected; X/y length
     mismatch named; ranking ties first-declared-wins; as_frame twin;
     plot_permutation_importance accepts BOTH the record and the frame;
     negative bars use the red palette slot, positive the teal one;
     a ZERO-feature record still draws (no crash);
  R. ecfp3: compute_feature_frame over families=['ecfp','ecfp3'] gives
     the 2048+2048 columns with the right prefixes; family_of_column is
     the exact inverse; a molecule whose radius-3 environments differ
     from radius-2 yields DIFFERENT bit sets; selection_metadata records
     ecfp3 in fingerprint_sizes;
  S. io.py parity edges the round-144/145 tests do not pin: pipe- and
     tab-delimited RAGGED files (sync vs progressive verdicts must
     agree), the both-separators-choke file, the header-only progressive
     read (no report fires, empty frame back), the 3-argument report
     arity, and the small-file passthrough.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# ------------------------------------------------------------------ P ---- #
from pharmaml.chem import prepare_pic50_dataset

mixed = pd.DataFrame(
    {
        "Ligand SMILES": ["CCO", "CCO", "CCO", "CCN", "", "CCC", "CCN", "CCO"],
        "EC50 (nM)": ["<50", 200.0, 100.0, "junk", 10.0, 2e9, ">3e9", 400.0],
    }
)
out, report = prepare_pic50_dataset(mixed, aggregate="median")
identity = (
    report.rows_valid + report.rows_missing_smiles + report.rows_unparsable
    + report.rows_out_of_range == report.rows_in
)
check("P1. accounting identity holds", identity, f"report={report.as_dict()}")
check(
    "P2. rows_out == rows_valid - duplicates_collapsed",
    report.rows_out == report.rows_valid - report.duplicates_collapsed,
)
# rows: CCO(<50,200,100,400) -> 4 valid, 1 group; CCN junk -> unparsable;
# '' -> missing smiles; CCC 2e9 -> out of range; CCN '>3e9' -> out of range
check(
    "P3. buckets land as designed",
    (report.rows_in, report.rows_valid, report.rows_unparsable,
     report.rows_missing_smiles, report.rows_out_of_range) == (8, 4, 1, 1, 2),
    f"got={(report.rows_in, report.rows_valid, report.rows_unparsable, report.rows_missing_smiles, report.rows_out_of_range)}",
)
check(
    "P4. censored-above-cap is OUT OF RANGE, not censored",
    report.rows_censored == 1 and report.rows_out_of_range == 2,
    f"censored={report.rows_censored} oor={report.rows_out_of_range}",
)
check(
    "P5. median collapse picked median(50,200,100,400)=150 nM",
    abs(float(out.loc[out["SMILES"] == "CCO", "endpoint_median"].iloc[0]) - 150.0) < 1e-9,
)

first_out, first_report = prepare_pic50_dataset(mixed, aggregate="first")
check(
    "P6. 'first' keeps the first replicate magnitude (50 nM)",
    abs(float(first_out.loc[first_out["SMILES"] == "CCO", "endpoint_median"].iloc[0]) - 50.0)
    < 1e-9,
)
keep_out, keep_report = prepare_pic50_dataset(mixed, aggregate="keep")
check(
    "P7. 'keep' collapses nothing",
    keep_report.duplicates_collapsed == 0 and len(keep_out) == keep_report.rows_valid,
)

# the out_col collision edge: a duplicate dict key silently wins -- the
# SMILES strings are LOST and the 'SMILES' column carries the pEC50 floats
collided, _rep = prepare_pic50_dataset(mixed, out_col="SMILES")
check(
    "P8. out_col='SMILES' collides: the SMILES column is silently REPLACED",
    list(collided.columns) == ["SMILES", "n_measurements", "endpoint_median"]
    and collided["SMILES"].dtype.kind == "f",
    f"columns={list(collided.columns)} dtype={collided['SMILES'].dtype}",
)

# ------------------------------------------------------------------ Q ---- #
from pharmaml.ml import permutation_feature_importance
from pharmaml.ml.evaluation import PermutationImportance
from pharmaml.viz import plot_permutation_importance

rng = np.random.default_rng(0)
X = rng.normal(size=(80, 4))
y = 2.0 * X[:, 0] - 0.5 * X[:, 3]
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X, y)

for bad in (True, 0, -1, 2.5):
    try:
        permutation_feature_importance(model, X, y, n_repeats=bad)
        check(f"Q1. n_repeats={bad!r} rejected", False, "no exception")
    except Exception as exc:
        check(f"Q1. n_repeats={bad!r} rejected", "n_repeats" in str(exc), type(exc).__name__)

try:
    permutation_feature_importance(model, X, y[:-1])
    check("Q2. X/y mismatch named", False, "no exception")
except Exception as exc:
    check("Q2. X/y mismatch named", "rows" in str(exc), str(exc)[:60])

imp = permutation_feature_importance(model, X, y, n_repeats=3)
check("Q3. positional names", imp.names == ["x0", "x1", "x2", "x3"])
tie = PermutationImportance(
    names=["a", "b"], means=np.array([0.5, -0.5]),
    stds=np.array([0.01, 0.01]), scoring="r2", n_repeats=1, n_samples=10,
)
check(
    "Q4. |mean| ties resolve first-declared-wins",
    [name for name, _ in tie.ranking()] == ["a", "b"],
    f"got={[n for n, _ in tie.ranking()]}",
)

fig_rec = plot_permutation_importance(tie, title="rec")
ax = fig_rec.axes[0]
rec_ticks = [t.get_text() for t in ax.get_yticklabels()]
fig_frame = plot_permutation_importance(tie.as_frame(), title="frame")
frame_ticks = [t.get_text() for t in fig_frame.axes[0].get_yticklabels()]
check("Q5. record and as_frame twins draw the same ticks", rec_ticks == frame_ticks == ["b", "a"])
# negative bar (b, mean=-0.5) rendered red, positive (a) teal
from pharmaml.viz.ml_plots import PALETTE

recs = [patch for patch in ax.patches]
colors = [patch.get_facecolor() for patch in recs]
import matplotlib.colors as mcolors

check(
    "Q6. negative bar red / positive teal",
    mcolors.to_hex(colors[0], keep_alpha=False).upper() == mcolors.to_hex(PALETTE[3], keep_alpha=False).upper()
    and mcolors.to_hex(colors[1], keep_alpha=False).upper() == mcolors.to_hex(PALETTE[0], keep_alpha=False).upper(),
    f"got={[mcolors.to_hex(c) for c in colors]} palette={[mcolors.to_hex(c) for c in PALETTE[:4]]}",
)
empty = PermutationImportance(
    names=[], means=np.array([]), stds=np.array([]),
    scoring="r2", n_repeats=1, n_samples=0,
)
try:
    fig_empty = plot_permutation_importance(empty)
    check("Q7. zero-feature record still draws", fig_empty.axes[0] is not None)
except Exception as exc:
    check("Q7. zero-feature record still draws", False, f"{type(exc).__name__}: {exc}")

# ------------------------------------------------------------------ R ---- #
from pharmaml.chem.features import (
    ECFP3_BITS,
    ECFP3_RADIUS,
    FEATURE_FAMILIES,
    compute_feature_frame,
    family_of_column,
)

check("R1. FEATURE_FAMILIES order: ecfp3 follows ecfp",
      FEATURE_FAMILIES.index("ecfp3") == FEATURE_FAMILIES.index("ecfp") + 1)
frame, freport = compute_feature_frame(["CCO", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"],
                                       families=["ecfp", "ecfp3"])
ecfp_cols = [c for c in frame.columns if c.startswith("ecfp_bit_")]
ecfp3_cols = [c for c in frame.columns if c.startswith("ecfp3_bit_")]
check("R2. ecfp + ecfp3 produce 2048 + 2048 columns",
      len(ecfp_cols) == 2048 and len(ecfp3_cols) == 2048,
      f"ecfp={len(ecfp_cols)} ecfp3={len(ecfp3_cols)}")
check("R3. family_of_column inverse for ecfp3", family_of_column("ecfp3_bit_7") == "ecfp3")
caffeine = frame.iloc[1]
r2_set = {int(c[len("ecfp_bit_"):]) for c in ecfp_cols if caffeine[c] == 1.0}
r3_set = {int(c[len("ecfp3_bit_"):]) for c in ecfp3_cols if caffeine[c] == 1.0}
check("R4. radius-3 sees a different bit set than radius-2 (caffeine)",
      r2_set != r3_set, f"|r2|={len(r2_set)} |r3|={len(r3_set)} shared={len(r2_set & r3_set)}")
check("R5. constants", ECFP3_BITS == 2048 and ECFP3_RADIUS == 3)
from pharmaml.chem.features import selection_metadata, select_top_features

scores = pd.Series(np.arange(len(frame.columns)), index=frame.columns)
sel = select_top_features(frame, scores, 10)
meta = selection_metadata(sel, freport, k=10, families=["ecfp", "ecfp3"])
check("R6. selection_metadata records ecfp3 in fingerprint_sizes",
      meta["fingerprint_sizes"].get("ecfp3") == 2048,
      f"sizes={meta['fingerprint_sizes']}")

# ------------------------------------------------------------------ S ---- #
from pharmaml.utils.io import DataValidationError, read_table, read_table_with_progress

tmp = Path(tempfile.mkdtemp(prefix="probe146b_"))


def both_ways(name: str, text: str) -> tuple[Any, Any]:
    path = tmp / name
    path.write_text(text, encoding="utf-8", newline="")
    sync = read_table(path)
    reports: list[tuple] = []
    prog = read_table_with_progress(
        path, lambda *a: reports.append(a), chunk_rows=2, size_threshold=1
    )
    return sync, prog, reports


# S1: pipe-delimited ragged body
sync, prog, _ = both_ways("pipe_ragged.csv", "a|b\n1|2\n3|4|5\n")
check("S1. pipe-ragged parity (both single-column mangled)",
      sync.shape == prog.shape and list(sync.columns) == list(prog.columns),
      f"sync={sync.shape} prog={prog.shape}")

# S2: tab-delimited ragged body
sync, prog, _ = both_ways("tab_ragged.csv", "a\tb\n1\t2\n3\t4\t5\n")
check("S2. tab-ragged parity (both single-column mangled)",
      sync.shape == prog.shape and list(sync.columns) == list(prog.columns),
      f"sync={sync.shape} prog={prog.shape}")

# S3a: comma-ragged body (extra fields) -- BOTH readers must give the
# SAME verdict (the semicolon fallback's mangled single-column frame)
sync, prog, _ = both_ways("comma_ragged.csv", "a,b\n1,2,3\n4,5,6,7\n")
check("S3a. comma-ragged parity (both mangled single-column)",
      sync.shape == prog.shape and list(sync.columns) == list(prog.columns)
      and sync.iloc[0, 0] == prog.iloc[0, 0],
      f"sync={sync.shape} {sync.columns.tolist()} prog={prog.shape} {prog.columns.tolist()}")

# S3b: too-FEW fields pad with NaN on both readers -- parity again
sync, prog, _ = both_ways("padded.csv", "a,b\n1;2;3\n4;5\n")
check("S3b. too-few-fields parity (both pad to the comma header)",
      sync.shape == prog.shape and list(sync.columns) == list(prog.columns)
      and bool(sync.isna().values.any()) == bool(prog.isna().values.any()),
      f"sync={sync.shape} prog={prog.shape}")

# S3c: THE PARITY BREAK (the round's real io.py finding) -- a ragged row
# carrying >= 2 extra fields that lands EXACTLY on a chunk boundary is
# SILENTLY TRUNCATED by pandas' chunked (low-memory) C reader while the
# plain read raises ParserError -> read_table raises DataValidationError
# but read_table_with_progress returns a frame with fields silently
# dropped.  Non-boundary ragged rows raise on BOTH paths (verified
# separately below), so the break needs the boundary alignment.
ragged_boundary = tmp / "ragged_boundary.csv"
ragged_boundary.write_text("a,b\n5,6\n7,8,9,0\n1;2;3\n", encoding="utf-8", newline="")
sync_exc = None
prog_frame = None
try:
    read_table(ragged_boundary)
except DataValidationError as exc:
    sync_exc = exc
try:
    prog_frame = read_table_with_progress(
        ragged_boundary, lambda *a: None, chunk_rows=2, size_threshold=1
    )
except DataValidationError:
    prog_frame = None
check(
    "S3c. CONFIRMED DIVERGENCE: sync raises, progressive silently truncates",
    sync_exc is not None and prog_frame is not None
    and list(prog_frame.columns) == ["a,b"]
    and prog_frame.iloc[2, 0] == 1,  # '1;2;3' truncated to 1
    f"sync={type(sync_exc).__name__} prog={None if prog_frame is None else prog_frame.values.tolist()}",
)
# control: the SAME kind of ragged row NOT on a chunk boundary raises on
# both readers (the comma read chokes at row 2 -> both fall to ';', and
# the mid-chunk ragged row raises in the chunked reader too)
ragged_inside = tmp / "ragged_inside.csv"
ragged_inside.write_text(
    "a,b\n5,6\n9,8,7\n7,8\n8,9\n1;2;3\n9,8\n", encoding="utf-8", newline=""
)
sync_exc2 = prog_exc2 = None
try:
    read_table(ragged_inside)
except DataValidationError as exc:
    sync_exc2 = exc
try:
    read_table_with_progress(
        ragged_inside, lambda *a: None, chunk_rows=3, size_threshold=1
    )
except DataValidationError as exc:
    prog_exc2 = exc
check(
    "S3d. control: non-boundary ragged row raises on BOTH readers",
    sync_exc2 is not None and prog_exc2 is not None,
    f"sync={type(sync_exc2).__name__} prog={type(prog_exc2).__name__}",
)
# control: pandas level -- engine='python' restores the raise (the C
# low-memory chunked reader is the culprit; on_bad_lines='error' does not)
import pandas as pd

with pd.read_csv(ragged_boundary, sep=";", chunksize=2) as reader:
    c_chunks = [chunk for chunk in reader]
try:
    pd.read_csv(ragged_boundary, sep=";")
    py_plain_raised = False
except pd.errors.ParserError:
    py_plain_raised = True
check(
    "S3e. control: plain C read raises; chunked C read truncates; boundary row value == 1",
    py_plain_raised and c_chunks[1].iloc[0, 0] == 1,
    f"plain_raised={py_plain_raised} chunk1={c_chunks[1].values.tolist()}",
)

# S4: header-only progressive read -- NO report, empty frame back
hdr = tmp / "header_only.csv"
hdr.write_text("a,b\n", encoding="utf-8", newline="")
reports: list[tuple] = []
frame = read_table_with_progress(hdr, lambda *a: reports.append(a), chunk_rows=2, size_threshold=1)
check("S4. header-only: empty frame, no progress blip",
      len(frame) == 0 and list(frame.columns) == ["a", "b"] and reports == [],
      f"rows={len(frame)} reports={reports}")

# S5: report arity is (done, total, note) -- three positional args
multi = tmp / "multi.csv"
multi.write_text("a,b\n" + "".join(f"{i},{i}\n" for i in range(6)), encoding="utf-8", newline="")
reports = []
frame = read_table_with_progress(multi, lambda *a: reports.append(a), chunk_rows=2, size_threshold=1)
check("S5. report carries 3 args, first = 2 (chunk size), total >= rows",
      reports and len(reports[0]) == 3 and reports[0][0] == 2 and reports[-1][0] == 6,
      f"reports={reports}")
check("S5b. values survive the chunked path", frame["a"].tolist() == list(range(6)))

# S6: small-file passthrough -- the plain reader, no reports
small = tmp / "small.csv"
small.write_text("a,b\n1,2\n", encoding="utf-8", newline="")
reports = []
frame = read_table_with_progress(small, lambda *a: reports.append(a))
check("S6. under the gate: no reports, same frame",
      reports == [] and frame["a"].tolist() == [1])

print()
print("PROBE_B_RESULT:", "ALL_PASS" if not fails else f"FAILURES={fails}")
