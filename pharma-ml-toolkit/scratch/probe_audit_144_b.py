"""Task 144-a probe B -- read_table_with_progress edge cases (library only).

Covers: header-only file on the progressive path, 0-byte file, the
ParserError mid-stream fallback, chunked-vs-plain parity (values AND
dtypes), the row-estimate math (incl. CRLF + the header-inclusive +1
the doctest pins), the None-estimate single-giant-line case, and the
duplicate-column hygiene shared by both readers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pharmaml.exceptions import DataValidationError
from pharmaml.utils.io import read_table, read_table_with_progress

tmp = Path(tempfile.mkdtemp(prefix="probe144b_"))
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- header-only
head = tmp / "header_only.csv"
head.write_text("ligand,pic50,notes\n", encoding="utf-8")
seen: list[tuple[int, int | None, str]] = []
frame = read_table_with_progress(head, lambda *a: seen.append(a), chunk_rows=10, size_threshold=1)
plain = read_table(head)
check("header-only: empty frame with the right columns",
      frame.empty and list(frame.columns) == ["ligand", "pic50", "notes"],
      f"shape={frame.shape}")
check("header-only: no progress reports on the empty chunk stream", seen == [], f"seen={seen}")
check("header-only: parity with the plain reader",
      frame.shape == plain.shape and list(frame.columns) == list(plain.columns))

# -------------------------------------------------------------------- 0-byte
zero = tmp / "zero.csv"
zero.write_bytes(b"")
try:
    read_table_with_progress(zero, lambda *a: None, size_threshold=1)
    check("0-byte: raises DataValidationError", False, "no exception")
except DataValidationError as exc:
    check("0-byte: raises DataValidationError", True, str(exc)[:80])

# ------------------------------------------------------- ParserError fallback
ragged = tmp / "ragged.csv"
ragged.write_text("a,b\n1,2\n3,4,5\n", encoding="utf-8")
seen.clear()
prog = read_table_with_progress(ragged, lambda *a: seen.append(a), chunk_rows=10, size_threshold=1)
sync = read_table(ragged)
check("ParserError fallback: progressive == sync shape",
      prog.shape == sync.shape, f"prog={prog.shape} sync={sync.shape}")
check("ParserError fallback: single mangled column both paths",
      list(prog.columns) == list(sync.columns), f"prog={list(prog.columns)}")

# a ragged body that ALSO fails the semicolon re-read -> honest error both ways
worse = tmp / "worse.csv"
worse.write_text("a;b\n1;2\n3;4;5\n", encoding="utf-8")
try:
    read_table_with_progress(worse, lambda *a: None, chunk_rows=10, size_threshold=1)
    prog_err = None
except DataValidationError as exc:
    prog_err = exc
try:
    read_table(worse)
    sync_err = None
except DataValidationError as exc:
    sync_err = exc
check("semicolon-ragged: progressive raises the plain reader's own error",
      prog_err is not None and sync_err is not None,
      f"prog={type(prog_err).__name__ if prog_err else None} sync={type(sync_err).__name__ if sync_err else None}")

# ---------------------------------------------------- chunked/concat parity
import pandas as pd  # noqa: E402

rows = []
for i in range(250):
    rows.append({
        "int_col": i,
        "float_col": i / 7.0 if i % 3 else None,   # NAs interleaved
        "text_col": f"L{i:03d}" if i % 2 else "",
        "mixed_col": str(i) if i % 5 == 0 else i,   # chunk-dependent look
    })
big = pd.DataFrame(rows)
parity_path = tmp / "parity.csv"
big.to_csv(parity_path, index=False)
seen.clear()
prog = read_table_with_progress(parity_path, lambda *a: seen.append(a), chunk_rows=40, size_threshold=1)
sync = read_table(parity_path)
check("parity: identical shape", prog.shape == sync.shape, f"{prog.shape} vs {sync.shape}")
check("parity: identical dtypes",
      list(prog.dtypes.astype(str)) == list(sync.dtypes.astype(str)),
      f"prog={list(prog.dtypes.astype(str))} sync={list(sync.dtypes.astype(str))}")
pd.testing.assert_frame_equal(prog, sync)
check("parity: assert_frame_equal passes", True)
check("parity: report fired per chunk with a stable estimate",
      len(seen) == 7 and seen[0][1] == 251 and seen[-1][0] == 250,
      f"calls={len(seen)} first={seen[0] if seen else None} last={seen[-1] if seen else None}")

# ------------------------------------------------------------ estimate maths
crlf = tmp / "crlf.csv"
crlf.write_bytes(b"a,b\r\n" + b"".join(f"{i},{i}.5\r\n".encode() for i in range(100)))
seen.clear()
frame = read_table_with_progress(crlf, lambda *a: seen.append(a), chunk_rows=25, size_threshold=1)
check("CRLF: 100 data rows parsed", len(frame) == 100, f"len={len(frame)}")
check("CRLF: estimate within 5% of the true data-row count",
      seen[0][1] is not None and abs(seen[0][1] - 101) <= 6,
      f"first estimate={seen[0][1]}")

giant = tmp / "giant_line.csv"
giant.write_bytes(b"a,b\n" + b"1," + b"2" * 200_000 + b"\n")  # first 64KB = one line
seen.clear()
frame = read_table_with_progress(giant, lambda *a: seen.append(a), chunk_rows=10, size_threshold=1)
check("single-giant-line: estimate degrades to None (open-ended)",
      bool(seen) and all(call[1] is None for call in seen),
      f"totals={[c[1] for c in seen[:3]]}")

# ------------------------------------------------- shared column hygiene
dup = tmp / "dup.csv"
dup.write_text("Ligand ,Ligand\n1,2\n", encoding="utf-8")
try:
    read_table(dup)
    check("duplicate-after-strip: plain reader refuses", False, "no error")
except DataValidationError:
    check("duplicate-after-strip: plain reader refuses", True)
try:
    read_table_with_progress(dup, lambda *a: None, chunk_rows=10, size_threshold=1)
    check("duplicate-after-strip: progressive reader refuses", False, "no error")
except DataValidationError:
    check("duplicate-after-strip: progressive reader refuses", True)

# small-file gate: under the default 8 MB threshold -> plain reader, no reports
small = tmp / "small.csv"
small.write_text("a,b\n1,2\n", encoding="utf-8")
seen.clear()
frame = read_table_with_progress(small, lambda *a: seen.append(a))
check("small file: default threshold routes to the plain reader, zero reports",
      seen == [] and len(frame) == 1)

print()
print("PROBE_B_RESULT:", "ALL_PASS" if not fails else f"FAILURES={fails}")
