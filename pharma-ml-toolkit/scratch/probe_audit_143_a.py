"""PROBE 143-a: session-restore label drift + converter flow checks.

Part A -- the Utilities pickers restored by the session (the seven
_SESSION_PATH_ATTRS entries) never re-render their info labels:
_refresh_restored_path_labels() only walks the docking helpers and the
Chemistry converter's _conv_info.  Control: _conv_info DOES refresh,
proving the mechanism works where wired.

Part C -- the converter's "SDF -> table" mode is blocked by the
"No output" guard even though the mode writes no output file
(the result lands in the tree + Save table CSV).

Run: DISPLAY=:99 python scratch/probe_audit_143_a.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import playground  # noqa: E402

RESULTS: list[str] = []


def pump(app: playground.PlaygroundApp, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.update_idletasks()
        app.update()
        time.sleep(0.02)


# quiet the dialogs, recording what fired
seen: list[tuple[str, str]] = []
for name in ("showinfo", "showwarning", "showerror"):
    def spy(title: str, message: str, _n: str = name, **_k: object) -> None:
        seen.append((_n, f"{title} :: {message}"))

    setattr(playground.messagebox, name, spy)

app = playground.PlaygroundApp(persist_session=False)
pump(app, 0.5)

# ---------------- Part A: restored utilities paths ---------------- #
root = Path("scratch").resolve()
merge_folder = root / "shoot142.py"            # an existing file path
cmp_a = root / "smoke140.py"
cmp_b = root / "probe_biology_edges.py"
tool_folder = Path("examples/data").resolve()
tool_out = Path("examples").resolve()
reindex_path = root / "junit140.xml"

# simulate exactly what _restore_session does for the 7 utility attrs
app._util_merge_folder = merge_folder
app._util_merge_split_file = merge_folder
app._util_cmp_a = cmp_a
app._util_cmp_b = cmp_b
app._util_tool_folder = tool_folder
app._util_tool_out = tool_out
app._util_reindex_path = reindex_path

before = {
    "merge": app._util_merge_info.cget("text"),
    "cmp": app._util_cmp_info.cget("text"),
    "tool": app._util_tool_info.cget("text"),
    "reindex": app._util_reindex_info.cget("text"),
}

app._refresh_restored_path_labels()
pump(app, 0.2)

after = {
    "merge": app._util_merge_info.cget("text"),
    "cmp": app._util_cmp_info.cget("text"),
    "tool": app._util_tool_info.cget("text"),
    "reindex": app._util_reindex_info.cget("text"),
}

print("=== Part A: utilities labels after _refresh_restored_path_labels ===")
stale = []
for key in before:
    print(f"[{key}] before={before[key]!r}")
    print(f"[{key}] after ={after[key]!r}")
    if before[key] == after[key] and "pick" in str(after[key]).lower() or before[key] == after[key]:
        stale.append(key)
print("STALE (label unchanged while the attr holds a restored path):", stale)

# control: the Chemistry converter label DOES refresh
app._conv_input = cmp_a
app._conv_output = None
app._refresh_restored_path_labels()
pump(app, 0.2)
print("control _conv_info:", repr(app._conv_info.cget("text")))
control_ok = "smoke140.py" in str(app._conv_info.cget("text"))
print("control refreshed (converter works, utilities do not):", control_ok)

# ---------------- Part C: SDF -> table needs no output ------------- #
print()
print("=== Part C: 'SDF -> table' mode blocked by the unused output guard ===")
seen.clear()
app._conv_mode.set("SDF → table")
app._conv_input = cmp_a
app._conv_output = None
app._conv_run()
pump(app, 0.5)
fired = [msg for kind, msg in seen]
print("dialogs fired:", fired)
blocked = any("No output" in msg for msg in fired)
print("BLOCKED by 'No output' guard (mode writes no file):", blocked)

# contrast: pIC50 mode, also in-memory, is NOT blocked
seen.clear()
app._conv_mode.set("append pIC50 column")
app._conv_input = cmp_a
app._conv_output = None
app._conv_run()
pump(app, 1.5)  # read_table over a .py fails in the worker -> error dialog
pic50_blocked = any("No output" in msg for kind, msg in seen)
print("pIC50 mode blocked by the same guard (it is not):", pic50_blocked)

app.destroy()
ok_a = bool(stale) and control_ok
print()
print(f"PROBE143A_PART_A={'CONFIRMED' if ok_a else 'REFUTED'} (stale={stale})")
print(f"PROBE143A_PART_C={'CONFIRMED' if blocked and not pic50_blocked else 'REFUTED'}")
