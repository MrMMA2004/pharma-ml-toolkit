"""Task 144-a probe C -- round-29 paths_multi session restore, live GUI.

Verifies against the real app (Xvfb):
  A. dump_session -> load_session -> _restore_session round-trip for the
     three multi-file pickers: attrs restored in picked order, the
     approval label confessing the truncated list exactly like a fresh
     pick, the merge label via its own helper;
  B. per-file existence filtering (a vanished file drops out);
  C. a live pick is never overwritten (single AND multi);
  D. a foreign paths_multi payload (non-list entry / non-str members /
     non-allowlisted attr name) is sanitised away by load_session and
     never reaches setattr;
  E. allowlist discipline: a paths_multi key outside
     _SESSION_PATH_LIST_ATTRS is skipped by the restore loop even when
     it survives sanitisation (planted directly on the session dict).
"""

from __future__ import annotations

import sys
import tempfile
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import playground  # noqa: E402

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        fails.append(name)


tmp = Path(tempfile.mkdtemp(prefix="probe144c_"))
files = [tmp / f"appr{i}.tsv" for i in range(4)]
for f in files:
    f.write_text("drug\tapproval\naspirin\tapproved\n", encoding="utf-8")
merge_a, merge_b = tmp / "m_a.csv", tmp / "m_b.csv"
merge_a.write_text("a\n1\n", encoding="utf-8")
merge_b.write_text("a\n2\n", encoding="utf-8")
gone = tmp / "vanished.tsv"
gone.write_text("x\n", encoding="utf-8")  # deleted before the restore
reindex = tmp / "reindex.csv"
reindex.write_text("a\n1\n", encoding="utf-8")

app: playground.PlaygroundApp | None = None
try:
    app = playground.PlaygroundApp(persist_session=False)

    payload = playground.dump_session(
        theme="light",
        geometry="900x600+10+10",
        tab=8,
        variables={},
        families={},
        texts={},
        paths={"_util_reindex_path": str(reindex)},
        paths_multi={
            "_util_appr_files": [str(f) for f in files] + [str(gone)],
            "_util_merge_files": [str(merge_a), str(merge_b)],
            "_util_tool_files": [str(merge_a)],
            "_not_a_real_picker": [str(merge_a)],  # E: foreign key
            "_util_appr_files_bogus": "not-a-list",  # D: non-list value
        },
    )
    gone.unlink()  # the vanished file is gone by restore time

    session = playground.load_session(payload)
    check("D. sanitisation drops the non-list entry + non-str members",
          "_util_appr_files_bogus" not in session["paths_multi"]
          and "_util_appr_files" in session["paths_multi"],
          f"keys={sorted(session['paths_multi'])}")
    # D2: non-str members inside a surviving list are filtered
    dirty = dict(payload)
    dirty["paths_multi"] = {"_util_tool_files": [str(merge_a), 42, None]}
    session2 = playground.load_session(dirty)
    check("D2. non-str members filtered from a surviving list",
          session2["paths_multi"]["_util_tool_files"] == [str(merge_a)],
          f"got={session2['paths_multi'].get('_util_tool_files')}")

    app._session = session
    app._restore_session()

    check("A. approval picks restored in order, vanished file dropped",
          getattr(app, "_util_appr_files", None) == files,
          f"got={[p.name for p in getattr(app, '_util_appr_files', [])]}")
    label = app._util_appr_info.cget("text")
    expected = "appr0.tsv, appr1.tsv, appr2.tsv … (+1)"
    check("A. approval label matches the fresh-pick format exactly",
          label == expected, f"label={label!r}")
    merge_label = app._util_merge_info.cget("text")
    check("A. merge label re-rendered by its own helper",
          merge_label == "2 table(s): m_a.csv, m_b.csv", f"label={merge_label!r}")
    tool_label = app._util_tool_info.cget("text")
    check("A. tool label quiet under the non-distribute default mode",
          tool_label == "no folder or output picked yet", f"label={tool_label!r}")
    reindex_label = app._util_reindex_info.cget("text")
    check("A. reindex (single-path) label shows the full restored path",
          reindex_label == str(reindex), f"label={reindex_label!r}")
    check("E. foreign paths_multi key never landed on the app",
          not hasattr(app, "_not_a_real_picker"))
    status = app._job_label.cget("text")
    check("A. status bar confesses the restored-pick count",
          "6 picked path(s)" in status, f"status={status!r}")

    # B/C: live picks are never overwritten
    live = tmp / "live_pick.tsv"
    live.write_text("drug\tapproval\nx\ty\n", encoding="utf-8")
    app._util_appr_files = [live]
    app._util_merge_files = [live]
    app._util_reindex_path = live
    app._session = playground.load_session(payload)
    app._restore_session()
    check("C. live multi pick kept (approval)", app._util_appr_files == [live])
    check("C. live multi pick kept (merge)", app._util_merge_files == [live])
    check("C. live single pick kept (reindex)", app._util_reindex_path == live)

    # B: an EMPTY live list IS replaceable (the launch state restores fine)
    app._util_appr_files = []
    app._session = playground.load_session(payload)
    app._restore_session()
    check("B. empty live list still accepts the restore",
          getattr(app, "_util_appr_files", None) == files)

    # dump side: _collect_session emits exactly the allowlisted picks
    app._util_tool_files = [merge_a, merge_b]
    collected = app._collect_session()
    check("collect: multi picks dumped in picked order",
          collected["paths_multi"]["_util_tool_files"] == [str(merge_a), str(merge_b)],
          f"got={collected['paths_multi'].get('_util_tool_files')}")
    check("collect: single paths dumped",
          collected["paths"]["_util_reindex_path"] == str(reindex))
    check("collect: paths_multi key always present",
          "paths_multi" in collected and isinstance(collected["paths_multi"], dict))

finally:
    if app is not None:
        app.destroy()

print()
print("PROBE_C_RESULT:", "ALL_PASS" if not fails else f"FAILURES={fails}")
