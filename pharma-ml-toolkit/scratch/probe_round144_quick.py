"""Round-144 quick probe: the paths_multi session machinery + the
progressive reader parity + the loader's determinate bar (throwaway).

Runs on Xvfb :99 with the toolkit venv; uses only temp paths (the
home-path leak gate).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip())
    if not ok:
        RESULTS.append("PROBE144_FAIL")


def main() -> int:
    import playground

    # -- dump/load: the paths_multi key ----------------------------- #
    payload = playground.dump_session(
        theme="light",
        geometry="",
        tab=0,
        variables={},
        families={},
        texts={},
        paths={},
        paths_multi={"_util_merge_files": ["a.csv", "b.csv"]},
    )
    check("dump carries paths_multi", payload.get("paths_multi") ==
          {"_util_merge_files": ["a.csv", "b.csv"]})
    loaded = playground.load_session(payload)
    assert loaded is not None
    check("load round-trips paths_multi", loaded["paths_multi"] ==
          {"_util_merge_files": ["a.csv", "b.csv"]})
    foreign = playground.load_session(
        {
            "format": playground.SESSION_FORMAT,
            "theme": "light",
            "geometry": "",
            "tab": 0,
            "variables": {},
            "families": {},
            "texts": {},
            "paths": {},
            "paths_multi": {"_util_merge_files": "not a list",
                            "_util_appr_files": ["ok.tsv", 42, None],
                            "bad_name": ["x.csv"]},
        }
    )
    assert foreign is not None
    check("load sanitises foreign paths_multi",
          foreign["paths_multi"] == {"_util_appr_files": ["ok.tsv"],
                                     "bad_name": ["x.csv"]},
          json.dumps(foreign["paths_multi"], sort_keys=True))
    old = playground.load_session(
        {
            "format": playground.SESSION_FORMAT,
            "theme": "light",
            "geometry": "",
            "tab": 0,
            "variables": {},
            "families": {},
            "texts": {},
            "paths": {},
        }
    )
    assert old is not None
    check("old payload gains empty paths_multi", old["paths_multi"] == {})

    # -- the reader: parity + reports -------------------------------- #
    from pharmaml.utils.io import read_table, read_table_with_progress

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        semi = tmp_path / "semi.csv"
        semi.write_text("ligand;pic50\nA;1.5\nB;2.5\nC;3.5\n", encoding="utf-8")
        seen: list[tuple[int, object, str]] = []
        frame = read_table_with_progress(
            semi, lambda d, t, n: seen.append((d, t, n)),
            chunk_rows=2, size_threshold=1,
        )
        plain = read_table(semi)
        check("semicolon parity", frame.equals(plain) and list(frame.columns) == ["ligand", "pic50"])
        check("semicolon reports", [s[0] for s in seen] == [2, 3] and seen[0][1] == 4,
              str(seen))

        tab_sep = tmp_path / "tab.tsv"
        tab_sep.write_text("ligand\tpic50\nA\t1.5\nB\t2.5\n", encoding="utf-8")
        seen.clear()
        frame = read_table_with_progress(
            tab_sep, lambda d, t, n: seen.append((d, t, n)), chunk_rows=1, size_threshold=1
        )
        check("tab parity + sniff", frame.equals(read_table(tab_sep)) and
              list(frame.columns) == ["ligand", "pic50"] and [s[0] for s in seen] == [1, 2])

        dup = tmp_path / "dup.csv"
        dup.write_text("a, a \n1,2\n3,4\n5,6\n7,8\n", encoding="utf-8")
        try:
            read_table_with_progress(dup, lambda *a: None, chunk_rows=2, size_threshold=1)
            check("duplicate columns refused", False)
        except Exception as exc:  # noqa: BLE001
            check("duplicate columns refused",
                  "Duplicate column name" in str(exc), type(exc).__name__)

        # -- the GUI round trip: collect -> restore ------------------ #
        app = playground.PlaygroundApp(persist_session=False)
        try:
            a = tmp_path / "a.csv"
            b = tmp_path / "b.csv"
            gone = tmp_path / "gone.csv"
            a.write_text("x\n1\n", encoding="utf-8")
            b.write_text("x\n2\n", encoding="utf-8")
            gone.write_text("x\n3\n", encoding="utf-8")
            app._util_merge_files = [a, gone, b]
            app._util_appr_files = [a, b]
            app._util_tool_files = [gone]
            app._util_merge_mode.set("merge picked tables")
            app._util_tool_mode.set("distribute files to sub-folders")
            collected = app._collect_session()
            check("collect rides the lists",
                  collected["paths_multi"] == {
                      "_util_appr_files": [str(a), str(b)],
                      "_util_merge_files": [str(a), str(gone), str(b)],
                      "_util_tool_files": [str(gone)],
                  }, json.dumps(collected["paths_multi"], sort_keys=True))
            gone.unlink()

            fresh = playground.PlaygroundApp(persist_session=False)
            try:
                fresh._session = playground.load_session(collected)
                assert fresh._session is not None
                fresh._restore_session()
                check("restore filters vanished, keeps order",
                      fresh._util_merge_files == [a, b],
                      str(fresh._util_merge_files))
                check("restore approval list", fresh._util_appr_files == [a, b])
                check("restore drops a fully-vanished list",
                      fresh._util_tool_files == [], str(fresh._util_tool_files))
                label = str(fresh._util_merge_info.cget("text"))
                check("merge label confesses", "2 table(s)" in label and
                      "a.csv, b.csv" in label, label)
                appr_label = str(fresh._util_appr_info.cget("text"))
                check("approval label confesses", appr_label == "a.csv, b.csv",
                      appr_label)
                status = str(fresh._job_label.cget("text"))
                check("status counts restored files", "4 picked path(s) back" in status,
                      status)
                # a live pick is never overwritten
                live = fresh.PlaygroundApp(persist_session=False) if False else None
            finally:
                fresh.destroy()

            live_app = playground.PlaygroundApp(persist_session=False)
            try:
                live_app._util_merge_files = [b]
                live_app._session = playground.load_session(collected)
                assert live_app._session is not None
                live_app._restore_session()
                check("live pick never overwritten", live_app._util_merge_files == [b])
            finally:
                live_app.destroy()

            # -- the loader's determinate bar ------------------------ #
            big = tmp_path / "big.csv"
            rows = ["smiles,value"] + [f"CCO,{i}.5" for i in range(250)]
            big.write_text("\n".join(rows) + "\n", encoding="utf-8")
            app._ml_load_file(big)
            deadline_wait = 0.0
            import time

            while app._busy and deadline_wait < 20.0:
                app.update()
                time.sleep(0.02)
                deadline_wait += 0.02
            check("big load finishes", not app._busy)
            check("big load adopts", app._ml_frame is not None and len(app._ml_frame) == 250)
            check("big load done label", "done in" in str(app._job_label.cget("text")))
        finally:
            app.destroy()

    print("PROBE144_RESULTS")
    for line in RESULTS:
        print(line)
    return 1 if any("PROBE144_FAIL" in line for line in RESULTS) else 0


if __name__ == "__main__":
    sys.exit(main())
