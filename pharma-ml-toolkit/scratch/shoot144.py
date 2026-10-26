"""SHOOT144 -- screenshot the round-144 surfaces (100x-144).

Opens the playground headlessly (Xvfb :99) and captures:
  * pg144-ml-determinate.png -- the Data & ML tab MID-JOB on a genuine
    60 MB / 1.5M-row CSV: the status bar carrying the DETERMINATE bar
    ("loading wide_library.csv — 500,000/1,500,001 · ~N s left"), the
    round's headline: a large CSV streams in with a rows/ETA counter
    instead of the open-ended marquee.
  * pg144-util-merge.png -- the Utilities tab's merge card after a
    paths_multi session restore: the info label confessing the restored
    FIVE picked tables ("5 table(s): ... …").
  * pg144-util-approvals.png -- the Approved drugs card below it: its
    inline label confessing the restored exports list in the picker's
    own "first three + … (+N)" format.

Run:  DISPLAY=:99 python scratch/shoot144.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PLAYGROUND_NO_SESSION", "1")

import playground  # noqa: E402

OUT = ROOT / "scratch"


def _grab(app: "playground.PlaygroundApp", filename: str) -> Path:
    from PIL import ImageGrab

    x, y = app.winfo_rootx(), app.winfo_rooty()
    w, h = app.winfo_width(), app.winfo_height()
    shot = OUT / filename
    ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(shot)
    print(f"wrote {shot}")
    return shot


def _select_tab(app: "playground.PlaygroundApp", name: str) -> None:
    for widget in app._walk_widgets(app):
        if widget.winfo_class() == "TNotebook":
            for tab_id in widget.tabs():
                if name in str(widget.tab(tab_id, "text")):
                    widget.select(tab_id)
                    return
    raise RuntimeError(f"tab not found: {name}")


def main() -> int:
    app = playground.PlaygroundApp(persist_session=False)
    try:
        app.geometry("1280x860+0+0")
        app.update_idletasks()
        app.update()
        OUT.mkdir(exist_ok=True)

        # -- shot 1: the determinate bar mid-read ------------------------ #
        _select_tab(app, "Data & ML")
        with tempfile.TemporaryDirectory(prefix="pg144_lib_") as tmp:
            big = Path(tmp) / "wide_library.csv"
            rows = ["smiles,mw,logp"]
            for i in range(1_500_000):
                rows.append(f"CCO{i % 97},{20.0 + (i % 800) / 10:.1f},{(i % 50) / 10:.1f}")
            big.write_text("\n".join(rows) + "\n", encoding="utf-8")
            size_mb = big.stat().st_size / 1_000_000

            app._ml_load_file(big)
            # poll for the poller's determinate switch, then grab -- the
            # 60 MB read keeps the window busy for well over a second
            mode_mid = ""
            label_mid = ""
            deadline = time.time() + 60.0
            while time.time() < deadline:
                app.update()
                mode_mid = str(app._progress.cget("mode"))
                label_mid = str(app._job_label.cget("text"))
                if mode_mid == "determinate" and "/" in label_mid:
                    break
                if not app._busy:
                    break
                time.sleep(0.02)
            _grab(app, "pg144-ml-determinate.png")
            # drain to completion
            deadline = time.time() + 120.0
            while app._busy and time.time() < deadline:
                app.update()
                time.sleep(0.02)
            label_done = str(app._job_label.cget("text"))
            info = str(app._ml_info.cget("text"))

        print("SHOOT144 facts (determinate loader):")
        print(f"  csv size: {size_mb:.1f} MB")
        print(f"  bar mode mid-job: {mode_mid!r}")
        print(f"  label mid-job: {label_mid!r}")
        print(f"  label done: {label_done!r}")
        print(f"  dataset info: {info!r}")
        ok_loader = (
            mode_mid == "determinate"
            and "loading wide_library.csv" in label_mid
            and "/" in label_mid
            and "left" in label_mid
            and "done in" in label_done
            and "1,500,000 rows" in info
        )

        # -- shots 2 + 3: the multi-file picks after a session restore --- #
        with tempfile.TemporaryDirectory(prefix="pg144_util_") as tmp:
            merge_tables = []
            for i in range(5):
                table = Path(tmp) / f"table_{chr(97 + i)}.csv"
                table.write_text("x\n1\n", encoding="utf-8")
                merge_tables.append(table)
            exports = []
            for i in range(5):
                export = Path(tmp) / f"approvals_{chr(97 + i)}.tsv"
                export.write_text("drug\tstatus\nx\tapproved\n", encoding="utf-8")
                exports.append(export)
            app._util_merge_mode.set("merge picked tables")
            app._util_tool_mode.set("distribute files into folders")
            app._session = playground.load_session(
                {
                    "format": 1,
                    "theme": "light",
                    "geometry": "",
                    "tab": 0,
                    "variables": {},
                    "families": {},
                    "texts": {},
                    "paths": {},
                    "paths_multi": {
                        "_util_merge_files": [str(p) for p in merge_tables],
                        "_util_appr_files": [str(p) for p in exports],
                    },
                }
            )
            app._restore_session()
            status = str(app._job_label.cget("text"))
            merge_info = str(app._util_merge_info.cget("text"))
            appr_info = str(app._util_appr_info.cget("text"))
            restored_merge = list(app._util_merge_files)
            restored_appr = list(app._util_appr_files)

            _select_tab(app, "Utilities")
            app._jump_to_card(8, "Table merge & split")
            app.update_idletasks()
            app.update()
            _grab(app, "pg144-util-merge.png")
            app._jump_to_card(8, "Approved drugs → report")
            app.update_idletasks()
            app.update()
            _grab(app, "pg144-util-approvals.png")

        print("SHOOT144 facts (multi-file session):")
        print(f"  status bar: {status!r}")
        print(f"  merge label: {merge_info!r}")
        print(f"  approval label: {appr_info!r}")
        print(f"  restored merge files: {len(restored_merge)}")
        print(f"  restored approval files: {len(restored_appr)}")
        ok_multi = (
            "10 picked path(s) back" in status
            and restored_merge == merge_tables
            and restored_appr == exports
            and merge_info.startswith("5 table(s): table_a.csv, table_b.csv")
            and appr_info == "approvals_a.tsv, approvals_b.tsv, approvals_c.tsv … (+2)"
        )

        print("SHOOT144_OK" if (ok_loader and ok_multi) else "SHOOT144_FAIL")
        return 0 if (ok_loader and ok_multi) else 1
    finally:
        app.destroy()


if __name__ == "__main__":
    sys.exit(main())
