"""SHOOT143 -- screenshot the round-143 surfaces (100x-143).

Opens the playground headlessly (Xvfb :99) and captures:
  * pg143-ml-load-job.png -- the Data & ML tab MID-JOB: a 300k-row CSV
    loading through the new worker-pool loader (the status bar carrying
    "loading wide_library.csv", the busy marquee, the action buttons
    disabled) -- the round's headline: a big CSV never freezes the
    window again.
  * pg143-ml-load-done.png -- the SAME tab after the job: the status
    bar's finished-job label ("loading ... — done in ..."), the dataset
    card's preview tree filled, the info label confessing the row count.
  * pg143-util-labels.png -- the Utilities tab after a session-style
    restore of the seven picked paths: the merge/compare/tools/reindex
    cards' info labels finally CONFESSING the restored paths (round 27
    restored them silently; the labels used to keep denying them).
  * pg143-maccs-mixed.png -- the Chemistry tab's MACCS card after a
    mixed parse run: the computed summary kept, the dropped count
    riding alongside (the false "no MACCS keys" clobber is gone).

Run:  DISPLAY=:99 python scratch/shoot143.py
"""

from __future__ import annotations

import os
import sys
import tempfile
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

        # -- shot 1 + 2: the job-based Data & ML loader ------------------ #
        _select_tab(app, "Data & ML")
        with tempfile.TemporaryDirectory(prefix="pg143_lib_") as tmp:
            big = Path(tmp) / "wide_library.csv"
            rows = ["smiles,mw,logp"]
            for i in range(300_000):
                rows.append(f"CCO{i % 97},{20.0 + (i % 800) / 10:.1f},{(i % 50) / 10:.1f}")
            big.write_text("\n".join(rows) + "\n", encoding="utf-8")
            size_mb = big.stat().st_size / 1_000_000

            app._ml_load_file(big)
            app.update_idletasks()
            app.update()
            # mid-job: the reader thread is still chewing the 13 MB table
            busy_mid = app._busy
            label_mid = str(app._job_label.cget("text"))
            if busy_mid:
                _grab(app, "pg143-ml-load-job.png")
            # drain to completion
            import time

            deadline = time.time() + 60.0
            while app._busy and time.time() < deadline:
                app.update()
                time.sleep(0.02)
            label_done = str(app._job_label.cget("text"))
            info = str(app._ml_info.cget("text"))
            preview_rows = len(app._preview_tree.get_children())
            _grab(app, "pg143-ml-load-done.png")

        print("SHOOT143 facts (loader):")
        print(f"  csv size: {size_mb:.1f} MB")
        print(f"  busy mid-job: {busy_mid}")
        print(f"  label mid-job: {label_mid!r}")
        print(f"  label done: {label_done!r}")
        print(f"  dataset info: {info!r}")
        print(f"  preview rows: {preview_rows}")
        ok_loader = (
            busy_mid
            and label_mid.startswith("loading wide_library.csv")
            and "done in" in label_done
            and "300,000 rows" in info
            and preview_rows == 12
        )

        # -- shot 3: the Utilities labels after a session-style restore -- #
        with tempfile.TemporaryDirectory(prefix="pg143_util_") as tmp:
            campaign = Path(tmp) / "campaign"
            campaign.mkdir()
            table_a = Path(tmp) / "screen_a.csv"
            table_a.write_text("x\n1\n", encoding="utf-8")
            table_b = Path(tmp) / "screen_b.csv"
            table_b.write_text("x\n2\n", encoding="utf-8")
            split_file = Path(tmp) / "big_table.csv"
            split_file.write_text("x\n3\n", encoding="utf-8")
            out_dir = Path(tmp) / "collected"
            out_dir.mkdir()
            reindex = Path(tmp) / "sparse_results.csv"
            reindex.write_text("n\n1\n", encoding="utf-8")
            # exactly the restore's own assignments
            app._util_merge_folder = campaign
            app._util_merge_split_file = split_file
            app._util_cmp_a = table_a
            app._util_cmp_b = table_b
            app._util_tool_folder = campaign
            app._util_tool_out = out_dir
            app._util_reindex_path = reindex
            app._util_merge_mode.set("merge whole folder")
            app._util_tool_mode.set("collect files (name/pattern)")
            app._refresh_restored_path_labels()

            _select_tab(app, "Utilities")
            # scroll the merge card to the top of the viewport so the
            # labels are in-frame (the tab is long)
            app._jump_to_card(8, "Table merge & split")
            app.update_idletasks()
            app.update()
            _grab(app, "pg143-util-labels.png")
            merge_info = str(app._util_merge_info.cget("text"))
            cmp_info = str(app._util_cmp_info.cget("text"))
            tool_info = str(app._util_tool_info.cget("text"))
            reindex_info = str(app._util_reindex_info.cget("text"))

        print("SHOOT143 facts (utilities):")
        print(f"  merge label: {merge_info!r}")
        print(f"  compare label: {cmp_info!r}")
        print(f"  tools label: {tool_info!r}")
        print(f"  reindex label: {reindex_info!r}")
        ok_utils = (
            f"folder: {campaign}" in merge_info
            and f"A: {table_a.name}" in cmp_info
            and "campaign" in tool_info
            and str(reindex) == reindex_info
        )

        # -- shot 4: the MACCS mixed-parse label ------------------------- #
        _select_tab(app, "Chemistry")
        app._maccs_text.delete("1.0", "end")
        app._maccs_text.insert("1.0", "CCO\nCC(=O)Oc1ccccc1C(=O)O\nNOT_A_SMILES")
        app._maccs_run()
        deadline = time.time() + 60.0
        while app._busy and time.time() < deadline:
            app.update()
            time.sleep(0.02)
        # jump + scroll the card into view before the grab (the Chemistry
        # tab is long; the card sits below the fold)
        app._jump_to_card(1, "MACCS key fingerprint")
        app.update_idletasks()
        app.update()
        _grab(app, "pg143-maccs-mixed.png")
        maccs_label = str(app._maccs_result.cget("text"))
        print("SHOOT143 facts (maccs):")
        print(f"  maccs label: {maccs_label[:180]!r}")
        ok_maccs = (
            "2 molecule(s) · mean" in maccs_label
            and "excluded from the mean" in maccs_label
        )

        print("SHOOT143_OK" if (ok_loader and ok_utils and ok_maccs) else "SHOOT143_FAIL")
        return 0 if (ok_loader and ok_utils and ok_maccs) else 1
    finally:
        app.destroy()


if __name__ == "__main__":
    sys.exit(main())
