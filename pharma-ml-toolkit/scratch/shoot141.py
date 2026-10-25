"""SHOOT141 -- screenshot the two new Docking cards (100x-141).

Opens the playground headlessly (Xvfb :99), jumps to the Docking tab,
scrolls the two new cards (Ligand structures / MGLTools batch) into
view and captures pg141-new-cards.png; then drives the resume checkbox
state and captures the vbatch card too (pg141-resume.png).

Run:  DISPLAY=:99 python scratch/shoot141.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PLAYGROUND_NO_SESSION", "1")

import playground  # noqa: E402

OUT = ROOT / "scratch"


def main() -> int:
    app = playground.PlaygroundApp(persist_session=False)
    try:
        # land on the Docking tab
        tabs = None
        for widget in app._walk_widgets(app):
            if widget.winfo_class() == "TNotebook":
                tabs = widget
                break
        if tabs is None:
            print("SHOOT141_FAIL: no notebook found")
            return 1
        for tab_id in tabs.tabs():
            if "Docking" in str(tabs.tab(tab_id, "text")):
                tabs.select(tab_id)
                break
        app.update_idletasks()
        app.update()

        # the window's grab rectangle (after the first layout pass)
        x, y = app.winfo_rootx(), app.winfo_rooty()
        w, h = app.winfo_width(), app.winfo_height()
        OUT.mkdir(exist_ok=True)

        # jump to the MGL card -- the gen card sits right above it, so
        # ONE screenshot shows both new cards; then the vbatch card.
        live = {
            title: (shell, index)
            for title, _subtitle, shell, index in app._live_cards()
        }
        for needed in (
            "Ligand structures (CSV → PDB + MOL2)",
            "PDB → PDBQT batch (MGLTools)",
            "Vina config batch",
        ):
            if needed not in live:
                print(f"SHOOT141_FAIL: card not live: {needed}")
                return 1
        targets = (
            ("Ligand structures (CSV → PDB + MOL2)", "pg141-new-cards.png"),
            ("Vina config batch", "pg141-resume.png"),
        )
        from PIL import ImageGrab

        for title, filename in targets:
            shell, index = live[title]
            app._jump_to_card(index, title)
            app.update_idletasks()
            app.update()
            shot = OUT / filename
            ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(shot)
            print(f"wrote {shot}")

        # live facts for the VLM-free check
        print("SHOOT141 facts:")
        print("  gen conformers default:", app._dock_gen_confs.get())
        print("  gen mol2 default:", bool(app._dock_gen_mol2_var.get()))
        print("  mgl timeout default:", app._dock_mgl_timeout.get())
        print("  vbatch resume default:", bool(app._dock_vbatch_skip_var.get()))
        print("SHOOT141_OK")
        return 0
    finally:
        app.destroy()


if __name__ == "__main__":
    sys.exit(main())
