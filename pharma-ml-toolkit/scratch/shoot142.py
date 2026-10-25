"""SHOOT142 -- screenshot the round-142 surfaces (100x-142).

Opens the playground headlessly (Xvfb :99), jumps to the Docking tab
and captures:
  * pg142-gen-bridge.png -- the ligand-structures card with its new
    "Send PDBs → MGLTools" relay button (visible beside Generate).
  * pg142-vbatch-count.png -- the Vina config batch card with the new
    "Count remaining" button and the resume checkbox.
  * pg142-bridge-landing.png -- AFTER clicking the relay: the MGLTools
    card's info label refreshed with the handed-over source folder.

Run:  DISPLAY=:99 python scratch/shoot142.py
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


def main() -> int:
    app = playground.PlaygroundApp(persist_session=False)
    try:
        for widget in app._walk_widgets(app):
            if widget.winfo_class() == "TNotebook":
                tabs = widget
                break
        else:
            print("SHOOT142_FAIL: no notebook found")
            return 1
        for tab_id in tabs.tabs():
            if "Docking" in str(tabs.tab(tab_id, "text")):
                tabs.select(tab_id)
                break
        app.update_idletasks()
        app.update()

        x, y = app.winfo_rootx(), app.winfo_rooty()
        w, h = app.winfo_width(), app.winfo_height()
        OUT.mkdir(exist_ok=True)

        live = {
            title: (shell, index)
            for title, _subtitle, shell, index in app._live_cards()
        }
        for needed in (
            "Ligand structures (CSV → PDB + MOL2)",
            "Vina config batch",
            "PDB → PDBQT batch (MGLTools)",
        ):
            if needed not in live:
                print(f"SHOOT142_FAIL: card not live: {needed}")
                return 1

        # a real picked folder so the relay has something to hand over
        with tempfile.TemporaryDirectory(prefix="pg142_gen_") as gen_dir:
            app._dock_gen_out = Path(gen_dir)
            from PIL import ImageGrab

            shots = (
                ("Ligand structures (CSV → PDB + MOL2)", "pg142-gen-bridge.png"),
                ("Vina config batch", "pg142-vbatch-count.png"),
            )
            for title, filename in shots:
                shell, index = live[title]
                app._jump_to_card(index, title)
                app.update_idletasks()
                app.update()
                shot = OUT / filename
                ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(shot)
                print(f"wrote {shot}")

            # click the relay: the MGL card must refresh + jump there
            app._dock_gen_to_mgl()
            app.update_idletasks()
            app.update()
            shot = OUT / "pg142-bridge-landing.png"
            ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(shot)
            print(f"wrote {shot}")
            print("SHOOT142 facts:")
            print("  mgl src after relay:", app._dock_mgl_src)
            print("  mgl out after relay:", app._dock_mgl_out)
            print("  gen summary:", app._dock_gen_summary.cget("text"))
            print("  mgl info:", repr(app._dock_mgl_info.cget("text")))

        # the census button label + resume checkbox state (facts line)
        print("  vbatch resume default:", bool(app._dock_vbatch_skip_var.get()))
        button_texts = [
            str(b.cget("text")) for b in app._action_buttons if b.winfo_exists()
        ]
        print("  Count remaining present:", "Count remaining" in button_texts)
        print("  relay button present:", "Send PDBs → MGLTools" in button_texts)
        print("SHOOT142_OK")
        return 0
    finally:
        app.destroy()


if __name__ == "__main__":
    sys.exit(main())
