"""Screenshots + artifacts for the 100x-131 round: the finders'
"Clear" affordance + the web session report's per-section control
(the web side's screenshots come from the agent-browser E2E walk --
shoot131-web).  Desktop shots: (1) the finder dialog with the
"Recently jumped" chip row AND its Clear button visible; (2) the
moment after a Clear click (the row collapsed, the memory gone);
(3) the desktop Generative tab with a topology grid drawn (the
surgery content the report's image section now embeds); (4) a REAL
session report HTML artifact with the surgery image section inside
(the round's bridge output, saved into download/ for inspection).

Run: DISPLAY=:99 python scratch/shoot131.py
"""
import json
import subprocess
import sys
import time

sys.path.insert(0, ".")
import playground

# keep any modal quiet in headless runs
playground.messagebox.showinfo = lambda *a, **k: None

SURGERY_BRIDGE = str(Path(__file__).resolve().parents[2] / "scripts" / "surgery_bridge.py")
REPORT_BRIDGE = str(Path(__file__).resolve().parents[2] / "scripts" / "report_bridge.py")

app = playground.PlaygroundApp()
app.geometry("1280x860+10+10")


def pump(timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def shot(name: str) -> None:
    app.update()
    app.update_idletasks()
    for _ in range(24):
        app.update()
        time.sleep(0.01)
    from PIL import ImageGrab  # type: ignore

    img = ImageGrab.grab(xdisplay=":99")
    img.save(f"download/{name}.png")
    print("shot", name)


try:
    pump()
    # seed the recency memory with real jumps (newest first)
    app._jump_to_card(1, "Affinity conversion")
    app._jump_to_card(0, "PharmaML Playground")
    live_cards = app._live_cards()
    for title in ("Ring topology moves", "Breed two molecules"):
        for card_title, _subtitle, _shell, index in live_cards:
            if card_title == title:
                app._jump_to_card(index, title)
                break
    pump()

    # (1) the dialog with the chip row + the Clear button
    app._find_card_dialog()
    import tkinter as tkmod

    dialogs = [
        child
        for child in app.winfo_children()
        if isinstance(child, tkmod.Toplevel) and child.title() == "Find a card"
    ]
    win = dialogs[0]
    shot("pg131-finder-clear-visible")

    # (2) the moment after a Clear click: row collapsed in place
    buttons: list = []

    def walk_buttons(widget):
        for child in widget.winfo_children():
            if child.winfo_class() == "TButton":
                buttons.append(child)
            walk_buttons(child)

    walk_buttons(win)
    clear = {str(b.cget("text")): b for b in buttons}["Clear"]
    clear.invoke()
    app.update()
    pump(timeout=10.0)
    assert app._finder_recent == []
    shot("pg131-finder-cleared")
    win.destroy()

    # (3) the Generative tab with a topology grid drawn (the surgery
    # content the report's image section embeds this round)
    live_cards = app._live_cards()
    for card_title, _subtitle, _shell, index in live_cards:
        if card_title == "Ring topology moves":
            app._jump_to_card(index, card_title)
            break
    app._topo_smiles.delete(0, "end")
    app._topo_smiles.insert(0, "C1CCNCC1")
    app._gen_topology_moves()
    assert pump(), "topology job hung"
    app._gen_draw_topology()
    assert pump(), "mutant grid job hung"
    shot("pg131-desktop-topology-card")

    # (4) a REAL session-report HTML artifact with the surgery image
    # section inside -- the bridge output, saved for inspection
    surgery = subprocess.run(
        [sys.executable, SURGERY_BRIDGE],
        input=json.dumps({"mode": "topology", "smiles": "C1CCNCC1"}),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert surgery.returncode == 0, surgery.stdout + surgery.stderr
    grid = json.loads(surgery.stdout)["gridPng"]
    report = subprocess.run(
        [sys.executable, REPORT_BRIDGE],
        input=json.dumps(
            {
                "theme": "dark",
                "subtitle": "100x-131 — the surgery feed",
                "sections": [
                    {
                        "kind": "metrics",
                        "title": "Ring topology moves",
                        "items": [
                            ["source", "C1CCNCC1"],
                            ["distinct mutants", "3"],
                        ],
                    },
                    {
                        "kind": "table",
                        "title": "Ring topology mutants",
                        "columns": ["#", "mutant SMILES"],
                        "rows": [
                            [1, "C1CCCC1"],
                            [2, "C1CCCNCC1"],
                            [3, "C1CCNC1"],
                        ],
                    },
                    {
                        "kind": "image",
                        "title": "Ring topology mutants — 3 structure panel(s)",
                        "dataUri": grid,
                    },
                ],
            }
        ),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert report.returncode == 0, report.stdout + report.stderr
    composed = json.loads(report.stdout)
    assert composed["figures"] == 1 and composed["tables"] == 1
    with open(
        "download/pg131-surgery-report.html",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(composed["html"])
    print("artifact pg131-surgery-report.html written")

    print("SHOOT131_OK")
finally:
    app.destroy()
    pump(timeout=30.0)
