"""Round 100x-130 smoke: the Ctrl+F card finder's "Recently jumped"
recency row -- every jump is remembered (most-recent-first,
de-duplicated, capped at 6, persisted with the session as JSON), the
dialog renders one-click chips for the LIVE remembered cards only, a
chip click jumps + closes + re-remembers, and the web twin's surgery
bridge answers the pinned contracts (the piperidine mutant set, the
seeded litter, the honest-null rows, the PNG grid) -- headless,
Xvfb-driven.

Run: DISPLAY=:99 /home/z/.venv/bin/python scratch/smoke130.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit")

import playground

BRIDGE = "/home/z/my-project/scripts/surgery_bridge.py"


def deadline_wait(app, label: str, timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    print(f"TIMEOUT waiting for {label}")
    return False


def main() -> int:
    app = playground.PlaygroundApp(persist_session=False)
    try:
        # -- the memory -------------------------------------------------- #
        app._jump_to_card(1, "Affinity conversion")
        app._jump_to_card(0, "PharmaML Playground")
        app._jump_to_card(1, "Affinity conversion")
        assert app._finder_recent == ["Affinity conversion", "PharmaML Playground"]
        print("memory: most-recent-first + de-duplication OK")

        titles = [t for t, _s, _sh in playground._CARD_INDEX[:8]]
        for index, title in enumerate(titles):
            app._jump_to_card(0, title)
        assert app._finder_recent == list(reversed(titles[-6:]))
        assert len(app._finder_recent) == playground._FINDER_RECENT_MAX == 6
        print("memory: capped at _FINDER_RECENT_MAX = 6 OK")

        # -- the dialog's chip row ---------------------------------------- #
        import tkinter as tkmod

        app._finder_recent = ["Affinity conversion", "Molar mass", "no such card"]
        app._find_card_dialog()
        dialogs = [
            child
            for child in app.winfo_children()
            if isinstance(child, tkmod.Toplevel) and child.title() == "Find a card"
        ]
        assert len(dialogs) == 1
        win = dialogs[0]
        buttons: list = []

        def walk_buttons(widget):
            for child in widget.winfo_children():
                if child.winfo_class() == "TButton":
                    buttons.append(child)
                walk_buttons(child)

        walk_buttons(win)
        labels = [str(b.cget("text")) for b in buttons]
        assert "Affinity conversion" in labels and "Molar mass" in labels, labels
        assert not any("no such card" in label for label in labels)
        print("chip row: live titles only OK")

        chips = {str(b.cget("text")): b for b in buttons}
        chips["Molar mass"].invoke()
        app.update()
        assert not bool(win.winfo_exists())
        assert app._finder_recent[0] == "Molar mass"
        print("chip click: jump + close + re-remember OK")

        # -- the session round trip ---------------------------------------- #
        payload = app._collect_session()
        raw = payload["texts"]["finder_recent"]
        assert json.loads(raw)[0] == "Molar mass"
        app._finder_recent = ["SMILES inspector"]
        app._session = {
            "theme": "light",
            "geometry": "",
            "tab": 0,
            "variables": {},
            "families": {},
            "texts": {"finder_recent": raw},
        }
        app._restore_session()
        assert app._finder_recent[0] == "Molar mass"
        print("session persistence: JSON round trip OK")

        # -- the documentation pins ----------------------------------------- #
        doc = playground.__doc__ or ""
        assert "100x-130" in doc and "Recently jumped" in doc
        print("docstring pins OK")

        # -- the web twin's surgery bridge (the pinned contracts) ----------- #
        result = subprocess.run(
            [sys.executable, BRIDGE],
            input=json.dumps({"mode": "topology", "smiles": "C1CCNCC1"}),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        response = json.loads(result.stdout)
        assert response["mutants"] == ["C1CCCC1", "C1CCCNCC1", "C1CCNC1"]
        assert response["gridPng"].startswith("data:image/png;base64,")
        print("bridge topology: the piperidine set + PNG grid OK")

        result = subprocess.run(
            [sys.executable, BRIDGE],
            input=json.dumps(
                {
                    "mode": "breed",
                    "parentA": "CC(=O)Oc1ccccc1C(=O)O",
                    "parentB": "CC(C)Cc1ccc(O)cc1",
                    "children": 4,
                    "seed": 42,
                }
            ),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        response = json.loads(result.stdout)
        assert len(response["litter"]) == 4
        assert response["viable"] == sum(1 for child in response["litter"] if child)
        assert response["gridPanels"] == 2 + response["viable"]
        print("bridge breed: seeded litter + honest nulls + parents grid OK")

        print("SMOKE130_OK")
        return 0
    finally:
        app.destroy()
        deadline_wait(app, "final destroy", timeout=30.0)


if __name__ == "__main__":
    sys.exit(main())
