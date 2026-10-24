"""Round 100x-131 smoke: the finders' "Clear" affordance (the desktop
side -- the button rides the recency row, one click empties the
session-persisted memory and collapses the row in place, the next
open renders no row at all, and no recency means no button) plus the
report bridge's new ``image`` section kind (a REAL surgery grid from
the surgery bridge composes into the HTML report as a full-bleed
embedded figure with its caption; broken payloads exit 2) --
headless, Xvfb-driven.

Run: DISPLAY=:99 /home/z/.venv/bin/python scratch/smoke131.py
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time

sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit")

import playground

SURGERY_BRIDGE = "/home/z/my-project/scripts/surgery_bridge.py"
REPORT_BRIDGE = "/home/z/my-project/scripts/report_bridge.py"


def deadline_wait(app, label: str, timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    print(f"TIMEOUT waiting for {label}")
    return False


def _run_bridge(script: str, payload: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )


def main() -> int:
    app = playground.PlaygroundApp(persist_session=False)
    try:
        import tkinter as tkmod

        # -- the Clear button rides the recency row ---------------------- #
        app._jump_to_card(1, "Affinity conversion")
        app._jump_to_card(0, "PharmaML Playground")
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
        assert "Clear" in labels, labels
        print("clear affordance: rides the recency row OK")

        # -- one click empties + collapses in place ----------------------- #
        chips = {str(b.cget("text")): b for b in buttons}
        chips["Clear"].invoke()
        app.update()
        assert app._finder_recent == []
        row_labels: list = []

        def walk_labels(widget):
            for child in widget.winfo_children():
                if child.winfo_class() == "TLabel":
                    row_labels.append(str(child.cget("text")))
                walk_labels(child)

        walk_labels(win)
        assert not any(l.startswith("Recently jumped") for l in row_labels)
        payload_session = app._collect_session()
        assert json.loads(payload_session["texts"]["finder_recent"]) == []
        print("clear click: memory emptied + row collapsed + session [] OK")

        # -- the next open renders no row at all --------------------------- #
        win.destroy()
        app._find_card_dialog()
        dialogs = [
            child
            for child in app.winfo_children()
            if isinstance(child, tkmod.Toplevel) and child.title() == "Find a card"
        ]
        win2 = dialogs[0]
        buttons2: list = []

        def walk_buttons2(widget):
            for child in widget.winfo_children():
                if child.winfo_class() == "TButton":
                    buttons2.append(child)
                walk_buttons2(child)

        walk_buttons2(win2)
        assert "Clear" not in [str(b.cget("text")) for b in buttons2]
        win2.destroy()
        print("next open: no row, no Clear button OK")

        # -- the documentation pins ----------------------------------------- #
        doc = playground.__doc__ or ""
        assert "100x-131" in doc and '"Clear" affordance' in doc
        dialog_doc = playground.PlaygroundApp._find_card_dialog.__doc__ or ""
        assert '"Clear" ghost button' in dialog_doc
        print("docstring pins OK")

        # -- the report bridge's image kind over a REAL surgery grid ------- #
        surgery = _run_bridge(
            SURGERY_BRIDGE, {"mode": "topology", "smiles": "C1CCNCC1"}
        )
        assert surgery.returncode == 0, surgery.stdout + surgery.stderr
        grid = json.loads(surgery.stdout)["gridPng"]
        assert grid.startswith("data:image/png;base64,")
        assert base64.b64decode(grid.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"

        report = _run_bridge(
            REPORT_BRIDGE,
            {
                "theme": "light",
                "subtitle": "smoke131",
                "sections": [
                    {
                        "kind": "image",
                        "title": "Ring topology mutants — 3 structure panel(s)",
                        "dataUri": grid,
                    },
                ],
            },
        )
        assert report.returncode == 0, report.stdout + report.stderr
        composed = json.loads(report.stdout)
        assert composed["figures"] == 1
        html = composed["html"]
        assert html.count("<img") == 1
        assert "Ring topology mutants — 3 structure panel(s)" in html
        assert "<link" not in html and "src='http" not in html
        print("report bridge image kind: real grid embedded + caption OK")

        # -- the broken payloads exit 2 with the contract's own text ------- #
        for data_uri, fragment in (
            ("data:image/png;base64,not-valid-!!", "not valid base64"),
            ("data:image/jpeg;base64,aGVsbG8=", "dataUri"),
            ("data:image/png;base64," + "A" * 64, "PNG payload"),
        ):
            bad = _run_bridge(
                REPORT_BRIDGE,
                {
                    "theme": "light",
                    "sections": [
                        {"kind": "image", "title": "broken", "dataUri": data_uri}
                    ],
                },
            )
            assert bad.returncode == 2, data_uri[:40]
            assert fragment in json.loads(bad.stdout)["error"]
        print("report bridge image kind: broken payloads exit 2 OK")

        print("SMOKE131_OK")
        return 0
    finally:
        app.destroy()
        deadline_wait(app, "final destroy", timeout=30.0)


if __name__ == "__main__":
    sys.exit(main())
