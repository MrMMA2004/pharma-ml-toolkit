"""SHOOT145 -- screenshot the round-145 surfaces (100x-145).

Opens the playground headlessly (Xvfb :99) and captures:
  * pg145-ml-prep.png -- the Data & ML Dataset card after an
    EC50 → pEC50 prep over a D2-style CSV (censored '<50', a duplicate
    SMILES, junk text, a blank SMILES, an out-of-range 1e10): the info
    line confessing the whole audit and the preview showing the
    ADOPTED prepared table (pEC50 pre-picked as the target).
  * pg145-ml-permutation.png -- the Results card after Train & evaluate
    (ridge, the regression example) plus the round's Permutation
    importance button: the signed bars with ±1 std whiskers, the noise
    features red.
  * pg145-chem-ecfp3.png -- the Chemistry tab's feature factory with
    the new ECFP3 checkbox ticked and the width label confessing
    "9 families selected · ~12,853 columns".

Run:  DISPLAY=:99 python scratch/shoot145.py
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


def _drain(app: "playground.PlaygroundApp", timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def main() -> int:
    app = playground.PlaygroundApp(persist_session=False)
    try:
        app.geometry("1280x900+0+0")
        app.update_idletasks()
        app.update()
        OUT.mkdir(exist_ok=True)

        # -- shot 1: the EC50 -> pEC50 prep card ------------------------- #
        _select_tab(app, "Data & ML")
        with tempfile.TemporaryDirectory(prefix="pg145_d2_") as tmp:
            d2_csv = Path(tmp) / "d2_ligands.csv"
            d2_csv.write_text(
                "Ligand SMILES,EC50 (nM)\n"
                "CC(=O)Oc1ccccc1C(=O)O,<50\n"
                "CC(=O)Oc1ccccc1C(=O)O,200.0\n"
                "CCN(C)CCc1ccccc1O,125.0\n"
                "CN1C=NC2=C1C(=O)N(C)C(=O)N2C,51.0\n"
                "CCO,junk\n"
                ",10.0\n"
                "CCOC(=O)c1ccccc1N,10000000000.0\n"
                "Cc1ccc(O)cc1,310.0\n",
                encoding="utf-8",
            )
            app._ml_load_file(d2_csv)
            assert _drain(app), "d2 load job hung"
            # auto-adopt the prepared set for the shot
            playground.messagebox.askyesno = lambda *a, **k: True  # type: ignore[assignment]
            app._ml_prep_endpoint()
            assert _drain(app), "prep job hung"
            info = str(app._ml_info.cget("text"))
            target = str(app._target_combo.get())
            preview_cols = list(app._preview_tree["columns"])
            n_rows = len(app._preview_tree.get_children())
            app._jump_to_card(5, "Dataset")
            app.update_idletasks()
            app.update()
            _grab(app, "pg145-ml-prep.png")

        print("SHOOT145 facts (pEC50 prep):")
        print(f"  info line: {info!r}")
        print(f"  target: {target!r}")
        print(f"  preview columns: {preview_cols}")
        print(f"  preview rows: {n_rows}")
        ok_prep = (
            "pEC50 prep" in info
            and "8 rows in" in info
            and "censored" in info
            and "duplicates collapsed" in info
            and target == "pEC50"
            and preview_cols == ["SMILES", "n_measurements", "endpoint_median", "pEC50"]
            and n_rows > 0
        )

        # -- shot 2: the permutation importance figure ------------------- #
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert _drain(app, timeout=240.0), "train job hung"
        app._ml_plot_permutation()
        assert _drain(app, timeout=240.0), "permutation job hung"
        note = str(app._ml_result_note.cget("text"))
        figure_children = len(app._ml_host.winfo_children())
        app._jump_to_card(5, "Results")
        app.update_idletasks()
        app.update()
        _grab(app, "pg145-ml-permutation.png")

        print("SHOOT145 facts (permutation):")
        print(f"  note: {note!r}")
        print(f"  figure children: {figure_children}")
        ok_perm = "permutation importance" in note and figure_children == 1

        # -- shot 3: the ecfp3 family checkbox --------------------------- #
        _select_tab(app, "Chemistry")
        app._jump_to_card(1, "Feature factory")
        app.update_idletasks()
        app.update()
        width_label = str(app._feat_width_label.cget("text"))
        ecfp3_ticked = bool(app._feat_family_vars["ecfp3"].get())
        _grab(app, "pg145-chem-ecfp3.png")

        print("SHOOT145 facts (ecfp3):")
        print(f"  width label: {width_label!r}")
        print(f"  ecfp3 ticked: {ecfp3_ticked}")
        ok_ecfp3 = "9 families selected" in width_label and ecfp3_ticked

        verdict = ok_prep and ok_perm and ok_ecfp3
        print(f"SHOOT145_OK: {verdict}")
        return 0 if verdict else 1
    finally:
        app.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
