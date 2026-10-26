"""SHOOT148 -- screenshot the round-148 surfaces (100x-148).

Opens the playground headlessly (Xvfb :99, same command) and captures:
  * pg148-ml-pr.png -- the Data & ML Results card after a multiclass
    Train & evaluate + the round's PR-curve button: the figure with
    the per-class AP legend + no-skill baselines, the per-class
    metrics line under the macro aggregates, and the ROC/PR buttons.
  * pg148-families.png -- the Chemistry tab's feature factory with
    the twenty-family picker and the ~27,759-column width label.
  * pg148-ml-triage.png -- the Predict card after the pIC50 -> IC50
    conversion + the 100 nM potency triage (the info line's honest
    "N of M row(s)" census).
  * pg148-ml-benchmark.png -- the Leaderboard after the family
    benchmark: 18 rows, one per 2D family, best first.

Run:  DISPLAY=:99 python scratch/shoot148.py  (Xvfb starts in-command)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PLAYGROUND_NO_SESSION", "1")

import playground  # noqa: E402

OUT = ROOT / "download"


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


def _drain(app: "playground.PlaygroundApp", timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def _scroll_to_widget(
    app: "playground.PlaygroundApp", widget: object, offset: int = 24
) -> None:
    """Scroll a deep widget (a card / figure host) into the viewport."""
    page = app._book.nametowidget(app._book.select())
    app.update()
    body_y = widget.winfo_rooty() - page.body.winfo_rooty()
    total = page._canvas.bbox("all")[3] or 1
    page._canvas.yview_moveto(max(0, body_y - offset) / total)
    app.update()


def _quiet() -> None:
    for name in ("showinfo", "showerror", "showwarning"):
        setattr(
            playground.messagebox,
            name,
            lambda *a, _n=name, **k: print(f"  [dialog {_n}]: {a}"),
        )


def main() -> int:
    _quiet()
    app = playground.PlaygroundApp(persist_session=False)
    ok_pr = ok_families = ok_triage = ok_bench = False
    facts: list[str] = []
    try:
        app.geometry("1280x900+0+0")
        app.update_idletasks()
        app.update()
        OUT.mkdir(exist_ok=True)

        # -- shot 1: the PR curve + per-class line ---------------------- #
        import pandas as pd

        _select_tab(app, "Data & ML")
        frame = pd.read_csv(ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._ml_adopt_frame(frame, ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._model_combo.set("logistic_regression")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert _drain(app), "train job hung"
        app._ml_plot_pr()
        assert _drain(app), "PR job hung"
        note = str(app._ml_result_note.cget("text"))
        _scroll_to_widget(app, app._ml_metrics, offset=100)
        _grab(app, "pg148-ml-pr.png")
        ok_pr = "PR curves replace the parity figure" in note and getattr(
            app._ml_host, "_pg_figure", None
        ) is not None
        facts.append(f"PR note: {note!r}")

        # -- shot 2: the twenty-family picker --------------------------- #
        _select_tab(app, "Chemistry")
        width_text = str(app._feat_width_label.cget("text"))
        _scroll_to_widget(app, app._feat_pasted_text, offset=0)
        _grab(app, "pg148-families.png")
        ok_families = len(app._feat_family_vars) == 20 and "20 families selected" in width_text
        facts.append(f"width label: {width_text!r}")

        # -- shot 3: the IC50 triage ------------------------------------- #
        _select_tab(app, "Data & ML")
        app._ml_predictions = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN", "CCCO", "CCCCO", "CC(=O)O", "CCC(=O)O"],
                "prediction": [9.0, 8.0, 7.0, 6.0, 5.5, 5.0],
            }
        )
        app._ml_predict_add_ic50()
        assert _drain(app), "IC50 job hung"
        app._predict_threshold.set("100")
        app._ml_predict_triage()
        assert _drain(app), "triage job hung"
        triage_text = str(app._ml_bundle_info.cget("text"))
        _scroll_to_widget(app, app._ml_bundle_info, offset=0)
        _grab(app, "pg148-ml-triage.png")
        ok_triage = "3 of 6 row(s) at or below 100 nM" in triage_text
        facts.append(f"triage: {triage_text!r}")

        # -- shot 4: the family benchmark board -------------------------- #
        bench_frame = pd.DataFrame(
            {
                "SMILES": [
                    "CCO", "CCCO", "CCCCO", "CCCOO", "CC(=O)O", "CCC(=O)O",
                    "O=C(O)C1CCCO1", "CCOC(=O)c1ccc(N)cc1",
                    "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)Nc1ccc(O)cc1",
                    "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "CC(=O)Oc1ccccc1C(=O)O",
                ],
                "pIC50": [2.0, 2.4, 2.8, 3.2, 4.0, 4.4, 4.8, 5.2, 6.0, 6.4, 6.8, 7.2],
            }
        )
        app._ml_adopt_frame(bench_frame, ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._target_combo.set("pIC50")
        app._task_combo.set("regression")
        app._bench_model.delete(0, "end")
        app._bench_model.insert(0, "ridge")
        app._bench_cv.delete(0, "end")
        app._bench_cv.insert(0, "2")
        app._ml_benchmark_families()
        assert _drain(app), "benchmark job hung"
        rows = app._board_tree.get_children()
        status = str(app._board_status.cget("text"))
        _scroll_to_widget(app, app._board_tree)
        _grab(app, "pg148-ml-benchmark.png")
        ok_bench = len(rows) == 18 and "18 2D families" in status
        facts.append(f"benchmark rows: {len(rows)}, status: {status!r}")

    finally:
        app.destroy()

    print()
    for key, value in (
        ("pr", ok_pr),
        ("families", ok_families),
        ("triage", ok_triage),
        ("benchmark", ok_bench),
    ):
        print(f"  {'OK ' if value else 'FAIL'} {key}")
    for fact in facts:
        print(f"  · {fact}")
    if ok_pr and ok_families and ok_triage and ok_bench:
        print("SHOOT148_OK")
        return 0
    print("SHOOT148_FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
