"""SMOKE148 -- the round-148 surfaces, fact-checked headlessly (100x-148).

Drives the one-by-one notebook-review round's additions:
  * the twenty-family picker (all families ticked, honest widths);
  * the per-class metrics line after a multiclass train;
  * the ROC / PR buttons drawing into the Results host;
  * the pIC50 -> IC50 conversion + the potency triage;
  * the family benchmark filling the Leaderboard board.

Run:  DISPLAY=:99 python scratch/smoke148.py
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

OK_FLAGS: dict[str, bool] = {}


def _drain(app: "playground.PlaygroundApp", timeout: float = 240.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


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
    facts: list[str] = []
    try:
        app.geometry("1280x900+0+0")
        app.update_idletasks()
        app.update()

        # -- 1: the twenty-family picker --------------------------------- #
        n_families = len(app._feat_family_vars)
        width_text = str(app._feat_width_label.cget("text"))
        OK_FLAGS["twenty_families"] = n_families == 20 and "20 families selected" in width_text
        facts.append(f"family picker: {n_families} families, width label: {width_text!r}")

        # -- 2: multiclass train + the per-class line ------------------- #
        import pandas as pd

        frame = pd.read_csv(ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._ml_adopt_frame(frame, ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._model_combo.set("logistic_regression")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert _drain(app), "train job hung"
        labels = [
            child.cget("text")
            for child in app._ml_metrics.winfo_children()
            if child.winfo_class() == "TLabel"
        ]
        per_class_line = next(
            (text for text in labels if ": P " in str(text) and "· R " in str(text)), ""
        )
        OK_FLAGS["per_class_line"] = bool(per_class_line) and "F1" in str(per_class_line)
        facts.append(f"per-class line: {str(per_class_line)[:120]!r}")

        # -- 3: the PR button draws --------------------------------------- #
        app._ml_plot_pr()
        assert _drain(app), "PR job hung"
        note = str(app._ml_result_note.cget("text"))
        OK_FLAGS["pr_button"] = (
            "PR curves replace the parity figure" in note
            and getattr(app._ml_host, "_pg_figure", None) is not None
        )
        facts.append(f"PR note: {note!r}")

        # -- 4: the IC50 conversion + triage ------------------------------ #
        app._ml_predictions = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN", "CCCO", "CCCCO", "CC(=O)O", "CCC(=O)O"],
                "prediction": [9.0, 8.0, 7.0, 6.0, 5.5, 5.0],
            }
        )
        app._ml_predict_add_ic50()
        assert _drain(app), "IC50 job hung"
        converted = app._ml_predictions["IC50_nM"].tolist()
        OK_FLAGS["ic50_conversion"] = converted == [1.0, 10.0, 100.0, 1000.0, 3162.2776601683795, 10000.0]
        facts.append(f"IC50_nM: {converted}")
        app._predict_threshold.set("100")
        app._ml_predict_triage()
        assert _drain(app), "triage job hung"
        triage_text = str(app._ml_bundle_info.cget("text"))
        OK_FLAGS["triage"] = "3 of 6 row(s) at or below 100 nM" in triage_text
        facts.append(f"triage: {triage_text!r}")

        # -- 5: the family benchmark board --------------------------------- #
        # the multiclass example carries no SMILES column (its guard fired
        # in the probe); the benchmark needs a SMILES dataset like the
        # notebook's own dop_train.csv
        import numpy as _np

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
        assert _drain(app, timeout=300.0), "benchmark job hung"
        rows = app._board_tree.get_children()
        status = str(app._board_status.cget("text"))
        first_row = app._board_tree.item(rows[0], "values") if rows else []
        OK_FLAGS["benchmark"] = len(rows) == 18 and "18 2D families" in status
        facts.append(f"benchmark rows: {len(rows)}, status: {status!r}")
        facts.append(f"benchmark best row: {first_row[:4]}")
        _ = _np.array([])  # keep the import honest for the linter

    finally:
        app.destroy()

    print()
    for key, value in OK_FLAGS.items():
        print(f"  {'OK ' if value else 'FAIL'} {key}")
    for fact in facts:
        print(f"  · {fact}")
    if all(OK_FLAGS.values()):
        print("SMOKE148_OK")
        return 0
    print("SMOKE148_FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
