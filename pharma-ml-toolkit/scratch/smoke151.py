"""SMOKE151 -- the round-151 prediction-intervals card, fact-checked
headlessly (100x-151).

Drives D2.py's uncertainty block as the Results card's buttons end to
end against the REAL library (pharmaml.ml.quantile_prediction_intervals
+ pharmaml.viz.plot_prediction_intervals):
  * the card's widgets (the alpha entry's 0.1 default);
  * the guards (no trained model, classification run, alpha bounds);
  * the full run: ridge on the regression example, the quantile trio's
    interval table (actual/pred_median/pred_lower/pred_upper/
    interval_width), the honest coverage/width/crossing note, and the
    band figure in the Results card's host;
  * the Save-intervals twin through a monkeypatched dialog;
  * the retirement: a new train clears the table.

Run:  DISPLAY=:99 python scratch/smoke151.py
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

        # -- 1: the card's widgets --------------------------------------- #
        OK_FLAGS["widgets"] = app._interval_alpha.get() == "0.1"
        facts.append(f"alpha entry: {app._interval_alpha.get()!r}")

        # -- 2: the guards ------------------------------------------------- #
        dialogs: list[str] = []
        for name in ("showinfo", "showerror", "showwarning"):
            setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=dialogs, **k: _seen.append(" ".join(str(x) for x in a)),
            )
        app._ml_plot_intervals()
        app.update()
        OK_FLAGS["guard_no_model"] = any("Train a model first" in d for d in dialogs)
        facts.append(f"no-model guard: {dialogs[-1][:60]!r}")

        import pandas as pd

        class_frame = pd.read_csv(ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._ml_adopt_frame(class_frame, ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._model_combo.set("logistic_regression")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert _drain(app), "classification train job hung"
        app._ml_plot_intervals()
        app.update()
        OK_FLAGS["guard_regression_only"] = any("Regression only" in d for d in dialogs)
        OK_FLAGS["guard_no_job"] = not app._busy
        facts.append(f"regression-only guard: {dialogs[-1][:60]!r}")

        # -- 3: the real run (ridge on the regression example) ------------ #
        _quiet()
        frame = pd.read_csv(ROOT / "examples" / "data" / "qsar_regression.csv")
        app._ml_adopt_frame(frame, ROOT / "examples" / "data" / "qsar_regression.csv")
        app._target_combo.set("pIC50")
        app._task_combo.set("regression")
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert _drain(app), "train job hung"
        # a fresh run retires any old table first
        OK_FLAGS["retired_by_train"] = app._ml_intervals_frame is None

        app._ml_plot_intervals()
        assert _drain(app), "intervals job hung"
        table = app._ml_intervals_frame
        note = str(app._ml_result_note.cget("text"))
        figure = getattr(app._ml_host, "_pg_figure", None)
        title = figure.axes[0].get_title() if figure is not None else ""
        y_test = app._ml_last_run[6]
        OK_FLAGS["real_run"] = (
            table is not None
            and list(table.columns)
            == ["actual", "pred_median", "pred_lower", "pred_upper", "interval_width"]
            and len(table) == len(y_test)
            and (table["pred_lower"] <= table["pred_upper"]).all()
            and "coverage" in note
            and "mean width" in note
            and "crossed bound(s) repaired" in note
            and figure is not None
            and title == "Prediction intervals — ridge (pIC50)"
        )
        facts.append(f"table rows: {len(table)}, columns: {list(table.columns)}")
        facts.append(f"note: {note!r}")
        facts.append(f"figure title: {title!r}")
        bounds = (table["pred_upper"] - table["pred_lower"]).describe()
        facts.append(f"width stats: min={bounds['min']:.3f} mean={bounds['mean']:.3f}")

        # -- 4: the Save-intervals twin ----------------------------------- #
        saved = ROOT / "scratch" / "smoke151-intervals.csv"
        playground.filedialog.asksaveasfilename = (  # type: ignore[assignment]
            lambda t="", _p=saved, **k: str(_p)
        )
        app._ml_intervals_save()
        app.update()
        head = saved.read_text(encoding="utf-8").splitlines()[:3]
        saved.unlink()
        OK_FLAGS["save_twin"] = (
            head[0] == "actual,pred_median,pred_lower,pred_upper,interval_width"
            and len(head) > 2
        )
        facts.append(f"saved csv head: {head!r}")

        # -- 5: a new train retires the table ------------------------------ #
        app._ml_train()
        assert _drain(app), "second train job hung"
        OK_FLAGS["retire_on_new_train"] = app._ml_intervals_frame is None

        # -- 6: the alpha parse guards (dialog, never a job) --------------- #
        dialogs.clear()
        for name in ("showinfo", "showerror", "showwarning"):
            setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=dialogs, **k: _seen.append(" ".join(str(x) for x in a)),
            )
        app._interval_alpha.delete(0, "end")
        app._interval_alpha.insert(0, "0.6")
        app._ml_plot_intervals()
        app.update()
        high_ok = any("alpha" in d for d in dialogs) and not app._busy
        dialogs.clear()
        app._interval_alpha.delete(0, "end")
        app._interval_alpha.insert(0, "abc")
        app._ml_plot_intervals()
        app.update()
        text_ok = any("alpha" in d for d in dialogs) and not app._busy
        OK_FLAGS["alpha_guards"] = high_ok and text_ok
        facts.append(f"alpha 0.6 guard: {high_ok}, 'abc' guard: {text_ok}")

    finally:
        app.destroy()

    print()
    for key, value in OK_FLAGS.items():
        print(f"  {'OK ' if value else 'FAIL'} {key}")
    for fact in facts:
        print(f"  · {fact}")
    if all(OK_FLAGS.values()):
        print("SMOKE151_OK")
        return 0
    print("SMOKE151_FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
