"""SMOKE149 -- the round-149 stacking card, fact-checked headlessly (100x-149).

Drives look.ipynb's absorbed card end to end against the REAL library
(pharmaml.ml.stack_oof_regressors -- Task 4-a's module):
  * the card's widgets (defaults, ticks, tooltips implicit);
  * the guards (no dataset, regression-only);
  * the full run: quantile target on, residual correction on, the
    shared board leaderboard with the +residual row, the status note
    naming folds/counts/transform, and the parity scatter in the
    card's own figure host;
  * the residual-off pass: the note and the title drop the correction.

Run:  DISPLAY=:99 python scratch/smoke149.py
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
        base_default = app._stack_base.get()
        meta_default = app._stack_meta.get()
        OK_FLAGS["widgets"] = (
            base_default == "ridge, random_forest, gradient_boosting, extra_trees"
            and meta_default == "ridge, huber_regression, lightgbm"
            and app._stack_folds.get() == "5"
            and app._stack_test.get() == "20"
            and not app._stack_quantile.get()
            and app._stack_residual.get()
        )
        facts.append(f"base entry: {base_default!r}")
        facts.append(f"meta entry: {meta_default!r}")

        # -- 2: the guards ------------------------------------------------- #
        dialogs: list[str] = []
        for name in ("showinfo", "showerror", "showwarning"):
            setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=dialogs, **k: _seen.append(" ".join(str(x) for x in a)),
            )
        app._ml_stack_oof()
        app.update()
        OK_FLAGS["guard_no_dataset"] = any("No dataset" in d for d in dialogs)
        facts.append(f"no-dataset guard: {dialogs[-1][:60]!r}")

        import pandas as pd

        class_frame = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "label": [0, 1, 0, 1]})
        app._ml_adopt_frame(class_frame, ROOT / "examples" / "data" / "qsar_regression.csv")
        app._target_combo.set("label")
        app._task_combo.set("classification")
        app._ml_stack_oof()
        app.update()
        OK_FLAGS["guard_regression_only"] = any("Regression only" in d for d in dialogs)
        facts.append(f"regression-only guard: {dialogs[-1][:60]!r}")
        _quiet()

        # -- 3: the real run (quantile on, residual on) ------------------- #
        notes: list[str] = []
        original_report = app._report_progress

        def spy(done: int, total: object = None, note: str = "") -> None:
            if note:
                notes.append(note)
            original_report(done, total, note)

        app._report_progress = spy
        frame = pd.read_csv(ROOT / "examples" / "data" / "qsar_regression.csv")
        app._ml_adopt_frame(frame, ROOT / "examples" / "data" / "qsar_regression.csv")
        app._target_combo.set("pIC50")
        app._task_combo.set("regression")
        app._stack_base.delete(0, "end")
        app._stack_base.insert(0, "ridge, knn")
        app._stack_meta.delete(0, "end")
        app._stack_meta.insert(0, "ridge")
        app._stack_folds.delete(0, "end")
        app._stack_folds.insert(0, "2")
        app._stack_test.delete(0, "end")
        app._stack_test.insert(0, "20")
        app._stack_quantile.set(True)
        app._ml_stack_oof()
        assert _drain(app, timeout=300.0), "stacking job hung"
        rows = app._board_tree.get_children()
        columns = list(app._board_tree["columns"])
        status = str(app._board_status.cget("text"))
        last_model = app._board_tree.item(rows[-1], "values")[0] if rows else ""
        figure = getattr(app._stack_host, "_pg_figure", None)
        title = figure.axes[0].get_title() if figure is not None else ""
        OK_FLAGS["real_run"] = (
            columns == ["model", "oof_r2", "test_r2"]
            and len(rows) == 2  # ridge + ridge+residual
            and str(last_model).endswith("+residual")
            and "2-fold OOF stacking" in status
            and "2 base model(s) × 1 meta-learner(s)" in status
            and "TRANSFORMED scale" in status
            and figure is not None
            and title == "OOF stacking — ridge + residual"
        )
        facts.append(f"board columns: {columns}, rows: {len(rows)}")
        facts.append(f"last row model: {last_model!r}")
        facts.append(f"status: {status!r}")
        facts.append(f"scatter title: {title!r}")
        facts.append(f"on_step notes (first 5): {notes[:5]}")

        # -- 4: residual off ------------------------------------------------ #
        app._stack_residual.set(False)
        app._stack_quantile.set(False)
        app._ml_stack_oof()
        assert _drain(app, timeout=300.0), "second stacking job hung"
        rows = app._board_tree.get_children()
        status = str(app._board_status.cget("text"))
        figure = getattr(app._stack_host, "_pg_figure", None)
        title = figure.axes[0].get_title() if figure is not None else ""
        OK_FLAGS["residual_off"] = (
            len(rows) == 1
            and "+residual" not in status
            and "raw target scale" in status
            and figure is not None
            and title == "OOF stacking — ridge"
        )
        facts.append(f"residual-off rows: {len(rows)}, status: {status!r}")
        facts.append(f"residual-off title: {title!r}")

    finally:
        app.destroy()

    print()
    for key, value in OK_FLAGS.items():
        print(f"  {'OK ' if value else 'FAIL'} {key}")
    for fact in facts:
        print(f"  · {fact}")
    if all(OK_FLAGS.values()):
        print("SMOKE149_OK")
        return 0
    print("SMOKE149_FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
