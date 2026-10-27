"""Tail of the live drive: dialog-neutralized guards + redraw/export."""
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("playground_live", ROOT / "playground.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

seen: list = []


def _quiet_dialogs() -> None:
    for name in ("showerror", "showwarning", "showinfo"):
        def make(_n):
            def _record(*a, **k):
                seen.append((_n, a))
            return _record
        setattr(module.messagebox, name, make(name))


def main() -> int:
    _quiet_dialogs()
    app = module.PlaygroundApp(persist_session=False)
    try:
        app._ml_load_path(ROOT / "examples" / "data" / "qsar_classification.csv")
        app._unsup_select_all()

        def run_and_wait(label):
            app._unsup_run()
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and app._busy:
                app.update_idletasks(); app.update(); time.sleep(0.02)
            assert not app._busy, label

        seen.clear()
        app._unsup_clear_selection()
        app._unsup_run(); app.update()
        assert seen and seen[0][0] == "showwarning", seen
        assert not app._busy
        print("no-selection guard ok:", seen[0][1][1], flush=True)

        frame_backup = app._ml_frame
        app._ml_frame = None
        seen.clear()
        app._unsup_select_all()
        app._unsup_run(); app.update()
        assert seen and seen[0][0] == "showwarning"
        app._ml_frame = frame_backup
        print("no-dataset guard ok:", seen[0][1][1], flush=True)

        seen.clear()
        app._unsup_select_all()
        app._unsup_method.set("K-Means (elbow + silhouette)")
        app._unsup_k.delete(0, "end"); app._unsup_k.insert(0, "bogus")
        app._unsup_run(); app.update()
        assert seen and seen[0][0] == "showwarning" and "k" in str(seen[0][1][1]).lower()
        print("bad-k guard ok:", seen[0][1][1], flush=True)

        app._unsup_k.delete(0, "end"); app._unsup_k.insert(0, "auto")
        app._unsup_method.set("PCA (variance ladder)")
        run_and_wait("PCA")
        app._unsup_plot(); app.update()
        assert app._unsup_host.winfo_children(), "redraw produced no canvas"
        app._unsup_colour.set(app._target_combo.get())
        app._unsup_scatter(); app.update()
        assert app._unsup_host.winfo_children()
        frame = app._unsup_result["record"].as_frame()
        assert frame.shape[0] == len(frame_backup)
        print("redraw + scatter + export frame ok", flush=True)

        app._unsup_result["record"] = object()  # pretend an old record
        app._ml_adopt_frame(frame_backup, ROOT / "examples" / "data" / "qsar_classification.csv")
        assert app._unsup_result is None
        assert app._unsup_summary.cget("text") == "no analysis yet"
        print("stale-state retirement ok", flush=True)
        print("TAIL CHECKS PASSED", flush=True)
        return 0
    finally:
        app.destroy()


if __name__ == "__main__":
    sys.exit(main())
