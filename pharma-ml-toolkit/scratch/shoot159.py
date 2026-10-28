"""Round-42 artifact shooter: the collapsible cards + the GUI metrics dialog."""
import importlib.util
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("playground_shoot", ROOT / "playground.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

app = module.PlaygroundApp(persist_session=False)
try:
    def shot(name: str) -> None:
        app.update()
        app.update_idletasks()
        for _ in range(24):
            app.update()
            time.sleep(0.01)
        from PIL import ImageGrab  # type: ignore

        img = ImageGrab.grab(xdisplay=":99")
        img.save(f"download/{name}.png")
        print("shot", name, flush=True)

    # -- 1. the PK tab with three calculators folded to their headings -- #
    app._book.select(app._book.tabs()[2])  # PK / PD tab
    for key in ("pk-oral", "pk-nca", "pk-tdm"):
        module._set_card_expanded(key, False)
    app.update()
    shot("pg159-collapsible-folded")

    # -- 2. the finder's unfold-on-jump: jump to a folded card ---------- #
    module._set_card_expanded("home-what-lives-where", False)
    app.update()
    app._jump_to_card(0, "What lives where")
    app.update()
    shot("pg159-finder-unfold")

    # -- 3. the GUI metrics dialog (light palette) ---------------------- #
    app._show_gui_metrics()
    app.update()
    shot("pg159-gui-metrics")

    # -- 4. the same dialog after a live theme flip --------------------- #
    app._theme_var.set("dark")
    app._on_theme_change()
    app.update()
    shot("pg159-gui-metrics-dark")

    # -- 5. the chevron states restored (launch look) -------------------- #
    app._theme_var.set("light")
    app._on_theme_change()
    app._reset_tab(2)
    app.update()
    shot("pg159-collapsible-reset")
    print("ALL SHOTS DONE", flush=True)
finally:
    app.destroy()
