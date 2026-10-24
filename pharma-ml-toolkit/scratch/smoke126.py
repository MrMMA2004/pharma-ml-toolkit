"""Round 100x-126 smoke: the HTML report's theme engine (the report
OPENS in the app's View > theme setting + the per-reader toggle) and
the Pareto front card's per-card "Save figure..." export (the figure
THIS card shows, not the last one drawn anywhere) -- headless,
Xvfb-driven.

Run: DISPLAY=:99 /home/z/.venv/bin/python scratch/smoke126.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit")

import playground  # noqa: E402


def main() -> None:
    app = playground.PlaygroundApp(persist_session=False)
    app.update_idletasks()
    app.update()

    # quiet ALL dialogs up front (headless: any modal would hang --
    # the round-123 trap, hit again in round 126's first test draft)
    seen: list[tuple[str, tuple]] = []
    for name in ("showinfo", "showerror", "showwarning"):
        setattr(
            playground.messagebox,
            name,
            lambda *a, _n=name, **k: seen.append((_n, a)),
        )

    # -- 1. the library theme engine ----------------------------------- #
    from pharmaml.viz import HTMLReport

    for theme, label in (
        ("auto", "◐ auto"),
        ("light", "☀ light"),
        ("dark", "☽ dark"),
    ):
        text = HTMLReport("smoke", theme=theme).render()
        assert f"<html lang='en' data-theme='{theme}'>" in text, theme
        assert f">{label}</button>" in text, theme
    plain = HTMLReport("smoke").render()
    assert plain.count("<script>") == 1 and "localStorage" in plain
    assert "html[data-theme='dark']" in plain and "html[data-theme='auto']" in plain
    assert plain.count("--bg: #0B1220") == 2  # forced block + media block
    assert "http://" not in plain and "https://" not in plain  # self-contained
    try:
        HTMLReport("smoke", theme="blue")
        raise AssertionError("invalid theme accepted")
    except Exception as exc:
        assert "theme" in str(exc)
    print("1. HTMLReport theme engine OK (3 themes, toggle, single inline script)")

    # -- 2. the Pareto per-card save: the CARD's figure, not the last - #
    app._book.select(app._book.tabs()[7])  # Generative
    app.update()
    app._objective_combo.set("Multi-objective (QED × MW × TPSA)")
    app._pop_size.set("12")
    app._generations.set("4")
    app._run_evolution()
    assert deadline_wait(app, "evolution"), "the evolution job hung"
    card_figure = app._pareto_host._pg_figure
    assert card_figure is not None  # the auto-scatter

    captured: dict[str, object] = {}
    import pharmaml.viz as viz_module

    real_save = viz_module.save_figure

    def spy(figure, path, *args, **kwargs):
        captured["figure"] = figure
        return real_save(figure, path, *args, **kwargs)

    viz_module.save_figure = spy
    try:
        playground.filedialog.asksaveasfilename = lambda **_k: "/tmp/smoke126-front.png"
        # the honest distinction: another figure "draws last" elsewhere
        import matplotlib.pyplot as plt

        decoy, decoy_axis = plt.subplots()
        decoy_axis.plot([0, 1], [1, 0])
        app._last_figure = decoy
        app._save_pareto_figure()
        assert captured["figure"] is card_figure, "saved a figure that is not the card's"
        assert captured["figure"] is not decoy
        with open("/tmp/smoke126-front.png", "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
        assert any("Figure saved" in str(a) for _n, a in seen), seen
        plt.close(decoy)
    finally:
        viz_module.save_figure = real_save
    print("2. Pareto per-card save OK (card's figure saved over the decoy _last_figure)")

    # -- 3. the guard path: nothing on the card yet --------------------- #
    seen.clear()
    app._close_embedded(app._pareto_host)
    app._pareto_host._pg_figure = None
    app._save_pareto_figure()
    assert any("No figure on the card yet" in str(a) for _n, a in seen), seen
    print("3. per-card save guard OK")

    # -- 4. the HTML report bakes the APP's theme ------------------------ #
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    app._render_figure(app._pareto_host, figure)
    app._theme_var.set("dark")
    playground.filedialog.asksaveasfilename = lambda **_k: "/tmp/smoke126-report.html"
    app._menu_save_session_report_html()
    assert deadline_wait(app, "html report"), "the HTML report job hung"
    with open("/tmp/smoke126-report.html", encoding="utf-8") as handle:
        text = handle.read()
    assert "<html lang='en' data-theme='dark'>" in text
    assert "id='theme-toggle'" in text and "localStorage" in text
    assert "data:image/png;base64," in text
    assert any("cycles it per reader" in str(a) for _n, a in seen), seen
    plt.close(figure)
    print("4. HTML session report OK (opens dark, toggle + embedded figure inside)")

    plt.close("all")
    app.destroy()
    print("SMOKE126_OK")


def deadline_wait(app: playground.PlaygroundApp, label: str, timeout: float = 240.0) -> bool:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    print(f"TIMEOUT waiting for {label}")
    return False


if __name__ == "__main__":
    import tkinter as tk  # noqa: F401 - guard: this smoke needs a window

    assert tk is not None
    main()
