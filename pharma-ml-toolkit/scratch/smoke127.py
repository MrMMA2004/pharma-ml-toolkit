"""Round 100x-127 smoke: the per-card "Save figure..." export
generalised to every Generative figure card (the two-host Results card
picks the figure that drew LAST on that card; both surgery cards carry
their own button) and the HTML report's bake guarded by the library's
public REPORT_THEMES contract -- headless, Xvfb-driven.

Run: DISPLAY=:99 /home/z/.venv/bin/python scratch/smoke127.py
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit")

import matplotlib.pyplot as plt  # noqa: E402
import playground  # noqa: E402


def deadline_wait(app, label: str, timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    print(f"TIMEOUT waiting for {label}")
    return False


def main() -> None:
    app = playground.PlaygroundApp(persist_session=False)
    app.update_idletasks()
    app.update()

    # quiet ALL dialogs up front (headless: any modal would hang --
    # the standing round-123 trap)
    seen: list[tuple[str, tuple]] = []
    for name in ("showinfo", "showerror", "showwarning"):
        setattr(
            playground.messagebox,
            name,
            lambda *a, _n=name, **k: seen.append((_n, a)),
        )

    # -- 1. the ring-topology card's own export ------------------------- #
    app._jump_to_tab(7)
    app._topo_smiles.delete(0, "end")
    app._topo_smiles.insert(0, "C1CCNCC1")  # piperidine: 3 mutants
    app._gen_topology_moves()
    assert deadline_wait(app, "topology"), "the topology job hung"
    app._gen_draw_topology()
    assert deadline_wait(app, "draw-topology"), "the draw-topology job hung"
    topo_grid = app._topo_host._pg_figure
    assert topo_grid is not None, "the mutant grid never landed"

    import pharmaml.viz as viz_module

    real_save = viz_module.save_figure
    captured: dict[str, object] = {}

    def spy(figure, path, *args, **kwargs):
        captured["figure"] = figure
        return real_save(figure, path, *args, **kwargs)

    viz_module.save_figure = spy
    try:
        playground.filedialog.asksaveasfilename = lambda **_k: "/tmp/smoke127-topo.png"
        decoy, decoy_axis = plt.subplots()
        decoy_axis.plot([0, 1], [1, 0])
        app._last_figure = decoy  # something else drew last ANYWHERE
        app._save_topo_figure()
        assert deadline_wait(app, "topo save"), "the topo save job hung"
        assert captured["figure"] is topo_grid, "saved a figure that is not the card's"
        assert captured["figure"] is not decoy
        with open("/tmp/smoke127-topo.png", "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
        assert any("Figure saved" in str(a) for _n, a in seen), seen
        plt.close(decoy)
        seen.clear()
    finally:
        viz_module.save_figure = real_save
    print("1. ring-topology per-card save OK (the card's grid over the decoy)")

    # -- 2. the Results card's TWO hosts: newest of the card's own ----- #
    history, history_axis = plt.subplots()
    history_axis.set_title("evolution history")
    grid, grid_axis = plt.subplots()
    grid_axis.set_title("leaderboard grid")
    again, again_axis = plt.subplots()
    again_axis.set_title("history again")
    try:
        app._render_figure(app._ga_host, history)
        app._render_figure(app._ga_grid_host, grid)
        assert (
            app._ga_host._pg_figure_seq < app._ga_grid_host._pg_figure_seq
        ), "render stamps are not ordered"
        viz_module.save_figure = spy
        try:
            playground.filedialog.asksaveasfilename = lambda **_k: "/tmp/smoke127-ga.png"
            app._save_ga_figure()
            assert deadline_wait(app, "ga save"), "the ga save job hung"
            assert captured["figure"] is grid, "the newest card figure lost the race"
            app._render_figure(app._ga_host, again)  # history re-drew last
            app._save_ga_figure()
            assert deadline_wait(app, "ga save 2"), "the second ga save job hung"
            assert captured["figure"] is again, "the newest card figure lost the race"
            with open("/tmp/smoke127-ga.png", "rb") as handle:
                assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
            seen.clear()
        finally:
            viz_module.save_figure = real_save
    finally:
        plt.close("all")
    print("2. Results two-host save OK (the newest of the card's own figures wins)")

    # -- 3. the guards: nothing on the cards yet ------------------------ #
    for host, saver, label in (
        (app._topo_host, app._save_topo_figure, "topology"),
        (app._breed_host, app._save_breed_figure, "breed"),
    ):
        saved = getattr(host, "_pg_figure", None)
        host._pg_figure = None
        saver()
        assert any("No figure on the card yet" in str(a) for _n, a in seen), (label, seen)
        host._pg_figure = saved
        seen.clear()
    saved_grid, saved_host = app._ga_grid_host._pg_figure, app._ga_host._pg_figure
    app._ga_grid_host._pg_figure = None
    app._ga_host._pg_figure = None
    app._save_ga_figure()
    assert any("No figure on the card yet" in str(a) for _n, a in seen), seen
    app._ga_grid_host._pg_figure, app._ga_host._pg_figure = saved_grid, saved_host
    seen.clear()
    print("3. per-card save guards OK (all three new buttons)")

    # -- 4. the four buttons ride the Generative tab -------------------- #
    buttons = [
        str(w.cget("text"))
        for w in app._walk_widgets(app._book.nametowidget(app._book.tabs()[7]))
        if w.winfo_class() == "TButton"
    ]
    assert buttons.count("Save figure…") == 4, buttons.count("Save figure…")
    print("4. four Save figure… buttons on the Generative tab OK")

    # -- 5. the HTML bake guards REPORT_THEMES -------------------------- #
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    app._render_figure(app._pareto_host, figure)
    app._theme_var.set("neon")  # a drifted value must NOT reach HTMLReport
    playground.filedialog.asksaveasfilename = lambda **_k: "/tmp/smoke127-report.html"
    app._menu_save_session_report_html()
    assert deadline_wait(app, "html report"), "the guarded HTML report job hung"
    with open("/tmp/smoke127-report.html", encoding="utf-8") as handle:
        text = handle.read()
    assert "<html lang='en' data-theme='auto'>" in text, "the guard did not fall back to auto"
    assert "data:image/png;base64," in text
    app._theme_var.set("auto")
    seen.clear()
    plt.close("all")
    print("5. REPORT_THEMES bake guard OK (drifted value -> auto)")

    app.destroy()
    assert not seen, f"unexpected dialogs: {seen}"
    print("SMOKE127_OK")


if __name__ == "__main__":
    main()
