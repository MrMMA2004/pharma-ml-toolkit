"""Round 100x-128 smoke: the per-card "Save figure..." export closes
app-wide -- every remaining figure card (ML results / leaderboard / AD,
DL results and the two-host training history, PK & PD figures, the
feature factory, the similarity scan, both docking cards, the NCA
profile, the utilities reconciliation) carries its own export button --
headless, Xvfb-driven.

Run: DISPLAY=:99 /home/z/.venv/bin/python scratch/smoke128.py
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit")

import matplotlib.pyplot as plt  # noqa: E402
import playground  # noqa: E402

NEW_CARDS = [
    ("_save_feat_figure", ("_feat_host",)),
    ("_save_scan_figure", ("_scan_host",)),
    ("_save_iv_figure", ("_iv_host",)),
    ("_save_pd_figure", ("_pd_host",)),
    ("_save_nca_figure", ("_nca_host",)),
    ("_save_acc_figure", ("_acc_host",)),
    ("_save_fam_figure", ("_fam_host",)),
    ("_save_dock_receptor_figure", ("_dock_receptor_host",)),
    ("_save_dock_results_figure", ("_dock_results_host",)),
    ("_save_ml_figure", ("_ml_host",)),
    ("_save_board_figure", ("_board_host",)),
    ("_save_ad_figure", ("_ad_host",)),
    ("_save_dl_figure", ("_dl_host",)),
    ("_save_dl_history_figure", ("_dl_hist_host", "_dl_curve_host")),
    ("_save_util_cmp_figure", ("_util_cmp_host",)),
]


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

    import pharmaml.viz as viz_module

    real_save = viz_module.save_figure
    captured: dict[str, object] = {}

    def spy(figure, path, *args, **kwargs):
        captured["figure"] = figure
        return real_save(figure, path, *args, **kwargs)

    # -- 1. a REAL path: the IV bolus card's own plot, then its export --- #
    app._iv_c0.delete(0, "end"); app._iv_c0.insert(0, "10")
    app._iv_thalf.delete(0, "end"); app._iv_thalf.insert(0, "4")
    app._iv_hours.delete(0, "end"); app._iv_hours.insert(0, "24")
    app._plot_iv_bolus()
    assert deadline_wait(app, "iv profile"), "the IV profile job hung"
    iv_figure = app._iv_host._pg_figure
    assert iv_figure is not None, "the IV decay curve never landed"
    viz_module.save_figure = spy
    try:
        playground.filedialog.asksaveasfilename = lambda **_k: "/tmp/smoke128-iv.png"
        decoy, decoy_axis = plt.subplots()
        decoy_axis.plot([0, 1], [1, 0])
        app._last_figure = decoy  # something else drew last ANYWHERE
        app._save_iv_figure()
        assert deadline_wait(app, "iv save"), "the IV save job hung"
        assert captured["figure"] is iv_figure, "saved a figure that is not the card's"
        assert captured["figure"] is not decoy
        with open("/tmp/smoke128-iv.png", "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
        assert any("Figure saved" in str(a) for _n, a in seen), seen
        plt.close(decoy)
        seen.clear()
    finally:
        viz_module.save_figure = real_save
    print("1. IV per-card save OK (the card's own real plot over the decoy)")

    # -- 2. the DL history card's TWO hosts: newest of the card's own ---- #
    losses, loss_axis = plt.subplots()
    loss_axis.set_title("loss curves")
    curve, curve_axis = plt.subplots()
    curve_axis.set_title("learning curve")
    again, again_axis = plt.subplots()
    again_axis.set_title("loss curves again")
    try:
        app._render_figure(app._dl_hist_host, losses)
        app._render_figure(app._dl_curve_host, curve)
        assert (
            app._dl_hist_host._pg_figure_seq < app._dl_curve_host._pg_figure_seq
        ), "render stamps are not ordered"
        viz_module.save_figure = spy
        try:
            playground.filedialog.asksaveasfilename = lambda **_k: "/tmp/smoke128-dl.png"
            app._save_dl_history_figure()
            assert deadline_wait(app, "dl save"), "the DL history save job hung"
            assert captured["figure"] is curve, "the newest card figure lost the race"
            app._render_figure(app._dl_hist_host, again)  # losses re-drew last
            app._save_dl_history_figure()
            assert deadline_wait(app, "dl save 2"), "the second DL save job hung"
            assert captured["figure"] is again, "the newest card figure lost the race"
            with open("/tmp/smoke128-dl.png", "rb") as handle:
                assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
            seen.clear()
        finally:
            viz_module.save_figure = real_save
    finally:
        plt.close("all")
    print("2. DL history two-host save OK (the newest of the card's own figures wins)")

    # -- 3. the guards: every new card's honest empty hint --------------- #
    for method_name, host_names in NEW_CARDS:
        for host_name in host_names:
            host = getattr(app, host_name)
            if getattr(host, "_pg_figure", None) is not None:
                host._pg_figure = None
        getattr(app, method_name)()
        assert any("No figure on the card yet" in str(a) for _n, a in seen), (method_name, seen)
        seen.clear()
    print("3. per-card save guards OK (all 15 new buttons)")

    # -- 4. the button census: every tab carries its share --------------- #
    for index, want in ((1, 2), (2, 5), (4, 2), (5, 3), (6, 2), (8, 1), (7, 4)):
        buttons = [
            str(w.cget("text"))
            for w in app._walk_widgets(app._book.nametowidget(app._book.tabs()[index]))
            if w.winfo_class() == "TButton"
        ]
        assert buttons.count("Save figure…") == want, (index, want, buttons.count("Save figure…"))
    print("4. button census OK (19 Save figure… buttons app-wide)")

    # -- 5. the documentation pins ---------------------------------------- #
    doc = playground.__doc__ or ""
    assert "100x-128" in doc and "closes app-wide" in doc
    print("5. module docstring records round 128 OK")

    app.destroy()
    assert not seen, f"unexpected dialogs: {seen}"
    print("SMOKE128_OK")


if __name__ == "__main__":
    main()
