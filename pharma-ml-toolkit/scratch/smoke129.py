"""Round 100x-129 smoke: the PDF session report's dark-ink variant rides
the View ▸ theme radios -- a dark app writes near-black pages with light
ink and white figure plates, a light app writes the classic colourless
paper, auto resolves through _resolve_auto_theme before the ink freezes
-- headless, Xvfb-driven.

Run: DISPLAY=:99 python scratch/smoke129.py
"""

from __future__ import annotations

import re
import sys
import time
import zlib

sys.path.insert(0, ".")

import playground  # noqa: E402

PAGE_BG = b"0.0824 0.102 0.0941 rg"
INK = b"0.9059 0.9255 0.9137 rg"
PLATE = b"1 1 1 rg"


def deadline_wait(app, label: str, timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    print(f"TIMEOUT waiting for {label}")
    return False


def pdf_streams(pdf_bytes: bytes) -> bytes:
    out: list[bytes] = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, re.S):
        try:
            out.append(zlib.decompress(match.group(1)))
        except zlib.error:
            pass
    return b"\n".join(out)


def save_report(app, target: str) -> bytes:
    playground.filedialog.asksaveasfilename = lambda **_k: target
    app._menu_save_session_report()
    assert deadline_wait(app, "session report"), "the session report job hung"
    with open(target, "rb") as handle:
        return handle.read()


def main() -> None:
    # -- 0. the library contract ---------------------------------------- #
    from pharmaml.viz import PDF_THEMES, save_pdf_report

    assert PDF_THEMES == ("light", "dark"), PDF_THEMES
    assert "auto" not in PDF_THEMES
    try:
        save_pdf_report({}, out_path="/tmp/smoke129-guard.pdf", theme="sepia")
    except ValueError as exc:
        assert "frozen at write time" in str(exc), exc
    else:  # pragma: no cover - the guard must raise
        raise AssertionError("the theme guard did not raise")
    print("0. library contract: PDF_THEMES + the frozen-ink guard OK")

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

    # -- 1. a REAL live figure + table for the report walk --------------- #
    app._iv_c0.delete(0, "end"); app._iv_c0.insert(0, "12")
    app._iv_thalf.delete(0, "end"); app._iv_thalf.insert(0, "6")
    app._iv_hours.delete(0, "end"); app._iv_hours.insert(0, "24")
    app._plot_iv_bolus()
    assert deadline_wait(app, "iv profile"), "the IV profile job hung"
    assert app._iv_host._pg_figure is not None, "the IV decay curve never landed"
    app._desc_tree.insert("", "end", values=("MolWt", "194.19"))
    print("1. live IV figure + descriptor row on the session registry OK")

    # -- 2. DARK: the radio writes the dark-ink variant ------------------ #
    app._theme_var.set("dark")
    app._on_theme_change()
    assert app._theme == "dark"
    dark_bytes = save_report(app, "/tmp/smoke129-dark.pdf")
    assert dark_bytes[:5] == b"%PDF-", "not a PDF"
    ops = pdf_streams(dark_bytes)
    assert PAGE_BG in ops, "dark pages never filled"
    assert PLATE in ops, "the figure plate never landed"
    assert INK in ops, "light body ink never landed"
    assert any("dark ink variant" in str(a) for _n, a in seen), seen
    assert any("View" in str(a) for _n, a in seen), "the flip hint never fired"
    seen.clear()
    print("2. dark radio -> dark ink (page fill + plate + light ink) OK")

    # -- 3. LIGHT: the classic paper stays colourless -------------------- #
    app._theme_var.set("light")
    app._on_theme_change()
    light_bytes = save_report(app, "/tmp/smoke129-light.pdf")
    assert light_bytes[:5] == b"%PDF-", "not a PDF"
    assert light_bytes != dark_bytes, "the variant changed nothing"
    assert b" rg" not in pdf_streams(light_bytes), "light paper gained colour ops"
    assert any("classic light paper" in str(a) for _n, a in seen), seen
    seen.clear()
    print("3. light radio -> classic paper, zero colour operators OK")

    # -- 4. AUTO resolves before the ink freezes ------------------------- #
    app._theme_var.set("auto")
    app._resolve_auto_theme = lambda: "dark"
    auto_bytes = save_report(app, "/tmp/smoke129-auto.pdf")
    assert PAGE_BG in pdf_streams(auto_bytes), "auto did not resolve to dark ink"
    app._theme_var.set("light")
    app._on_theme_change()
    seen.clear()
    print("4. auto resolves through _resolve_auto_theme at save time OK")

    # -- 5. the View menu caption pins the riding ------------------------ #
    import tkinter as tkmod

    bar = app.nametowidget(app.cget("menu"))
    labels: list[str] = []
    for index in range(bar.index("end") + 1):
        if bar.type(index) != "cascade":
            continue
        if str(bar.entrycget(index, "label")).strip() != "View":
            continue
        sub = app.nametowidget(bar.entrycget(index, "menu"))
        for j in range(sub.index("end") + 1):
            if sub.type(j) in ("command", "radiobutton"):
                labels.append(str(sub.entrycget(j, "label")))
    assert any("session reports save in this theme" in label for label in labels), labels
    assert "Follow system" in labels and "Dark" in labels, labels
    print("5. View menu caption '(both session reports save in this theme)' OK")

    app.destroy()
    print("SMOKE129_OK")


if __name__ == "__main__":
    main()
