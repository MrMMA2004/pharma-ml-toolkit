"""100x-51 live driver: session round-trip + dark theme under Xvfb."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "pharma-ml-toolkit"))

import playground as pg  # noqa: E402


def phase1() -> None:
    app = pg.PlaygroundApp()
    for _ in range(2):
        app.update_idletasks()
        app.update()
    app._aff_value.delete(0, "end")
    app._aff_value.insert(0, "3.14")
    app._tani_a.delete(0, "end")
    app._tani_a.insert(0, "c1ccccc1")
    app._tune_var.set(True)
    app._dock_remove_waters_var.set(False)
    app._feat_family_vars["morgan_scaffold"].set(False)
    app._feat_pasted_text.delete("1.0", "end")
    app._feat_pasted_text.insert("1.0", "CCO\nCCN # a comment")
    app._corpus_text.delete("1.0", "end")
    app._corpus_text.insert("1.0", "CCC\nCCCO")
    app._feat_k_entry.delete(0, "end")
    app._feat_k_entry.insert(0, "777")
    app._book.select(4)
    app._theme_var.set("dark")
    app._on_theme_change()
    ok = app._write_session_file()
    print("phase1 write:", ok, "· theme now:", pg._ACTIVE_THEME, "· BG:", pg.BG)
    print("phase1 dark root bg:", app.cget("background"),
          "· corpus text bg:", app._corpus_text.cget("background"))
    app.destroy()
    payload = json.loads(pg.session_path().read_text(encoding="utf-8"))
    assert payload["format"] == 1, payload
    assert payload["theme"] == "dark", payload["theme"]
    assert payload["tab"] == 4, payload["tab"]
    assert payload["variables"]["_aff_value"] == "3.14"
    assert payload["variables"]["_tune_var"] is True
    assert payload["variables"]["_dock_remove_waters_var"] is False
    assert payload["variables"]["_feat_k_entry"] == "777"
    assert payload["families"]["morgan_scaffold"] is False
    assert payload["families"]["descriptors"] is True
    assert payload["texts"]["pasted"] == "CCO\nCCN # a comment"
    assert payload["texts"]["corpus"] == "CCC\nCCCO"
    assert pg.parse_geometry(payload["geometry"]) is not None, payload["geometry"]
    print("phase1 file payload OK")


def phase2() -> None:
    assert pg._ACTIVE_THEME == "light", "module must import light"
    app = pg.PlaygroundApp()
    for _ in range(2):
        app.update_idletasks()
        app.update()
    checks = {
        "aff": app._aff_value.get() == "3.14",
        "tani": app._tani_a.get() == "c1ccccc1",
        "tune": bool(app._tune_var.get()) is True,
        "waters": bool(app._dock_remove_waters_var.get()) is False,
        "morgan": bool(app._feat_family_vars["morgan_scaffold"].get()) is False,
        "desc": bool(app._feat_family_vars["descriptors"].get()) is True,
        "k": app._feat_k_entry.get() == "777",
        "pasted": app._feat_pasted_text.get("1.0", "end-1c") == "CCO\nCCN # a comment",
        "corpus": app._corpus_text.get("1.0", "end-1c") == "CCC\nCCCO",
        "tab": app._book.index(app._book.select()) == 4,
        "theme_dark": pg._ACTIVE_THEME == "dark",
        "root_dark": str(app.cget("background")) == pg._DARK_PALETTE["BG"],
        "text_dark": str(app._corpus_text.cget("background")) == pg._DARK_PALETTE["CARD"],
        "status_note": "session restored" in str(app._job_label.cget("text")),
    }
    bad = [name for name, ok in checks.items() if not ok]
    print("phase2 checks:", sum(checks.values()), "/", len(checks),
          "· failing:", bad or "none")
    app._theme_var.set("light")
    app._on_theme_change()
    for _ in range(2):
        app.update_idletasks()
        app.update()
    light_checks = {
        "light_root": str(app.cget("background")) == pg._LIGHT_PALETTE["BG"],
        "light_text": str(app._corpus_text.cget("background")) == pg._LIGHT_PALETTE["CARD"],
        "light_family": bool(app._feat_family_vars["morgan_scaffold"].get()) is False,
        "theme_persists_in_memory": app._theme == "light",
    }
    bad_light = [name for name, ok in light_checks.items() if not ok]
    print("phase2 light-flip:", sum(light_checks.values()), "/", len(light_checks),
          "· failing:", bad_light or "none")
    app._on_close()
    payload = json.loads(pg.session_path().read_text(encoding="utf-8"))
    assert payload["theme"] == "light", payload["theme"]
    assert payload["variables"]["_aff_value"] == "3.14"
    print("phase2 close-save OK (theme light, inputs intact)")
    if bad or bad_light:
        raise SystemExit("ROUND-TRIP FAILURES: " + ", ".join(bad + bad_light))
    print("ROUND-TRIP GREEN")


if __name__ == "__main__":
    {"phase1": phase1, "phase2": phase2}[sys.argv[1]]()
