#!/usr/bin/env python3
"""(100x-159) The second DearPyGui pass on playground.py.

Round 41 (1.2.94) shipped the first pass -- figure toolbars + crosshair,
table chrome, the font scale -- but the session died mid-close-out (the
worklog entry, the notebook re-execution and three green-suite fixes
never ran; this round closes that debt too).

This patch lands the two STRETCH rows of docs/COVERAGE.md's Round-41
disposition table, translated honestly to tkinter:

  1. ``add_collapsing_header`` -- the ``CollapsibleCard`` retrofit:
     ``make_card(collapsible=True, key=...)`` folds a card to its
     heading via a \\u25be/\\u25b8 chevron; the fold persists with the
     session (``collapsed_cards`` text key, the tolerant JSON trick),
     a finder jump unfolds the target first, and the per-tab reset
     restores the launch look (expanded).  Retrofitted onto the two
     static Home reference cards + all thirteen PK-tab calculators
     (the tallest tab -- fold the calculators you are not using).
  2. The developer-tools row -- View \\u25b8 GUI metrics...: the live
     widget census by Tk class, card/tree/toolbar/job totals, the
     active preferences, and the palette inspector (one chip + hex per
     palette key, re-tinted live on theme flips).

Plus two fixes the review found on the way: the round-41 menu build
re-assigned ``_menu_tints`` AFTER appending the font cascade (the font
submenu silently lost its theme tint on every flip), and the retint
walker had no ``Toplevel`` branch (the metrics + finder dialogs kept
the old window background).

Run from the toolkit root; asserts every anchor landed exactly once.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "playground.py"
src = TARGET.read_text(encoding="utf-8")


def patch(anchor: str, replacement: str, *, count: int = 1) -> None:
    global src
    found = src.count(anchor)
    assert found == count, f"anchor x{found} (want {count}): {anchor[:80]!r}"
    src = src.replace(anchor, replacement, count)


# ------------------------------------------------------------------ #
# 1. the registry + helpers, inserted just before make_card           #
# ------------------------------------------------------------------ #
REGISTRY = '''#: (100x-159) DearPyGui ``add_collapsing_header`` parity: the registry of
#: collapsible cards -- stable key -> bookkeeping (title / shell /
#: content / chevron button / expanded flag).  The session's
#: ``collapsed_cards`` text key persists which cards sit folded, so a
#: fold survives restarts exactly like any other input; the per-tab
#: reset unfolds its own cards (launch defaults = expanded).
_COLLAPSE_CARDS: dict[str, dict[str, Any]] = {}


def _set_card_expanded(key: str, expanded: bool) -> None:
    """Fold/unfold one registered card (never raises on a dead shell)."""
    entry = _COLLAPSE_CARDS.get(key)
    if entry is None:
        return
    try:
        if expanded:
            entry["content"].pack(fill="both", expand=True)
        else:
            entry["content"].pack_forget()
        entry["button"].configure(text="\\u25be" if expanded else "\\u25b8")
        entry["expanded"] = bool(expanded)
    except tk.TclError:  # pragma: no cover - destroyed mid-toggle
        return


def _toggle_collapsible_card(key: str) -> None:
    """The chevron button's command -- flip one card's fold state."""
    entry = _COLLAPSE_CARDS.get(key)
    if entry is not None:
        _set_card_expanded(key, not bool(entry.get("expanded", True)))


def collapsed_card_keys() -> list[str]:
    """Which registered cards currently sit folded (sorted, stable)."""
    return sorted(
        key
        for key, entry in _COLLAPSE_CARDS.items()
        if not bool(entry.get("expanded", True))
    )


def make_card(
'''
patch("def make_card(\n", REGISTRY, count=1)

# ------------------------------------------------------------------ #
# 2. the signature + docstring + collapsible body                     #
# ------------------------------------------------------------------ #
patch(
    "    span: int = 1,\n) -> Any:\n",
    '    span: int = 1,\n    collapsible: bool = False,\n    key: str = "",\n) -> Any:\n',
)
patch(
    "    subtitle, shell) -- the Ctrl+F card finder's directory (100x-120).\n    \"\"\"\n",
    "    subtitle, shell) -- the Ctrl+F card finder's directory (100x-120).\n\n"
    "    (100x-159) ``collapsible=True`` adds the DearPyGui\n"
    "    ``add_collapsing_header`` affordance: a \\u25be/\\u25b8 chevron on\n"
    "    the header's right folds the card to its heading (inputs and\n"
    "    all -- they keep their values, only the body hides).  ``key`` is\n"
    "    the stable registry/session key (the title stands in when\n"
    "    omitted); a finder jump to a folded card unfolds it first, the\n"
    "    per-tab reset restores the launch look, and the fold state\n"
    "    persists with the session (``collapsed_cards`` text key).\n"
    '    """\n',
)
patch(
    "    content.columnconfigure(1, weight=1)\n    return content\n",
    '''    content.columnconfigure(1, weight=1)
    if collapsible:
        card_key = key or title
        chevron = ttk.Button(
            title_row,
            text="\\u25be",
            style="Ghost.TButton",
            width=3,
            command=lambda bound=card_key: _toggle_collapsible_card(bound),
        )
        chevron.pack(side="right", anchor="ne")
        Tooltip(
            chevron,
            "Collapse this card to its heading (or expand it back) -- "
            "the inputs keep their values, and the fold state is saved "
            "with the session.",
        )
        _COLLAPSE_CARDS[card_key] = {
            "title": title,
            "shell": shell,
            "content": content,
            "button": chevron,
            "expanded": True,
        }
    return content
''',
)

# ------------------------------------------------------------------ #
# 3. the View menu entry + the _menu_tints overwrite fix              #
# ------------------------------------------------------------------ #
patch(
    '        view_menu.add_cascade(label="Font size", menu=font_menu)\n'
    "        self._menu_tints.append(font_menu)\n"
    '        bar.add_cascade(label="View", menu=view_menu)\n',
    '        view_menu.add_cascade(label="Font size", menu=font_menu)\n'
    "        self._menu_tints.append(font_menu)\n"
    "        view_menu.add_separator()\n"
    "        # (100x-159) DearPyGui's developer tools, translated honestly:\n"
    "        # the metrics window's census + the tool inspector's palette.\n"
    "        view_menu.add_command(\n"
    '            label="GUI metrics\\u2026",\n'
    "            command=self._show_gui_metrics,\n"
    "        )\n"
    '        bar.add_cascade(label="View", menu=view_menu)\n',
)
patch(
    "        self.configure(menu=bar)\n"
    "        self._menu_tints = [bar, file_menu, view_menu, help_menu]\n",
    "        self.configure(menu=bar)\n"
    "        # (100x-159) the font cascade rides the tint list too -- the\n"
    "        # round-41 build appended it ABOVE this reassignment, so the\n"
    "        # font submenu silently lost its tint on every theme flip.\n"
    "        self._menu_tints = [bar, file_menu, view_menu, font_menu, help_menu]\n",
)

# ------------------------------------------------------------------ #
# 4. the metrics dialog state, declared with its 158-era siblings     #
# ------------------------------------------------------------------ #
patch(
    "        self._ui_font_size = 10\n",
    "        self._ui_font_size = 10\n"
    "        # (100x-159) the GUI-metrics dialog's live handles -- None\n"
    "        # when closed; _refresh_metrics_dialog prunes stale refs.\n"
    "        self._metrics_dialog: Any = None\n"
    "        self._metrics_palette: Any = None\n"
    "        self._metrics_text: Any = None\n",
)

# ------------------------------------------------------------------ #
# 5. theme flips refresh the metrics dialog                            #
# ------------------------------------------------------------------ #
patch(
    "        for _host, toolbar in self._live_toolbars:\n"
    "            with contextlib.suppress(tk.TclError):\n"
    "                if toolbar.winfo_exists():\n"
    "                    self._style_figure_toolbar(toolbar)\n",
    "        for _host, toolbar in self._live_toolbars:\n"
    "            with contextlib.suppress(tk.TclError):\n"
    "                if toolbar.winfo_exists():\n"
    "                    self._style_figure_toolbar(toolbar)\n"
    "        # (100x-159) the metrics dialog's palette block follows the flip\n"
    "        self._refresh_metrics_dialog()\n",
)

# ------------------------------------------------------------------ #
# 6. the retint walker's Toplevel branch                              #
# ------------------------------------------------------------------ #
patch(
    "                    elif klass == \"Label\":\n"
    "                        # the toolbars' coordinate-readout labels AND the PNG\n"
    "                        # figure fallback labels (a latent dark-theme straggler\n"
    "                        # before this branch existed)\n"
    "                        if str(child.cget(\"background\")) == previous[\"CARD\"]:\n"
    "                            child.configure(background=CARD)\n",
    "                    elif klass == \"Label\":\n"
    "                        # the toolbars' coordinate-readout labels AND the PNG\n"
    "                        # figure fallback labels (a latent dark-theme straggler\n"
    "                        # before this branch existed)\n"
    "                        if str(child.cget(\"background\")) == previous[\"CARD\"]:\n"
    "                            child.configure(background=CARD)\n"
    "                    elif klass == \"Toplevel\":\n"
    "                        # (100x-159) the metrics dialog (and the card finder):\n"
    "                        # their window background follows the flip too\n"
    "                        if str(child.cget(\"background\")) == previous[\"BG\"]:\n"
    "                            child.configure(background=BG)\n",
)

# ------------------------------------------------------------------ #
# 7. session collect: the collapsed_cards text key                    #
# ------------------------------------------------------------------ #
patch(
    "                    if kind in (\"table\", \"figure\")\n"
    "                    and isinstance(labels, list)\n"
    "                }\n"
    "            ),\n"
    "        }\n",
    "                    if kind in (\"table\", \"figure\")\n"
    "                    and isinstance(labels, list)\n"
    "                }\n"
    "            ),\n"
    "            # 100x-159: which collapsible cards sit folded -- the same\n"
    "            # JSON-in-a-text trick, one sorted list of stable card keys\n"
    "            # (the restore path parses tolerantly; a foreign value just\n"
    "            # leaves every card open, never a half-folded window).\n"
    '            "collapsed_cards": json.dumps(collapsed_card_keys()),\n'
    "        }\n",
)

# ------------------------------------------------------------------ #
# 8. session restore: the tolerant branch                             #
# ------------------------------------------------------------------ #
patch(
    "                        and all(isinstance(label, str) for label in labels)\n"
    "                    }\n"
    "                continue\n"
    "            widget = text_widgets.get(key)\n",
    "                        and all(isinstance(label, str) for label in labels)\n"
    "                    }\n"
    "                continue\n"
    '            if key == "collapsed_cards":\n'
    "                # 100x-159: the collapsible cards' fold state -- a JSON\n"
    "                # list of stable card keys parsed with the SAME\n"
    "                # tolerance (a corrupt value leaves everything open);\n"
    "                # unknown keys (renamed cards, older sessions) skip\n"
    "                # quietly, exactly like every other tolerant restore.\n"
    "                try:\n"
    "                    stored_cards = json.loads(content)\n"
    "                except (TypeError, ValueError):\n"
    "                    stored_cards = []\n"
    "                if isinstance(stored_cards, list):\n"
    "                    for card_key in stored_cards:\n"
    "                        if isinstance(card_key, str) and card_key in _COLLAPSE_CARDS:\n"
    "                            _set_card_expanded(card_key, False)\n"
    "                continue\n"
    "            widget = text_widgets.get(key)\n",
)

# ------------------------------------------------------------------ #
# 9. the finder's unfold-on-jump                                       #
# ------------------------------------------------------------------ #
patch(
    "            ):\n"
    "                continue\n"
    "            page = self._book.nametowidget(self._book.tabs()[tab_index])\n",
    "            ):\n"
    "                continue\n"
    "            # (100x-159) a jump to a folded card unfolds it first --\n"
    "            # scrolling to a hidden body would be a lie.\n"
    "            for collapse_key, entry in _COLLAPSE_CARDS.items():\n"
    '                if entry["shell"] is shell and not bool(entry.get("expanded", True)):\n'
    "                    _set_card_expanded(collapse_key, True)\n"
    "                    self.update_idletasks()\n"
    "            page = self._book.nametowidget(self._book.tabs()[tab_index])\n",
)

# ------------------------------------------------------------------ #
# 10. the per-tab reset restores the launch look                       #
# ------------------------------------------------------------------ #
patch(
    "        if label.strip() == \"Data & ML\":\n"
    "            self._ml_params_var.set(\"{}\")\n"
    "            with contextlib.suppress(Exception):  # best effort\n"
    "                self._ml_refresh_hyperparam_panel()\n"
    "        self._job_label.configure(\n"
    "            text=f\"{label.lower()} tab reset \u00b7 {applied} inputs back to defaults\"\n"
    "        )\n",
    "        if label.strip() == \"Data & ML\":\n"
    "            self._ml_params_var.set(\"{}\")\n"
    "            with contextlib.suppress(Exception):  # best effort\n"
    "                self._ml_refresh_hyperparam_panel()\n"
    "        # (100x-159) launch defaults mean EXPANDED cards too -- every\n"
    "        # folded collapsible card on THIS tab unfolds with the reset\n"
    "        # (other tabs keep their folds -- the per-tab rule).\n"
    "        pages = [self._book.nametowidget(tab) for tab in self._book.tabs()]\n"
    "        unfolded = 0\n"
    "        for card_key, entry in _COLLAPSE_CARDS.items():\n"
    '            shell = entry["shell"]\n'
    "            with contextlib.suppress(tk.TclError):\n"
    "                if (\n"
    '                    not bool(entry.get("expanded", True))\n'
    "                    and bool(shell.winfo_exists())\n"
    "                    and self._card_tab_index(shell, pages) == index\n"
    "                ):\n"
    "                    _set_card_expanded(card_key, True)\n"
    "                    unfolded += 1\n"
    '        extra = f" \u00b7 {unfolded} card(s) unfolded" if unfolded else ""\n'
    "        self._job_label.configure(\n"
    '            text=f"{label.lower()} tab reset \u00b7 {applied} inputs back to defaults{extra}"\n'
    "        )\n",
)

# ------------------------------------------------------------------ #
# 11. the GUI metrics + palette inspector methods                      #
# ------------------------------------------------------------------ #
METRICS = '''    # -- (100x-159) GUI metrics + palette inspector ------------------- #
    def _gui_metrics_snapshot(self) -> str:
        """The census/preferences text block the metrics dialog shows.

        Pure read-side: the widget census walks the live tree once
        (``_walk_widgets`` -- the same walk the chrome sweep uses), the
        card/tree/toolbar/job totals come from the registries that
        already exist, and the preferences line reads the four
        View-menu variables.  A SNAPSHOT, honestly labelled: live
        counters would burn CPU for nobody's benefit; the palette block
        below the text is the part that follows theme flips live.
        """
        cards = self._live_cards()
        live_keys: set[str] = set()
        folded: list[str] = []
        for card_key, entry in _COLLAPSE_CARDS.items():
            with contextlib.suppress(tk.TclError):
                if not bool(entry["shell"].winfo_exists()):
                    continue
            live_keys.add(card_key)
            if not bool(entry.get("expanded", True)):
                folded.append(card_key)
        trees = 0
        tree_rows = 0
        counts: dict[str, int] = {}
        for widget in self._walk_widgets(self):
            klass = widget.winfo_class()
            counts[klass] = counts.get(klass, 0) + 1
            if klass == "Treeview":
                trees += 1
                with contextlib.suppress(tk.TclError):
                    tree_rows += len(widget.get_children(""))
        try:
            queued = self._jobs.qsize()
        except NotImplementedError:  # pragma: no cover - exotic queues
            queued = 0
        session_file = session_path()
        if not self._persist_session:
            session_line = "session persistence off (--no-session)"
        elif session_file.exists():
            stamp = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(session_file.stat().st_mtime)
            )
            session_line = f"{session_file} \\u00b7 last saved {stamp}"
        else:
            session_line = f"{session_file} \\u00b7 not saved yet"
        lines = [
            f"CARDS      {len(cards)} live \\u00b7 {len(live_keys)} collapsible \\u00b7 {len(folded)} folded",
            f"TREES      {trees} Treeviews \\u00b7 {tree_rows:,} rows",
            f"TOOLBARS   {len(self._live_toolbars)} live figure toolbars",
            f"JOBS       {'busy' if self._busy else 'idle'} \\u00b7 queue \\u2248 {queued} pending",
            "PREFS      "
            f"theme {self._theme} (resolved {_ACTIVE_THEME}) \\u00b7 "
            f"font {self._font_var.get()} pt \\u00b7 "
            f"toolbars {'on' if self._toolbars_var.get() else 'off'} \\u00b7 "
            f"shortcut hints {'on' if self._hints_var.get() else 'off'}",
            f"SESSION    {session_line}",
            "",
            "WIDGET CENSUS (by Tk class)",
        ]
        for klass, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"{klass:<12} {count:>7,}")
        return "\\n".join(lines)

    def _show_gui_metrics(self) -> None:
        """View \\u25b8 GUI metrics -- the census + the palette inspector.

        (100x-159) DearPyGui's developer tools (the metrics window, the
        theme inspector) translated honestly: the census is a snapshot
        at open, while the PALETTE block under it follows theme flips
        live -- every key of the active palette as a colour chip with
        its hex value, so "which colour is HOVER?" is one glance, not a
        source dive.  Re-opening lifts the existing dialog instead of
        stacking a second one.
        """
        if tk is None:  # pragma: no cover - headless import
            return
        existing = self._metrics_dialog
        if existing is not None:
            with contextlib.suppress(tk.TclError):
                if bool(existing.winfo_exists()):
                    existing.lift()
                    return
        win = tk.Toplevel(self)
        win.title("GUI metrics")
        win.configure(background=BG)
        win.transient(self)
        try:
            win.geometry(f"+{self.winfo_rootx() + 120}+{self.winfo_rooty() + 90}")
        except tk.TclError:  # pragma: no cover - unmapped parent
            pass
        ttk.Label(
            win,
            text="Widget census + preferences (a snapshot at open) -- "
            "the palette below follows theme flips live.",
            style="CardMuted.TLabel",
            wraplength=520,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(12, 4))
        text = tk.Text(
            win,
            height=22,
            width=72,
            background=CARD,
            foreground=INK,
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            padx=8,
            pady=6,
            font=("TkFixedFont", 10),
            wrap="none",
        )
        text.insert("1.0", self._gui_metrics_snapshot())
        text.configure(state="disabled")
        text.pack(fill="x", padx=14, pady=(0, 8))
        Tooltip(
            text,
            "The widget census by Tk class, the card / tree / toolbar / "
            "job totals and the active preferences -- a snapshot from "
            "the moment the dialog opened.",
        )
        self._metrics_text = text
        self._metrics_palette_title = ttk.Label(
            win,
            text=f"Palette \\u2014 {_ACTIVE_THEME} "
            f"({len(_PALETTES[_ACTIVE_THEME])} keys)",
            style="CardTitle.TLabel",
        )
        self._metrics_palette_title.pack(anchor="w", padx=14, pady=(0, 4))
        self._metrics_palette = ttk.Frame(win, style="Card.TFrame")
        self._metrics_palette.pack(fill="x", padx=14, pady=(0, 8))
        self._populate_metrics_palette()
        close = ttk.Button(win, text="Close", style="Ghost.TButton", command=win.destroy)
        close.pack(anchor="e", padx=14, pady=(0, 12))
        Tooltip(close, "Close the metrics dialog (View \\u25b8 GUI metrics re-opens it).")
        self._metrics_dialog = win

    def _populate_metrics_palette(self) -> None:
        """(Re)build the palette chip grid -- one row per palette key."""
        holder = self._metrics_palette
        if holder is None:
            return
        with contextlib.suppress(tk.TclError):
            for child in holder.winfo_children():
                child.destroy()
        for row, (pal_key, value) in enumerate(sorted(_PALETTES[_ACTIVE_THEME].items())):
            chip = tk.Frame(
                holder,
                width=18,
                height=12,
                background=value,
                highlightbackground=BORDER,
                highlightthickness=1,
            )
            chip.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=1)
            chip.pack_propagate(False)
            ttk.Label(holder, text=pal_key, style="Field.TLabel").grid(
                row=row, column=1, sticky="w"
            )
            ttk.Label(holder, text=value, style="CardMuted.TLabel").grid(
                row=row, column=2, sticky="w", padx=(10, 0)
            )
        holder.columnconfigure(2, weight=1)

    def _refresh_metrics_dialog(self) -> None:
        """Theme flips re-tint an open metrics dialog's palette block."""
        win = self._metrics_dialog
        if win is None:
            return
        try:
            alive = bool(win.winfo_exists())
        except tk.TclError:  # pragma: no cover - destroyed mid-flip
            alive = False
        if not alive:
            self._metrics_dialog = None
            self._metrics_palette = None
            self._metrics_text = None
            self._metrics_palette_title = None
            return
        self._populate_metrics_palette()
        with contextlib.suppress(tk.TclError):
            self._metrics_palette_title.configure(
                text=f"Palette \\u2014 {_ACTIVE_THEME} "
                f"({len(_PALETTES[_ACTIVE_THEME])} keys)"
            )

    def _collect_session(self) -> dict[str, Any]:
'''
patch(
    "    def _collect_session(self) -> dict[str, Any]:\n",
    METRICS,
)

# ------------------------------------------------------------------ #
# 12. the fifteen retrofitted cards                                    #
# ------------------------------------------------------------------ #
COLLAPSIBLE: list[tuple[str, str]] = [
    ("Dependency status", "home-dependency-status"),
    ("What lives where", "home-what-lives-where"),
    ("Half-life \u2194 elimination", "pk-half-life"),
    ("IV bolus concentration profile", "pk-iv-bolus"),
    ("Oral absorption PK", "pk-oral"),
    ("Cockcroft\u2013Gault creatinine clearance", "pk-crcl"),
    ("Loading & maintenance dosing", "pk-loading"),
    ("BSA & weight-based dosing", "pk-bsa"),
    ("Pediatric dose rules", "pk-pediatric"),
    ("Emax pharmacodynamics", "pk-emax"),
    ("Non-compartmental analysis (NCA)", "pk-nca"),
    ("IV infusion & distribution", "pk-infusion"),
    ("TDM trough adjustment", "pk-tdm"),
    ("PD model explorer", "pk-pd-explorer"),
    ("Unit converter", "pk-units"),
]

lines = src.splitlines(keepends=True)
out: list[str] = []
inside = 0  # lines since a make_card( opener
patched: set[str] = set()
for line in lines:
    if "make_card(" in line and not line.strip().startswith("#"):
        if line.rstrip().endswith(")"):
            # single-line call: rewrite if it carries a target title
            hit = None
            for title, key in COLLAPSIBLE:
                if f'"{title}"' in line:
                    hit = (title, key)
                    break
            if hit is not None:
                title, key = hit
                assert title not in patched, f"twice: {title}"
                patched.add(title)
                line = line.rstrip()[:-1].rstrip() + (
                    f', collapsible=True, key="{key}")\n'
                )
            out.append(line)
            inside = 0
            continue
        inside = 1
        out.append(line)
        continue
    if inside:
        stripped = line.strip()
        hit = None
        for title, key in COLLAPSIBLE:
            if stripped == f'"{title}",':
                hit = (title, key)
                break
        if hit is not None:
            title, key = hit
            assert title not in patched, f"twice: {title}"
            patched.add(title)
            indent = line[: len(line) - len(line.lstrip())]
            out.append(line)
            out.append(f"{indent}collapsible=True,\n")
            out.append(f'{indent}key="{key}",\n')
            inside = 0
            continue
        if stripped.startswith(")"):
            inside = 0
        else:
            inside += 1
            if inside > 8:  # a call is never this deep without its title
                inside = 0
    out.append(line)

missing = [title for title, _key in COLLAPSIBLE if title not in patched]
assert not missing, f"cards not patched: {missing}"
src = "".join(out)
# 13 pk- keys + 2 home- keys (the docstring's ``collapsible=True`` mention
# is deliberately excluded from this count)
assert src.count('key="pk-') == 13 and src.count('key="home-') == 2

# ------------------------------------------------------------------ #
# write + verify                                                       #
# ------------------------------------------------------------------ #
ast.parse(src)
TARGET.write_text(src, encoding="utf-8")
print(f"patched playground.py: {len(COLLAPSIBLE)} cards collapsible, "
      f"{len(src.splitlines())} lines, ast OK")
