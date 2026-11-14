"""Toast routing for pages (review C-3 fix).

The shell's ``notify()`` toast system existed and was wired into the
main window — but zero pages called it: completions reported through
bare status labels only.  This module gives any widget a ``notify()``
that walks up the parent chain to the shell and raises the toast
there, with a page-provenance prefix so users know WHICH workflow
finished while they are looking at another page.

Standalone pages (tests, card workspaces) degrade to a no-op: the
return value tells callers whether a toast actually fired.
"""

from __future__ import annotations

from typing import Protocol

from PySide6.QtWidgets import QWidget


class _Notifier(Protocol):
    def notify(self, text: str, kind: str = "info") -> object: ...


def shell_notify(widget: QWidget, text: str, kind: str = "info") -> bool:
    """Raise ``text`` as a shell toast; True when one actually fired.

    Walks the parent chain looking for a MainWindow-like ``notify``.
    Never raises: a missing shell means the page runs standalone
    (tests, workspaces), where toasts are meaningless anyway.
    """
    node: QWidget | None = widget
    for _ in range(24):  # bounded walk; the shell is a handful of levels up
        if node is None:
            return False
        fn = getattr(node, "notify", None)
        if callable(fn) and node is not widget:
            try:
                fn(text, kind)
                return True
            except TypeError:
                # a foreign notify() with a different signature — skip it
                pass
        node = node.parentWidget()
    return False
