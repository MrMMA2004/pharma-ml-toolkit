"""Toast-style notifications for PharmaML Studio (directive 16.7/16.12).

Transient feedback messages (success / info / warning / danger) shown
in a corner overlay; they auto-dismiss after ``AUTO_MS`` and never
block interaction (unlike modal message boxes, which are reserved for
destructive confirmations and fatal errors).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLabel, QWidget

from ..design_tokens import tokens

#: kind -> semantic token name pair (status foreground, soft background).
#: The Mineral status tokens (§4.1) — one source for QSS, toasts and
#: tests; no Material-era hex pairs inline (the migration's follow-up).
_KIND_TOKENS = {
    "success": ("success", "success_bg"),
    "info": ("info", "info_bg"),
    "warning": ("warning", "warning_bg"),
    "danger": ("error", "error_bg"),
}
AUTO_MS = 4_000


def kind_style(kind: str, theme: str = "light") -> tuple[str, str]:
    """The (background, foreground) pair for a toast kind from tokens."""
    t = tokens(theme)
    fg_name, bg_name = _KIND_TOKENS.get(kind, _KIND_TOKENS["info"])
    return getattr(t, bg_name), getattr(t, fg_name)


class NotificationBar(QLabel):
    """One toast row (private -- use :class:`Notifications`)."""

    dismissed = Signal(object)

    def __init__(self, text: str, kind: str = "info", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        bg, fg = kind_style(kind)
        # Phase 5 accessibility: screen readers announce the toast as a
        # notification of its severity, not as an anonymous label.
        self.setAccessibleName(f"{kind.title()} notification: {text}")
        self.setStyleSheet(
            f"QLabel {{ background: {bg}; color: {fg}; border-radius: 6px;"
            " padding: 8px 12px; font-weight: 600; }"
        )
        self.setWordWrap(True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(0.97)
        self.setGraphicsEffect(effect)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._dismiss)
        self._timer.start(AUTO_MS)

    def mousePressEvent(self, event: object) -> None:
        self._dismiss()

    def _dismiss(self) -> None:
        self.dismissed.emit(self)
        self.close()


class Notifications(QWidget):
    """A vertical stack of toasts anchored to a parent widget's corner."""

    def __init__(self, anchor: QWidget) -> None:
        super().__init__(anchor)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self._anchor = anchor
        self._toasts: list[NotificationBar] = []

    def notify(self, text: str, kind: str = "info") -> NotificationBar:
        toast = NotificationBar(text, kind, self)
        toast.dismissed.connect(self._forget)
        self._toasts.append(toast)
        self._relayout()
        return toast

    # -- internals ---------------------------------------------------------------
    def _forget(self, toast: NotificationBar) -> None:
        if toast in self._toasts:
            self._toasts.remove(toast)
        self._relayout()

    def _relayout(self) -> None:
        """Position the stack bottom-right of the anchor."""
        if self._anchor is None:
            return
        width = min(420, max(240, self._anchor.width() // 3))
        height = sum(t.sizeHint().height() + 6 for t in self._toasts)
        self.setGeometry(self._anchor.width() - width - 16, self._anchor.height() - height - 16, width, max(height, 1))
        y = 0
        for toast in self._toasts:
            toast.setGeometry(0, y, width, toast.sizeHint().height())
            toast.show()
            toast.raise_()
            y += toast.sizeHint().height() + 6
