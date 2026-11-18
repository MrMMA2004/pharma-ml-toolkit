"""Validated manual ``n_jobs`` entry (part8 Phase 6).

The audit round found parallelism locked to a fixed ``1/2/4`` combo in the
Chemistry Lab and nowhere else.  This module gives every featurization /
modelling surface the same editable field with joblib semantics:

- any nonzero integer is valid (``-1`` = all cores, the default offered),
- ``0`` is rejected (joblib treats it as an error, not as serial),
- blanks / non-integers are flagged without raising.

The widget is deliberately dependency-free (PySide6 only) so pages can
import it without touching the chemistry backend.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QWidget

#: Placeholder text shown while the field is empty.
_HINT = "integer, -1 = all cores (0 invalid)"


class NJobsField(QComboBox):
    """Editable combo accepting any backend-valid ``n_jobs`` integer.

    The dropdown offers the common presets (``-1, 1, 2, 4, 8``) for quick
    picking, but the user may TYPE any nonzero integer — the restrictive
    fixed-value selection is gone.  ``text()`` always returns the current
    raw text; :meth:`value` validates it and answers ``None`` when it is
    not a backend-valid count, so callers branch on ``None`` to explain
    instead of raising.
    """

    #: Emitted after every edit settles (text changed or preset picked).
    valueChanged = Signal(str)

    def __init__(self, default: int = -1, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        for preset in (-1, 1, 2, 4, 8):
            self.addItem(str(preset))
        self.setCurrentText(str(default))
        self.setToolTip(
            "Worker count for parallel backends (joblib convention): "
            "-1 = all cores, 1 = serial. 0 is invalid."
        )
        self.lineEdit().setPlaceholderText(_HINT)
        self.currentTextChanged.connect(self._on_text)

    # -- API -----------------------------------------------------------------
    def _on_text(self, text: str) -> None:
        self.valueChanged.emit(text)

    def value(self) -> int | None:
        """The validated ``n_jobs`` count, or ``None`` when invalid.

        Valid means: a bare integer (sign allowed, no separators) that is
        not zero.  Booleans never reach here (a QLineEdit cannot produce
        one) and whitespace is trimmed before parsing.
        """
        stripped = self.currentText().strip()
        try:
            parsed = int(stripped)
        except ValueError:
            return None
        if parsed == 0:
            return None
        return parsed

    def is_valid(self) -> bool:
        return self.value() is not None
