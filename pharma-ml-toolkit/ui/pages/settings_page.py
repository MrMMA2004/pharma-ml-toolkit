"""Appearance & behaviour settings (directive 16.9.3/16.15).

Theme provider selection (Qt-Material primary / QDarkStyleSheet
alternative / native fallback), theme and density switching at runtime
without restarting, persisted through QSettings. Reset clears every
stored UI preference.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import PROVIDERS, ThemeManager


class SettingsPage(QWidget):
    """The appearance page: live theme switching, honest notes."""

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(12)

        title = QLabel("Appearance")
        title.setObjectName("studioPageTitle")
        lay.addWidget(title)

        form = QFormLayout()
        self._provider_combo = QComboBox()
        for provider in PROVIDERS:
            self._provider_combo.addItem(provider)
        self._provider_combo.setCurrentText(theme.provider)
        self._provider_combo.currentTextChanged.connect(self._provider_changed)
        form.addRow("Style provider:", self._provider_combo)

        self._theme_combo = QComboBox()
        self._repopulate_themes()
        self._theme_combo.setCurrentText(theme.theme)
        self._theme_combo.currentTextChanged.connect(self._apply)
        form.addRow("Theme:", self._theme_combo)

        self._density_combo = QComboBox()
        for label, value in (("comfortable (−1)", -1), ("standard (0)", 0), ("compact (+1)", 1), ("dense (+2)", 2)):
            self._density_combo.addItem(label, value)
        self._density_combo.setCurrentIndex(1)
        self._density_combo.currentIndexChanged.connect(self._apply)
        form.addRow("Density:", self._density_combo)
        lay.addLayout(form)

        buttons = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._apply)
        reset = QPushButton("Reset all UI preferences…")
        reset.clicked.connect(self._reset)
        buttons.addWidget(self._apply_btn)
        buttons.addWidget(reset)
        buttons.addStretch(1)
        lay.addLayout(buttons)

        note = QLabel(
            "Qt-Material is the primary Material Design provider; QDarkStyleSheet is the "
            "verified alternative (single dark theme, Qt6-styled widgets); native is the "
            "unstyled fallback. Switching providers replaces the active stylesheet "
            "completely — the two engines never compete — and every PyQtGraph chart "
            "re-colours itself to match the new theme. Preferences persist via QSettings."
        )
        note.setObjectName("studioMigrationNote")
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addStretch(1)

    def _repopulate_themes(self) -> None:
        provider = self._provider_combo.currentText()
        self._theme_combo.blockSignals(True)
        self._theme_combo.clear()
        self._theme_combo.addItems(list(self._theme.available_themes(provider)))
        self._theme_combo.blockSignals(False)

    def _provider_changed(self, _provider: str) -> None:
        self._repopulate_themes()
        self._apply()

    def _apply(self) -> None:
        density = self._density_combo.currentData()
        if density is None:
            density = 0
        try:
            self._theme.apply(
                self._provider_combo.currentText(),
                self._theme_combo.currentText() or None,
                density_scale=int(density),
            )
            self._theme.save_state()
        except ValueError as exc:  # pragma: no cover - guarded combo values
            QMessageBox.warning(self, "Theme", str(exc))

    def _reset(self) -> None:
        answer = QMessageBox.question(
            self,
            "Reset UI preferences",
            "Clear every stored UI preference (theme, window geometry, navigation)?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            from ..app_settings import AppSettings

            AppSettings().clear()
            self._provider_combo.setCurrentText("qt-material")
            self._repopulate_themes()
            self._theme_combo.setCurrentText("dark_teal.xml")
            self._apply()
