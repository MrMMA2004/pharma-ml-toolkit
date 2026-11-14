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

        # -- execution preset + resource readout (§17 E, Phase 5) ------------
        perf_title = QLabel("Execution & resources")
        perf_title.setObjectName("studioPageTitle")
        lay.addWidget(perf_title)
        self._preset_combo = QComboBox()
        from pharmaml.utils.resources import (
            PRESETS, preset_for_low_memory, snapshot, stored_preset,
        )

        for preset in PRESETS:
            self._preset_combo.addItem(f"{preset.label} — {preset.trades}",
                                       preset.key)
        # the recommendation is only as confident as the memory evidence
        recommended = preset_for_low_memory()
        self._preset_combo.setCurrentIndex(
            next((i for i in range(self._preset_combo.count())
                  if self._preset_combo.itemData(i) == recommended), 0)
        )
        if recommended == "eco":
            hint = (" (recommended: this machine's available memory was "
                    "evidenced under 2 GiB)")
        else:
            hint = ""
        self._preset_combo.setToolTip(
            "The resource policy for background jobs: worker count and "
            "the streaming reader's chunk size" + hint
        )
        self._preset_combo.currentIndexChanged.connect(self._preset_changed)
        preset_form = QFormLayout()
        preset_form.addRow("Execution preset:", self._preset_combo)
        lay.addLayout(preset_form)

        snap = snapshot()
        readout = QLabel(
            f"Live probe: {snap.cpu_count} CPUs · "
            f"available memory {snap.memory_gib()} "
            f"(source: {snap.memory_source}). GPUs are not probed — "
            "no GPU claim is made without evidence."
        )
        readout.setObjectName("studioMigrationNote")
        readout.setWordWrap(True)
        lay.addWidget(readout)

        # -- dependency transparency (§12, Phase 5): the doctor in the GUI --
        dep_title = QLabel("Dependency diagnostics")
        dep_title.setObjectName("studioPageTitle")
        lay.addWidget(dep_title)
        from ..doctor import collect_checks

        checks = collect_checks()
        ok = sum(1 for c in checks if c.status == "OK")
        dep_summary = QLabel(
            f"Environment doctor: {ok}/{len(checks)} checks passed "
            "(the same checks as `python studio.py --doctor`)."
        )
        dep_summary.setObjectName("studioMigrationNote")
        dep_summary.setWordWrap(True)
        lay.addWidget(dep_summary)
        self._doctor_detail = QLabel()
        self._doctor_detail.setWordWrap(True)
        from PySide6.QtCore import Qt

        self._doctor_detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        failed_lines = [
            f"✗ {c.name}: {c.fix or c.detail}" for c in checks if c.status != "OK"
        ]
        if failed_lines:
            self._doctor_detail.setText("\n".join(failed_lines))
        else:
            self._doctor_detail.setText("All launch-environment checks pass.")
        lay.addWidget(self._doctor_detail)

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
            "The studio provider carries the Design System identity: Mineral light "
            "and dark (the canonical direction) plus the two alternative directions "
            "as switchable themes — Spectral (warm paper, indigo actions, rounded "
            "panels) and Catalyst (navy frame, ivory surfaces, lime selection, "
            "squared cards), each in light and dark. Qt-Material is the primary "
            "Material Design provider; QDarkStyleSheet is the verified alternative "
            "(single dark theme, Qt6-styled widgets); native is the unstyled "
            "fallback. Switching providers replaces the active stylesheet "
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

    def _preset_changed(self, index: int) -> None:
        """Persist the §17 E execution preset (session echo + QSettings)."""
        from pharmaml.utils.resources import find_preset, store_preset

        key = self._preset_combo.itemData(index) if index >= 0 else "balanced"
        preset = find_preset(str(key))
        store_preset(preset.key)
        try:  # the durable store; failures never block the switch
            from PySide6.QtCore import QSettings

            QSettings().setValue("execution/preset", preset.key)
        except Exception:  # noqa: BLE001 - settings must never block launch
            pass

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
