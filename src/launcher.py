from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _enable_windows_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        if hasattr(user32, "SetProcessDPIAware"):
            user32.SetProcessDPIAware()
    except Exception:
        pass


_enable_windows_dpi_awareness()

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    QT_API = "PyQt6"
except ImportError:
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QFont
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QMessageBox,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
        QT_API = "PyQt5"
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit("PyQt6 or PyQt5 is required to run launcher.py") from exc


APP_TITLE = "EyeAssist OS"
ROOT_DIR = Path(__file__).resolve().parent.parent
MAIN_SCRIPT = ROOT_DIR / "src" / "main.py"


class LaunchCard(QFrame):
    def __init__(self, accent: str, title: str, subtitle: str, facts: list[str], button_text: str, callback) -> None:
        super().__init__()
        self.setObjectName("launchCard")
        self.setProperty("accent", accent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)

        accent_bar = QFrame()
        accent_bar.setObjectName("accentBar")
        accent_bar.setFixedHeight(6)

        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")

        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("cardSubtitle")
        subtitle_label.setWordWrap(True)

        facts_box = QVBoxLayout()
        facts_box.setSpacing(6)
        for fact in facts:
            fact_label = QLabel(fact)
            fact_label.setObjectName("factLabel")
            fact_label.setWordWrap(True)
            facts_box.addWidget(fact_label)

        button = QPushButton(button_text)
        button.setObjectName("primaryButton")
        button.clicked.connect(callback)

        layout.addWidget(accent_bar)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addLayout(facts_box)
        layout.addStretch(1)
        layout.addWidget(button)


class EyeAssistLauncher(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1120, 720)
        self._build_ui()
        self._refresh_launch_hint()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)

        hero = QFrame()
        hero.setObjectName("heroCard")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(28, 26, 28, 26)
        hero_layout.setSpacing(10)

        badge_row = QHBoxLayout()
        badge_row.setSpacing(8)
        for text in ("Finals Build", "Windows Control", "Safe F12 Exit"):
            badge = QLabel(text)
            badge.setObjectName("pill")
            badge_row.addWidget(badge)
        badge_row.addStretch(1)

        title = QLabel(APP_TITLE)
        title.setObjectName("heroTitle")
        title.setFont(QFont("Segoe UI", 26, self._font_bold_weight()))

        subtitle = QLabel(
            "Launch demo or OS control with the correct finals flow: calibration starts first, then the app "
            "transitions automatically into the requested mode without extra clicks or head movement."
        )
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)

        hero_layout.addLayout(badge_row)
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)

        settings_row = QHBoxLayout()
        settings_row.setSpacing(16)

        settings_card = QFrame()
        settings_card.setObjectName("settingsCard")
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(22, 20, 22, 20)
        settings_layout.setSpacing(12)

        settings_title = QLabel("Launch Settings")
        settings_title.setObjectName("sectionTitle")

        self.assist_checkbox = QCheckBox("Smart Magnetism Assist")
        self.assist_checkbox.setChecked(True)
        self.assist_checkbox.toggled.connect(self._refresh_launch_hint)

        self.dwell_checkbox = QCheckBox("Dwell Click")
        self.dwell_checkbox.setChecked(False)
        self.dwell_checkbox.toggled.connect(self._refresh_launch_hint)

        self.launch_hint = QLabel()
        self.launch_hint.setObjectName("launchHint")
        self.launch_hint.setWordWrap(True)

        settings_layout.addWidget(settings_title)
        settings_layout.addWidget(self.assist_checkbox)
        settings_layout.addWidget(self.dwell_checkbox)
        settings_layout.addSpacing(8)
        settings_layout.addWidget(self.launch_hint)
        settings_layout.addStretch(1)

        safety_card = QFrame()
        safety_card.setObjectName("safetyCard")
        safety_layout = QVBoxLayout(safety_card)
        safety_layout.setContentsMargins(22, 20, 22, 20)
        safety_layout.setSpacing(10)

        safety_title = QLabel("Field Conditions Safety")
        safety_title.setObjectName("sectionTitle")

        safety_text = QLabel(
            "OS Mode hides the OpenCV sandbox and drives the real Windows cursor. "
            "Press F12 or ESC at any time to instantly stop control."
        )
        safety_text.setObjectName("warningText")
        safety_text.setWordWrap(True)

        safety_steps = QLabel(
            "Finals flow:\n"
            "1. Choose Demo Mode or OS Mode\n"
            "2. Calibration starts automatically\n"
            "3. Tracking continues immediately in the selected mode"
        )
        safety_steps.setObjectName("guideText")

        safety_layout.addWidget(safety_title)
        safety_layout.addWidget(safety_text)
        safety_layout.addWidget(safety_steps)
        safety_layout.addStretch(1)

        settings_row.addWidget(settings_card, 3)
        settings_row.addWidget(safety_card, 2)

        cards_grid = QGridLayout()
        cards_grid.setHorizontalSpacing(16)
        cards_grid.setVerticalSpacing(16)

        demo_card = LaunchCard(
            "cool",
            "Launch Demo Mode",
            "Starts calibration first, then stays inside the presentation sandbox for rehearsal and finals.",
            [
                "No extra click needed after calibration finishes.",
                "Best mode for rehearsal and stage presentation.",
            ],
            "Open Demo",
            self.launch_demo,
        )
        os_card = LaunchCard(
            "green",
            "Launch Field Conditions (OS Mode)",
            "Starts calibration first, then transitions directly into real Windows cursor control.",
            [
                "Prevents baseline drift from a second launcher click.",
                "Failsafe: F12 or ESC instantly exits control.",
            ],
            "Take Control",
            self.launch_os_mode,
        )

        cards_grid.addWidget(demo_card, 0, 0)
        cards_grid.addWidget(os_card, 0, 1)

        footer = QLabel(
            "Backend: EyeAssist tracking + calibration + smart magnetism + dwell selection"
        )
        footer.setObjectName("footerText")

        root.addWidget(hero)
        root.addLayout(settings_row)
        root.addLayout(cards_grid, 1)
        root.addWidget(footer)

        self.setStyleSheet(
            """
            QWidget {
                background: #0b1016;
                color: #edf4fb;
                font-family: Segoe UI;
                font-size: 14px;
            }
            QFrame#heroCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #142334, stop:0.55 #1b3041, stop:1 #0f1822);
                border: 1px solid #28445b;
                border-radius: 24px;
            }
            QFrame#settingsCard, QFrame#safetyCard, QFrame#launchCard {
                background: #131d28;
                border: 1px solid #233444;
                border-radius: 20px;
            }
            QFrame#accentBar {
                border: none;
                border-radius: 3px;
                background: #3f82ff;
            }
            QFrame#launchCard[accent="warm"] QFrame#accentBar {
                background: #ff9a52;
            }
            QFrame#launchCard[accent="cool"] QFrame#accentBar {
                background: #49b0ff;
            }
            QFrame#launchCard[accent="green"] QFrame#accentBar {
                background: #52d28c;
            }
            QLabel#heroTitle {
                color: #ffffff;
            }
            QLabel#heroSubtitle, QLabel#cardSubtitle, QLabel#footerText {
                color: #b8c8d8;
            }
            QLabel#sectionTitle, QLabel#cardTitle {
                color: #ffffff;
                font-size: 17px;
                font-weight: 600;
            }
            QLabel#factLabel {
                color: #c4d3e1;
                padding-left: 2px;
            }
            QLabel#launchHint {
                color: #9fd0ff;
                background: #0f1822;
                border: 1px solid #203243;
                border-radius: 12px;
                padding: 10px 12px;
            }
            QLabel#warningText {
                color: #ffd79a;
                background: #2a2115;
                border: 1px solid #58401f;
                border-radius: 12px;
                padding: 10px 12px;
            }
            QLabel#guideText {
                color: #d8e4ef;
                background: #101924;
                border: 1px solid #1e3040;
                border-radius: 12px;
                padding: 10px 12px;
            }
            QLabel#pill {
                color: #dceeff;
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(180, 220, 255, 0.18);
                border-radius: 12px;
                padding: 4px 10px;
            }
            QPushButton#primaryButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1e75dd, stop:1 #2c98ff);
                border: none;
                border-radius: 14px;
                padding: 13px 16px;
                color: white;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton#primaryButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2b84ed, stop:1 #40a8ff);
            }
            QPushButton#primaryButton:pressed {
                background: #1e6ece;
            }
            QCheckBox {
                spacing: 10px;
                color: #edf4fb;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #47637d;
                border-radius: 5px;
                background: #0e151d;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #2b8cff;
                border-radius: 5px;
                background: #2b8cff;
            }
            """
        )

    @staticmethod
    def _font_bold_weight() -> int:
        if QT_API == "PyQt6":
            return int(QFont.Weight.Bold)
        return int(QFont.Bold)

    def _refresh_launch_hint(self) -> None:
        assist_text = "ON" if self.assist_checkbox.isChecked() else "OFF"
        dwell_text = "ON" if self.dwell_checkbox.isChecked() else "OFF"
        self.launch_hint.setText(
            f"Current launch profile: Smart Magnetism {assist_text} | Dwell Click {dwell_text}"
        )

    def _build_backend_args(self, mode: str, auto_calibrate: bool = False) -> list[str]:
        args = [sys.executable, str(MAIN_SCRIPT), "--mode", mode]
        args.extend(["--assist", "on" if self.assist_checkbox.isChecked() else "off"])
        args.extend(["--click", "dwell" if self.dwell_checkbox.isChecked() else "off"])
        args.extend(["--os-click", "on" if self.dwell_checkbox.isChecked() else "off"])
        if auto_calibrate:
            args.extend(["--auto-calibrate", "on"])
        return args

    def _launch(self, args: list[str], label: str) -> None:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        src_path = str(ROOT_DIR / "src")
        env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
        try:
            subprocess.Popen(args, cwd=str(ROOT_DIR), env=env)
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Failed to launch {label}.\n\n{exc}")

    def launch_demo(self) -> None:
        self._launch(
            self._build_backend_args("demo", auto_calibrate=True) + ["--post-calibration-mode", "demo"],
            "demo mode",
        )

    def launch_os_mode(self) -> None:
        answer = QMessageBox.question(
            self,
            APP_TITLE,
            "OS Mode will start calibration first and then automatically switch into real Windows control.\n\nUse F12 or ESC as a kill-switch.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            if QT_API == "PyQt6"
            else QMessageBox.Yes | QMessageBox.No,
        )
        yes_value = QMessageBox.StandardButton.Yes if QT_API == "PyQt6" else QMessageBox.Yes
        if answer != yes_value:
            return
        self._launch(
            self._build_backend_args("demo", auto_calibrate=True) + ["--post-calibration-mode", "os"],
            "OS mode",
        )


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyle("Fusion")
    window = EyeAssistLauncher()
    window.show()
    if QT_API == "PyQt6":
        sys.exit(app.exec())
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
