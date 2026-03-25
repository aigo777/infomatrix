from __future__ import annotations

import argparse
import json
import os
import sys
import threading
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


def _get_windows_desktop_rect() -> tuple[int, int, int, int]:
    if os.name != "nt":
        return (0, 0, 1920, 1080)
    try:
        import ctypes

        user32 = ctypes.windll.user32
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        return (0, 0, max(1, width), max(1, height))
    except Exception:
        return (0, 0, 1920, 1080)


_enable_windows_dpi_awareness()

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QApplication,
        QLabel,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    QT_API = "PyQt6"
except ImportError:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import (
        QApplication,
        QLabel,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    QT_API = "PyQt5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EyeAssist floating accessibility toolbar")
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--edge", choices=("left", "right"), default="left")
    return parser.parse_args()


class FloatingToolbar(QWidget):
    voice_finished = pyqtSignal(str)

    def __init__(self, state_file: Path, edge: str = "right") -> None:
        super().__init__()
        self.state_file = Path(state_file)
        self.edge = edge
        self.tracking_paused = False
        self.next_click_button = "left"
        self.voice_finished.connect(self._finish_voice_type)
        self._build_ui()
        self._load_state()
        self._sync_state()
        self._dock_to_edge()

    def _build_ui(self) -> None:
        flags = Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        if QT_API == "PyQt5":
            flags = Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setObjectName("toolbarRoot")
        self.setFixedWidth(132)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        title = QLabel("EyeAssist")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter if QT_API == "PyQt6" else Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 13, self._font_bold_weight()))

        self.keyboard_btn = self._make_button("⌨️\nKeyboard", self.open_keyboard)
        self.voice_btn = self._make_button("🎙️\nVoice Type", self.voice_type)
        self.right_click_btn = self._make_button("🖱️\nRight Click", self.toggle_right_click)
        self.pause_btn = self._make_button("⏸️\nPause", self.toggle_pause)

        self.state_label = QLabel()
        self.state_label.setObjectName("stateLabel")
        self.state_label.setWordWrap(True)

        root.addWidget(title)
        root.addWidget(self.keyboard_btn)
        root.addWidget(self.voice_btn)
        root.addWidget(self.right_click_btn)
        root.addWidget(self.pause_btn)
        root.addStretch(1)
        root.addWidget(self.state_label)

        self.setStyleSheet(
            """
            QWidget#toolbarRoot {
                background: #111a24;
                border: 1px solid #2a3b4a;
                border-radius: 22px;
            }
            QLabel#titleLabel {
                color: #ffffff;
                padding: 8px 0 2px 0;
            }
            QLabel#stateLabel {
                color: #bdd0e1;
                background: #0d141c;
                border: 1px solid #243645;
                border-radius: 14px;
                padding: 10px;
            }
            QPushButton {
                min-width: 100px;
                min-height: 100px;
                border: none;
                border-radius: 18px;
                color: #ffffff;
                background: #183043;
                font-size: 15px;
                font-weight: 600;
                padding: 8px;
            }
            QPushButton:hover {
                background: #20425c;
            }
            QPushButton:pressed {
                background: #152b3b;
            }
            QPushButton[state="armed"] {
                background: #2f7d3e;
            }
            QPushButton[state="paused"] {
                background: #7e5a22;
            }
            """
        )

    @staticmethod
    def _font_bold_weight() -> int:
        if QT_API == "PyQt6":
            return int(QFont.Weight.Bold)
        return int(QFont.Bold)

    def _make_button(self, text: str, callback) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    def _dock_to_edge(self) -> None:
        vx, vy, width_px, height_px = _get_windows_desktop_rect()
        width = 132
        height = min(620, max(480, int(height_px * 0.72)))
        y = vy + max(20, int((height_px - height) * 0.16))
        if self.edge == "left":
            x = vx + 16
        else:
            x = vx + width_px - width - 16
        self.setGeometry(x, y, width, height)

    def _load_state(self) -> None:
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return
        self.tracking_paused = bool(payload.get("paused", False))
        self.next_click_button = "right" if payload.get("next_click_button") == "right" else "left"

    def _sync_state(self) -> None:
        payload = {
            "paused": bool(self.tracking_paused),
            "next_click_button": "right" if self.next_click_button == "right" else "left",
        }
        tmp_path = str(self.state_file) + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp_path, self.state_file)
        except Exception:
            pass
        self._refresh_labels()

    def _refresh_labels(self) -> None:
        pause_text = "Paused" if self.tracking_paused else "Tracking Live"
        click_text = "Right Click armed" if self.next_click_button == "right" else "Left Click default"
        self.state_label.setText(f"{pause_text}\n{click_text}")

        self.right_click_btn.setProperty("state", "armed" if self.next_click_button == "right" else "")
        self.pause_btn.setProperty("state", "paused" if self.tracking_paused else "")
        self.pause_btn.setText("▶️\nResume" if self.tracking_paused else "⏸️\nPause")
        self._polish_button(self.right_click_btn)
        self._polish_button(self.pause_btn)

    @staticmethod
    def _polish_button(button: QPushButton) -> None:
        style = button.style()
        style.unpolish(button)
        style.polish(button)
        button.update()

    def open_keyboard(self) -> None:
        try:
            os.system("osk")
        except Exception as exc:
            self._show_error(f"Failed to open On-Screen Keyboard.\n\n{exc}")

    def voice_type(self) -> None:
        self.voice_btn.setEnabled(False)
        self.voice_btn.setText("🎙️\nListening…")

        def worker() -> None:
            error_message = None
            try:
                import pyautogui
                import speech_recognition as sr

                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
                text = recognizer.recognize_google(audio)
                pyautogui.write(text, interval=0.02)
            except Exception as exc:  # pragma: no cover - depends on runtime devices
                error_message = str(exc)
            self.voice_finished.emit("" if error_message is None else error_message)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_voice_type(self, error_message: str) -> None:
        self.voice_btn.setEnabled(True)
        self.voice_btn.setText("🎙️\nVoice Type")
        if error_message:
            self._show_error(
                "Voice typing failed.\n\nMake sure microphone access, speech_recognition, and pyautogui are available.\n\n"
                + error_message
            )

    def toggle_right_click(self) -> None:
        self.next_click_button = "left" if self.next_click_button == "right" else "right"
        self._sync_state()

    def toggle_pause(self) -> None:
        self.tracking_paused = not self.tracking_paused
        self._sync_state()

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "EyeAssist Toolbar", message)


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("EyeAssistToolbar")
    toolbar = FloatingToolbar(Path(args.state_file), edge=args.edge)
    toolbar.show()
    if QT_API == "PyQt6":
        sys.exit(app.exec())
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
