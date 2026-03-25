from __future__ import annotations

import argparse
import json
import os
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
    from PyQt6.QtCore import QPointF, QRect, Qt, QTimer
    from PyQt6.QtGui import QColor, QPainter, QPen
    from PyQt6.QtWidgets import QApplication, QWidget
    QT_API = "PyQt6"
except ImportError:
    from PyQt5.QtCore import QPointF, QRect, Qt, QTimer
    from PyQt5.QtGui import QColor, QPainter, QPen
    from PyQt5.QtWidgets import QApplication, QWidget
    QT_API = "PyQt5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IrisKeys transparent OS overlay")
    parser.add_argument("--state-file", required=True)
    return parser.parse_args()


class GazeOverlay(QWidget):
    def __init__(self, state_file: Path) -> None:
        super().__init__()
        self.state_file = Path(state_file)
        self.state_mtime_ns = -1
        self.cursor_x = 0
        self.cursor_y = 0
        self.active = False
        self.magnetized = False
        self.desktop_rect = _get_windows_desktop_rect()
        self._init_window()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_state)
        self.timer.start(16)

    def _init_window(self) -> None:
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        if QT_API == "PyQt5":
            flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._apply_windows_click_through()

    def _apply_windows_click_through(self) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            current = int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, current | WS_EX_LAYERED | WS_EX_TRANSPARENT)
        except Exception:
            pass

    def _set_geometry_from_state(self) -> None:
        vx, vy, vw, vh = self.desktop_rect
        self.setGeometry(QRect(int(vx), int(vy), int(max(1, vw)), int(max(1, vh))))

    def refresh_state(self) -> None:
        try:
            stat = self.state_file.stat()
        except FileNotFoundError:
            self.active = False
            self.update()
            return
        if stat.st_mtime_ns == self.state_mtime_ns:
            return
        self.state_mtime_ns = stat.st_mtime_ns
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return

        self.active = bool(payload.get("active", False))
        self.magnetized = bool(payload.get("magnetized", False))
        self.cursor_x = int(payload.get("x", self.cursor_x))
        self.cursor_y = int(payload.get("y", self.cursor_y))
        rect = payload.get("desktop_rect")
        if isinstance(rect, list) and len(rect) == 4:
            self.desktop_rect = tuple(int(v) for v in rect)
            self._set_geometry_from_state()
        self.update()

    def paintEvent(self, _event) -> None:  # pragma: no cover - GUI rendering
        if not self.active:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        vx, vy, _, _ = self.desktop_rect
        local_x = float(self.cursor_x - vx)
        local_y = float(self.cursor_y - vy)
        center = QPointF(local_x, local_y)

        if self.magnetized:
            outer = QColor(60, 220, 120, 220)
            inner = QColor(180, 255, 200, 240)
        else:
            outer = QColor(70, 165, 255, 220)
            inner = QColor(215, 240, 255, 245)

        pen_outer = QPen(outer, 3.0)
        pen_inner = QPen(inner, 1.5)

        painter.setPen(pen_outer)
        painter.drawEllipse(center, 18.0, 18.0)
        painter.drawLine(int(local_x - 26), int(local_y), int(local_x + 26), int(local_y))
        painter.drawLine(int(local_x), int(local_y - 26), int(local_x), int(local_y + 26))

        painter.setPen(pen_inner)
        painter.drawEllipse(center, 7.0, 7.0)
        painter.drawLine(int(local_x - 10), int(local_y), int(local_x + 10), int(local_y))
        painter.drawLine(int(local_x), int(local_y - 10), int(local_x), int(local_y + 10))


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("IrisKeysOverlay")
    overlay = GazeOverlay(Path(args.state_file))
    overlay._set_geometry_from_state()
    overlay.show()
    if QT_API == "PyQt6":
        sys.exit(app.exec())
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
