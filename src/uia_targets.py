from __future__ import annotations

import ctypes
from ctypes import wintypes

WindowInfo = dict[str, int | str | None]


def _empty_info() -> WindowInfo:
    return {"hwnd": None, "title": None, "pid": None}


def _get_user32():
    try:
        user32 = ctypes.windll.user32
    except Exception:
        return None

    try:
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int
    except Exception:
        return None
    return user32


def get_foreground_window_info() -> WindowInfo:
    default = _empty_info()
    try:
        user32 = _get_user32()
        if user32 is None:
            return default

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return default
        hwnd_value = int(hwnd)

        pid_value: int | None = None
        try:
            pid = wintypes.DWORD(0)
            user32.GetWindowThreadProcessId(wintypes.HWND(hwnd_value), ctypes.byref(pid))
            pid_value = int(pid.value)
        except Exception:
            pid_value = None

        title_value: str | None = None
        try:
            title_len = int(user32.GetWindowTextLengthW(wintypes.HWND(hwnd_value)))
            if title_len < 0:
                title_len = 0
            buf = ctypes.create_unicode_buffer(title_len + 1)
            user32.GetWindowTextW(wintypes.HWND(hwnd_value), buf, title_len + 1)
            title_value = str(buf.value)
        except Exception:
            title_value = None

        return {"hwnd": hwnd_value, "title": title_value, "pid": pid_value}
    except Exception:
        return default


class TargetRegistry:
    def __init__(self) -> None:
        self.last_hwnd: int | None = None

    def refresh(self) -> WindowInfo:
        info = get_foreground_window_info()
        hwnd_value = info.get("hwnd")
        self.last_hwnd = int(hwnd_value) if isinstance(hwnd_value, int) else None
        return info


__all__ = ["TargetRegistry", "get_foreground_window_info"]
