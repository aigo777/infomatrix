from __future__ import annotations

import ctypes
import time
from collections import deque
from ctypes import wintypes
from typing import Any, TypedDict

WindowInfo = dict[str, int | str | None]
Rect = tuple[int, int, int, int]
Point = tuple[int, int]


class Target(TypedDict):
    kind: str
    name: str
    rect: Rect
    center: Point
    hwnd: int


ALLOW_KINDS = {
    "Button",
    "Hyperlink",
    "Edit",
    "CheckBox",
    "RadioButton",
    "ComboBox",
    "TabItem",
    "MenuItem",
}

_KIND_CANONICAL: dict[str, str] = {
    "button": "Button",
    "hyperlink": "Hyperlink",
    "edit": "Edit",
    "checkbox": "CheckBox",
    "radiobutton": "RadioButton",
    "combobox": "ComboBox",
    "tabitem": "TabItem",
    "menuitem": "MenuItem",
}


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
    def __init__(
        self,
        max_targets: int = 200,
        visited_limit: int = 2000,
        min_scan_interval_s: float = 0.25,
    ) -> None:
        self.max_targets = max(1, int(max_targets))
        self.visited_limit = max(1, int(visited_limit))
        self.min_scan_interval_s = float(max(0.0, min_scan_interval_s))

        self.last_hwnd: int | None = None
        self.last_window_rect: Rect | None = None
        self.targets: list[Target] = []
        self._last_scan_time = 0.0
        self._last_refresh_info: WindowInfo = _empty_info()

    def refresh(self) -> WindowInfo:
        now = time.monotonic()
        if now - self._last_scan_time < self.min_scan_interval_s:
            return dict(self._last_refresh_info)

        info = get_foreground_window_info()
        hwnd_value = info.get("hwnd")
        hwnd = int(hwnd_value) if isinstance(hwnd_value, int) else None
        self.last_hwnd = hwnd

        self.targets = []
        self.last_window_rect = None
        if hwnd is None:
            self._last_refresh_info = dict(info)
            self._last_scan_time = now
            return dict(info)

        attach_result = self._attach_uia_internal(hwnd)
        if not bool(attach_result.get("ok")):
            self._last_refresh_info = dict(info)
            self._last_scan_time = now
            return dict(info)

        rect_obj = attach_result.get("rect")
        window = attach_result.get("window")
        if (
            isinstance(rect_obj, tuple)
            and len(rect_obj) == 4
            and window is not None
            and all(isinstance(v, int) for v in rect_obj)
        ):
            win_rect: Rect = (int(rect_obj[0]), int(rect_obj[1]), int(rect_obj[2]), int(rect_obj[3]))
            self.last_window_rect = win_rect
            self.targets = self._scan_targets(window, hwnd, win_rect)

        self._last_refresh_info = dict(info)
        self._last_scan_time = now
        return dict(info)

    def get_targets(self) -> list[Target]:
        return list(self.targets)

    def attach_uia(self, hwnd: int) -> dict:
        result = self._attach_uia_internal(hwnd)
        if not bool(result.get("ok")):
            return {"ok": False, "error": result.get("error")}
        return {
            "ok": True,
            "uia_title": result.get("uia_title"),
            "rect": result.get("rect"),
        }

    def _attach_uia_internal(self, hwnd: int) -> dict[str, Any]:
        if not hwnd:
            return {"ok": False, "error": "no_hwnd"}

        try:
            from pywinauto import Desktop
        except ImportError:
            return {"ok": False, "error": "pywinauto_not_installed"}
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}

        try:
            window = Desktop(backend="uia").window(handle=int(hwnd))
            try:
                window.wait("exists", timeout=0.5)
            except Exception:
                pass

            uia_title = str(window.window_text() or "")
            rect_tuple = _coerce_rect(window.rectangle())
            if rect_tuple is None:
                return {"ok": False, "error": "invalid_window_rect"}
            return {"ok": True, "window": window, "uia_title": uia_title, "rect": rect_tuple}
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}

    def _scan_targets(self, window: Any, hwnd: int, win_rect: Rect) -> list[Target]:
        try:
            root = window.wrapper_object() if hasattr(window, "wrapper_object") else window
        except Exception:
            return []

        try:
            initial_children = root.children()
        except Exception:
            return []

        queue: deque[Any] = deque(initial_children)

        out: list[Target] = []
        visited = 0
        while queue:
            elem = queue.popleft()
            visited += 1
            if visited > self.visited_limit:
                break
            if len(out) >= self.max_targets:
                break

            target = self._element_to_target(elem, hwnd=hwnd, win_rect=win_rect)
            if target is not None:
                out.append(target)

            if visited >= self.visited_limit or len(out) >= self.max_targets:
                continue

            try:
                children = elem.children()
            except Exception:
                children = []
            if children:
                queue.extend(children)
        return out

    def _element_to_target(self, elem: Any, hwnd: int, win_rect: Rect) -> Target | None:
        kind = _element_kind(elem)
        if kind not in ALLOW_KINDS:
            return None

        enabled = _is_enabled(elem)
        if enabled is not True:
            return None

        if _is_offscreen(elem):
            return None

        rect = _element_rect(elem)
        if rect is None:
            return None
        if not _rect_is_within_window(rect, win_rect, tol=2):
            return None

        name = _element_name(elem)
        left, top, right, bottom = rect
        center = ((left + right) // 2, (top + bottom) // 2)
        return {
            "kind": kind,
            "name": name,
            "rect": rect,
            "center": center,
            "hwnd": int(hwnd),
        }


def _coerce_rect(rect: Any) -> Rect | None:
    try:
        left = int(rect.left)
        top = int(rect.top)
        right = int(rect.right)
        bottom = int(rect.bottom)
    except Exception:
        return None
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _normalize_kind(value: str) -> str | None:
    key = "".join(ch for ch in value if ch.isalnum()).lower()
    return _KIND_CANONICAL.get(key)


def _element_kind(elem: Any) -> str | None:
    raw_kind = ""
    try:
        raw_kind = str(elem.friendly_class_name() or "")
    except Exception:
        raw_kind = ""

    kind = _normalize_kind(raw_kind)
    if kind is not None:
        return kind

    try:
        raw_kind = str(getattr(elem.element_info, "control_type", "") or "")
    except Exception:
        raw_kind = ""
    return _normalize_kind(raw_kind)


def _is_enabled(elem: Any) -> bool | None:
    enabled_fn = getattr(elem, "is_enabled", None)
    if callable(enabled_fn):
        try:
            return bool(enabled_fn())
        except Exception:
            return None
    try:
        enabled_attr = getattr(elem.element_info, "enabled", None)
    except Exception:
        enabled_attr = None
    if enabled_attr is None:
        return None
    try:
        return bool(enabled_attr)
    except Exception:
        return None


def _is_offscreen(elem: Any) -> bool:
    offscreen_fn = getattr(elem, "is_offscreen", None)
    if callable(offscreen_fn):
        try:
            return bool(offscreen_fn())
        except Exception:
            # If offscreen check fails, skip this filter safely.
            return False
    return False


def _element_rect(elem: Any) -> Rect | None:
    try:
        rect = _coerce_rect(elem.rectangle())
    except Exception:
        return None
    if rect is None:
        return None

    left, top, right, bottom = rect
    width = right - left
    height = bottom - top
    if width <= 4 or height <= 4:
        return None
    return rect


def _rect_is_within_window(rect: Rect, win_rect: Rect, tol: int = 2) -> bool:
    left, top, right, bottom = rect
    win_l, win_t, win_r, win_b = win_rect
    if left < win_l - tol:
        return False
    if top < win_t - tol:
        return False
    if right > win_r + tol:
        return False
    if bottom > win_b + tol:
        return False
    return True


def _element_name(elem: Any) -> str:
    text = ""
    try:
        text = str(elem.window_text() or "").strip()
    except Exception:
        text = ""
    if text:
        return text

    try:
        text = str(getattr(elem.element_info, "name", "") or "").strip()
    except Exception:
        text = ""
    return text if text else "<unnamed>"


__all__ = ["Target", "TargetRegistry", "get_foreground_window_info"]
