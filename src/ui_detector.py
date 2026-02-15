from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]


@dataclass(frozen=True)
class MagnetTarget:
    id: str
    center_px: Point
    radius_px: int
    kind: str
    weight: float = 1.0
    window_rect: Optional[Rect] = None


def get_foreground_window_handle() -> Optional[int]:
    return None


def get_window_rect(_hwnd: int) -> Optional[Rect]:
    return None


def get_window_pid(_hwnd: int) -> Optional[int]:
    return None


def build_standard_targets(window_rect: Rect, include_back: bool = True) -> List[MagnetTarget]:
    left, top, right, bottom = (int(window_rect[0]), int(window_rect[1]), int(window_rect[2]), int(window_rect[3]))
    if right <= left or bottom <= top:
        return []

    width = right - left
    height = bottom - top
    radius = max(12, int(0.018 * min(width, height)))
    title_y = top + max(radius + 4, int(0.04 * height))
    gap = max(10, radius + 8)
    x_close = right - max(18, radius + 8)

    targets = [
        MagnetTarget(
            id="close",
            center_px=(x_close, title_y),
            radius_px=radius,
            kind="window_control",
            weight=1.2,
            window_rect=(left, top, right, bottom),
        ),
        MagnetTarget(
            id="maximize",
            center_px=(x_close - gap, title_y),
            radius_px=radius,
            kind="window_control",
            weight=1.0,
            window_rect=(left, top, right, bottom),
        ),
        MagnetTarget(
            id="minimize",
            center_px=(x_close - 2 * gap, title_y),
            radius_px=radius,
            kind="window_control",
            weight=1.0,
            window_rect=(left, top, right, bottom),
        ),
    ]
    if include_back:
        targets.append(
            MagnetTarget(
                id="back",
                center_px=(left + max(22, radius + 8), title_y),
                radius_px=radius,
                kind="window_nav",
                weight=0.9,
                window_rect=(left, top, right, bottom),
            )
        )
    return targets


class WindowTargetProvider:
    def __init__(self, poll_interval_ms: int = 250, stale_after_ms: int = 500, include_back: bool = True) -> None:
        self.poll_interval_ms = int(max(1, poll_interval_ms))
        self.stale_after_ms = int(max(1, stale_after_ms))
        self.include_back = bool(include_back)

        self._running = False
        self._last_poll_ms: Optional[int] = None
        self._cached_at_ms: Optional[int] = None
        self._cached_targets: List[MagnetTarget] = []
        self._cached_reason = "no_cache"

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
        self._cached_targets = []
        self._cached_reason = "stopped"
        self._cached_at_ms = None

    def _set_cache(self, targets: List[MagnetTarget], reason: str, now_ms: int) -> None:
        self._cached_targets = list(targets)
        self._cached_reason = str(reason)
        self._cached_at_ms = int(now_ms)

    def _poll_once(self, now_ms: int) -> None:
        now_i = int(now_ms)
        if not self._running:
            return
        if self._last_poll_ms is not None and now_i - self._last_poll_ms < self.poll_interval_ms:
            return
        self._last_poll_ms = now_i

        hwnd = get_foreground_window_handle()
        if hwnd is None:
            self._set_cache([], "no_window", now_i)
            return

        window_rect = get_window_rect(hwnd)
        if window_rect is None:
            self._set_cache([], "no_rect", now_i)
            return

        pid = get_window_pid(hwnd)
        if pid == os.getpid():
            self._set_cache([], "self_window", now_i)
            return

        targets = build_standard_targets(window_rect, include_back=self.include_back)
        if not targets:
            self._set_cache([], "no_targets", now_i)
            return

        self._set_cache(targets, "ok", now_i)

    def get_targets(self, now_ms: int) -> Tuple[List[MagnetTarget], str]:
        now_i = int(now_ms)
        if self._cached_at_ms is None:
            return [], "no_cache"
        if now_i - self._cached_at_ms > self.stale_after_ms:
            return [], "stale_cache"
        if self._cached_reason != "ok":
            return [], self._cached_reason
        return list(self._cached_targets), "ok"


__all__ = [
    "MagnetTarget",
    "WindowTargetProvider",
    "build_standard_targets",
    "get_foreground_window_handle",
    "get_window_pid",
    "get_window_rect",
]
