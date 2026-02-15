from __future__ import annotations

import os
import unittest
from unittest import mock

from ui_detector import MagnetTarget, WindowTargetProvider


class WindowTargetProviderTests(unittest.TestCase):
    def _sample_target(self) -> MagnetTarget:
        return MagnetTarget(
            id="close",
            center_px=(100, 20),
            radius_px=18,
            kind="window_control",
            weight=1.0,
            window_rect=(0, 0, 800, 600),
        )

    def test_self_window_is_excluded(self) -> None:
        provider = WindowTargetProvider(poll_interval_ms=250, stale_after_ms=500, include_back=True)
        provider._running = True
        with mock.patch("ui_detector.get_foreground_window_handle", return_value=11), mock.patch(
            "ui_detector.get_window_rect", return_value=(0, 0, 800, 600)
        ), mock.patch("ui_detector.get_window_pid", return_value=os.getpid()):
            provider._poll_once(now_ms=1000)
            targets, reason = provider.get_targets(now_ms=1000)
        self.assertEqual(reason, "self_window")
        self.assertEqual(targets, [])

    def test_stale_cache_falls_back_to_raw(self) -> None:
        provider = WindowTargetProvider(poll_interval_ms=250, stale_after_ms=500, include_back=True)
        provider._running = True
        fake_target = self._sample_target()
        with mock.patch("ui_detector.get_foreground_window_handle", return_value=12), mock.patch(
            "ui_detector.get_window_rect", return_value=(0, 0, 800, 600)
        ), mock.patch("ui_detector.get_window_pid", return_value=42424), mock.patch(
            "ui_detector.build_standard_targets", return_value=[fake_target]
        ):
            provider._poll_once(now_ms=1000)
            targets, reason = provider.get_targets(now_ms=1700)
        self.assertEqual(reason, "stale_cache")
        self.assertEqual(targets, [])

    def test_no_targets_reason(self) -> None:
        provider = WindowTargetProvider(poll_interval_ms=250, stale_after_ms=500, include_back=True)
        provider._running = True
        with mock.patch("ui_detector.get_foreground_window_handle", return_value=13), mock.patch(
            "ui_detector.get_window_rect", return_value=(0, 0, 800, 600)
        ), mock.patch("ui_detector.get_window_pid", return_value=42424), mock.patch(
            "ui_detector.build_standard_targets", return_value=[]
        ):
            provider._poll_once(now_ms=1000)
            targets, reason = provider.get_targets(now_ms=1050)
        self.assertEqual(reason, "no_targets")
        self.assertEqual(targets, [])


if __name__ == "__main__":
    unittest.main()
