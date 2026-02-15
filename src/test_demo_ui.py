from __future__ import annotations

import unittest

from demo_ui import DemoUI, local_to_desktop_px, normalized_to_local_px


class DemoMappingTests(unittest.TestCase):
    def test_normalized_local_mapping_bounds(self) -> None:
        w, h = 1493, 933
        self.assertEqual(normalized_to_local_px((0.0, 0.0), w, h), (0, 0))
        self.assertEqual(normalized_to_local_px((1.0, 1.0), w, h), (w - 1, h - 1))
        mid = normalized_to_local_px((0.5, 0.5), w, h)
        self.assertTrue(0 <= mid[0] < w)
        self.assertTrue(0 <= mid[1] < h)

    def test_local_desktop_mapping_multimon(self) -> None:
        screen_w, screen_h = 1920, 1080
        vx, vy, vw, vh = -1920, 0, 3840, 1080

        left_top = local_to_desktop_px((0, 0), screen_w, screen_h, vx, vy, vw, vh)
        right_bottom = local_to_desktop_px((screen_w - 1, screen_h - 1), screen_w, screen_h, vx, vy, vw, vh)
        center = local_to_desktop_px((screen_w // 2, screen_h // 2), screen_w, screen_h, vx, vy, vw, vh)

        self.assertEqual(left_top, (vx, vy))
        self.assertEqual(right_bottom, (vx + vw - 1, vy + vh - 1))
        self.assertTrue(vx <= center[0] <= vx + vw - 1)
        self.assertTrue(vy <= center[1] <= vy + vh - 1)


class DemoUiBehaviorTests(unittest.TestCase):
    def test_invalid_or_out_of_bounds_gaze_does_not_pin_corner(self) -> None:
        demo = DemoUI(1000, 600, assist_on=False)
        demo.update(1000, None, face_detected=False, raw_desktop_px=None)
        self.assertIsNone(demo.raw_gaze_px)
        self.assertIsNone(demo.assist_px)

        demo.update(1033, (-100, 700), face_detected=True, raw_desktop_px=(-500, 700))
        self.assertIsNone(demo.raw_gaze_px)
        self.assertIsNone(demo.assist_px)

    def test_assist_off_cursor_follows_raw_local(self) -> None:
        demo = DemoUI(1200, 700, assist_on=False)
        demo.update(1000, (200, 300), face_detected=True, raw_desktop_px=(200, 300))
        self.assertEqual(demo.raw_gaze_px, (200, 300))
        self.assertEqual(demo.assist_px, (200, 300))

        demo.update(1033, (245, 315), face_detected=True, raw_desktop_px=(245, 315))
        self.assertEqual(demo.raw_gaze_px, (245, 315))
        self.assertEqual(demo.assist_px, (245, 315))


if __name__ == "__main__":
    unittest.main()
