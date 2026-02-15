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

    def test_local_desktop_mapping_center_point(self) -> None:
        screen_w, screen_h = 101, 101
        vx, vy, vw, vh = 100, 200, 301, 401
        center_local = (screen_w // 2, screen_h // 2)
        center_desktop = local_to_desktop_px(center_local, screen_w, screen_h, vx, vy, vw, vh)
        self.assertEqual(center_desktop, (250, 400))


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

    def test_assist_on_sets_none_when_no_target_in_range(self) -> None:
        demo = DemoUI(1200, 700, assist_on=True)
        demo.update(1000, (0, 0), face_detected=True, raw_desktop_px=(0, 0))
        self.assertIsNone(demo.assist_px)

    def test_assist_hysteresis_keeps_target_until_snap_out(self) -> None:
        demo = DemoUI(1200, 700, assist_on=True)
        demo.snap_in_r = 130.0
        demo.snap_out_r = 280.0
        cx, cy = demo.target_centers_px["C"]
        rx, _ = demo.target_centers_px["R"]

        demo.update(1000, (cx + 120, cy), face_detected=True, raw_desktop_px=(cx + 120, cy))
        self.assertEqual(demo._snapped_id, "C")

        demo.update(1033, (cx + 261, cy), face_detected=True, raw_desktop_px=(cx + 261, cy))
        self.assertEqual(demo._snapped_id, "C")

        demo.update(1066, (rx - 80, cy), face_detected=True, raw_desktop_px=(rx - 80, cy))
        self.assertEqual(demo._snapped_id, "R")


class DemoUiTargetsTests(unittest.TestCase):
    def test_get_targets_returns_five_unique_expected_ids(self) -> None:
        demo = DemoUI(1200, 700, assist_on=False)
        targets = demo.get_targets()
        ids = [target["id"] for target in targets]
        self.assertEqual(len(targets), 5)
        self.assertEqual(set(ids), {"C", "U", "D", "L", "R"})
        self.assertEqual(len(ids), len(set(ids)))

    def test_get_targets_match_demo_layout_and_radius(self) -> None:
        demo = DemoUI(1200, 700, assist_on=True)
        targets = demo.get_targets()
        for target in targets:
            tid = target["id"]
            self.assertEqual(target["radius_px"], demo.target_radius)
            self.assertEqual(target["center_px"], demo.target_centers_px[tid])


if __name__ == "__main__":
    unittest.main()
