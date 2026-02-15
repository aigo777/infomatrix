from __future__ import annotations

import unittest

from intent_predictor import (
    MagnetismParams,
    MagnetismState,
    compute_magnetized_target,
    compute_magnetized_target_with_state,
)


class MagnetismTests(unittest.TestCase):
    def test_selects_nearest_candidate_within_radius(self) -> None:
        params = MagnetismParams()
        raw_px = (100, 100)
        targets = [
            {"id": "near", "center_px": (130, 100), "radius_px": 40, "weight": 1.0},
            {"id": "far", "center_px": (170, 100), "radius_px": 40, "weight": 1.0},
        ]

        target_px, target_id, strength = compute_magnetized_target(
            raw_px=raw_px,
            targets=targets,
            prev_target_id=None,
            now_ms=1000,
            gaze_speed_px_s=0.0,
            params=params,
        )
        self.assertEqual(target_id, "near")
        self.assertGreater(strength, 0.0)
        self.assertGreater(target_px[0], raw_px[0])

    def test_preferred_target_grace_then_release(self) -> None:
        params = MagnetismParams(preferred_release_ms=250)
        state = MagnetismState()
        targets = [{"id": "A", "center_px": (130, 100), "radius_px": 40, "weight": 1.0}]

        first = compute_magnetized_target_with_state(
            raw_px=(110, 100),
            targets=targets,
            state=state,
            now_ms=1000,
            gaze_speed_px_s=0.0,
            params=params,
        )
        self.assertEqual(first.target_id, "A")

        # 120 px from center: outside both influence and grace radii.
        second = compute_magnetized_target_with_state(
            raw_px=(250, 100),
            targets=targets,
            state=first.state,
            now_ms=1100,
            gaze_speed_px_s=0.0,
            params=params,
        )
        self.assertEqual(second.state.preferred_target_id, "A")

        third = compute_magnetized_target_with_state(
            raw_px=(250, 100),
            targets=targets,
            state=second.state,
            now_ms=1400,
            gaze_speed_px_s=0.0,
            params=params,
        )
        self.assertIsNone(third.state.preferred_target_id)
        self.assertIsNone(third.target_id)
        self.assertEqual(third.target_px, (250, 100))

    def test_speed_break_temporarily_disables_assist(self) -> None:
        params = MagnetismParams(speed_break_px_s=1200.0, speed_break_hold_ms=150)
        targets = [{"id": "A", "center_px": (140, 100), "radius_px": 40, "weight": 1.0}]

        first = compute_magnetized_target_with_state(
            raw_px=(110, 100),
            targets=targets,
            state=MagnetismState(),
            now_ms=1000,
            gaze_speed_px_s=1500.0,
            params=params,
        )
        self.assertIsNone(first.target_id)
        self.assertEqual(first.strength, 0.0)
        self.assertGreater(first.state.disabled_until_ms, 1000)

        second = compute_magnetized_target_with_state(
            raw_px=(110, 100),
            targets=targets,
            state=first.state,
            now_ms=1070,
            gaze_speed_px_s=0.0,
            params=params,
        )
        self.assertIsNone(second.target_id)

        third = compute_magnetized_target_with_state(
            raw_px=(110, 100),
            targets=targets,
            state=second.state,
            now_ms=1200,
            gaze_speed_px_s=0.0,
            params=params,
        )
        self.assertEqual(third.target_id, "A")
        self.assertGreater(third.strength, 0.0)

    def test_falls_back_to_raw_when_no_targets(self) -> None:
        raw_px = (222, 333)
        target_px, target_id, strength = compute_magnetized_target(
            raw_px=raw_px,
            targets=[],
            prev_target_id=None,
            now_ms=1000,
            gaze_speed_px_s=0.0,
            params=MagnetismParams(),
        )
        self.assertEqual(target_px, raw_px)
        self.assertIsNone(target_id)
        self.assertEqual(strength, 0.0)


if __name__ == "__main__":
    unittest.main()
