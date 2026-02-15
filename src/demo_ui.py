from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, TypedDict

import cv2
import numpy as np

Point = Tuple[int, int]


class Target(TypedDict):
    id: str
    center_px: Point
    radius_px: int


def normalized_to_local_px(gaze_norm: Tuple[float, float], screen_w: int, screen_h: int) -> Point:
    """Map normalized gaze [0..1] to window-local pixel coordinates."""
    w = int(max(1, screen_w))
    h = int(max(1, screen_h))
    gx = float(np.clip(gaze_norm[0], 0.0, 1.0))
    gy = float(np.clip(gaze_norm[1], 0.0, 1.0))
    return int(round(gx * (w - 1))), int(round(gy * (h - 1)))


def local_to_desktop_px(
    local_px: Point,
    screen_w: int,
    screen_h: int,
    vx: int,
    vy: int,
    vw: int,
    vh: int,
) -> Point:
    """Project window-local pixel coordinates into desktop bounds."""
    w = int(max(1, screen_w))
    h = int(max(1, screen_h))
    span_w = int(max(1, vw))
    span_h = int(max(1, vh))
    lx = int(np.clip(local_px[0], 0, w - 1))
    ly = int(np.clip(local_px[1], 0, h - 1))
    ux = lx / max(1.0, float(w - 1))
    uy = ly / max(1.0, float(h - 1))
    dx = int(vx + round(ux * (span_w - 1)))
    dy = int(vy + round(uy * (span_h - 1)))
    return dx, dy


class DemoUI:
    def __init__(self, screen_w: int, screen_h: int, assist_on: bool = False) -> None:
        self.screen_w = int(max(1, screen_w))
        self.screen_h = int(max(1, screen_h))
        self.assist_on = bool(assist_on)

        self.state = "BROWSE"
        self.paused = False
        self.successes = 0
        self.false_selects = 0

        self.dwell_ms = 900
        self.fixation_window_ms = 600
        self.fixation_min_duration_ms = 180
        self.fixation_dispersion_px = 80.0
        self.off_target_timeout_ms = 350
        self.armed_timeout_ms = 4000
        self.success_flash_ms = 300

        self.assist_radius_factor = 2.2
        self.assist_k = 0.35
        self.assist_p = 1.8
        self.preferred_grace_factor = 1.25
        self.preferred_release_ms = 250
        self.snap_in_r = 0.0
        self.snap_out_r = 0.0

        self._last_update_ms: Optional[int] = None
        self._fixation_buffer: Deque[Tuple[int, int, int]] = deque()
        self.fixation_ok = False
        self.fixation_dispersion = (0.0, 0.0)

        self.raw_gaze_px: Optional[Point] = None
        self.assist_px: Optional[Point] = None
        self.hover_target_id: Optional[str] = None
        self.assist_strength = 0.0
        self.preferred_target_id: Optional[str] = None
        self._preferred_far_since_ms: Optional[int] = None
        self._snapped_id: Optional[str] = None

        self.dwell_target_id: Optional[str] = None
        self.dwell_elapsed_ms = 0.0

        self.armed_target_id: Optional[str] = None
        self.armed_since_ms: Optional[int] = None
        self.off_target_since_ms: Optional[int] = None

        self.success_until_ms: Optional[int] = None
        self.last_success_target_id: Optional[str] = None
        self.face_detected = False
        self.raw_desktop_px: Optional[Point] = None

        self._recompute_layout()

    def set_screen_size(self, screen_w: int, screen_h: int) -> None:
        sw = int(max(1, screen_w))
        sh = int(max(1, screen_h))
        if sw == self.screen_w and sh == self.screen_h:
            return
        self.screen_w = sw
        self.screen_h = sh
        self._recompute_layout()

    def get_targets(self) -> List[Target]:
        """Return current magnetism targets in LOCAL window pixel coordinates."""
        ordered_ids = ("C", "U", "D", "L", "R")
        return [
            {"id": tid, "center_px": self.target_centers_px[tid], "radius_px": self.target_radius}
            for tid in ordered_ids
        ]

    def get_target_by_id(self, tid: str) -> Optional[Target]:
        center = self.target_centers_px.get(tid)
        if center is None:
            return None
        return {"id": tid, "center_px": center, "radius_px": self.target_radius}

    def set_assist_enabled(self, enabled: bool) -> None:
        self.assist_on = bool(enabled)
        if not self.assist_on:
            self.assist_strength = 0.0
            self.preferred_target_id = None
            self._preferred_far_since_ms = None
            self._snapped_id = None

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def cancel_armed(self) -> None:
        if self.state != "ARMED":
            return
        self.state = "BROWSE"
        self.armed_target_id = None
        self.armed_since_ms = None
        self.off_target_since_ms = None
        self._reset_dwell()

    def confirm(self, now_ms: int) -> Dict[str, object]:
        if self.paused:
            return {"type": "ignored"}
        if self.state == "SUCCESS":
            return {"type": "ignored"}

        if self.state == "ARMED" and self.armed_target_id is not None and self.hover_target_id == self.armed_target_id:
            self.successes += 1
            self.state = "SUCCESS"
            self.success_until_ms = int(now_ms + self.success_flash_ms)
            self.last_success_target_id = self.armed_target_id
            click_px = self.assist_px
            if click_px is None:
                cx, cy = self.target_centers_px[self.armed_target_id]
                click_px = (int(cx), int(cy))
            self.armed_target_id = None
            self.armed_since_ms = None
            self.off_target_since_ms = None
            self._reset_dwell()
            return {"type": "success", "click_px": click_px}

        self.false_selects += 1
        return {"type": "false_select"}

    def update(
        self,
        now_ms: int,
        raw_gaze_px: Optional[Point],
        drift_offset_px: Point = (0, 0),
        face_detected: bool = False,
        raw_desktop_px: Optional[Point] = None,
    ) -> None:
        now_ms = int(now_ms)
        dt_ms = 0.0
        if self._last_update_ms is not None:
            dt_ms = float(max(0, now_ms - self._last_update_ms))
        self._last_update_ms = now_ms

        self.face_detected = bool(face_detected)
        self.raw_desktop_px = raw_desktop_px
        safe_raw = self._validate_local_point(raw_gaze_px, require_in_bounds=True)
        self.raw_gaze_px = safe_raw
        self._update_fixation(now_ms, safe_raw)

        base_assist = safe_raw
        if safe_raw is not None and self.assist_on:
            base_assist = self._apply_assist(now_ms, safe_raw)
        else:
            self.assist_strength = 0.0
            if safe_raw is None:
                self._snapped_id = None

        if base_assist is not None:
            ax = int(round(base_assist[0] + drift_offset_px[0]))
            ay = int(round(base_assist[1] + drift_offset_px[1]))
            self.assist_px = self._validate_local_point((ax, ay), require_in_bounds=False)
        else:
            self.assist_px = None

        self.hover_target_id = self._hit_test(self.assist_px)

        if self.state == "SUCCESS" and self.success_until_ms is not None and now_ms >= self.success_until_ms:
            self.state = "BROWSE"
            self.success_until_ms = None

        if self.paused:
            return

        if self.state == "BROWSE":
            if self.hover_target_id is not None and self.fixation_ok:
                if self.dwell_target_id != self.hover_target_id:
                    self.dwell_target_id = self.hover_target_id
                    self.dwell_elapsed_ms = dt_ms
                else:
                    self.dwell_elapsed_ms += dt_ms

                if self.dwell_elapsed_ms >= self.dwell_ms:
                    self.state = "ARMED"
                    self.armed_target_id = self.dwell_target_id
                    self.armed_since_ms = now_ms
                    self.off_target_since_ms = None
                    self._reset_dwell()
            else:
                self._reset_dwell()
            return

        if self.state == "ARMED":
            if self.hover_target_id == self.armed_target_id:
                self.off_target_since_ms = None
            elif self.hover_target_id is not None and self.hover_target_id != self.armed_target_id:
                self.cancel_armed()
                return
            else:
                if self.off_target_since_ms is None:
                    self.off_target_since_ms = now_ms
                elif now_ms - self.off_target_since_ms >= self.off_target_timeout_ms:
                    self.cancel_armed()
                    return

            if self.armed_since_ms is not None and now_ms - self.armed_since_ms >= self.armed_timeout_ms:
                self.cancel_armed()

    def render(self, frame: np.ndarray, now_ms: int) -> None:
        frame[:] = (10, 10, 10)

        for tid, center in self.target_centers_px.items():
            cx, cy = center
            base_fill = (40, 40, 40)
            outline = (95, 95, 95)
            if self.hover_target_id == tid:
                base_fill = (80, 85, 105)
                outline = (160, 170, 215)

            cv2.circle(frame, (cx, cy), self.target_radius, base_fill, -1)
            cv2.circle(frame, (cx, cy), self.target_radius, outline, 3)

            if self.state == "ARMED" and self.armed_target_id == tid:
                cv2.circle(frame, (cx, cy), self.target_radius + 10, (0, 180, 255), 5)

            if self.state == "SUCCESS" and self.last_success_target_id == tid and self.success_until_ms is not None:
                remain = max(0.0, float(self.success_until_ms - now_ms)) / max(1.0, float(self.success_flash_ms))
                pulse = int(round((1.0 - remain) * 28.0))
                cv2.circle(frame, (cx, cy), self.target_radius + 14 + pulse, (80, 220, 120), 4)
                cv2.putText(
                    frame,
                    "Selected",
                    (max(20, cx - 70), max(36, cy - self.target_radius - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (120, 240, 160),
                    2,
                )

            cv2.putText(
                frame,
                tid,
                (cx - 10, cy + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (230, 230, 230),
                2,
            )

        if self.state == "BROWSE" and self.dwell_target_id is not None and self.dwell_elapsed_ms > 0.0:
            progress = float(np.clip(self.dwell_elapsed_ms / max(1.0, float(self.dwell_ms)), 0.0, 1.0))
            cx, cy = self.target_centers_px[self.dwell_target_id]
            angle = int(round(360.0 * progress))
            cv2.ellipse(
                frame,
                (cx, cy),
                (self.target_radius + 16, self.target_radius + 16),
                0.0,
                -90.0,
                -90.0 + angle,
                (0, 220, 255),
                7,
            )

        if self.assist_px is not None:
            draw_assist = self._clamp_point(self.assist_px)
            cv2.circle(frame, draw_assist, 18, (0, 220, 255), -1)
            cv2.circle(frame, draw_assist, 24, (230, 255, 255), 2)
        if self.raw_gaze_px is not None:
            draw_raw = self._clamp_point(self.raw_gaze_px)
            cv2.circle(frame, draw_raw, 6, (155, 155, 155), -1)

        state_text = self.state if not self.paused else f"PAUSED/{self.state}"
        cv2.putText(
            frame,
            f"STATE: {state_text}",
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"SUCCESS: {self.successes}   FALSE: {self.false_selects}",
            (24, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
        )
        cv2.putText(
            frame,
            f"ASSIST {'ON' if self.assist_on else 'OFF'} preferred={self.preferred_target_id or '-'} strength={self.assist_strength:.2f}",
            (24, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )
        cv2.putText(
            frame,
            "SPACE confirm | ESC cancel | P pause | Q quit",
            (24, self.screen_h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
        )
        raw_local_dbg = self.raw_gaze_px if self.raw_gaze_px is not None else ("-", "-")
        assist_dbg = self.assist_px if self.assist_px is not None else ("-", "-")
        raw_desktop_dbg = self.raw_desktop_px if self.raw_desktop_px is not None else ("-", "-")
        cv2.putText(
            frame,
            f"raw_local={raw_local_dbg} assist_local={assist_dbg} face={int(self.face_detected)} raw_desktop={raw_desktop_dbg}",
            (24, 136),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            2,
        )

    def _recompute_layout(self) -> None:
        self.target_radius = max(120, int(0.09 * min(self.screen_w, self.screen_h)))
        self.snap_in_r = 1.05 * float(self.target_radius)
        self.snap_out_r = 1.35 * float(self.target_radius)
        w = self.screen_w - 1
        h = self.screen_h - 1
        self.target_centers_px: Dict[str, Point] = {
            "C": (int(0.50 * w), int(0.50 * h)),
            "U": (int(0.50 * w), int(0.18 * h)),
            "D": (int(0.50 * w), int(0.82 * h)),
            "L": (int(0.18 * w), int(0.50 * h)),
            "R": (int(0.82 * w), int(0.50 * h)),
        }

    def _reset_dwell(self) -> None:
        self.dwell_target_id = None
        self.dwell_elapsed_ms = 0.0

    def _clamp_point(self, pt: Point) -> Point:
        return (
            int(np.clip(pt[0], 0, self.screen_w - 1)),
            int(np.clip(pt[1], 0, self.screen_h - 1)),
        )

    def _hit_test(self, pt: Optional[Point]) -> Optional[str]:
        if pt is None:
            return None
        x, y = pt
        for tid, center in self.target_centers_px.items():
            d = math.hypot(float(x - center[0]), float(y - center[1]))
            if d <= self.target_radius:
                return tid
        return None

    def _update_fixation(self, now_ms: int, raw_gaze_px: Optional[Point]) -> None:
        if raw_gaze_px is None:
            self._fixation_buffer.clear()
            self.fixation_ok = False
            self.fixation_dispersion = (0.0, 0.0)
            return

        self._fixation_buffer.append((now_ms, int(raw_gaze_px[0]), int(raw_gaze_px[1])))
        min_ts = now_ms - self.fixation_window_ms
        while self._fixation_buffer and self._fixation_buffer[0][0] < min_ts:
            self._fixation_buffer.popleft()

        if len(self._fixation_buffer) < 2:
            self.fixation_ok = False
            self.fixation_dispersion = (0.0, 0.0)
            return

        duration_ms = self._fixation_buffer[-1][0] - self._fixation_buffer[0][0]
        xs = [p[1] for p in self._fixation_buffer]
        ys = [p[2] for p in self._fixation_buffer]
        disp_x = float(max(xs) - min(xs))
        disp_y = float(max(ys) - min(ys))
        self.fixation_dispersion = (disp_x, disp_y)
        self.fixation_ok = (
            duration_ms >= self.fixation_min_duration_ms
            and disp_x <= self.fixation_dispersion_px
            and disp_y <= self.fixation_dispersion_px
        )

    def _apply_assist(self, now_ms: int, raw_gaze_px: Point) -> Optional[Point]:
        raw_x, raw_y = raw_gaze_px
        influence_radius = self.assist_radius_factor * float(self.target_radius)
        snap_in_r = max(1.0, float(self.snap_in_r))
        snap_out_r = max(snap_in_r, float(self.snap_out_r))

        distance_by_id: Dict[str, float] = {}
        candidate_ids = []
        for tid, center in self.target_centers_px.items():
            d = math.hypot(float(raw_x - center[0]), float(raw_y - center[1]))
            distance_by_id[tid] = d
            if d <= influence_radius:
                candidate_ids.append((d, tid))

        chosen_id = None
        chosen_dist = float("inf")
        if self._snapped_id is not None:
            snapped_dist = distance_by_id.get(self._snapped_id)
            if snapped_dist is not None and snapped_dist <= snap_out_r:
                chosen_id = self._snapped_id
                chosen_dist = snapped_dist
            else:
                self._snapped_id = None

        if chosen_id is None and candidate_ids:
            candidate_ids.sort(key=lambda item: item[0])
            nearest_dist, nearest_id = candidate_ids[0]
            if nearest_dist <= snap_in_r:
                chosen_dist = nearest_dist
                chosen_id = nearest_id
                self._snapped_id = nearest_id

        self.preferred_target_id = self._snapped_id
        if self.preferred_target_id is None:
            self._preferred_far_since_ms = None

        if chosen_id is None:
            self.assist_strength = 0.0
            return None

        center = self.target_centers_px[chosen_id]
        s = float(np.clip(1.0 - (chosen_dist / max(influence_radius, 1e-6)), 0.0, 1.0))
        s = s ** self.assist_p
        if s < 0.05:
            s = 0.0
        self.assist_strength = s

        ax = raw_x + (center[0] - raw_x) * (self.assist_k * s)
        ay = raw_y + (center[1] - raw_y) * (self.assist_k * s)
        return int(round(ax)), int(round(ay))

    def _validate_local_point(self, pt: Optional[Point], require_in_bounds: bool) -> Optional[Point]:
        if pt is None:
            return None
        try:
            x = int(round(float(pt[0])))
            y = int(round(float(pt[1])))
        except (TypeError, ValueError, IndexError):
            return None
        if require_in_bounds and (x < 0 or x >= self.screen_w or y < 0 or y >= self.screen_h):
            return None
        return x, y
