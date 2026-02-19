from __future__ import annotations
a=0
import argparse
import os
import time
from collections import deque
import math
import cv2
import numpy as np
from gaze_tracker import GazeTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="IrisKeys - Stage 3.0")
    parser.add_argument(
        "--drift",
        choices=("off", "on"),
        default="off",
        help="Enable drift integrator (default: off).",
    )
    parser.add_argument(
        "--drift-max",
        type=float,
        default=0.002,
        help="Absolute drift cap when --drift on (default: 0.002).",
    )
    parser.add_argument(
        "--os-click",
        choices=("off", "on"),
        default="off",
        help="Emit OS left click on SUCCESS (default: off).",
    )
    parser.add_argument(
        "--assist",
        choices=("off", "on"),
        default="off",
        help="Enable gentle assistive magnetism for demo targets (default: off).",
    )
    args = parser.parse_args()
    drift_enabled = args.drift == "on"
    drift_max = float(np.clip(args.drift_max, 0.0, 0.01))
    os_click_enabled = args.os_click == "on"
    assist_enabled = args.assist == "on"

    print("IrisKeys - Stage 3.0")
    print(
        "Controls: q quit | esc cancel selection | m toggle mouse | p pause | space confirm | s screenshot | k calibrate | r reset | l load calibration | t test | [/] edge_gain | 9/0 y_edge_gain | i/k y_scale | o/l y_offset | y flip | ,/. spring_k"
    )
    print(f"Drift integrator: {'ON' if drift_enabled else 'OFF'} (max={drift_max:.4f})")
    print(f"OS click on success: {'ON' if os_click_enabled else 'OFF'}")
    print(f"Assist magnetism: {'ON' if assist_enabled else 'OFF'}")

    tracker = GazeTracker()
    cap = cv2.VideoCapture(tracker.camera_index)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    debug_win = "IrisKeys Debug"
    pointer_win = "IrisKeys Pointer"
    cv2.namedWindow(pointer_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(debug_win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(pointer_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    calib_dir = os.path.join(base_dir, "calibration")
    calib_path = os.path.join(calib_dir, "calibration_data.json")
    os.makedirs(calib_dir, exist_ok=True)

    if tracker.load_calibration(calib_path):
        print(f"Loaded calibration from {calib_path}")

    calib_targets = [
        ("tl", "TOP-LEFT", (0.08, 0.08)),
        ("t", "TOP", (0.50, 0.08)),
        ("tr", "TOP-RIGHT", (0.92, 0.08)),
        ("l", "LEFT", (0.08, 0.50)),
        ("center", "CENTER", (0.50, 0.50)),
        ("r", "RIGHT", (0.92, 0.50)),
        ("bl", "BOTTOM-LEFT", (0.08, 0.92)),
        ("b", "BOTTOM", (0.50, 0.92)),
        ("br", "BOTTOM-RIGHT", (0.92, 0.92)),
    ]
    calib_active = False
    calib_phase = "idle"
    calib_index = 0
    calib_samples: list[tuple[float, float, float]] = []
    calib_data: dict[str, tuple[float, float]] = {}
    calib_quality: dict[str, dict[str, float]] = {}
    calib_open: dict[str, float] = {}
    center_open_list: list[float] = []
    center_gy_list: list[float] = []
    calib_settle_s = 0.8
    calib_samples_needed = 75
    calib_phase_start = 0.0
    calib_pad = 0.03
    calib_mad_thresh = 0.015
    test_active = False
    test_start = 0.0
    test_duration = 1.0

    screen_w = None
    screen_h = None
    vx = 0
    vy = 0
    vw = None
    vh = None
    cursor_backend = "ctypes"
    user32 = None
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004

    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        screen_w = int(user32.GetSystemMetrics(0))
        screen_h = int(user32.GetSystemMetrics(1))
        vx = int(user32.GetSystemMetrics(76))
        vy = int(user32.GetSystemMetrics(77))
        vw = int(user32.GetSystemMetrics(78))
        vh = int(user32.GetSystemMetrics(79))
    except Exception:
        screen_w = None
        screen_h = None
        vx = 0
        vy = 0
        vw = None
        vh = None

    if vw is None or vh is None or vw <= 0 or vh <= 0:
        vw = screen_w
        vh = screen_h

    mouse_enabled = False
    mouse_paused = False
    lost_frames = 0
    last_face_time = time.time()
    last_cursor_time = time.time()
    sx = None
    sy = None
    vx_c = 0.0
    vy_c = 0.0
    edge_gain = 0.25
    # Cursor dynamics (tune for smoothness)
    spring_k = 55.0
    spring_d = 14.0
    max_speed = 2200.0
    max_accel = 9000.0
    drift_x = 0.0
    drift_y = 0.0
    y_scale = 1.35
    y_offset = 0.0
    y_flip = False
    y_edge_gain = 0.18

    fixation_window_ms = 260
    fixation_min_duration_ms = 180
    dispersion_threshold_px = 60.0
    dwell_ms = 700
    off_target_cancel_ms = 150
    armed_timeout_ms = 1200
    success_flash_ms = 420
    space_debounce_ms = 180
    assist_radius_factor = 1.6
    assist_k = 0.35
    assist_power = 2.0
    assist_release_far_factor = 1.2
    assist_stick_grace_ms = 180
    assist_deadzone_strength = 0.05
    demo_targets_norm = [
        ("tl", (0.22, 0.22)),
        ("tr", (0.78, 0.22)),
        ("center", (0.50, 0.50)),
        ("bl", (0.22, 0.78)),
        ("br", (0.78, 0.78)),
    ]

    interaction_state = "BROWSE"
    success_count = 0
    false_count = 0
    gaze_buffer: deque[tuple[float, int, int]] = deque(maxlen=120)
    fixation_since = None
    current_dispersion_px = None
    dwell_target_id = None
    dwell_start_ts = None
    dwell_progress = 0.0
    armed_target_id = None
    armed_since = None
    off_target_since = None
    last_space_ts = 0.0
    success_anim_target_id = None
    success_anim_until = 0.0
    preferred_target_id = None
    preferred_far_since = None
    assist_strength = 0.0

    print("Cursor backend: ctypes")

    def clamp01(val: float) -> float:
        return float(np.clip(val, 0.0, 1.0))

    def soft_edge_curve(u: float, strength: float) -> float:
        # strength in [0..1], 0 = linear, 1 = strong
        u = float(np.clip(u, 0.0, 1.0))
        s = float(np.clip(strength, 0.0, 1.0))
        smooth = u * u * (3.0 - 2.0 * u)
        return (1.0 - s) * u + s * smooth

    def mid_edge_expand(u: float, v: float, strength: float = 0.18) -> float:
        """
        Expands coordinate u when the orthogonal coordinate v is near center.
        Used to fix mid-edge compression.
        """
        v_dist = abs(v - 0.5) * 2.0
        center_weight = 1.0 - np.clip(v_dist, 0.0, 1.0)
        expand = strength * center_weight
        return float(np.clip(0.5 + (u - 0.5) * (1.0 + expand), 0.0, 1.0))

    def vertical_extreme_damp(u: float, strength: float = 0.35) -> float:
        """
        Damp movement near top/bottom edges to prevent snapping.
        """
        d = abs(u - 0.5) * 2.0
        if d < 0.75:
            return u
        t = (d - 0.75) / 0.25
        damp = 1.0 - strength * t * t
        return float(np.clip(0.5 + (u - 0.5) * damp, 0.0, 1.0))

    def cornerness(x: float, y: float) -> float:
        # 0 = center, 1 = corner
        return max(abs(x - 0.5), abs(y - 0.5)) * 2.0

    def transform_gaze(gaze: tuple[float, float]) -> tuple[float, float]:
        gx = clamp01(gaze[0])
        gy = gaze[1]
        if y_flip:
            gy = 1.0 - gy
        gy = (gy - 0.5) * y_scale + 0.5 + y_offset
        gy = clamp01(gy)

        #gx = mid_edge_expand(gx, gy, strength=0.18)
        #gy = mid_edge_expand(gy, gx, strength=0.08)

        c = cornerness(gx, gy)
        if c < 0.75:
            gx = soft_edge_curve(gx, edge_gain)
            gy = soft_edge_curve(gy, y_edge_gain)
            gy = vertical_extreme_damp(gy, strength=0.30)

        #reach = 1.08
        #gx = clamp01(0.5 + (gx - 0.5) * reach)
        #gy = clamp01(0.5 + (gy - 0.5) * reach)

        vel = tracker._last_velocity
        #if vel is not None and vel < 0.015:
            #precision_gain = 1.12
            #gx = clamp01(0.5 + (gx - 0.5) * precision_gain)
            #gy = clamp01(0.5 + (gy - 0.5) * precision_gain)
        return gx, gy

    def get_cursor_pos() -> tuple[int, int]:
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)

    def set_cursor_pos(x_px: int, y_px: int) -> None:
        user32.SetCursorPos(int(x_px), int(y_px))

    def emit_left_click() -> None:
        if user32 is None:
            print("OS click unavailable: user32 not initialized.")
            return
        try:
            get_cursor_pos()
            user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        except Exception as exc:
            print(f"OS click failed: {exc}")

    def reset_dwell() -> None:
        nonlocal dwell_target_id, dwell_start_ts, dwell_progress
        dwell_target_id = None
        dwell_start_ts = None
        dwell_progress = 0.0

    def reset_armed() -> None:
        nonlocal interaction_state, armed_target_id, armed_since, off_target_since
        interaction_state = "BROWSE"
        armed_target_id = None
        armed_since = None
        off_target_since = None

    frame_times = deque(maxlen=30)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: failed to read frame.")
            break

        frame = cv2.flip(frame, 1)

        result = tracker.process_frame(frame)
        face_detected = bool(result.get("face_detected"))
        gaze_raw = result.get("gaze_raw_uncal")
        eye_open = result.get("eye_openness")
        gaze_pointer = tracker.get_mapped_gaze(result)

        if isinstance(gaze_pointer, tuple):
            if drift_enabled and mouse_enabled:
                drift_rate = 0.002
                drift_x = float(
                    np.clip(drift_x + (gaze_pointer[0] - 0.5) * drift_rate, -drift_max, drift_max)
                )
                drift_y = float(
                    np.clip(drift_y + (gaze_pointer[1] - 0.5) * drift_rate, -drift_max, drift_max)
                )
            elif drift_enabled:
                drift_x *= 0.995
                drift_y *= 0.995
            else:
                drift_x = 0.0
                drift_y = 0.0
        elif not drift_enabled:
            drift_x = 0.0
            drift_y = 0.0

        display_gaze = None
        target_x = None
        target_y = None
        if isinstance(gaze_pointer, tuple) and vw is not None and vh is not None:
            display_gaze = transform_gaze(gaze_pointer)
            dx_off = drift_x if drift_enabled else 0.0
            dy_off = drift_y if drift_enabled else 0.0
            dx = clamp01(display_gaze[0] + dx_off)
            dy = clamp01(display_gaze[1] + dy_off)
            target_x = vx + int(dx * (vw - 1))
            target_y = vy + int(dy * (vh - 1))

        if screen_w is None or screen_h is None:
            screen_h, screen_w = frame.shape[:2]
        if vw is None or vh is None or vw <= 0 or vh <= 0:
            vw = screen_w
            vh = screen_h

        now = time.time()
        if face_detected:
            last_face_time = now
            lost_frames = 0
        else:
            lost_frames += 1
            if now - last_face_time > 0.25 and mouse_enabled:
                mouse_enabled = False
                sx = None
                sy = None
                vx_c = 0.0
                vy_c = 0.0
                print("Mouse control auto-disabled (face lost).")

        if calib_active:
            if calib_index >= len(calib_targets):
                calib_active = False
                calib_phase = "idle"
            else:
                if calib_phase == "settle":
                    if time.time() - calib_phase_start >= calib_settle_s:
                        calib_phase = "capture"
                        calib_samples = []
                elif calib_phase == "capture":
                    if face_detected and isinstance(gaze_raw, tuple) and isinstance(eye_open, float):
                        calib_samples.append((gaze_raw[0], gaze_raw[1], eye_open))
                    if len(calib_samples) >= calib_samples_needed:
                        gaze_only = [(s[0], s[1]) for s in calib_samples]
                        opens = [s[2] for s in calib_samples]
                        med, mad = tracker.compute_median_mad(gaze_only)
                        open_med = float(np.median(opens)) if opens else 0.0
                        if max(mad[0], mad[1]) > calib_mad_thresh:
                            print(f"Calibration point unstable ({calib_targets[calib_index][1]}), retrying.")
                            calib_phase = "settle"
                            calib_phase_start = time.time()
                            calib_samples = []
                            continue
                        name = calib_targets[calib_index][0]
                        calib_data[name] = (float(med[0]), float(med[1]))
                        calib_open[name] = open_med
                        if name == "center":
                            center_open_list = opens[:]
                            center_gy_list = [s[1] for s in calib_samples]
                        calib_quality[name] = {
                            "median_x": float(med[0]),
                            "median_y": float(med[1]),
                            "mad_x": float(mad[0]),
                            "mad_y": float(mad[1]),
                        }
                        calib_index += 1
                        calib_phase = "settle"
                        calib_phase_start = time.time()
                        calib_samples = []

                if calib_index >= len(calib_targets):
                    calib_active = False
                    calib_phase = "idle"
                    success = tracker.set_full_calibration(calib_data, pad=calib_pad)
                    if success:
                        open_ref = calib_open.get("center")
                        beta_y = 0.0
                        if center_open_list:
                            open_arr = np.array(center_open_list, dtype=np.float32)
                            gy_arr = np.array(center_gy_list, dtype=np.float32)
                            mean_open = float(np.mean(open_arr))
                            mean_gy = float(np.mean(gy_arr))
                            cov = float(np.mean((gy_arr - mean_gy) * (open_arr - mean_open)))
                            var = float(np.mean((open_arr - mean_open) ** 2))
                            if var > 1e-6:
                                beta_y = cov / var
                        if open_ref is not None:
                            tracker.set_openness_compensation(open_ref, beta_y)
                        tracker.set_calibration_quality(calib_quality)
                        tracker.save_calibration(calib_path)
                        top_vals = [calib_data[k][1] for k in ("tl", "t", "tr") if k in calib_data]
                        bot_vals = [calib_data[k][1] for k in ("bl", "b", "br") if k in calib_data]
                        top_vals = [tracker.apply_axis_flip((0.5, y))[1] for y in top_vals]
                        bot_vals = [tracker.apply_axis_flip((0.5, y))[1] for y in bot_vals]
                        if top_vals and bot_vals:
                            span = abs(float(np.median(bot_vals)) - float(np.median(top_vals)))
                            if span < 0.10:
                                print("Vertical span too small; keep head fixed, avoid eyebrow movement; try again.")
                                y_scale = float(np.clip(0.18 / max(span, 1e-3), 0.8, 2.5))
                        print(f"Calibration saved to {calib_path}")
                        
                    else:
                        print("Calibration failed: range invalid.")

        raw_gaze_px = None
        assist_px = None
        if isinstance(display_gaze, tuple) and screen_w is not None and screen_h is not None:
            raw_gaze_px = (
                int(np.clip(display_gaze[0], 0.0, 1.0) * (screen_w - 1)),
                int(np.clip(display_gaze[1], 0.0, 1.0) * (screen_h - 1)),
            )

        demo_targets = []
        demo_targets_by_id = {}
        if screen_w is not None and screen_h is not None:
            demo_radius = max(48, int(min(screen_w, screen_h) * 0.075))
            for target_id, (nx, ny) in demo_targets_norm:
                cx = int(nx * (screen_w - 1))
                cy = int(ny * (screen_h - 1))
                target = {"id": target_id, "cx": cx, "cy": cy, "r": demo_radius}
                demo_targets.append(target)
                demo_targets_by_id[target_id] = target

        assist_strength = 0.0
        if isinstance(raw_gaze_px, tuple):
            assist_px = raw_gaze_px
        if not assist_enabled:
            preferred_target_id = None
            preferred_far_since = None
        if preferred_target_id not in demo_targets_by_id:
            preferred_target_id = None
            preferred_far_since = None

        if assist_enabled and isinstance(raw_gaze_px, tuple) and demo_targets:
            gx_raw, gy_raw = raw_gaze_px
            best_id = None
            best_score = -1e9
            for target in demo_targets:
                cx = target["cx"]
                cy = target["cy"]
                r = target["r"]
                r_assist = assist_radius_factor * r
                d_raw = float(math.hypot(gx_raw - cx, gy_raw - cy))
                if d_raw > r_assist:
                    continue
                score = r_assist - d_raw
                if preferred_target_id == target["id"]:
                    score += 0.15 * r_assist
                if interaction_state == "ARMED" and armed_target_id == target["id"]:
                    score += 0.25 * r_assist
                if score > best_score:
                    best_score = score
                    best_id = target["id"]

            if preferred_target_id is None and best_id is not None:
                preferred_target_id = best_id
                preferred_far_since = None

            if preferred_target_id is not None and preferred_target_id in demo_targets_by_id:
                pref = demo_targets_by_id[preferred_target_id]
                pref_r_assist = assist_radius_factor * pref["r"]
                pref_far = assist_release_far_factor * pref_r_assist
                d_pref = float(math.hypot(gx_raw - pref["cx"], gy_raw - pref["cy"]))
                if d_pref > pref_far:
                    if preferred_far_since is None:
                        preferred_far_since = now
                    elif (now - preferred_far_since) * 1000.0 >= assist_stick_grace_ms:
                        preferred_target_id = None
                        preferred_far_since = None
                else:
                    preferred_far_since = None

            if preferred_target_id is None and best_id is not None:
                preferred_target_id = best_id

            force_id = preferred_target_id if preferred_target_id in demo_targets_by_id else best_id
            if force_id is not None:
                target = demo_targets_by_id[force_id]
                cx = target["cx"]
                cy = target["cy"]
                r_assist = assist_radius_factor * target["r"]
                d_raw = float(math.hypot(gx_raw - cx, gy_raw - cy))
                u = float(np.clip(1.0 - d_raw / max(r_assist, 1e-6), 0.0, 1.0))
                s = float(u ** assist_power)
                if s < assist_deadzone_strength:
                    s = 0.0
                assist_strength = s
                a = assist_k * s
                ax = int(round(gx_raw + (cx - gx_raw) * a))
                ay = int(round(gy_raw + (cy - gy_raw) * a))
                assist_px = (
                    int(np.clip(ax, 0, max(screen_w - 1, 0))),
                    int(np.clip(ay, 0, max(screen_h - 1, 0))),
                )

        hover_target_id = None
        if isinstance(assist_px, tuple):
            gx_px, gy_px = assist_px
            for target in demo_targets:
                if math.hypot(gx_px - target["cx"], gy_px - target["cy"]) <= target["r"]:
                    hover_target_id = target["id"]
                    break

        tracking_ok = (
            isinstance(raw_gaze_px, tuple)
            and face_detected
            and tracker.has_full_calibration()
            and not calib_active
            and not mouse_paused
        )
        fixation_true = False
        if tracking_ok and isinstance(raw_gaze_px, tuple):
            gaze_buffer.append((now, raw_gaze_px[0], raw_gaze_px[1]))
            while gaze_buffer and (now - gaze_buffer[0][0]) * 1000.0 > fixation_window_ms:
                gaze_buffer.popleft()

            if len(gaze_buffer) >= 2:
                xs = [p[1] for p in gaze_buffer]
                ys = [p[2] for p in gaze_buffer]
                current_dispersion_px = float(math.hypot(max(xs) - min(xs), max(ys) - min(ys)))
                if current_dispersion_px <= dispersion_threshold_px:
                    if fixation_since is None:
                        fixation_since = now
                else:
                    fixation_since = None
            else:
                current_dispersion_px = 0.0
                fixation_since = None
            fixation_true = (
                fixation_since is not None
                and (now - fixation_since) * 1000.0 >= fixation_min_duration_ms
            )
        else:
            gaze_buffer.clear()
            fixation_since = None
            current_dispersion_px = None
            if interaction_state == "BROWSE":
                reset_dwell()

        key = cv2.waitKey(1) & 0xFF
        now_ms = now * 1000.0
        space_rising = False
        if key == ord(" ") and now_ms - last_space_ts >= space_debounce_ms:
            space_rising = True
            last_space_ts = now_ms
        esc_pressed = key == 27

        if interaction_state == "BROWSE":
            if tracking_ok and fixation_true and hover_target_id is not None:
                if dwell_target_id != hover_target_id or dwell_start_ts is None:
                    dwell_target_id = hover_target_id
                    dwell_start_ts = now
                    dwell_progress = 0.0
                else:
                    dwell_progress = float(np.clip((now - dwell_start_ts) * 1000.0 / dwell_ms, 0.0, 1.0))
                if dwell_progress >= 1.0:
                    interaction_state = "ARMED"
                    armed_target_id = hover_target_id
                    armed_since = now
                    off_target_since = None
                    reset_dwell()
            else:
                reset_dwell()

            if space_rising:
                false_count += 1
                print(
                    f"[{time.strftime('%H:%M:%S')}] FALSE_SELECT reason=space_in_browse "
                    f"success={success_count} false={false_count}"
                )

            if esc_pressed:
                reset_dwell()
        elif interaction_state == "ARMED":
            on_armed_target = armed_target_id is not None and hover_target_id == armed_target_id
            if esc_pressed:
                print(f"[{time.strftime('%H:%M:%S')}] CANCEL reason=esc")
                reset_armed()
                reset_dwell()
            elif space_rising:
                if on_armed_target and armed_target_id is not None:
                    selected_target = armed_target_id
                    success_count += 1
                    success_anim_target_id = selected_target
                    success_anim_until = now + success_flash_ms / 1000.0
                    print(
                        f"[{time.strftime('%H:%M:%S')}] SUCCESS target={selected_target} "
                        f"success={success_count} false={false_count}"
                    )
                    if os_click_enabled:
                        emit_left_click()
                else:
                    false_count += 1
                    print(
                        f"[{time.strftime('%H:%M:%S')}] FALSE_SELECT reason=space_off_target "
                        f"success={success_count} false={false_count}"
                    )
                reset_armed()
                reset_dwell()
            else:
                cancel_reason = None
                if armed_since is not None and (now - armed_since) * 1000.0 >= armed_timeout_ms:
                    cancel_reason = "armed_timeout"
                elif hover_target_id is not None and armed_target_id is not None and hover_target_id != armed_target_id:
                    cancel_reason = "target_changed"
                else:
                    if on_armed_target:
                        off_target_since = None
                    else:
                        if off_target_since is None:
                            off_target_since = now
                        elif (now - off_target_since) * 1000.0 >= off_target_cancel_ms:
                            cancel_reason = "off_target_timeout"
                if cancel_reason is not None:
                    print(f"[{time.strftime('%H:%M:%S')}] CANCEL reason={cancel_reason}")
                    reset_armed()
                    reset_dwell()

        frame_times.append(now)
        fps = 0.0
        if len(frame_times) >= 2:
            span = frame_times[-1] - frame_times[0]
            fps = (len(frame_times) - 1) / span if span > 0 else 0.0

        h, w = frame.shape[:2]
        calib_status = "FULL" if tracker.has_full_calibration() else "NONE"
        if calib_active and calib_index < len(calib_targets):
            target_label = calib_targets[calib_index][1]
            if calib_phase == "settle":
                calib_status = f"IN_PROGRESS {target_label} (settle)"
            else:
                calib_status = f"IN_PROGRESS {target_label} ({len(calib_samples)}/{calib_samples_needed})"

        result["calib_status"] = calib_status
        debug_frame = tracker.draw_debug(frame, result)

        cv2.putText(
            debug_frame,
            f"FPS: {fps:5.1f}",
            (w - 140, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        mouse_status = f"mouse={'ON' if mouse_enabled else 'OFF'} pause={'ON' if mouse_paused else 'OFF'}"
        backend_text = f"backend={cursor_backend} lost={lost_frames}"
        drift_text = (
            f"drift={'ON' if drift_enabled else 'OFF'} max={drift_max:.4f} "
            f"({drift_x:+.4f},{drift_y:+.4f})"
        )
        desktop_text = f"desktop=({vx},{vy},{vw},{vh}) target=({target_x},{target_y})"
        cv2.putText(
            debug_frame,
            mouse_status,
            (10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            debug_frame,
            backend_text,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            debug_frame,
            drift_text,
            (10, h - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            debug_frame,
            desktop_text,
            (10, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow(debug_win, debug_frame)

        pointer_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        interaction_visible = tracker.has_full_calibration() and not calib_active
        if interaction_visible:
            for target in demo_targets:
                target_id = target["id"]
                center = (target["cx"], target["cy"])
                radius = target["r"]
                fill = (36, 36, 36)
                stroke = (150, 150, 150)
                thickness = 2

                if target_id == hover_target_id:
                    fill = (62, 92, 136)
                    stroke = (100, 220, 255)
                    thickness = 3
                if interaction_state == "ARMED" and target_id == armed_target_id:
                    fill = (52, 130, 52)
                    stroke = (70, 255, 150)
                    thickness = 4
                if success_anim_target_id == target_id and now <= success_anim_until:
                    pulse = int((now * 14.0) % 2)
                    fill = (52, 190, 52) if pulse == 0 else (52, 235, 52)
                    stroke = (255, 255, 255)
                    thickness = 5

                cv2.circle(pointer_frame, center, radius, fill, -1)
                cv2.circle(pointer_frame, center, radius, stroke, thickness)
                cv2.putText(
                    pointer_frame,
                    target_id.upper(),
                    (center[0] - radius // 3, center[1] + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        if interaction_state == "BROWSE" and dwell_target_id in demo_targets_by_id and dwell_progress > 0.0:
            target = demo_targets_by_id[dwell_target_id]
            end_angle = int(-90 + 360.0 * dwell_progress)
            cv2.ellipse(
                pointer_frame,
                (target["cx"], target["cy"]),
                (target["r"] + 14, target["r"] + 14),
                0,
                -90,
                end_angle,
                (0, 255, 255),
                5,
            )
        if interaction_state == "ARMED" and armed_target_id in demo_targets_by_id:
            target = demo_targets_by_id[armed_target_id]
            cv2.ellipse(
                pointer_frame,
                (target["cx"], target["cy"]),
                (target["r"] + 14, target["r"] + 14),
                0,
                0,
                360,
                (0, 255, 140),
                5,
            )
            cv2.putText(
                pointer_frame,
                "ARMED",
                (target["cx"] - 42, target["cy"] - target["r"] - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 140),
                2,
            )
        if success_anim_target_id in demo_targets_by_id and now <= success_anim_until:
            target = demo_targets_by_id[success_anim_target_id]
            cv2.putText(
                pointer_frame,
                "Selected",
                (target["cx"] - 52, target["cy"] - target["r"] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        if assist_enabled and isinstance(raw_gaze_px, tuple) and isinstance(assist_px, tuple):
            if raw_gaze_px != assist_px:
                cv2.line(pointer_frame, raw_gaze_px, assist_px, (120, 120, 120), 1)
            cv2.circle(pointer_frame, raw_gaze_px, 6, (180, 180, 180), -1)
        if isinstance(assist_px, tuple):
            cv2.circle(pointer_frame, assist_px, 20, (255, 0, 0), -1)
            cv2.circle(pointer_frame, assist_px, 24, (255, 255, 255), 2)

        dwell_pct = int(100 if interaction_state == "ARMED" else round(dwell_progress * 100.0))
        disp_text = f"{current_dispersion_px:.1f}" if isinstance(current_dispersion_px, float) else "None"
        assist_line = (
            f"ASSIST: {'ON' if assist_enabled else 'OFF'} "
            f"preferred={preferred_target_id or '-'} strength={assist_strength:.2f}"
        )
        hud_lines = [
            f"state={interaction_state} hover={hover_target_id or '-'} armed={armed_target_id or '-'}",
            f"fixation={'YES' if fixation_true else 'NO'} dispersion_px={disp_text} dwell={dwell_pct}%",
            f"success_count={success_count} false_count={false_count} os_click={'ON' if os_click_enabled else 'OFF'}",
            assist_line,
            f"mouse={'ON' if mouse_enabled else 'OFF'} pause={'ON' if mouse_paused else 'OFF'}",
            "SPACE confirm | ESC cancel | P pause | Q quit | K calibrate | M mouse",
        ]
        y_hud = 28
        for line in hud_lines:
            cv2.putText(
                pointer_frame,
                line,
                (20, y_hud),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7 if y_hud == 28 else 0.62,
                (255, 255, 255),
                2,
            )
            y_hud += 28

        if test_active:
            elapsed = time.time() - test_start
            if elapsed >= test_duration * 2:
                test_active = False
            else:
                phase = 0 if elapsed < test_duration else 1
                label = "TEST TOP" if phase == 0 else "TEST BOTTOM"
                pos = (0.5, 0.15) if phase == 0 else (0.5, 0.85)
                tx = int(pos[0] * (screen_w - 1))
                ty = int(pos[1] * (screen_h - 1))
                cv2.line(pointer_frame, (tx - 25, ty), (tx + 25, ty), (255, 255, 255), 2)
                cv2.line(pointer_frame, (tx, ty - 25), (tx, ty + 25), (255, 255, 255), 2)

                raw_text = "raw gy=None"
                if isinstance(gaze_raw, tuple):
                    raw_text = f"raw gy={gaze_raw[1]:.3f}"
                trans_text = "gy2=None"
                if isinstance(display_gaze, tuple):
                    trans_text = f"gy2={display_gaze[1]:.3f}"
                target_text = f"target_y={target_y}"
                cv2.putText(
                    pointer_frame,
                    f"{label} {raw_text} {trans_text} {target_text}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
        elif calib_active and calib_index < len(calib_targets):
            _, label, pos = calib_targets[calib_index]
            tx = int(pos[0] * (screen_w - 1))
            ty = int(pos[1] * (screen_h - 1))
            cv2.line(pointer_frame, (tx - 25, ty), (tx + 25, ty), (255, 255, 255), 2)
            cv2.line(pointer_frame, (tx, ty - 25), (tx, ty + 25), (255, 255, 255), 2)

            instruction = f"LOOK AT {label}"
            size, _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(
                pointer_frame,
                instruction,
                ((screen_w - size[0]) // 2, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
        elif not tracker.has_full_calibration():
            instruction = "PRESS K TO CALIBRATE"
            size, _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(
                pointer_frame,
                instruction,
                ((screen_w - size[0]) // 2, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

        cv2.imshow(pointer_win, pointer_frame)

        should_move = (
            mouse_enabled
            and not mouse_paused
            and tracker.has_full_calibration()
            and not calib_active
            and face_detected
            and isinstance(display_gaze, tuple)
            and target_x is not None
            and target_y is not None
        )
        if should_move:
            try:
                if sx is None or sy is None:
                    cx, cy = get_cursor_pos()
                    sx = float(cx)
                    sy = float(cy)
                    vx_c = 0.0
                    vy_c = 0.0

                dt = float(np.clip(now - last_cursor_time, 1e-4, 0.05))
                last_cursor_time = now

                ex = float(target_x - sx)
                ey = float(target_y - sy)
                ax = spring_k * ex - spring_d * vx_c
                edge_y = abs((sy / vh) - 0.5) * 2.0 if vh else 0.0
                y_damp = 1.0 - 0.35 * np.clip(edge_y, 0.0, 1.0)
                ay = (spring_k * y_damp) * ey - spring_d * vy_c
                ax = float(np.clip(ax, -max_accel, max_accel))
                ay = float(np.clip(ay, -max_accel, max_accel))
                vx_c += ax * dt
                vy_c += ay * dt
                vx_c = float(np.clip(vx_c, -max_speed, max_speed))
                vy_c = float(np.clip(vy_c, -max_speed, max_speed))
                sx += vx_c * dt
                sy += vy_c * dt
                set_cursor_pos(int(round(sx)), int(round(sy)))
            except Exception as exc:
                print(f"Mouse control error: {exc}")
                mouse_enabled = False

        if key == ord("q"):
            break
        if key == ord("s"):
            filename = time.strftime("screenshot_%Y%m%d_%H%M%S.png")
            path = os.path.join(os.getcwd(), filename)
            cv2.imwrite(path, debug_frame)
            print(f"Saved screenshot: {path}")
        if key == ord("k"):
            if mouse_enabled or mouse_paused:
                y_scale = float(np.clip(y_scale - 0.05, 0.6, 2.5))
                print(f"y_scale={y_scale:.2f}")
            else:
                calib_active = True
                calib_phase = "settle"
                calib_index = 0
                calib_phase_start = time.time()
            calib_samples = []
            calib_data = {}
            calib_quality = {}
            calib_open = {}
            center_open_list = []
            center_gy_list = []
            tracker.reset_calibration()
            mouse_enabled = False
            sx = None
            sy = None
            vx_c = 0.0
            vy_c = 0.0
            print("Calibration started: 3x3 grid (top-left -> top -> top-right -> left -> center -> right -> bottom-left -> bottom -> bottom-right)")
        if key == ord("r"):
            tracker.reset_calibration()
            calib_active = False
            calib_phase = "idle"
            calib_index = 0
            calib_samples = []
            calib_data = {}
            calib_quality = {}
            calib_open = {}
            center_open_list = []
            center_gy_list = []
            mouse_enabled = False
            sx = None
            sy = None
            vx_c = 0.0
            vy_c = 0.0
            print("Calibration reset.")
        if key == ord("l"):
            if mouse_enabled or mouse_paused:
                y_offset = float(np.clip(y_offset - 0.01, -0.25, 0.25))
                print(f"y_offset={y_offset:.2f}")
            else:
                if tracker.load_calibration(calib_path):
                    print(f"Loaded calibration from {calib_path}")
                else:
                    print("Load failed: no calibration data found.")
        if key == ord("m"):
            if not tracker.has_full_calibration():
                print("Mouse control requires full calibration.")
            elif calib_active:
                print("Mouse control disabled during calibration.")
            else:
                mouse_enabled = not mouse_enabled
                if mouse_enabled:
                    cx, cy = get_cursor_pos()
                    sx = float(cx)
                    sy = float(cy)
                    vx_c = 0.0
                    vy_c = 0.0
                print(f"Mouse control {'enabled' if mouse_enabled else 'disabled'}.")
        if key == ord("p") or key == ord("P"):
            mouse_paused = not mouse_paused
        if key == ord("["):
            edge_gain = float(np.clip(edge_gain + 0.02, 0.0, 0.5))
            print(f"edge_gain={edge_gain:.2f}")
        if key == ord("]"):
            edge_gain = float(np.clip(edge_gain - 0.02, 0.0, 0.5))
            print(f"edge_gain={edge_gain:.2f}")
        if key == ord(","):
            spring_k = float(np.clip(spring_k - 5.0, 20.0, 120.0))
            print(f"spring_k={spring_k:.0f}")
        if key == ord("."):
            spring_k = float(np.clip(spring_k + 5.0, 20.0, 120.0))
            print(f"spring_k={spring_k:.0f}")
        if key == ord("i"):
            y_scale = float(np.clip(y_scale + 0.05, 0.6, 2.5))
            print(f"y_scale={y_scale:.2f}")
        if key == ord("o"):
            y_offset = float(np.clip(y_offset + 0.01, -0.25, 0.25))
            print(f"y_offset={y_offset:.2f}")
        if key == ord("y"):
            y_flip = not y_flip
            print(f"y_flip={y_flip}")
        if key == ord("9"):
            y_edge_gain = float(np.clip(y_edge_gain + 0.02, 0.0, 0.5))
            print(f"y_edge_gain={y_edge_gain:.2f}")
        if key == ord("0"):
            y_edge_gain = float(np.clip(y_edge_gain - 0.02, 0.0, 0.5))
            print(f"y_edge_gain={y_edge_gain:.2f}")
        if key == ord("t"):
            test_active = True
            test_start = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
