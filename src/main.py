from __future__ import annotations
a=0
import os
import time
from collections import deque
import math
import cv2
import numpy as np
from gaze_tracker import GazeTracker


def main() -> None:
    print("IrisKeys - Stage 3.0")
    print(
        "Controls: q quit | esc emergency stop | m toggle mouse | space pause | s screenshot | k calibrate | r reset | l load calibration | t test | [/] edge_gain | 9/0 y_edge_gain | i/k y_scale | o/l y_offset | y flip | ,/. spring_k"
    )

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

    calib_dir = os.path.join(os.getcwd(), "calibration")
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
    drift_max = 0.04
    y_scale = 1.35
    y_offset = 0.0
    y_flip = False
    y_edge_gain = 0.18

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

    def transform_gaze(gaze: tuple[float, float]) -> tuple[float, float]:
        gx = clamp01(gaze[0])
        gy = gaze[1]
        if y_flip:
            gy = 1.0 - gy
        gy = (gy - 0.5) * y_scale + 0.5 + y_offset
        gy = clamp01(gy)

        gx = mid_edge_expand(gx, gy, strength=0.22)
        gy = mid_edge_expand(gy, gx, strength=0.10)

        gx = soft_edge_curve(gx, edge_gain)
        gy = soft_edge_curve(gy, y_edge_gain)
        gy = vertical_extreme_damp(gy, strength=0.35)
        return gx, gy

    def get_cursor_pos() -> tuple[int, int]:
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)

    def set_cursor_pos(x_px: int, y_px: int) -> None:
        user32.SetCursorPos(int(x_px), int(y_px))

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
            if mouse_enabled:
                drift_rate = 0.002
                drift_x = float(
                    np.clip(drift_x + (gaze_pointer[0] - 0.5) * drift_rate, -drift_max, drift_max)
                )
                drift_y = float(
                    np.clip(drift_y + (gaze_pointer[1] - 0.5) * drift_rate, -drift_max, drift_max)
                )
            else:
                drift_x *= 0.995
                drift_y *= 0.995

        display_gaze = None
        target_x = None
        target_y = None
        if isinstance(gaze_pointer, tuple) and vw is not None and vh is not None:
            display_gaze = transform_gaze(gaze_pointer)
            dx = clamp01(display_gaze[0] + drift_x)
            dy = clamp01(display_gaze[1] + drift_y)
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
            desktop_text,
            (10, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow(debug_win, debug_frame)

        pointer_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        if isinstance(display_gaze, tuple):
            px = int(np.clip(display_gaze[0], 0.0, 1.0) * (screen_w - 1))
            py = int(np.clip(display_gaze[1], 0.0, 1.0) * (screen_h - 1))
            cv2.circle(pointer_frame, (px, py), 22, (255, 0, 0), -1)
            label = f"{display_gaze[0]:.2f},{display_gaze[1]:.2f}"
            cv2.putText(
                pointer_frame,
                label,
                (min(px + 24, screen_w - 140), max(py - 24, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        flip_text = f"flipX={tracker.axis_flip_x} flipY={tracker.axis_flip_y}"
        cv2.putText(
            pointer_frame,
            flip_text,
            (20, screen_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        alpha_text = f"spring_k={spring_k:.0f} edge={edge_gain:.2f}"
        cv2.putText(
            pointer_frame,
            alpha_text,
            (screen_w - 160, screen_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        open_val = f"{eye_open:.3f}" if isinstance(eye_open, float) else "None"
        ref_val = f"{tracker._open_ref:.3f}" if isinstance(tracker._open_ref, float) else "None"
        open_text = f"open={open_val} ref={ref_val} beta_y={tracker._open_beta_y:.4f}"
        y_text = f"y_scale={y_scale:.2f} y_off={y_offset:.2f} y_edge={y_edge_gain:.2f} flip={y_flip}"
        cv2.putText(
            pointer_frame,
            y_text,
            (20, screen_h - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            pointer_frame,
            open_text,
            (20, screen_h - 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        status_text = f"mouse={'ON' if mouse_enabled else 'OFF'} {cursor_backend}"
        cv2.putText(
            pointer_frame,
            status_text,
            (20, screen_h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        desktop_text = f"desktop=({vx},{vy},{vw},{vh})"
        target_text = f"target=({target_x},{target_y})"
        cv2.putText(
            pointer_frame,
            desktop_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            pointer_frame,
            target_text,
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

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

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 27:
            mouse_enabled = False
            print("Emergency stop.")
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
        if key == ord(" "):
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
