from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

Point = Tuple[int, int]


@dataclass(frozen=True)
class MagnetismParams:
    assist_radius_factor: float = 2.2
    assist_k: float = 0.35
    assist_p: float = 1.8
    preferred_grace_factor: float = 1.25
    preferred_release_ms: int = 250
    speed_break_px_s: float = 1500.0
    speed_break_hold_ms: int = 150


@dataclass
class MagnetismState:
    preferred_target_id: Optional[str] = None
    preferred_far_since_ms: Optional[int] = None
    disabled_until_ms: int = 0


@dataclass
class MagnetismResult:
    target_px: Point
    target_id: Optional[str]
    strength: float
    state: MagnetismState


@dataclass(frozen=True)
class _TargetEntry:
    id: str
    center_px: Point
    radius_px: int
    weight: float
    distance: float
    influence_radius: float


def _copy_state(state: MagnetismState) -> MagnetismState:
    return MagnetismState(
        preferred_target_id=state.preferred_target_id,
        preferred_far_since_ms=state.preferred_far_since_ms,
        disabled_until_ms=int(state.disabled_until_ms),
    )


def _to_target_entry(raw_px: Point, target: object, params: MagnetismParams) -> Optional[_TargetEntry]:
    if isinstance(target, Mapping):
        tid = target.get("id")
        center = target.get("center_px")
        radius = target.get("radius_px")
        weight = target.get("weight", 1.0)
    else:
        tid = getattr(target, "id", None)
        center = getattr(target, "center_px", None)
        radius = getattr(target, "radius_px", None)
        weight = getattr(target, "weight", 1.0)

    if not isinstance(tid, str):
        return None
    if not isinstance(center, tuple) or len(center) != 2:
        return None
    try:
        cx = int(round(float(center[0])))
        cy = int(round(float(center[1])))
        radius_i = int(round(float(radius)))
        weight_f = float(weight)
    except (TypeError, ValueError):
        return None
    if radius_i <= 0 or weight_f <= 0.0:
        return None

    distance = math.hypot(float(raw_px[0] - cx), float(raw_px[1] - cy))
    influence_radius = float(params.assist_radius_factor) * float(radius_i)
    return _TargetEntry(
        id=tid,
        center_px=(cx, cy),
        radius_px=radius_i,
        weight=weight_f,
        distance=distance,
        influence_radius=influence_radius,
    )


def _blend_toward(raw_px: Point, target: _TargetEntry, params: MagnetismParams) -> Tuple[Point, float]:
    s = 1.0 - (target.distance / max(target.influence_radius, 1e-6))
    s = float(max(0.0, min(1.0, s)))
    s = s ** float(params.assist_p)
    if s < 0.05:
        return raw_px, 0.0

    alpha = float(params.assist_k) * s
    tx = raw_px[0] + (target.center_px[0] - raw_px[0]) * alpha
    ty = raw_px[1] + (target.center_px[1] - raw_px[1]) * alpha
    return (int(round(tx)), int(round(ty))), float(s)


def compute_magnetized_target_with_state(
    raw_px: Point,
    targets: Sequence[object],
    state: MagnetismState,
    now_ms: int,
    gaze_speed_px_s: float,
    params: MagnetismParams,
) -> MagnetismResult:
    now_i = int(now_ms)
    speed = float(gaze_speed_px_s)
    safe_state = _copy_state(state)
    raw_pt = (int(raw_px[0]), int(raw_px[1]))

    if speed >= float(params.speed_break_px_s):
        safe_state.disabled_until_ms = int(now_i + int(params.speed_break_hold_ms))
        safe_state.preferred_target_id = None
        safe_state.preferred_far_since_ms = None
        return MagnetismResult(target_px=raw_pt, target_id=None, strength=0.0, state=safe_state)

    if now_i < int(safe_state.disabled_until_ms):
        return MagnetismResult(target_px=raw_pt, target_id=None, strength=0.0, state=safe_state)

    entries = []
    by_id = {}
    for target in targets:
        entry = _to_target_entry(raw_pt, target, params)
        if entry is None:
            continue
        entries.append(entry)
        by_id[entry.id] = entry

    if not entries:
        return MagnetismResult(target_px=raw_pt, target_id=None, strength=0.0, state=safe_state)

    chosen: Optional[_TargetEntry] = None
    preferred_id = safe_state.preferred_target_id
    if preferred_id is not None:
        pref = by_id.get(preferred_id)
        if pref is None:
            safe_state.preferred_target_id = None
            safe_state.preferred_far_since_ms = None
        else:
            grace_radius = float(params.preferred_grace_factor) * pref.influence_radius
            if pref.distance <= grace_radius:
                chosen = pref
                safe_state.preferred_far_since_ms = None
            else:
                if safe_state.preferred_far_since_ms is None:
                    safe_state.preferred_far_since_ms = now_i
                elif now_i - int(safe_state.preferred_far_since_ms) >= int(params.preferred_release_ms):
                    safe_state.preferred_target_id = None
                    safe_state.preferred_far_since_ms = None

    if chosen is None:
        candidates = [entry for entry in entries if entry.distance <= entry.influence_radius]
        if candidates:
            candidates.sort(key=lambda entry: entry.distance / max(entry.weight, 1e-6))
            chosen = candidates[0]
            safe_state.preferred_target_id = chosen.id
            safe_state.preferred_far_since_ms = None

    if chosen is None:
        return MagnetismResult(target_px=raw_pt, target_id=None, strength=0.0, state=safe_state)

    target_px, strength = _blend_toward(raw_pt, chosen, params)
    if strength <= 0.0:
        return MagnetismResult(target_px=raw_pt, target_id=None, strength=0.0, state=safe_state)

    return MagnetismResult(target_px=target_px, target_id=chosen.id, strength=strength, state=safe_state)


def compute_magnetized_target(
    raw_px: Point,
    targets: Sequence[object],
    prev_target_id: Optional[str],
    now_ms: int,
    gaze_speed_px_s: float,
    params: MagnetismParams,
) -> Tuple[Point, Optional[str], float]:
    result = compute_magnetized_target_with_state(
        raw_px=raw_px,
        targets=targets,
        state=MagnetismState(preferred_target_id=prev_target_id),
        now_ms=now_ms,
        gaze_speed_px_s=gaze_speed_px_s,
        params=params,
    )
    return result.target_px, result.target_id, result.strength


__all__ = [
    "MagnetismParams",
    "MagnetismResult",
    "MagnetismState",
    "compute_magnetized_target",
    "compute_magnetized_target_with_state",
]
