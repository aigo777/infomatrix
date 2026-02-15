# python test_uia_attach.py

from __future__ import annotations

import time

from uia_targets import TargetRegistry


def _safe_title(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    return "<no title>"


def _attach_signature(result: dict) -> tuple[bool, str, str, tuple[int, int, int, int] | None]:
    ok = bool(result.get("ok"))
    error = str(result.get("error") or "")
    title = str(result.get("uia_title") or "")
    rect_obj = result.get("rect")
    rect: tuple[int, int, int, int] | None = None
    if isinstance(rect_obj, tuple) and len(rect_obj) == 4:
        try:
            rect = (int(rect_obj[0]), int(rect_obj[1]), int(rect_obj[2]), int(rect_obj[3]))
        except Exception:
            rect = None
    return (ok, error, title, rect)


def main() -> None:
    registry = TargetRegistry()
    last_hwnd: int | None = None
    last_attach_sig: tuple[bool, str, str, tuple[int, int, int, int] | None] | None = None
    attach_result: dict = {"ok": False, "error": "no_hwnd"}

    print("Ctrl+C to exit")
    while True:
        info = registry.refresh()
        hwnd_obj = info.get("hwnd")
        hwnd = int(hwnd_obj) if isinstance(hwnd_obj, int) else None
        pid_obj = info.get("pid")
        pid = int(pid_obj) if isinstance(pid_obj, int) else None
        ctypes_title = _safe_title(info.get("title"))

        hwnd_changed = hwnd != last_hwnd
        if hwnd_changed:
            attach_result = registry.attach_uia(hwnd if hwnd is not None else 0)
            last_hwnd = hwnd

        attach_sig = _attach_signature(attach_result)
        if hwnd_changed or attach_sig != last_attach_sig:
            ok = attach_sig[0]
            uia_title = attach_result.get("uia_title") if ok else "<n/a>"
            rect = attach_sig[3] if ok else "<n/a>"
            error = attach_result.get("error", "<n/a>") if not ok else "<n/a>"

            print(f"[HWND] {hwnd} PID={pid}")
            print(f"ctypes_title: {ctypes_title}")
            print(f"uia_ok: {ok}")
            print(f"uia_title: {uia_title if uia_title else '<n/a>'}")
            print(f"rect: {rect}")
            print(f"error: {error}")
            print()

            last_attach_sig = attach_sig

        time.sleep(1.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
