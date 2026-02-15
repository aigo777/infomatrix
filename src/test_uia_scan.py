# python test_uia_scan.py

from __future__ import annotations

import time

from uia_targets import TargetRegistry


def _display_title(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    return "<no title>"


def main() -> None:
    registry = TargetRegistry(max_targets=200, visited_limit=2000, min_scan_interval_s=0.25)
    print("Ctrl+C to exit")

    while True:
        start = time.time()
        info = registry.refresh()
        dt_ms = (time.time() - start) * 1000.0

        hwnd = info.get("hwnd")
        pid = info.get("pid")
        title = _display_title(info.get("title"))
        targets = registry.get_targets()
        count = len(targets)

        cap_note = ""
        if count >= registry.max_targets:
            cap_note = f" (capped at {registry.max_targets})"

        print(f"[HWND] {hwnd} PID={pid}")
        print(f"title: {title}")
        print(f"targets: {count}{cap_note}")
        print(f"scan_ms: {dt_ms:.1f}")
        if count > registry.max_targets:
            print(f"warning: count exceeded cap ({count} > {registry.max_targets})")

        for target in targets[:10]:
            kind = target.get("kind", "")
            name = target.get("name", "")
            rect = target.get("rect", "")
            print(f"{kind} | {name} | {rect}")
        print()

        time.sleep(1.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
