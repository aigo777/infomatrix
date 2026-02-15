# Run:
#   python test_uia_targets.py

from __future__ import annotations

import time

from uia_targets import TargetRegistry


def main() -> None:
    registry = TargetRegistry()
    last_hwnd: int | None = None
    last_title: str | None = None

    print("Press Ctrl+C to exit")
    while True:
        info = registry.refresh()
        hwnd = info.get("hwnd")
        pid = info.get("pid")
        raw_title = info.get("title")
        title = raw_title if isinstance(raw_title, str) and raw_title else "<no title>"

        if hwnd != last_hwnd or title != last_title:
            print(f"hwnd={hwnd} pid={pid} title={title}")
            last_hwnd = hwnd if isinstance(hwnd, int) else None
            last_title = title

        time.sleep(1.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
