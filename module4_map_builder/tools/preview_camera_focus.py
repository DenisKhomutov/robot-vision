"""Preview a V4L2 camera and adjust its UVC focus interactively."""

from __future__ import annotations

import argparse
import subprocess

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview camera and tune focus")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--focus-step", type=int, default=10)
    return parser.parse_args()


def v4l2_get(device: str, control: str) -> int | None:
    result = subprocess.run(
        ["v4l2-ctl", "-d", device, f"--get-ctrl={control}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.rsplit(":", 1)[1].strip())
    except (IndexError, ValueError):
        return None


def v4l2_set(device: str, control: str, value: int) -> bool:
    result = subprocess.run(
        ["v4l2-ctl", "-d", device, f"--set-ctrl={control}={value}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr.strip() or f"Failed to set {control}")
        return False
    return True


def main() -> int:
    args = parse_args()
    source: int | str = int(args.device) if args.device.isdecimal() else args.device
    camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise SystemExit(f"Cannot open camera {args.device}")

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    autofocus = v4l2_get(args.device, "focus_automatic_continuous")
    focus = v4l2_get(args.device, "focus_absolute") or 0
    window = "Camera focus | A auto | -/+ manual focus | Q quit"

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise SystemExit("Cannot read camera frame")

            mode = "AUTO" if autofocus else "MANUAL"
            text = f"1280x720  focus={mode}:{focus}   A: auto   -/+: focus   Q: quit"
            cv2.rectangle(frame, (0, 0), (1280, 42), (0, 0, 0), -1)
            cv2.putText(frame, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("a"):
                autofocus = 0 if autofocus else 1
                v4l2_set(args.device, "focus_automatic_continuous", autofocus)
                focus = v4l2_get(args.device, "focus_absolute") or focus
            elif key in (ord("-"), ord("[")):
                if autofocus:
                    autofocus = 0
                    v4l2_set(args.device, "focus_automatic_continuous", 0)
                focus = max(0, focus - args.focus_step)
                v4l2_set(args.device, "focus_absolute", focus)
            elif key in (ord("+"), ord("="), ord("]")):
                if autofocus:
                    autofocus = 0
                    v4l2_set(args.device, "focus_automatic_continuous", 0)
                focus = min(1023, focus + args.focus_step)
                v4l2_set(args.device, "focus_absolute", focus)
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Final focus mode: {'auto' if autofocus else 'manual'}, focus_absolute={focus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
