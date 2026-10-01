"""Capture one 1280x720 image for camera calibration."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "calibration" / "images"
IMAGE_NAME = re.compile(r"^calib_(\d{6})\.jpg$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one numbered calibration image")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup-frames", type=int, default=30)
    return parser.parse_args()


def next_path(output: Path) -> tuple[int, Path]:
    last_id = 0
    for path in output.glob("calib_*.jpg"):
        match = IMAGE_NAME.fullmatch(path.name)
        if match:
            last_id = max(last_id, int(match.group(1)))
    image_id = last_id + 1
    return image_id, output / f"calib_{image_id:06d}.jpg"


def main() -> int:
    args = parse_args()
    if args.warmup_frames < 0:
        raise SystemExit("--warmup-frames must be non-negative")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source: int | str = int(args.device) if args.device.isdecimal() else args.device
    camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise SystemExit(f"Cannot open camera {args.device}")

    try:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame = None
        for _ in range(args.warmup_frames + 1):
            ok, frame = camera.read()
            if not ok or frame is None:
                raise SystemExit(f"Cannot read from camera {args.device}")

        height, width = frame.shape[:2]
        if (width, height) != (1280, 720):
            raise SystemExit(f"Camera returned {width}x{height}; expected native 1280x720. Nothing saved.")

        image_id, path = next_path(output)
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise SystemExit(f"Cannot save {path}")
    finally:
        camera.release()

    print(f"Saved calibration image id={image_id}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
