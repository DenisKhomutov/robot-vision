"""
Прогон 2D-карты + расстояний до краёв по видео (аналог predict_video, но с BEV).

Каждый кадр -> панель как у 21_edge.jpg: слева оригинал, посередине BEV (фото сверху),
справа размеченный BEV (классы + осевая + L/R/offset/width). Пишет mp4.

CPU: undistort + depth + сегментация ~5 с/кадр, поэтому кадры прореживаются (STRIDE).

Запуск:
    uv run python experiment/edge_video.py
    uv run python experiment/edge_video.py yolo_train/data/video_2026-06-04_14-29-40.mp4
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from depth_model import load_model
from edge_distance import INTRINSICS, process

VIDEO = "yolo_train/data/video_2026-07-08_14-48-31.mp4"
OUT = "experiment/edge_out"
STRIDE = 15  # обрабатывать каждый N-й кадр
MAX_FRAMES = 40  # максимум обработанных кадров


def main() -> None:
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else STRIDE
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_FRAMES
    if not Path(video).exists():
        print(f"[ERROR] нет видео: {video}")
        return

    intr = np.load(INTRINSICS)
    k, dist, calib = intr["K"], intr["dist"], tuple(int(x) for x in intr["image_size"])
    processor, model, device = load_model()

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    Path(OUT).mkdir(parents=True, exist_ok=True)
    out_path = f"{OUT}/{Path(video).stem}_edge.mp4"
    writer = None

    read = done = 0
    while done < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if read % stride:
            read += 1
            continue
        read += 1
        if (frame.shape[1], frame.shape[0]) != calib:  # под размер калибровки
            frame = cv2.resize(frame, calib)

        panel, _ = process(frame, k, dist, processor, model, device)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, max(1.0, fps / stride), (panel.shape[1], panel.shape[0]))
        writer.write(panel)
        done += 1
        print(f"[INFO] кадр {done}/{max_frames}")

    if writer is not None:
        writer.release()
    cap.release()
    print(f"[INFO] готово -> {out_path}")


if __name__ == "__main__":
    main()
