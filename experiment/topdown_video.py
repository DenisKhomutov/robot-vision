"""
Топ-даун карта сверху на видео (аналог yolo_train/segmentation/predict_video.py).

Покадрово: depth + сегментация + IPM-варп -> вид сверху. Выводит mp4:
слева исходный кадр, справа карта классов сверху.

CPU: depth+seg ~5 с/кадр, поэтому кадры прореживаются (STRIDE) и ограничены (MAX_FRAMES).

Запуск:
    uv run python experiment/topdown_video.py
    uv run python experiment/topdown_video.py yolo_train/data/video_2026-07-08_14-48-31.mp4
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from depth_model import infer_depth, load_model
from topdown_map import (
    HFOV_DEG,
    COLORS,
    backproject,
    draw_rings,
    fit_ground_plane,
    ground_homography,
    metric_to_px,
    segment,
)

VIDEO = "yolo_train/data/video_2026-06-04_14-29-40.mp4"
OUT = "experiment/topdown_out"
STRIDE = 15  # обрабатывать каждый N-й кадр
MAX_FRAMES = 40  # максимум обработанных кадров


def frame_topdown(frame_bgr, processor, model, device):
    """Кадр BGR -> вид сверху (классы)."""
    depth = infer_depth(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)), processor, model, device)
    cls_map = segment(frame_bgr)
    h, w = depth.shape
    if cls_map.shape != (h, w):
        cls_map = cv2.resize(cls_map, (w, h), interpolation=cv2.INTER_NEAREST)

    pts = backproject(depth, HFOV_DEG)
    u = np.tile(np.arange(w, dtype=np.float32), h)
    v = np.repeat(np.arange(h, dtype=np.float32), w)
    lower = (v > 0.55 * h) & np.isfinite(pts).all(axis=1)
    centroid, up = fit_ground_plane(pts, lower)

    A, out_w, out_h = metric_to_px()
    H_out = A @ ground_homography(pts, u, v, centroid, up)
    td_seg = cv2.warpPerspective(COLORS[cls_map], H_out, (out_w, out_h), flags=cv2.INTER_NEAREST)
    draw_rings(td_seg)
    return td_seg


def main() -> None:
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else STRIDE
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_FRAMES
    if not Path(video).exists():
        print(f"[ERROR] нет видео: {video}")
        return

    processor, model, device = load_model()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    Path(OUT).mkdir(parents=True, exist_ok=True)
    out_path = f"{OUT}/{Path(video).stem}_topdown.mp4"
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

        td = frame_topdown(frame, processor, model, device)
        scale = td.shape[0] / frame.shape[0]
        left = cv2.resize(frame, (int(frame.shape[1] * scale), td.shape[0]))
        panel = np.hstack([left, td])
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
