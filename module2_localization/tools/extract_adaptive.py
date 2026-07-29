import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = {
    "video1": "yolo_train/data/video_2026-06-04_14-29-40.mp4",
    "video2": "yolo_train/data/video_2026-07-08_14-48-31.mp4",
}
GRID = 24
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def grid_points(w, h, n=GRID):
    xs = np.linspace(w * 0.1, w * 0.9, n)
    ys = np.linspace(h * 0.1, h * 0.9, n)
    return np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 1, 2).astype(np.float32)


def displacement(prev_gray, gray, pts):
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK)
    ok = st.ravel() == 1
    if ok.sum() < 20:
        return None
    d = np.linalg.norm(nxt[ok] - pts[ok], axis=2).ravel()
    return float(np.median(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="video1|video2 или путь к файлу")
    ap.add_argument("--shift", type=float, default=22.0,
                    help="целевой сдвиг картинки между соседними кадрами, пикселей")
    ap.add_argument("--min-gap", type=int, default=2, help="не чаще, чем каждый N-й кадр видео")
    ap.add_argument("--max-gap", type=int, default=15, help="не реже, чем каждый N-й")
    ap.add_argument("--out", default=None, help="по умолчанию frames_4f_adaptive_<v1|v2>")
    args = ap.parse_args()

    src = (ROOT.parent / VIDEOS[args.video]) if args.video in VIDEOS else Path(args.video)
    if not src.exists():
        log(f"ОШИБКА: нет {src}")
        return 1
    out = ROOT / "data" / (args.out or f"frames_4f_adaptive_{args.video.replace('video', 'v')}")
    probe = cv2.VideoCapture(str(src)); _w = probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280; probe.release()
    shift = args.shift * max(_w, 1) / 1280.0
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok:
        log("не читается видео")
        return 1
    h, w = frame.shape[:2]
    pts = grid_points(w, h)
    log(f"адаптивная нарезка {src.name}: {total} кадров {w}x{h} @ {fps:.0f} fps")
    log(f"режу при сдвиге >= {shift:.1f}px (порог {args.shift:.0f} для 1280), шаг {args.min_gap}-{args.max_gap}")

    cv2.imwrite(str(out / "00000.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    anchor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kept, idx, gap = 1, 0, 0
    gaps, shifts = [], []
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        gap += 1
        if gap < args.min_gap:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        d = displacement(anchor, gray, pts)
        if d is None or d >= shift or gap >= args.max_gap:
            cv2.imwrite(str(out / f"{kept:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            gaps.append(gap)
            shifts.append(d if d is not None else -1)
            anchor, kept, gap = gray, kept + 1, 0
            if kept % 25 == 0:
                el = time.perf_counter() - t0
                log(f"  {kept:4d} кадров | видео {100*idx/total:5.1f}% | шаг сейчас {gaps[-1]:2d} "
                    f"| сдвиг {shifts[-1]:5.1f}px | {el:5.0f}с")
    cap.release()

    g = np.array(gaps)
    log(f"нарезано {kept} кадров -> {out}")
    log(f"шаг по видео: медиана {int(np.median(g))}, мин {g.min()}, макс {g.max()}")
    log(f"эквивалент fps: медиана {fps/np.median(g):.1f}, в поворотах до {fps/g.min():.1f}, "
        f"на прямых от {fps/g.max():.1f}")
    log(f"для сравнения, равномерные 3 fps дали бы {total//10} кадров")
    return 0


if __name__ == "__main__":
    sys.exit(main())
