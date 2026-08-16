import argparse
import json
import logging
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
LOGGER = logging.getLogger("extract_adaptive")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")


def log(message: str) -> None:
    LOGGER.info(message)


def grid_points(w: int, h: int, n: int = GRID) -> np.ndarray:
    xs = np.linspace(w * 0.1, w * 0.9, n)
    ys = np.linspace(h * 0.1, h * 0.9, n)
    return np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 1, 2).astype(np.float32)


def displacement(prev_gray: np.ndarray, gray: np.ndarray, pts: np.ndarray) -> float | None:
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK)
    if nxt is None or st is None:
        return None
    back, st_back, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **LK)
    if back is None or st_back is None:
        return None
    fb_error = np.linalg.norm(back - pts, axis=2).ravel()
    ok = (st.ravel() == 1) & (st_back.ravel() == 1) & (fb_error < 1.5)
    if ok.sum() < 20:
        return None
    d = np.linalg.norm(nxt[ok] - pts[ok], axis=2).ravel()
    return float(np.median(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="video1|video2 или путь к файлу")
    ap.add_argument("--shift", type=float, default=22.0,
                    help="целевой сдвиг картинки между соседними кадрами, пикселей")
    ap.add_argument("--min-gap", type=int, default=2, help="не чаще, чем каждый N-й кадр видео")
    ap.add_argument("--max-gap", type=int, default=15, help="не реже, чем каждый N-й")
    ap.add_argument("--fixed-gap", type=int, default=None,
                    help="фиксированный шаг по исходному видео; отключает учащение на поворотах. "
                         "Для уличной SfM рекомендуется 8")
    ap.add_argument("--min-move", type=float, default=2.0,
                    help="сдвиг картинки (px при 1280) ниже которого робот считается СТОЯЩИМ -> "
                         "кадр не берём (иначе стоянка копит нулевые базы и схлопывает карту)")
    ap.add_argument("--out", default=None, help="по умолчанию frames_4f_adaptive_<v1|v2>")
    ap.add_argument("--overwrite", action="store_true", help="явно заменить существующий каталог кадров")
    args = ap.parse_args()

    if args.fixed_gap is not None:
        if args.fixed_gap < 1:
            log("ОШИБКА: --fixed-gap должен быть >= 1")
            return 2
        args.min_gap = args.fixed_gap
        args.max_gap = args.fixed_gap
        # В фиксированном режиме порог shift не должен вызывать раннее сохранение.
        args.shift = 1_000_000.0

    if args.min_gap < 1 or args.max_gap < args.min_gap:
        log("ОШИБКА: нужно 1 <= min-gap <= max-gap")
        return 2
    if args.shift <= 0 or args.min_move < 0 or args.min_move >= args.shift:
        log("ОШИБКА: нужно 0 <= min-move < shift")
        return 2
    src = (ROOT.parent / VIDEOS[args.video]) if args.video in VIDEOS else Path(args.video)
    if not src.exists():
        log(f"ОШИБКА: нет {src}")
        return 1
    if args.out:
        output_name = args.out
    elif args.video in VIDEOS:
        output_name = f"frames_4f_adaptive_{args.video.replace('video', 'v')}"
    else:
        output_name = f"frames_adaptive_{Path(args.video).stem}"
    if Path(output_name).name != output_name:
        log("ОШИБКА: --out должен быть именем каталога внутри module2_localization/data")
        return 2
    out = ROOT / "data" / output_name
    probe = cv2.VideoCapture(str(src))
    source_width = probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280
    probe.release()
    shift = args.shift * max(source_width, 1) / 1280.0
    min_move = args.min_move * max(source_width, 1) / 1280.0
    if out.exists():
        if not args.overwrite:
            log(f"ОШИБКА: {out} уже существует; задайте другой --out или добавьте --overwrite")
            return 2
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cap = cv2.VideoCapture(str(src))
    total_value = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total = int(total_value) if 0 < total_value < 1_000_000_000 else 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok:
        log("не читается видео")
        return 1
    h, w = frame.shape[:2]
    pts = grid_points(w, h)
    log(f"адаптивная нарезка {src.name}: {total} кадров {w}x{h} @ {fps:.0f} fps")
    if args.fixed_gap is not None:
        log(f"фиксированная нарезка: шаг {args.fixed_gap} кадров видео, "
            f"стояние < {min_move:.1f}px исключается")
    else:
        log(f"режу при сдвиге >= {shift:.1f}px (порог {args.shift:.0f} для 1280), "
            f"шаг {args.min_gap}-{args.max_gap}")

    if not cv2.imwrite(str(out / "000000.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        log("ОШИБКА: не удалось записать первый кадр")
        return 1
    anchor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kept, idx, gap = 1, 0, 0
    gaps, shifts = [], []
    source_indices = [0]
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
        if d is not None and d < min_move and gap >= args.max_gap:
            # Якорь сохраняем: медленное движение должно накопиться, а не исчезнуть
            # из-за периодического сброса каждые max-gap кадров.
            continue
        if d is None or d >= shift or gap >= args.max_gap:
            if not cv2.imwrite(str(out / f"{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                log(f"ОШИБКА: не удалось записать исходный кадр {idx}")
                cap.release()
                return 1
            gaps.append(gap)
            shifts.append(d if d is not None else -1)
            source_indices.append(idx)
            anchor, kept, gap = gray, kept + 1, 0
            if kept % 25 == 0:
                el = time.perf_counter() - t0
                progress = f"видео {100*idx/total:5.1f}%" if total else f"исх. кадр {idx}"
                log(f"  {kept:4d} кадров | {progress} | шаг сейчас {gaps[-1]:2d} "
                    f"| сдвиг {shifts[-1]:5.1f}px | {el:5.0f}с")
    cap.release()

    g = np.array(gaps)
    log(f"нарезано {kept} кадров -> {out}")
    if len(g):
        log(f"шаг по видео: медиана {int(np.median(g))}, мин {g.min()}, макс {g.max()}")
        log(f"эквивалент fps: медиана {fps/np.median(g):.1f}, в поворотах до {fps/g.min():.1f}, "
            f"на прямых от {fps/g.max():.1f}")
    (out / "extraction.json").write_text(json.dumps({
        "source": str(src.resolve()),
        "source_fps": fps,
        "source_frames": source_indices,
        "shift_px_1280": args.shift,
        "min_move_px_1280": args.min_move,
        "min_gap": args.min_gap,
        "max_gap": args.max_gap,
        "fixed_gap": args.fixed_gap,
    }, ensure_ascii=False, indent=2))
    log(f"для сравнения, равномерные 3 fps дали бы {total//10} кадров")
    return 0


if __name__ == "__main__":
    sys.exit(main())
