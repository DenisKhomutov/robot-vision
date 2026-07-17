import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = {
    "july": "yolo_train/data/video_2026-07-08_14-48-31.mp4",
    "june": "yolo_train/data/video_2026-06-04_14-29-40.mp4",
}
FPS = 3.0
PROGRESS_EVERY = 10
SEQ_OVERLAP = 10
VOCAB_TREE = ROOT / "data" / "vocab_tree_flickr100K_words32K.bin"
VOCAB_URL = "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract(video, out_dir, fps, blur_thresh):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        log(f"ОШИБКА: не открывается {video}")
        return 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / fps))
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"нарезка {video.name}: {total} кадров @ {src_fps:.1f} fps -> каждый {step}-й (~{fps} fps)")
    kept = skipped_blur = idx = 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if sharp < blur_thresh:
                skipped_blur += 1
            else:
                cv2.imwrite(str(out_dir / f"{kept:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                kept += 1
                if kept % PROGRESS_EVERY == 0:
                    done = idx / max(total, 1)
                    el = time.perf_counter() - t0
                    eta = el / max(done, 1e-6) - el
                    log(f"  сохранено {kept:4d} | видео {100 * done:5.1f}% | смазанных откинуто {skipped_blur} "
                        f"| прошло {el:5.1f}с | осталось ~{eta:5.1f}с")
        idx += 1
    cap.release()
    log(f"нарезка готова: {kept} кадров -> {out_dir}  (смазанных откинуто: {skipped_blur})")
    return kept


def have(binary):
    return shutil.which(binary) is not None


def run(cmd, tag):
    log(f"--- {tag}")
    log(f"    {' '.join(str(c) for c in cmd)}")
    t0 = time.perf_counter()
    p = subprocess.run(cmd)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        log(f"ОШИБКА на шаге '{tag}' (код {p.returncode}), прошло {dt:.0f}с")
        return False
    log(f"--- {tag}: готово за {dt:.0f}с")
    return True


def fetch_vocab_tree():
    if VOCAB_TREE.exists():
        return True
    log(f"качаю vocab tree (~200МБ) -> {VOCAB_TREE}")
    VOCAB_TREE.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["curl", "-L", "-o", str(VOCAB_TREE), VOCAB_URL], check=True)
        return True
    except Exception as e:
        log(f"не скачался vocab tree: {e}. Петли искаться не будут.")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="july", choices=list(VIDEOS))
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--blur", type=float, default=40.0)
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--overlap", type=int, default=SEQ_OVERLAP)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    video = Path(VIDEOS[args.video])
    if not video.exists():
        log(f"ОШИБКА: нет видео {video}")
        return 1

    tag = args.out or args.video
    work = ROOT / "maps" / tag
    images = ROOT / "data" / tag
    db = work / "database.db"
    sparse = work / "sparse"

    if not args.skip_extract:
        if extract(video, images, args.fps, args.blur) == 0:
            return 1
    n = len(list(images.glob("*.jpg")))
    log(f"кадров на входе SfM: {n}")
    if args.extract_only:
        return 0

    missing = [b for b in ("colmap", "glomap") if not have(b)]
    if missing:
        log(f"нет бинарей: {', '.join(missing)}")
        log("  colmap: sudo apt-get install -y colmap")
        log("  glomap: на pypi нет, собирать из github.com/colmap/glomap")
        return 1

    work.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)

    if not run(["colmap", "feature_extractor",
                "--database_path", db,
                "--image_path", images,
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "OPENCV",
                "--SiftExtraction.use_gpu", "1"], "feature_extractor (SIFT)"):
        return 1

    match = ["colmap", "sequential_matcher",
             "--database_path", db,
             "--SequentialMatching.overlap", str(args.overlap),
             "--SiftMatching.use_gpu", "1"]
    if fetch_vocab_tree():
        match += ["--SequentialMatching.loop_detection", "1",
                  "--SequentialMatching.vocab_tree_path", str(VOCAB_TREE)]
    if not run(match, "sequential_matcher"):
        return 1

    if not run(["glomap", "mapper",
                "--database_path", db,
                "--image_path", images,
                "--output_path", sparse], "glomap mapper"):
        return 1

    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    log(f"моделей собрано: {len(models)}")
    for m in models:
        if not run(["colmap", "model_analyzer", "--path", m], f"model_analyzer {m.name}"):
            continue
    log(f"смотреть: colmap gui --import_path {sparse}/0 --database_path {db} --image_path {images}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
