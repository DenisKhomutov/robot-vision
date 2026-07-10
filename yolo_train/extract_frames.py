"""
Нарезка видео на кадры для датасета.

Берёт все *.mp4 из VIDEO_DIR, сохраняет каждый EVERY_SEC-й кадр в OUT_DIR.
Имя кадра = <имя_видео>_fNNNNNN.jpg — источник виден (важно для сплита по видео,
чтобы не поймать утечку) и индекс кадра прослеживается.

Запуск из корня проекта:
    uv run python yolo_train/extract_frames.py
"""

from pathlib import Path

import cv2

VIDEO_DIR = Path("yolo_train/data")
OUT_DIR = Path("yolo_train/data/frames")
EVERY_SEC = 1.0  # интервал между сохраняемыми кадрами (сек). 0.5 — плотнее, 2.0 — реже


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    for vpath in sorted(VIDEO_DIR.glob("*.mp4")):
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(fps * EVERY_SEC))
        stem = vpath.stem

        idx = saved = 0
        while True:
            ok, frame = cap.read()  # последовательное чтение надёжнее seek по кадрам
            if not ok:
                break
            if idx % step == 0:
                cv2.imwrite(str(OUT_DIR / f"{stem}_f{idx:06d}.jpg"), frame)
                saved += 1
            idx += 1
        cap.release()

        print(f"[INFO] {vpath.name}: сохранено {saved} кадров (шаг {step} кадров = {EVERY_SEC}с)")
        total += saved

    print(f"[INFO] ИТОГО: {total} кадров → {OUT_DIR}")


if __name__ == "__main__":
    main()
