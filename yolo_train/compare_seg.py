"""
Сравнение 3 сегментаторов на реальных кадрах.

Берёт N случайных кадров (разный шаг) из FRAMES_DIR → копирует в EXP_DIR,
затем прогоняет каждую модель из WEIGHTS_DIR → визуализации в RESULT_DIR/<модель>/.

Запуск из корня проекта:
    uv run python yolo_train/compare_seg.py
"""

import random
import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = PROJECT_ROOT / "yolo_train" / "data" / "frames"
EXP_DIR = PROJECT_ROOT / "yolo_train" / "exp"
RESULT_DIR = PROJECT_ROOT / "yolo_train" / "exp_result"
WEIGHTS_DIR = PROJECT_ROOT / "yolo_train" / "weights"

N = 20
IMGSZ = 768


def main() -> None:
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"Нет кадров в {FRAMES_DIR} — сначала extract_frames.py")

    # НЕ чистим exp/exp_result — добавляем. Берём только ещё не прогнанные кадры.
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    already = {p.name for p in EXP_DIR.glob("*.jpg")}
    candidates = [p for p in frames if p.name not in already]
    if not candidates:
        raise RuntimeError("Все кадры уже прогнаны — новых нет.")

    picked = random.sample(candidates, min(N, len(candidates)))  # random.sample = разный шаг
    for p in picked:
        shutil.copy(p, EXP_DIR / p.name)
    print(f"[INFO] добавлено {len(picked)} новых кадров → {EXP_DIR}")

    new_sources = [str(EXP_DIR / p.name) for p in picked]  # прогоняем ТОЛЬКО новые
    models = sorted(WEIGHTS_DIR.glob("*.pt"))
    for mp in models:
        model = YOLO(str(mp))
        model.predict(
            source=new_sources,
            imgsz=IMGSZ,
            save=True,
            project=str(RESULT_DIR),  # абсолютный путь — не улетит в чужой runs_dir
            name=mp.stem,
            exist_ok=True,  # добавляем в существующую папку, старое не трогаем
            verbose=False,
        )
        print(f"[INFO] {mp.stem}: +{len(picked)} → {RESULT_DIR / mp.stem}")

    print(f"[INFO] Смотри результаты: {RESULT_DIR}")


if __name__ == "__main__":
    main()
