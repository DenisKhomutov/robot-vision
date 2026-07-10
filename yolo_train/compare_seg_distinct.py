"""
Перекраска ТОЛЬКО пересекающихся классов (для презентации).
Оставляет оригинальную палитру ultralytics, меняет лишь классы, которые сливаются:
  - person (был как terrain, маджента) → коричневый
  - car    (был как vegetation, зелёный) → серый
Прогоняет все модели из weights на кадрах exp/ → exp_result_distinct/<модель>/.
Плюс легенда цвет→класс для yolo26s-sem.

Запуск из корня проекта:
    uv run python yolo_train/compare_seg_distinct.py
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import plotting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "yolo_train" / "exp"
OUT_DIR = PROJECT_ROOT / "yolo_train" / "exp_result_distinct"
WEIGHTS_DIR = PROJECT_ROOT / "yolo_train" / "weights"
IMGSZ = 768

# Только пересекающиеся классы Cityscapes: индекс -> новый RGB (остальное не трогаем)
OVERRIDES = {
    11: (139, 90, 43),   # person → коричневый (был как terrain)
    13: (128, 128, 128),  # car → серый (был как vegetation)
}


def patch_palette() -> list:
    """Берём оригинальную палитру и меняем только нужные индексы."""
    n = plotting.colors.n
    pal = [plotting.colors(i, bgr=False) for i in range(n)]
    for idx, rgb in OVERRIDES.items():
        if idx < n:
            pal[idx] = rgb
    plotting.colors.palette = [tuple(c) for c in pal]
    plotting.colors.n = len(pal)
    return pal


def save_legend(names: dict[int, str], palette: list, path: Path) -> None:
    step = 46
    img = np.full((step * len(names), 540, 3), 30, np.uint8)
    for i, name in names.items():
        r, g, b = palette[i % len(palette)]
        cv2.rectangle(img, (10, i * step + 8), (60, i * step + step - 8), (b, g, r), -1)
        cv2.putText(img, f"{i:2d} {name}", (75, i * step + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main() -> None:
    palette = patch_palette()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mp in sorted(WEIGHTS_DIR.glob("*.pt")):
        model = YOLO(str(mp))
        model.predict(
            source=str(EXP_DIR),
            imgsz=IMGSZ,
            save=True,
            project=str(OUT_DIR),
            name=mp.stem,
            exist_ok=True,
            verbose=False,
        )
        print(f"[INFO] {mp.stem}: готово")
        if len(model.names) == 19:
            save_legend(model.names, palette, OUT_DIR / f"_legend_{mp.stem}.jpg")

    print(f"[INFO] Результаты: {OUT_DIR}")


if __name__ == "__main__":
    main()
