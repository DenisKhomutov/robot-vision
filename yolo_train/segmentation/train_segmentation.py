"""
Обучение сегментации поверхностей (YOLO11s-seg).
Классы: sidewalk / road / crosswalk / non_driveable.

Старт со стоковых COCO-весов (yolo11s-seg.pt) — transfer learning.
Претрен на публичном урбан-датасете (Mapillary Vistas), затем дообучение на реальных.

Запуск из корня проекта:
    uv run python yolo_train/segmentation/train_segmentation.py

⚠️ Нужна GPU. И укажи DATA — сейчас это заглушка.
"""

from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ЗАГЛУШКА: подставь путь к data.yaml сегментации (4 класса, YOLO-seg формат)
DATA = "PATH/TO/segmentation/data.yaml"
# абсолютный путь вывода: иначе ultralytics шлёт результат в свой глобальный runs_dir
OUT_PROJECT = str(PROJECT_ROOT / "yolo_train" / "segmentation" / "runs")


def main() -> None:
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] CUDA недоступна — сегментацию на CPU обучать нереально. Нужна GPU.")

    model = YOLO("yolo11s-seg.pt")  # small, COCO-претрен

    model.train(
        # данные / вывод
        data=DATA,
        project=OUT_PROJECT,
        name="surfaces_yolo11s_seg",
        exist_ok=False,
        # расписание
        epochs=100,
        patience=20,
        batch=8,  # seg тяжелее детекции; при OOM снижай, при запасе VRAM подними или -1
        imgsz=768,  # поверхности крупные, 768 хватает; тонкие границы/зебра — можно 1024
        # производительность
        device=device,
        workers=8,
        amp=True,
        cache="disk",
        cos_lr=True,
        # оптимизатор / LR
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # аугментации: цвет/свет (день/ночь/погода)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        # аугментации: геометрия
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,  # камера смотрит вперёд: верх=небо, низ=земля — вертикальный флип бессмыслен
        fliplr=0.5,
        # аугментации: композиция
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,  # для семантических поверхностей не нужен
    )

    print(f"[INFO] Готово. Веса: {OUT_PROJECT}/surfaces_yolo11s_seg/weights/best.pt")


if __name__ == "__main__":
    main()
