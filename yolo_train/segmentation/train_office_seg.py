"""
Дообучение семантической сегментации офисного маршрута (allowed_path / obstacle).

База — yolo26m-sem.pt (претрейн Cityscapes, 19 классов: road/sidewalk/wall/person/...).
С нуля НЕ обучаем: на 381 кадре семантика с нуля не сойдётся, а тут нужные понятия
(пол/стена/человек) уже выучены — остаётся переложить их на наши 3 класса.

Датасет: yolo_train/data/office_sem_seg_train (сделан coco_to_semantic.py из
Roboflow COCO-экспорта). Классы: 0=background, 1=allowed_path, 2=obstacle.

Запуск из корня проекта (нужна GPU):
    uv run python yolo_train/segmentation/train_office_seg.py
Веса: yolo_train/segmentation/runs/office_sem/weights/best.pt
"""

from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL = str(PROJECT_ROOT / "experiment" / "weights" / "yolo26m-sem-ade20k.pt")  # ADE20K ближе к офису, чем Cityscapes
DATA = str(PROJECT_ROOT / "yolo_train" / "data" / "office_sem_seg_train" / "data.yaml")
OUT_PROJECT = str(PROJECT_ROOT / "yolo_train" / "segmentation" / "runs")

# freeze=0 -> заморожено 0 слоёв, т.е. учится вся сеть. Домен сильно другой
# (улица -> помещение), поэтому полный fine-tune оправдан. Если val начнёт
# расходиться (переобучение) — поставить 10 (заморозить backbone).
FREEZE = 0
IMGSZ = 960  # боевые кадры 1280x720; 960 — баланс качества и скорости
EPOCHS = 80


def main() -> None:
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] CUDA недоступна. Семантику на CPU обучать нереально — нужна GPU.")

    YOLO(MODEL).train(
        data=DATA,
        project=OUT_PROJECT,
        name="office_sem",
        exist_ok=False,
        epochs=EPOCHS,
        patience=20,
        batch=4,
        imgsz=IMGSZ,
        device=device,
        workers=8,
        amp=True,
        cache="disk",
        cos_lr=True,
        freeze=FREEZE,
        optimizer="AdamW",
        lr0=1e-4,  # маленький: дообучаем претрейн, а не ломаем его
        lrf=0.01,
        warmup_epochs=1.0,
        weight_decay=0.0005,
        # аугментации: свет в офисе скачет -> hsv щедро; геометрию не крутим сильно,
        # т.к. пол/стены имеют устойчивую перспективу
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=3.0,
        translate=0.1,
        scale=0.4,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,  # пол остаётся полом при отражении — безопасно
        mosaic=0.4,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
    )
    print(f"[INFO] Готово. Веса: {OUT_PROJECT}/office_sem/weights/best.pt")


if __name__ == "__main__":
    main()
