"""
Дообучение семантической сегментации поверхностей.
Запуск из корня проекта: uv run python yolo_train/segmentation/train_segmentation.py
"""

from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL = str(PROJECT_ROOT / "yolo_train" / "weights" / "yolo26m-sem.pt")
DATA = "PATH/TO/segmentation/data.yaml"
OUT_PROJECT = str(PROJECT_ROOT / "yolo_train" / "segmentation" / "runs")
FREEZE = 0


def main() -> None:
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] CUDA недоступна — сегментацию на CPU обучать нереально. Нужна GPU.")

    model = YOLO(MODEL)

    model.train(
        data=DATA,
        project=OUT_PROJECT,
        name="surfaces_ft",
        exist_ok=False,
        epochs=60,
        patience=15,
        batch=4,
        imgsz=1280,
        device=device,
        workers=8,
        amp=True,
        cache="disk",
        cos_lr=True,
        freeze=FREEZE,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.01,
        warmup_epochs=1.0,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
    )

    print(f"[INFO] Готово. Веса: {OUT_PROJECT}/surfaces_ft/weights/best.pt")


if __name__ == "__main__":
    main()
