"""
Дообучение детектора для ОТВЕРЖЕНИЯ автомобильных светофоров (hard negatives).

Старт с уже обученного пешеходного детектора (weights/best_det.pt) — сохраняем его
recall и лишь учим отвергать машинные. Датасет тот же + добавленные негативы:
кадры с АВТОМОБИЛЬНЫМИ светофорами, у которых .txt пустой/отсутствует → фон.

Подготовка датасета (руками, до запуска):
  - положи машинные кадры в data/pedestrian_merged/images/train (часть — в .../val)
  - .txt для них НЕ создавай (или пустой) → YOLO трактует как background
  - держи негативы ~5-10% от объёма, разнообразные (ракурсы/дистанции/свет)

Запуск из корня проекта:
    uv run python yolo_train/finetune_negatives.py

⚠️ imgsz=960 — как у базовой модели. GPU обязателен.
"""

from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = "data/pedestrian_merged/traffic_light.yaml"
MODEL = str(PROJECT_ROOT / "weights" / "best_det.pt")  # стартуем со своего детектора
# абсолютный путь: иначе ultralytics шлёт вывод в чужой глобальный runs_dir
OUT_PROJECT = str(PROJECT_ROOT / "yolo_train" / "runs")


def main() -> None:
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] CUDA недоступна — дообучение на CPU нереально по времени. Нужна GPU.")

    model = YOLO(MODEL)

    model.train(
        # данные / вывод
        data=DATA,
        project=OUT_PROJECT,
        name="pedestrian_neg_ft",
        exist_ok=False,
        # расписание — короткое дообучение
        epochs=50,
        patience=15,  # оборвёт, если val перестанет улучшаться
        batch=10,
        imgsz=960,  # КАК У БАЗОВОЙ МОДЕЛИ — не менять
        # производительность
        device=device,
        workers=8,
        amp=True,
        cache="disk",
        cos_lr=True,
        # оптимизатор — низкий LR, чтобы не забыть пешеходных (catastrophic forgetting)
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.01,
        warmup_epochs=1.0,
        weight_decay=0.0005,
        # веса лоссов
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # аугментации: цвет/свет
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        # аугментации: геометрия
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,  # верх/низ семантичны
        fliplr=0.5,
        # аугментации: композиция (легче, чем при обучении с нуля)
        mosaic=0.5,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
    )

    print(f"[INFO] Готово. Веса: {OUT_PROJECT}/pedestrian_neg_ft/weights/best.pt")


if __name__ == "__main__":
    main()
