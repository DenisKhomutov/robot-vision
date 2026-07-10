from ultralytics import YOLO

DATA = "data/pedestrian_merged/traffic_light.yaml"

def main() -> None:
    device = 0

    model = YOLO("yolo11s.pt")

    model.train(
        # данные / вывод
        data=DATA,
        project="yolo_train/runs",
        name="pedestrian_yolo11s",
        exist_ok=False,
        # расписание
        epochs=150,
        patience=30,
        batch=14,
        imgsz=960,
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
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # веса лоссов
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # аугментации: цвет/свет (день/ночь/контровой) 
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5, 
        # аугментации: геометрия
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        # аугментации: композиция
        mosaic=0.5,
        close_mosaic=15,
        mixup=0.0,
        copy_paste=0.0,
    )

    print("[INFO] Готово. Лучшие веса: yolo_train/runs/pedestrian_yolo11s/weights/best.pt")


if __name__ == "__main__":
    main()
