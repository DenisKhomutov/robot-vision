from ultralytics import YOLO


def main():
    model = YOLO("runs/detect/runs_traffic_light/yolo11s_traffic_light_v1_6_merged/weights/last.pt")

    model.train(
        resume = True,
        
        data="data/datasets/pedestrian_merged/traffic_light.yaml",

        epochs=50,
        imgsz=1280,
        batch=10,
        device=0,
        workers=8,
        
        pretrained=True,
        optimizer="AdamW",

        lr0=1e-4,
        lrf=0.01,
        cos_lr=True,

        patience=20,

        amp=True,

        cache="disk",


        project="runs_traffic_light",
        name="yolo11s_traffic_light_v1_6_merged",
        exist_ok=True,

        save=True,
        save_period=10,
        val=True,

        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.35,

        degrees=3.0,
        translate=0.08,
        scale=0.35,

        fliplr=0.5,
        flipud=0.0,

        mosaic=0.5,
        mixup=0.0,

        close_mosaic=15,
    )

    metrics = model.val(
        data="data/datasets/traffic_light_merged/traffic_light.yaml",
        imgsz=1280,
        batch=8,
        device=0,
    )

    print(metrics)


if __name__ == "__main__":
    main()
