from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL = "weights/best_det.pt"
SOURCE = "data/test"
CONF = 0.25
IMGSZ = 960
OUT_PROJECT = str(PROJECT_ROOT / "detection" / "pred_val")


def main() -> None:
    model = YOLO(MODEL)

    results = model.predict(
        source=SOURCE,
        conf=CONF,
        imgsz=IMGSZ,
        save=True,
        project=OUT_PROJECT,
        name="best_det",
        exist_ok=True,
        stream=False,
        verbose=False,
    )

    n_img = n_with_det = n_det = 0
    for r in results:
        n_img += 1
        k = len(r.boxes)
        n_det += k
        if k > 0:
            n_with_det += 1

    empty = n_img - n_with_det
    print(f"[INFO] кадров: {n_img}")
    print(f"[INFO] с детекцией: {n_with_det}  без детекции: {empty} ({empty / max(n_img, 1) * 100:.1f}%)")
    print(f"[INFO] всего боксов: {n_det}")
    print("[INFO] размеченные кадры: yolo_train/pred_val/best_det/")


if __name__ == "__main__":
    main()
