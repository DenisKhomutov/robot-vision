from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import plotting

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL = str(PROJECT_ROOT / "yolo_train" / "weights" / "surfaces.pt")
SOURCE = str(PROJECT_ROOT / "yolo_train" / "data" / "segm" / "images" / "test")
OUT = str(PROJECT_ROOT / "yolo_train" / "exp_result")
IMGSZ = 1280
OVERRIDES = {1: (255, 255, 0), 3: (255, 0, 0), 4: (144, 238, 144)}


def main() -> None:
    n = plotting.colors.n
    pal = [plotting.colors(i, bgr=False) for i in range(n)]
    for idx, rgb in OVERRIDES.items():
        if idx < n:
            pal[idx] = rgb
    plotting.colors.palette = [tuple(c) for c in pal]
    plotting.colors.n = len(pal)

    YOLO(MODEL).predict(
        source=SOURCE,
        imgsz=IMGSZ,
        save=True,
        project=OUT,
        name="surfaces_test_colored",
        exist_ok=True,
        verbose=False,
    )
    print("[INFO] готово -> exp_result/surfaces_test_colored")


if __name__ == "__main__":
    main()
