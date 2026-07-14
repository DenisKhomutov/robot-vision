from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL = str(PROJECT_ROOT / "yolo_train" / "weights" / "surfaces.pt")
VIDEO = str(PROJECT_ROOT / "yolo_train" / "data" / "video_2026-06-04_14-29-40.mp4")
OUT = str(PROJECT_ROOT / "yolo_train" / "exp_result" / "surfaces_video_morph.mp4")
IMGSZ = 1280
ALPHA = 0.5
MIN_AREA = 400
COLORS = {0: (4, 42, 255), 1: (255, 255, 0), 2: (243, 243, 243), 3: (255, 0, 0), 4: (144, 238, 144), 5: (255, 111, 221)}


def clean(cmap: np.ndarray) -> np.ndarray:
    out = cmap.copy()
    for c in range(1, 6):
        binm = (cmap == c).astype(np.uint8)
        num, lbl, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
                out[lbl == i] = 0
    return out


def render(orig: np.ndarray, cmap: np.ndarray) -> np.ndarray:
    ov = orig.copy()
    for c, (r, g, b) in COLORS.items():
        ov[cmap == c] = (b, g, r)
    return cv2.addWeighted(orig, 1 - ALPHA, ov, ALPHA, 0)


def main() -> None:
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = YOLO(MODEL)
    for r in model.predict(source=VIDEO, imgsz=IMGSZ, stream=True, verbose=False):
        cmap = np.asarray(r.semantic_mask.data.cpu()).astype(np.uint8)
        out = render(r.orig_img, clean(cmap))
        vw.write(out)
        cv2.imshow("surfaces", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    vw.release()
    cv2.destroyAllWindows()
    print(f"[INFO] готово -> {OUT}")


if __name__ == "__main__":
    main()
