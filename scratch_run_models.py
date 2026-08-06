import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "exp_weights"
DET_W = OUT / "traffic_birdseye.pt"
SEG_W = ROOT / "module3_segmentation" / "weights" / "best_seg_street.pt"

SEG_COLORS = {
    0: (0, 0, 0), 1: (0, 235, 235), 2: (0, 150, 255),
    3: (60, 60, 240), 4: (80, 220, 80), 5: (120, 90, 60),
}

imgs = sys.argv[1:]
det = YOLO(str(DET_W))
seg = YOLO(str(SEG_W))

for p in imgs:
    img = cv2.imread(p)
    if img is None:
        print("не читается", p)
        continue
    name = Path(p).stem

    r = det(img, verbose=False)[0]
    det_vis = img.copy()
    for b, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        x1, y1, x2, y2 = map(int, b)
        lbl = f"{r.names[int(c)]} {cf:.2f}"
        cv2.rectangle(det_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(det_vis, lbl, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    rs = seg(img, verbose=False)[0]
    mask = rs.semantic_mask.data.cpu().numpy()
    if mask.ndim == 3:
        mask = mask.argmax(0) if mask.shape[0] <= 8 else mask[0]
    mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    overlay = np.zeros_like(img)
    for cid, col in SEG_COLORS.items():
        overlay[mask == cid] = col
    seg_vis = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

    both = np.hstack([det_vis, seg_vis])
    out = OUT / f"{name}_result.jpg"
    cv2.imwrite(str(out), both)
    print("сохранено", out)
