import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
SEG_W = ROOT / "module3_segmentation" / "weights" / "best_seg_street.pt"
OUT = ROOT / "data" / "exp_weights" / "seg_out"
OUT.mkdir(parents=True, exist_ok=True)

SEG_COLORS = {
    0: (0, 0, 0), 1: (0, 235, 235), 2: (0, 150, 255),
    3: (60, 60, 240), 4: (80, 220, 80), 5: (120, 90, 60),
}
NAMES = {0: "bg", 1: "crosswalk", 2: "curb", 3: "road", 4: "sidewalk", 5: "terrain"}

seg = YOLO(str(SEG_W))
paths = sorted(sys.argv[1:])

for p in paths:
    img = cv2.imread(p)
    if img is None:
        print("не читается", p)
        continue
    r = seg(img, verbose=False)[0]
    mask = r.semantic_mask.data.cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = np.zeros_like(img)
    for cid, col in SEG_COLORS.items():
        overlay[mask == cid] = col
    vis = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    both = np.hstack([img, vis])
    out = OUT / f"{Path(p).stem}_seg.jpg"
    cv2.imwrite(str(out), both)
    present = [NAMES[c] for c in np.unique(mask) if c in NAMES]
    print(f"{Path(p).name:12} -> {out.name}   классы: {', '.join(present)}")
