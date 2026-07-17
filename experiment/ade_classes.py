"""
Полная семантика ADE20K на кадрах маршрута — все 150 классов, не только пол.

Показывает, что модель видит целиком: пол, стены, потолок, люди, двери, мебель.
Полезно, чтобы решить, какие классы брать под проходимую зону и препятствия.

Панели: оригинал | классы | наложение (+легенда по классам кадра с их долей)
Выход: experiment/ade_classes_out/

Запуск:
    uv run python experiment/ade_classes.py
    uv run python experiment/ade_classes.py data/route_reference/route_B/forward/front
"""

import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SEG_WEIGHTS = "experiment/weights/yolo26m-sem-ade20k.pt"
OUT_DIR = Path("experiment/ade_classes_out")
DEFAULT_DIR = "data/route_reference/route_A/forward/front"
IMGSZ = 640
MIN_FRAC = 0.005  # в легенду — классы крупнее 0.5% кадра

# ключевым классам — запоминающиеся цвета (BGR), остальным авто-палитра
FIXED = {
    3: ((144, 238, 144), "floor"),
    0: ((60, 60, 220), "wall"),
    5: ((200, 200, 90), "ceiling"),
    12: ((0, 215, 255), "person"),
    14: ((0, 140, 255), "door"),
    8: ((230, 130, 230), "window"),
    19: ((120, 90, 200), "chair"),
    15: ((90, 160, 210), "table"),
}


def palette(n=150):
    """Стабильная палитра: ключевые классы фиксированы, прочие разведены по тону."""
    pal = np.zeros((n, 3), np.uint8)
    for i in range(n):
        hue = int(179 * (i * 47 % n) / n)  # шаг 47 разводит соседние id по цвету
        pal[i] = cv2.cvtColor(np.uint8([[[hue, 170, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
    for i, (col, _) in FIXED.items():
        pal[i] = col
    return pal


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    paths = sorted(glob.glob(f"{src}/*.jpg"), key=lambda p: int("".join(filter(str.isdigit, Path(p).stem)) or 0))
    if not paths:
        print(f"[ERROR] нет кадров: {src}")
        return

    model = YOLO(SEG_WEIGHTS)
    names = model.names
    pal = palette(len(names))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in paths:
        bgr = cv2.imread(p)
        h, w = bgr.shape[:2]
        r = model.predict(bgr, imgsz=IMGSZ, verbose=False)[0]
        sm = np.asarray(r.semantic_mask.data).astype(np.uint8)
        if sm.shape != (h, w):
            sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST)

        colored = pal[sm]
        blend = cv2.addWeighted(bgr, 0.5, colored, 0.5, 0)

        # легенда: классы кадра по убыванию площади
        ids, cnt = np.unique(sm, return_counts=True)
        order = np.argsort(-cnt)
        y = 26
        for i in order:
            cid, frac = int(ids[i]), cnt[i] / sm.size
            if frac < MIN_FRAC:
                continue
            col = tuple(int(c) for c in pal[cid])
            cv2.rectangle(blend, (10, y - 13), (30, y + 3), col, -1)
            cv2.rectangle(blend, (10, y - 13), (30, y + 3), (255, 255, 255), 1)
            cv2.putText(blend, f"{names[cid]} {100 * frac:.0f}%", (36, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(blend, f"{names[cid]} {100 * frac:.0f}%", (36, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y += 22

        panel = np.hstack([bgr, colored, blend])
        stem = Path(p).stem
        cv2.imwrite(str(OUT_DIR / f"{stem}_ade.jpg"), cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2)))
        top = [f"{names[int(ids[i])]}={100 * cnt[i] / sm.size:.0f}%" for i in order[:5] if cnt[i] / sm.size >= MIN_FRAC]
        print(f"{stem:<9} {' '.join(top)}")

    print(f"\n[INFO] {len(paths)} кадров -> {OUT_DIR}/  (оригинал | классы | наложение+легенда)")


if __name__ == "__main__":
    main()
