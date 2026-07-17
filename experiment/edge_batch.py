"""
Прогон edge_distance по всем front-кадрам route_reference.

Каждый кадр -> отдельная панель в experiment/edge_batch_out/<route>_<dir>/.
В конце — сводка по маршрутам (медиана/ст.откл ширины), чтобы видеть стабильность.

Запуск:
    uv run python experiment/edge_batch.py
"""

import glob
from pathlib import Path

import cv2
import numpy as np

from edge_distance import INTRINSICS, process

OUT_DIR = Path("experiment/edge_batch_out")
SRC_ROOT = Path("data/route_reference")


def frame_no(p: str) -> int:
    return int("".join(filter(str.isdigit, Path(p).stem)) or 0)


def main() -> None:
    intr = np.load(INTRINSICS)
    k, dist = intr["K"], intr["dist"]
    dirs = sorted(SRC_ROOT.glob("*/*/front"))
    if not dirs:
        print(f"[ERROR] нет кадров: {SRC_ROOT}")
        return

    for d in dirs:
        tag = f"{d.parts[-3]}_{d.parts[-2]}"  # route_A_forward
        out = OUT_DIR / tag
        out.mkdir(parents=True, exist_ok=True)
        widths = []
        print(f"\n=== {tag} ===")
        for p in sorted(glob.glob(f"{d}/*.jpg"), key=frame_no):
            panel, res = process(cv2.imread(p), k, dist)
            stem = Path(p).stem
            cv2.imwrite(str(out / f"{stem}_edge.jpg"), panel)
            if res["valid"]:
                widths.append(res["width_m"])
                print(f"{stem:<9} width={res['width_m']:.2f}  left={res['left_clearance_m']:.2f}  "
                      f"right={res['right_clearance_m']:.2f}  offset={res['offset_from_center_m']:+.2f}  "
                      f"near={res['near_dist_m']:.2f}")
            else:
                print(f"{stem:<9} no floor")
        if widths:
            w = np.array(widths)
            print(f"--- width: медиана {np.median(w):.2f}  ст.откл {w.std():.2f}  "
                  f"мин {w.min():.2f}  макс {w.max():.2f}  ({len(w)}/{len(list(d.glob('*.jpg')))} валидных)")

    print(f"\n[INFO] панели -> {OUT_DIR}/  (оригинал | маска | вид сверху | карта | метры)")


if __name__ == "__main__":
    main()
