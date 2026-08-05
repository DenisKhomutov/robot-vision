import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CITYSCAPES_DYNAMIC = {10, 11, 12, 13, 14, 15, 16, 17, 18}
NAMES = {8: "vegetation", 10: "sky",
         11: "person", 12: "rider", 13: "car", 14: "truck",
         15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="module2_localization/data/frames_street_1")
    ap.add_argument("--weights", default="module3_segmentation/weights/yolo26m-sem.pt")
    ap.add_argument("--out", default="module2_localization/out/seg_preview")
    ap.add_argument("--sample", type=int, default=30, help="равномерно взять N кадров")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dilate", type=int, default=7, help="расширить маску на N px (захватить кромки/тени)")
    ap.add_argument("--build-out", default=None,
                    help="папка чистого датасета: ВСЕ кадры, вырезка под теми же именами, без оверлеев")
    args = ap.parse_args()

    build = args.build_out is not None
    files = sorted(Path(args.frames).glob("*.jpg"))
    if not files:
        print(f"нет кадров в {args.frames}")
        return 1
    if build:
        picks = files
        out = Path(args.build_out)
    else:
        idx = np.linspace(0, len(files) - 1, min(args.sample, len(files))).astype(int)
        picks = [files[i] for i in dict.fromkeys(idx)]
        out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    half = torch.cuda.is_available()

    for p in picks:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        res = model(img, conf=args.conf, imgsz=args.imgsz, half=half, verbose=False)[0]
        mask = np.zeros((h, w), np.uint8)
        found = {}
        sm = res.semantic_mask
        if sm is not None:
            sm = sm.data.cpu().numpy() if hasattr(sm, "data") else np.asarray(sm)
            sm = np.asarray(sm).astype(np.int32)
            if sm.shape != (h, w):
                sm = cv2.resize(sm.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
            for c in CITYSCAPES_DYNAMIC:
                cnt = int((sm == c).sum())
                if cnt:
                    mask[sm == c] = 255
                    found[NAMES[c]] = cnt
        if args.dilate:
            mask = cv2.dilate(mask, np.ones((args.dilate, args.dilate), np.uint8))

        pct = 100 * (mask > 0).mean()

        if build:
            # пишем МАСКУ (255=выкинуть точки), не трогая кадр: чёрная заливка дала бы
            # ложные точки на кромке. build_map по маске отбрасывает keypoints.
            cv2.imwrite(str(out / f"{p.stem}.png"), mask)
            if (picks.index(p) + 1) % 100 == 0:
                print(f"  {picks.index(p) + 1}/{len(picks)}", flush=True)
        else:
            cut = img.copy()
            cut[mask > 0] = 0
            overlay = img.copy()
            overlay[mask > 0] = (0, 0, 255)
            overlay = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
            cv2.putText(overlay, f"{p.name}  вырезано {pct:.1f}%  " + str(found),
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(out / f"{p.stem}_cut.jpg"), cut)
            cv2.imwrite(str(out / f"{p.stem}_overlay.jpg"), overlay)
            print(f"{p.name}: вырезано {pct:.1f}%  {found}")

    if build:
        print(f"\nчистый датасет готов: {len(picks)} кадров -> {out}/")
    else:
        print(f"\nготово -> {out}/  (пары *_overlay.jpg = что нашла, *_cut.jpg = что уйдёт из карты)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
