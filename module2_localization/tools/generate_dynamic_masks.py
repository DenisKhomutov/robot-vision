"""Generate COLMAP-compatible masks for dynamic street objects."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--last", required=True)
    parser.add_argument("--dilate", type=int, default=15)
    args = parser.parse_args()

    paths = [p for p in sorted(args.images.glob("*.jpg")) if args.first <= p.stem <= args.last]
    if not paths:
        raise SystemExit("No images in requested range")
    args.out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    dynamic = {"person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}
    dynamic_ids = {idx for idx, name in model.names.items() if name in dynamic}
    kernel = np.ones((args.dilate, args.dilate), np.uint8)
    results = model.predict(
        source=[str(p) for p in paths], device=0, half=True, imgsz=640,
        conf=0.25, retina_masks=True, stream=True, verbose=False,
    )
    total_instances = 0
    for path, result in zip(paths, results, strict=True):
        h, w = result.orig_shape
        mask = np.zeros((h, w), np.uint8)
        if result.semantic_mask is not None:
            classes = result.semantic_mask.data.cpu().numpy()
            mask[np.isin(classes, list(dynamic_ids))] = 255
            total_instances += int(any(np.any(classes == idx) for idx in dynamic_ids))
        elif result.masks is not None and result.boxes is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for cls, polygon in zip(classes, result.masks.xy, strict=True):
                if str(result.names[cls]) not in dynamic or len(polygon) < 3:
                    continue
                cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 255)
                total_instances += 1
        if args.dilate > 1:
            mask = cv2.dilate(mask, kernel)
        cv2.imwrite(str(args.out / f"{path.stem}.png"), mask)
    print(f"frames={len(paths)} dynamic_instances={total_instances} out={args.out}")


if __name__ == "__main__":
    main()
