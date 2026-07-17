"""
COCO-segmentation (Roboflow) -> семантический датасет ultralytics.

Берёт полигоны инстансов и растеризует их в PNG-маски, где значение пикселя = id класса.
Инстансы при этом схлопываются в классы (что нам и нужно: «пол / не пол»).

Классы: 0=background, 1=allowed_path, 2=obstacle (id совпадают с COCO-категориями).
Препятствия рисуются ПОВЕРХ пола: если пиксель размечен и тем, и другим — он препятствие.

В экспорте только сплит train, поэтому режем на train/val сами (по VAL_FRACTION,
детерминированно по имени файла, чтобы сплит не менялся между прогонами).

Запуск из корня проекта:
    uv run python yolo_train/segmentation/coco_to_semantic.py
Дальше: yolo_train/segmentation/train_segmentation.py (указав получившийся data.yaml)
"""

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "data" / "MyProject.coco-segmentation" / "train"
DST = PROJECT_ROOT / "yolo_train" / "data" / "office_sem_seg_train"
NAMES = {0: "background", 1: "allowed_path", 2: "obstacle"}
DRAW_ORDER = [1, 2]  # сначала пол, препятствия поверх
VAL_FRACTION = 0.15


def is_val(file_name: str) -> bool:
    """Детерминированный сплит: хэш имени -> стабильно между запусками."""
    h = int(hashlib.md5(file_name.encode()).hexdigest()[:8], 16)
    return (h % 1000) < VAL_FRACTION * 1000


def build_mask(anns, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    for cid in DRAW_ORDER:
        for a in anns:
            if a["category_id"] != cid:
                continue
            for seg in a.get("segmentation", []):
                if len(seg) < 6:  # нужно >=3 точек
                    continue
                poly = np.array(seg, np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [poly], cid)
    return mask


def main() -> None:
    ann_path = SRC / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"[ERROR] нет аннотаций: {ann_path}")
        return
    data = json.load(ann_path.open())

    by_image: dict[int, list] = {}
    for a in data["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)

    for split in ("train", "val"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "masks" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "empty": 0}
    px = np.zeros(len(NAMES), np.int64)
    for im in data["images"]:
        anns = by_image.get(im["id"], [])
        if not anns:
            stats["empty"] += 1
            continue
        src_img = SRC / im["file_name"]
        if not src_img.exists():
            continue
        mask = build_mask(anns, im["width"], im["height"])
        for i in range(len(NAMES)):
            px[i] += int((mask == i).sum())

        split = "val" if is_val(im["file_name"]) else "train"
        stem = Path(im["file_name"]).stem
        shutil.copy(src_img, DST / "images" / split / f"{stem}.jpg")
        cv2.imwrite(str(DST / "masks" / split / f"{stem}.png"), mask)
        stats[split] += 1

    cfg = {
        "path": DST.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "masks_dir": "masks",
        "names": NAMES,
    }
    (DST / "data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    total_px = px.sum()
    print(f"[INFO] train={stats['train']}  val={stats['val']}  без аннотаций пропущено={stats['empty']}")
    print("[INFO] доля пикселей по классам:")
    for i, n in NAMES.items():
        print(f"       {n:<14} {100 * px[i] / total_px:5.1f}%")
    print(f"[INFO] data.yaml -> {DST / 'data.yaml'}")


if __name__ == "__main__":
    main()
