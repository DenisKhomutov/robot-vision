"""
Переразложить Roboflow png-mask-semantic экспорт в структуру ultralytics semantic.
Запуск из корня проекта: uv run python yolo_train/segmentation/prepare_seg_dataset.py
"""

import shutil
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "yolo_train" / "data" / "dataset for segm"
DST = PROJECT_ROOT / "yolo_train" / "data" / "segm"
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}
NAMES = {0: "background", 1: "crosswalk", 2: "curb", 3: "road", 4: "sidewalk", 5: "terrain"}


def main() -> None:
    total = 0
    for src_split, dst_split in SPLIT_MAP.items():
        src = SRC / src_split
        if not src.exists():
            continue
        img_dst = DST / "images" / dst_split
        msk_dst = DST / "masks" / dst_split
        img_dst.mkdir(parents=True, exist_ok=True)
        msk_dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for mask in src.glob("*_mask.png"):
            stem = mask.name.replace("_mask.png", "")
            img = src / f"{stem}.jpg"
            if not img.exists():
                continue
            shutil.copy(img, img_dst / f"{stem}.jpg")
            shutil.copy(mask, msk_dst / f"{stem}.png")
            n += 1
        print(f"[INFO] {src_split} -> {dst_split}: {n} пар")
        total += n

    cfg = {
        "path": DST.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "masks_dir": "masks",
        "names": NAMES,
    }
    (DST / "data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[INFO] всего {total} пар. data.yaml: {DST / 'data.yaml'}")


if __name__ == "__main__":
    main()
