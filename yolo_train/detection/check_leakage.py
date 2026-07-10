"""
Проверка утечки train↔val по перцептивному хэшу (dHash). Без нейросетей — cv2+numpy.
Для каждого val-кадра ищет ближайший train-кадр по hamming-расстоянию хэшей.

    dist == 0  → точный/почти точный дубликат
    dist <= 5  → почти-дубликат (соседний кадр видео, кроп, ре-энкод)

Запуск из корня проекта:
    uv run python yolo_train/check_leakage.py
"""

from pathlib import Path

import cv2
import numpy as np

TRAIN_DIR = Path("data/pedestrian_only/images/train")
VAL_DIR = Path("data/pedestrian_only/images/val")
NEAR = 5  # порог hamming для «почти-дубликата»
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash(path: Path) -> int | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]  # 8x8 сравнений соседей
    return int(np.packbits(bits.flatten()).view(np.uint64)[0])


def hashes(folder: Path) -> tuple[np.ndarray, list[str]]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    hs, names = [], []
    for p in paths:
        h = dhash(p)
        if h is not None:
            hs.append(h)
            names.append(p.name)
    return np.array(hs, dtype=np.uint64), names


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> 1) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def main() -> None:
    print("[INFO] хэширую train...")
    train_h, _ = hashes(TRAIN_DIR)
    print(f"[INFO] train: {len(train_h)} кадров")
    print("[INFO] хэширую val...")
    val_h, val_names = hashes(VAL_DIR)
    print(f"[INFO] val: {len(val_h)} кадров")

    exact = near = 0
    examples: list[str] = []
    for i, vh in enumerate(val_h):
        dist = popcount64(train_h ^ vh)
        m = int(dist.min())
        if m == 0:
            exact += 1
        if m <= NEAR:
            near += 1
            if len(examples) < 10:
                examples.append(f"{val_names[i]} (dist={m})")

    n = len(val_h)
    print(f"\n[RESULT] точных дубликатов (dist=0): {exact}/{n} ({exact / max(n, 1) * 100:.1f}%)")
    print(f"[RESULT] почти-дубликатов (dist<={NEAR}): {near}/{n} ({near / max(n, 1) * 100:.1f}%)")
    print("[RESULT] примеры утёкших val:")
    for e in examples:
        print("   ", e)


if __name__ == "__main__":
    main()
