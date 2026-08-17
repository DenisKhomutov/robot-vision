"""Remove masked keypoints and their matches from a copy of a COLMAP database."""

import argparse
import sqlite3
from pathlib import Path

import cv2
import numpy as np

MAX_IMAGE_ID = 2_147_483_647


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    args = parser.parse_args()

    db = sqlite3.connect(args.database)
    names = {name: image_id for image_id, name in db.execute("SELECT image_id, name FROM images")}
    remaps: dict[int, np.ndarray] = {}
    removed = 0
    for mask_path in sorted(args.masks.glob("*.png")):
        name = f"{mask_path.stem}.jpg"
        image_id = names.get(name)
        if image_id is None:
            continue
        row = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
        desc = db.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?", (image_id,)).fetchone()
        if row is None:
            continue
        keypoints = np.frombuffer(row[2], np.float32).reshape(row[0], row[1]).copy()
        descriptors = None if desc is None else np.frombuffer(desc[2], np.uint8).reshape(desc[0], desc[1]).copy()
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        x = np.clip(np.rint(keypoints[:, 0]).astype(int), 0, mask.shape[1] - 1)
        y = np.clip(np.rint(keypoints[:, 1]).astype(int), 0, mask.shape[0] - 1)
        keep = mask[y, x] == 0
        remap = np.full(len(keep), -1, np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        remaps[image_id] = remap
        removed += int((~keep).sum())
        keypoints = keypoints[keep]
        db.execute("UPDATE keypoints SET rows=?, data=? WHERE image_id=?", (len(keypoints), keypoints.tobytes(), image_id))
        if descriptors is not None:
            descriptors = descriptors[keep]
            db.execute("UPDATE descriptors SET rows=?, data=? WHERE image_id=?", (len(descriptors), descriptors.tobytes(), image_id))

    changed_pairs = 0
    removed_matches = 0
    for pair_id, rows, cols, blob in db.execute("SELECT pair_id, rows, cols, data FROM matches").fetchall():
        image_id1, image_id2 = pair_id // MAX_IMAGE_ID, pair_id % MAX_IMAGE_ID
        if image_id1 not in remaps and image_id2 not in remaps:
            continue
        matches = np.frombuffer(blob, np.uint32).reshape(rows, cols).copy()
        keep = np.ones(rows, bool)
        if image_id1 in remaps:
            mapped = remaps[image_id1][matches[:, 0]]
            keep &= mapped >= 0
            matches[:, 0] = np.maximum(mapped, 0)
        if image_id2 in remaps:
            mapped = remaps[image_id2][matches[:, 1]]
            keep &= mapped >= 0
            matches[:, 1] = np.maximum(mapped, 0)
        removed_matches += int((~keep).sum())
        matches = matches[keep]
        db.execute("UPDATE matches SET rows=?, data=? WHERE pair_id=?", (len(matches), matches.tobytes(), pair_id))
        changed_pairs += 1

    db.execute("DELETE FROM two_view_geometries")
    db.commit()
    db.close()
    print(f"masked_images={len(remaps)} removed_keypoints={removed} changed_pairs={changed_pairs} removed_matches={removed_matches}")


if __name__ == "__main__":
    main()
