"""Create a COLMAP database containing only a named subset of images."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path


MAX_IMAGE_ID = 2_147_483_647


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True,
                        help="keep the first N images ordered by image name")
    args = parser.parse_args()

    if args.first < 2:
        raise SystemExit("--first must be at least 2")
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input, args.output)

    connection = sqlite3.connect(args.output)
    try:
        rows = connection.execute(
            "SELECT image_id, name FROM images ORDER BY name LIMIT ?", (args.first,)
        ).fetchall()
        if len(rows) != args.first:
            raise SystemExit(f"database contains only {len(rows)} images")
        kept = {int(row[0]) for row in rows}

        connection.execute("BEGIN")
        for table in ("matches", "two_view_geometries"):
            pair_ids = connection.execute(f"SELECT pair_id FROM {table}").fetchall()
            rejected = [
                (pair_id,) for (pair_id,) in pair_ids
                if pair_id // MAX_IMAGE_ID not in kept or pair_id % MAX_IMAGE_ID not in kept
            ]
            connection.executemany(f"DELETE FROM {table} WHERE pair_id = ?", rejected)
        placeholders = ",".join("?" for _ in kept)
        for table in ("keypoints", "descriptors", "pose_priors", "images"):
            connection.execute(
                f"DELETE FROM {table} WHERE image_id NOT IN ({placeholders})", tuple(kept)
            )
        connection.commit()
        connection.execute("VACUUM")

        print(f"kept {len(kept)} images: {rows[0][1]} .. {rows[-1][1]}")
        for table in ("images", "keypoints", "matches", "two_view_geometries"):
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table}: {count}")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
