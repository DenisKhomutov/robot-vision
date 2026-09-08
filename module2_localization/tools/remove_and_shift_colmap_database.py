import argparse
import shutil
import sqlite3
from pathlib import Path


MAX_IMAGE_ID = 2147483647


def map_image_id(value: int, remove_first: int, remove_last: int, shift_from: int, shift: int):
    if remove_first <= value <= remove_last:
        return None
    return value + shift if value >= shift_from else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remove-first", type=int, required=True)
    parser.add_argument("--remove-last", type=int, required=True)
    parser.add_argument("--shift-from", type=int, required=True)
    parser.add_argument("--shift", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    shutil.copy2(args.input, args.output)
    connection = sqlite3.connect(str(args.output))

    def mapped(value):
        return map_image_id(
            value, args.remove_first, args.remove_last, args.shift_from, args.shift
        )

    for table in ("keypoints", "descriptors", "pose_priors"):
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            continue
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
        rows = connection.execute(f"SELECT * FROM {table}").fetchall()
        connection.execute(f"DELETE FROM {table}")
        placeholders = ",".join("?" for _ in columns)
        for row in rows:
            row = list(row)
            image_id = mapped(row[0])
            if image_id is None:
                continue
            row[0] = image_id
            connection.execute(f"INSERT INTO {table} VALUES ({placeholders})", row)

    image_columns = [row[1] for row in connection.execute("PRAGMA table_info(images)")]
    image_rows = connection.execute("SELECT * FROM images").fetchall()
    connection.execute("DELETE FROM images")
    placeholders = ",".join("?" for _ in image_columns)
    for row in image_rows:
        row = list(row)
        image_id = mapped(row[0])
        if image_id is None:
            continue
        row[0] = image_id
        connection.execute(f"INSERT INTO images VALUES ({placeholders})", row)

    for table in ("matches", "two_view_geometries"):
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
        rows = connection.execute(f"SELECT * FROM {table}").fetchall()
        connection.execute(f"DELETE FROM {table}")
        placeholders = ",".join("?" for _ in columns)
        for row in rows:
            row = list(row)
            first = mapped(row[0] // MAX_IMAGE_ID)
            second = mapped(row[0] % MAX_IMAGE_ID)
            if first is None or second is None:
                continue
            row[0] = min(first, second) * MAX_IMAGE_ID + max(first, second)
            connection.execute(f"INSERT INTO {table} VALUES ({placeholders})", row)

    connection.commit()
    result = connection.execute("PRAGMA integrity_check").fetchone()[0]
    images = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    pairs = connection.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
    connection.close()
    print(f"images={images} pairs={pairs} integrity={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
