import argparse
import sqlite3
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import pycolmap


MAX_IMAGE_ID = 2147483647
CALIBRATED = 2


def pair_id(first: int, second: int) -> int:
    return first * MAX_IMAGE_ID + second


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--database", required=True)
    args = parser.parse_args()

    reconstruction = pycolmap.Reconstruction(str(Path(args.model).resolve()))
    database = Path(args.database).resolve()
    connection = sqlite3.connect(str(database))
    database_images = dict(connection.execute("SELECT image_id,name FROM images"))
    model_images = {image.image_id: image for image in reconstruction.images.values()}
    for image_id, image in model_images.items():
        if database_images.get(image_id) != image.name:
            raise SystemExit(f"несовпадение image_id={image_id}: {image.name}")

    order = {
        image.image_id: index
        for index, image in enumerate(sorted(model_images.values(), key=lambda item: item.name))
    }
    grouped = defaultdict(list)
    edges = 0
    for point in reconstruction.points3D.values():
        observations = sorted(point.track.elements, key=lambda item: order[item.image_id])
        for first, second in pairwise(observations):
            if first.image_id < second.image_id:
                image_a, point_a = first.image_id, first.point2D_idx
                image_b, point_b = second.image_id, second.point2D_idx
            else:
                image_a, point_a = second.image_id, second.point2D_idx
                image_b, point_b = first.image_id, first.point2D_idx
            grouped[pair_id(image_a, image_b)].append((point_a, point_b))
            edges += 1

    for index, (identifier, values) in enumerate(grouped.items(), 1):
        matches = np.ascontiguousarray(values, dtype=np.uint32)
        row = (identifier, len(matches), 2, matches.tobytes())
        connection.execute("INSERT OR REPLACE INTO matches VALUES (?,?,?,?)", row)
        connection.execute(
            "INSERT OR REPLACE INTO two_view_geometries "
            "(pair_id,rows,cols,data,config) VALUES (?,?,?,?,?)",
            (*row, CALIBRATED),
        )
        if index % 1000 == 0:
            connection.commit()
            print(f"pairs {index}/{len(grouped)}", flush=True)
    connection.commit()
    connection.close()
    print(f"готово: pairs={len(grouped)} observation_edges={edges}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
