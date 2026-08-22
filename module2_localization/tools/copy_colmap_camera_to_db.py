"""Replace a COLMAP database camera with one from a proven reconstruction."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pycolmap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()

    reconstruction = pycolmap.Reconstruction(str(args.model))
    if len(reconstruction.cameras) != 1:
        raise SystemExit(f"expected one camera, found {len(reconstruction.cameras)}")
    camera = next(iter(reconstruction.cameras.values()))
    model_id = int(camera.model)
    params = np.asarray(camera.params, dtype=np.float64).tobytes()

    connection = sqlite3.connect(args.database)
    try:
        rows = connection.execute("SELECT camera_id FROM cameras").fetchall()
        if len(rows) != 1:
            raise SystemExit(f"expected one database camera, found {len(rows)}")
        camera_id = int(rows[0][0])
        connection.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 "
            "WHERE camera_id=?",
            (model_id, camera.width, camera.height, params, camera_id),
        )
        connection.commit()
    finally:
        connection.close()

    print(f"camera_id={camera_id}")
    print(f"model={camera.model_name} ({model_id})")
    print(f"size={camera.width}x{camera.height}")
    print("params=" + ",".join(f"{value:.12g}" for value in camera.params))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
