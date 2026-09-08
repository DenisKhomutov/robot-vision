import argparse
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pycolmap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-db", required=True)
    args = parser.parse_args()

    model = Path(args.model).resolve()
    output = Path(args.output_db).resolve()
    if not model.is_dir():
        raise SystemExit(f"нет модели: {model}")
    if output.exists():
        raise SystemExit(f"уже существует: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    reconstruction = pycolmap.Reconstruction(str(model))
    if len(reconstruction.cameras) != 1:
        raise SystemExit(f"ожидалась одна камера, найдено {len(reconstruction.cameras)}")

    subprocess.run(
        ["colmap", "database_creator", "--database_path", str(output)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    connection = sqlite3.connect(str(output))
    camera = next(iter(reconstruction.cameras.values()))
    connection.execute(
        "INSERT INTO cameras VALUES (?,?,?,?,?,?)",
        (
            camera.camera_id,
            int(camera.model.value),
            camera.width,
            camera.height,
            np.asarray(camera.params, dtype=np.float64).tobytes(),
            1,
        ),
    )

    images = sorted(reconstruction.images.values(), key=lambda image: image.image_id)
    for image in images:
        connection.execute(
            "INSERT INTO images (image_id, name, camera_id) VALUES (?,?,?)",
            (image.image_id, image.name, image.camera_id),
        )
        keypoints = np.asarray([point.xy for point in image.points2D], dtype=np.float32)
        connection.execute(
            "INSERT INTO keypoints VALUES (?,?,?,?)",
            (image.image_id, len(keypoints), 2, keypoints.tobytes()),
        )
    connection.commit()
    connection.close()
    print(
        f"готово: {output} | images={len(images)} "
        f"id_max={max(image.image_id for image in images)} "
        f"camera={camera.model.name} {camera.width}x{camera.height}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
