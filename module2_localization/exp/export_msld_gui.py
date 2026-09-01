import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pycolmap

from .. import config


def export_gui_map(experimental_map: Path) -> Path:
    metadata = json.loads((experimental_map / "metadata.json").read_text())
    shard_name = metadata["source_map"]
    shard_dir = config.MAPS_DIR / shard_name
    shard = json.loads((shard_dir / "shard.json").read_text())
    source_dir = config.MAPS_DIR / shard["source_map"]
    source_sparse = source_dir / "sparse" / "0"
    if not source_sparse.exists():
        raise RuntimeError(f"нет исходной COLMAP-модели: {source_sparse}")

    with np.load(experimental_map / "aliked_bank.npz") as bank:
        selected_xyz = bank["xyz"]
    with np.load(experimental_map / "runtime.npz") as runtime:
        selected_names = set(runtime["names"].tolist())

    reconstruction = pycolmap.Reconstruction(str(source_sparse))
    point_by_xyz = {
        np.asarray(point.xyz, np.float64).tobytes(): point_id
        for point_id, point in reconstruction.points3D.items()
    }
    selected_ids = set()
    missing_points = 0
    for xyz in selected_xyz:
        point_id = point_by_xyz.get(np.asarray(xyz, np.float64).tobytes())
        if point_id is None:
            missing_points += 1
        else:
            selected_ids.add(point_id)
    if missing_points:
        raise RuntimeError(f"{missing_points} экспериментальных точек отсутствуют в исходной модели")

    for frame_id in list(reconstruction.reg_frame_ids()):
        frame = reconstruction.frames[frame_id]
        if not any(reconstruction.images[data_id.id].name in selected_names for data_id in frame.image_ids):
            reconstruction.deregister_frame(frame_id)
    for point_id in list(reconstruction.point3D_ids()):
        if point_id not in selected_ids:
            reconstruction.delete_point3D(point_id)

    gui = experimental_map / "gui"
    if gui.exists():
        shutil.rmtree(gui)
    sparse = gui / "sparse" / "0"
    images = gui / "images"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)

    registered = 0
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.images[image_id]
        source = Path(image.name)
        if not source.exists():
            raise RuntimeError(f"нет фотографии исходной карты: {source}")
        target = images / source.name
        if target.exists() and target.resolve() != source.resolve():
            raise RuntimeError(f"совпадающие имена фотографий: {source.name}")
        if not target.exists():
            os.symlink(source, target)
        image.name = source.name
        registered += 1
    reconstruction.write_binary(str(sparse))
    info = {
        "source_map": shard["source_map"],
        "source_shard": shard_name,
        "registered_images": registered,
        "points3D": reconstruction.num_points3D(),
        "images_are_symlinks": True,
        "import_path": str(sparse.resolve()),
        "image_path": str(images.resolve()),
    }
    (gui / "gui.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return gui


def main():
    parser = argparse.ArgumentParser(description="экспорт MSLD-карты для COLMAP GUI")
    parser.add_argument("maps", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.maps:
        gui = export_gui_map(path)
        print(f"[gui] {path}: {json.loads((gui / 'gui.json').read_text())}")


if __name__ == "__main__":
    main()
