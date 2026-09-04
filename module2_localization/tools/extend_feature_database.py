import argparse
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MAX_IMAGE_ID = 2147483647
MIN_MATCHES = 15


def pair_id(first: int, second: int) -> int:
    if first > second:
        first, second = second, first
    return first * MAX_IMAGE_ID + second


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--output-db", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--overlap", type=int, default=25)
    parser.add_argument("--kpts", type=int, default=4096)
    parser.add_argument("--det-threshold", type=float, default=0.04)
    args = parser.parse_args()

    source = Path(args.source_db)
    output = Path(args.output_db)
    image_dir = ROOT / "data" / args.images
    names = sorted(path.name for path in image_dir.glob("*.jpg"))
    if not source.is_file() or not names:
        raise SystemExit("нет исходной базы или изображений")
    if output.exists():
        raise SystemExit(f"уже существует {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)

    connection = sqlite3.connect(str(output))
    existing_rows = connection.execute(
        "SELECT image_id, name FROM images ORDER BY image_id"
    ).fetchall()
    existing_names = [row[1] for row in existing_rows]
    if names[:len(existing_names)] != existing_names:
        raise SystemExit("существующие изображения не являются началом объединённого датасета")

    sys.path.insert(0, str(ROOT / "core"))
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image
    from model_weights import configure_local_model_weights

    configure_local_model_weights()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ALIKED(
        max_num_keypoints=args.kpts,
        detection_threshold=args.det_threshold,
    ).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)

    existing_count = len(existing_names)
    new_names = names[existing_count:]
    first_context = max(0, existing_count - args.overlap)
    feature_names = names[first_context:]
    features = {}
    started = time.perf_counter()
    for index, name in enumerate(feature_names, 1):
        with torch.inference_mode():
            image = load_image(str(image_dir / name)).to(device)
            features[name] = extractor.extract(image)
        if index % 100 == 0:
            elapsed = time.perf_counter() - started
            left = elapsed / index * (len(feature_names) - index)
            print(f"features {index}/{len(feature_names)} ~{left:.0f}s", flush=True)

    next_image_id = max((row[0] for row in existing_rows), default=0) + 1
    name_to_id = {name: image_id for image_id, name in existing_rows}
    for offset, name in enumerate(new_names):
        image_id = next_image_id + offset
        name_to_id[name] = image_id
        connection.execute(
            "INSERT INTO images (image_id, name, camera_id) VALUES (?,?,1)",
            (image_id, name),
        )
        keypoints = features[name]["keypoints"][0].cpu().numpy().astype(np.float32)
        connection.execute(
            "INSERT INTO keypoints VALUES (?,?,?,?)",
            (image_id, keypoints.shape[0], 2, keypoints.tobytes()),
        )
    connection.commit()

    pairs = []
    for first_index in range(first_context, len(names)):
        for offset in range(1, args.overlap + 1):
            second_index = first_index + offset
            if second_index >= len(names):
                break
            if second_index < existing_count:
                continue
            pairs.append((names[first_index], names[second_index]))

    strong_pairs = []
    weak = 0
    started = time.perf_counter()
    for index, (first_name, second_name) in enumerate(pairs, 1):
        with torch.inference_mode():
            result = matcher({
                "image0": features[first_name],
                "image1": features[second_name],
            })
        matches = result["matches"][0].cpu().numpy().astype(np.uint32)
        if len(matches) >= MIN_MATCHES:
            first_id = name_to_id[first_name]
            second_id = name_to_id[second_name]
            if first_id > second_id:
                first_id, second_id = second_id, first_id
                matches = matches[:, ::-1]
            matches = np.ascontiguousarray(matches, np.uint32)
            connection.execute(
                "INSERT OR REPLACE INTO matches VALUES (?,?,?,?)",
                (pair_id(first_id, second_id), matches.shape[0], 2, matches.tobytes()),
            )
            strong_pairs.append((first_name, second_name))
        else:
            weak += 1
        if index % 500 == 0:
            connection.commit()
            elapsed = time.perf_counter() - started
            left = elapsed / index * (len(pairs) - index)
            print(
                f"pairs {index}/{len(pairs)} strong={len(strong_pairs)} weak={weak} ~{left:.0f}s",
                flush=True,
            )
    connection.commit()
    connection.close()

    pair_list = output.with_suffix(".new_pairs.txt")
    pair_list.write_text(
        "".join(f"{first} {second}\n" for first, second in strong_pairs),
        encoding="utf-8",
    )
    result = subprocess.run([
        "colmap", "matches_importer",
        "--database_path", str(output),
        "--match_list_path", str(pair_list),
        "--match_type", "pairs",
        "--SiftMatching.num_threads", "1",
        "--SiftMatching.use_gpu", "0",
    ])
    if result.returncode:
        return result.returncode
    connection = sqlite3.connect(str(output))
    totals = {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("images", "keypoints", "matches", "two_view_geometries")
    }
    connection.close()
    print(
        f"done existing={existing_count} new={len(new_names)} "
        f"new_pairs={len(strong_pairs)} weak={weak} totals={totals}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
