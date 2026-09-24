"""Replace a runtime-node range in a COLMAP model with a separately built segment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pycolmap


def convert_to_text(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "colmap", "model_converter",
        "--input_path", str(source),
        "--output_path", str(output),
        "--output_type", "TXT",
    ], check=True)


def read_image_text(path: Path):
    lines = path.read_text().splitlines()
    header = []
    records = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line or line.startswith("#"):
            header.append(line)
            index += 1
            continue
        pose = line
        points = lines[index + 1] if index + 1 < len(lines) else ""
        fields = pose.split()
        image_id = int(fields[0])
        name = Path(fields[-1]).name
        records[name] = {
            "old_id": image_id,
            "pose_fields": fields,
            "points_line": points,
        }
        index += 2
    return header, records


def iter_points2d(points_line: str):
    fields = points_line.split()
    for index in range(0, len(fields), 3):
        yield fields[index], fields[index + 1], int(fields[index + 2])


def rewrite_points2d(points_line: str, point_id_map: dict[int, int]) -> str:
    rewritten = []
    for x, y, point_id in iter_points2d(points_line):
        rewritten.extend((x, y, str(point_id_map.get(point_id, -1))))
    return " ".join(rewritten)


def read_points3d_text(path: Path):
    lines = path.read_text().splitlines()
    header = []
    records = []
    for line in lines:
        if not line or line.startswith("#"):
            header.append(line)
            continue
        fields = line.split()
        records.append(fields)
    return header, records


def image_centers_by_name(reconstruction: pycolmap.Reconstruction) -> dict[str, np.ndarray]:
    return {
        Path(image.name).name: np.asarray(image.projection_center())
        for image in reconstruction.images.values()
        if image.has_pose
    }


def estimate_sim3(source: np.ndarray, target: np.ndarray):
    source_mean = source.mean(0)
    target_mean = target.mean(0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vt
    variance = np.sum(source_centered * source_centered) / len(source)
    scale = np.trace(np.diag(singular) @ sign) / variance
    translation = target_mean - scale * rotation @ source_mean
    fitted = (scale * (rotation @ source.T)).T + translation
    residuals = np.linalg.norm(fitted - target, axis=1)
    return scale, rotation, translation, residuals


def diagnostics_pairs(path: Path, fragment_names: set[str], runtime_names: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("record_type") != "frame" or not item.get("accepted"):
            continue
        processed = item.get("processed")
        node = item.get("global_node")
        if processed is None or node is None:
            continue
        fragment_name = f"r21fix_{int(processed):06d}.jpg"
        if fragment_name not in fragment_names:
            continue
        node = int(node)
        if 0 <= node < len(runtime_names):
            pairs.append((fragment_name, Path(str(runtime_names[node])).name))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-map", type=Path, required=True)
    parser.add_argument("--base-runtime", type=Path, required=True)
    parser.add_argument("--segment-map", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remove-first-node", type=int, required=True)
    parser.add_argument("--remove-last-node", type=int, required=True)
    parser.add_argument("--min-track-length", type=int, default=2)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    runtime = np.load(args.base_runtime, allow_pickle=True)
    runtime_names = [Path(str(name)).name for name in runtime["names"]]
    prefix_names = runtime_names[:args.remove_first_node]
    suffix_names = runtime_names[args.remove_last_node + 1:]

    base_rec = pycolmap.Reconstruction(str(args.base_map))
    segment_rec = pycolmap.Reconstruction(str(args.segment_map))
    base_centers = image_centers_by_name(base_rec)
    segment_centers = image_centers_by_name(segment_rec)
    segment_names = sorted(segment_centers)

    pairs = diagnostics_pairs(args.diagnostics, set(segment_names), runtime_names)
    source = []
    target = []
    for segment_name, base_name in pairs:
        if segment_name in segment_centers and base_name in base_centers:
            source.append(segment_centers[segment_name])
            target.append(base_centers[base_name])
    if len(source) < 3:
        raise SystemExit(f"not enough alignment pairs: {len(source)}")
    source = np.asarray(source)
    target = np.asarray(target)
    scale, rotation, translation, residuals = estimate_sim3(source, target)

    with tempfile.TemporaryDirectory(prefix="merge_repaired_segment_") as temporary:
        temp = Path(temporary)
        aligned_segment = temp / "segment_aligned"
        transformed = pycolmap.Reconstruction(str(args.segment_map))
        transformed.transform(pycolmap.Sim3d(np.c_[scale * rotation, translation]))
        aligned_segment.mkdir()
        transformed.write(str(aligned_segment))

        base_text = temp / "base_text"
        segment_text = temp / "segment_text"
        merged_text = temp / "merged_text"
        merged_bin = temp / "merged_bin"
        convert_to_text(args.base_map, base_text)
        convert_to_text(aligned_segment, segment_text)
        merged_text.mkdir()
        merged_bin.mkdir()

        shutil.copy2(base_text / "cameras.txt", merged_text / "cameras.txt")

        _, base_images = read_image_text(base_text / "images.txt")
        _, segment_images = read_image_text(segment_text / "images.txt")
        missing = [name for name in prefix_names + suffix_names if name not in base_images]
        if missing:
            raise SystemExit(f"base images are missing: {missing[:5]}")
        missing = [name for name in segment_names if name not in segment_images]
        if missing:
            raise SystemExit(f"segment images are missing: {missing[:5]}")

        final_names = prefix_names + segment_names + suffix_names
        new_image_id = {name: index + 1 for index, name in enumerate(final_names)}

        base_old_image_id_to_name = {}
        for name, record in base_images.items():
            base_old_image_id_to_name[record["old_id"]] = name
        segment_old_image_id_to_name = {}
        for name, record in segment_images.items():
            segment_old_image_id_to_name[record["old_id"]] = name

        base_point_header, base_points = read_points3d_text(base_text / "points3D.txt")
        _, segment_points = read_points3d_text(segment_text / "points3D.txt")
        max_base_point_id = max((int(fields[0]) for fields in base_points), default=0)
        point_id_map: dict[int, int] = {}
        merged_points = []


        # Keep original base point IDs, offset segment IDs above the base range.
        for fields in base_points:
            old_point_id = int(fields[0])
            track = []
            for index in range(8, len(fields), 2):
                old_image_id = int(fields[index])
                image_name = base_old_image_id_to_name.get(old_image_id)
                if image_name not in new_image_id:
                    continue
                track.extend((str(new_image_id[image_name]), fields[index + 1]))
            if len(track) < 2 * args.min_track_length:
                continue
            point_id_map[old_point_id] = old_point_id
            merged_points.append([str(old_point_id)] + fields[1:8] + track)

        segment_point_id_map = {}
        for fields in segment_points:
            old_point_id = int(fields[0])
            new_point_id = max_base_point_id + old_point_id
            track = []
            for index in range(8, len(fields), 2):
                old_image_id = int(fields[index])
                image_name = segment_old_image_id_to_name.get(old_image_id)
                if image_name not in new_image_id:
                    continue
                track.extend((str(new_image_id[image_name]), fields[index + 1]))
            if len(track) < 2 * args.min_track_length:
                continue
            segment_point_id_map[old_point_id] = new_point_id
            merged_points.append([str(new_point_id)] + fields[1:8] + track)

        (merged_text / "points3D.txt").write_text(
            "\n".join(base_point_header + [" ".join(fields) for fields in merged_points]) + "\n"
        )

        image_lines = ["# Image list with two lines of data per image:"]
        for name in final_names:
            if name in base_images:
                record = base_images[name]
                local_point_map = point_id_map
            else:
                record = segment_images[name]
                local_point_map = segment_point_id_map
            pose_fields = list(record["pose_fields"])
            pose_fields[0] = str(new_image_id[name])
            pose_fields[-1] = name
            image_lines.append(" ".join(pose_fields))
            image_lines.append(rewrite_points2d(record["points_line"], local_point_map))
        (merged_text / "images.txt").write_text("\n".join(image_lines) + "\n")

        subprocess.run([
            "colmap", "model_converter",
            "--input_path", str(merged_text),
            "--output_path", str(merged_bin),
            "--output_type", "BIN",
        ], check=True)
        shutil.copytree(merged_bin, args.output)

    print(f"alignment_pairs={len(source)}")
    print(f"alignment_scale={scale:.12g}")
    print(f"alignment_rms={np.sqrt(np.mean(residuals ** 2)):.12g}")
    print(f"alignment_p90={np.quantile(residuals, 0.9):.12g}")
    print(f"alignment_max={residuals.max():.12g}")
    print(f"images={len(final_names)}")
    print(f"points={len(merged_points)}")
    print(f"removed_nodes={args.remove_first_node}..{args.remove_last_node}")
    print(f"inserted_images={len(segment_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
