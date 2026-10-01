"""Export raw DPVO poses and patch points as a COLMAP viewer model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from evo.tools import file_interface

from module3_map_builder.dpvo_state import load_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = load_state(args.state)
    graph = state["graph"]
    camera = state["camera"]
    trajectory = file_interface.read_tum_trajectory_file(str(args.trajectory))

    points = graph["points"].numpy().reshape(-1, 3)
    colors = graph["colors"].numpy().reshape(-1, 3)[: len(points)]
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    colors = colors[valid]
    if colors.dtype != np.uint8:
        if colors.size and float(colors.max()) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    intrinsics = graph["intrinsics"].numpy()
    fx, fy, cx, cy = map(float, intrinsics[0])
    width = int(camera["width"])
    height = int(camera["height"])

    sparse = args.out_dir.resolve() / "sparse_text"
    sparse.mkdir(parents=True, exist_ok=True)
    (sparse / "cameras.txt").write_text(
        f"# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n",
        encoding="utf-8",
    )

    image_lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    for image_id, (center, quat_cw, timestamp) in enumerate(
        zip(
            trajectory.positions_xyz,
            trajectory.orientations_quat_wxyz,
            trajectory.timestamps,
            strict=True,
        ),
        start=1,
    ):
        qw, qx, qy, qz = map(float, quat_cw)
        quat_wc = np.array([qw, -qx, -qy, -qz])
        w, x, y, z = quat_wc
        rotation_wc = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        translation = -rotation_wc @ np.asarray(center)
        image_lines.append(
            f"{image_id} {quat_wc[0]:.12g} {quat_wc[1]:.12g} {quat_wc[2]:.12g} "
            f"{quat_wc[3]:.12g} {translation[0]:.12g} {translation[1]:.12g} "
            f"{translation[2]:.12g} 1 frame_{int(timestamp):06d}.jpg"
        )
        image_lines.append("")
    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")

    point_lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]"]
    for point_id, (point, color) in enumerate(zip(points, colors, strict=True), start=1):
        point_lines.append(
            f"{point_id} {point[0]:.12g} {point[1]:.12g} {point[2]:.12g} "
            f"{int(color[0])} {int(color[1])} {int(color[2])} 0"
        )
    (sparse / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")

    print(f"DPVO-only COLMAP model: {sparse}")
    print(f"poses={trajectory.num_poses} points={len(points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
