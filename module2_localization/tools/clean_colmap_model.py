"""Create a navigation copy of a COLMAP model without two-view 3D points."""

from __future__ import annotations

import argparse
from pathlib import Path

import pycolmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-track-length", type=int, default=3)
    parser.add_argument("--min-retained-ratio", type=float, default=0.7)
    parser.add_argument(
        "--absolute-image-root",
        type=Path,
        help="Записать абсолютные пути к изображениям для импорта в COLMAP GUI",
    )
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    reconstruction = pycolmap.Reconstruction(str(args.input))
    original_points = reconstruction.num_points3D()
    remove = [
        point_id
        for point_id, point in reconstruction.points3D.items()
        if point.track.length() < args.min_track_length
    ]
    retained_ratio = (original_points - len(remove)) / max(original_points, 1)
    if retained_ratio < args.min_retained_ratio:
        raise RuntimeError(
            f"Refusing destructive filtering: retained ratio {retained_ratio:.1%} "
            f"is below {args.min_retained_ratio:.1%}"
        )

    for point_id in remove:
        reconstruction.delete_point3D(point_id)
    if args.absolute_image_root:
        root = args.absolute_image_root.resolve()
        for image in reconstruction.images.values():
            image.name = str(root / image.name)
    reconstruction.write(str(args.output))

    print(f"IMAGES={reconstruction.num_reg_images()}")
    print(f"POINTS_BEFORE={original_points}")
    print(f"REMOVED={len(remove)}")
    print(f"POINTS_AFTER={reconstruction.num_points3D()}")
    print(f"RETAINED={retained_ratio:.2%}")


if __name__ == "__main__":
    main()
