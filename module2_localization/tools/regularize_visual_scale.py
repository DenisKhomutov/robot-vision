"""Remove monocular scale drift using the cadence of adaptive frame extraction.

`extract_adaptive` selects a frame after a target amount of visual motion.  The
selected-frame cadence is therefore a usable visual motion prior: translation
is proportional to the source-frame gap, capped by the extractor's max gap.
Camera rotations from SfM are retained and the route is integrated along the
front camera optical axis.  The resulting fixed poses can be passed to COLMAP's
point_triangulator to rebuild consistent 3D points.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pycolmap


def _key(image: pycolmap.Image) -> tuple[int, int]:
    prefix, number = Path(image.name).stem.split("_", 1)
    return int(prefix[1:]), int(number)


def regularize(
    input_model: Path,
    output_model: Path,
    max_gap: int,
    distance_per_max_gap: float,
    p2_tail_start: int | None,
) -> None:
    reconstruction = pycolmap.Reconstruction(input_model)
    images = sorted(reconstruction.images.values(), key=_key)

    rotations = []
    for image in images:
        pose = image.cam_from_world().inverse()
        rotations.append(np.asarray(pose.rotation.matrix()))
    rotations = np.asarray(rotations)


    axes = rotations[:, :, 2]
    smoothed_axes = np.empty_like(axes)
    for index in range(len(axes)):
        lo, hi = max(0, index - 2), min(len(axes), index + 3)
        vector = axes[lo:hi].sum(axis=0)
        smoothed_axes[index] = vector / np.linalg.norm(vector)

    centers = np.zeros((len(images), 3), dtype=float)
    for index in range(1, len(images)):
        previous_part, previous_frame = _key(images[index - 1])
        part, frame = _key(images[index])
        if part != previous_part:

            centers[index] = centers[index - 1]
            continue
        frame_gap = max(0, frame - previous_frame)
        distance = distance_per_max_gap * min(frame_gap, max_gap) / max_gap
        direction = smoothed_axes[index - 1] + smoothed_axes[index]
        direction /= np.linalg.norm(direction)
        centers[index] = centers[index - 1] + distance * direction

    for point3d_id in list(reconstruction.point3D_ids()):
        reconstruction.delete_point3D(point3d_id)

    for index, image in enumerate(images):
        part, frame = _key(image)
        if p2_tail_start is not None and part == 2 and frame > p2_tail_start:
            if image.has_pose:
                reconstruction.deregister_frame(image.frame_id)
            continue
        camera_to_world_rotation = rotations[index]
        world_to_camera_rotation = camera_to_world_rotation.T
        translation = -world_to_camera_rotation @ centers[index]
        reconstruction.frames[image.frame_id].rig_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(world_to_camera_rotation), translation
        )

    output_model.mkdir(parents=True, exist_ok=True)
    reconstruction.write(output_model)
    registered = reconstruction.num_reg_images()
    distances = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    print(f"Regularized images: {registered} / {len(images)}")
    print(f"Route length: {distances.sum():.3f}; endpoint: {np.linalg.norm(centers[-1]):.3f}")
    print(
        "Step median/p90/max: "
        f"{np.median(distances):.5f} / {np.quantile(distances, 0.9):.5f} / "
        f"{distances.max():.5f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gap", type=int, default=15)
    parser.add_argument("--distance-per-max-gap", type=float, default=0.05)
    parser.add_argument(
        "--p2-tail-start",
        type=int,
        default=3975,
        help="Deregister sparse stationary tail after this original p2 frame",
    )
    args = parser.parse_args()
    regularize(
        args.input_model,
        args.output,
        args.max_gap,
        args.distance_per_max_gap,
        args.p2_tail_start,
    )


if __name__ == "__main__":
    main()
